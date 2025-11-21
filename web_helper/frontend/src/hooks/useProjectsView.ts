import { useState, useMemo, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useProjects, useProjectExperiments, useRuns } from './useProjects';
import { apiClient } from '../lib/api-client';
import { adaptExperiments } from '../lib/dataAdapter';
import { extractColumns, getValue} from '../lib/table-columns';
import { applyFilters, type FilterRule } from '../components/ui/advanced-filter';
import type { SavedView } from '../lib/saved-views';
import html2canvas from 'html2canvas';
import { devError } from '../lib/dev-utils';
import {
  listColumnMappings,
  createColumnMapping,
  updateColumnMapping,
  deleteColumnMapping,
  generateColumnSuggestions,
  type ColumnMapping,
  type ColumnSuggestion,
  type ColumnMappingCreate,
} from '../lib/api/column-mappings';
import {
  getOutlierSummary,
  type OutlierSummaryResponse,
  type OutlierMethod,
} from '../lib/api/outliers';

export function useProjectsView() {
  const { projectName: activeProject } = useParams<{ projectName: string }>();
  const navigate = useNavigate();

  const { data: projects = [], isLoading: projectsLoading, isError: _projectsError, error: _projectsErrorObj } = useProjects();
  const { data: allRuns = [], isLoading: runsLoading } = useRuns();
  const { data: experimentData, isLoading: experimentsLoading } = useProjectExperiments(activeProject || null);

  const [flattenParams, setFlattenParams] = useState(false);
  const [diffMode, setDiffMode] = useState(false);
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>(() => {
    // Load from sessionStorage on init
    try {
      const saved = sessionStorage.getItem(`selectedExperiments_${activeProject}`);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [activeView, setActiveView] = useState('overview');
  const [selectedRunDetail, setSelectedRunDetail] = useState<Record<string, unknown> | null>(null);
  const [runMetricsData, setRunMetricsData] = useState<Record<string, unknown>[]>([]);
  const [filters, setFilters] = useState<FilterRule[]>([]);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);
  const [visibleHyperparams, setVisibleHyperparams] = useState<Set<string>>(new Set());
  const [visibleMetrics, setVisibleMetrics] = useState<Set<string>>(new Set());
  const [pinnedLeftHyperparams, setPinnedLeftHyperparams] = useState<Set<string>>(new Set());
  const [pinnedRightMetrics, setPinnedRightMetrics] = useState<Set<string>>(new Set());
  const [hyperparamOrder, setHyperparamOrder] = useState<string[]>([]);
  const [metricOrder, setMetricOrder] = useState<string[]>([]);
  const [isComparisonModalOpen, setIsComparisonModalOpen] = useState(false);
  const [runDetailTab, setRunDetailTab] = useState<'metrics' | 'config' | 'info'>('metrics');
  const [activeModalTab, setActiveModalTab] = useState('execution');
  const [configContent, setConfigContent] = useState<string>('');
  const [logsContent, setLogsContent] = useState<string>('');
  const [isLoadingConfig, setIsLoadingConfig] = useState(false);
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [logSearch, setLogSearch] = useState('');
  const [selectedMetric, setSelectedMetric] = useState<string>('val/acc');
  const [metricDirection, setMetricDirection] = useState<'maximize' | 'minimize'>('maximize');

  // Auto-select direction based on metric name patterns
  useEffect(() => {
    const metricLower = selectedMetric.toLowerCase();

    // Patterns that should be minimized
    const minimizePatterns = ['loss', 'error', 'err', 'mse', 'mae', 'rmse', 'cost'];
    const shouldMinimize = minimizePatterns.some(pattern => metricLower.includes(pattern));

    // Patterns that should be maximized
    const maximizePatterns = ['acc', 'accuracy', 'precision', 'recall', 'f1', 'score', 'auc', 'map', 'iou'];
    const shouldMaximize = maximizePatterns.some(pattern => metricLower.includes(pattern));

    // Auto-select direction (maximize takes precedence if both match)
    if (shouldMaximize) {
      setMetricDirection('maximize');
    } else if (shouldMinimize) {
      setMetricDirection('minimize');
    }
    // If neither matches, keep current direction (user can manually change)
  }, [selectedMetric]);

  // Column mapping state
  const [columnMappings, setColumnMappings] = useState<ColumnMapping[]>([]);
  const [columnSuggestions, setColumnSuggestions] = useState<ColumnSuggestion[]>([]);
  const [isColumnMappingDialogOpen, setIsColumnMappingDialogOpen] = useState(false);
  const [isLoadingColumnMappings, setIsLoadingColumnMappings] = useState(false);
  const [isGeneratingSuggestions, setIsGeneratingSuggestions] = useState(false);

  // Outlier detection state
  const [outlierSummary, setOutlierSummary] = useState<OutlierSummaryResponse | null>(null);
  const [outlierMethod, setOutlierMethod] = useState<OutlierMethod>('iqr');
  const [outlierThreshold, setOutlierThreshold] = useState(1.5);
  const [isLoadingOutliers, setIsLoadingOutliers] = useState(false);
  const [cellOutliers, setCellOutliers] = useState<Set<string>>(new Set());

  // Highlighted run state (for outlier/recommendation selection)
  const [highlightedRun, setHighlightedRun] = useState<string | null>(null);

  const chartsRef = useRef<HTMLDivElement>(null);

  const projectSummaries = useMemo(() => {
    return projects.map(project => {
      const projectRunsList = allRuns.filter(run => run.project === project.name);
      const completedRuns = projectRunsList.filter(run => run.status === 'completed');

      const durationsMs = completedRuns
        .filter(run => run.started_at && run.finished_at)
        .map(run => {
          const start = new Date(run.started_at!).getTime();
          const end = new Date(run.finished_at!).getTime();
          return end - start;
        });

      const avgDurationMs = durationsMs.length > 0
        ? durationsMs.reduce((a, b) => a + b, 0) / durationsMs.length
        : 0;

      const avgDuration = avgDurationMs > 0
        ? `${Math.round(avgDurationMs / (1000 * 60))}m`
        : '-';

      const sortedRuns = [...projectRunsList].sort((a, b) => {
        const dateA = a.started_at ? new Date(a.started_at) : new Date(0);
        const dateB = b.started_at ? new Date(b.started_at) : new Date(0);
        return dateB.getTime() - dateA.getTime();
      });

      const lastActivity = sortedRuns[0]?.started_at
        ? new Date(sortedRuns[0].started_at).toLocaleDateString()
        : '-';

      return {
        name: project.name,
        run_count: projectRunsList.length, // Calculate count on the client-side
        completed_count: completedRuns.length,
        avg_duration: avgDuration,
        last_activity: lastActivity,
      };
    });
  }, [projects, allRuns]);

  const experimentsData = useMemo(() => {
    if (!experimentData?.experiments) return [];
    let data = experimentData.experiments;

    // Apply advanced filters
    if (filters.length > 0) {
      data = applyFilters(data, filters);
    }

    return data;
  }, [experimentData, filters]);

  const { hyperparamColumns, metricColumns } = useMemo(() => {
    if (experimentsData.length === 0) return { hyperparamColumns: [], metricColumns: [] };

    const allHyperparams = new Set<string>();
    const allMetrics = new Set<string>();

    experimentsData.forEach(exp => {
      if (exp.hyperparameters) {
        const cols = extractColumns(exp.hyperparameters, flattenParams);
        cols.forEach(key => allHyperparams.add(key));
      }
      if (exp.final_metrics) {
        const cols = extractColumns(exp.final_metrics, flattenParams);
        cols.forEach(key => {
          if (key !== 'step' && key !== 'epoch') {
            allMetrics.add(key);
          }
        });
      }
    });

    let hyperparamList = Array.from(allHyperparams);
    let metricList = Array.from(allMetrics);

    if (diffMode && experimentsData.length > 1) {
      hyperparamList = hyperparamList.filter(param => {
        const values = experimentsData.map(exp =>
          getValue(exp.hyperparameters || {}, param, flattenParams)
        );
        return new Set(values.map(v => JSON.stringify(v))).size > 1;
      });

      metricList = metricList.filter(metric => {
        const values = experimentsData.map(exp =>
          getValue(exp.final_metrics || {}, metric, flattenParams)
        );
        return new Set(values.map(v => JSON.stringify(v))).size > 1;
      });
    }

    // Add Run ID, Status, and Tags at the beginning (always visible columns)
    // Note: Tags is not a hyperparam but we include it here for column ordering
    return {
      hyperparamColumns: ['Run ID', 'Status', 'Tags', ...hyperparamList],
      metricColumns: metricList
    };
  }, [experimentsData, diffMode, flattenParams]);

  useEffect(() => {
    if (hyperparamColumns.length > 0) {
      setVisibleHyperparams(new Set(hyperparamColumns));
      // Update order only if it's empty or columns changed significantly
      setHyperparamOrder(prev => {
        if (prev.length === 0) return hyperparamColumns;
        // Keep existing order for columns that still exist, add new ones at the end
        const existing = prev.filter(col => hyperparamColumns.includes(col));
        const newCols = hyperparamColumns.filter(col => !prev.includes(col));
        return [...existing, ...newCols];
      });
    }
  }, [hyperparamColumns]);

  useEffect(() => {
    if (metricColumns.length > 0) {
      setVisibleMetrics(new Set(metricColumns));
      // Update order only if it's empty or columns changed significantly
      setMetricOrder(prev => {
        if (prev.length === 0) return metricColumns;
        // Keep existing order for columns that still exist, add new ones at the end
        const existing = prev.filter(col => metricColumns.includes(col));
        const newCols = metricColumns.filter(col => !prev.includes(col));
        return [...existing, ...newCols];
      });
    }
  }, [metricColumns]);

  const clearSelection = () => {
    setSelectedExperiments([]);
  };

  // Persist selectedExperiments to sessionStorage
  useEffect(() => {
    if (activeProject) {
      sessionStorage.setItem(`selectedExperiments_${activeProject}`, JSON.stringify(selectedExperiments));
    }
  }, [selectedExperiments, activeProject]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        clearSelection();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedExperiments]);

  const fetchRunConfiguration = async (run: Record<string, unknown>) => {
    if (!run || !activeProject) return;
    setIsLoadingConfig(true);
    try {
      const response = await apiClient.getRunConfig(activeProject, run.run_name as string);
      const content = response.content || 'No configuration data available.';
      setConfigContent(content);
    } catch (error) {
      const content = `Error loading configuration: ${error instanceof Error ? error.message : 'Unknown error'}`;
      setConfigContent(content);
    } finally {
      setIsLoadingConfig(false);
    }
  };

  const fetchRunLogs = async (run: Record<string, unknown>) => {
    if (!run || !activeProject) return;
    setIsLoadingLogs(true);
    try {
      const response = await apiClient.getRunLogs(activeProject, run.run_name as string);
      const content = response.content || 'No log data available.';
      setLogsContent(content);
    } catch (error) {
      const content = `Error loading logs: ${error instanceof Error ? error.message : 'Unknown error'}`;
      setLogsContent(content);
    } finally {
      setIsLoadingLogs(false);
    }
  };

  const exportRuns = async (format: 'json' | 'markdown' | 'png' | 'svg' | 'csv') => {
    const runsToExport = selectedExperiments.length > 0
      ? experimentsData.filter(exp => selectedExperiments.includes(exp.run_name))
      : experimentsData;

    if (runsToExport.length === 0) {
      alert('No runs to export');
      return;
    }

    const timestamp = new Date().toISOString().split('T')[0];
    const projectPrefix = activeProject ? `${activeProject}_` : '';

    switch (format) {
      case 'json': {
        const data = JSON.stringify(runsToExport, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${projectPrefix}runs_${timestamp}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        break;
      }

      case 'csv': {
        // Generate CSV with hyperparameters and metrics
        const headers = ['run_name', ...hyperparamColumns.map(h => `hp_${h}`), ...metricColumns.map(m => `metric_${m}`)];
        const rows = runsToExport.map(run => {
          const row = [run.run_name];

          // Add hyperparameters
          hyperparamColumns.forEach(param => {
            const value = getValue(run.hyperparameters || {}, param, flattenParams);
            row.push(value !== undefined && value !== null ? String(value) : '');
          });

          // Add metrics
          metricColumns.forEach(metric => {
            const value = getValue(run.final_metrics || {}, metric, flattenParams);
            row.push(value !== undefined && value !== null ? String(value) : '');
          });

          return row;
        });

        const csv = [headers.join(','), ...rows.map(r => r.map(cell => {
          const str = String(cell);
          return str.includes(',') ? `"${str.replace(/"/g, '""')}"` : str;
        }).join(','))].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${projectPrefix}runs_${timestamp}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        break;
      }

      case 'markdown': {
        // Generate Markdown table
        let md = `# ${activeProject || 'Project'} Runs Export\n\n`;
        md += `Generated: ${new Date().toISOString()}\n\n`;
        md += `## Summary\n\n`;
        md += `- Total Runs: ${runsToExport.length}\n`;
        md += `- Selected Runs: ${selectedExperiments.length || 'All'}\n\n`;
        md += `## Runs Table\n\n`;

        // Table header
        const headers = ['Run Name', ...hyperparamColumns.slice(0, 5), ...metricColumns.slice(0, 5)];
        md += `| ${headers.join(' | ')} |\n`;
        md += `| ${headers.map(() => '---').join(' | ')} |\n`;

        // Table rows
        runsToExport.forEach(run => {
          const cells = [run.run_name];

          hyperparamColumns.slice(0, 5).forEach(param => {
            const value = getValue(run.hyperparameters || {}, param, flattenParams);
            cells.push(value !== undefined && value !== null ? String(value) : 'N/A');
          });

          metricColumns.slice(0, 5).forEach(metric => {
            const value = getValue(run.final_metrics || {}, metric, flattenParams);
            cells.push(value !== undefined && value !== null
              ? (typeof value === 'number' ? value.toFixed(4) : String(value))
              : 'N/A');
          });

          md += `| ${cells.join(' | ')} |\n`;
        });

        const blob = new Blob([md], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${projectPrefix}runs_${timestamp}.md`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        break;
      }

      case 'png': {
        if (!chartsRef.current) {
          alert('Charts view not available. Switch to Charts tab first.');
          return;
        }

        try {
          const canvas = await html2canvas(chartsRef.current, {
            backgroundColor: '#ffffff',
            scale: 2,
          });

          canvas.toBlob((blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${projectPrefix}charts_${timestamp}.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
          });
        } catch (error) {
          devError('Failed to export as PNG:', error);
          alert('Failed to export as PNG. Please try again.');
        }
        break;
      }
    }
  };

  const getSelectedExperimentData = () => {
    const selectedExps = experimentsData.filter(exp => selectedExperiments.includes(exp.run_name));
    return adaptExperiments(selectedExps, activeProject || '');
  };

  const fetchRunDetails = async (run: Record<string, unknown>) => {
    if (!activeProject) return;

    setIsLoadingMetrics(true);
    fetchRunConfiguration(run);
    fetchRunLogs(run);

    try {
      const runName = typeof run.run_name === 'string' ? run.run_name : '';
      const metricsResponse = await apiClient.getRunMetrics(activeProject, runName);
      const rawMetrics = metricsResponse.data || [];

      // Transform raw CSV data to timeseries format for RunDetailModal
      const timeseries = rawMetrics.map((row: Record<string, unknown>) => {
        const { step, epoch, ...values } = row;
        return {
          step: typeof step === 'number' ? step : 0,
          epoch: typeof epoch === 'number' ? epoch : 0,
          values: values as Record<string, number>
        };
      });

      // Attach timeseries data to run object
      const runWithMetrics = {
        ...run,
        project: activeProject, // Ensure project is always present
        metrics: {
          final: (run as any).final_metrics || {},
          max: {},
          min: {},
          mean: {},
          timeseries
        }
      };

      setSelectedRunDetail(runWithMetrics);
      setRunMetricsData(rawMetrics);
    } catch (error) {
      devError('Failed to fetch run metrics:', error);
      setSelectedRunDetail({...run, project: activeProject}); // Add project in error case too
      setRunMetricsData([]);
    } finally {
      setIsLoadingMetrics(false);
    }
  };

  const getBestRun = useMemo(() => {
    if (experimentsData.length === 0) return null;

    // Use user-selected metric or fallback to common defaults
    let keyMetric = selectedMetric;

    // Determine which metrics source to use based on direction
    const metricsSource = metricDirection === 'maximize' ? 'max_metrics' : 'min_metrics';

    // Verify selected metric exists in data, otherwise fallback
    const hasSelectedMetric = experimentsData.some(run =>
      run[metricsSource] && typeof run[metricsSource][keyMetric] === 'number'
    );

    if (!hasSelectedMetric) {
      const fallbackMetrics = ['val/acc', 'test/acc', 'accuracy', 'acc', 'test_acc', 'val_acc'];
      keyMetric = fallbackMetrics.find(metric =>
        experimentsData.some(run => run[metricsSource] && typeof run[metricsSource][metric] === 'number')
      ) || metricColumns[0] || 'val/acc';
    }

    let bestRun: any = null;
    let bestValue = metricDirection === 'maximize' ? -Infinity : Infinity;

    experimentsData.forEach(run => {
      // Use max_metrics for maximize, min_metrics for minimize
      const metricsToCheck = run[metricsSource];
      if (!metricsToCheck) return;

      const value = metricsToCheck[keyMetric];
      if (typeof value !== 'number' || isNaN(value)) return;

      const isBetter = metricDirection === 'maximize'
        ? value > bestValue
        : value < bestValue;

      if (isBetter) {
        bestValue = value;
        bestRun = run;
      }
    });

    // If still no best run, just return the first completed run
    if (!bestRun) {
      bestRun = experimentsData.find(run => run.status === 'completed' && run.final_metrics);
    }

    // Transform to Run type format expected by BestRunCard
    if (bestRun) {
      return {
        run_name: bestRun.run_name,
        project: activeProject || '',
        status: bestRun.status,
        started_at: bestRun.started_at,
        finished_at: bestRun.finished_at,
        hyperparameters: bestRun.hyperparameters,
        metrics: {
          final: bestRun.final_metrics || {},
          max: bestRun.max_metrics || {},
          min: bestRun.min_metrics || {},
          mean: bestRun.mean_metrics || {}
        }
      };
    }

    return null;
  }, [experimentsData, activeProject, selectedMetric, metricDirection, metricColumns]);

  // Column mapping functions
  const loadColumnMappings = async () => {
    if (!activeProject) return;
    setIsLoadingColumnMappings(true);
    try {
      const mappings = await listColumnMappings(activeProject);
      setColumnMappings(mappings);
    } catch (error) {
      devError('Failed to load column mappings:', error);
    } finally {
      setIsLoadingColumnMappings(false);
    }
  };

  const generateSuggestions = async (columnType: 'hyperparam' | 'metric' = 'hyperparam') => {
    if (!activeProject) return;
    setIsGeneratingSuggestions(true);
    try {
      const result = await generateColumnSuggestions(activeProject, columnType, 0.5);
      setColumnSuggestions(result.suggestions);
    } catch (error) {
      devError('Failed to generate suggestions:', error);
      setColumnSuggestions([]);
    } finally {
      setIsGeneratingSuggestions(false);
    }
  };

  const handleAcceptSuggestion = async (suggestion: ColumnSuggestion) => {
    if (!activeProject) return;
    try {
      await createColumnMapping(activeProject, {
        source_column: suggestion.source_column,
        target_column: suggestion.target_column,
        column_type: suggestion.column_type,
        mapping_method: 'verified',
        confidence_score: suggestion.confidence_score,
        algorithm: suggestion.algorithm,
      });
      // Remove from suggestions and reload mappings
      setColumnSuggestions(prev => prev.filter(s => s.source_column !== suggestion.source_column));
      await loadColumnMappings();
    } catch (error) {
      devError('Failed to accept suggestion:', error);
      alert(`Failed to accept suggestion: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleRejectSuggestion = (suggestion: ColumnSuggestion) => {
    setColumnSuggestions(prev => prev.filter(s => s.source_column !== suggestion.source_column));
  };

  const handleCreateMapping = async (mapping: ColumnMappingCreate) => {
    if (!activeProject) return;
    try {
      await createColumnMapping(activeProject, mapping);
      await loadColumnMappings();
    } catch (error) {
      devError('Failed to create mapping:', error);
      alert(`Failed to create mapping: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleUpdateMapping = async (id: number, target: string) => {
    if (!activeProject) return;
    try {
      await updateColumnMapping(activeProject, id, { target_column: target });
      await loadColumnMappings();
    } catch (error) {
      devError('Failed to update mapping:', error);
      alert(`Failed to update mapping: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleDeleteMapping = async (id: number) => {
    if (!activeProject) return;
    try {
      await deleteColumnMapping(activeProject, id);
      await loadColumnMappings();
    } catch (error) {
      devError('Failed to delete mapping:', error);
      alert(`Failed to delete mapping: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  // Load column mappings and generate suggestions when project changes
  useEffect(() => {
    if (activeProject) {
      loadColumnMappings();
      generateSuggestions('hyperparam');
    }
  }, [activeProject]);

  // Outlier detection functions
  const loadOutlierSummary = async () => {
    if (!activeProject) return;
    setIsLoadingOutliers(true);
    try {
      const summary = await getOutlierSummary(activeProject, outlierMethod, outlierThreshold);
      setOutlierSummary(summary);
      // Parse cell_outliers object into Set for efficient lookup
      // Keys are in format: "run_name|column"
      setCellOutliers(new Set(Object.keys(summary.cell_outliers)));
    } catch (error) {
      devError('Failed to load outlier summary:', error);
      setOutlierSummary(null);
      setCellOutliers(new Set());
    } finally {
      setIsLoadingOutliers(false);
    }
  };

  // Load outliers when project or detection parameters change
  useEffect(() => {
    if (activeProject) {
      loadOutlierSummary();
    }
  }, [activeProject, outlierMethod, outlierThreshold]);

  return {
    // Data
    activeProject,
    projects,
    projectSummaries,
    experimentsData,
    hyperparamColumns,
    metricColumns,
    getBestRun,
    selectedMetric,
    setSelectedMetric,
    metricDirection,
    setMetricDirection,

    // Loaders
    projectsLoading,
    runsLoading,
    experimentsLoading,
    isLoadingMetrics,
    isLoadingConfig,
    isLoadingLogs,

    // State & Setters
    flattenParams, setFlattenParams,
    diffMode, setDiffMode,
    selectedExperiments, setSelectedExperiments,
    activeView, setActiveView,
    selectedRunDetail, setSelectedRunDetail,
    runMetricsData, setRunMetricsData,
    visibleHyperparams, setVisibleHyperparams,
    visibleMetrics, setVisibleMetrics,
    pinnedLeftHyperparams, setPinnedLeftHyperparams,
    pinnedRightMetrics, setPinnedRightMetrics,
    hyperparamOrder, setHyperparamOrder,
    metricOrder, setMetricOrder,
    isComparisonModalOpen, setIsComparisonModalOpen,
    runDetailTab, setRunDetailTab,
    activeModalTab, setActiveModalTab,
    configContent, setConfigContent,
    logsContent, setLogsContent,
    logSearch, setLogSearch,
    filters, setFilters,

    // Column mapping state
    columnMappings,
    columnSuggestions,
    isColumnMappingDialogOpen,
    setIsColumnMappingDialogOpen,
    isLoadingColumnMappings,
    isGeneratingSuggestions,

    // Outlier detection state
    outlierSummary,
    outlierMethod,
    setOutlierMethod,
    outlierThreshold,
    setOutlierThreshold,
    isLoadingOutliers,
    cellOutliers,
    highlightedRun,
    setHighlightedRun,

    // Functions
    navigate,
    clearSelection,
    exportRuns,
    getSelectedExperimentData,
    fetchRunDetails,
    handleAcceptSuggestion,
    handleRejectSuggestion,
    handleCreateMapping,
    handleUpdateMapping,
    handleDeleteMapping,
    generateSuggestions,
    loadOutlierSummary,

    // Refs
    chartsRef,
  };
}