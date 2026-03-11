import { useState, useRef, useCallback, useEffect } from 'react';
import { apiClient } from '@/shared/api/api-client';
import { devError } from '@/shared/lib/utils';
import {
  listColumnMappings,
  createColumnMapping,
  updateColumnMapping,
  deleteColumnMapping,
  generateColumnSuggestions,
  type ColumnMapping,
  type ColumnSuggestion,
  type ColumnMappingCreate,
} from '@/shared/api/column-mappings';

export interface RunDetailState {
  selectedRunDetail: Record<string, unknown> | null;
  runMetricsData: Record<string, unknown>[];
  configContent: string;
  logsContent: string;
  isLoadingMetrics: boolean;
  isLoadingConfig: boolean;
  isLoadingLogs: boolean;
  logSearch: string;
  runDetailTab: 'metrics' | 'config' | 'info';
}

export interface ColumnMappingState {
  columnMappings: ColumnMapping[];
  columnSuggestions: ColumnSuggestion[];
  isColumnMappingDialogOpen: boolean;
  isLoadingColumnMappings: boolean;
  isGeneratingSuggestions: boolean;
}

export interface ActionsResult {
  runDetailState: RunDetailState;
  columnMappingState: ColumnMappingState;

  // Run detail actions
  setSelectedRunDetail: (run: Record<string, unknown> | null) => void;
  setRunDetailTab: (tab: 'metrics' | 'config' | 'info') => void;
  setLogSearch: (search: string) => void;
  fetchRunDetails: (run: Record<string, unknown>) => Promise<void>;
  fetchRunConfiguration: (run: Record<string, unknown>) => Promise<void>;
  fetchRunLogs: (run: Record<string, unknown>) => Promise<void>;

  // Selection actions
  selectedExperiments: string[];
  setSelectedExperiments: React.Dispatch<React.SetStateAction<string[]>>;
  clearSelection: () => void;

  // Column mapping actions
  setIsColumnMappingDialogOpen: (open: boolean) => void;
  loadColumnMappings: () => Promise<void>;
  generateSuggestions: (columnType?: 'hyperparam' | 'metric') => Promise<void>;
  handleAcceptSuggestion: (suggestion: ColumnSuggestion) => Promise<void>;
  handleRejectSuggestion: (suggestion: ColumnSuggestion) => void;
  handleCreateMapping: (mapping: ColumnMappingCreate) => Promise<void>;
  handleUpdateMapping: (id: number, target: string) => Promise<void>;
  handleDeleteMapping: (id: number) => Promise<void>;
}

interface UseProjectActionsOptions {
  activeProject?: string | null;
}

/**
 * Hook for managing CRUD actions and data fetching.
 * Extracted from useProjectsView for better separation of concerns.
 */
export function useProjectActions(options: UseProjectActionsOptions): ActionsResult {
  const { activeProject } = options;

  // Run detail state
  const [selectedRunDetail, setSelectedRunDetail] = useState<Record<string, unknown> | null>(null);
  const [runMetricsData, setRunMetricsData] = useState<Record<string, unknown>[]>([]);
  const [configContent, setConfigContent] = useState<string>('');
  const [logsContent, setLogsContent] = useState<string>('');
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);
  const [isLoadingConfig, setIsLoadingConfig] = useState(false);
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [logSearch, setLogSearch] = useState('');
  const [runDetailTab, setRunDetailTab] = useState<'metrics' | 'config' | 'info'>('metrics');

  // Selection state
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>(() => {
    try {
      const saved = sessionStorage.getItem(`selectedExperiments_${activeProject}`);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Column mapping state
  const [columnMappings, setColumnMappings] = useState<ColumnMapping[]>([]);
  const [columnSuggestions, setColumnSuggestions] = useState<ColumnSuggestion[]>([]);
  const [isColumnMappingDialogOpen, setIsColumnMappingDialogOpen] = useState(false);
  const [isLoadingColumnMappings, setIsLoadingColumnMappings] = useState(false);
  const [isGeneratingSuggestions, setIsGeneratingSuggestions] = useState(false);

  // AbortController refs
  const configAbortControllerRef = useRef<AbortController | null>(null);
  const logsAbortControllerRef = useRef<AbortController | null>(null);
  const metricsAbortControllerRef = useRef<AbortController | null>(null);

  // Persist selectedExperiments to sessionStorage
  useEffect(() => {
    if (activeProject) {
      sessionStorage.setItem(`selectedExperiments_${activeProject}`, JSON.stringify(selectedExperiments));
    }
  }, [activeProject, selectedExperiments]);

  // Clear selection
  const clearSelection = useCallback(() => {
    setSelectedExperiments([]);
  }, []);

  // Fetch run configuration
  const fetchRunConfiguration = useCallback(async (run: Record<string, unknown>) => {
    if (configAbortControllerRef.current) {
      configAbortControllerRef.current.abort();
    }
    configAbortControllerRef.current = new AbortController();
    const signal = configAbortControllerRef.current.signal;

    setIsLoadingConfig(true);
    try {
      const response = await apiClient.getRunConfig(
        (run.project as string) || activeProject || '',
        run.run_name as string,
        { signal }
      );
      if (!signal.aborted) {
        setConfigContent(response.content || 'No configuration available');
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return;
      devError('Failed to fetch run configuration:', error);
      if (!signal.aborted) {
        setConfigContent('Failed to load configuration');
      }
    } finally {
      if (!signal.aborted) {
        setIsLoadingConfig(false);
      }
    }
  }, [activeProject]);

  // Fetch run logs
  const fetchRunLogs = useCallback(async (run: Record<string, unknown>) => {
    if (logsAbortControllerRef.current) {
      logsAbortControllerRef.current.abort();
    }
    logsAbortControllerRef.current = new AbortController();
    const signal = logsAbortControllerRef.current.signal;

    setIsLoadingLogs(true);
    try {
      const response = await apiClient.getRunLogs(
        (run.project as string) || activeProject || '',
        run.run_name as string,
        { signal }
      );
      if (!signal.aborted) {
        setLogsContent(response.content || 'No logs available');
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return;
      devError('Failed to fetch run logs:', error);
      if (!signal.aborted) {
        setLogsContent('Failed to load logs');
      }
    } finally {
      if (!signal.aborted) {
        setIsLoadingLogs(false);
      }
    }
  }, [activeProject]);

  // Fetch run details (metrics)
  const fetchRunDetails = useCallback(async (run: Record<string, unknown>) => {
    if (metricsAbortControllerRef.current) {
      metricsAbortControllerRef.current.abort();
    }
    metricsAbortControllerRef.current = new AbortController();
    const signal = metricsAbortControllerRef.current.signal;

    setIsLoadingMetrics(true);
    setSelectedRunDetail({...run, project: activeProject});
    setRunMetricsData([]);

    try {
      const response = await apiClient.getRunMetrics(
        (run.project as string) || activeProject || '',
        run.run_name as string,
        { signal }
      );
      if (!signal.aborted) {
        setRunMetricsData(response.data || []);
        setSelectedRunDetail({...run, project: activeProject});
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return;
      devError('Failed to fetch run metrics:', error);
      setSelectedRunDetail({...run, project: activeProject});
      setRunMetricsData([]);
    } finally {
      if (!signal.aborted) {
        setIsLoadingMetrics(false);
      }
    }
  }, [activeProject]);

  // Column mapping functions
  const loadColumnMappings = useCallback(async () => {
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
  }, [activeProject]);

  const generateSuggestions = useCallback(async (columnType: 'hyperparam' | 'metric' = 'hyperparam') => {
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
  }, [activeProject]);

  const handleAcceptSuggestion = useCallback(async (suggestion: ColumnSuggestion) => {
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
      // Remove from suggestions and refresh mappings
      setColumnSuggestions(prev => prev.filter(s =>
        s.source_column !== suggestion.source_column || s.target_column !== suggestion.target_column
      ));
      await loadColumnMappings();
    } catch (error) {
      devError('Failed to accept suggestion:', error);
    }
  }, [activeProject, loadColumnMappings]);

  const handleRejectSuggestion = useCallback((suggestion: ColumnSuggestion) => {
    setColumnSuggestions(prev => prev.filter(s =>
      s.source_column !== suggestion.source_column || s.target_column !== suggestion.target_column
    ));
  }, []);

  const handleCreateMapping = useCallback(async (mapping: ColumnMappingCreate) => {
    if (!activeProject) return;
    try {
      await createColumnMapping(activeProject, mapping);
      await loadColumnMappings();
    } catch (error) {
      devError('Failed to create mapping:', error);
    }
  }, [activeProject, loadColumnMappings]);

  const handleUpdateMapping = useCallback(async (id: number, target: string) => {
    if (!activeProject) return;
    try {
      await updateColumnMapping(activeProject, id, { target_column: target });
      await loadColumnMappings();
    } catch (error) {
      devError('Failed to update mapping:', error);
    }
  }, [activeProject, loadColumnMappings]);

  const handleDeleteMapping = useCallback(async (id: number) => {
    if (!activeProject) return;
    try {
      await deleteColumnMapping(activeProject, id);
      await loadColumnMappings();
    } catch (error) {
      devError('Failed to delete mapping:', error);
    }
  }, [activeProject, loadColumnMappings]);

  // Load column mappings on project change
  useEffect(() => {
    if (activeProject) {
      loadColumnMappings();
      generateSuggestions('hyperparam');
    }
  }, [activeProject, loadColumnMappings, generateSuggestions]);

  return {
    runDetailState: {
      selectedRunDetail,
      runMetricsData,
      configContent,
      logsContent,
      isLoadingMetrics,
      isLoadingConfig,
      isLoadingLogs,
      logSearch,
      runDetailTab,
    },
    columnMappingState: {
      columnMappings,
      columnSuggestions,
      isColumnMappingDialogOpen,
      isLoadingColumnMappings,
      isGeneratingSuggestions,
    },

    // Run detail actions
    setSelectedRunDetail,
    setRunDetailTab,
    setLogSearch,
    fetchRunDetails,
    fetchRunConfiguration,
    fetchRunLogs,

    // Selection actions
    selectedExperiments,
    setSelectedExperiments,
    clearSelection,

    // Column mapping actions
    setIsColumnMappingDialogOpen,
    loadColumnMappings,
    generateSuggestions,
    handleAcceptSuggestion,
    handleRejectSuggestion,
    handleCreateMapping,
    handleUpdateMapping,
    handleDeleteMapping,
  };
}
