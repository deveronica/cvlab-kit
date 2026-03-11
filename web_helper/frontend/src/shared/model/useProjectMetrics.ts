import { useState, useMemo, useEffect } from 'react';
import { devError } from '@/shared/lib/utils';
import {
  getOutlierSummary,
  type OutlierSummaryResponse,
  type OutlierMethod,
} from '@/shared/api/outliers';

export interface BestRun {
  run_name: string;
  project: string;
  status: string;
  started_at: string;
  finished_at: string;
  hyperparameters: Record<string, unknown>;
  metrics: {
    final: Record<string, unknown>;
    max: Record<string, unknown>;
    min: Record<string, unknown>;
    mean: Record<string, unknown>;
  };
}

export interface MetricsState {
  selectedMetric: string;
  metricDirection: 'maximize' | 'minimize';
  outlierMethod: OutlierMethod;
  outlierThreshold: number;
  outlierSummary: OutlierSummaryResponse | null;
  cellOutliers: Set<string>;
  isLoadingOutliers: boolean;
  highlightedRun: string | null;
}

export interface MetricsActions {
  setSelectedMetric: (value: string) => void;
  setMetricDirection: (value: 'maximize' | 'minimize') => void;
  setOutlierMethod: (value: OutlierMethod) => void;
  setOutlierThreshold: (value: number) => void;
  setHighlightedRun: (value: string | null) => void;
  loadOutlierSummary: () => Promise<void>;
}

export interface MetricsResult {
  state: MetricsState;
  actions: MetricsActions;
  getBestRun: BestRun | null;
}

interface UseProjectMetricsOptions {
  activeProject?: string | null;
  experimentsData: Record<string, unknown>[];
  metricColumns: string[];
}

/**
 * Hook for managing metric calculations, best run detection, and outlier analysis.
 * Extracted from useProjectsView for better separation of concerns.
 */
export function useProjectMetrics(options: UseProjectMetricsOptions): MetricsResult {
  const { activeProject, experimentsData, metricColumns } = options;

  // Metric selection state
  const [selectedMetric, setSelectedMetric] = useState<string>('val/acc');
  const [metricDirection, setMetricDirection] = useState<'maximize' | 'minimize'>('maximize');

  // Outlier detection state
  const [outlierSummary, setOutlierSummary] = useState<OutlierSummaryResponse | null>(null);
  const [outlierMethod, setOutlierMethod] = useState<OutlierMethod>('iqr');
  const [outlierThreshold, setOutlierThreshold] = useState(1.5);
  const [isLoadingOutliers, setIsLoadingOutliers] = useState(false);
  const [cellOutliers, setCellOutliers] = useState<Set<string>>(new Set());

  // Highlight state
  const [highlightedRun, setHighlightedRun] = useState<string | null>(null);

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
  }, [selectedMetric]);

  // Calculate best run
  const getBestRun = useMemo((): BestRun | null => {
    if (experimentsData.length === 0) return null;

    // Use user-selected metric or fallback to common defaults
    let keyMetric = selectedMetric;

    // Determine which metrics source to use based on direction
    const metricsSource = metricDirection === 'maximize' ? 'max_metrics' : 'min_metrics';

    // Verify selected metric exists in data, otherwise fallback
    const hasSelectedMetric = experimentsData.some(run => {
      const metrics = run[metricsSource] as Record<string, unknown> | undefined;
      return metrics && typeof metrics[keyMetric] === 'number';
    });

    if (!hasSelectedMetric) {
      const fallbackMetrics = ['val/acc', 'test/acc', 'accuracy', 'acc', 'test_acc', 'val_acc'];
      keyMetric = fallbackMetrics.find(metric =>
        experimentsData.some(run => {
          const metrics = run[metricsSource] as Record<string, unknown> | undefined;
          return metrics && typeof metrics[metric] === 'number';
        })
      ) || metricColumns[0] || 'val/acc';
    }

    let bestRun: Record<string, unknown> | null = null;
    let bestValue = metricDirection === 'maximize' ? -Infinity : Infinity;

    experimentsData.forEach(run => {
      const metricsToCheck = run[metricsSource] as Record<string, unknown> | undefined;
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
      bestRun = experimentsData.find(run =>
        run.status === 'completed' && run.final_metrics
      ) as Record<string, unknown> | undefined ?? null;
    }

    // Transform to BestRun type format
    if (bestRun) {
      return {
        run_name: bestRun.run_name as string,
        project: activeProject || '',
        status: bestRun.status as string,
        started_at: bestRun.started_at as string,
        finished_at: bestRun.finished_at as string,
        hyperparameters: (bestRun.hyperparameters || {}) as Record<string, unknown>,
        metrics: {
          final: (bestRun.final_metrics || {}) as Record<string, unknown>,
          max: (bestRun.max_metrics || {}) as Record<string, unknown>,
          min: (bestRun.min_metrics || {}) as Record<string, unknown>,
          mean: (bestRun.mean_metrics || {}) as Record<string, unknown>,
        }
      };
    }

    return null;
  }, [experimentsData, activeProject, selectedMetric, metricDirection, metricColumns]);

  // Load outlier summary
  const loadOutlierSummary = async () => {
    if (!activeProject) return;
    setIsLoadingOutliers(true);
    try {
      const summary = await getOutlierSummary(activeProject, outlierMethod, outlierThreshold);
      setOutlierSummary(summary);
      // Parse cell_outliers object into Set for efficient lookup
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
    state: {
      selectedMetric,
      metricDirection,
      outlierMethod,
      outlierThreshold,
      outlierSummary,
      cellOutliers,
      isLoadingOutliers,
      highlightedRun,
    },
    actions: {
      setSelectedMetric,
      setMetricDirection,
      setOutlierMethod,
      setOutlierThreshold,
      setHighlightedRun,
      loadOutlierSummary,
    },
    getBestRun,
  };
}
