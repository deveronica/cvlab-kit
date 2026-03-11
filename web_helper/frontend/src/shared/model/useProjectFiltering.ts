import { useState, useMemo, useEffect } from 'react';
import { extractColumns, getValue } from '@/shared/lib/table-columns';
import { applyFilters, type FilterRule } from '@/shared/../shared/ui/advanced-filter';
import { adaptExperiments } from '@/shared/lib/dataAdapter';

export interface FilteringState {
  filters: FilterRule[];
  flattenParams: boolean;
  diffMode: boolean;
  visibleHyperparams: Set<string>;
  visibleMetrics: Set<string>;
  pinnedLeftHyperparams: Set<string>;
  pinnedRightMetrics: Set<string>;
  hyperparamOrder: string[];
  metricOrder: string[];
}

export interface FilteringActions {
  setFilters: (filters: FilterRule[]) => void;
  setFlattenParams: (value: boolean) => void;
  setDiffMode: (value: boolean) => void;
  setVisibleHyperparams: (value: Set<string>) => void;
  setVisibleMetrics: (value: Set<string>) => void;
  setPinnedLeftHyperparams: (value: Set<string>) => void;
  setPinnedRightMetrics: (value: Set<string>) => void;
  setHyperparamOrder: (value: string[]) => void;
  setMetricOrder: (value: string[]) => void;
}

export interface FilteringResult {
  state: FilteringState;
  actions: FilteringActions;
  experimentsData: Record<string, unknown>[];
  hyperparamColumns: string[];
  metricColumns: string[];
}

interface UseProjectFilteringOptions {
  experimentData?: Record<string, unknown>[];
}

/**
 * Hook for managing project filtering and column visibility state.
 * Extracted from useProjectsView for better separation of concerns.
 */
export function useProjectFiltering(options: UseProjectFilteringOptions): FilteringResult {
  const { experimentData = [] } = options;

  // Filtering state
  const [filters, setFilters] = useState<FilterRule[]>([]);
  const [flattenParams, setFlattenParams] = useState(false);
  const [diffMode, setDiffMode] = useState(false);

  // Column visibility state
  const [visibleHyperparams, setVisibleHyperparams] = useState<Set<string>>(new Set());
  const [visibleMetrics, setVisibleMetrics] = useState<Set<string>>(new Set());
  const [pinnedLeftHyperparams, setPinnedLeftHyperparams] = useState<Set<string>>(new Set());
  const [pinnedRightMetrics, setPinnedRightMetrics] = useState<Set<string>>(new Set());
  const [hyperparamOrder, setHyperparamOrder] = useState<string[]>([]);
  const [metricOrder, setMetricOrder] = useState<string[]>([]);

  // Apply filters to experiment data
  const experimentsData = useMemo(() => {
    let data = experimentData || [];

    if (filters.length > 0) {
      data = applyFilters(data, filters);
    }

    return data;
  }, [experimentData, filters]);

  // Extract columns from experiments
  const { hyperparamColumns, metricColumns } = useMemo(() => {
    if (experimentsData.length === 0) return { hyperparamColumns: [], metricColumns: [] };

    const allHyperparams = new Set<string>();
    const allMetrics = new Set<string>();

    experimentsData.forEach(exp => {
      if (exp.hyperparameters) {
        const cols = extractColumns(exp.hyperparameters as Record<string, unknown>, flattenParams);
        cols.forEach(key => allHyperparams.add(key));
      }
      if (exp.final_metrics) {
        const cols = extractColumns(exp.final_metrics as Record<string, unknown>, flattenParams);
        cols.forEach(key => {
          if (key !== 'step' && key !== 'epoch') {
            allMetrics.add(key);
          }
        });
      }
    });

    let hyperparamList = Array.from(allHyperparams);
    let metricList = Array.from(allMetrics);

    // In diff mode, only show columns with different values
    if (diffMode && experimentsData.length > 1) {
      hyperparamList = hyperparamList.filter(param => {
        const values = experimentsData.map(exp =>
          getValue((exp.hyperparameters || {}) as Record<string, unknown>, param, flattenParams)
        );
        return new Set(values.map(v => JSON.stringify(v))).size > 1;
      });

      metricList = metricList.filter(metric => {
        const values = experimentsData.map(exp =>
          getValue((exp.final_metrics || {}) as Record<string, unknown>, metric, flattenParams)
        );
        return new Set(values.map(v => JSON.stringify(v))).size > 1;
      });
    }

    // Add Run ID, Status, and Tags at the beginning (always visible columns)
    return {
      hyperparamColumns: ['Run ID', 'Status', 'Tags', ...hyperparamList],
      metricColumns: metricList
    };
  }, [experimentsData, diffMode, flattenParams]);

  // Sync visible hyperparams with available columns
  useEffect(() => {
    if (hyperparamColumns.length > 0) {
      setVisibleHyperparams(new Set(hyperparamColumns));
      setHyperparamOrder(prev => {
        if (prev.length === 0) return hyperparamColumns;
        const existing = prev.filter(col => hyperparamColumns.includes(col));
        const newCols = hyperparamColumns.filter(col => !prev.includes(col));
        return [...existing, ...newCols];
      });
    }
  }, [hyperparamColumns]);

  // Sync visible metrics with available columns
  useEffect(() => {
    if (metricColumns.length > 0) {
      setVisibleMetrics(new Set(metricColumns));
      setMetricOrder(prev => {
        if (prev.length === 0) return metricColumns;
        const existing = prev.filter(col => metricColumns.includes(col));
        const newCols = metricColumns.filter(col => !prev.includes(col));
        return [...existing, ...newCols];
      });
    }
  }, [metricColumns]);

  return {
    state: {
      filters,
      flattenParams,
      diffMode,
      visibleHyperparams,
      visibleMetrics,
      pinnedLeftHyperparams,
      pinnedRightMetrics,
      hyperparamOrder,
      metricOrder,
    },
    actions: {
      setFilters,
      setFlattenParams,
      setDiffMode,
      setVisibleHyperparams,
      setVisibleMetrics,
      setPinnedLeftHyperparams,
      setPinnedRightMetrics,
      setHyperparamOrder,
      setMetricOrder,
    },
    experimentsData,
    hyperparamColumns,
    metricColumns,
  };
}
