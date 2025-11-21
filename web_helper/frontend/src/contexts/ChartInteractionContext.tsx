/**
 * Chart Interaction Context
 *
 * Provides global state and handlers for chart interactions:
 * - Drill-down to run details
 * - Cross-chart selection and highlighting
 * - Brushing and filtering
 * - Synchronized hover states
 */

import React, { createContext, useContext, useState, useCallback, useMemo, ReactNode } from 'react';
import type { Run } from '../lib/types';

interface ChartInteractionState {
  // Selected run for detail view
  selectedRun: Run | null;
  selectedRunName: string | null;

  // Highlighted runs (hover state)
  highlightedRuns: Set<string>;

  // Brushed/filtered data range
  brushedRange: [number, number] | null;
  brushedMetric: string | null;

  // Focused metric pair (for correlation drill-down)
  focusedMetricPair: { x: string; y: string } | null;
}

interface ChartInteractionContextValue extends ChartInteractionState {
  // Actions
  selectRun: (run: Run | null) => void;
  selectRunByName: (runName: string | null) => void;
  highlightRuns: (runNames: string[]) => void;
  clearHighlight: () => void;
  setBrushedRange: (range: [number, number] | null, metric?: string) => void;
  setFocusedMetricPair: (pair: { x: string; y: string } | null) => void;
  reset: () => void;
}

const ChartInteractionContext = createContext<ChartInteractionContextValue | undefined>(
  undefined
);

interface ChartInteractionProviderProps {
  children: ReactNode;
}

export function ChartInteractionProvider({ children }: ChartInteractionProviderProps) {
  const [state, setState] = useState<ChartInteractionState>({
    selectedRun: null,
    selectedRunName: null,
    highlightedRuns: new Set(),
    brushedRange: null,
    brushedMetric: null,
    focusedMetricPair: null,
  });

  // Select a run for detail view
  const selectRun = useCallback((run: Run | null) => {
    setState(prev => ({
      ...prev,
      selectedRun: run,
      selectedRunName: run?.run_name || null,
    }));
  }, []);

  // Select run by name (when full Run object not available)
  const selectRunByName = useCallback((runName: string | null) => {
    setState(prev => ({
      ...prev,
      selectedRunName: runName,
    }));
  }, []);

  // Highlight runs (for hover effects)
  const highlightRuns = useCallback((runNames: string[]) => {
    setState(prev => ({
      ...prev,
      highlightedRuns: new Set(runNames),
    }));
  }, []);

  // Clear all highlights
  const clearHighlight = useCallback(() => {
    setState(prev => ({
      ...prev,
      highlightedRuns: new Set(),
    }));
  }, []);

  // Set brushed range
  const setBrushedRange = useCallback(
    (range: [number, number] | null, metric?: string) => {
      setState(prev => ({
        ...prev,
        brushedRange: range,
        brushedMetric: metric || prev.brushedMetric,
      }));
    },
    []
  );

  // Set focused metric pair
  const setFocusedMetricPair = useCallback(
    (pair: { x: string; y: string } | null) => {
      setState(prev => ({
        ...prev,
        focusedMetricPair: pair,
      }));
    },
    []
  );

  // Reset all interaction state
  const reset = useCallback(() => {
    setState({
      selectedRun: null,
      selectedRunName: null,
      highlightedRuns: new Set(),
      brushedRange: null,
      brushedMetric: null,
      focusedMetricPair: null,
    });
  }, []);

  const value: ChartInteractionContextValue = {
    ...state,
    selectRun,
    selectRunByName,
    highlightRuns,
    clearHighlight,
    setBrushedRange,
    setFocusedMetricPair,
    reset,
  };

  return (
    <ChartInteractionContext.Provider value={value}>
      {children}
    </ChartInteractionContext.Provider>
  );
}

/**
 * Hook to use chart interaction context
 */
export function useChartInteraction() {
  const context = useContext(ChartInteractionContext);
  if (!context) {
    throw new Error('useChartInteraction must be used within ChartInteractionProvider');
  }
  return context;
}

/**
 * Hook for run selection with detail modal
 */
export function useRunDetail(runs: Run[]) {
  const { selectedRunName, selectRun, selectRunByName } = useChartInteraction();

  const selectedRun = useMemo(() => {
    if (!selectedRunName) return null;
    return runs.find(r => r.run_name === selectedRunName) || null;
  }, [runs, selectedRunName]);

  const openRunDetail = useCallback(
    (runName: string) => {
      const run = runs.find(r => r.run_name === runName);
      if (run) {
        selectRun(run);
      } else {
        selectRunByName(runName);
      }
    },
    [runs, selectRun, selectRunByName]
  );

  const closeRunDetail = useCallback(() => {
    selectRun(null);
  }, [selectRun]);

  return {
    selectedRun,
    openRunDetail,
    closeRunDetail,
  };
}

/**
 * Hook for chart highlighting
 */
export function useChartHighlight() {
  const { highlightedRuns, highlightRuns, clearHighlight } = useChartInteraction();

  const isHighlighted = useCallback(
    (runName: string) => {
      return highlightedRuns.has(runName) || highlightedRuns.size === 0;
    },
    [highlightedRuns]
  );

  return {
    highlightedRuns,
    highlightRuns,
    clearHighlight,
    isHighlighted,
  };
}

/**
 * Hook for brushing interactions
 */
export function useChartBrushing() {
  const { brushedRange, brushedMetric, setBrushedRange } = useChartInteraction();

  const isInBrush = useCallback(
    (value: number) => {
      if (!brushedRange) return true;
      const [min, max] = brushedRange;
      return value >= min && value <= max;
    },
    [brushedRange]
  );

  const clearBrush = useCallback(() => {
    setBrushedRange(null);
  }, [setBrushedRange]);

  return {
    brushedRange,
    brushedMetric,
    setBrushedRange,
    isInBrush,
    clearBrush,
  };
}
