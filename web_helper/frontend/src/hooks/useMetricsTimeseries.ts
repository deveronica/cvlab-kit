/**
 * useMetricsTimeseries Hook
 *
 * Loads timeseries data for multiple runs in parallel with React Query caching.
 * Merges timeseries into Run objects for chart visualization.
 */

import { useQueries } from '@tanstack/react-query';
import { useMemo } from 'react';
import { apiClient } from '../lib/api-client';
import { mergeTimeseriesIntoRun } from '../lib/dataAdapter';
import type { Run } from '../lib/types';

interface UseMetricsTimeseriesOptions {
  /** Downsample data points for performance (default: 500) */
  downsample?: number;
  /** Enable/disable automatic fetching (default: true) */
  enabled?: boolean;
  /** Cache time in milliseconds (default: 5 minutes) */
  staleTime?: number;
}

interface UseMetricsTimeseriesResult {
  /** Runs with timeseries data merged */
  runs: Run[];
  /** True if any query is loading */
  isLoading: boolean;
  /** True if any query has error */
  hasError: boolean;
  /** Array of errors from failed queries */
  errors: Error[];
  /** True if all queries succeeded */
  isSuccess: boolean;
}

/**
 * Fetch and merge timeseries data for multiple runs
 *
 * @example
 * ```tsx
 * const { runs, isLoading } = useMetricsTimeseries(selectedRuns, {
 *   downsample: 500,
 *   enabled: selectedRuns.length > 0
 * });
 *
 * <AdvancedMetricsChart runs={runs} metricKey="val/acc" />
 * ```
 */
export function useMetricsTimeseries(
  runs: Run[],
  options: UseMetricsTimeseriesOptions = {}
): UseMetricsTimeseriesResult {
  const {
    downsample = 500,
    enabled = true,
    staleTime = 5 * 60 * 1000, // 5 minutes
  } = options;

  // Create parallel queries for each run
  const queries = useQueries({
    queries: runs.map(run => ({
      queryKey: ['metrics', 'timeseries', run.project, run.run_name, downsample],
      queryFn: async () => {
        const response = await apiClient.getRunMetrics(
          run.project,
          run.run_name,
          downsample
        );
        return {
          runName: run.run_name,
          data: response.data,
          metadata: {
            file_path: response.file_path,
            total_points: response.total_steps,
            downsampled: downsample ? response.total_steps < response.data.length : false,
            columns: response.columns,
          },
        };
      },
      enabled: enabled && !!run.project && !!run.run_name,
      staleTime,
      retry: 2,
      retryDelay: 1000,
    })),
  });

  // Merge timeseries data into runs
  const runsWithTimeseries = useMemo(() => {
    return runs.map((run, index) => {
      const query = queries[index];

      // If data is available, merge it
      if (query.isSuccess && query.data) {
        return mergeTimeseriesIntoRun(run, query.data.data);
      }

      // Otherwise return original run
      return run;
    });
  }, [runs, queries]);

  // Aggregate loading/error states
  const isLoading = queries.some(q => q.isLoading);
  const hasError = queries.some(q => q.isError);
  const isSuccess = queries.every(q => q.isSuccess);
  const errors = queries
    .filter(q => q.error)
    .map(q => q.error as Error);

  return {
    runs: runsWithTimeseries,
    isLoading,
    hasError,
    errors,
    isSuccess,
  };
}
