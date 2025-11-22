/**
 * Data Adapter
 *
 * Converts backend API responses to the format expected by chart components.
 * Handles the mismatch between flat structure (final_metrics) and nested structure (metrics.final).
 */

import type { Run } from './types';

/**
 * Convert experiment data from API response to chart-compatible Run format
 */
export function adaptExperimentToRun(experiment: any, project: string): Run {
  return {
    run_name: experiment.run_name,
    project: project,
    status: experiment.status || 'completed',
    started_at: experiment.started_at,
    finished_at: experiment.finished_at,
    hyperparameters: experiment.hyperparameters || {},

    // Convert flat structure to nested structure
    metrics: {
      final: experiment.final_metrics || {},
      max: experiment.max_metrics || {},
      min: experiment.min_metrics || {},
      mean: experiment.mean_metrics || {},
      timeseries: [], // Will be loaded separately
    },

    config: {},
  };
}

/**
 * Batch convert multiple experiments
 */
export function adaptExperiments(experiments: any[], project: string): Run[] {
  return experiments.map(exp => adaptExperimentToRun(exp, project));
}

/**
 * Convert timeseries API response to Run metrics format
 */
export function adaptTimeseriesData(apiData: any[]): import('./types').TimeseriesMetric[] {
  if (!apiData || apiData.length === 0) return [];

  return apiData.map((point, index) => {
    const { step, epoch, ...values } = point;

    return {
      step: step ?? index,
      epoch: epoch ?? 0,
      values: values,
    };
  });
}

/**
 * Merge timeseries data into a Run object
 */
export function mergeTimeseriesIntoRun(run: Run, timeseriesData: any[]): Run {
  return {
    ...run,
    metrics: {
      final: run.metrics?.final || {},
      max: run.metrics?.max || {},
      min: run.metrics?.min || {},
      mean: run.metrics?.mean || {},
      timeseries: adaptTimeseriesData(timeseriesData),
    },
  };
}
