/**
 * Statistics calculation utilities for Results View
 */

export interface MetricStats {
  min: number;
  max: number;
  mean: number;
  count: number;
}

export interface RunMetrics {
  [key: string]: number;
}

/**
 * Calculate statistics for a list of numeric values
 */
export const calculateStats = (values: number[]): MetricStats => {
  if (values.length === 0) {
    return { min: 0, max: 0, mean: 0, count: 0 };
  }

  const validValues = values.filter(v => typeof v === 'number' && !isNaN(v));

  if (validValues.length === 0) {
    return { min: 0, max: 0, mean: 0, count: 0 };
  }

  return {
    min: Math.min(...validValues),
    max: Math.max(...validValues),
    mean: validValues.reduce((a, b) => a + b, 0) / validValues.length,
    count: validValues.length
  };
};

/**
 * Extract final metrics from a run's metrics data
 * Assumes metrics data is in format: [{ step: 0, train_loss: 0.5, val_acc: 0.8 }, ...]
 */
export const extractFinalMetrics = (metricsData: any[]): RunMetrics => {
  if (!Array.isArray(metricsData) || metricsData.length === 0) {
    return {};
  }

  // Get the last entry which should have the final metrics
  const lastEntry = metricsData[metricsData.length - 1];
  const metrics: RunMetrics = {};

  Object.keys(lastEntry).forEach(key => {
    if (key !== 'step' && key !== 'runName' && typeof lastEntry[key] === 'number') {
      metrics[key] = lastEntry[key];
    }
  });

  return metrics;
};

/**
 * Get common metrics across all runs
 */
export const getCommonMetrics = (runsMetrics: Record<string, RunMetrics>): string[] => {
  const allMetricKeys = Object.values(runsMetrics)
    .map(metrics => Object.keys(metrics))
    .flat();

  const metricCounts = allMetricKeys.reduce((acc, metric) => {
    acc[metric] = (acc[metric] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const totalRuns = Object.keys(runsMetrics).length;

  // Return metrics that appear in at least 50% of runs
  return Object.keys(metricCounts)
    .filter(metric => metricCounts[metric] >= Math.max(1, totalRuns * 0.5))
    .sort();
};

/**
 * Format number for display in statistics columns
 */
export const formatStatValue = (value: number, precision: number = 3): string => {
  if (value === 0) return '0';
  if (Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(precision);
};