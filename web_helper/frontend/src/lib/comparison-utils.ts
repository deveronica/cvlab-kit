import { Run, ParameterImpact, ConvergenceInfo } from './types';

/**
 * Flatten nested object into dot notation
 * Example: { model: { name: "resnet18" } } -> { "model.name": "resnet18" }
 */
export function flattenObject(obj: Record<string, any>, prefix = ''): Record<string, any> {
  const flattened: Record<string, any> = {};

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const newKey = prefix ? `${prefix}.${key}` : key;

      if (obj[key] !== null && typeof obj[key] === 'object' && !Array.isArray(obj[key])) {
        Object.assign(flattened, flattenObject(obj[key], newKey));
      } else {
        flattened[newKey] = obj[key];
      }
    }
  }

  return flattened;
}

/**
 * Unflatten dot notation back to nested object
 * Example: { "model.name": "resnet18" } -> { model: { name: "resnet18" } }
 */
export function unflattenObject(obj: Record<string, any>): Record<string, any> {
  const unflattened: Record<string, any> = {};

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const keys = key.split('.');
      let current = unflattened;

      for (let i = 0; i < keys.length - 1; i++) {
        if (!(keys[i] in current)) {
          current[keys[i]] = {};
        }
        current = current[keys[i]];
      }

      current[keys[keys.length - 1]] = obj[key];
    }
  }

  return unflattened;
}

/**
 * Get all unique parameter keys from multiple runs
 */
export function getAllParameterKeys(runs: Run[]): string[] {
  const allKeys = new Set<string>();

  runs.forEach(run => {
    if (run.hyperparameters) {
      Object.keys(run.hyperparameters).forEach(key => allKeys.add(key));
    }
    if (run.config) {
      Object.keys(run.config).forEach(key => allKeys.add(key));
    }
  });

  return Array.from(allKeys).sort();
}

/**
 * Get all unique metric keys from multiple runs
 */
export function getAllMetricKeys(runs: Run[]): string[] {
  const allKeys = new Set<string>();

  runs.forEach(run => {
    if (run.metrics) {
      if (run.metrics.final) Object.keys(run.metrics.final).forEach(key => allKeys.add(`${key}_final`));
      if (run.metrics.max) Object.keys(run.metrics.max).forEach(key => allKeys.add(`${key}_max`));
      if (run.metrics.min) Object.keys(run.metrics.min).forEach(key => allKeys.add(`${key}_min`));
      if (run.metrics.mean) Object.keys(run.metrics.mean).forEach(key => allKeys.add(`${key}_mean`));
    }
  });

  return Array.from(allKeys).sort();
}

/**
 * Filter parameters that differ between runs (for diff mode)
 */
export function getDifferingParameters(runs: Run[]): string[] {
  if (runs.length <= 1) return [];

  const allParams = getAllParameterKeys(runs);
  const differingParams: string[] = [];

  allParams.forEach(param => {
    const values = new Set();
    runs.forEach(run => {
      const value = getParameterValue(run, param);
      values.add(JSON.stringify(value)); // Use JSON.stringify to handle objects
    });

    if (values.size > 1) {
      differingParams.push(param);
    }
  });

  return differingParams;
}

/**
 * Get parameter value from run (supports both hyperparameters and config)
 */
export function getParameterValue(run: Run, paramKey: string): any {
  // Try hyperparameters first
  if (run.hyperparameters && paramKey in run.hyperparameters) {
    return run.hyperparameters[paramKey];
  }

  // Fall back to config
  if (run.config && paramKey in run.config) {
    return run.config[paramKey];
  }

  // Support dot notation
  if (paramKey.includes('.')) {
    const keys = paramKey.split('.');
    let value = run.hyperparameters || run.config;

    for (const key of keys) {
      if (value && typeof value === 'object' && key in value) {
        value = value[key];
      } else {
        return undefined;
      }
    }

    return value;
  }

  return undefined;
}

/**
 * Get metric value from run
 */
export function getMetricValue(run: Run, metricKey: string): number | undefined {
  if (!run.metrics) return undefined;

  // Parse metric key (e.g., "val_acc_final" -> type: "final", metric: "val_acc")
  const parts = metricKey.split('_');
  const type = parts[parts.length - 1]; // final, max, min, mean
  const metric = parts.slice(0, -1).join('_');

  switch (type) {
    case 'final':
      return run.metrics.final?.[metric];
    case 'max':
      return run.metrics.max?.[metric];
    case 'min':
      return run.metrics.min?.[metric];
    case 'mean':
      return run.metrics.mean?.[metric];
    default:
      // Direct metric name
      return run.metrics.final?.[metricKey];
  }
}

/**
 * Calculate parameter impact on a specific metric
 */
export function calculateParameterImpact(runs: Run[], targetMetric: string): ParameterImpact[] {
  const paramKeys = getDifferingParameters(runs);
  const impacts: ParameterImpact[] = [];

  paramKeys.forEach(param => {
    const paramValues: number[] = [];
    const metricValues: number[] = [];

    runs.forEach(run => {
      const paramValue = getParameterValue(run, param);
      const metricValue = getMetricValue(run, targetMetric);

      if (paramValue !== undefined && metricValue !== undefined) {
        // Convert parameter to number if possible
        const numericParam = typeof paramValue === 'number' ? paramValue : parseFloat(paramValue);
        if (!isNaN(numericParam)) {
          paramValues.push(numericParam);
          metricValues.push(metricValue);
        }
      }
    });

    if (paramValues.length >= 2) {
      const correlation = calculateCorrelation(paramValues, metricValues);
      const variance = calculateVariance(paramValues);

      impacts.push({
        parameter: param,
        impact: getImpactLevel(Math.abs(correlation)),
        correlation,
        variance
      });
    }
  });

  return impacts.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
}

/**
 * Calculate correlation coefficient between two arrays
 */
function calculateCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length === 0) return 0;

  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
  const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
  const sumY2 = y.reduce((acc, yi) => acc + yi * yi, 0);

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  return denominator === 0 ? 0 : numerator / denominator;
}

/**
 * Calculate variance of an array
 */
function calculateVariance(values: number[]): number {
  if (values.length === 0) return 0;

  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;

  return variance;
}

/**
 * Determine impact level based on correlation strength
 */
function getImpactLevel(absCorrelation: number): 'high' | 'medium' | 'low' {
  if (absCorrelation >= 0.7) return 'high';
  if (absCorrelation >= 0.3) return 'medium';
  return 'low';
}

/**
 * Analyze convergence for each run
 */
export function analyzeConvergence(runs: Run[], targetMetric: string): ConvergenceInfo[] {
  const analysis: ConvergenceInfo[] = [];

  runs.forEach(run => {
    if (!run.metrics?.timeseries || run.metrics.timeseries.length === 0) {
      analysis.push({
        run_name: run.run_name,
        converged: false,
        final_value: getMetricValue(run, targetMetric) || 0,
        convergence_speed: 'slow'
      });
      return;
    }

    const timeseries = run.metrics.timeseries;
    const metricName = targetMetric.replace(/_final$|_max$|_min$|_mean$/, '');
    const values = timeseries.map(t => t.values[metricName]).filter(v => v !== undefined);

    if (values.length === 0) {
      analysis.push({
        run_name: run.run_name,
        converged: false,
        final_value: 0,
        convergence_speed: 'slow'
      });
      return;
    }

    // Simple convergence detection: check if the metric stabilizes
    const windowSize = Math.min(10, Math.floor(values.length / 4));
    const recentValues = values.slice(-windowSize);
    const variance = calculateVariance(recentValues);
    const finalValue = values[values.length - 1];

    const converged = variance < 0.001; // Threshold for convergence
    const convergenceEpoch = converged ? findConvergenceEpoch(values, variance) : undefined;
    const speed = getConvergenceSpeed(convergenceEpoch, values.length);

    analysis.push({
      run_name: run.run_name,
      converged,
      convergence_epoch: convergenceEpoch,
      final_value: finalValue,
      convergence_speed: speed
    });
  });

  return analysis;
}

/**
 * Find the epoch where convergence started
 */
function findConvergenceEpoch(values: number[], targetVariance: number): number | undefined {
  const windowSize = Math.min(10, Math.floor(values.length / 4));

  for (let i = windowSize; i <= values.length; i++) {
    const window = values.slice(i - windowSize, i);
    const variance = calculateVariance(window);

    if (variance <= targetVariance) {
      return i - windowSize;
    }
  }

  return undefined;
}

/**
 * Determine convergence speed
 */
function getConvergenceSpeed(convergenceEpoch: number | undefined, totalEpochs: number): 'fast' | 'medium' | 'slow' {
  if (!convergenceEpoch) return 'slow';

  const ratio = convergenceEpoch / totalEpochs;
  if (ratio < 0.3) return 'fast';
  if (ratio < 0.7) return 'medium';
  return 'slow';
}

/**
 * Generate comparison name based on selected runs
 */
export function generateComparisonName(runs: Run[]): string {
  if (runs.length === 0) return 'Empty Comparison';
  if (runs.length === 1) return `Single Run: ${runs[0].run_name.substring(0, 8)}`;
  if (runs.length === 2) return `${runs[0].run_name.substring(0, 8)} vs ${runs[1].run_name.substring(0, 8)}`;

  return `${runs.length} Runs Comparison`;
}