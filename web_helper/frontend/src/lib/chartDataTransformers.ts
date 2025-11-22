/**
 * Chart Data Transformers
 *
 * Transforms Run data into chart-specific formats:
 * - Correlation data (scatter plot)
 * - Distribution data (histogram)
 * - Parallel coordinates data
 * - Scatter matrix data
 */

import type { Run } from './types';
import {
  calculateLinearRegression,
  calculatePearsonCorrelation,
  createHistogramBins,
  calculateStatistics,
  normalize,
  detectOutliers,
  getNestedValue,
  type Point,
  type RegressionResult,
  type Statistics,
  type HistogramBin,
} from './statistics';

export interface CorrelationDataPoint extends Point {
  runName: string;
  isOutlier?: boolean;
}

export interface CorrelationData {
  points: CorrelationDataPoint[];
  regression: RegressionResult;
  pearsonR: number;
  stats: {
    x: Statistics;
    y: Statistics;
  };
}

export interface DistributionData {
  bins: HistogramBin[];
  stats: Statistics;
  outliers: number[];
}

export interface ParallelDimension {
  name: string;
  values: number[];
  normalized: number[];
  min: number;
  max: number;
}

export interface ParallelData {
  runs: {
    runName: string;
    values: Record<string, number>;
  }[];
  dimensions: ParallelDimension[];
}

export interface ScatterMatrixCell {
  metricX: string;
  metricY: string;
  points: CorrelationDataPoint[];
  pearsonR?: number;
}

export interface ScatterMatrixData {
  metrics: string[];
  cells: ScatterMatrixCell[][];
}

/**
 * Transform runs data for correlation chart
 */
export function transformForCorrelation(
  runs: Run[],
  hyperparamKey: string,
  metricKey: string,
  metricDisplayMode: 'final' | 'max' | 'min' = 'final'
): CorrelationData {
  // Extract points
  const points: CorrelationDataPoint[] = runs
    .map(run => {
      const x = getNestedValue(run.hyperparameters, hyperparamKey);

      // Get metric value based on display mode
      let y: number | undefined;
      switch (metricDisplayMode) {
        case 'max':
          y = run.metrics?.max?.[metricKey];
          break;
        case 'min':
          y = run.metrics?.min?.[metricKey];
          break;
        case 'final':
        default:
          y = run.metrics?.final?.[metricKey];
          break;
      }

      if (x === undefined || y === undefined || typeof x !== 'number' || typeof y !== 'number') {
        return null;
      }

      return {
        x,
        y,
        runName: run.run_name,
      };
    })
    .filter((p): p is CorrelationDataPoint => p !== null);

  if (points.length === 0) {
    return {
      points: [],
      regression: { slope: 0, intercept: 0, rSquared: 0 },
      pearsonR: 0,
      stats: {
        x: {
          min: 0,
          max: 0,
          mean: 0,
          median: 0,
          std: 0,
          q1: 0,
          q3: 0,
          count: 0,
        },
        y: {
          min: 0,
          max: 0,
          mean: 0,
          median: 0,
          std: 0,
          q1: 0,
          q3: 0,
          count: 0,
        },
      },
    };
  }

  // Calculate regression
  const regression = calculateLinearRegression(points);
  const pearsonR = calculatePearsonCorrelation(points);

  // Detect outliers
  const yValues = points.map(p => p.y);
  const outlierIndices = detectOutliers(yValues);
  points.forEach((p, i) => {
    p.isOutlier = outlierIndices.includes(i);
  });

  // Calculate statistics
  const xValues = points.map(p => p.x);
  const stats = {
    x: calculateStatistics(xValues),
    y: calculateStatistics(yValues),
  };

  return {
    points,
    regression,
    pearsonR,
    stats,
  };
}

/**
 * Transform runs data for distribution chart
 */
export function transformForDistribution(
  runs: Run[],
  metricKey: string,
  binCount?: number
): DistributionData {
  console.log('transformForDistribution runs', runs);
  // Extract metric values
  const values = runs
    .map(run => run.metrics?.final?.[metricKey])
    .filter((v): v is number => typeof v === 'number' && !isNaN(v));

  if (values.length === 0) {
    return {
      bins: [],
      stats: {
        min: 0,
        max: 0,
        mean: 0,
        median: 0,
        std: 0,
        q1: 0,
        q3: 0,
        count: 0,
      },
      outliers: [],
    };
  }

  // Create histogram bins
  const bins = createHistogramBins(values, binCount);

  // Calculate statistics
  const stats = calculateStatistics(values);

  // Detect outliers
  const outliers = detectOutliers(values);

  return {
    bins,
    stats,
    outliers,
  };
}

/**
 * Transform runs data for parallel coordinates chart
 */
export function transformForParallel(
  runs: Run[],
  hyperparamKeys: string[],
  metricKeys: string[]
): ParallelData {
  // Validate input
  if (!runs || runs.length === 0) {
    return { runs: [], dimensions: [] };
  }

  if (!hyperparamKeys || !metricKeys || (hyperparamKeys.length === 0 && metricKeys.length === 0)) {
    return { runs: [], dimensions: [] };
  }

  const allKeys = [...hyperparamKeys, ...metricKeys];

  // Extract all dimension values
  const runData = runs.map(run => {
    const values: Record<string, number> = {};

    hyperparamKeys.forEach(key => {
      const value = getNestedValue(run.hyperparameters, key);
      if (typeof value === 'number' && !isNaN(value)) {
        values[key] = value;
      }
    });

    metricKeys.forEach(key => {
      const value = run.metrics?.final?.[key];
      if (typeof value === 'number' && !isNaN(value)) {
        values[key] = value;
      }
    });

    return {
      runName: run.run_name,
      values,
    };
  });

  // Filter out runs with no valid values
  const validRunData = runData.filter(run => Object.keys(run.values).length > 0);

  if (validRunData.length === 0) {
    return { runs: [], dimensions: [] };
  }

  // Calculate dimensions
  const dimensions: ParallelDimension[] = allKeys.map(key => {
    const values = validRunData
      .map(r => r.values[key])
      .filter((v): v is number => typeof v === 'number' && !isNaN(v));

    const min = values.length > 0 ? Math.min(...values) : 0;
    const max = values.length > 0 ? Math.max(...values) : 1;
    const normalized = normalize(values);

    return {
      name: key,
      values,
      normalized,
      min,
      max,
    };
  }).filter(dim => dim.values.length > 0); // Only include dimensions with valid data

  return {
    runs: validRunData,
    dimensions,
  };
}

/**
 * Transform runs data for scatter matrix chart
 */
export function transformForScatterMatrix(
  runs: Run[],
  metricKeys: string[]
): ScatterMatrixData {
  const n = metricKeys.length;
  const cells: ScatterMatrixCell[][] = [];

  // Create n×n grid
  for (let i = 0; i < n; i++) {
    const row: ScatterMatrixCell[] = [];

    for (let j = 0; j < n; j++) {
      const metricX = metricKeys[j];
      const metricY = metricKeys[i];

      // Extract points
      const points: CorrelationDataPoint[] = runs
        .map(run => {
          const x = run.metrics?.final?.[metricX];
          const y = run.metrics?.final?.[metricY];

          if (typeof x !== 'number' || typeof y !== 'number') {
            return null;
          }

          return {
            x,
            y,
            runName: run.run_name,
          };
        })
        .filter((p): p is CorrelationDataPoint => p !== null);

      // Calculate correlation for off-diagonal cells
      const pearsonR = i !== j ? calculatePearsonCorrelation(points) : undefined;

      row.push({
        metricX,
        metricY,
        points,
        pearsonR,
      });
    }

    cells.push(row);
  }

  return {
    metrics: metricKeys,
    cells,
  };
}

/**
 * Downsample timeseries data using LTTB (Largest-Triangle-Three-Buckets) algorithm
 * Preserves visual characteristics while reducing data points
 */
export function downsampleTimeseries(
  data: { step: number; value: number }[],
  targetPoints: number
): { step: number; value: number }[] {
  if (data.length <= targetPoints) {
    return data;
  }

  const sampled: { step: number; value: number }[] = [];

  // Always include first point
  sampled.push(data[0]);

  const bucketSize = (data.length - 2) / (targetPoints - 2);

  let a = 0; // Previously selected point

  for (let i = 0; i < targetPoints - 2; i++) {
    // Calculate point average for next bucket (used in the next iteration)
    const avgRangeStart = Math.floor((i + 1) * bucketSize) + 1;
    const avgRangeEnd = Math.floor((i + 2) * bucketSize) + 1;
    const avgRangeLength = avgRangeEnd - avgRangeStart;

    let avgX = 0;
    let avgY = 0;

    for (let j = avgRangeStart; j < avgRangeEnd && j < data.length; j++) {
      avgX += data[j].step;
      avgY += data[j].value;
    }
    avgX /= avgRangeLength;
    avgY /= avgRangeLength;

    // Get the range for this bucket
    const rangeStart = Math.floor(i * bucketSize) + 1;
    const rangeEnd = Math.floor((i + 1) * bucketSize) + 1;

    // Point a (previously selected)
    const pointAX = data[a].step;
    const pointAY = data[a].value;

    let maxArea = -1;
    let maxAreaPoint = rangeStart;

    for (let j = rangeStart; j < rangeEnd && j < data.length; j++) {
      // Calculate triangle area
      const area = Math.abs(
        (pointAX - avgX) * (data[j].value - pointAY) -
        (pointAX - data[j].step) * (avgY - pointAY)
      ) * 0.5;

      if (area > maxArea) {
        maxArea = area;
        maxAreaPoint = j;
      }
    }

    sampled.push(data[maxAreaPoint]);
    a = maxAreaPoint;
  }

  // Always include last point
  sampled.push(data[data.length - 1]);

  return sampled;
}
