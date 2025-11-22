export interface Point {
  x: number;
  y: number;
}

export interface RegressionResult {
  slope: number;
  intercept: number;
  rSquared: number;
}

export interface Statistics {
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
  q1: number;
  q3: number;
  count: number;
}

export interface HistogramBin {
  x0: number;
  x1: number;
  length: number;
}

function rank(data: number[]): number[] {
  const sorted = [...data].map((d, i) => [d, i]).sort((a, b) => a[0] - b[0]);
  const ranks = new Array(data.length);
  let i = 0;
  while (i < sorted.length) {
    let j = i;
    while (j < sorted.length - 1 && sorted[j][0] === sorted[j + 1][0]) {
      j++;
    }
    const avgRank = (i + j + 2) / 2;
    for (let k = i; k <= j; k++) {
      ranks[sorted[k][1]] = avgRank;
    }
    i = j + 1;
  }
  return ranks;
}

function pearson(x: number[], y: number[]): number {
  const n = x.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;
  let sumY2 = 0;

  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
    sumY2 += y[i] * y[i];
  }

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  if (denominator === 0) {
    return 0;
  }

  return numerator / denominator;
}

export function spearmanr(x: number[], y: number[]): { correlation: number } {
  const rankedX = rank(x.map(v => (isNaN(v) ? Infinity : v)));
  const rankedY = rank(y.map(v => (isNaN(v) ? Infinity : v)));

  const validX = rankedX.filter(v => v !== undefined);
  const validY = rankedY.filter(v => v !== undefined);

  if (validX.length !== validY.length) {
    return { correlation: NaN };
  }

  return { correlation: pearson(validX, validY) };
}

export function calculateLinearRegression(data: Point[]): RegressionResult {
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;
  let sumYY = 0;
  let count = 0;

  for (let i = 0; i < data.length; i++) {
    const { x, y } = data[i];
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
    sumYY += y * y;
    count++;
  }

  const slope = (count * sumXY - sumX * sumY) / (count * sumXX - sumX * sumX);
  const intercept = (sumY / count) - (slope * sumX) / count;

  const rSquared = Math.pow((count * sumXY - sumX * sumY) / Math.sqrt((count * sumXX - sumX * sumX) * (count * sumYY - sumY * sumY)), 2);

  return { slope, intercept, rSquared };
}

export function calculatePearsonCorrelation(data: Point[]): number {
  const x = data.map(p => p.x);
  const y = data.map(p => p.y);
  return pearson(x, y);
}

export function createHistogramBins(data: number[], binCount = 10): HistogramBin[] {
  if (!data || data.length === 0) return [];

  const min = Math.min(...data);
  const max = Math.max(...data);

  // Handle case where all values are the same
  if (min === max) {
    return [{
      x0: min - 0.5,
      x1: max + 0.5,
      length: data.length
    }];
  }

  const binSize = (max - min) / binCount;
  const bins: HistogramBin[] = [];

  for (let i = 0; i < binCount; i++) {
    bins.push({ x0: min + i * binSize, x1: min + (i + 1) * binSize, length: 0 });
  }

  for (const d of data) {
    // Safe division with NaN check
    const rawIndex = (d - min) / binSize;
    if (!isNaN(rawIndex) && isFinite(rawIndex)) {
      const binIndex = Math.min(Math.floor(rawIndex), binCount - 1);
      if (binIndex >= 0 && binIndex < bins.length) {
        bins[binIndex].length++;
      }
    }
  }

  return bins;
}

export function calculateStatistics(data: number[]): Statistics {
  if (!data || data.length === 0) return { min: 0, max: 0, mean: 0, median: 0, std: 0, q1: 0, q3: 0, count: 0 };
  const n = data.length;
  const sorted = [...data].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[n - 1];
  const sum = data.reduce((a, b) => a + b, 0);
  const mean = sum / n;
  const mid = Math.floor(n / 2);
  const median = n % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  const q1 = sorted[Math.floor(n / 4)];
  const q3 = sorted[Math.floor((n * 3) / 4)];
  const std = Math.sqrt(data.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / n);
  return { min, max, mean, median, std, q1, q3, count: n };
}

export function normalize(data: number[]): number[] {
  const min = Math.min(...data);
  const max = Math.max(...data);
  return data.map(d => (d - min) / (max - min));
}

export function detectOutliers(data: number[]): number[] {
  if (!data || data.length === 0) return [];
  const stats = calculateStatistics(data);
  const iqr = stats.q3 - stats.q1;
  const lowerFence = stats.q1 - 1.5 * iqr;
  const upperFence = stats.q3 + 1.5 * iqr;
  return data.reduce((outliers, d, i) => {
    if (d < lowerFence || d > upperFence) {
      outliers.push(i);
    }
    return outliers;
  }, [] as number[]);
}

export function getNestedValue(obj: any, key: string): any {
  return key.split('.').reduce((o, i) => (o ? o[i] : undefined), obj);
}
