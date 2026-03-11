/**
 * API client for trend analysis endpoints.
 */

const API_BASE = '/api/trends';

export type TrendDirection = 'improving' | 'degrading' | 'stable';
export type TrendStrength = 'strong' | 'moderate' | 'weak' | 'none';

export interface TrendAnalysisRequest {
  project: string;
  metrics?: string[];
  minimize_metrics?: string[];
  significance_level?: number;
}

export interface MetricTrend {
  metric: string;
  direction: TrendDirection;
  strength: TrendStrength;
  slope: number;
  r_squared: number;
  p_value: number;
  is_significant: boolean;
  prediction: {
    value: number;
    confidence_interval: [number, number];
    confidence_level: number;
  };
  improvement_rate: number;
  data_points: number;
  minimize: boolean;
  message?: string;
}

export interface TrendSummary {
  improving: string[];
  degrading: string[];
  stable: string[];
  significant_trends: Array<{
    metric: string;
    direction: TrendDirection;
    strength: TrendStrength;
    improvement_rate: number;
  }>;
}

export interface TrendAnalysisResponse {
  project: string;
  total_runs: number;
  trends: Record<string, MetricTrend>;
  summary: TrendSummary;
  total_metrics: number;
  analyzed_metrics: number;
}

export interface TrendSummaryResponse {
  project: string;
  total_runs: number;
  analyzed_metrics: number;
  summary: TrendSummary;
  health_score: number;
  significance_level: number;
  message?: string;
}

export interface MetricTrendDetailsResponse {
  project: string;
  metric: string;
  trend: MetricTrend;
  runs_analyzed: number;
}

/**
 * Analyze trends for experiment metrics.
 */
export async function analyzeTrends(
  request: TrendAnalysisRequest,
): Promise<TrendAnalysisResponse> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to analyze trends');
  }

  return response.json();
}

/**
 * Get trend summary for a project.
 */
export async function getTrendSummary(
  project: string,
  significanceLevel: number = 0.05,
): Promise<TrendSummaryResponse> {
  const params = new URLSearchParams({
    significance_level: significanceLevel.toString(),
  });

  const response = await fetch(`${API_BASE}/${project}/summary?${params}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get trend summary');
  }

  return response.json();
}

/**
 * Get detailed trend analysis for a specific metric.
 */
export async function getMetricTrendDetails(
  project: string,
  metric: string,
  significanceLevel: number = 0.05,
): Promise<MetricTrendDetailsResponse> {
  const params = new URLSearchParams({
    significance_level: significanceLevel.toString(),
  });

  const response = await fetch(
    `${API_BASE}/${project}/${metric}/details?${params}`,
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get metric trend details');
  }

  return response.json();
}
