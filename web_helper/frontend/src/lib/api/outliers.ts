/**
 * API client for outlier detection endpoints.
 */

const API_BASE = '/api/outliers';

export type OutlierMethod = 'iqr' | 'zscore' | 'modified_zscore';

export interface OutlierDetectionRequest {
  project: string;
  run_names?: string[];
  hyperparam_columns?: string[];
  metric_columns?: string[];
  method?: OutlierMethod;
  threshold?: number;
}

export interface OutlierColumnResult {
  outlier_indices: number[];
  outlier_runs: string[];
  scores: number[];
  statistics: {
    q1?: number;
    q3?: number;
    iqr?: number;
    lower_bound?: number;
    upper_bound?: number;
    mean?: number;
    median?: number;
    std?: number;
    mad?: number;
    threshold?: number;
  };
  method: string;
  threshold?: number;
  message?: string;
}

export interface OutlierAnalysisResult {
  column_results: Record<string, OutlierColumnResult>;
  summary: {
    total_outlier_runs: number;
    outlier_runs: string[];
    outlier_counts: Record<string, number>;
    method: string;
    threshold: number;
  };
}

export interface OutlierDetectionResponse {
  project: string;
  total_runs: number;
  hyperparameters: OutlierAnalysisResult;
  metrics: OutlierAnalysisResult;
  method: string;
  threshold: number;
}

export interface OutlierSummaryResponse {
  project: string;
  total_runs: number;
  total_outlier_runs: number;
  outlier_percentage: number;
  top_outlier_runs: Array<{
    run_name: string;
    outlier_count: number;
  }>;
  columns_with_outliers: Record<string, number>;
  cell_outliers: Record<string, boolean>; // NEW: {run_name|column: true}
  method: string;
  threshold: number;
}

/**
 * Detect outliers in experiment runs.
 */
export async function detectOutliers(
  request: OutlierDetectionRequest,
): Promise<OutlierDetectionResponse> {
  const response = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to detect outliers');
  }

  return response.json();
}

/**
 * Get outlier summary for a project.
 */
export async function getOutlierSummary(
  project: string,
  method: OutlierMethod = 'iqr',
  threshold: number = 1.5,
): Promise<OutlierSummaryResponse> {
  const params = new URLSearchParams({
    method,
    threshold: threshold.toString(),
  });

  const response = await fetch(`${API_BASE}/${project}/summary?${params}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get outlier summary');
  }

  return response.json();
}
