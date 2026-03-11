/**
 * API client for best run recommendation endpoints.
 */

const API_BASE = '/api/recommendations';

export type RecommendationStrategy = 'pareto' | 'weighted' | 'rank';

export interface RecommendationRequest {
  project: string;
  objectives?: string[];
  strategy?: RecommendationStrategy;
  weights?: number[];
  minimize?: boolean[];
  top_k?: number;
}

export interface ParetoRecommendation {
  run_name: string;
  is_pareto_optimal: boolean;
  dominated_by_count: number;
  dominates_count: number;
  objective_values: Record<string, number>;
  rank: number;
}

export interface WeightedRecommendation {
  run_name: string;
  score: number;
  objective_values: Record<string, number>;
  normalized_values: Record<string, number>;
}

export interface RecommendationResponse {
  recommendations: Array<ParetoRecommendation | WeightedRecommendation>;
  strategy: string;
  total_runs: number;
  objectives: string[];
  minimize: boolean[] | null;
  pareto_optimal_count?: number;
  weights?: number[];
}

export interface RecommendationSummaryResponse {
  project: string;
  total_runs: number;
  auto_selected_objectives: string[];
  pareto_optimal_count: number;
  top_recommendations: ParetoRecommendation[];
  message?: string;
}

export interface ParetoFrontierResponse {
  project: string;
  objectives: string[];
  pareto_optimal_runs: ParetoRecommendation[];
  total_pareto_optimal: number;
  total_runs: number;
  pareto_percentage: number;
}

/**
 * Find best runs using multi-objective optimization.
 */
export async function findBestRuns(
  request: RecommendationRequest,
): Promise<RecommendationResponse> {
  const response = await fetch(`${API_BASE}/find`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to find best runs');
  }

  return response.json();
}

/**
 * Get Pareto-optimal runs for a project.
 */
export async function getParetoFrontier(
  project: string,
  objectives: string[],
): Promise<ParetoFrontierResponse> {
  const params = new URLSearchParams();
  objectives.forEach((obj) => params.append('objectives', obj));

  const response = await fetch(`${API_BASE}/${project}/pareto?${params}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get Pareto frontier');
  }

  return response.json();
}

/**
 * Get recommendation summary for a project.
 */
export async function getRecommendationSummary(
  project: string,
  topK: number = 3,
): Promise<RecommendationSummaryResponse> {
  const params = new URLSearchParams({
    top_k: topK.toString(),
  });

  const response = await fetch(`${API_BASE}/${project}/summary?${params}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get recommendation summary');
  }

  return response.json();
}
