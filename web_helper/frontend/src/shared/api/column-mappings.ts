/**
 * Column Mapping API Client
 *
 * Provides functions for interacting with column mapping endpoints.
 */

const API_BASE = '/api/projects';

export interface ColumnMapping {
  id: number;
  project: string;
  source_column: string;
  target_column: string;
  column_type: 'hyperparam' | 'metric';
  mapping_method: 'auto' | 'manual' | 'verified';
  confidence_score?: number;
  algorithm?: string;
  metadata?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  created_by?: string;
}

export interface ColumnSuggestion {
  source_column: string;
  target_column: string;
  confidence_score: number;
  algorithm: string;
  reason: string;
  column_type: 'hyperparam' | 'metric';
  similarity_details?: Record<string, any>;
}

export interface ColumnMappingSuggestionResponse {
  project: string;
  suggestions: ColumnSuggestion[];
  total_suggestions: number;
  high_confidence_count: number;
  medium_confidence_count: number;
  low_confidence_count: number;
}

export interface ColumnMappingCreate {
  source_column: string;
  target_column: string;
  column_type: 'hyperparam' | 'metric';
  mapping_method?: 'auto' | 'manual' | 'verified';
  confidence_score?: number;
  algorithm?: string;
  metadata?: string;
  is_active?: boolean;
}

export interface ColumnMappingUpdate {
  target_column?: string;
  mapping_method?: 'auto' | 'manual' | 'verified';
  is_active?: boolean;
  metadata?: string;
}

/**
 * List all column mappings for a project
 */
export async function listColumnMappings(
  project: string,
  columnType?: 'hyperparam' | 'metric',
  activeOnly: boolean = true
): Promise<ColumnMapping[]> {
  const params = new URLSearchParams();
  if (columnType) params.set('column_type', columnType);
  params.set('active_only', String(activeOnly));

  const response = await fetch(
    `${API_BASE}/${encodeURIComponent(project)}/column-mappings?${params.toString()}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch column mappings: ${response.statusText}`);
  }

  const result = await response.json();
  return result.data;
}

/**
 * Create a new column mapping
 */
export async function createColumnMapping(
  project: string,
  mapping: ColumnMappingCreate
): Promise<ColumnMapping> {
  const response = await fetch(
    `${API_BASE}/${encodeURIComponent(project)}/column-mappings`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(mapping),
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to create column mapping: ${response.statusText}`);
  }

  const result = await response.json();
  return result.data;
}

/**
 * Update an existing column mapping
 */
export async function updateColumnMapping(
  project: string,
  mappingId: number,
  update: ColumnMappingUpdate
): Promise<ColumnMapping> {
  const response = await fetch(
    `${API_BASE}/${encodeURIComponent(project)}/column-mappings/${mappingId}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(update),
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to update column mapping: ${response.statusText}`);
  }

  const result = await response.json();
  return result.data;
}

/**
 * Delete a column mapping
 */
export async function deleteColumnMapping(
  project: string,
  mappingId: number
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/${encodeURIComponent(project)}/column-mappings/${mappingId}`,
    {
      method: 'DELETE',
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to delete column mapping: ${response.statusText}`);
  }
}

/**
 * Generate automatic column mapping suggestions
 */
export async function generateColumnSuggestions(
  project: string,
  columnType: 'hyperparam' | 'metric' = 'hyperparam',
  minConfidence: number = 0.5
): Promise<ColumnMappingSuggestionResponse> {
  const params = new URLSearchParams({
    column_type: columnType,
    min_confidence: String(minConfidence),
  });

  const response = await fetch(
    `${API_BASE}/${encodeURIComponent(project)}/column-suggestions?${params.toString()}`,
    {
      method: 'POST',
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Failed to generate suggestions: ${response.statusText}`);
  }

  const result = await response.json();
  return result.data;
}
