/**
 * Backend API response types
 * 
 * These types extend the core domain types with backend-specific fields.
 */

import type { Run, Project, ComponentInfo, RunStatus } from '../model/types';

/** Generic API response wrapper */
export interface ApiResponse<T = unknown> {
  success?: boolean;
  message?: string;
  data?: T;
  error?: string | null;
}

/** Experiment (Run) response from backend */
export interface ExperimentResponse extends Partial<Run> { [key: string]: any;
  experiment_uid: string;
  status: RunStatus;
  created_at: string;
}

/** Project response from API */
export interface ApiProject extends Partial<Project> {
  project_id: string;
  name: string;
}

/** Component registry response */
export interface ApiComponentInfo extends ComponentInfo {
  implementation?: string; hash?: string;
}

export interface ComponentRegistryResponse {
  components: ApiComponentInfo[];
  categories?: string[];
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page?: number;
  per_page?: number;
  has_more?: boolean;
}
