/**
 * Core Domain Types (Single Source of Truth)
 * 
 * This file defines the fundamental entities used across the application.
 * All feature-specific types should extend or compose these base types.
 */

// =============================================================================
// 1. Projects & Runs (The core tracking units)
// =============================================================================

export type RunStatus = 'completed' | 'running' | 'failed' | 'pending' | 'queued' | 'paused' | 'cancelled' | 'orphaned' | 'archived';

/** Single execution unit (previously referred to as Experiment in some contexts) */
export interface Run {
  run_name: string;
  project: string;
  status: RunStatus;
  started_at: string | null;
  finished_at: string | null;
  created_at?: string;
  experiment_uid?: string; // Backend compatibility
  config?: Record<string, any>;
  metrics?: RunMetrics;
  final_metrics?: Record<string, number> | null;
  hyperparameters?: Record<string, any>;
  metadata?: Record<string, any>;
  notes?: string;
  tags?: string[];
  assigned_device?: string | null;
  
  // Added reproducibility field
  reproducibility?: {
    git?: {
      branch?: string;
      commit_hash?: string;
      is_dirty?: boolean;
      diff_patch?: string;
    };
    environment?: Record<string, unknown>;
    packages?: Record<string, unknown>;
    hardware?: {
      gpu_count?: number;
      gpu_name?: string;
    };
    seed?: {
      global_seed?: number | null;
    };
  };

  // Index signature for flexible backend data
  [key: string]: any;
}

/** Collection of Runs */
export interface Project {
  name: string;
  description?: string;
  run_count?: number;
  created_at?: string;
  runs: Run[];
}

// =============================================================================
// 2. Metrics & Plotting
// =============================================================================

export interface TimeseriesMetric {
  step: number;
  epoch: number;
  values: Record<string, number>;
  timestamp?: string;
}

export interface RunMetrics {
  final: Record<string, number>;
  max: Record<string, number>;
  min: Record<string, number>;
  mean: Record<string, number>;
  timeseries?: TimeseriesMetric[];
}

// =============================================================================
// 3. Components & Agents (The architecture units)
// =============================================================================

export type ComponentCategory = 'model' | 'optimizer' | 'loss' | 'dataset' | 'dataloader' | 'transform' | 'scheduler' | 'metric' | 'unknown';

export interface ComponentInfo {
  name: string;
  category: ComponentCategory;
  type?: string;
  path: string;
  description?: string;
  parameters: Record<string, any>;
  examples?: Array<Record<string, any>>;
  methods?: ComponentMethod[];
}

export interface ComponentMethod {
  name: string;
  signature?: string;
  docstring?: string;
  params?: any;
}

// =============================================================================
// 4. Hardware & Infrastructure
// =============================================================================

export interface GPU {
  id: number;
  name: string;
  util: number;
  vram_used: number;
  vram_total: number;
  temperature?: number | null;
  power_usage?: number | null;
}

export interface Device {
  host_id: string;
  name?: string;
  status: 'healthy' | 'unhealthy' | 'offline';
  cpu_util: number | null;
  memory_used: number | null;
  memory_total: number | null;
  gpus?: GPU[];
  last_heartbeat: string | null;
  
  // Extended fields for specific views
  vram_used?: number | null;
  vram_total?: number | null;
}

// =============================================================================
// 5. Analysis & Comparison
// =============================================================================

export interface Comparison {
  id: string;
  name: string;
  description?: string;
  project: string;
  runs: string[]; // run_names
  created_at: string;
  updated_at: string;
  settings: any;
}

export interface ComparisonSettings {
  viewType: string;
  flattenParams: boolean;
  diffMode: boolean;
  selectedMetrics: string[];
  selectedParams: string[];
}

// =============================================================================
// Re-exports for Backward Compatibility
// =============================================================================

export * from './config-types';
export * from './charts/data-types';
export * from '../lib/charts/types';
