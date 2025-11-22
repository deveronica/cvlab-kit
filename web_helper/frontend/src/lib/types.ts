export interface Project {
  name: string;
  run_count?: number;
  runs: Array<{
    run_name: string;
    status: string;
    started_at: string | null;
    finished_at: string | null;
  }>;
}

export interface GPU {
  id: number;
  name: string;
  util: number;
  vram_used: number;
  vram_total: number;
  temperature: number | null;
  power_usage: number | null;
}

export interface Device {
  id?: string;
  name?: string;
  host_id: string;
  gpu_util?: number | null;
  gpu_utilization?: number | null;
  vram_used: number | null;
  vram_total: number | null;
  gpu_count?: number;
  gpus?: GPU[];
  cpu_util: number | null;
  memory_used: number | null;
  memory_usage?: number | null;
  memory_total: number | null;
  status: string;
  last_heartbeat: string | null;
}

export interface QueueJob {
  id?: string;
  config_path?: string;
  device_id?: string;
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'paused';
  started_at?: string | null;
  completed_at?: string | null;
  queued_at?: string | null;
}

export interface QueueResponse {
  jobs: QueueJob[];
  total: number;
}

export interface ComponentInfo {
  name: string;
  type: string;
  path: string;
  description?: string;
  parameters: Record<string, any>;
  examples: Array<Record<string, any>>;
}

export interface ComponentCategory {
  category: string;
  count: number;
  components: ComponentInfo[];
}

// === Comparison System Types ===

export interface Run {
  run_name: string;
  project: string;
  status: 'completed' | 'running' | 'failed' | 'pending';
  started_at: string | null;
  finished_at: string | null;
  config?: Record<string, any>;  // YAML configuration
  metrics?: RunMetrics;          // Final metrics
  hyperparameters?: Record<string, any>;  // Extracted hyperparameters
  notes?: string;                // User notes
  tags?: string[];               // User tags
}

export interface RunMetrics {
  final: Record<string, number>;    // Final values (val_acc_final, loss_final)
  max: Record<string, number>;      // Maximum values (val_acc_max)
  min: Record<string, number>;      // Minimum values (loss_min)
  mean: Record<string, number>;     // Average values (val_acc_mean)
  timeseries?: TimeseriesMetric[];  // Training curves
}

export interface TimeseriesMetric {
  step: number;
  epoch: number;
  values: Record<string, number>;   // loss, accuracy, lr, etc.
  timestamp?: string;
}

export interface Comparison {
  id: string;
  name: string;
  description?: string;
  project: string;
  runs: string[];               // Array of run_names
  created_at: string;
  updated_at: string;
  settings: ComparisonSettings;
  analysis?: ComparisonAnalysis;
}

export interface ComparisonSettings {
  viewType: 'training_curves' | 'hyperparams' | 'performance' | 'convergence' | 'summary';
  flattenParams: boolean;
  diffMode: boolean;
  selectedMetrics: string[];    // Which metrics to compare
  selectedParams: string[];     // Which hyperparameters to compare
  chartConfig?: ChartConfig;
}

export interface ChartConfig {
  type: 'line' | 'scatter' | 'bar' | 'heatmap' | 'parallel_coordinates';
  xAxis: string;
  yAxis: string | string[];
  groupBy?: string;
  colorBy?: string;
}

export interface ComparisonAnalysis {
  bestPerformer: {
    run_name: string;
    metric: string;
    value: number;
  };
  worstPerformer: {
    run_name: string;
    metric: string;
    value: number;
  };
  parameterImpact: ParameterImpact[];
  convergenceAnalysis: ConvergenceInfo[];
  recommendations?: string[];
}

export interface ParameterImpact {
  parameter: string;
  impact: 'high' | 'medium' | 'low';
  correlation: number;        // Correlation with main metric
  variance: number;          // How much this parameter varies
}

export interface ConvergenceInfo {
  run_name: string;
  converged: boolean;
  convergence_epoch?: number;
  final_value: number;
  convergence_speed: 'fast' | 'medium' | 'slow';
}
