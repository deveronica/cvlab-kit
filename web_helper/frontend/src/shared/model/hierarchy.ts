/**
 * Hierarchical Node Graph Types for Simulink-style visualization.
 *
 * Matches backend models/hierarchy.py schema.
 */

// =============================================================================
// Enums
// =============================================================================

export type HierarchyLevel = "method" | "component" | "layer" | "operation";

export type ComponentCategory =
  | "model"
  | "optimizer"
  | "scheduler"
  | "loss"
  | "dataset"
  | "dataloader"
  | "transform"
  | "metric"
  | "sampler"
  | "checkpoint"
  | "callback"
  | "logger"
  | "agent"
  | "global"
  | "unknown";

/**
 * Method categories for Agent methods.
 * Matches backend hierarchy.py MethodCategory enum.
 */
export type MethodCategory =
  | "setup"       // setup() - initialization
  | "train"       // train_step() - training logic
  | "validation"  // val_step() - validation logic
  | "lifecycle"   // on_train_start, on_train_end, etc.
  | "utility";    // Other helper methods

export type FlowType =
  | "tensor"
  | "parameters"
  | "gradient"
  | "control"
  | "config"
  | "reference";

export type OriginType =
  | "create_call"
  | "assignment"
  | "method_call"
  | "unpack"
  | "input"
  | "output";

export type ImplSource = "yaml" | "default" | "hardcode";

/**
 * Value source for properties (5-state system).
 * Matches backend hierarchy.py ValueSource enum.
 */
export const ValueSource = {
  REQUIRED: "required",   // No default value, must be provided
  CONFIG: "config",       // Value from YAML config file
  DEFAULT: "default",     // Has default value in code
  HARDCODE: "hardcode",   // Fixed in code, not configurable
  CONNECTED: "connected", // Value from connected node edge
} as const;

export type ValueSource = typeof ValueSource[keyof typeof ValueSource];

/**
 * Property information with value source for UI display.
 * Used in SmartNodeView to show properties with clear source indicators.
 */
export interface PropertyInfo {
  name: string;
  value: unknown;
  source: ValueSource;
  is_grid_search?: boolean;
  grid_count?: number;
  default_value?: unknown;
  description?: string;
}

/**
 * Summary counts of properties by source for badge display.
 * Shows "⚠️ 2 required, 3 config" on node surface.
 */
export interface PropertySummary {
  required_count: number;
  config_count: number;
  default_count: number;
  hardcode_count: number;
  connected_count: number;
}

// =============================================================================
// Types
// =============================================================================

export interface SourceLocation {
  file: string;
  line: number;
  column?: number;
  end_line?: number;
  end_column?: number;
}

export interface NodeOrigin {
  type: OriginType;
  create_path?: string[];
  code_snippet: string;
  source?: SourceLocation;
  impl_source?: ImplSource;
}

export interface DrillTarget {
  type: "component" | "layer";
  path: string;
}

export interface Port {
  name: string;
  type: string;
  kind?: "exec" | "data";
}

export interface HierarchyPath {
  level: HierarchyLevel;
  label: string;
  node_id: string;
  graph_id: string;
  // Optional: Category for visual styling (e.g., "model", "optimizer")
  category?: ComponentCategory;
  // Optional: Implementation name (e.g., "resnet18")
  implementation?: string;
}

export interface HierarchyNode {
  id: string;
  label: string;
  level: HierarchyLevel;

  // Code origin
  origin?: NodeOrigin;

  // Drill-down support
  can_drill: boolean;
  drill_target?: DrillTarget;

  // Visual category
  category?: ComponentCategory;

  // Ports
  inputs: Port[];
  outputs: Port[];

  // Source location
  source?: SourceLocation;

  // Properties with ValueSource (Phase 1)
  properties?: PropertyInfo[];
  property_summary?: PropertySummary;

  // Additional metadata
  metadata?: Record<string, unknown>;
}

export interface CodeFlowEdge {
  id: string;
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;

  // Flow type
  flow_type: FlowType;
  edge_type?: "execution" | "data";

  // Variable name
  variable_name?: string;

  // Extraction source
  extracted_from?: "setup" | "train_step";

  // Creation order (for setup phase: 1, 2, 3, ...)
  sequence_index?: number;

  // Source location
  source?: SourceLocation;

  // Additional metadata
  metadata?: Record<string, unknown>;
}

export interface Hierarchy {
  parent_graph_id?: string;
  parent_node_id?: string;
  depth: number;
  path: HierarchyPath[];
}

export interface HierarchicalNodeGraph {
  id: string;
  label: string;
  level: HierarchyLevel;

  // Hierarchy context
  hierarchy: Hierarchy;

  // Graph content
  nodes: HierarchyNode[];
  edges: CodeFlowEdge[];

  // Agent name
  agent_name?: string;

  // Source file
  source_file?: string;

  // Code coverage info (for 1:1 code-node mapping)
  method_range?: SourceLocation;
  covered_lines?: number[];
  uncovered_lines?: number[];

  // Additional metadata
  metadata?: Record<string, unknown>;
}

// =============================================================================
// API Response Types
// =============================================================================

export interface HierarchyResponse {
  success: boolean;
  data?: HierarchicalNodeGraph;
  error?: string;
}

// =============================================================================
// Navigation State
// =============================================================================

export interface NavigationState {
  agentName: string;
  currentPath: string;
  graph: HierarchicalNodeGraph | null;
  history: Array<{
    path: string;
    graph: HierarchicalNodeGraph;
    viewport?: { x: number; y: number; zoom: number };
  }>;
}

// =============================================================================
// Theme Types
// =============================================================================

export interface CategoryTheme {
  // Legacy aliases used across the Builder UI
  bg: string;
  badge: string;
  background: string;
  border: string;
  text: string;
  icon?: string;
}

export const CATEGORY_THEMES: Record<ComponentCategory, CategoryTheme> = {
  model: {
    bg: "bg-blue-50 dark:bg-blue-950/30",
    badge: "bg-blue-50 dark:bg-blue-950/30",
    background: "bg-blue-50 dark:bg-blue-950/30",
    border: "border-blue-200 dark:border-blue-800",
    text: "text-blue-700 dark:text-blue-300",
    icon: "cube",
  },
  optimizer: {
    bg: "bg-green-50 dark:bg-green-950/30",
    badge: "bg-green-50 dark:bg-green-950/30",
    background: "bg-green-50 dark:bg-green-950/30",
    border: "border-green-200 dark:border-green-800",
    text: "text-green-700 dark:text-green-300",
    icon: "trending-up",
  },
  loss: {
    bg: "bg-red-50 dark:bg-red-950/30",
    badge: "bg-red-50 dark:bg-red-950/30",
    background: "bg-red-50 dark:bg-red-950/30",
    border: "border-red-200 dark:border-red-800",
    text: "text-red-700 dark:text-red-300",
    icon: "target",
  },
  dataset: {
    bg: "bg-purple-50 dark:bg-purple-950/30",
    badge: "bg-purple-50 dark:bg-purple-950/30",
    background: "bg-purple-50 dark:bg-purple-950/30",
    border: "border-purple-200 dark:border-purple-800",
    text: "text-purple-700 dark:text-purple-300",
    icon: "database",
  },
  dataloader: {
    bg: "bg-indigo-50 dark:bg-indigo-950/30",
    badge: "bg-indigo-50 dark:bg-indigo-950/30",
    background: "bg-indigo-50 dark:bg-indigo-950/30",
    border: "border-indigo-200 dark:border-indigo-800",
    text: "text-indigo-700 dark:text-indigo-300",
    icon: "loader",
  },
  transform: {
    bg: "bg-amber-50 dark:bg-amber-950/30",
    badge: "bg-amber-50 dark:bg-amber-950/30",
    background: "bg-amber-50 dark:bg-amber-950/30",
    border: "border-amber-200 dark:border-amber-800",
    text: "text-amber-700 dark:text-amber-300",
    icon: "wand",
  },
  metric: {
    bg: "bg-cyan-50 dark:bg-cyan-950/30",
    badge: "bg-cyan-50 dark:bg-cyan-950/30",
    background: "bg-cyan-50 dark:bg-cyan-950/30",
    border: "border-cyan-200 dark:border-cyan-800",
    text: "text-cyan-700 dark:text-cyan-300",
    icon: "bar-chart",
  },
  scheduler: {
    bg: "bg-orange-50 dark:bg-orange-950/30",
    badge: "bg-orange-50 dark:bg-orange-950/30",
    background: "bg-orange-50 dark:bg-orange-950/30",
    border: "border-orange-200 dark:border-orange-800",
    text: "text-orange-700 dark:text-orange-300",
    icon: "clock",
  },
  sampler: {
    bg: "bg-pink-50 dark:bg-pink-950/30",
    badge: "bg-pink-50 dark:bg-pink-950/30",
    background: "bg-pink-50 dark:bg-pink-950/30",
    border: "border-pink-200 dark:border-pink-800",
    text: "text-pink-700 dark:text-pink-300",
    icon: "shuffle",
  },
  checkpoint: {
    bg: "bg-slate-50 dark:bg-slate-950/30",
    badge: "bg-slate-50 dark:bg-slate-950/30",
    background: "bg-slate-50 dark:bg-slate-950/30",
    border: "border-slate-200 dark:border-slate-800",
    text: "text-slate-700 dark:text-slate-300",
    icon: "save",
  },
  callback: {
    bg: "bg-teal-50 dark:bg-teal-950/30",
    badge: "bg-teal-50 dark:bg-teal-950/30",
    background: "bg-teal-50 dark:bg-teal-950/30",
    border: "border-teal-200 dark:border-teal-800",
    text: "text-teal-700 dark:text-teal-300",
    icon: "phone-callback",
  },
  logger: {
    bg: "bg-lime-50 dark:bg-lime-950/30",
    badge: "bg-lime-50 dark:bg-lime-950/30",
    background: "bg-lime-50 dark:bg-lime-950/30",
    border: "border-lime-200 dark:border-lime-800",
    text: "text-lime-700 dark:text-lime-300",
    icon: "file-text",
  },
  agent: {
    bg: "bg-violet-50 dark:bg-violet-950/30",
    badge: "bg-violet-50 dark:bg-violet-950/30",
    background: "bg-violet-50 dark:bg-violet-950/30",
    border: "border-violet-200 dark:border-violet-800",
    text: "text-violet-700 dark:text-violet-300",
    icon: "user",
  },
  global: {
    bg: "bg-stone-50 dark:bg-stone-950/30",
    badge: "bg-stone-50 dark:bg-stone-950/30",
    background: "bg-stone-50 dark:bg-stone-950/30",
    border: "border-stone-200 dark:border-stone-800",
    text: "text-stone-700 dark:text-stone-300",
    icon: "globe",
  },
  unknown: {
    bg: "bg-gray-50 dark:bg-gray-950/30",
    badge: "bg-gray-50 dark:bg-gray-950/30",
    background: "bg-gray-50 dark:bg-gray-950/30",
    border: "border-gray-200 dark:border-gray-800",
    text: "text-gray-700 dark:text-gray-300",
    icon: "help-circle",
  },
};

// =============================================================================
// Flow Type Themes
// =============================================================================

export interface FlowTypeTheme {
  stroke: string;
  strokeDasharray?: string;
  label: string;
}

export const FLOW_TYPE_THEMES: Record<FlowType, FlowTypeTheme> = {
  tensor: {
    stroke: "stroke-blue-500 dark:stroke-blue-400",
    label: "Tensor",
  },
  parameters: {
    stroke: "stroke-green-500 dark:stroke-green-400",
    strokeDasharray: "5,5",
    label: "Parameters",
  },
  gradient: {
    stroke: "stroke-red-500 dark:stroke-red-400",
    strokeDasharray: "3,3",
    label: "Gradient",
  },
  control: {
    stroke: "stroke-gray-500 dark:stroke-gray-400",
    strokeDasharray: "10,5",
    label: "Control",
  },
  config: {
    stroke: "stroke-purple-500 dark:stroke-purple-400",
    strokeDasharray: "1,3",
    label: "Config",
  },
  reference: {
    stroke: "stroke-gray-500 dark:stroke-gray-400",
    label: "Reference",
  },
};

// =============================================================================
// Impl Source Themes (for value origin indicator)
// =============================================================================

export interface ImplSourceTheme {
  /** Badge background color */
  background: string;
  /** Badge text color */
  text: string;
  /** Badge border */
  border: string;
  /** Display label */
  label: string;
  /** Description for tooltip */
  description: string;
  /** Whether value is editable via YAML */
  editable: boolean;
}

export const IMPL_SOURCE_THEMES: Record<ImplSource, ImplSourceTheme> = {
  yaml: {
    background: "bg-blue-100 dark:bg-blue-900/50",
    text: "text-blue-700 dark:text-blue-300",
    border: "border-blue-300 dark:border-blue-700",
    label: "YAML",
    description: "Value from YAML config (editable)",
    editable: true,
  },
  default: {
    background: "bg-amber-100 dark:bg-amber-900/50",
    text: "text-amber-700 dark:text-amber-300",
    border: "border-amber-300 dark:border-amber-700",
    label: "Default",
    description: "Code default value (add to YAML to override)",
    editable: false,
  },
  hardcode: {
    background: "bg-red-100 dark:bg-red-900/50",
    text: "text-red-700 dark:text-red-300",
    border: "border-red-300 dark:border-red-700",
    label: "Hardcode",
    description: "Hardcoded in source code (code edit required)",
    editable: false,
  },
};
