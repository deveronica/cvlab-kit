/**
 * Node Types
 *
 * Unified node data structure for both Execute Tab and Builder Tab.
 * This is the SINGLE source of truth - prevents version mismatch bugs.
 */

import type { NodeMode } from './tab';
import type { UnifiedPort } from './port';
import type { SourceLocation } from './edge';

/**
 * Component category for theming and port defaults
 */
export const ComponentCategory = {
  MODEL: 'model',
  OPTIMIZER: 'optimizer',
  LOSS: 'loss',
  DATASET: 'dataset',
  DATALOADER: 'dataloader',
  TRANSFORM: 'transform',
  METRIC: 'metric',
  SCHEDULER: 'scheduler',
  SAMPLER: 'sampler',
  CHECKPOINT: 'checkpoint',
  CALLBACK: 'callback',
  LOGGER: 'logger',
  AGENT: 'agent',
  GLOBAL: 'global',
  UNKNOWN: 'unknown',
} as const;

export type ComponentCategory = (typeof ComponentCategory)[keyof typeof ComponentCategory];

/**
 * Parameter type for editable fields
 */
export type ParamType = 'string' | 'number' | 'boolean' | 'select';

/**
 * Parameter definition
 */
export interface UnifiedParam {
  name: string;
  value: string | number | boolean;
  type: ParamType;
  options?: string[];              // For 'select' type
  defaultValue?: string | number | boolean;
  description?: string;
  min?: number;                    // For 'number' type
  max?: number;                    // For 'number' type
  step?: number;                   // For 'number' type
}

/**
 * Implementation option for dropdown selection
 */
export interface ImplementationOption {
  name: string;
  description?: string;
  defaultParams?: UnifiedParam[];
}

/**
 * Special node types (ADR-008 Hybrid Approach)
 */
export const SpecialNodeType = {
  /** Code block node - Hybrid Approach for complex logic */
  CODE_BLOCK: 'code_block',
  /** Input node for Hierarchical Navigation */
  INPUT: 'input',
  /** Output node for Hierarchical Navigation */
  OUTPUT: 'output',
  /** Subsystem node for drill-down */
  SUBSYSTEM: 'subsystem',
  /** Standard component node */
  COMPONENT: 'component',
  /** Control flow: if/else */
  CONTROL_IF: 'control_if',
  /** Control flow: loop */
  CONTROL_LOOP: 'control_loop',
  /** Control flow: loop exit */
  CONTROL_LOOP_EXIT: 'control_loop_exit',
  /** Control flow: with/no_grad */
  CONTROL_WITH: 'control_with',
  /** Control flow: with exit */
  CONTROL_WITH_EXIT: 'control_with_exit',
  /** Control flow: merge */
  CONTROL_MERGE: 'control_merge',
  /** Control flow: generic step */
  CONTROL_STEP: 'control_step',
} as const;

export type SpecialNodeType = typeof SpecialNodeType[keyof typeof SpecialNodeType];

/**
 * UnifiedNodeData - Single source of truth for all node types
 *
 * This consolidates all previous node data interfaces into one.
 * The `mode` property determines how the node is rendered.
 *
 * @example
 * // Execute Tab: shows implementation selector + params
 * { mode: 'execute', params: [...], availableImplementations: [...] }
 *
 * // Builder Tab: compact with implementation badge
 * { mode: 'builder', implementation: 'resnet18' }
 *
 * // Flow visualization: ports only
 * { mode: 'flow' }
 */
export interface UnifiedNodeData {
  // ==========================================================================
  // Identity
  // ==========================================================================
  /** Unique node ID */
  id: string;
  /** What this node does (e.g., "model", "train_dataset", "loss") */
  role: string;
  /** Category for theming and default ports */
  category: ComponentCategory;
  /** Special node type (component, code_block, subsystem, etc.) */
  nodeType?: SpecialNodeType;

  // ==========================================================================
  // Implementation
  // ==========================================================================
  /** Current implementation name (e.g., "resnet18", "adam") */
  implementation?: string;
  /** Property summary from backend */
  propertySummary?: {
    config_count: number;
    hardcode_count: number;
    required_count: number;
    default_count: number;
  };
  /** Code snippet for tooltips */
  codeSnippet?: string;
  /** Available implementations for dropdown */
  availableImplementations?: ImplementationOption[];

  // ==========================================================================
  // Ports (4-Zone Layout)
  // ==========================================================================
  /** Input ports */
  inputs: UnifiedPort[];
  /** Output ports */
  outputs: UnifiedPort[];
  /** Init input ports (for Execute Tab header) */
  initInputs?: UnifiedPort[];
  /** Self output port (for Execute Tab header) */
  selfOutput?: UnifiedPort;
  /** Execution pins (for flow control) */
  executionPins?: UnifiedPort[];

  /** Additional metadata (row_index, etc.) */
  metadata?: Record<string, any>;

  // ==========================================================================
  // Parameters (Execute mode)
  // ==========================================================================
  params?: UnifiedParam[];

  // ==========================================================================
  // Display State
  // ==========================================================================
  /** Rendering mode */
  mode: NodeMode;
  /** Currently selected */
  selected?: boolean;
  /** Config differs from defaults */
  isModified?: boolean;
  /** Collapsed state (for complex nodes) */
  collapsed?: boolean;

  // ==========================================================================
  // Source Code Mapping (1:1)
  // ==========================================================================
  /** Source code location */
  source?: SourceLocation;
  /** Code path not executed/analyzed */
  isUncovered?: boolean;

  // ==========================================================================
  // Hierarchical Navigation
  // ==========================================================================
  /** Can drill into this node (model internals, etc.) */
  canDrill?: boolean;
  /** Parent node ID if this is inside a subsystem */
  parentId?: string;
  /** Child node IDs if this is a subsystem */
  childIds?: string[];

  // ==========================================================================
  // Callbacks
  // ==========================================================================
  onImplementationChange?: (nodeId: string, impl: string) => void;
  onParamChange?: (nodeId: string, param: string, value: string | number | boolean) => void;
  onCodeClick?: (source: SourceLocation) => void;
  onDrill?: (nodeId: string) => void;
}

/**
 * Full Graph structure from API
 */
export interface NodeGraph {
  id: string;
  name: string;
  nodes: any[]; // Use any[] for compatibility with BackendNode during transform
  edges: any[];
}

/**
 * Node dimensions for layout (dagre)
 * Standardized to 180px width (9 grid dots)
 */
export const NODE_DIMENSIONS = {
  execute: { width: 180, height: 160 },
  builder: { width: 180, height: 120 },
  flow: { width: 180, height: 100 },
  codeBlock: { width: 180, height: 120 },
  subsystem: { width: 180, height: 140 },
  control: { width: 180, height: 80 },
} as const;

/**
 * Snap grid size for node placement (Item 1)
 */
export const SNAP_GRID: [number, number] = [20, 20];

/**
 * Quantize a value to the nearest grid step
 */
export const quantize = (val: number) => Math.round(val / SNAP_GRID[0]) * SNAP_GRID[0];

/**
 * Get node dimensions based on mode and type
 */
export function getNodeDimensions(
  mode: NodeMode,
  nodeType?: SpecialNodeType
): { width: number; height: number } {
  if (nodeType === SpecialNodeType.CODE_BLOCK) {
    return NODE_DIMENSIONS.codeBlock;
  }
  if (nodeType === SpecialNodeType.SUBSYSTEM) {
    return NODE_DIMENSIONS.subsystem;
  }
  if (nodeType && nodeType.startsWith('control_')) {
    return NODE_DIMENSIONS.control;
  }
  return NODE_DIMENSIONS[mode];
}
