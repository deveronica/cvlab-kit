/**
 * Node Graph Type Definitions
 *
 * TypeScript interfaces mirroring backend Pydantic models for node graph visualization.
 * Used for hierarchical code representation and drill-down navigation.
 *
 * Note: PortType and ValueSource are defined in hierarchy.ts and re-exported here
 * for backward compatibility. New code should import from hierarchy.ts or node-system/model/port.ts
 */

import type { Config } from '@/shared/model/types';
import type { PortType } from '@/entities/node-system/model/port';
import type { PropertyInfo, PropertySummary, ValueSource } from './hierarchy';

export enum NodeType {
  AGENT = 'agent',
  COMPONENT = 'component',
  CONFIG = 'config',  // Config-centric node (uses ConfigNode renderer)
  MODULE = 'module',
  LAYER = 'layer',
  OPERATION = 'operation',
  FUNCTION = 'function',
  INPUT = 'input',
  OUTPUT = 'output',
  PARAMETER = 'parameter',
}

// Re-export PortType from node-system for consistency
// Authoritative source: src/features/node-system/model/port.ts
export { PortType } from '@/entities/node-system/model/port';

/**
 * Classification of input ports (3-way split for visualization)
 * - COMPONENT: Core data flow (tensors, modules) - left/right sides
 * - PROPERTY: Config parameters (lr, num_classes) - top side
 * - DEPENDENCY: Runtime injection (model.parameters()) - bottom side
 */
export enum InputKind {
  COMPONENT = 'component',
  PROPERTY = 'property',
  DEPENDENCY = 'dependency',
}

// Re-export from hierarchy for consistency
// Authoritative sources:
// - ValueSource: src/model/hierarchy.ts
// - PropertyInfo: src/model/hierarchy.ts
// - PropertySummary: src/model/hierarchy.ts
export { type PropertyInfo, type PropertySummary } from './hierarchy';
export { ValueSource } from './hierarchy';

export enum EdgeType {
  TENSOR = 'tensor',      // Tensor data flow (forward pass)
  GRADIENT = 'gradient',  // Gradient flow (backward pass)
  CONFIG = 'config',      // Configuration injection
  PARAMETERS = 'parameters', // Parameter passing (e.g., model.parameters())
  CONTROL = 'control',    // Control flow (execution order)
}

export enum ComponentCategory {
  MODEL = 'model',
  OPTIMIZER = 'optimizer',
  SCHEDULER = 'scheduler',
  LOSS = 'loss',
  DATASET = 'dataset',
  DATALOADER = 'dataloader',
  TRANSFORM = 'transform',
  METRIC = 'metric',
  SAMPLER = 'sampler',
  CHECKPOINT = 'checkpoint',
  CALLBACK = 'callback',
  LOGGER = 'logger',
  AGENT = 'agent',
  GLOBAL = 'global',
  UNKNOWN = 'unknown',
}

/**
 * Value state of a port (for property ports)
 */
export interface PortValue {
  source: ValueSource;
  actual?: unknown;
  default?: unknown;
  config_key?: string;
  is_grid_search?: boolean;
  grid_values?: unknown[];
}

export interface Port {
  name: string;
  type: PortType;

  // ComfyUI-style type compatibility
  compatible_types?: PortType[];

  // 3-way classification for visualization
  input_kind?: InputKind;

  // Value state (for property ports)
  value?: PortValue;

  // Editing mode: 'inline' for simple values, 'panel' for complex ones
  edit_mode?: 'inline' | 'panel';

  // Visual hints
  shape?: string;
  dtype?: string;
  optional?: boolean;
  default?: unknown;
}

/**
 * Ports classified by input kind (for 3-way layout)
 */
export interface ClassifiedPorts {
  component: Port[];  // Core data flow ports (left/right sides) - tensors, modules
  property: Port[];   // Config parameter ports (top side) - lr, num_classes, etc.
  dependency: Port[]; // Runtime injection ports (bottom side) - model.parameters()
}

/**
 * Hierarchy information for a node (Simulink-style drill-down)
 */
export interface NodeHierarchy {
  parent_id?: string;
  depth: number;
  path: string[];  // e.g., ['agent', 'train_step', 'model']
}

/**
 * Mapping between external and internal ports (shown when drilled into)
 */
export interface PortMapping {
  external_port: string;
  internal_node: string;
  internal_port: string;
  direction: 'input' | 'output';
}

/**
 * Config parameters with priority levels (runtime > component > global)
 * Mirrors CVLab-Kit's Creator parameter merging behavior
 */
export interface ConfigParams {
  global: Config;                  // Parameters inherited from global config
  component: Config;               // Component-specific parameters
  runtime: string[];               // Runtime injection arguments
}

export interface Edge {
  id: string;
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
  edge_type?: EdgeType;  // Type of connection (tensor, gradient, config, etc.)
  label?: string;
}

export interface SourceLocation {
  file: string;
  line_start: number;
  line_end: number;
  column_start?: number;
  column_end?: number;
}

export interface Node {
  id: string;
  label: string;
  type: NodeType;

  // Config-Centric: Role vs Implementation (What vs How)
  role?: string;              // Role/purpose in pipeline (What) - e.g., 'classifier', 'generator'
  implementation?: string;    // Implementation class (How) - e.g., 'resnet18', 'unet'
  category?: ComponentCategory; // CVLab-Kit component category

  // Config parameters (Config-Centric)
  config_params?: ConfigParams; // Parameters with priority levels
  config?: Config;              // Legacy: flat config dict

  // Ports - NEW: 3-way classified ports for visualization
  ports?: ClassifiedPorts;
  // Legacy: flat port lists (for backward compatibility)
  inputs?: Port[];
  outputs?: Port[];

  // Hierarchy - NEW: Simulink-style drill-down info
  hierarchy?: NodeHierarchy;
  // Legacy drill-down fields
  has_children?: boolean;
  children_type?: string;     // 'layer' | 'operation'

  // Port mapping - NEW: shown when drilled into a node
  port_mapping?: PortMapping[];

  // Source code mapping
  source?: SourceLocation;
  metadata?: Record<string, unknown>;
}

/**
 * Single item in logical hierarchy breadcrumb (Simulink-style)
 */
export interface BreadcrumbItem {
  type: string;  // 'agent', 'method', 'component', 'layer', 'operation'
  label: string;
  node_id: string;
  graph_id: string;
}

/**
 * Hierarchy information for a NodeGraph (Simulink-style drill-down)
 */
export interface GraphHierarchy {
  parent_graph_id?: string;
  parent_node_id?: string;
  depth: number;
  breadcrumb: BreadcrumbItem[];
}

/**
 * External port mapping when drilled into a graph
 */
export interface ExternalPorts {
  inputs: Port[];   // Parent node's input ports mapped to internal nodes
  outputs: Port[];  // Internal nodes mapped to parent node's output ports
}

export interface NodeGraph {
  // Graph identity
  id: string;
  label: string;
  type: string;

  // Graph structure
  nodes: Node[];
  edges: Edge[];

  // Hierarchy - NEW: Simulink-style drill-down context
  hierarchy?: GraphHierarchy;
  // External ports - NEW: shown when inside a drilled-into graph
  external_ports?: ExternalPorts;
  // Legacy hierarchy fields (for backward compatibility)
  parent_id?: string;
  breadcrumb?: string[];

  // Source mapping
  source?: SourceLocation;
  metadata?: Record<string, unknown>;
}

export interface AvailableMethodsResponse {
  methods: string[];
  default: string;
}

export interface NodeGraphAPIResponse {
  success: boolean;
  data: NodeGraph;
  error?: string;
}

export interface AvailableMethodsAPIResponse {
  success: boolean;
  data: AvailableMethodsResponse;
  error?: string;
}

// =============================================================================
// Node System Types - 1:1 Code-Node Mapping System
// =============================================================================

/**
 * Data types that flow through ports and edges.
 * Matches backend DataType enum.
 */
export enum DataType {
  TENSOR = 'tensor',       // torch.Tensor
  MODULE = 'module',       // nn.Module or callable
  PARAMS = 'params',       // Iterator[Parameter]
  SCALAR = 'scalar',       // Python numeric (int, float)
  BOOL = 'bool',           // Python bool
  STRING = 'string',       // Python str
  LIST = 'list',           // Python list
  DICT = 'dict',           // Python dict
  CONFIG = 'config',       // Configuration dict
  DEVICE = 'device',       // torch.device
  ANY = 'any',             // Any type
}

/**
 * Flow types for edges - determines visual style.
 */
export enum FlowType {
  TENSOR = 'tensor',           // Tensor data flow (blue, solid, animated)
  PARAMETERS = 'parameters',   // Parameter reference (green, dashed)
  GRADIENT = 'gradient',       // Gradient flow (red, dashed, animated)
  CONTROL = 'control',         // Control flow / sequence (gray, dashed)
  CONFIG = 'config',           // Configuration (purple, dotted)
  REFERENCE = 'reference',     // Variable reference (gray, solid)
}

/**
 * Node type classification - matches code constructs 1:1.
 */
export enum NodeKind {
  // Component Nodes (from setup)
  COMPONENT = 'component',         // self.x = self.create.category()

  // Operation Nodes (from train_step)
  FORWARD = 'forward',             // output = self.component(input)
  LOSS = 'loss',                   // loss = self.loss_fn(pred, target)
  BACKWARD = 'backward',           // loss.backward()
  OPTIMIZER_STEP = 'step',         // self.optimizer.step()
  OPTIMIZER_ZERO = 'zero_grad',    // self.optimizer.zero_grad()
  METHOD_CALL = 'method',          // x.method()

  // Control Flow Nodes
  IF = 'if',                       // if condition:
  ELIF = 'elif',                   // elif condition:
  ELSE = 'else',                   // else:
  FOR = 'for',                     // for x in y:
  WHILE = 'while',                 // while condition:
  WITH = 'with',                   // with context:

  // Data Nodes
  ASSIGN = 'assign',               // x = expression
  UNPACK = 'unpack',               // a, b = expression
  RETURN = 'return',               // return value

  // Special Nodes
  COMMENT = 'comment',             // # comment
  PASS = 'pass',                   // pass
}

/**
 * Method definition - callable operation on a node's output.
 * Example: model.parameters() -> MethodDefinition("parameters", PARAMS)
 */
export interface MethodDefinition {
  name: string;                    // Method name (e.g., "parameters", "eval")
  returns: DataType;               // Return type
  args?: string[];                 // Required arguments
  description?: string;            // Tooltip description
}

/**
 * Port definition - connection point on a node.
 */
export interface PortDefinition {
  name: string;                    // Port identifier (e.g., "in", "out", "params")
  data_type: DataType;             // Expected data type
  label?: string;                  // Display label
  required?: boolean;              // Must be connected
  multiple?: boolean;              // Allow multiple connections
  default_value?: string;          // Default value if not connected
  description?: string;            // Tooltip description
}

/**
 * Node type definition - template for a node type.
 * Contains ports, methods, and code generation template.
 */
export interface NodeTypeDefinition {
  node_type: NodeKind;
  category?: ComponentCategory;    // For component nodes

  // Code generation
  code_template?: string;          // Python code template with {placeholders}

  // Ports
  input_ports: PortDefinition[];
  output_ports: PortDefinition[];

  // Methods callable on output
  methods: MethodDefinition[];

  // Visual properties
  icon?: string;                   // Lucide icon name
  color?: string;                  // Theme color
  can_drill?: boolean;             // Can drill down into this node

  // Metadata
  description?: string;
}

// Re-export from edge types for consistency
// Authoritative source: src/features/node-system/model/edge.ts
export type { EdgeStyle } from '@/entities/node-system/model/edge';
export { EDGE_STYLES } from '@/entities/node-system/model/edge';

/**
 * Code-Node mapping for bidirectional sync.
 */
export interface CodeNodeMapping {
  node_id: string;
  line_start: number;
  line_end: number;
  code_snippet: string;
  ast_type: string;                // AST node type (Assign, Call, If, etc.)
}

/**
 * Methods API response - returns methods available for a category.
 */
export interface CategoryMethodsResponse {
  category: string;
  methods: MethodDefinition[];
}

export interface CategoryMethodsAPIResponse {
  success: boolean;
  data: CategoryMethodsResponse;
  error?: string;
}
