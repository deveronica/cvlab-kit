/**
 * Node System Types - Public API
 *
 * This is the SINGLE source of truth for all node system types.
 * Import from '@/entities/node-system/model/port' instead of scattered locations.
 */

// Tab system
export {
  type TabMode,
  NodeMode,
  getDefaultNodeMode,
  isConfigGraph,
  isDataFlowGraph,
} from './tab';

// Port system
export {
  PortType,
  type PortPosition,
  type PortZone,
  type UnifiedPort,
  type PortDefinition,
  type CategoryPorts,
  toPortType,
  createPort,
} from './port';

// Edge/Flow system
export {
  FlowType,
  type SourceLocation,
  type CodeFlowEdge,
  type EdgeStyle,
  type EdgeLane,
  EDGE_STYLES,
  FILTERED_FLOW_TYPES,
  getEdgeLane,
  getEdgeStyle,
  isEdgeVisibleInTab,
  generateEdgeId,
} from './edge';

// Node system
export {
  ComponentCategory,
  type ParamType,
  type UnifiedParam,
  type ImplementationOption,
  SpecialNodeType,
  type UnifiedNodeData,
  type NodeGraph,
  NODE_DIMENSIONS,
  SNAP_GRID,
  getNodeDimensions,
} from './node';
