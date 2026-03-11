/**
 * Edge/Flow Types
 *
 * Based on L3_TSD/node-system/엣지-플로우-타입.md
 *
 * Loss-Centric Scope:
 * - Builder Tab: TENSOR, PARAMETERS, REFERENCE, CONTROL (active)
 * - Execute Tab: CONFIG, REFERENCE (active)
 * - GRADIENT: parsed but filtered out (handled by base Agent class)
 */

import type { TabMode } from './tab';

/**
 * Flow type for edges based on actual code patterns
 */
export const FlowType = {
  // Builder Tab (Data Flow Graph) - active
  TENSOR: 'tensor',           // 텐서 데이터 흐름 (Input → Loss)
  PARAMETERS: 'parameters',   // model.parameters() → optimizer

  // Execute Tab (Config Graph) - active
  CONFIG: 'config',           // 설정 파라미터 전달
  REFERENCE: 'reference',     // 컴포넌트 인스턴스 참조

  // Loss-Centric Scope 외 (파싱은 유지, 그래프에서 필터링)
  GRADIENT: 'gradient',       // loss.backward() - 범위 외
  CONTROL: 'control',         // optimizer.step() 등 - 범위 외
} as const;

export type FlowType = typeof FlowType[keyof typeof FlowType];

/**
 * Source code location for 1:1 code mapping
 */
interface OldSourceLocation {
  file: string;
  line: number;
  endLine?: number;
  column?: number;
  codeSnippet?: string;
}

/**
 * Edge data from backend
 */
import { SourceLocation, TabMode, CodeFlowEdge } from './types';

export interface CodeFlowEdgeOverride {
  id: string;
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
  flow_type: FlowType | string;
  edge_type?: 'execution' | 'data';
  variable_name?: string;
  extracted_from?: 'setup' | 'train_step';
  source?: SourceLocation;
  metadata?: Record<string, unknown>;
}

/**
 * Edge style configuration
 */
export interface EdgeStyle {
  color: string;
  strokeDasharray?: string;
  animated?: boolean;
  strokeWidth?: number;
}

/**
 * FlowType → Edge style mapping
 */
export const EDGE_STYLES: Record<FlowType, EdgeStyle> = {
  [FlowType.TENSOR]: {
    color: '#38bdf8',
    animated: false,
    strokeWidth: 2,
  },
  [FlowType.PARAMETERS]: {
    color: '#22c55e',
    strokeDasharray: '5,5',
    strokeWidth: 2,
  },
  [FlowType.REFERENCE]: {
    color: '#94a3b8',
    strokeWidth: 1.8,
  },
  [FlowType.CONFIG]: {
    color: '#f59e0b',
    strokeDasharray: '4,4',
    strokeWidth: 1.8,
  },
  // Out of scope (filtered, but define for completeness)
  [FlowType.GRADIENT]: {
    color: '#ef4444',
    strokeDasharray: '3,3',
    animated: true,
    strokeWidth: 2,
  },
  [FlowType.CONTROL]: {
    color: '#f8fafc',
    strokeWidth: 4.5,
    animated: true,
  },
};

export type EdgeLane = 'execution' | 'data';

export function getEdgeLane(flowType: FlowType | string | undefined): EdgeLane {
  if (flowType === FlowType.CONTROL) {
    return 'execution';
  }
  return 'data';
}

/**
 * FlowTypes that should be filtered out (Loss-Centric Scope)
 */
export const FILTERED_FLOW_TYPES = new Set<FlowType>([
  FlowType.GRADIENT,
  // FlowType.CONTROL is now allowed for execution pins
]);

/**
 * Get edge style for a flow type
 */
export function getEdgeStyle(flowType: FlowType | string): EdgeStyle {
  const ft = flowType as FlowType;
  return EDGE_STYLES[ft] || EDGE_STYLES[FlowType.REFERENCE];
}

/**
 * Check if edge should be visible in the given tab
 */
export function isEdgeVisibleInTab(edge: CodeFlowEdge, tab: TabMode): boolean {
  const flowType = edge.flow_type as FlowType;

  // Filter out-of-scope types
  if (FILTERED_FLOW_TYPES.has(flowType)) {
    return false;
  }

  // Tab-specific filtering
  if (tab === 'execute') {
    // Execute Tab: CONFIG, REFERENCE, PARAMETERS, CONTROL, TENSOR
    // Allowed all types for better connectivity visualization in setup view
    const allowed = new Set<FlowType>([
      FlowType.CONFIG,
      FlowType.REFERENCE,
      FlowType.PARAMETERS,
      FlowType.CONTROL,
      FlowType.TENSOR,
    ]);
    return allowed.has(flowType);
  }

  // Builder Tab: TENSOR, REFERENCE, CONTROL, PARAMETERS
  const allowed = new Set<FlowType>([
    FlowType.TENSOR,
    FlowType.REFERENCE,
    FlowType.CONTROL,
    FlowType.PARAMETERS,
  ]);
  return allowed.has(flowType);
}

/**
 * Generate unique edge ID
 */
export function generateEdgeId(
  sourceNode: string,
  targetNode: string,
  flowType: FlowType | string
): string {
  return `${sourceNode}-${targetNode}-${flowType}`;
}
