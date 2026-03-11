/**
 * useNodeGraph - Hook for fetching and managing node graph data
 *
 * Fetches hierarchy graph from backend and transforms it for ReactFlow.
 * Works with both Execute Tab and Builder Tab.
 *
 * Features position memory and collision-free layout.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Node, Edge, XYPosition } from 'reactflow';
import * as dagre from 'dagre';

import { useNodeStore } from '@/entities/node-system/model/port';
import { useBuilderStore } from '@/features/builder/model/builderStore';
import type {
  TabMode,
  UnifiedNodeData,
  CodeFlowEdge,
  UnifiedPort,
} from '@/entities/node-system/model/port';
import { getDefaultNodeMode, getEdgeStyle, getNodeDimensions, getEdgeLane } from '@/entities/node-system/model/port';
import { isEdgeVisibleInTab } from '@/entities/node-system/model/edge';
import { createPort, PortType } from '@/entities/node-system/model/port';
import type { SpecialNodeType } from '@/entities/node-system/model/node';
import { getExecuteTabPorts, getBuilderTabPorts } from '@/entities/node-system/config/port-registry';

// =============================================================================
// API Types (from backend)
// =============================================================================

interface BackendNode {
  id: string;
  role?: string;
  label?: string;
  category: string;
  implementation?: string;
  params?: Record<string, unknown>;
  source?: {
    file: string;
    line: number;
    end_line?: number;
  };
  canDrill?: boolean;
  can_drill?: boolean; // Support both naming styles for reliability
  metadata?: Record<string, unknown>;
  inputs?: any[];
  outputs?: any[];
}

interface BackendEdge {
  id: string;
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
  flow_type: string;
  variable_name?: string; edge_type?: 'execution' | 'data' | 'ref';
  extracted_from?: string;
  metadata?: {
    is_gapped?: boolean;
    comment?: string;
    line_gap?: number;
  };
}

interface HierarchyGraphResponse {
  nodes: BackendNode[];
  edges: BackendEdge[];
  agent_name: string;
  agent_path?: string;
  covered_lines?: number[];
  uncovered_lines?: number[];
}

// =============================================================================
// HELPER: Position Memory
// =============================================================================

const STORAGE_KEY_PREFIX = 'cvlab-node-positions:';

function getSavedPositions(agentName: string, tab: string): Record<string, XYPosition> {
  try {
    const data = localStorage.getItem(`${STORAGE_KEY_PREFIX}${agentName}:${tab}`);
    return data ? JSON.parse(data) : {};
  } catch {
    return {};
  }
}

function savePositions(agentName: string, tab: string, nodes: Node[]) {
  const positions: Record<string, XYPosition> = {};
  nodes.forEach(n => {
    positions[n.id] = n.position;
  });
  localStorage.setItem(`${STORAGE_KEY_PREFIX}${agentName}:${tab}`, JSON.stringify(positions));
}

// =============================================================================
// Transform Functions
// =============================================================================

/**
 * Transform backend node to ReactFlow node
 */
function transformNode(
  backendNode: BackendNode,
  tab: TabMode
): Node<UnifiedNodeData> {
  const role = backendNode.role || backendNode.label || backendNode.id;
  const category = (backendNode.category || 'unknown') as ComponentCategory;
  const mode = getDefaultNodeMode(tab);
  const nodeType = backendNode.metadata?.node_type as SpecialNodeType | undefined;

  // Requirement 5: Robust handle matching
  // 1. Process data ports (좌우)
  const dataInputs = (backendNode.metadata?.data_inputs as string[] | undefined) || [];
  const dataOutputs = (backendNode.metadata?.data_outputs as string[] | undefined) || [];
  
  // Use explicit backend ports if available, otherwise fallback to metadata names
  const inputs = (backendNode.inputs && backendNode.inputs.length > 0)
    ? backendNode.inputs.map(p => ({ ...createPort(p.name, p.type as any), kind: p.kind || 'data' }))
    : dataInputs.map(name => ({ ...createPort(name, PortType.ANY), kind: 'data' }));
    
  const outputs = (backendNode.outputs && backendNode.outputs.length > 0)
    ? backendNode.outputs.map(p => ({ ...createPort(p.name, p.type as any), kind: p.kind || 'data' }))
    : dataOutputs.map(name => ({ ...createPort(name, PortType.ANY), kind: 'data' }));
  
  // Always ensure 'self' output exists for references (Requirement 5 fallback)
  if (!outputs.some(p => p.id === 'self')) {
    outputs.push({ ...createPort('self', PortType.MODULE), kind: 'data' });
  }

  // 2. Process execution pins (상하)
  const controlInputs = (backendNode.metadata?.control_inputs as string[] | undefined) || [];
  const controlOutputs = (backendNode.metadata?.control_outputs as string[] | undefined) || [];
  
  let finalExecPins: UnifiedPort[] = [];
  
  // Map backend 'control' ports to standardized 'in'/'out' handles
  controlInputs.forEach(name => finalExecPins.push({ ...createPort(name === 'control' ? 'in' : name, PortType.EXECUTION), kind: 'exec' }));
  controlOutputs.forEach(name => finalExecPins.push({ ...createPort(name === 'control' ? 'out' : name, PortType.EXECUTION), kind: 'exec' }));
  
  // Requirement 5: Always provide mandatory in/out handles for connectivity reliability
  if (!finalExecPins.some(p => p.id === 'in')) finalExecPins.push({ ...createPort('in', PortType.EXECUTION), kind: 'exec' });
  if (!finalExecPins.some(p => p.id === 'out' || p.id === 'true' || p.id === 'false')) {
    finalExecPins.push({ ...createPort('out', PortType.EXECUTION), kind: 'exec' });
  }

  // Transform params
  const params = backendNode.params
    ? Object.entries(backendNode.params).map(([name, value]) => ({
        name,
        value: value as string | number | boolean,
        type: typeof value === 'boolean' ? 'boolean' as const :
              typeof value === 'number' ? 'number' as const : 'string' as const,
      }))
    : [];

  return {
    id: backendNode.id,
    type: 'unified',
    position: { x: 0, y: 0 },
    data: {
      id: backendNode.id,
      role,
      category,
      nodeType,
      implementation: backendNode.implementation,
      inputs,
      outputs,
      executionPins: finalExecPins,
      usedConfigKeys: (backendNode as any).used_config_keys || [],
      params,
      mode,
      source: backendNode.source ? {
        ...backendNode.source,
        endLine: backendNode.source.end_line // Map snake_case to camelCase
      } : undefined,
      canDrill: backendNode.canDrill || backendNode.can_drill,
    },
  };
}

/**
 * Transform backend edge to ReactFlow edge
 */
function transformEdge(backendEdge: BackendEdge, tab: TabMode): Edge | null {
  const flowType = backendEdge.flow_type || 'reference';
  const edgeType = backendEdge.edge_type || getEdgeLane(flowType);

  // Requirement 5: Standardized handle naming for 1:1 mapping
  // Map backend 'control' ports to frontend 'in'/'out' handles
  const sourceHandleRaw = (backendEdge.source_port === 'control' || !backendEdge.source_port) ? 'out' : backendEdge.source_port;
  const targetHandleRaw = (backendEdge.target_port === 'control' || !backendEdge.target_port) ? 'in' : backendEdge.target_port;

  // IMPORTANT: UnifiedNode uses prefixed IDs for handles: target:id and source:id
  const sourceHandle = `source:${sourceHandleRaw}`;
  const targetHandle = `target:${targetHandleRaw}`;

  const codeFlowEdge: CodeFlowEdge = {
    id: backendEdge.id,
    source_node: backendEdge.source_node,
    source_port: sourceHandleRaw, // Keep raw for visibility check
    target_node: backendEdge.target_node,
    target_port: targetHandleRaw, // Keep raw for visibility check
    flow_type: flowType,
    variable_name: backendEdge.variable_name,
    extracted_from: backendEdge.extracted_from as 'setup' | 'train_step' | undefined,
  };

  // Filter by tab
  if (!isEdgeVisibleInTab(codeFlowEdge, tab)) {
    return null;
  }

  const style = getEdgeStyle(flowType);

  return {
    id: backendEdge.id,
    source: backendEdge.source_node,
    target: backendEdge.target_node,
    sourceHandle: sourceHandle,
    targetHandle: targetHandle,
    type: 'custom',
    data: {
      flowType: flowType,
      variableName: backendEdge.variable_name,
      edgeType: edgeType,
      sequenceIndex: (backendEdge as any).sequence_index,
      extracted_from: backendEdge.extracted_from as any,
      is_gapped: backendEdge.metadata?.is_gapped,
      comment: backendEdge.metadata?.comment,
    },
    style: {
      stroke: style.color,
      strokeWidth: style.strokeWidth || 2,
      strokeDasharray: style.strokeDasharray,
    },
    animated: style.animated || edgeType === 'data',
  };
}

/**
 * Row-based Grid Layout Algorithm (Requirement 3)
 * - Continuous code lines (gap <= 1) -> Same Row (LR)
 * - Empty line gaps (gap > 1) -> New Row
 */
function applyRowLayout(
  nodes: Node<UnifiedNodeData>[],
  edges: Edge[],
  savedPositions: Record<string, XYPosition> = {}
): Node<UnifiedNodeData>[] {
  if (nodes.length === 0) return [];

  // 1. Sort nodes by line number
  const sortedNodes = [...nodes].sort((a, b) => {
    const lineA = (a.data.source?.line || 0);
    const lineB = (b.data.source?.line || 0);
    return lineA - lineB;
  });

  // 2. Group into rows based on gaps (Requirement 3.2, 3.3)
  const rows: Node<UnifiedNodeData>[][] = [];
  let currentRow: Node<UnifiedNodeData>[] = [];
  let lastLineEnd = -1;

  for (const node of sortedNodes) {
    const currentLineStart = node.data.source?.line || 0;
    const currentLineEnd = node.data.source?.end_line || currentLineStart;
    const hasGap = lastLineEnd !== -1 && (currentLineStart - lastLineEnd) > 1;

    if (hasGap && currentRow.length > 0) {
      rows.push(currentRow);
      currentRow = [];
    }
    currentRow.push(node);
    lastLineEnd = currentLineEnd;
  }
  if (currentRow.length > 0) rows.push(currentRow);

  // 3. Position based on row height
  const layoutedNodes: Node<UnifiedNodeData>[] = [];
  const PADDING_X = 80;
  const PADDING_Y = 160;
  const START_X = 100;
  let currentY = 100;

  rows.forEach((row, rowIndex) => {
    let maxRowHeight = 0;
    row.forEach(n => {
      const { height } = getNodeDimensions(n.data.mode, n.data.nodeType as unknown as SpecialNodeType);
      maxRowHeight = Math.max(maxRowHeight, height);
    });

    let currentX = START_X;
    row.forEach((node) => {
      const { height, width } = getNodeDimensions(node.data.mode, node.data.nodeType as unknown as SpecialNodeType);
      const verticalOffset = (maxRowHeight - height) / 2;
      
      layoutedNodes.push({
        ...node,
        position: { x: currentX, y: rowIndex * 400 + 100 + verticalOffset },
        data: { ...node.data, metadata: { ...node.data.metadata, row_index: rowIndex } }
      });
      
      currentX += width + PADDING_X;
    });
  });

  return layoutedNodes;
}

// =============================================================================
// Hook
// =============================================================================

interface UseNodeGraphOptions {
  agentName?: string;
  autoFetch?: boolean;
  onCodeClick?: (source: { file: string; line: number }) => void;
}

export function useNodeGraph(options: UseNodeGraphOptions = {}) {
  const { agentName, autoFetch = true, onCodeClick } = options;

  const queryClient = useQueryClient();

  // Store actions
  const currentTab = useNodeStore((s) => s.currentTab);
  const setNodes = useNodeStore((s) => s.setNodes);
  const setEdges = useNodeStore((s) => s.setEdges);
  const setCodeFlowEdges = useNodeStore((s) => s.setCodeFlowEdges);
  const setAgent = useNodeStore((s) => s.setAgent);
  const setLoading = useNodeStore((s) => s.setLoading);
  const setError = useNodeStore((s) => s.setError);

  const selectedConfig = useBuilderStore((s) => s.selectedConfig);

  // Fetch query
  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery<HierarchyGraphResponse>({
    queryKey: ['hierarchy-graph', agentName, currentTab],
    queryFn: async () => {
      if (!agentName) throw new Error('Agent name required');

      // Strip extensions if present
      const cleanAgentName = agentName.replace(/\.(py|yaml|yml)$/, '');

      // Map currentTab (execute/builder) to backend phase (initialize/flow)
      const phase = currentTab === 'execute' ? 'initialize' : 'flow';

      const configPath = selectedConfig?.path;
      const configQuery = configPath ? `&config_path=${encodeURIComponent(configPath)}` : '';

      const response = await fetch(
        `/api/nodes/hierarchy/${encodeURIComponent(cleanAgentName)}?phase=${phase}${configQuery}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch graph: ${response.statusText}`);
      }

      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch graph');
      }
      return result.data;
    },
    enabled: autoFetch && !!agentName,
    staleTime: 30000, // 30 seconds
  });

  // Transform and apply data when fetched
  useEffect(() => {
    if (!data) return;

    setLoading(true);

    try {
      const builderStore = useBuilderStore.getState();
      builderStore.setUncoveredLines(data.uncovered_lines || []);
      
      const savedPositions = agentName ? getSavedPositions(agentName, currentTab) : {};

      let nodes = data.nodes.map((n) => transformNode(n, currentTab));
      const edges = data.edges
        .map((e) => transformEdge(e, currentTab))
        .filter((e): e is Edge => e !== null);

      const codeFlowEdges: CodeFlowEdge[] = data.edges.map((e) => ({
        id: e.id,
        source_node: e.source_node,
        source_port: e.source_port,
        target_node: e.target_node,
        target_port: e.target_port,
        flow_type: e.flow_type,
        variable_name: e.variable_name,
        extracted_from: e.extracted_from as 'setup' | 'train_step' | undefined,
      }));

      // Requirement 3: Apply Row Layout immediately after transformation
      const layoutedNodes = applyRowLayout(nodes, edges, savedPositions);

      // Add callbacks to node data
      const finalNodes = layoutedNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          onCodeClick,
        },
      }));

      // Update store with fully layouted nodes
      setNodes(finalNodes);
      setEdges(edges);
      setCodeFlowEdges(codeFlowEdges);
      setAgent(data.agent_name, data.agent_path);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [data, currentTab, setNodes, setEdges, setCodeFlowEdges, setAgent, setLoading, setError, onCodeClick, agentName]);

  // Sync loading state
  useEffect(() => {
    setLoading(isLoading);
  }, [isLoading, setLoading]);

  // Sync error state
  useEffect(() => {
    if (error) {
      setError(error instanceof Error ? error.message : 'Unknown error');
    }
  }, [error, setError]);

  // Manual save trigger (could be called onNodeChange)
  const persistPositions = useCallback((nodes: Node[]) => {
    if (agentName) {
      savePositions(agentName, currentTab, nodes);
    }
  }, [agentName, currentTab]);

  // Refresh function
  const refresh = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['hierarchy-graph', agentName] });
    refetch();
  }, [queryClient, agentName, refetch]);

  return {
    isLoading,
    error: error instanceof Error ? error.message : error ? String(error) : null,
    refresh,
    persistPositions
  };
}
