import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { Node, Edge } from 'reactflow';

import type {
  TabMode,
  NodeMode,
  UnifiedNodeData,
  CodeFlowEdge,
} from './types';
import { getDefaultNodeMode } from './tab';
import { isEdgeVisibleInTab, getEdgeLane, getEdgeStyle } from './edge';

// Re-export core types for convenience (Barrel-style but safe)
export type { TabMode, NodeMode, UnifiedNodeData, CodeFlowEdge } from './types';
export { getEdgeLane, getEdgeStyle } from './edge';

export interface BreadcrumbItem {
  level: 'root' | 'agent' | 'method' | 'subsystem';
  label: string;
  path: string;
  nodeId?: string;
}

export interface SyncState {
  isDirty: boolean;
  lastSyncedCode: string;
  lastSyncedNodes: Record<string, unknown> | null;
  isSyncing: boolean;
  syncError: string | null;
}

export interface DraftState {
  draftId: string | null;
  isDirty: boolean;
  status: 'idle' | 'active' | 'saving' | 'committed' | 'discarded';
  addedNodes: Array<{
    id: string;
    category: string;
    implementation?: string;
    role?: string;
    position?: { x: number; y: number };
  }>;
  removedNodeIds: string[];
  modifiedNodes: Map<string, {
    implementation?: string;
    config?: Record<string, unknown>;
  }>;
  addedEdges: Array<{
    id: string;
    source: string;
    target: string;
    sourcePort: string;
    targetPort: string;
    flowType?: string;
  }>;
  removedEdgeIds: string[];
  saveError: string | null;
}

export interface Conflict {
  nodeId: string;
  type: 'code_changed' | 'node_changed' | 'both_changed';
  codeVersion: string;
  nodeVersion: Record<string, unknown>;
}

// Internal helper for state logic
export function getVisibleNodesFromState(
  nodes: Node<UnifiedNodeData>[],
  currentSubsystemId: string | null,
  nodeMode: NodeMode
): Node<UnifiedNodeData>[] {
  const filtered = currentSubsystemId
    ? nodes.filter((n) => (n.data as any).parentId === currentSubsystemId)
    : nodes.filter((n) => !(n.data as any).parentId);

  return filtered.map((node) => ({
    ...node,
    data: {
      ...node.data,
      mode: nodeMode,
    },
  }));
}

export function getVisibleEdgesFromState(
  edges: Edge[],
  codeFlowEdges: CodeFlowEdge[],
  currentTab: TabMode,
  visibleNodeIds: Set<string>
): Edge[] {
  let visibleEdges = edges.filter(
    (edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
  );

  const edgeFlowTypes = new Map(codeFlowEdges.map((edge) => [edge.id, edge]));

  visibleEdges = visibleEdges.filter((edge) => {
    const flowEdge = edgeFlowTypes.get(edge.id);
    if (!flowEdge) return true;
    return isEdgeVisibleInTab(flowEdge, currentTab);
  });

  return visibleEdges;
}

interface NodeStoreState {
  currentTab: TabMode;
  nodeMode: NodeMode;
  nodes: Node<UnifiedNodeData>[];
  edges: Edge[];
  codeFlowEdges: CodeFlowEdge[];
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  breadcrumb: BreadcrumbItem[];
  currentSubsystemId: string | null;
  agentName: string | null;
  agentPath: string | null;
  syncState: SyncState;
  conflicts: Conflict[];
  draftState: DraftState;
  isLoading: boolean;
  error: string | null;
  setTab: (tab: TabMode) => void;
  setNodeMode: (mode: NodeMode) => void;
  setNodes: (nodes: Node<UnifiedNodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  setCodeFlowEdges: (edges: CodeFlowEdge[]) => void;
  updateNode: (nodeId: string, data: Partial<UnifiedNodeData>) => void;
  addNode: (node: Node<UnifiedNodeData>) => void;
  removeNode: (nodeId: string) => void;
  selectNode: (nodeId: string | null) => void;
  hoverNode: (nodeId: string | null) => void;
  setBreadcrumb: (breadcrumb: BreadcrumbItem[]) => void;
  drillInto: (nodeId: string, label: string) => void;
  drillOut: () => void;
  navigateTo: (index: number) => void;
  setAgent: (name: string | null, path?: string | null) => void;
  updateSyncState: (state: Partial<SyncState>) => void;
  addConflict: (conflict: Conflict) => void;
  clearConflicts: () => void;
  markSynced: (code: string, nodes: Record<string, unknown>) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  initDraft: (draftId: string) => void;
  draftAddNode: (node: DraftState['addedNodes'][0]) => void;
  draftRemoveNode: (nodeId: string) => void;
  draftModifyNode: (nodeId: string, changes: { implementation?: string; config?: Record<string, unknown> }) => void;
  draftAddEdge: (edge: DraftState['addedEdges'][0]) => void;
  draftRemoveEdge: (edgeId: string) => void;
  getDraftChanges: () => {
    addedNodes: DraftState['addedNodes'];
    removedNodeIds: string[];
    modifiedNodes: Array<{ nodeId: string; changes: { implementation?: string; config?: Record<string, unknown> } }>;
    addedEdges: DraftState['addedEdges'];
    removedEdgeIds: string[];
  };
  setDraftStatus: (status: DraftState['status']) => void;
  setDraftError: (error: string | null) => void;
  clearDraft: () => void;
  hasDraftChanges: () => boolean;
  getVisibleNodes: () => Node<UnifiedNodeData>[];
  getVisibleEdges: () => Edge[];
  getSelectedNode: () => Node<UnifiedNodeData> | null;
}

export const useNodeStore = create<NodeStoreState>()(
  subscribeWithSelector((set, get) => ({
    currentTab: 'builder',
    nodeMode: 'builder',
    nodes: [],
    edges: [],
    codeFlowEdges: [],
    selectedNodeId: null,
    hoveredNodeId: null,
    breadcrumb: [{ level: 'root', label: 'Agent', path: '' }],
    currentSubsystemId: null,
    agentName: null,
    agentPath: null,
    syncState: { isDirty: false, lastSyncedCode: '', lastSyncedNodes: null, isSyncing: false, syncError: null },
    conflicts: [],
    draftState: { draftId: null, isDirty: false, status: 'idle', addedNodes: [], removedNodeIds: [], modifiedNodes: new Map(), addedEdges: [], removedEdgeIds: [], saveError: null },
    isLoading: false,
    error: null,
    setTab: (tab) => set({ currentTab: tab, nodeMode: getDefaultNodeMode(tab) }),
    setNodeMode: (mode) => set({ nodeMode: mode }),
    setNodes: (nodes) => set({ nodes }),
    setEdges: (edges) => set({ edges }),
    setCodeFlowEdges: (codeFlowEdges) => set({ codeFlowEdges }),
    updateNode: (nodeId, data) => set((state) => ({ nodes: state.nodes.map((node) => node.id === nodeId ? { ...node, data: { ...node.data, ...data } } : node) })),
    addNode: (node) => set((state) => ({ nodes: [...state.nodes, node] })),
    removeNode: (nodeId) => set((state) => ({ nodes: state.nodes.filter((n) => n.id !== nodeId), edges: state.edges.filter((e) => e.source !== nodeId && e.target !== nodeId), selectedNodeId: state.selectedNodeId === nodeId ? null : state.selectedNodeId })),
    selectNode: (nodeId) => set({ selectedNodeId: nodeId }),
    hoverNode: (nodeId) => set({ hoveredNodeId: nodeId }),
    setBreadcrumb: (breadcrumb) => set({ breadcrumb }),
    drillInto: (nodeId, label) => { const state = get(); const newBreadcrumb: BreadcrumbItem[] = [...state.breadcrumb, { level: 'subsystem', label, path: nodeId, nodeId }]; set({ breadcrumb: newBreadcrumb, currentSubsystemId: nodeId, selectedNodeId: null }); },
    drillOut: () => { const state = get(); if (state.breadcrumb.length <= 1) return; const newBreadcrumb = state.breadcrumb.slice(0, -1); const parentItem = newBreadcrumb[newBreadcrumb.length - 1]; set({ breadcrumb: newBreadcrumb, currentSubsystemId: parentItem.nodeId || null, selectedNodeId: null }); },
    navigateTo: (index) => { const state = get(); if (index < 0 || index >= state.breadcrumb.length) return; const newBreadcrumb = state.breadcrumb.slice(0, index + 1); const targetItem = newBreadcrumb[newBreadcrumb.length - 1]; set({ breadcrumb: newBreadcrumb, currentSubsystemId: targetItem.nodeId || null, selectedNodeId: null }); },
    setAgent: (name, path = null) => set({ agentName: name, agentPath: path, breadcrumb: name ? [{ level: 'root', label: name, path: path || '' }] : [{ level: 'root', label: 'Agent', path: '' }], currentSubsystemId: null, selectedNodeId: null }),
    updateSyncState: (state) => set((prev) => ({ syncState: { ...prev.syncState, ...state } })),
    addConflict: (conflict) => set((state) => ({ conflicts: [...state.conflicts, conflict] })),
    clearConflicts: () => set({ conflicts: [] }),
    markSynced: (code, nodes) => set((state) => ({ syncState: { ...state.syncState, isDirty: false, lastSyncedCode: code, lastSyncedNodes: nodes, syncError: null }, conflicts: [] })),
    setLoading: (loading) => set({ isLoading: loading }),
    setError: (error) => set({ error }),
    initDraft: (draftId) => set((state) => ({ draftState: { ...state.draftState, draftId, isDirty: false, status: 'active', addedNodes: [], removedNodeIds: [], modifiedNodes: new Map(), addedEdges: [], removedEdgeIds: [], saveError: null } })),
    draftAddNode: (node) => set((state) => ({ draftState: { ...state.draftState, isDirty: true, addedNodes: [...state.draftState.addedNodes, node] } })),
    draftRemoveNode: (nodeId) => set((state) => { const wasAdded = state.draftState.addedNodes.some((n) => n.id === nodeId); if (wasAdded) return { draftState: { ...state.draftState, isDirty: true, addedNodes: state.draftState.addedNodes.filter((n) => n.id !== nodeId) } }; return { draftState: { ...state.draftState, isDirty: true, removedNodeIds: [...state.draftState.removedNodeIds, nodeId] } }; }),
    draftModifyNode: (nodeId, changes) => set((state) => { const newModifiedNodes = new Map(state.draftState.modifiedNodes); const existing = newModifiedNodes.get(nodeId) || {}; newModifiedNodes.set(nodeId, { ...existing, ...changes }); return { draftState: { ...state.draftState, isDirty: true, modifiedNodes: newModifiedNodes } }; }),
    draftAddEdge: (edge) => set((state) => ({ draftState: { ...state.draftState, isDirty: true, addedEdges: [...state.draftState.addedEdges, edge] } })),
    draftRemoveEdge: (edgeId) => set((state) => { const wasAdded = state.draftState.addedEdges.some((e) => e.id === edgeId); if (wasAdded) return { draftState: { ...state.draftState, isDirty: true, addedEdges: state.draftState.addedEdges.filter((e) => e.id !== edgeId) } }; return { draftState: { ...state.draftState, isDirty: true, removedEdgeIds: [...state.draftState.removedEdgeIds, edgeId] } }; }),
    getDraftChanges: () => { const { draftState } = get(); return { addedNodes: draftState.addedNodes, removedNodeIds: draftState.removedNodeIds, modifiedNodes: Array.from(draftState.modifiedNodes.entries()).map(([nodeId, changes]) => ({ nodeId, changes })), addedEdges: draftState.addedEdges, removedEdgeIds: draftState.removedEdgeIds }; },
    setDraftStatus: (status) => set((state) => ({ draftState: { ...state.draftState, status } })),
    setDraftError: (error) => set((state) => ({ draftState: { ...state.draftState, saveError: error, status: error ? 'active' : state.draftState.status } })),
    clearDraft: () => set(() => ({ draftState: { draftId: null, isDirty: false, status: 'idle', addedNodes: [], removedNodeIds: [], modifiedNodes: new Map(), addedEdges: [], removedEdgeIds: [], saveError: null } })),
    hasDraftChanges: () => { const { draftState } = get(); return draftState.addedNodes.length > 0 || draftState.removedNodeIds.length > 0 || draftState.modifiedNodes.size > 0 || draftState.addedEdges.length > 0 || draftState.removedEdgeIds.length > 0; },
    getVisibleNodes: () => { const state = get(); return getVisibleNodesFromState(state.nodes, state.currentSubsystemId, state.nodeMode); },
    getVisibleEdges: () => { const state = get(); const visibleNodes = getVisibleNodesFromState(state.nodes, state.currentSubsystemId, state.nodeMode); const visibleNodeIds = new Set(visibleNodes.map((node) => node.id)); return getVisibleEdgesFromState(state.edges, state.codeFlowEdges, state.currentTab, visibleNodeIds); },
    getSelectedNode: () => { const state = get(); return state.nodes.find((n) => n.id === state.selectedNodeId) || null; },
  }))
);
