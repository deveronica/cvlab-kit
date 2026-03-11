import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { Node, Edge } from 'reactflow';
import type { CodeBlock, SyncState, CodeNodeMapping, CursorPosition } from '@/shared/model/ast-blocks';

// =============================================================================
// Types
// =============================================================================

// 2-Tab System (ADR-008: Node System Design)
// - Execute Tab: Config Graph (setup dependencies)
// - Builder Tab: Data Flow Graph (Input → Loss, Loss-Centric Scope)
export type BuilderTab = 'execute' | 'builder';

// Node state for PyTorch-specific features
export interface NodeState {
  id: string;
  mode: 'train' | 'eval';  // For model/loss nodes
  device: string;          // 'cpu' | 'cuda:0' | 'cuda:1' etc.
  requiresGrad: boolean;   // Gradient tracking
  selected: boolean;
}

export interface AgentFile {
  type: 'agent' | 'config';
  path: string;
  name: string;
  content?: string;
}

// Method definition for context menu
export interface MethodAction {
  name: string;
  returns: string;
  args?: string[];
  description?: string;
}

// Context menu state
export interface ContextMenuState {
  visible: boolean;
  x: number;
  y: number;
  nodeId: string | null;
  nodeCategory: string | null;
}

// Component types
interface ComponentVersion {
  hash: string;
  path: string;
  category: string;
  name: string;
  is_active: boolean;
  created_at: string;
  content?: string;
}

interface ComponentItem {
  path: string;
  category: string;
  name: string;
  active_hash: string | null;
  version_count: number;
  updated_at: string | null;
}

interface CodeSymbol {
  name: string;
  type: 'class' | 'function' | 'method';
  line: number;
  indent: number;
  parent?: string;
}

export type BreadcrumbItem = {
  level: 'root' | 'package' | 'subpackage' | 'file' | 'symbol' | 'method';
  label: string;
  path: string;
};

type BuilderState = {
  // Global UI State
  leftPanelOpen: boolean;
  rightPanelOpen: boolean;
  activeViewMode: 'components' | 'code' | 'nodes'; // Renamed from 'mode' to avoid conflict

  // 2-Tab System (ADR-008)
  builderTab: BuilderTab;

  // Breadcrumb for navigation
  breadcrumb: BreadcrumbItem[];

  // State for ComponentsMode
  components: ComponentItem[];
  selectedComponent: ComponentItem | null;
  componentVersions: ComponentVersion[]; // Renamed from 'versions'
  activeComponentVersion: ComponentVersion | null;
  codeSymbols: CodeSymbol[];

  // =========================================================================
  // Node Graph State (for Builder tab)
  // =========================================================================
  nodes: Node[];
  edges: Edge[];
  nodeStates: Map<string, NodeState>;  // PyTorch-specific state per node
  selectedNodeId: string | null;
  selectedEdgeId: string | null;  // Req 4: Edge selection for Inspector editing

  // Context Menu
  contextMenu: ContextMenuState;

  // =========================================================================
  // Draft State (for Node Graph editing with API sync)
  // =========================================================================
  selectedAgent: AgentFile | null;
  selectedConfig: AgentFile | null;
  isEditingMode: boolean;
  draftId: string | null;
  draftStatus: 'clean' | 'modified' | 'committed' | 'discarded' | null;
  agentName: string | null;
  isLoading: boolean;
  error: string | null;

  // =========================================================================
  // Bidirectional Sync State (Code ↔ Nodes)
  // =========================================================================
  codeBlocks: CodeBlock[];
  uncoveredLines: number[];
  syncState: SyncState;
  codeNodeMappings: CodeNodeMapping[];
  cursorPosition: CursorPosition;
  typeColors: Record<string, string>; // 항목 2: 동적 타입 색상

  // Actions
  toggleLeftPanel: () => void;
  toggleRightPanel: () => void;
  setActiveViewMode: (mode: 'components' | 'code' | 'nodes') => void;
  setBuilderTab: (tab: BuilderTab) => void;
  setBreadcrumb: (breadcrumb: BreadcrumbItem[]) => void;
  setTypeColors: (colors: Record<string, string>) => void; // 항목 2: 색상 로드 액션

  // Component related actions
  setComponents: (components: ComponentItem[]) => void;
  setSelectedComponent: (component: ComponentItem | null) => void;
  setComponentVersions: (versions: ComponentVersion[]) => void;
  setActiveComponentVersion: (version: ComponentVersion | null) => void;
  setCodeSymbols: (symbols: CodeSymbol[]) => void;

  // Initial fetch status
  isComponentsLoaded: boolean;
  setComponentsLoaded: (loaded: boolean) => void;

  // =========================================================================
  // Node Graph Actions
  // =========================================================================
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  updateNodes: (updater: (nodes: Node[]) => Node[]) => void;
  updateEdges: (updater: (edges: Edge[]) => Edge[]) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
  setSelectedEdgeId: (edgeId: string | null) => void;  // Req 4: Select edge

  // Node state actions (PyTorch-specific)
  setNodeMode: (nodeId: string, mode: 'train' | 'eval') => void;
  setNodeDevice: (nodeId: string, device: string) => void;
  toggleNodeGrad: (nodeId: string) => void;

  // Context menu actions
  showContextMenu: (x: number, y: number, nodeId: string, category: string) => void;
  hideContextMenu: () => void;

  // =========================================================================
  // Draft Actions
  // =========================================================================
  setSelectedAgent: (agent: AgentFile | null) => void;
  setSelectedConfig: (config: AgentFile | null) => void;
  setIsEditingMode: (editing: boolean) => void;
  setIsDirty: (isDirty: boolean) => void;
  setDraft: (draftId: string | null, status: 'clean' | 'modified' | 'committed' | 'discarded' | null) => void;
  setAgentName: (agentName: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  deleteNodeWithApi: (nodeId: string) => Promise<boolean>;

  // =========================================================================
  // Bidirectional Sync Actions
  // =========================================================================
  setCodeBlocks: (blocks: CodeBlock[]) => void;
  setUncoveredLines: (lines: number[]) => void;
  updateSyncState: (state: Partial<SyncState>) => void;
  setCodeNodeMappings: (mappings: CodeNodeMapping[]) => void;
  setCursorPosition: (position: CursorPosition) => void;
  markSynced: (code: string, nodes: Record<string, unknown>) => void;
  addConflict: (conflict: import('@/shared/model/ast-blocks').Conflict) => void;
  clearConflicts: () => void;
};

const STORAGE_KEY = 'builder-panel-state';

function loadPanelState(): { left: boolean; right: boolean } {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return JSON.parse(saved);
  } catch {}
  return { left: true, right: true }; // Default to open
}

export const useBuilderStore = create<BuilderState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    leftPanelOpen: loadPanelState().left,
    rightPanelOpen: loadPanelState().right,
    activeViewMode: 'components',
    builderTab: 'builder', // Default to Builder Tab (Data Flow Graph)
    breadcrumb: [{ level: 'root', label: 'cvlabkit', path: '' }],
    components: [],
    selectedComponent: null,
    componentVersions: [],
    activeComponentVersion: null,
    codeSymbols: [],
    isComponentsLoaded: false,

    // Node graph state
    nodes: [],
    edges: [],
    nodeStates: new Map<string, NodeState>(),
    selectedNodeId: null,
    selectedEdgeId: null,  // Req 4: Edge selection

    // Context menu (hidden by default)
    contextMenu: {
      visible: false,
      x: 0,
      y: 0,
      nodeId: null,
      nodeCategory: null,
    },

    // Draft state
    selectedAgent: null,
    selectedConfig: null,
    isEditingMode: false,
    draftId: null,
    draftStatus: null,
    agentName: null,
    isLoading: false,
    error: null,

    // Bidirectional sync state
    codeBlocks: [],
    uncoveredLines: [],
    syncState: {
      isDirty: false,
      lastSyncedCode: '',
      lastSyncedNodes: null,
      conflicts: [],
      isSyncing: false,
      syncError: null,
    },
    codeNodeMappings: [],
    cursorPosition: { line: 1, col: 0 },
    typeColors: { execution: '#ffffff' },

    // Actions
    setTypeColors: (typeColors) => set({ typeColors: { ...typeColors, execution: '#ffffff' } }),
    setSelectedAgent: (agent) => set({ selectedAgent: agent, agentName: agent?.name.replace('.py', '') || null }),
    setSelectedConfig: (config) => set({ selectedConfig: config }),
    setIsEditingMode: (isEditingMode) => set({ isEditingMode }),
    setIsDirty: (isDirty) => set((state) => ({ 
      syncState: { ...state.syncState, isDirty } 
    })),
    toggleLeftPanel: () => set((state) => {
      const newLeft = !state.leftPanelOpen;
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ left: newLeft, right: state.rightPanelOpen }));
      return { leftPanelOpen: newLeft };
    }),
    toggleRightPanel: () => set((state) => {
      const newRight = !state.rightPanelOpen;
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ left: state.leftPanelOpen, right: newRight }));
      return { rightPanelOpen: newRight };
    }),
    setActiveViewMode: (mode) => set({ activeViewMode: mode }),
    setBuilderTab: (tab) => set({ builderTab: tab }),
    setBreadcrumb: (breadcrumb) => set({ breadcrumb }),

    // Component actions
    setComponents: (components) => set({ components }),
    setSelectedComponent: (component) => set({ selectedComponent: component }),
    setComponentVersions: (versions) => set({ componentVersions: versions }),
    setActiveComponentVersion: (version) => set({ activeComponentVersion: version }),
    setCodeSymbols: (symbols) => set({ codeSymbols: symbols }),
    setComponentsLoaded: (loaded) => set({ isComponentsLoaded: loaded }),

    // Node graph actions
    setNodes: (nodes) => set({ nodes }),
    setEdges: (edges) => set({ edges }),
    updateNodes: (updater) => set((state) => ({ nodes: updater(state.nodes) })),
    updateEdges: (updater) => set((state) => ({ edges: updater(state.edges) })),
    setSelectedNodeId: (nodeId) => set({ selectedNodeId: nodeId, selectedEdgeId: null }),  // Clear edge selection
    setSelectedEdgeId: (edgeId) => set({ selectedEdgeId: edgeId, selectedNodeId: null }),  // Clear node selection (Req 4)

    // Node state actions
    setNodeMode: (nodeId, mode) => set((state) => {
      const newStates = new Map(state.nodeStates);
      const nodeState = newStates.get(nodeId) || { id: nodeId, mode, device: 'cpu', requiresGrad: true, selected: false };
      newStates.set(nodeId, { ...nodeState, mode });
      return { nodeStates: newStates };
    }),
    setNodeDevice: (nodeId, device) => set((state) => {
      const newStates = new Map(state.nodeStates);
      const nodeState = newStates.get(nodeId) || { id: nodeId, mode: 'train', device, requiresGrad: true, selected: false };
      newStates.set(nodeId, { ...nodeState, device });
      return { nodeStates: newStates };
    }),
    toggleNodeGrad: (nodeId) => set((state) => {
      const newStates = new Map(state.nodeStates);
      const nodeState = newStates.get(nodeId) || { id: nodeId, mode: 'train', device: 'cpu', requiresGrad: true, selected: false };
      newStates.set(nodeId, { ...nodeState, requiresGrad: !nodeState.requiresGrad });
      return { nodeStates: newStates };
    }),

    // Context menu actions
    showContextMenu: (x, y, nodeId, category) => set({
      contextMenu: { visible: true, x, y, nodeId, nodeCategory: category }
    }),
    hideContextMenu: () => set((state) => ({
      contextMenu: { ...state.contextMenu, visible: false }
    })),

    // Draft actions
    setDraft: (draftId, status) => set({ draftId, draftStatus: status }),
    setAgentName: (agentName) => set({ agentName }),
    setLoading: (loading) => set({ isLoading: loading }),
    setError: (error) => set({ error }),

    deleteNodeWithApi: async (nodeId: string) => {
      try {
        const response = await fetch('/api/node-graph/nodes/' + nodeId, { method: 'DELETE' });
        if (response.ok) {
          get().updateNodes((nodes) => nodes.filter((n) => n.id !== nodeId));
          return true;
        }
        return false;
      } catch (error) {
        console.error('Failed to delete node:', error);
        return false;
      }
    },

    // Bidirectional sync actions
    setCodeBlocks: (blocks) => set({ codeBlocks: blocks }),
    setUncoveredLines: (lines) => set({ uncoveredLines: lines }),
    updateSyncState: (newState) => set((state) => ({ 
      syncState: { ...state.syncState, ...newState } 
    })),
    setCodeNodeMappings: (mappings) => set({ codeNodeMappings: mappings }),
    setCursorPosition: (position) => set({ cursorPosition: position }),

    markSynced: (code, nodes) => set((state) => ({
      syncState: {
        ...state.syncState,
        isDirty: false,
        lastSyncedCode: code,
        lastSyncedNodes: nodes,
        conflicts: [],
      }
    })),

    addConflict: (conflict) => set((state) => ({
      syncState: {
        ...state.syncState,
        conflicts: [...state.syncState.conflicts, conflict],
      }
    })),

    clearConflicts: () => set((state) => ({
      syncState: { ...state.syncState, conflicts: [] }
    })),
  }))
);;
