/**
 * NodeCanvas - Unified ReactFlow canvas for Execute Tab and Builder Tab
 *
 * This is the SINGLE node graph component used by both tabs.
 * The tab mode determines:
 * - Which nodes/edges are visible (filtering)
 * - How nodes are rendered (mode prop)
 * - Edge styling
 *
 * Key Features:
 * - Tab-aware filtering (Execute: Config Graph, Builder: Data Flow Graph)
 * - Hierarchical navigation (Simulink-style drill-down)
 * - Bidirectional sync with code
 * - Consistent visual appearance across tabs
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Panel,
  type NodeTypes,
  type EdgeTypes,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  type OnConnectStart,
  type OnConnectEnd,
  type OnNodesDelete,
  type OnEdgesDelete,
  type Node,
  type Edge,
  type Connection,
  type IsValidConnection,
  applyNodeChanges,
  applyEdgeChanges,
  BackgroundVariant,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { cn } from '@/shared/lib/utils';
import { getVisibleEdgesFromState, getVisibleNodesFromState, useNodeStore } from '@/entities/node-system/model/nodeStore';
import { useBuilderStore } from '@/features/builder/model/builderStore';
import type { UnifiedNodeData } from '@/entities/node-system/model/types';
import { getEdgeLane, getEdgeStyle } from '@/entities/node-system/model/edge';
import { getCategoryTheme } from '@/entities/node-system/config/themes';

import { UnifiedNode } from './UnifiedNode';
import { CustomEdge } from './CustomEdge';
import { Breadcrumb } from './Breadcrumb';
import { TabSwitcher } from './TabSwitcher';

// =============================================================================
// Node Types Registration
// =============================================================================

const nodeTypes: NodeTypes = {
  unified: UnifiedNode,
};

const edgeTypes: EdgeTypes = {
  custom: CustomEdge,
};

// =============================================================================
// Props
// =============================================================================

/**
 * Editing callbacks for NodeCanvas
 * These are called by the canvas when editing operations occur.
 * The parent component can implement the actual logic.
 */
export interface EditingCallbacks {
  /** Called when a connection is started (for port highlighting) */
  onConnectStart?: OnConnectStart;
  /** Called when a connection ends (stop highlighting) */
  onConnectEnd?: OnConnectEnd;
  /** Called when nodes are about to be deleted */
  onNodesDelete?: OnNodesDelete;
  /** Called when edges are about to be deleted */
  onEdgesDelete?: OnEdgesDelete;
  /** Called to validate a connection before it's created */
  isValidConnection?: IsValidConnection;
  /** Called when something is dragged over the canvas */
  onDragOver?: (event: DragEvent<HTMLDivElement>) => void;
  /** Called when something is dropped on the canvas */
  onDrop?: (event: DragEvent<HTMLDivElement>) => void;
  /** Called when a key is pressed (for keyboard shortcuts) */
  onKeyDown?: (event: React.KeyboardEvent<HTMLDivElement>) => void;
  /** Called when a new connection is created */
  onConnectionCreate?: (connection: Connection) => void;
  /** Called when nodes are successfully deleted */
  onNodesDeleteComplete?: (nodes: Node<UnifiedNodeData>[]) => void;
  /** Called when edges are successfully deleted */
  onEdgesDeleteComplete?: (edges: Edge[]) => void;
}

export interface NodeCanvasProps extends EditingCallbacks {
  /** Custom class name */
  className?: string;
  /** Show minimap */
  showMinimap?: boolean;
  /** Show controls */
  showControls?: boolean;
  /** Show background grid */
  showBackground?: boolean;
  /** Background variant for grid */
  backgroundVariant?: BackgroundVariant;
  /** Allow editing (node creation, deletion) */
  editable?: boolean;
  /** Allow node dragging even when not editable */
  nodesDraggable?: boolean;
  /** Allow selecting elements even when not editable */
  elementsSelectable?: boolean;
  /** Show tab switcher */
  showTabSwitcher?: boolean;
  /** Show breadcrumb navigation */
  showBreadcrumb?: boolean;
  /** Custom node types (extend default unified node) */
  customNodeTypes?: NodeTypes;
  /** Custom edge types (extend default custom edge) */
  customEdgeTypes?: EdgeTypes;
  /** Additional panel content (renders after default panels) */
  children?: React.ReactNode;
  /** Enable drill-down on double click */
  drillEnabled?: boolean;
  /** Custom drill handler for hierarchy navigation */
  onDrillInto?: (node: Node<UnifiedNodeData>) => void;
  /** Called when a node is selected */
  onNodeSelect?: (node: Node<UnifiedNodeData>) => void;
  /** Called when node positions should be persisted */
  onNodesPersist?: (nodes: Node<UnifiedNodeData>[]) => void;
  initialFitTrigger?: string;
}

// =============================================================================
// Component
// =============================================================================

export const NodeCanvas = memo(function NodeCanvas({
  className,
  showMinimap = true,
  showControls = true,
  showBackground = true,
  backgroundVariant = BackgroundVariant.Lines,
  editable = false,
  nodesDraggable,
  elementsSelectable,
  showTabSwitcher = true,
  showBreadcrumb = true,
  drillEnabled = false,
  onDrillInto,
  onNodeSelect,
  onNodesPersist,
  initialFitTrigger,
  customNodeTypes,
  customEdgeTypes,
  children,
  // Editing callbacks
  onConnectStart,
  onConnectEnd,
  onNodesDelete,
  onEdgesDelete,
  onDragOver,
  onDrop,
  onKeyDown,
  onConnectionCreate,
}: NodeCanvasProps) {
  // Store state
  const typeColors = useBuilderStore((s) => s.typeColors);
  const currentTab = useNodeStore((s) => s.currentTab);
  const currentSubsystemId = useNodeStore((s) => s.currentSubsystemId);
  const nodeMode = useNodeStore((s) => s.nodeMode);
  const nodes = useNodeStore((s) => s.nodes);
  const edges = useNodeStore((s) => s.edges);
  const codeFlowEdges = useNodeStore((s) => s.codeFlowEdges);
  const setTab = useNodeStore((s) => s.setTab);
  const setNodes = useNodeStore((s) => s.setNodes);
  const setEdges = useNodeStore((s) => s.setEdges);
  const selectNode = useNodeStore((s) => s.selectNode);
  const breadcrumb = useNodeStore((s) => s.breadcrumb);
  const drillInto = useNodeStore((s) => s.drillInto);
  const navigateTo = useNodeStore((s) => s.navigateTo);

  const visibleNodes = useMemo(
    () => getVisibleNodesFromState(nodes, currentSubsystemId, nodeMode),
    [nodes, currentSubsystemId, nodeMode]
  );
  const fitViewTriggered = useRef(false);
  const reactFlowInstanceRef = useRef<{ fitView: (options?: { padding?: number; minZoom?: number; maxZoom?: number }) => void } | null>(null);
  const [isFlowReady, setIsFlowReady] = useState(false);
  const layeredNodes = useMemo(() => {
    return visibleNodes.map((node) => (node.zIndex !== undefined && node.zIndex !== 1) ? node : { ...node, zIndex: 1 });
  }, [visibleNodes]);
  const visibleEdges = useMemo(() => {
    const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
    const visibleNodesById = new Map(visibleNodes.map((node) => [node.id, node]));
    const baseEdges = getVisibleEdgesFromState(edges, codeFlowEdges, currentTab, visibleNodeIds);

    const hasSourceHandle = (node: Node<UnifiedNodeData>, handle?: string | null) => {
      if (!handle) return true;
      const normalized = handle.replace(/^source:/, '');
      if (normalized === 'out' || normalized === 'exec:out') return true;
      return node.data.outputs.some((port) => port.name === normalized || port.id === normalized);
    };

    const hasTargetHandle = (node: Node<UnifiedNodeData>, handle?: string | null) => {
      if (!handle) return true;
      const normalized = handle.replace(/^target:/, '');
      if (normalized === 'in' || normalized === 'exec:in') return true;
      return node.data.inputs.some((port) => port.name === normalized || port.id === normalized);
    };

    return baseEdges.filter((edge) => {
      const sourceNode = visibleNodesById.get(edge.source);
      const targetNode = visibleNodesById.get(edge.target);
      if (!sourceNode || !targetNode) return false;
      if (!hasSourceHandle(sourceNode, edge.sourceHandle)) return false;
      if (!hasTargetHandle(targetNode, edge.targetHandle)) return false;
      return true;
    });
  }, [edges, codeFlowEdges, currentTab, visibleNodes]);

  useEffect(() => {
    fitViewTriggered.current = false;
  }, [initialFitTrigger]);

  useEffect(() => {
    if (fitViewTriggered.current || !isFlowReady || layeredNodes.length === 0 || !reactFlowInstanceRef.current) return;
    fitViewTriggered.current = true;
    requestAnimationFrame(() => {
      reactFlowInstanceRef.current?.fitView({ padding: 0.2, minZoom: 0.5, maxZoom: 2 });
    });
  }, [layeredNodes, isFlowReady]);

  // Apply edge styles based on flow type
  const styledEdges = useMemo(() => {
    return visibleEdges.map((edge) => {
      const flowType = edge.data?.flowType;
      const edgeType = edge.data?.edgeType || getEdgeLane(flowType ? String(flowType) : undefined);
      
      // 항목 4: 연결선 색상을 입력 포트 기준으로 동기화 (백엔드 flowType 사용)
      const edgeColor = edgeType === 'execution' 
        ? '#ffffff' 
        : (typeColors[flowType] || typeColors['any'] || '#94a3b8');

      return {
        ...edge,
        type: 'custom',
        zIndex: edgeType === 'execution' ? 30 : 20,
        style: {
          stroke: edgeColor,
          strokeWidth: edgeType === 'execution' ? 3 : 2.5,
          strokeDasharray: edge.data?.is_gapped ? '6,4' : undefined,
        },
        animated: edgeType === 'data',
      };
    });
  }, [visibleEdges, typeColors]);

  const draggable = nodesDraggable ?? editable;
  const selectable = elementsSelectable ?? (editable || draggable);

  // 항목 6: 같은 타입 끼리만 연결되는 비즈니스 로직 추가
  const isValidConnection: IsValidConnection = useCallback((connection) => {
    const sourceNode = nodes.find(n => n.id === connection.source);
    const targetNode = nodes.find(n => n.id === connection.target);
    if (!sourceNode || !targetNode) return false;

    const sourcePort = sourceNode.data.outputs.find(p => p.id === connection.sourceHandle);
    const targetPort = targetNode.data.inputs.find(p => p.id === connection.targetHandle);
    
    // 실행핀은 실행핀끼리, 데이터핀은 같은 타입끼리
    if (!sourcePort || !targetPort) {
      // 실행핀 체크 (Handle ID가 in/out인 경우)
      const isSourceExec = connection.sourceHandle === 'out';
      const isTargetExec = connection.targetHandle === 'in';
      return isSourceExec && isTargetExec;
    }

    return sourcePort.type === targetPort.type || sourcePort.type === 'any' || targetPort.type === 'any';
  }, [nodes]);

  // Node change handler
  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      const currentNodes = useNodeStore.getState().nodes;
      const updated = applyNodeChanges(changes, currentNodes);
      setNodes(updated as Node<UnifiedNodeData>[]);

      // Requirement 1: Sync selection state to store
      const selectChange = changes.find(c => c.type === 'select');
      if (selectChange) {
        const id = (selectChange as any).id;
        const selected = (selectChange as any).selected;
        if (selected) selectNode(id);
        else if (useNodeStore.getState().selectedNodeId === id) selectNode(null);
      }

      // Requirement 4: Persist positions on change (when dragging stops)
      if (changes.some(c => c.type === 'position' && (c as any).dragging === false)) {
        onNodesPersist?.(updated as Node<UnifiedNodeData>[]);
      }
    },
    [setNodes, onNodesPersist, selectNode]
  );

  // Edge change handler
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      const edges = useNodeStore.getState().edges;
      setEdges(applyEdgeChanges(changes, edges));
    },
    [setEdges]
  );

  // Merge custom node/edge types with defaults
  const mergedNodeTypes = useMemo(() => ({
    ...nodeTypes,
    ...customNodeTypes,
  }), [customNodeTypes]);

  const mergedEdgeTypes = useMemo(() => ({
    ...edgeTypes,
    ...customEdgeTypes,
  }), [customEdgeTypes]);

  // Connection handler (for editable mode)
  const handleConnect: OnConnect = useCallback(
    (connection) => {
      if (!editable) return;
      // Call parent callback if provided
      if (onConnectionCreate) {
        onConnectionCreate(connection);
      }
    },
    [editable, onConnectionCreate]
  );

  // Drag over handler (for drag & drop)
  const handleDragOver = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (!editable) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
      onDragOver?.(event);
    },
    [editable, onDragOver]
  );

  // Drop handler (for drag & drop)
  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (!editable) return;
      event.preventDefault();
      onDrop?.(event);
    },
    [editable, onDrop]
  );

  // Keyboard handler
  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (!editable) return;
      onKeyDown?.(event);
    },
    [editable, onKeyDown]
  );

  // Node click handler
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node<UnifiedNodeData>) => {
      selectNode(node.id);
      onNodeSelect?.(node);
    },
    [selectNode, onNodeSelect]
  );

  // Node double-click handler (drill-down)
  const onNodeDoubleClick = useCallback(
    (event: React.MouseEvent, node: Node<UnifiedNodeData>) => {
      // Prevent default to avoid side effects
      event.preventDefault();
      event.stopPropagation();
      
      if (!drillEnabled || !node.data.canDrill) return;
      if (onDrillInto) {
        onDrillInto(node);
        return;
      }
      drillInto(node.id, node.data.role);
    },
    [drillEnabled, drillInto, onDrillInto]
  );

  // Pane click handler (deselect)
  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  // Edge click handler (Req 4: Select edge for Inspector)
  const onEdgeClick = useCallback((_: React.MouseEvent, edge: Edge) => {
    const setSelectedEdgeId = useBuilderStore((s) => s.setSelectedEdgeId);
    setSelectedEdgeId(edge.id);
  }, []);

  // MiniMap node color
  const minimapNodeColor = useCallback((node: Node<UnifiedNodeData>) => {
    const theme = getCategoryTheme(node.data.category);
    // Extract color from theme (simplified)
    if (theme.icon.includes('blue')) return '#3b82f6';
    if (theme.icon.includes('green')) return '#22c55e';
    if (theme.icon.includes('red')) return '#ef4444';
    if (theme.icon.includes('purple')) return '#a855f7';
    return '#6b7280';
  }, []);

  return (
    <div
      className={cn(
        'w-full h-full rounded-lg border border-border/60 bg-gradient-to-br from-muted/30 via-background to-muted/10 shadow-sm',
        className
      )}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onKeyDown={handleKeyDown}
      tabIndex={editable ? 0 : undefined}
    >
      <ReactFlow
        nodes={layeredNodes}
        edges={styledEdges}
        nodeTypes={mergedNodeTypes}
        edgeTypes={mergedEdgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={handleConnect}
        onConnectStart={editable ? onConnectStart : undefined}
        onConnectEnd={editable ? onConnectEnd : undefined}
        onNodesDelete={editable ? onNodesDelete : undefined}
        onEdgesDelete={editable ? onEdgesDelete : undefined}
        isValidConnection={editable ? isValidConnection : undefined}
        onNodeClick={onNodeClick}
        onNodeDoubleClick={onNodeDoubleClick}
        onEdgeClick={onEdgeClick}
        onPaneClick={onPaneClick}
        onInit={(instance) => {
          reactFlowInstanceRef.current = instance;
          setIsFlowReady(true);
        }}
        
        // fitView removed to prevent auto-reset on update
        fitViewOptions={{ padding: 0.2, minZoom: 0.5, maxZoom: 2 }}
        minZoom={0.5}
        maxZoom={2}
        snapToGrid
        snapGrid={[20, 20]}
        deleteKeyCode={editable ? 'Delete' : null}
        selectionKeyCode={editable ? 'Shift' : null}
        nodesDraggable={draggable}
        nodesConnectable={editable}
        elementsSelectable={selectable}
        proOptions={{ hideAttribution: true }}
      >
        {/* Top-right: Tab Switcher - Removed internal panel for global header integration */}

        {/* Background */}
        {showBackground && (
          <Background
            variant={BackgroundVariant.Dots}
            gap={20}
            size={1}
            color="hsl(var(--foreground)/0.1)"
            className="bg-transparent"
          />
        )}
        
        {/* Global Edge Marker Definition */}
        <svg style={{ position: 'absolute', top: 0, left: 0, width: 0, height: 0 }}>
          <defs>
            <marker
              id="flow-marker"
              viewBox="0 0 10 10"
              refX="9" // Arrow tip position
              refY="5"
              markerWidth="4" // Reduced size
              markerHeight="4" // Reduced size
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" />
            </marker>
          </defs>
        </svg>

        {/* Controls - Styled for IDE */}
        {showControls && (
          <Controls
            showInteractive={editable}
            className="!bg-background !border !border-border/40 !rounded-md !shadow-xl m-4 !overflow-hidden !flex !flex-col !gap-0"
            showFitView={true}
            showZoom={true}
          />
        )}

        {/* MiniMap */}
        {showMinimap && (
          <MiniMap
            nodeColor={minimapNodeColor}
            maskColor="hsl(var(--foreground) / 0.04)"
            className="bg-background border rounded-md shadow-sm m-2"
            pannable
            zoomable
            style={{ width: 120, height: 80 }}
          />
        )}

        {/* Custom children (additional panels, etc.) */}
        {children}
      </ReactFlow>
    </div>
  );
});

NodeCanvas.displayName = 'NodeCanvas';
