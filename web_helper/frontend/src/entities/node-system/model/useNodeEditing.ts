/**
 * useNodeEditing - Bridge hook for editing functionality
 *
 * This hook bridges the existing AgentBuilderContext with the new NodeCanvas.
 * It provides all the editing callbacks needed by NodeCanvas.
 *
 * Usage:
 * ```tsx
 * const { editable, editingCallbacks } = useNodeEditing();
 *
 * <NodeCanvas
 *   editable={editable}
 *   {...editingCallbacks}
 * />
 * ```
 */

import { useCallback, useMemo, useRef } from 'react';
import type {
  OnConnectStart,
  OnConnectEnd,
  OnNodesDelete,
  OnEdgesDelete,
  IsValidConnection,
  Connection,
} from 'reactflow';

import type { ComponentCategory } from '@/entities/node-system/model/port';
import { useNodeStore } from '@/entities/node-system/model/port';
import type { EditingCallbacks } from '../NodeCanvas';

// =============================================================================
// Types
// =============================================================================

/**
 * Port info for connection validation
 */
interface PortInfo {
  nodeId: string;
  portId: string;
  portType: string;
  handleType: 'source' | 'target';
}

/**
 * Drag data for component drop
 */
export interface ComponentDragData {
  type: 'component';
  category: ComponentCategory;
  name: string;
  implementation?: string;
}

/**
 * Options for useNodeEditing hook
 */
interface UseNodeEditingOptions {
  /** Whether editing is currently enabled */
  isEditingMode?: boolean;
  /** Called when a node is added via drag & drop */
  onNodeAdd?: (
    category: ComponentCategory,
    name: string,
    position: { x: number; y: number },
    implementation?: string
  ) => void;
  /** Called when nodes are deleted */
  onNodesRemove?: (nodeIds: string[]) => void;
  /** Called when edges are deleted */
  onEdgesRemove?: (edgeIds: string[]) => void;
  /** Called when a connection is created */
  onConnectionCreate?: (connection: Connection) => void;
  /** Called to sync a node click to code */
  onNodeSyncToCode?: (nodeId: string) => void;
  /** Port type compatibility checker */
  checkPortCompatibility?: (
    sourceType: string,
    targetType: string
  ) => boolean;
}

/**
 * Return type for useNodeEditing hook
 */
interface UseNodeEditingResult {
  /** Whether editing is enabled */
  editable: boolean;
  /** All editing callbacks for NodeCanvas */
  editingCallbacks: EditingCallbacks;
  /** Currently dragging port info (for highlighting) */
  draggingPort: PortInfo | null;
  /** Start dragging from a port */
  startPortDrag: (port: PortInfo) => void;
  /** End port drag */
  endPortDrag: () => void;
}

// =============================================================================
// Hook Implementation
// =============================================================================

export function useNodeEditing(
  options: UseNodeEditingOptions = {}
): UseNodeEditingResult {
  const {
    isEditingMode = false,
    onNodeAdd,
    onNodesRemove,
    onEdgesRemove,
    onConnectionCreate,
    checkPortCompatibility,
  } = options;

  // Node store state
  const setNodes = useNodeStore((s) => s.setNodes);
  const setEdges = useNodeStore((s) => s.setEdges);
  const nodes = useNodeStore((s) => s.nodes);
  const edges = useNodeStore((s) => s.edges);

  // Local state for dragging using ref to persist across renders
  const currentDraggingPortRef = useRef<PortInfo | null>(null);

  // ==========================================================================
  // Connection handlers
  // ==========================================================================

  const handleConnectStart: OnConnectStart = useCallback(
    (_event, { nodeId, handleId, handleType }) => {
      if (!nodeId || !handleId) return;

      // Find the node to get port info
      const node = nodes.find((n) => n.id === nodeId);
      if (!node) return;

      // Determine port type from node data
      let portType = 'ANY';

      if (handleType === 'source') {
        const outputs = node.data.outputs || [];
        const port = outputs.find((p) => p.id === handleId || p.name === handleId);
        if (port) {
          portType = String(port.type);
        } else if (handleId === 'self') {
          portType = 'MODULE';
        }
      } else {
        const inputs = node.data.inputs || [];
        const port = inputs.find((p) => p.id === handleId || p.name === handleId);
        if (port) {
          portType = String(port.type);
        }
      }

      currentDraggingPortRef.current = {
        nodeId,
        portId: handleId,
        portType,
        handleType: handleType as 'source' | 'target',
      };
    },
    [nodes]
  );

  const handleConnectEnd: OnConnectEnd = useCallback(() => {
    currentDraggingPortRef.current = null;
  }, []);

  // ==========================================================================
  // Connection validation
  // ==========================================================================

  const isValidConnection: IsValidConnection = useCallback(
    (connection) => {
      const { source, target, sourceHandle, targetHandle } = connection;

      // Prevent self-connection
      if (source === target) return false;

      // Find nodes
      const sourceNode = nodes.find((n) => n.id === source);
      const targetNode = nodes.find((n) => n.id === target);
      if (!sourceNode || !targetNode) return false;

      // Find ports
      const sourceOutputs = sourceNode.data.outputs || [];
      const targetInputs = targetNode.data.inputs || [];

      const sourcePort = sourceOutputs.find(
        (p) => p.id === sourceHandle || p.name === sourceHandle
      );
      const targetPort = targetInputs.find(
        (p) => p.id === targetHandle || p.name === targetHandle
      );

      if (!sourcePort || !targetPort) {
        // Allow connection if handles exist but ports not found (legacy support)
        return true;
      }

      // Check port type compatibility
      if (checkPortCompatibility) {
        return checkPortCompatibility(
          String(sourcePort.type),
          String(targetPort.type)
        );
      }

      // Default: allow all connections
      return true;
    },
    [nodes, checkPortCompatibility]
  );

  // ==========================================================================
  // Deletion handlers
  // ==========================================================================

  const handleNodesDelete: OnNodesDelete = useCallback(
    (deletedNodes) => {
      if (!isEditingMode) return;

      const nodeIds = deletedNodes.map((n) => n.id);

      // Remove from store
      const remainingNodes = nodes.filter((n) => !nodeIds.includes(n.id));
      setNodes(remainingNodes);

      // Also remove connected edges
      const remainingEdges = edges.filter(
        (e) => !nodeIds.includes(e.source) && !nodeIds.includes(e.target)
      );
      setEdges(remainingEdges);

      // Notify parent
      onNodesRemove?.(nodeIds);
    },
    [isEditingMode, nodes, edges, setNodes, setEdges, onNodesRemove]
  );

  const handleEdgesDelete: OnEdgesDelete = useCallback(
    (deletedEdges) => {
      if (!isEditingMode) return;

      const edgeIds = deletedEdges.map((e) => e.id);

      // Remove from store
      const remainingEdges = edges.filter((e) => !edgeIds.includes(e.id));
      setEdges(remainingEdges);

      // Notify parent
      onEdgesRemove?.(edgeIds);
    },
    [isEditingMode, edges, setEdges, onEdgesRemove]
  );

  // ==========================================================================
  // Connection creation
  // ==========================================================================

  const handleConnectionCreate = useCallback(
    (connection: Connection) => {
      if (!isEditingMode) return;
      onConnectionCreate?.(connection);
    },
    [isEditingMode, onConnectionCreate]
  );

  // ==========================================================================
  // Drag & Drop handlers
  // ==========================================================================

  const handleDragOver = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      if (!isEditingMode) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
    },
    [isEditingMode]
  );

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      if (!isEditingMode) return;
      event.preventDefault();

      // Get drag data
      const dataStr = event.dataTransfer.getData('application/json');
      if (!dataStr) return;

      try {
        const data = JSON.parse(dataStr) as ComponentDragData;
        if (data.type !== 'component') return;

        // Get drop position relative to canvas
        const rect = event.currentTarget.getBoundingClientRect();
        const position = {
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
        };

        // Notify parent
        onNodeAdd?.(data.category, data.name, position, data.implementation);
      } catch {
        console.error('Invalid drag data');
      }
    },
    [isEditingMode, onNodeAdd]
  );

  // ==========================================================================
  // Keyboard handlers
  // ==========================================================================

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (!isEditingMode) return;

      // Delete key is handled by ReactFlow's deleteKeyCode prop
      // Add other shortcuts here if needed
      if (event.key === 'Escape') {
        // Cancel current operation
        currentDraggingPortRef.current = null;
      }
    },
    [isEditingMode]
  );

  // ==========================================================================
  // Port drag helpers
  // ==========================================================================

  const startPortDrag = useCallback((port: PortInfo) => {
    currentDraggingPortRef.current = port;
  }, []);

  const endPortDrag = useCallback(() => {
    currentDraggingPortRef.current = null;
  }, []);

  // ==========================================================================
  // Assemble callbacks
  // ==========================================================================

  const editingCallbacks: EditingCallbacks = useMemo(
    () => ({
      onConnectStart: handleConnectStart,
      onConnectEnd: handleConnectEnd,
      onNodesDelete: handleNodesDelete,
      onEdgesDelete: handleEdgesDelete,
      isValidConnection,
      onDragOver: handleDragOver,
      onDrop: handleDrop,
      onKeyDown: handleKeyDown,
      onConnectionCreate: handleConnectionCreate,
    }),
    [
      handleConnectStart,
      handleConnectEnd,
      handleNodesDelete,
      handleEdgesDelete,
      isValidConnection,
      handleDragOver,
      handleDrop,
      handleKeyDown,
      handleConnectionCreate,
    ]
  );

  return {
    editable: isEditingMode,
    editingCallbacks,
    draggingPort: currentDraggingPortRef.current,
    startPortDrag,
    endPortDrag,
  };
}
