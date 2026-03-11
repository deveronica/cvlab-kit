/**
 * useNodeSync - Hook for bidirectional Node ↔ Code synchronization
 *
 * This hook provides API integration for:
 * - Draft management (create, commit, discard)
 * - Node CRUD operations
 * - Edge CRUD operations
 * - Conflict detection and resolution
 *
 * Architecture (ADR-008):
 * ┌──────────────────────────────────────────────────────────────┐
 * │                    Draft Mode Flow                           │
 * ├──────────────────────────────────────────────────────────────┤
 * │                                                              │
 * │  [Node Edit] → draftStore → [Save] → API → [Code Modified]  │
 * │                                                              │
 * └──────────────────────────────────────────────────────────────┘
 */

import { useCallback, useRef } from 'react';
import { useNodeStore, type DraftState } from '@/entities/node-system/model/port';
import type { ComponentCategory } from '@/entities/node-system/model/port';

// =============================================================================
// Types
// =============================================================================

interface ApiResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
}

interface CreateDraftResponse {
  success: boolean;
  draft_id: string;
  agent_name: string;
  status: string;
  created_at: string;
}

interface CommitDraftResponse {
  success: boolean;
  version_id?: string;
  version_number?: number;
  code_changes?: Array<{ type: string; details: string }>;
  error?: string;
}

interface AddNodeResponse {
  success: boolean;
  node_id: string;
  role: string;
  mode: 'draft' | 'direct';
  edit?: Record<string, unknown>;
  error?: string;
}

interface DeleteNodeResponse {
  success: boolean;
  node_id: string;
  mode: 'draft' | 'direct';
  edit?: Record<string, unknown>;
  error?: string;
}

interface AddEdgeResponse {
  success: boolean;
  edge_id: string;
  mode: 'draft' | 'direct';
  code_change?: string;
  edit?: Record<string, unknown>;
  error?: string;
}

export interface UseNodeSyncOptions {
  /** Agent name (without .py) */
  agentName: string | null;
  /** Whether to use draft mode (recommended) */
  useDraftMode?: boolean;
  /** Callback when save completes */
  onSaveComplete?: () => void;
  /** Callback when error occurs */
  onError?: (error: string) => void;
}

export interface UseNodeSyncResult {
  // Draft management
  initializeDraft: () => Promise<boolean>;
  commitDraft: (description?: string) => Promise<boolean>;
  discardDraft: () => Promise<boolean>;

  // Node operations
  addNode: (
    category: ComponentCategory,
    implementation: string,
    position: { x: number; y: number },
    role?: string
  ) => Promise<AddNodeResponse | null>;
  removeNode: (nodeId: string) => Promise<boolean>;
  updateNodeImpl: (nodeId: string, implementation: string) => Promise<boolean>;

  // Edge operations
  addEdge: (
    source: string,
    target: string,
    sourcePort: string,
    targetPort: string,
    flowType?: string
  ) => Promise<AddEdgeResponse | null>;
  removeEdge: (edgeId: string) => Promise<boolean>;

  // Sync operations
  refreshGraph: () => Promise<void>;

  // State
  isSaving: boolean;
  draftId: string | null;
  isDirty: boolean;
  hasDraftChanges: boolean;
}

// =============================================================================
// API Helper
// =============================================================================

const API_BASE = '/api/nodes';

async function apiCall<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResult<T>> {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        success: false,
        error: data.detail || data.error || `HTTP ${response.status}`,
      };
    }

    return { success: true, data };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error',
    };
  }
}

// =============================================================================
// Hook Implementation
// =============================================================================

export function useNodeSync(options: UseNodeSyncOptions): UseNodeSyncResult {
  const { agentName, useDraftMode = true, onSaveComplete, onError } = options;

  // Store actions
  const initDraft = useNodeStore((s) => s.initDraft);
  const clearDraft = useNodeStore((s) => s.clearDraft);
  const setDraftStatus = useNodeStore((s) => s.setDraftStatus);
  const setDraftError = useNodeStore((s) => s.setDraftError);
  const draftAddNode = useNodeStore((s) => s.draftAddNode);
  const draftRemoveNode = useNodeStore((s) => s.draftRemoveNode);
  const draftModifyNode = useNodeStore((s) => s.draftModifyNode);
  const draftAddEdge = useNodeStore((s) => s.draftAddEdge);
  const draftRemoveEdge = useNodeStore((s) => s.draftRemoveEdge);
  const getDraftChanges = useNodeStore((s) => s.getDraftChanges);
  const hasDraftChangesCheck = useNodeStore((s) => s.hasDraftChanges);

  // Store state
  const draftState = useNodeStore((s) => s.draftState);
  const setLoading = useNodeStore((s) => s.setLoading);

  // Ref for preventing concurrent operations
  const operationInProgress = useRef(false);

  // ==========================================================================
  // Draft Management
  // ==========================================================================

  const initializeDraft = useCallback(async (): Promise<boolean> => {
    if (!agentName || !useDraftMode) return true;
    if (operationInProgress.current) return false;

    operationInProgress.current = true;

    try {
      const result = await apiCall<CreateDraftResponse>(
        `/hierarchy/${agentName}/draft`,
        { method: 'POST' }
      );

      if (!result.success || !result.data) {
        const errorMsg = result.error || 'Failed to create draft';
        setDraftError(errorMsg);
        onError?.(errorMsg);
        return false;
      }

      initDraft(result.data.draft_id);
      return true;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Failed to create draft';
      setDraftError(errorMsg);
      onError?.(errorMsg);
      return false;
    } finally {
      operationInProgress.current = false;
    }
  }, [agentName, useDraftMode, initDraft, setDraftError, onError]);

  const commitDraft = useCallback(async (description = ''): Promise<boolean> => {
    if (!agentName) return false;
    if (operationInProgress.current) return false;

    const draftId = draftState.draftId;

    if (useDraftMode && !draftId) {
      onError?.('No active draft to commit');
      return false;
    }

    operationInProgress.current = true;
    setDraftStatus('saving');

    try {
      if (useDraftMode && draftId) {
        // Commit via draft endpoint
        const result = await apiCall<CommitDraftResponse>(
          `/hierarchy/${agentName}/draft/${draftId}/commit?description=${encodeURIComponent(description)}`,
          { method: 'POST' }
        );

        if (!result.success || !result.data?.success) {
          const errorMsg = result.error || result.data?.error || 'Failed to commit draft';
          setDraftError(errorMsg);
          setDraftStatus('active');
          onError?.(errorMsg);
          return false;
        }

        setDraftStatus('committed');
        clearDraft();
        onSaveComplete?.();
        return true;
      } else {
        // Direct mode: apply changes immediately
        const changes = getDraftChanges();

        // Apply each change directly
        for (const node of changes.addedNodes) {
          await apiCall<AddNodeResponse>(
            `/hierarchy/${agentName}/nodes`,
            {
              method: 'POST',
              body: JSON.stringify({
                category: node.category,
                implementation: node.implementation,
                role: node.role,
                position: node.position,
              }),
            }
          );
        }

        for (const nodeId of changes.removedNodeIds) {
          await apiCall(`/hierarchy/${agentName}/nodes/${nodeId}`, {
            method: 'DELETE',
          });
        }

        for (const { nodeId, changes: nodeChanges } of changes.modifiedNodes) {
          await apiCall(`/hierarchy/${agentName}/nodes/${nodeId}`, {
            method: 'PUT',
            body: JSON.stringify(nodeChanges),
          });
        }

        for (const edge of changes.addedEdges) {
          await apiCall(`/hierarchy/${agentName}/edges`, {
            method: 'POST',
            body: JSON.stringify(edge),
          });
        }

        for (const edgeId of changes.removedEdgeIds) {
          await apiCall(`/hierarchy/${agentName}/${edgeId}`, {
            method: 'DELETE',
          });
        }

        clearDraft();
        onSaveComplete?.();
        return true;
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Failed to save changes';
      setDraftError(errorMsg);
      setDraftStatus('active');
      onError?.(errorMsg);
      return false;
    } finally {
      operationInProgress.current = false;
    }
  }, [
    agentName,
    useDraftMode,
    draftState.draftId,
    setDraftStatus,
    setDraftError,
    clearDraft,
    getDraftChanges,
    onSaveComplete,
    onError,
  ]);

  const discardDraft = useCallback(async (): Promise<boolean> => {
    if (!agentName) return false;
    if (operationInProgress.current) return false;

    const draftId = draftState.draftId;

    if (useDraftMode && draftId) {
      operationInProgress.current = true;
      try {
        await apiCall(`/hierarchy/${agentName}/draft/${draftId}`, {
          method: 'DELETE',
        });
      } finally {
        operationInProgress.current = false;
      }
    }

    setDraftStatus('discarded');
    clearDraft();
    return true;
  }, [agentName, useDraftMode, draftState.draftId, setDraftStatus, clearDraft]);

  // ==========================================================================
  // Node Operations
  // ==========================================================================

  const addNode = useCallback(async (
    category: ComponentCategory,
    implementation: string,
    position: { x: number; y: number },
    role?: string
  ): Promise<AddNodeResponse | null> => {
    if (!agentName) return null;

    const nodeId = `${role || category}_${Date.now()}`;
    const nodeData: DraftState['addedNodes'][0] = {
      id: nodeId,
      category,
      implementation,
      role,
      position,
    };

    if (useDraftMode) {
      // Record in draft
      draftAddNode(nodeData);

      // If draft is active, also send to backend for validation
      if (draftState.draftId) {
        const result = await apiCall<AddNodeResponse>(
          `/hierarchy/${agentName}/nodes?draft_id=${draftState.draftId}`,
          {
            method: 'POST',
            body: JSON.stringify({
              category,
              implementation,
              role,
              position,
            }),
          }
        );

        if (result.success && result.data) {
          return result.data;
        }
      }

      return {
        success: true,
        node_id: nodeId,
        role: role || category,
        mode: 'draft',
      };
    } else {
      // Direct mode
      const result = await apiCall<AddNodeResponse>(
        `/hierarchy/${agentName}/nodes`,
        {
          method: 'POST',
          body: JSON.stringify({
            category,
            implementation,
            role,
            position,
          }),
        }
      );

      if (result.success && result.data) {
        return result.data;
      }

      onError?.(result.error || 'Failed to add node');
      return null;
    }
  }, [agentName, useDraftMode, draftState.draftId, draftAddNode, onError]);

  const removeNode = useCallback(async (nodeId: string): Promise<boolean> => {
    if (!agentName) return false;

    if (useDraftMode) {
      draftRemoveNode(nodeId);

      if (draftState.draftId) {
        await apiCall(
          `/hierarchy/${agentName}/nodes/${nodeId}?draft_id=${draftState.draftId}`,
          { method: 'DELETE' }
        );
      }

      return true;
    } else {
      const result = await apiCall<DeleteNodeResponse>(
        `/hierarchy/${agentName}/nodes/${nodeId}`,
        { method: 'DELETE' }
      );

      if (!result.success) {
        onError?.(result.error || 'Failed to remove node');
        return false;
      }

      return true;
    }
  }, [agentName, useDraftMode, draftState.draftId, draftRemoveNode, onError]);

  const updateNodeImpl = useCallback(async (
    nodeId: string,
    implementation: string
  ): Promise<boolean> => {
    if (!agentName) return false;

    if (useDraftMode) {
      draftModifyNode(nodeId, { implementation });

      if (draftState.draftId) {
        await apiCall(
          `/hierarchy/${agentName}/nodes/${nodeId}?draft_id=${draftState.draftId}`,
          {
            method: 'PUT',
            body: JSON.stringify({ implementation }),
          }
        );
      }

      return true;
    } else {
      const result = await apiCall(
        `/hierarchy/${agentName}/nodes/${nodeId}`,
        {
          method: 'PUT',
          body: JSON.stringify({ implementation }),
        }
      );

      if (!result.success) {
        onError?.(result.error || 'Failed to update node');
        return false;
      }

      return true;
    }
  }, [agentName, useDraftMode, draftState.draftId, draftModifyNode, onError]);

  // ==========================================================================
  // Edge Operations
  // ==========================================================================

  const addEdge = useCallback(async (
    source: string,
    target: string,
    sourcePort: string,
    targetPort: string,
    flowType = 'reference'
  ): Promise<AddEdgeResponse | null> => {
    if (!agentName) return null;

    const edgeId = `${source}_${sourcePort}_to_${target}_${targetPort}`;
    const edgeData: DraftState['addedEdges'][0] = {
      id: edgeId,
      source,
      target,
      sourcePort,
      targetPort,
      flowType,
    };

    if (useDraftMode) {
      draftAddEdge(edgeData);

      if (draftState.draftId) {
        const result = await apiCall<AddEdgeResponse>(
          `/hierarchy/${agentName}/edges?draft_id=${draftState.draftId}`,
          {
            method: 'POST',
            body: JSON.stringify({
              source,
              target,
              source_port: sourcePort,
              target_port: targetPort,
              flow_type: flowType,
            }),
          }
        );

        if (result.success && result.data) {
          return result.data;
        }
      }

      return {
        success: true,
        edge_id: edgeId,
        mode: 'draft',
      };
    } else {
      const result = await apiCall<AddEdgeResponse>(
        `/hierarchy/${agentName}/edges`,
        {
          method: 'POST',
          body: JSON.stringify({
            source,
            target,
            source_port: sourcePort,
            target_port: targetPort,
            flow_type: flowType,
          }),
        }
      );

      if (result.success && result.data) {
        return result.data;
      }

      onError?.(result.error || 'Failed to add edge');
      return null;
    }
  }, [agentName, useDraftMode, draftState.draftId, draftAddEdge, onError]);

  const removeEdge = useCallback(async (edgeId: string): Promise<boolean> => {
    if (!agentName) return false;

    if (useDraftMode) {
      draftRemoveEdge(edgeId);

      if (draftState.draftId) {
        await apiCall(
          `/hierarchy/${agentName}/${edgeId}?draft_id=${draftState.draftId}`,
          { method: 'DELETE' }
        );
      }

      return true;
    } else {
      const result = await apiCall(
        `/hierarchy/${agentName}/${edgeId}`,
        { method: 'DELETE' }
      );

      if (!result.success) {
        onError?.(result.error || 'Failed to remove edge');
        return false;
      }

      return true;
    }
  }, [agentName, useDraftMode, draftState.draftId, draftRemoveEdge, onError]);

  // ==========================================================================
  // Sync Operations
  // ==========================================================================

  const refreshGraph = useCallback(async (): Promise<void> => {
    if (!agentName) return;

    setLoading(true);
    try {
      // This will trigger a re-fetch of the graph from the hierarchy endpoint
      // The actual fetching is handled by useHierarchyNavigation hook
      // Here we just signal that a refresh is needed
      window.dispatchEvent(
        new CustomEvent('cvlabkit:refresh-graph', { detail: { agentName } })
      );
    } finally {
      setLoading(false);
    }
  }, [agentName, setLoading]);

  return {
    // Draft management
    initializeDraft,
    commitDraft,
    discardDraft,

    // Node operations
    addNode,
    removeNode,
    updateNodeImpl,

    // Edge operations
    addEdge,
    removeEdge,

    // Sync operations
    refreshGraph,

    // State
    isSaving: draftState.status === 'saving',
    draftId: draftState.draftId,
    isDirty: draftState.isDirty,
    hasDraftChanges: hasDraftChangesCheck(),
  };
}
