/**
 * useNodeSync - Nodes → Code Synchronization Hook
 *
 * Manages synchronization from node graph changes to code generation.
 * Features:
 * - Immediate code generation on node edits
 * - LibCST-based formatting preservation
 * - Conflict detection
 * - Code editor update
 */

import { useCallback } from 'react';
import { useBuilderStore } from '@/features/builder/model/builderStore';
import type {
  GenerateCodeFromNodesRequest,
  GenerateCodeFromNodesResponse,
} from '@/shared/model/ast-blocks';

interface UseNodeSyncOptions {
  agentName: string;
  originalCode?: string;
  enabled?: boolean;
}

export function useNodeSync({
  agentName,
  originalCode,
  enabled = true,
}: UseNodeSyncOptions) {
  const store = useBuilderStore();

  /**
   * Generate code from current node graph
   */
  const syncNodesToCode = useCallback(
    async (nodeGraph?: Record<string, unknown>) => {
      if (!enabled || !agentName) return null;

      const lastSyncedNodes = store.syncState.lastSyncedNodes as {
        metadata?: Record<string, unknown>;
      } | null;

      // Use provided nodeGraph or build from current store state
      const graph = nodeGraph || {
        nodes: store.nodes.map(n => ({
          id: n.id,
          type: n.type,
          data: n.data,
          position: n.position,
        })),
        edges: store.edges.map(e => ({
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: e.sourceHandle,
          targetHandle: e.targetHandle,
          data: e.data,
        })),
        metadata: lastSyncedNodes?.metadata ?? {},
      };

      // Set syncing state
      store.updateSyncState({ isSyncing: true, syncError: null });

      try {
        const requestBody: GenerateCodeFromNodesRequest = {
          agent_name: agentName,
          node_graph: graph,
          original_code: originalCode || store.syncState.lastSyncedCode || null,
          // preserve_sections: Line ranges [[start, end], ...] to preserve from original_code
          // Future: Track user-edited sections via diff detection between auto-generated
          // and manually edited code. Requires editor integration to detect manual edits.
          preserve_sections: [],
        };

        const response = await fetch('/api/nodes/generate-from-nodes', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Code generation failed: ${response.statusText}`);
        }

        const result: GenerateCodeFromNodesResponse = await response.json();

        if (!result.success || !result.code) {
          throw new Error('Code generation failed');
        }

        // Clear syncing state
        store.updateSyncState({ isSyncing: false });

        // Mark as synced
        store.markSynced(result.code, graph);

        return {
          code: result.code,
          modifiedLines: result.modified_lines || [],
        };
      } catch (error) {
        if (error instanceof Error) {
          console.error('Node sync error:', error);
          store.updateSyncState({
            isSyncing: false,
            syncError: error.message,
          });
        }
        return null;
      }
    },
    [agentName, originalCode, enabled, store]
  );

  /**
   * Handle node port update
   * Immediately triggers code generation
   */
  const handleNodePortUpdate = useCallback(
    async (nodeId: string, portId: string, newValue: unknown) => {
      // Update node in store
      store.updateNodes((nodes) =>
        nodes.map((node) => {
          if (node.id === nodeId) {
            return {
              ...node,
              data: {
                ...node.data,
                ports: node.data?.ports?.map((port: any) =>
                  port.id === portId ? { ...port, value: newValue } : port
                ),
              },
            };
          }
          return node;
        })
      );

      // Trigger code generation
      return syncNodesToCode();
    },
    [store, syncNodesToCode]
  );

  /**
   * Handle node delete
   * Immediately triggers code generation
   */
  const handleNodeDelete = useCallback(
    async (nodeId: string) => {
      // Remove node and connected edges from store
      store.updateNodes((nodes) => nodes.filter((n) => n.id !== nodeId));
      store.updateEdges((edges) =>
        edges.filter((e) => e.source !== nodeId && e.target !== nodeId)
      );

      // Trigger code generation
      return syncNodesToCode();
    },
    [store, syncNodesToCode]
  );

  /**
   * Handle edge creation
   * Immediately triggers code generation
   */
  const handleEdgeCreate = useCallback(
    async (source: string, target: string, sourceHandle?: string, targetHandle?: string) => {
      // Clean up port prefixes if present
      const sourcePortRaw = sourceHandle?.replace(/^source:/, '') || 'out';
      const targetPortRaw = targetHandle?.replace(/^target:/, '') || 'in';
      
      // Use canonical backend edge ID format
      const edgeId = `${source}_${sourcePortRaw}_to_${target}_${targetPortRaw}`;

      // Add edge to store
      const newEdge = {
        id: edgeId,
        source,
        target,
        sourceHandle: sourceHandle || null,
        targetHandle: targetHandle || null,
      };

      store.updateEdges((edges) => [...edges, newEdge]);

      // Trigger code generation
      return syncNodesToCode();
    },
    [store, syncNodesToCode]
  );

  /**
   * Handle property/config update
   * Updates YAML config file via backend API
   *
   * @param nodeId - Node ID (role name like "model", "optimizer")
   * @param propertyName - Property name (e.g., "lr", "num_classes")
   * @param newValue - New value for the property
   * @param category - Optional category for nested YAML keys (e.g., "optimizer")
   * @param yamlPath - Optional custom YAML file path
   */
  const handlePropertyUpdate = useCallback(
    async (
      nodeId: string,
      propertyName: string,
      newValue: unknown,
      category?: string,
      yamlPath?: string
    ): Promise<{ success: boolean; error?: string }> => {
      if (!enabled || !agentName) {
        return { success: false, error: 'Sync not enabled' };
      }

      store.updateSyncState({ isSyncing: true, syncError: null });

      try {
        const response = await fetch(
          `/api/nodes/hierarchy/${agentName}/nodes/${nodeId}`,
          {
            method: 'PUT',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              config: { [propertyName]: newValue },
              category,
              yaml_path: yamlPath,
            }),
          }
        );

        if (!response.ok) {
          throw new Error(`Property update failed: ${response.statusText}`);
        }

        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || 'Property update failed');
        }

        // Update local store
        store.updateNodes((nodes) =>
          nodes.map((node) => {
            if (node.id === nodeId && node.data?.properties) {
              return {
                ...node,
                data: {
                  ...node.data,
                  properties: node.data.properties.map((prop: any) =>
                    prop.name === propertyName
                      ? { ...prop, value: newValue }
                      : prop
                  ),
                },
              };
            }
            return node;
          })
        );

        store.updateSyncState({ isSyncing: false });
        return { success: true };
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        console.error('Property update error:', error);
        store.updateSyncState({
          isSyncing: false,
          syncError: message,
        });
        return { success: false, error: message };
      }
    },
    [agentName, enabled, store]
  );

  return {
    syncNodesToCode,
    handleNodePortUpdate,
    handleNodeDelete,
    handleEdgeCreate,
    handlePropertyUpdate,
    isSyncing: store.syncState.isSyncing,
    syncError: store.syncState.syncError,
  };
}
