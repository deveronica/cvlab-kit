/**
 * useCodeSync - Bidirectional Code ↔ Nodes Synchronization Hook
 *
 * Manages real-time synchronization between code edits and node graph.
 * Features:
 * - Code → Nodes: Parse code and update node graph
 * - Nodes → Code: Generate code from node graph with merge support
 * - 500ms debounced API calls
 * - Conflict detection
 * - User code preservation (with smart merge)
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useBuilderStore } from '@/features/builder/model/builderStore';
import { useInvalidateHierarchyCache } from '@/shared/model/useHierarchyGraph';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import type {
  SyncCodeToNodesRequest,
  SyncCodeToNodesResponse,
  GenerateCodeFromNodesRequest,
  GenerateCodeFromNodesResponse,
  CodeNodeMapping,
  MergeMode,
  PreservedRegion,
} from '@/shared/model/ast-blocks';

const DEBOUNCE_MS = 500;

interface UseCodeSyncOptions {
  agentName: string;
  enabled?: boolean;
  defaultMergeMode?: MergeMode;
  /** Draft ID for draft mode sync (Code → Nodes) */
  draftId?: string | null;
  /** Callback when code-to-node sync detects changes */
  onSyncComplete?: (result: DraftSyncResult) => void;
}

/** Result from Code → Node draft sync */
export interface DraftSyncResult {
  success: boolean;
  added: string[];
  removed: string[];
  updated: string[];
  errors: string[];
  draft_status?: string;
}

export function useCodeSync({
  agentName,
  enabled = true,
  defaultMergeMode = 'smart',
  draftId = null,
  onSyncComplete,
}: UseCodeSyncOptions) {
  const store = useBuilderStore();
  const invalidateHierarchyCache = useInvalidateHierarchyCache();
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const nodesSyncAbortRef = useRef<AbortController | null>(null);

  // Additional state for Nodes → Code sync
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [preservedRegions, setPreservedRegions] = useState<PreservedRegion[]>([]);
  const [mergeMode, setMergeMode] = useState<MergeMode>(defaultMergeMode);
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);

  /**
   * Sync code to nodes with debouncing
   */
  const syncCodeToNodes = useCallback(
    async (code: string) => {
      if (!enabled || !agentName) return;

      // Cancel previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Create new abort controller
      abortControllerRef.current = new AbortController();

      // Set syncing state
      store.updateSyncState({ isSyncing: true, syncError: null });

      try {
        const requestBody: SyncCodeToNodesRequest = {
          agent_name: agentName,
          code_content: code,
          last_synced_code: store.syncState.lastSyncedCode || null,
          last_synced_nodes: store.syncState.lastSyncedNodes || null,
        };

        const response = await fetch('/api/nodes/sync', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`Sync failed: ${response.statusText}`);
        }

        const result: SyncCodeToNodesResponse = await response.json();

        if (!result.success) {
          throw new Error(result.error || 'Sync failed');
        }

        // Update code blocks
        if (result.code_blocks) {
          store.setCodeBlocks(result.code_blocks);

          // Build code-node mappings
          const mappings: CodeNodeMapping[] = result.code_blocks.map((block) => ({
            blockId: block.id,
            block,
            nodeIds: block.node_ids,
            startLine: block.start_line,
            endLine: block.end_line,
          }));
          store.setCodeNodeMappings(mappings);
        }

        store.setUncoveredLines(result.uncovered_lines || []);

        // Invalidate hierarchy cache so node-system views refetch updated graph
        // This triggers react-query to refetch, keeping node graph views in sync
        if (result.node_graph) {
          invalidateHierarchyCache();
        }

        // Handle conflicts
        if (result.conflicts && result.conflicts.length > 0) {
          result.conflicts.forEach((conflict) => {
            store.addConflict(conflict);
          });
        } else {
          // No conflicts, mark as synced
          store.markSynced(code, result.node_graph || {});
        }

        // Clear syncing state
        store.updateSyncState({ isSyncing: false });
      } catch (error) {
        if (error instanceof Error) {
          if (error.name === 'AbortError') {
            // Request was cancelled, ignore
            return;
          }

          console.error('Code sync error:', error);
          store.updateSyncState({
            isSyncing: false,
            syncError: error.message,
          });
          store.setUncoveredLines([]);
        }
      }
    },
    [agentName, enabled, store, invalidateHierarchyCache]
  );

  /**
   * Sync code to nodes in DRAFT mode
   * Uses the draft-specific endpoint for Code → Node sync
   */
  const syncCodeToNodesDraft = useCallback(
    async (code: string): Promise<DraftSyncResult | null> => {
      if (!enabled || !agentName || !draftId) {
        console.warn('syncCodeToNodesDraft: Missing agentName or draftId');
        return null;
      }

      // Cancel previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      abortControllerRef.current = new AbortController();
      store.updateSyncState({ isSyncing: true, syncError: null });

      try {
        const response = await fetch(
          `/api/nodes/hierarchy/${encodeURIComponent(agentName)}/draft/${encodeURIComponent(draftId)}/sync-from-code`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ code_content: code }),
            signal: abortControllerRef.current.signal,
          }
        );

        if (!response.ok) {
          throw new Error(`Draft sync failed: ${response.statusText}`);
        }

        const result: DraftSyncResult = await response.json();

        if (!result.success) {
          throw new Error(result.errors?.join(', ') || 'Draft sync failed');
        }

        // Update sync state
        store.updateSyncState({ isSyncing: false });

        // Invalidate hierarchy cache if changes were detected
        if (result.added.length > 0 || result.removed.length > 0 || result.updated.length > 0) {
          invalidateHierarchyCache();
        }

        // Notify callback
        onSyncComplete?.(result);

        return result;
      } catch (error) {
        if (error instanceof Error) {
          if (error.name === 'AbortError') {
            return null;
          }

          console.error('Draft code sync error:', error);
          store.updateSyncState({
            isSyncing: false,
            syncError: error.message,
          });
        }
        return null;
      }
    },
    [agentName, draftId, enabled, store, invalidateHierarchyCache, onSyncComplete]
  );

  /**
   * Debounced sync function - uses draft mode if draftId is available
   */
  const debouncedSync = useCallback(
    (code: string) => {
      // Clear existing timer
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }

      // Set new timer
      debounceTimerRef.current = setTimeout(() => {
        // Use draft sync if draftId is available, otherwise use legacy sync
        if (draftId) {
          syncCodeToNodesDraft(code);
        } else {
          syncCodeToNodes(code);
        }
      }, DEBOUNCE_MS);
    },
    [draftId, syncCodeToNodes, syncCodeToNodesDraft]
  );

  /**
   * Find and select node at cursor position
   */
  const selectNodeAtCursor = useCallback(
    (line: number) => {
      const blocksWithNodes = store.codeBlocks.filter((block) => block.node_ids.length > 0);
      if (blocksWithNodes.length === 0) {
        return;
      }

      const getLineDistance = (startLine: number, endLine: number): number => {
        if (line < startLine) {
          return startLine - line;
        }
        if (line > endLine) {
          return line - endLine;
        }
        return 0;
      };

      const getBlockSpan = (startLine: number, endLine: number): number => {
        return Math.max(0, endLine - startLine);
      };

      const getBlockDistance = (startLine: number, endLine: number): number => {
        return getLineDistance(startLine, endLine);
      };

      const containingBlocks = blocksWithNodes.filter(
        (block) => block.start_line <= line && line <= block.end_line
      );

      const targetBlock =
        containingBlocks.length > 0
          ? [...containingBlocks].sort((a, b) => {
              const spanDiff = getBlockSpan(a.start_line, a.end_line) - getBlockSpan(b.start_line, b.end_line);
              if (spanDiff !== 0) {
                return spanDiff;
              }
              return b.start_line - a.start_line;
            })[0]
          : [...blocksWithNodes].sort((a, b) => {
              const distanceDiff =
                getBlockDistance(a.start_line, a.end_line) - getBlockDistance(b.start_line, b.end_line);
              if (distanceDiff !== 0) {
                return distanceDiff;
              }

              const spanDiff = getBlockSpan(a.start_line, a.end_line) - getBlockSpan(b.start_line, b.end_line);
              if (spanDiff !== 0) {
                return spanDiff;
              }

              return a.start_line - b.start_line;
            })[0];

      if (!targetBlock || targetBlock.node_ids.length === 0) {
        return;
      }

      const nodeLineRanges = new Map<string, { startLine: number; endLine: number }>();

      const lastSyncedNodes = store.syncState.lastSyncedNodes;
      if (lastSyncedNodes && typeof lastSyncedNodes === 'object') {
        const candidateNodes = (lastSyncedNodes as { nodes?: unknown }).nodes;
        if (Array.isArray(candidateNodes)) {
          for (const rawNode of candidateNodes) {
            if (!rawNode || typeof rawNode !== 'object') {
              continue;
            }

            const node = rawNode as Record<string, unknown>;
            const nodeId = typeof node.id === 'string' ? node.id : null;
            if (!nodeId) {
              continue;
            }

            let startLine: number | null = null;
            let endLine: number | null = null;

            const source = node.source;
            if (source && typeof source === 'object') {
              const sourceData = source as Record<string, unknown>;
              const sourceStart = sourceData.line_start;
              const sourceEnd = sourceData.line_end;
              const sourceLine = sourceData.line;
              const sourceEndLine = sourceData.end_line;
              const sourceEndLineCamel = sourceData.endLine;

              if (typeof sourceStart === 'number' && Number.isFinite(sourceStart)) {
                startLine = sourceStart;
              }
              if (typeof sourceEnd === 'number' && Number.isFinite(sourceEnd)) {
                endLine = sourceEnd;
              }
              if (startLine === null && typeof sourceLine === 'number' && Number.isFinite(sourceLine)) {
                startLine = sourceLine;
              }
              if (endLine === null) {
                if (typeof sourceEndLine === 'number' && Number.isFinite(sourceEndLine)) {
                  endLine = sourceEndLine;
                } else if (typeof sourceEndLineCamel === 'number' && Number.isFinite(sourceEndLineCamel)) {
                  endLine = sourceEndLineCamel;
                }
              }
            }

            const metadata = node.metadata;
            if (startLine === null && metadata && typeof metadata === 'object') {
              const metadataData = metadata as Record<string, unknown>;
              const metadataStart = metadataData.source_line_start;
              const metadataEnd = metadataData.source_line_end;

              if (typeof metadataStart === 'number' && Number.isFinite(metadataStart)) {
                startLine = metadataStart;
              }
              if (typeof metadataEnd === 'number' && Number.isFinite(metadataEnd)) {
                endLine = metadataEnd;
              }
            }

            if (startLine !== null) {
              nodeLineRanges.set(nodeId, {
                startLine,
                endLine: endLine ?? startLine,
              });
            }
          }
        }
      }

      if (nodeLineRanges.size === 0) {
        const activeNodes = useNodeStore.getState().nodes;
        for (const node of activeNodes) {
          const source = (node.data as {
            source?: { line_start?: number; line_end?: number; line?: number; end_line?: number; endLine?: number };
          }).source;
          if (!source) {
            continue;
          }

          const start = typeof source.line_start === 'number'
            ? source.line_start
            : (typeof source.line === 'number' ? source.line : null);

          if (start === null) {
            continue;
          }

          const end = typeof source.line_end === 'number'
            ? source.line_end
            : (typeof source.end_line === 'number'
              ? source.end_line
              : (typeof source.endLine === 'number' ? source.endLine : start));

          nodeLineRanges.set(node.id, {
            startLine: start,
            endLine: end,
          });
        }
      }

      const blockDistanceByNode = new Map<string, number>();
      for (const block of blocksWithNodes) {
        const blockDistance = getBlockDistance(block.start_line, block.end_line);
        for (const nodeId of block.node_ids) {
          const previousDistance = blockDistanceByNode.get(nodeId);
          if (previousDistance === undefined || blockDistance < previousDistance) {
            blockDistanceByNode.set(nodeId, blockDistance);
          }
        }
      }

      const candidateNodeIds = [...new Set(targetBlock.node_ids)];
      const selectedNodeId = candidateNodeIds.reduce<string | null>((bestNodeId, nodeId) => {
        const nodeRange = nodeLineRanges.get(nodeId);
        const nodeDistance = nodeRange
          ? getLineDistance(nodeRange.startLine, nodeRange.endLine)
          : (blockDistanceByNode.get(nodeId) ?? Number.MAX_SAFE_INTEGER);

        if (!bestNodeId) {
          return nodeId;
        }

        const bestRange = nodeLineRanges.get(bestNodeId);
        const bestDistance = bestRange
          ? getLineDistance(bestRange.startLine, bestRange.endLine)
          : (blockDistanceByNode.get(bestNodeId) ?? Number.MAX_SAFE_INTEGER);

        if (nodeDistance < bestDistance) {
          return nodeId;
        }

        if (nodeDistance === bestDistance) {
          const nodeSpan = nodeRange
            ? getBlockSpan(nodeRange.startLine, nodeRange.endLine)
            : Number.MAX_SAFE_INTEGER;
          const bestSpan = bestRange
            ? getBlockSpan(bestRange.startLine, bestRange.endLine)
            : Number.MAX_SAFE_INTEGER;

          if (nodeSpan < bestSpan) {
            return nodeId;
          }

          if (nodeSpan === bestSpan && nodeId < bestNodeId) {
            return nodeId;
          }
        }

        return bestNodeId;
      }, null);

      if (selectedNodeId) {
        useNodeStore.getState().selectNode(selectedNodeId);
        store.setSelectedNodeId(selectedNodeId);
      }
    },
    [store]
  );

  // =========================================================================
  // Nodes → Code Synchronization
  // =========================================================================

  /**
   * Generate code from node graph
   */
  const syncNodesToCode = useCallback(
    async (
      nodeGraph: Record<string, unknown>,
      originalCode?: string | null,
      mode?: MergeMode
    ): Promise<string | null> => {
      if (!enabled || !agentName) return null;

      // Cancel previous request
      if (nodesSyncAbortRef.current) {
        nodesSyncAbortRef.current.abort();
      }

      nodesSyncAbortRef.current = new AbortController();
      setIsGenerating(true);
      setGenerateError(null);

      try {
        const requestBody: GenerateCodeFromNodesRequest = {
          agent_name: agentName,
          node_graph: nodeGraph,
          original_code: originalCode ?? store.syncState.lastSyncedCode ?? null,
          merge_mode: mode ?? mergeMode,
        };

        const response = await fetch('/api/nodes/generate-from-nodes', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
          signal: nodesSyncAbortRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`Generation failed: ${response.statusText}`);
        }

        const result: GenerateCodeFromNodesResponse = await response.json();

        if (!result.success) {
          throw new Error('Code generation failed');
        }

        // Store the generated code
        if (result.code) {
          setGeneratedCode(result.code);
        }

        // Store preserved regions for UI highlighting
        if (result.preserved_regions) {
          setPreservedRegions(result.preserved_regions);
        }

        // Log warnings
        if (result.warnings && result.warnings.length > 0) {
          console.warn('Code generation warnings:', result.warnings);
        }

        setIsGenerating(false);
        return result.code ?? null;
      } catch (error) {
        if (error instanceof Error) {
          if (error.name === 'AbortError') {
            return null;
          }

          console.error('Node to code sync error:', error);
          setGenerateError(error.message);
        }
        setIsGenerating(false);
        return null;
      }
    },
    [agentName, enabled, mergeMode, store.syncState.lastSyncedCode]
  );

  /**
   * Debounced nodes → code sync
   */
  const debouncedSyncNodesToCode = useCallback(
    (nodeGraph: Record<string, unknown>, originalCode?: string | null) => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }

      debounceTimerRef.current = setTimeout(() => {
        syncNodesToCode(nodeGraph, originalCode);
      }, DEBOUNCE_MS);
    },
    [syncNodesToCode]
  );

  /**
   * Switch merge mode
   */
  const changeMergeMode = useCallback((newMode: MergeMode) => {
    setMergeMode(newMode);
    // Clear preserved regions when switching modes
    setPreservedRegions([]);
  }, []);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (nodesSyncAbortRef.current) {
        nodesSyncAbortRef.current.abort();
      }
    };
  }, []);

  return {
    // Code → Nodes
    syncCodeToNodes,
    syncCodeToNodesDraft,
    debouncedSync,
    selectNodeAtCursor,
    isSyncing: store.syncState.isSyncing,
    syncError: store.syncState.syncError,
    conflicts: store.syncState.conflicts,

    // Nodes → Code
    syncNodesToCode,
    debouncedSyncNodesToCode,
    isGenerating,
    generateError,
    generatedCode,

    // Merge mode management
    mergeMode,
    changeMergeMode,
    preservedRegions,

    // Draft mode info
    isDraftMode: !!draftId,
  };
}
