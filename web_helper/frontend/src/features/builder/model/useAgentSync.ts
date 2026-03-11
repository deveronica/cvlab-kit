/**
 * useAgentSync - Code ↔ Node synchronization hook
 *
 * Features:
 * - Fetches node graph from API
 * - Builds code-node mappings
 * - Handles file content loading
 * - Syncs selection between code and nodes
 */

import { useCallback, useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { NodeGraph } from '@/shared/model/node-graph';
import { AgentFile, CodeNodeMapping } from '@/entities/node-system/model/AgentBuilderContext';

interface UseAgentSyncOptions {
  selectedFile: AgentFile | null;
  onCodeLoaded?: (code: string) => void;
  onGraphLoaded?: (graph: NodeGraph) => void;
  onMappingsBuilt?: (mappings: CodeNodeMapping[]) => void;
}

interface UseAgentSyncResult {
  code: string;
  graph: NodeGraph | null;
  mappings: CodeNodeMapping[];
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

// Fetch file content
async function fetchFileContent(file: AgentFile): Promise<string> {
  let endpoint: string;

  if (file.type === 'agent') {
    // Agent source: /api/ui/source
    const agentName = file.name.replace('.py', '');
    endpoint = `/api/ui/source`;

    const response = await fetch(endpoint);
    if (!response.ok) throw new Error('Failed to fetch agent source');

    // Returns raw source code string
    return await response.text();
  } else {
    // Config: /api/config?path={path}
    endpoint = `/api/config?path=${encodeURIComponent(file.path)}`;

    const response = await fetch(endpoint);
    if (!response.ok) throw new Error('Failed to fetch config content');

    const data = await response.json();
    return data.content || '';
  }
}

// Fetch node graph
async function fetchNodeGraph(file: AgentFile): Promise<NodeGraph | null> {
  let endpoint: string;

  if (file.type === 'agent') {
    // Agent graph: /api/nodes/agent/{name}
    const agentName = file.name.replace('.py', '');
    endpoint = `/api/nodes/agent/${encodeURIComponent(agentName)}`;
  } else {
    // Config graph: /api/nodes/config/{path}
    endpoint = `/api/nodes/config/${encodeURIComponent(file.path)}`;
  }

  const response = await fetch(endpoint);
  if (!response.ok) return null;

  const data = await response.json();
  return data.success ? data.data : null;
}

// Build code-node mappings from graph source location
function buildMappings(graph: NodeGraph | null): CodeNodeMapping[] {
  if (!graph) return [];

  const mappings: CodeNodeMapping[] = [];

  for (const node of graph.nodes) {
    // Check if node has source location (preferred) or metadata fallback
    const source = (node as any).source;
    const metadata = (node as any).metadata;

    let lineStart: number | undefined;
    let lineEnd: number | undefined;

    if (typeof source?.line_start === 'number' && typeof source?.line_end === 'number') {
      lineStart = source.line_start;
      lineEnd = source.line_end;
    } else if (typeof source?.line === 'number') {
      lineStart = source.line;
      lineEnd = typeof source?.end_line === 'number'
        ? source.end_line
        : (typeof source?.endLine === 'number' ? source.endLine : source.line);
    } else if (metadata?.source_line_start && metadata?.source_line_end) {
      lineStart = metadata.source_line_start;
      lineEnd = metadata.source_line_end;
    }

    if (lineStart !== undefined && lineEnd !== undefined) {
      mappings.push({
        nodeId: node.id,
        lineStart,
        lineEnd,
        type: metadata?.mapping_type || 'component_create',
      });
    }
  }

  // Sort by line number
  mappings.sort((a, b) => a.lineStart - b.lineStart);

  return mappings;
}

export function useAgentSync({
  selectedFile,
  onCodeLoaded,
  onGraphLoaded,
  onMappingsBuilt,
}: UseAgentSyncOptions): UseAgentSyncResult {
  const [code, setCode] = useState('');
  const [mappings, setMappings] = useState<CodeNodeMapping[]>([]);

  // Fetch file content
  const {
    data: contentData,
    isLoading: isLoadingContent,
    error: contentError,
    refetch: refetchContent,
  } = useQuery({
    queryKey: ['agent-sync-content', selectedFile?.path],
    queryFn: () => (selectedFile ? fetchFileContent(selectedFile) : Promise.resolve('')),
    enabled: !!selectedFile,
    staleTime: 10000,
  });

  // Fetch node graph
  const {
    data: graphData,
    isLoading: isLoadingGraph,
    error: graphError,
    refetch: refetchGraph,
  } = useQuery({
    queryKey: ['agent-sync-graph', selectedFile?.path],
    queryFn: () => (selectedFile ? fetchNodeGraph(selectedFile) : Promise.resolve(null)),
    enabled: !!selectedFile,
    staleTime: 10000,
  });

  // Update code when content loads
  useEffect(() => {
    if (contentData) {
      setCode(contentData);
      onCodeLoaded?.(contentData);
    }
  }, [contentData, onCodeLoaded]);

  // Update mappings when graph loads
  useEffect(() => {
    if (graphData) {
      onGraphLoaded?.(graphData);
      const newMappings = buildMappings(graphData);
      setMappings(newMappings);
      onMappingsBuilt?.(newMappings);
    }
  }, [graphData, onGraphLoaded, onMappingsBuilt]);

  // Combined refetch
  const refetch = useCallback(() => {
    refetchContent();
    refetchGraph();
  }, [refetchContent, refetchGraph]);

  return {
    code,
    graph: graphData || null,
    mappings,
    isLoading: isLoadingContent || isLoadingGraph,
    error: contentError || graphError || null,
    refetch,
  };
}

// Helper: Find node by line number
export function findNodeByLine(mappings: CodeNodeMapping[], line: number): string | null {
  for (const mapping of mappings) {
    if (line >= mapping.lineStart && line <= mapping.lineEnd) {
      return mapping.nodeId;
    }
  }
  return null;
}

// Helper: Get line range for node
export function getLineRangeForNode(
  mappings: CodeNodeMapping[],
  nodeId: string
): { start: number; end: number } | null {
  const mapping = mappings.find((m) => m.nodeId === nodeId);
  if (!mapping) return null;
  return { start: mapping.lineStart, end: mapping.lineEnd };
}

// Helper: Get all lines for highlighting
export function getHighlightLines(
  mappings: CodeNodeMapping[],
  nodeId: string
): number[] {
  const range = getLineRangeForNode(mappings, nodeId);
  if (!range) return [];

  const lines: number[] = [];
  for (let i = range.start; i <= range.end; i++) {
    lines.push(i);
  }
  return lines;
}
