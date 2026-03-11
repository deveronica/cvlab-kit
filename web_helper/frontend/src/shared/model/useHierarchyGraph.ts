/**
 * React hooks for Hierarchy Graph API (Simulink-style navigation)
 */

import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import type {
  HierarchicalNodeGraph,
  HierarchyResponse,
} from "../model/hierarchy";

const API_BASE = "/api/nodes";

// Phase type for filtering
// - "initialize": setup phase showing component dependencies
// - "flow": train_step phase showing data flow
// - "config": initialize phase + YAML config node with auto-binding
export type BuilderPhase = "initialize" | "flow" | "config";

// Query keys
export const hierarchyKeys = {
  all: ["hierarchy"] as const,
  graph: (agentName: string, path: string, phase?: BuilderPhase, method?: string, impl?: string, configPath?: string) =>
    [...hierarchyKeys.all, "graph", agentName, path, phase, method, impl, configPath] as const,
  agents: () => [...hierarchyKeys.all, "agents"] as const,
};

/**
 * Fetch hierarchy graph from API
 *
 * Uses phase-separated endpoints:
 * - initialize (setup): GET /hierarchy/{agent_name}/setup
 * - flow (train_step): GET /hierarchy/{agent_name}/train-step
 */
async function fetchHierarchyGraph(
  agentName: string,
  path: string,
  phase?: BuilderPhase,
  method?: string,
  impl?: string,
  configPath?: string
): Promise<HierarchicalNodeGraph> {
  let endpoint: string;
  const params = new URLSearchParams();

  // Route to phase-specific endpoint
  if (phase === "flow") {
    // Train-step phase: data flow visualization
    endpoint = `${API_BASE}/hierarchy/${agentName}/train-step`;
    params.set("path", path);
    if (method) params.set("method", method);
    if (impl) params.set("impl", impl);
  } else if (phase === "initialize" || phase === "config") {
    // Setup phase: component dependencies (initialize and config both use setup endpoint)
    endpoint = `${API_BASE}/hierarchy/${agentName}/setup`;
    params.set("path", path);
    if (impl) params.set("impl", impl);
    if (configPath) params.set("config_path", configPath);
  } else {
    // Fallback: default to setup
    endpoint = `${API_BASE}/hierarchy/${agentName}/setup`;
    params.set("path", path);
  }

  const response = await fetch(`${endpoint}?${params.toString()}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Failed to fetch hierarchy: ${response.statusText}`);
  }

  const result: HierarchyResponse = await response.json();

  if (!result.success || !result.data) {
    throw new Error(result.error || "Failed to fetch hierarchy graph");
  }

  return result.data;
}

/**
 * Hook to fetch hierarchy graph for a given agent and path
 *
 * @param agentName - Agent file name
 * @param path - Drill-down path
 * @param phase - Phase filter
 * @param method - Method name for flow phase
 * @param impl - Implementation name for drill-down (from AST or YAML)
 */
export function useHierarchyGraph(
  agentName: string,
  path: string = "",
  phase?: BuilderPhase,
  method?: string,
  impl?: string,
  configPath?: string
) {
  return useQuery({
    queryKey: hierarchyKeys.graph(agentName, path, phase, method, impl, configPath),
    queryFn: () => fetchHierarchyGraph(agentName, path, phase, method, impl, configPath),
    enabled: !!agentName,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    retry: 1,
  });
}

/**
 * Hook to prefetch hierarchy graph (for hover prefetch)
 */
export function usePrefetchHierarchyGraph() {
  const queryClient = useQueryClient();

  return useCallback(
    (agentName: string, path: string, phase?: BuilderPhase, method?: string, impl?: string) => {
      queryClient.prefetchQuery({
        queryKey: hierarchyKeys.graph(agentName, path, phase, method, impl),
        queryFn: () => fetchHierarchyGraph(agentName, path, phase, method, impl),
        staleTime: 5 * 60 * 1000,
      });
    },
    [queryClient]
  );
}

/**
 * Navigation state for Simulink-style drill-down
 */
interface HistoryEntry {
  path: string;
  impl?: string;
  viewport?: { x: number; y: number; zoom: number };
}

export interface NavigationState {
  currentPath: string;
  currentImpl?: string;
  history: HistoryEntry[];
  canGoBack: boolean;
}

/**
 * Hook for managing hierarchy navigation state
 *
 * Supports impl parameter for drill-down:
 * - When drilling into a component, pass impl from node.metadata.impl
 * - For YAML context, the caller can provide impl from parsed YAML config
 */
export function useHierarchyNavigation(agentName: string, phase?: BuilderPhase, method?: string, configPath?: string) {
  const [currentPath, setCurrentPath] = useState("");
  const [currentImpl, setCurrentImpl] = useState<string | undefined>(undefined);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const queryClient = useQueryClient();
  const prefetch = usePrefetchHierarchyGraph();

  // Current graph query with phase, method, impl, and configPath filtering
  const graphQuery = useHierarchyGraph(agentName, currentPath, phase, method, currentImpl, configPath);

  /**
   * Navigate to a path (drill-down)
   *
   * @param newPath - The new drill-down path
   * @param options - Navigation options
   * @param options.viewport - Current viewport state to save for back navigation
   * @param options.impl - Implementation name for the target component
   *                       Priority: YAML config > AST metadata.impl
   */
  const navigateTo = useCallback(
    (newPath: string, options?: {
      viewport?: { x: number; y: number; zoom: number };
      impl?: string;
    }) => {
      // Save current state to history
      setHistory((prev) => [...prev, { path: currentPath, impl: currentImpl, viewport: options?.viewport }]);
      setCurrentPath(newPath);
      setCurrentImpl(options?.impl);
    },
    [currentPath, currentImpl]
  );

  // Go back to previous level
  const goBack = useCallback(() => {
    if (history.length === 0) return null;

    const prevEntry = history[history.length - 1];
    setHistory((prev) => prev.slice(0, -1));
    setCurrentPath(prevEntry.path);
    setCurrentImpl(prevEntry.impl);

    return prevEntry.viewport;
  }, [history]);

  // Navigate to specific breadcrumb level
  const navigateToBreadcrumb = useCallback(
    (index: number) => {
      if (!graphQuery.data) return;

      const breadcrumbs = graphQuery.data.hierarchy.path;

      if (index < 0) {
        // Navigate to root (Level 0)
        setCurrentPath("");
        setHistory([]);
      } else if (index < breadcrumbs.length) {
        // Navigate to specific breadcrumb
        const targetPath = breadcrumbs
          .slice(0, index + 1)
          .map((p) => p.node_id)
          .join("/");

        // Remove history entries after this point
        setHistory((prev) => prev.slice(0, index + 1));
        setCurrentPath(targetPath);
      }
    },
    [graphQuery.data]
  );

  // Prefetch on hover (for better UX)
  const prefetchPath = useCallback(
    (path: string, impl?: string) => {
      prefetch(agentName, path, phase, method, impl);
    },
    [agentName, prefetch, phase, method]
  );

  // Clear navigation state
  const reset = useCallback(() => {
    setCurrentPath("");
    setCurrentImpl(undefined);
    setHistory([]);
  }, []);

  return {
    // Query state
    graph: graphQuery.data,
    isLoading: graphQuery.isLoading,
    error: graphQuery.error,
    refetch: graphQuery.refetch,

    // Navigation state
    currentPath,
    currentImpl,
    history,
    canGoBack: history.length > 0,

    // Navigation actions
    navigateTo,
    goBack,
    navigateToBreadcrumb,
    prefetchPath,
    reset,
  };
}

/**
 * Hook to get available agents
 */
export function useAvailableAgents() {
  return useQuery({
    queryKey: hierarchyKeys.agents(),
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/agents`);

      if (!response.ok) {
        throw new Error("Failed to fetch available agents");
      }

      const result = await response.json();
      return result.agents as string[];
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
    gcTime: 30 * 60 * 1000, // 30 minutes
  });
}

/**
 * Invalidate hierarchy cache (e.g., after code changes)
 */
export function useInvalidateHierarchyCache() {
  const queryClient = useQueryClient();

  return useCallback(() => {
    queryClient.invalidateQueries({ queryKey: hierarchyKeys.all });
  }, [queryClient]);
}
