/**
 * API hooks for fetching node graphs
 *
 * Multi-View Architecture (Simulink-style):
 * - Agent View (Primary): setup() components + high-level data flow
 * - Execution View: Detailed train_step flow (separate tab)
 * - Config View: YAML config file parsing
 * - Drill-down: Component internal structure
 *
 * Legacy hooks kept for backward compatibility.
 */

import { useQuery, useMutation } from '@tanstack/react-query';
import {
  NodeGraph,
  NodeGraphAPIResponse,
  AvailableMethodsAPIResponse,
} from '@/shared/model/node-graph';

// ============================================================================
// Primary Hooks (Simulink-style Multi-View Architecture)
// ============================================================================

/**
 * Fetch Agent View: setup() components + high-level data flow.
 *
 * This is the PRIMARY hook showing:
 * - Components defined in setup() as nodes (6-8 nodes)
 * - High-level data flow between components as edges
 * - NO internal operations (unpack, backward, zero_grad, etc.)
 * - All components have has_children=True for drill-down support
 */
export const useAgentView = (agentName: string | null) => {
  return useQuery<NodeGraph, Error>({
    queryKey: ['nodeGraph', 'agentView', agentName],
    queryFn: async () => {
      if (!agentName) throw new Error('Agent name is required');

      const response = await fetch(`/api/nodes/agent-view/${agentName}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch agent view');
      }

      const result: NodeGraphAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!agentName,
  });
};

/**
 * Fetch Execution View: detailed train_step flow (separate tab).
 *
 * Shows ALL operations including:
 * - Tuple unpacking (inputs, labels = batch)
 * - Component calls (self.model(inputs))
 * - Method calls (loss.backward(), optimizer.step())
 * - Full data flow including internal variables
 */
export const useExecutionView = (
  agentName: string | null,
  method: string = 'train_step'
) => {
  return useQuery<NodeGraph, Error>({
    queryKey: ['nodeGraph', 'executionView', agentName, method],
    queryFn: async () => {
      if (!agentName) throw new Error('Agent name is required');

      const response = await fetch(
        `/api/nodes/execution-view/${agentName}?method=${method}`
      );
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch execution view');
      }

      const result: NodeGraphAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!agentName,
  });
};

/**
 * Drill into a component's internal structure.
 *
 * Supports all component categories:
 * - model: PyTorch layers (Conv2d, BatchNorm, etc.)
 * - loss: Sub-losses
 * - optimizer: Parameter groups
 * - transform: Pipeline steps
 * - dataset: Sample structure
 * - metric: Internal computation
 */
export const useDrillComponent = (
  componentId: string | null,
  agentName: string | null,
  category: string | null
) => {
  return useQuery<NodeGraph, Error>({
    queryKey: ['nodeGraph', 'drill', agentName, componentId, category],
    queryFn: async () => {
      if (!componentId || !agentName || !category) {
        throw new Error('Component ID, agent name, and category are required');
      }

      const response = await fetch(
        `/api/nodes/drill/${componentId}?agent=${agentName}&category=${category}`
      );
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to drill into component');
      }

      const result: NodeGraphAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!componentId && !!agentName && !!category,
  });
};

// ============================================================================
// Legacy Hooks (kept for backward compatibility)
// ============================================================================

/**
 * @deprecated Use useAgentView instead
 * Fetch unified node graph using Agent-First design.
 */
export const useUnifiedGraph = (
  agentName: string | null,
  configPath: string | null = null,
  method: string = 'train_step'
) => {
  return useQuery<NodeGraph, Error>({
    queryKey: ['nodeGraph', 'unified', agentName, configPath, method],
    queryFn: async () => {
      if (!agentName) throw new Error('Agent name is required');

      let url = `/api/nodes/unified/${agentName}?method=${method}`;
      if (configPath) {
        url += `&config_path=${encodeURIComponent(configPath)}`;
      }

      const response = await fetch(url);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch unified graph');
      }

      const result: NodeGraphAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!agentName,
  });
};

// ============================================================================
// Config-Centric Hooks
// ============================================================================

/**
 * Fetch node graph from a YAML config file
 */
export const useConfigGraph = (configPath: string | null) => {
  return useQuery<NodeGraph, Error>({
    queryKey: ['nodeGraph', 'config', configPath],
    queryFn: async () => {
      if (!configPath) throw new Error('Config path is required');

      const response = await fetch(`/api/nodes/config/${encodeURIComponent(configPath)}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch config graph');
      }

      const result: NodeGraphAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!configPath,
  });
};

/**
 * Fetch detailed agent flow graph with AST analysis
 */
export const useAgentFlowGraph = (agentName: string | null, method: string = 'train_step') => {
  return useQuery<NodeGraph, Error>({
    queryKey: ['nodeGraph', 'agentFlow', agentName, method],
    queryFn: async () => {
      if (!agentName) throw new Error('Agent name is required');

      const response = await fetch(`/api/nodes/agent-flow/${agentName}?method=${method}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch agent flow graph');
      }

      const result: NodeGraphAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!agentName,
  });
};

/**
 * Fetch combined Config + Agent flow graph (richest visualization)
 */
export const useCombinedGraph = (
  agentName: string | null,
  configPath: string | null,
  method: string = 'train_step'
) => {
  return useQuery<NodeGraph, Error>({
    queryKey: ['nodeGraph', 'combined', agentName, configPath, method],
    queryFn: async () => {
      if (!agentName) throw new Error('Agent name is required');

      let url = `/api/nodes/combined/${agentName}?method=${method}`;
      if (configPath) {
        url += `&config_path=${encodeURIComponent(configPath)}`;
      }

      const response = await fetch(url);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch combined graph');
      }

      const result: NodeGraphAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!agentName,
  });
};

// ============================================================================
// Legacy Hooks (Backward Compatibility)
// ============================================================================

export const useAgentGraph = (agentName: string | null, method: string = 'train_step') => {
  return useQuery<NodeGraph, Error>({
    queryKey: ['nodeGraph', 'agent', agentName, method],
    queryFn: async () => {
      if (!agentName) throw new Error('Agent name is required');

      const response = await fetch(`/api/nodes/agent/${agentName}?method=${method}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch agent graph');
      }

      const result: NodeGraphAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!agentName,
  });
};

export const useComponentGraph = (category: string | null, name: string | null) => {
  return useQuery<NodeGraph, Error>({
    queryKey: ['nodeGraph', 'component', category, name],
    queryFn: async () => {
      if (!category || !name) throw new Error('Category and name are required');

      const response = await fetch(`/api/nodes/component/${category}/${name}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch component graph');
      }

      const result: NodeGraphAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!category && !!name,
  });
};

export const useAvailableMethods = (agentName: string | null) => {
  return useQuery<{ methods: string[]; default: string }, Error>({
    queryKey: ['availableMethods', agentName],
    queryFn: async () => {
      if (!agentName) throw new Error('Agent name is required');

      const response = await fetch(`/api/nodes/available-methods/${agentName}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch available methods');
      }

      const result: AvailableMethodsAPIResponse = await response.json();
      return result.data;
    },
    enabled: !!agentName,
  });
};
