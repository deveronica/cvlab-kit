/**
 * useConfigAutoBinding - Auto-bind implementation when Config connects
 *
 * Phase 3: Auto Config Binding
 *
 * When a YamlConfigNode connects to a ComponentNode:
 * - Extract the config value (implementation name)
 * - Auto-set the target node's implementation field
 * - Update value source to CONFIG
 *
 * Example:
 * - Config node: model = "resnet18"
 * - Component node: [Model] (empty)
 * - After connection: [Model] resnet18 🔧
 */

import { useCallback } from 'react';
import type { Node, Edge, Connection } from 'reactflow';

// Types
interface ConfigBinding {
  targetNodeId: string;
  category: string;
  implementation: string;
  configPath: string;
}

interface BindingResult {
  success: boolean;
  binding?: ConfigBinding;
  error?: string;
}

interface UseConfigAutoBindingOptions {
  nodes: Node[];
  updateNode: (nodeId: string, updates: Partial<Node['data']>) => void;
}

/**
 * Hook for auto-binding implementation when Config node connects
 */
export function useConfigAutoBinding({
  nodes,
  updateNode,
}: UseConfigAutoBindingOptions) {

  /**
   * Check if a node is a YamlConfigNode
   */
  const isConfigNode = useCallback((node: Node | undefined): boolean => {
    if (!node) return false;
    return node.type === 'yamlConfig' || node.data?.nodeType === 'yamlConfig';
  }, []);

  /**
   * Check if a node is a ComponentNode that accepts implementation
   */
  const isComponentNode = useCallback((node: Node | undefined): boolean => {
    if (!node) return false;
    return (
      node.type === 'component' ||
      node.type === 'config' ||
      node.data?.category != null
    );
  }, []);

  /**
   * Extract config value from sourceHandle
   *
   * YamlConfigNode uses handles like "config_model" or "config_lr"
   * We extract the key and look up the value in the node's properties
   */
  const extractConfigValue = useCallback(
    (sourceNode: Node, sourceHandle: string): { key: string; value: unknown } | null => {
      // Handle format: "config_{key}" or just "{key}"
      const key = sourceHandle.replace(/^config_/, '');

      // Look up in node's properties
      const properties = sourceNode.data?.properties || [];
      const prop = properties.find((p: { key: string }) => p.key === key);

      if (prop) {
        return { key: prop.key, value: prop.value };
      }

      // Fallback: check if node has direct config data
      if (sourceNode.data?.config?.[key] != null) {
        return { key, value: sourceNode.data.config[key] };
      }

      return null;
    },
    []
  );

  /**
   * Process a connection to check for config binding
   */
  const processConnection = useCallback(
    (connection: Connection): BindingResult => {
      if (!connection.source || !connection.target) {
        return { success: false, error: 'Missing source or target' };
      }

      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetNode = nodes.find((n) => n.id === connection.target);

      // Check if this is a config → component connection
      if (!isConfigNode(sourceNode) || !isComponentNode(targetNode)) {
        return { success: false, error: 'Not a config binding connection' };
      }

      // Extract config value
      const configValue = extractConfigValue(
        sourceNode!,
        connection.sourceHandle || 'config'
      );

      if (!configValue) {
        return {
          success: false,
          error: `No config value found for handle: ${connection.sourceHandle}`,
        };
      }

      // Determine target category
      const targetCategory = targetNode!.data?.category || 'unknown';

      // Check if config key matches target category
      // e.g., "model" config → Model component
      const keyCategory = configValue.key.toLowerCase();
      const nodeCategory = targetCategory.toLowerCase();

      // Allow binding if categories match or target has no impl yet
      const currentImpl = targetNode!.data?.implementation;
      const shouldBind =
        keyCategory === nodeCategory ||
        keyCategory.includes(nodeCategory) ||
        nodeCategory.includes(keyCategory) ||
        !currentImpl;

      if (!shouldBind) {
        return {
          success: false,
          error: `Category mismatch: config key "${configValue.key}" vs node "${targetCategory}"`,
        };
      }

      // Create binding
      const binding: ConfigBinding = {
        targetNodeId: connection.target,
        category: targetCategory,
        implementation: String(configValue.value),
        configPath: sourceNode!.data?.configPath || 'config.yaml',
      };

      return { success: true, binding };
    },
    [nodes, isConfigNode, isComponentNode, extractConfigValue]
  );

  /**
   * Apply config binding to target node
   */
  const applyBinding = useCallback(
    (binding: ConfigBinding): void => {
      updateNode(binding.targetNodeId, {
        implementation: binding.implementation,
        // Mark value source as CONFIG for visual feedback
        implSource: 'config',
        configPath: binding.configPath,
        // Add binding indicator
        isBoundFromConfig: true,
      });

      console.log(
        `🔧 Auto-bound "${binding.implementation}" to ${binding.category} node (from ${binding.configPath})`
      );
    },
    [updateNode]
  );

  /**
   * Main handler: process connection and apply binding if valid
   */
  const handleConnection = useCallback(
    (connection: Connection): BindingResult => {
      const result = processConnection(connection);

      if (result.success && result.binding) {
        applyBinding(result.binding);
      }

      return result;
    },
    [processConnection, applyBinding]
  );

  /**
   * Process edge addition (for use with edge change events)
   */
  const handleEdgeAdd = useCallback(
    (edge: Edge): BindingResult => {
      const connection: Connection = {
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle || null,
        targetHandle: edge.targetHandle || null,
      };

      return handleConnection(connection);
    },
    [handleConnection]
  );

  /**
   * Unbind implementation when config edge is removed
   */
  const handleEdgeRemove = useCallback(
    (edge: Edge): void => {
      const targetNode = nodes.find((n) => n.id === edge.target);

      // Only clear if the node was bound from config
      if (targetNode?.data?.isBoundFromConfig) {
        updateNode(edge.target, {
          implementation: undefined,
          implSource: 'default',
          isBoundFromConfig: false,
          configPath: undefined,
        });

        console.log(
          `🔓 Unbound config from ${targetNode.data?.category || 'node'} (edge removed)`
        );
      }
    },
    [nodes, updateNode]
  );

  return {
    handleConnection,
    handleEdgeAdd,
    handleEdgeRemove,
    isConfigNode,
    isComponentNode,
    processConnection,
  };
}

export type { ConfigBinding, BindingResult, UseConfigAutoBindingOptions };
