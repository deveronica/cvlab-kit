/**
 * Tab System Types
 *
 * ADR-008: 2-Tab System for Node Graph
 * - Execute Tab: Config Graph (setup dependencies)
 * - Builder Tab: Data Flow Graph (Input → Loss, Loss-Centric Scope)
 *
 * This is the SINGLE source of truth for tab mode.
 * Both Execute and Builder views import from here.
 */

/**
 * Tab mode - determines which graph to display
 *
 * @example
 * // Execute Tab shows setup() dependencies (Config Graph)
 * // Builder Tab shows train_step() data flow (Data Flow Graph)
 */
export type TabMode = 'execute' | 'builder';

/**
 * Node rendering mode within a tab
 */
export const NodeMode = {
  EXECUTE: 'execute',
  BUILDER: 'builder',
  FLOW: 'flow',
} as const;

export type NodeMode = (typeof NodeMode)[keyof typeof NodeMode];

/**
 * Get default node mode for a tab
 */
export function getDefaultNodeMode(tab: TabMode): NodeMode {
  switch (tab) {
    case 'execute':
      return 'execute';
    case 'builder':
      return 'builder';
    default:
      return 'builder';
  }
}

/**
 * Check if tab shows config graph (setup dependencies)
 */
export function isConfigGraph(tab: TabMode): boolean {
  return tab === 'execute';
}

/**
 * Check if tab shows data flow graph (train_step)
 */
export function isDataFlowGraph(tab: TabMode): boolean {
  return tab === 'builder';
}
