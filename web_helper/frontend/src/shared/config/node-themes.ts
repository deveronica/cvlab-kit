/**
 * React Flow Node Themes and Styling
 *
 * Color schemes and styles for different node types in the node editor.
 * Follows Builder tab design system with semantic colors and dark mode support.
 */

import { NodeType } from '@/shared/model/node-graph';

export interface NodeTheme {
  background: string;
  border: string;
  icon: string;
  hover: string;
}

export const nodeThemes: Record<NodeType, NodeTheme> = {
  [NodeType.AGENT]: {
    background: 'bg-amber-50 dark:bg-amber-950/20',
    border: 'border-amber-500',
    icon: 'text-amber-600 dark:text-amber-400',
    hover: 'hover:bg-amber-100 dark:hover:bg-amber-950/30',
  },
  [NodeType.COMPONENT]: {
    background: 'bg-blue-50 dark:bg-blue-950/20',
    border: 'border-blue-500',
    icon: 'text-blue-600 dark:text-blue-400',
    hover: 'hover:bg-blue-100 dark:hover:bg-blue-950/30',
  },
  [NodeType.CONFIG]: {
    background: 'bg-blue-50 dark:bg-blue-950/20',
    border: 'border-blue-500',
    icon: 'text-blue-600 dark:text-blue-400',
    hover: 'hover:bg-blue-100 dark:hover:bg-blue-950/30',
  },
  [NodeType.MODULE]: {
    background: 'bg-purple-50 dark:bg-purple-950/20',
    border: 'border-purple-500',
    icon: 'text-purple-600 dark:text-purple-400',
    hover: 'hover:bg-purple-100 dark:hover:bg-purple-950/30',
  },
  [NodeType.LAYER]: {
    background: 'bg-purple-50 dark:bg-purple-950/20',
    border: 'border-purple-500',
    icon: 'text-purple-600 dark:text-purple-400',
    hover: 'hover:bg-purple-100 dark:hover:bg-purple-950/30',
  },
  [NodeType.OPERATION]: {
    background: 'bg-green-50 dark:bg-green-950/20',
    border: 'border-green-500',
    icon: 'text-green-600 dark:text-green-400',
    hover: 'hover:bg-green-100 dark:hover:bg-green-950/30',
  },
  [NodeType.FUNCTION]: {
    background: 'bg-cyan-50 dark:bg-cyan-950/20',
    border: 'border-cyan-500',
    icon: 'text-cyan-600 dark:text-cyan-400',
    hover: 'hover:bg-cyan-100 dark:hover:bg-cyan-950/30',
  },
  [NodeType.INPUT]: {
    background: 'bg-muted',
    border: 'border-border',
    icon: 'text-muted-foreground',
    hover: 'hover:bg-muted/80',
  },
  [NodeType.OUTPUT]: {
    background: 'bg-muted',
    border: 'border-border',
    icon: 'text-muted-foreground',
    hover: 'hover:bg-muted/80',
  },
  [NodeType.PARAMETER]: {
    background: 'bg-muted',
    border: 'border-border',
    icon: 'text-muted-foreground',
    hover: 'hover:bg-muted/80',
  },
};

export interface EdgeTheme {
  stroke: string;
  strokeWidth: number;
  animated?: boolean;
}

export const edgeThemes = {
  default: {
    stroke: 'hsl(var(--border))',
    strokeWidth: 2,
  },
  selected: {
    stroke: 'hsl(var(--primary))',
    strokeWidth: 2.5,
  },
  animated: {
    stroke: 'hsl(var(--primary))',
    strokeWidth: 2.5,
    animated: true,
  },
};

// Edge themes by EdgeType (Config-Centric)
export const edgeTypeThemes = {
  tensor: {
    stroke: 'hsl(var(--chart-1))',  // Blue - tensor data flow
    strokeWidth: 2,
    strokeDasharray: undefined,
    animated: false,
  },
  gradient: {
    stroke: 'hsl(var(--chart-2))',  // Green - gradient flow
    strokeWidth: 1.5,
    strokeDasharray: '5,5',
    animated: true,
  },
  config: {
    stroke: 'hsl(var(--muted-foreground))',  // Gray - config injection
    strokeWidth: 1,
    strokeDasharray: '3,3',
    animated: false,
  },
  parameters: {
    stroke: 'hsl(var(--chart-4))',  // Orange - parameter passing
    strokeWidth: 1.5,
    strokeDasharray: '8,4',
    animated: false,
  },
  control: {
    stroke: 'hsl(var(--muted-foreground))',  // Gray - control flow
    strokeWidth: 1,
    strokeDasharray: '2,2',
    animated: false,
  },
};

// Component category themes (Config-Centric)
export const categoryThemes = {
  model: {
    background: 'bg-blue-50 dark:bg-blue-950/20',
    border: 'border-blue-500',
    icon: 'text-blue-600 dark:text-blue-400',
    badge: 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300',
  },
  optimizer: {
    background: 'bg-amber-50 dark:bg-amber-950/20',
    border: 'border-amber-500',
    icon: 'text-amber-600 dark:text-amber-400',
    badge: 'bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300',
  },
  scheduler: {
    background: 'bg-amber-50 dark:bg-amber-950/20',
    border: 'border-amber-500',
    icon: 'text-amber-600 dark:text-amber-400',
    badge: 'bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300',
  },
  loss: {
    background: 'bg-red-50 dark:bg-red-950/20',
    border: 'border-red-500',
    icon: 'text-red-600 dark:text-red-400',
    badge: 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300',
  },
  dataset: {
    background: 'bg-green-50 dark:bg-green-950/20',
    border: 'border-green-500',
    icon: 'text-green-600 dark:text-green-400',
    badge: 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300',
  },
  dataloader: {
    background: 'bg-green-50 dark:bg-green-950/20',
    border: 'border-green-500',
    icon: 'text-green-600 dark:text-green-400',
    badge: 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300',
  },
  transform: {
    background: 'bg-purple-50 dark:bg-purple-950/20',
    border: 'border-purple-500',
    icon: 'text-purple-600 dark:text-purple-400',
    badge: 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300',
  },
  metric: {
    background: 'bg-cyan-50 dark:bg-cyan-950/20',
    border: 'border-cyan-500',
    icon: 'text-cyan-600 dark:text-cyan-400',
    badge: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/50 dark:text-cyan-300',
  },
  callback: {
    background: 'bg-gray-50 dark:bg-gray-950/20',
    border: 'border-gray-500',
    icon: 'text-gray-600 dark:text-gray-400',
    badge: 'bg-gray-100 text-gray-700 dark:bg-gray-900/50 dark:text-gray-300',
  },
  logger: {
    background: 'bg-gray-50 dark:bg-gray-950/20',
    border: 'border-gray-500',
    icon: 'text-gray-600 dark:text-gray-400',
    badge: 'bg-gray-100 text-gray-700 dark:bg-gray-900/50 dark:text-gray-300',
  },
  agent: {
    background: 'bg-amber-50 dark:bg-amber-950/20',
    border: 'border-amber-500',
    icon: 'text-amber-600 dark:text-amber-400',
    badge: 'bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300',
  },
  global: {
    background: 'bg-slate-50 dark:bg-slate-950/20',
    border: 'border-slate-500',
    icon: 'text-slate-600 dark:text-slate-400',
    badge: 'bg-slate-100 text-slate-700 dark:bg-slate-900/50 dark:text-slate-300',
  },
};

export const getCategoryTheme = (category: string) => {
  return categoryThemes[category as keyof typeof categoryThemes] || categoryThemes.model;
};

export const getEdgeTypeTheme = (edgeType: string) => {
  return edgeTypeThemes[edgeType as keyof typeof edgeTypeThemes] || edgeTypeThemes.tensor;
};

export interface NodeDimensions {
  width: number;
  height: number;
}

export const nodeDimensions: Record<NodeType, NodeDimensions> = {
  [NodeType.AGENT]: { width: 240, height: 100 },
  [NodeType.COMPONENT]: { width: 240, height: 160 },  // ConfigNode: min-w-[200px] max-w-[280px]
  [NodeType.CONFIG]: { width: 240, height: 160 },     // ConfigNode: header + properties + deps
  [NodeType.MODULE]: { width: 180, height: 80 },
  [NodeType.LAYER]: { width: 180, height: 80 },
  [NodeType.OPERATION]: { width: 140, height: 60 },
  [NodeType.FUNCTION]: { width: 160, height: 65 },
  [NodeType.INPUT]: { width: 150, height: 55 },
  [NodeType.OUTPUT]: { width: 150, height: 55 },
  [NodeType.PARAMETER]: { width: 140, height: 50 },
};

export const getNodeTheme = (nodeType: NodeType | string): NodeTheme => {
  // Handle string type (from backend) or enum
  const key = nodeType as NodeType;
  return nodeThemes[key] || nodeThemes[NodeType.OPERATION];
};

export const getNodeDimensions = (nodeType: NodeType | string): NodeDimensions => {
  // Handle string type (from backend) or enum
  // Map 'config' and 'component' to their dimensions
  const typeStr = typeof nodeType === 'string' ? nodeType : String(nodeType);

  // Direct lookup first
  const key = typeStr as NodeType;
  if (nodeDimensions[key]) {
    return nodeDimensions[key];
  }

  // Default fallback with reasonable size for unknown types
  return { width: 200, height: 100 };
};
