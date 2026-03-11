/**
 * Node System Themes
 *
 * Category-based visual theming for consistent node appearance
 * across both Execute Tab and Builder Tab.
 */

import type { ComponentCategory } from '@/entities/node-system/model/types';

/**
 * Theme configuration for a category
 */
export interface CategoryTheme {
  /** Background color class */
  background: string;
  /** Border color class */
  border: string;
  /** Badge/accent color class */
  badge: string;
  /** Icon color class */
  icon: string;
  /** Display label */
  label: string;
}

/**
 * Category themes using Tailwind classes
 *
 * Color semantics:
 * - Model (blue): Primary processing unit
 * - Optimizer (green): Optimization/learning
 * - Loss (red): Error/loss computation
 * - Dataset (purple): Data source
 * - DataLoader (indigo): Data pipeline
 * - Transform (cyan): Data transformation
 * - Metric (orange): Evaluation
 * - Scheduler (teal): Learning rate control
 * - Sampler (pink): Data sampling
 * - Checkpoint (yellow): State persistence
 */
export const CATEGORY_THEMES: Record<ComponentCategory, CategoryTheme> = {
  model: {
    background: 'bg-blue-50 dark:bg-blue-950/30',
    border: 'border-blue-200 dark:border-blue-800',
    badge: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    icon: 'text-blue-600 dark:text-blue-400',
    label: 'Model',
  },
  optimizer: {
    background: 'bg-green-50 dark:bg-green-950/30',
    border: 'border-green-200 dark:border-green-800',
    badge: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    icon: 'text-green-600 dark:text-green-400',
    label: 'Optimizer',
  },
  loss: {
    background: 'bg-red-50 dark:bg-red-950/30',
    border: 'border-red-200 dark:border-red-800',
    badge: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    icon: 'text-red-600 dark:text-red-400',
    label: 'Loss',
  },
  dataset: {
    background: 'bg-purple-50 dark:bg-purple-950/30',
    border: 'border-purple-200 dark:border-purple-800',
    badge: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    icon: 'text-purple-600 dark:text-purple-400',
    label: 'Dataset',
  },
  dataloader: {
    background: 'bg-indigo-50 dark:bg-indigo-950/30',
    border: 'border-indigo-200 dark:border-indigo-800',
    badge: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200',
    icon: 'text-indigo-600 dark:text-indigo-400',
    label: 'DataLoader',
  },
  transform: {
    background: 'bg-cyan-50 dark:bg-cyan-950/30',
    border: 'border-cyan-200 dark:border-cyan-800',
    badge: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200',
    icon: 'text-cyan-600 dark:text-cyan-400',
    label: 'Transform',
  },
  metric: {
    background: 'bg-orange-50 dark:bg-orange-950/30',
    border: 'border-orange-200 dark:border-orange-800',
    badge: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    icon: 'text-orange-600 dark:text-orange-400',
    label: 'Metric',
  },
  scheduler: {
    background: 'bg-teal-50 dark:bg-teal-950/30',
    border: 'border-teal-200 dark:border-teal-800',
    badge: 'bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200',
    icon: 'text-teal-600 dark:text-teal-400',
    label: 'Scheduler',
  },
  sampler: {
    background: 'bg-pink-50 dark:bg-pink-950/30',
    border: 'border-pink-200 dark:border-pink-800',
    badge: 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200',
    icon: 'text-pink-600 dark:text-pink-400',
    label: 'Sampler',
  },
  checkpoint: {
    background: 'bg-yellow-50 dark:bg-yellow-950/30',
    border: 'border-yellow-200 dark:border-yellow-800',
    badge: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    icon: 'text-yellow-600 dark:text-yellow-400',
    label: 'Checkpoint',
  },
  callback: {
    background: 'bg-slate-50 dark:bg-slate-950/30',
    border: 'border-slate-200 dark:border-slate-800',
    badge: 'bg-slate-100 text-slate-800 dark:bg-slate-900 dark:text-slate-200',
    icon: 'text-slate-600 dark:text-slate-400',
    label: 'Callback',
  },
  logger: {
    background: 'bg-slate-50 dark:bg-slate-950/30',
    border: 'border-slate-200 dark:border-slate-800',
    badge: 'bg-slate-100 text-slate-800 dark:bg-slate-900 dark:text-slate-200',
    icon: 'text-slate-600 dark:text-slate-400',
    label: 'Logger',
  },
  agent: {
    background: 'bg-slate-50 dark:bg-slate-950/30',
    border: 'border-slate-200 dark:border-slate-800',
    badge: 'bg-slate-100 text-slate-800 dark:bg-slate-900 dark:text-slate-200',
    icon: 'text-slate-600 dark:text-slate-400',
    label: 'Agent',
  },
  global: {
    background: 'bg-slate-50 dark:bg-slate-950/30',
    border: 'border-slate-200 dark:border-slate-800',
    badge: 'bg-slate-100 text-slate-800 dark:bg-slate-900 dark:text-slate-200',
    icon: 'text-slate-600 dark:text-slate-400',
    label: 'Global',
  },
  unknown: {
    background: 'bg-gray-50 dark:bg-gray-950/30',
    border: 'border-gray-200 dark:border-gray-800',
    badge: 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200',
    icon: 'text-gray-600 dark:text-gray-400',
    label: 'Unknown',
  },
};

/**
 * Get theme for a category
 */
export function getCategoryTheme(category: ComponentCategory | string): CategoryTheme {
  const cat = category as ComponentCategory;
  return CATEGORY_THEMES[cat] || CATEGORY_THEMES.unknown;
}

/**
 * Port type color mapping
 */
export const PORT_COLORS: Record<string, string> = {
  tensor: '#3b82f6',     // blue
  module: '#22c55e',     // green
  parameters: '#22c55e', // green (same as module)
  scalar: '#f97316',     // orange
  bool: '#06b6d4',       // cyan
  string: '#8b5cf6',     // violet
  list: '#ec4899',       // pink
  dict: '#a855f7',       // purple
  config: '#a855f7',     // purple
  device: '#6b7280',     // gray
  execution: '#2563eb',  // blue-600
  any: '#9ca3af',        // gray-400
};

/**
 * Get color for a port type
 */
export function getPortColor(type: string): string {
  return PORT_COLORS[type.toLowerCase()] || PORT_COLORS.any;
}
