/**
 * Chart Interaction Utilities
 *
 * Helper functions for implementing cross-chart interactions:
 * - Synchronized hover highlighting
 * - Brushing and filtering
 * - Color consistency across charts
 */

import type { Run } from './types';

/**
 * Generate consistent color for a run based on its name
 * Uses hash function to ensure same run always gets same color
 */
export function getRunColor(runName: string, colorPalette?: string[]): string {
  const defaultPalette = [
    '#8b5cf6', // violet
    '#06b6d4', // cyan
    '#10b981', // emerald
    '#f59e0b', // amber
    '#ef4444', // red
    '#f97316', // orange
    '#84cc16', // lime
    '#ec4899', // pink
    '#6366f1', // indigo
    '#14b8a6', // teal
  ];

  const palette = colorPalette || defaultPalette;

  // Simple hash function
  let hash = 0;
  for (let i = 0; i < runName.length; i++) {
    hash = runName.charCodeAt(i) + ((hash << 5) - hash);
  }

  const index = Math.abs(hash) % palette.length;
  return palette[index];
}

/**
 * Apply opacity based on highlight state
 */
export function applyHighlightOpacity(
  isHighlighted: boolean,
  highlightedCount: number,
  baseOpacity = 0.8,
  dimOpacity = 0.2
): number {
  if (highlightedCount === 0) return baseOpacity;
  return isHighlighted ? baseOpacity : dimOpacity;
}

/**
 * Filter runs based on brushed range
 */
export function filterByBrush(
  runs: Run[],
  brushedRange: [number, number] | null,
  metricKey: string
): Run[] {
  if (!brushedRange) return runs;

  const [min, max] = brushedRange;
  return runs.filter(run => {
    const value = run.metrics?.final?.[metricKey];
    if (typeof value !== 'number') return false;
    return value >= min && value <= max;
  });
}

/**
 * Get runs within a value range
 */
export function getRunsInRange(
  runs: Run[],
  metricKey: string,
  min: number,
  max: number
): string[] {
  return runs
    .filter(run => {
      const value = run.metrics?.final?.[metricKey];
      if (typeof value !== 'number') return false;
      return value >= min && value <= max;
    })
    .map(run => run.run_name);
}

/**
 * Synchronize Recharts hover events
 * Returns event handlers for synced hover
 */
export function createSyncedHoverHandlers(
  onHoverStart: (runName: string) => void,
  onHoverEnd: () => void
) {
  return {
    onMouseEnter: (data: any) => {
      if (data && data.name) {
        onHoverStart(data.name);
      } else if (data && data.runName) {
        onHoverStart(data.runName);
      }
    },
    onMouseLeave: () => {
      onHoverEnd();
    },
  };
}

/**
 * Create Recharts brush event handler
 */
export function createBrushHandler(
  onBrush: (range: [number, number] | null) => void
) {
  return {
    onChange: (brushArea: any) => {
      if (brushArea && brushArea.startIndex !== undefined && brushArea.endIndex !== undefined) {
        // Extract actual data range from brush indices
        // This is a simplified version - real implementation depends on data structure
        onBrush([brushArea.startIndex, brushArea.endIndex]);
      } else {
        onBrush(null);
      }
    },
  };
}

/**
 * Apply consistent styling to chart elements based on interaction state
 */
export function getChartElementStyle(
  runName: string,
  highlightedRuns: Set<string>,
  selectedRunName: string | null
): {
  opacity: number;
  strokeWidth?: number;
  emphasis?: boolean;
} {
  const isHighlighted = highlightedRuns.has(runName);
  const isSelected = selectedRunName === runName;
  const hasHighlights = highlightedRuns.size > 0;

  if (isSelected) {
    return {
      opacity: 1,
      strokeWidth: 3,
      emphasis: true,
    };
  }

  if (hasHighlights) {
    return {
      opacity: isHighlighted ? 0.8 : 0.2,
      strokeWidth: isHighlighted ? 2 : 1,
    };
  }

  return {
    opacity: 0.7,
    strokeWidth: 2,
  };
}

/**
 * Debounce function for performance optimization
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

/**
 * Create tooltip content with consistent formatting
 */
export function formatTooltipContent(
  runName: string,
  metrics: Record<string, number | string>,
  highlightKeys?: string[]
): string {
  let html = `<div style="font-weight: bold; margin-bottom: 8px;">${runName}</div>`;

  Object.entries(metrics).forEach(([key, value]) => {
    const isHighlighted = highlightKeys?.includes(key);
    const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;

    html += `
      <div style="display: flex; justify-content: space-between; margin: 4px 0; ${
        isHighlighted ? 'font-weight: bold;' : ''
      }">
        <span style="margin-right: 16px; color: ${isHighlighted ? '#000' : '#888'};">${key}:</span>
        <span style="font-family: monospace;">${formattedValue}</span>
      </div>
    `;
  });

  return html;
}

/**
 * Calculate brush range from chart coordinates
 */
export function calculateBrushRange(
  startCoord: { x: number; y: number },
  endCoord: { x: number; y: number },
  chartArea: { x: number; y: number; width: number; height: number },
  dataRange: { min: number; max: number }
): [number, number] | null {
  const startPercent = (startCoord.x - chartArea.x) / chartArea.width;
  const endPercent = (endCoord.x - chartArea.x) / chartArea.width;

  const min = dataRange.min + startPercent * (dataRange.max - dataRange.min);
  const max = dataRange.min + endPercent * (dataRange.max - dataRange.min);

  return [Math.min(min, max), Math.max(min, max)];
}
