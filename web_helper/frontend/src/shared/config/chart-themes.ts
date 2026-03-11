/**
 * Chart Theme System
 *
 * Provides predefined color schemes and styling for charts.
 * Supports multiple themes optimized for different use cases:
 * - default: Standard colorful theme
 * - publication: Black & white, printer-friendly
 * - colorblind: Colorblind-safe palette
 * - dark: Dark mode optimized
 */

export interface ChartTheme {
  name: string;
  displayName: string;
  description: string;
  colors: string[];
  backgroundColor: string;
  gridColor: string;
  textColor: string;
  tooltipBg: string;
  tooltipBorder: string;
  axisColor: string;
}

export const CHART_THEMES: Record<string, ChartTheme> = {
  default: {
    name: 'default',
    displayName: 'Default',
    description: 'Vibrant colors for general use',
    colors: [
      '#8b5cf6', // violet-500
      '#06b6d4', // cyan-500
      '#10b981', // emerald-500
      '#f59e0b', // amber-500
      '#ef4444', // red-500
      '#f97316', // orange-500
      '#84cc16', // lime-500
      '#ec4899', // pink-500
      '#6366f1', // indigo-500
      '#14b8a6', // teal-500
    ],
    backgroundColor: '#ffffff',
    gridColor: '#e5e7eb',
    textColor: '#374151',
    tooltipBg: '#ffffff',
    tooltipBorder: '#d1d5db',
    axisColor: '#9ca3af',
  },

  publication: {
    name: 'publication',
    displayName: 'Publication',
    description: 'Black & white, printer-friendly',
    colors: [
      '#000000', // solid black
      '#4b5563', // gray-600
      '#6b7280', // gray-500
      '#9ca3af', // gray-400
      '#d1d5db', // gray-300
    ],
    backgroundColor: '#ffffff',
    gridColor: '#e5e7eb',
    textColor: '#000000',
    tooltipBg: '#ffffff',
    tooltipBorder: '#000000',
    axisColor: '#000000',
  },

  colorblind: {
    name: 'colorblind',
    displayName: 'Colorblind Safe',
    description: 'Accessible for color vision deficiency',
    colors: [
      '#0173B2', // Blue
      '#DE8F05', // Orange
      '#029E73', // Green
      '#CC78BC', // Purple
      '#CA9161', // Brown
      '#FBAFE4', // Pink
      '#949494', // Gray
      '#ECE133', // Yellow
    ],
    backgroundColor: '#ffffff',
    gridColor: '#e5e7eb',
    textColor: '#374151',
    tooltipBg: '#ffffff',
    tooltipBorder: '#d1d5db',
    axisColor: '#9ca3af',
  },

  dark: {
    name: 'dark',
    displayName: 'Dark Mode',
    description: 'Optimized for dark backgrounds',
    colors: [
      '#a78bfa', // violet-400
      '#22d3ee', // cyan-400
      '#34d399', // emerald-400
      '#fbbf24', // amber-400
      '#f87171', // red-400
      '#fb923c', // orange-400
      '#a3e635', // lime-400
      '#f472b6', // pink-400
      '#818cf8', // indigo-400
      '#2dd4bf', // teal-400
    ],
    backgroundColor: '#1f2937',
    gridColor: '#374151',
    textColor: '#f3f4f6',
    tooltipBg: '#1f2937',
    tooltipBorder: '#4b5563',
    axisColor: '#9ca3af',
  },

  pastel: {
    name: 'pastel',
    displayName: 'Pastel',
    description: 'Soft, muted colors',
    colors: [
      '#c4b5fd', // violet-300
      '#67e8f9', // cyan-300
      '#6ee7b7', // emerald-300
      '#fcd34d', // amber-300
      '#fca5a5', // red-300
      '#fdba74', // orange-300
      '#bef264', // lime-300
      '#f9a8d4', // pink-300
      '#a5b4fc', // indigo-300
      '#5eead4', // teal-300
    ],
    backgroundColor: '#ffffff',
    gridColor: '#e5e7eb',
    textColor: '#374151',
    tooltipBg: '#ffffff',
    tooltipBorder: '#d1d5db',
    axisColor: '#9ca3af',
  },

  monochrome: {
    name: 'monochrome',
    displayName: 'Monochrome',
    description: 'Grayscale gradient',
    colors: [
      '#111827', // gray-900
      '#1f2937', // gray-800
      '#374151', // gray-700
      '#4b5563', // gray-600
      '#6b7280', // gray-500
      '#9ca3af', // gray-400
      '#d1d5db', // gray-300
      '#e5e7eb', // gray-200
    ],
    backgroundColor: '#ffffff',
    gridColor: '#e5e7eb',
    textColor: '#111827',
    tooltipBg: '#ffffff',
    tooltipBorder: '#d1d5db',
    axisColor: '#9ca3af',
  },
};

/**
 * Get theme by name, fallback to default if not found
 */
export function getTheme(themeName?: string): ChartTheme {
  if (!themeName || !CHART_THEMES[themeName]) {
    return CHART_THEMES.default;
  }
  return CHART_THEMES[themeName];
}

/**
 * Get color from theme by index (with wrapping)
 */
export function getThemeColor(theme: ChartTheme, index: number): string {
  return theme.colors[index % theme.colors.length];
}

/**
 * Get all available theme names
 */
export function getAvailableThemes(): string[] {
  return Object.keys(CHART_THEMES);
}

/**
 * Get theme metadata for UI display
 */
export function getThemeMetadata(themeName: string) {
  const theme = getTheme(themeName);
  return {
    name: theme.name,
    displayName: theme.displayName,
    description: theme.description,
    colorCount: theme.colors.length,
    previewColors: theme.colors.slice(0, 5),
  };
}
