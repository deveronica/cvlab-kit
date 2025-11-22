/**
 * Chart Settings Store
 *
 * Manages per-chart settings with localStorage persistence
 * Each chart type has its own customizable settings
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Common settings available to all chart types
 */
interface BaseChartSettings {
  colorScheme?: 'default' | 'nord' | 'warm' | 'cool' | 'monochrome';
  showLegend?: boolean;
  showGrid?: boolean;
  fontSize?: 'xs' | 'sm' | 'base' | 'lg';
}

/**
 * Histogram-specific settings
 */
export interface HistogramSettings extends BaseChartSettings {
  binCount?: number;
  showDensity?: boolean;
  showMean?: boolean;
  showMedian?: boolean;
}

/**
 * Scatter plot settings
 */
export interface ScatterSettings extends BaseChartSettings {
  pointSize?: number;
  showTrendLine?: boolean;
  showCorrelation?: boolean;
  opacity?: number;
}

/**
 * Line chart settings
 */
export interface LineChartSettings extends BaseChartSettings {
  lineWidth?: number;
  showPoints?: boolean;
  smoothing?: boolean;
  fillArea?: boolean;
}

/**
 * Bar chart settings
 */
export interface BarChartSettings extends BaseChartSettings {
  orientation?: 'vertical' | 'horizontal';
  stacked?: boolean;
  showValues?: boolean;
  barWidth?: number;
}

/**
 * Heatmap settings
 */
export interface HeatmapSettings extends BaseChartSettings {
  interpolation?: 'linear' | 'step' | 'basis';
  showValues?: boolean;
  minColor?: string;
  maxColor?: string;
}

/**
 * Box plot settings
 */
export interface BoxPlotSettings extends BaseChartSettings {
  showOutliers?: boolean;
  showMean?: boolean;
  notched?: boolean;
  orientation?: 'vertical' | 'horizontal';
}

/**
 * All chart types
 */
export type ChartType =
  | 'histogram'
  | 'scatter'
  | 'line'
  | 'bar'
  | 'heatmap'
  | 'boxplot'
  | 'parallel-coordinates'
  | 'metric-distribution'
  | 'run-timeline'
  | 'correlation';

/**
 * Union type for all chart settings
 */
export type ChartSettings =
  | HistogramSettings
  | ScatterSettings
  | LineChartSettings
  | BarChartSettings
  | HeatmapSettings
  | BoxPlotSettings
  | BaseChartSettings;

/**
 * Default settings for each chart type
 */
const DEFAULT_SETTINGS: Record<ChartType, ChartSettings> = {
  histogram: {
    binCount: 20,
    colorScheme: 'nord',
    showLegend: true,
    showGrid: true,
    showDensity: false,
    showMean: true,
    showMedian: false,
    fontSize: 'sm',
  } as HistogramSettings,
  scatter: {
    pointSize: 4,
    colorScheme: 'nord',
    showLegend: true,
    showGrid: true,
    showTrendLine: false,
    showCorrelation: true,
    opacity: 0.7,
    fontSize: 'sm',
  } as ScatterSettings,
  line: {
    lineWidth: 2,
    colorScheme: 'nord',
    showLegend: true,
    showGrid: true,
    showPoints: false,
    smoothing: false,
    fillArea: false,
    fontSize: 'sm',
  } as LineChartSettings,
  bar: {
    orientation: 'vertical',
    colorScheme: 'nord',
    showLegend: true,
    showGrid: true,
    stacked: false,
    showValues: false,
    barWidth: 20,
    fontSize: 'sm',
  } as BarChartSettings,
  heatmap: {
    interpolation: 'linear',
    colorScheme: 'nord',
    showLegend: true,
    showGrid: false,
    showValues: false,
    fontSize: 'sm',
  } as HeatmapSettings,
  boxplot: {
    showOutliers: true,
    colorScheme: 'nord',
    showLegend: true,
    showGrid: true,
    showMean: true,
    notched: false,
    orientation: 'vertical',
    fontSize: 'sm',
  } as BoxPlotSettings,
  'parallel-coordinates': {
    colorScheme: 'nord',
    showLegend: true,
    showGrid: false,
    fontSize: 'sm',
  },
  'metric-distribution': {
    colorScheme: 'nord',
    showLegend: true,
    showGrid: true,
    fontSize: 'sm',
  },
  'run-timeline': {
    colorScheme: 'nord',
    showLegend: true,
    showGrid: true,
    fontSize: 'sm',
  },
  correlation: {
    colorScheme: 'nord',
    showLegend: true,
    showGrid: true,
    fontSize: 'sm',
  },
};

/**
 * Chart settings store state
 */
interface ChartSettingsState {
  // Settings by chart type
  settings: Partial<Record<ChartType, ChartSettings>>;

  // Get settings for a specific chart type
  getSettings: <T extends ChartSettings>(chartType: ChartType) => T;

  // Update settings for a specific chart type
  updateSettings: (chartType: ChartType, settings: Partial<ChartSettings>) => void;

  // Reset settings for a specific chart type to defaults
  resetSettings: (chartType: ChartType) => void;

  // Reset all settings to defaults
  resetAllSettings: () => void;
}

/**
 * Create chart settings store with localStorage persistence
 */
export const useChartSettings = create<ChartSettingsState>()(
  persist(
    (set, get) => ({
      settings: {},

      getSettings: <T extends ChartSettings>(chartType: ChartType): T => {
        const currentSettings = get().settings[chartType];
        const defaultSettings = DEFAULT_SETTINGS[chartType];

        // Merge default settings with user settings
        return {
          ...defaultSettings,
          ...currentSettings,
        } as T;
      },

      updateSettings: (chartType: ChartType, newSettings: Partial<ChartSettings>) => {
        set((state) => ({
          settings: {
            ...state.settings,
            [chartType]: {
              ...state.settings[chartType],
              ...newSettings,
            },
          },
        }));
      },

      resetSettings: (chartType: ChartType) => {
        set((state) => ({
          settings: {
            ...state.settings,
            [chartType]: { ...DEFAULT_SETTINGS[chartType] },
          },
        }));
      },

      resetAllSettings: () => {
        // Reset all chart types to their default settings
        const resetSettings: Partial<Record<ChartType, ChartSettings>> = {};
        Object.keys(DEFAULT_SETTINGS).forEach((chartType) => {
          resetSettings[chartType as ChartType] = {
            ...DEFAULT_SETTINGS[chartType as ChartType]
          };
        });
        set({ settings: resetSettings });
      },
    }),
    {
      name: 'chart-settings-storage',
      version: 1,
    }
  )
);

/**
 * Color scheme definitions for charts
 */
export const COLOR_SCHEMES = {
  default: {
    primary: 'hsl(var(--primary))',
    secondary: 'hsl(var(--secondary))',
    accent: 'hsl(var(--accent))',
    muted: 'hsl(var(--muted))',
  },
  nord: {
    primary: '#5E81AC',
    secondary: '#88C0D0',
    accent: '#81A1C1',
    muted: '#4C566A',
    success: '#A3BE8C',
    warning: '#EBCB8B',
    error: '#BF616A',
  },
  warm: {
    primary: '#F59E0B',
    secondary: '#F97316',
    accent: '#EF4444',
    muted: '#78350F',
  },
  cool: {
    primary: '#3B82F6',
    secondary: '#06B6D4',
    accent: '#8B5CF6',
    muted: '#1E3A8A',
  },
  monochrome: {
    primary: '#525252',
    secondary: '#737373',
    accent: '#404040',
    muted: '#A3A3A3',
  },
};

/**
 * Get colors for a specific color scheme
 */
export function getColorScheme(scheme: string = 'nord'): Record<string, string> {
  return COLOR_SCHEMES[scheme as keyof typeof COLOR_SCHEMES] || COLOR_SCHEMES.nord;
}
