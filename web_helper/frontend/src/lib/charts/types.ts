import React from "react";
/**
 * Universal Chart Types
 *
 * Common interfaces for chart abstraction layer.
 * Supports multiple chart libraries (Recharts, Victory, Nivo, ECharts, etc.)
 */

/**
 * Supported chart library renderers
 * - recharts: React + D3, most popular (7.9M weekly downloads)
 * - chartjs: Chart.js wrapper, stable and performant (2.2M weekly downloads)
 * - plotly: Scientific visualization, 3D charts (284K weekly downloads)
 */
export type ChartRenderer = 'recharts' | 'chartjs' | 'plotly';

/**
 * Chart types supported by the abstraction layer
 */
export type ChartType =
  | 'line'
  | 'bar'
  | 'area'
  | 'scatter'
  | 'pie'
  | 'radar'
  | 'heatmap';

/**
 * Common data point structure
 * All adapters normalize data to this format
 */
export interface DataPoint {
  [key: string]: string | number | null | undefined;
}

/**
 * Series configuration for multi-series charts
 */
export interface SeriesConfig {
  /** Unique key for the series (maps to data field) */
  dataKey: string;
  /** Display name for the series */
  name?: string;
  /** Color for this series */
  color?: string;
  /** Series type (for mixed charts) */
  type?: 'line' | 'bar' | 'area';
  /** Whether to show this series */
  visible?: boolean;
  /** Line style (solid, dashed, dotted) */
  strokeStyle?: 'solid' | 'dashed' | 'dotted';
  /** Point marker shape */
  pointShape?: 'circle' | 'square' | 'triangle' | 'diamond';
}

/**
 * Axis configuration
 */
export interface AxisConfig {
  /** Axis label */
  label?: string;
  /** Data key for this axis */
  dataKey?: string;
  /** Axis type */
  type?: 'number' | 'category' | 'time';
  /** Domain (min, max) */
  domain?: [number | 'auto', number | 'auto'];
  /** Tick format function */
  tickFormat?: (value: any) => string;
  /** Show grid lines */
  showGrid?: boolean;
  /** Scale type */
  scale?: 'linear' | 'log' | 'sqrt';
}

/**
 * Legend configuration
 */
export interface LegendConfig {
  /** Show legend */
  show?: boolean;
  /** Legend position */
  position?: 'top' | 'right' | 'bottom' | 'left';
  /** Alignment */
  align?: 'start' | 'center' | 'end';
  /** Orientation */
  orientation?: 'horizontal' | 'vertical';
}

/**
 * Tooltip configuration
 */
export interface TooltipConfig {
  /** Show tooltip */
  show?: boolean;
  /** Shared tooltip (show all series) */
  shared?: boolean;
  /** Custom formatter */
  formatter?: (data: any) => React.ReactNode;
}

/**
 * Interaction handlers
 */
export interface ChartInteractionHandlers {
  /** Click handler */
  onClick?: (data: DataPoint, seriesKey?: string) => void;
  /** Hover handler */
  onHover?: (data: DataPoint | null, seriesKey?: string) => void;
  /** Brush/zoom handler */
  onBrush?: (domain: [number, number]) => void;
}

/**
 * Universal chart configuration
 * This is the unified interface that all adapters must support
 */
export interface UniversalChartConfig {
  /** Chart type */
  type: ChartType;

  /** Data array */
  data: DataPoint[];

  /** Series configurations */
  series: SeriesConfig[];

  /** X-axis configuration */
  xAxis?: AxisConfig;

  /** Y-axis configuration */
  yAxis?: AxisConfig;

  /** Legend configuration */
  legend?: LegendConfig;

  /** Tooltip configuration */
  tooltip?: TooltipConfig;

  /** Interaction handlers */
  interactions?: ChartInteractionHandlers;

  /** Chart title */
  title?: string;

  /** Chart description */
  description?: string;

  /** Chart width (if not responsive) */
  width?: number | string;

  /** Chart height */
  height?: number | string;

  /** Theme mode */
  theme?: 'light' | 'dark' | 'auto';

  /** Color palette */
  colors?: string[];

  /** Animation enabled */
  animation?: boolean;

  /** Responsive behavior */
  responsive?: boolean;

  /** Custom styles */
  style?: React.CSSProperties;

  /** Additional library-specific options (escape hatch) */
  libraryOptions?: Record<string, any>;
}

/**
 * Export options for charts
 */
export interface ChartExportOptions {
  /** Export format */
  format: 'png' | 'svg' | 'csv' | 'json';
  /** Filename (without extension) */
  filename?: string;
  /** Image quality for PNG (0-1) */
  quality?: number;
  /** Resolution scale multiplier */
  scale?: number;
  /** Background color */
  backgroundColor?: string;
}

/**
 * Chart adapter interface
 * All chart library adapters must implement this interface
 */
export interface ChartAdapter {
  /** Adapter name */
  readonly name: ChartRenderer;

  /** Supported chart types */
  readonly supportedTypes: ChartType[];

  /**
   * Render the chart
   * @param config - Universal chart configuration
   * @returns React component
   */
  render(config: UniversalChartConfig): React.ReactElement;

  /**
   * Export the chart
   * @param config - Chart configuration
   * @param options - Export options
   * @returns Promise that resolves when export completes
   */
  export?(config: UniversalChartConfig, options: ChartExportOptions): Promise<void>;

  /**
   * Check if chart type is supported
   * @param type - Chart type to check
   * @returns true if supported
   */
  supportsType(type: ChartType): boolean;
}

/**
 * Adapter factory return type
 */
export type AdapterFactory = () => ChartAdapter;

/**
 * Chart preferences (stored in localStorage)
 */
export interface ChartPreferences {
  /** Default renderer */
  defaultRenderer: ChartRenderer;
  /** Per-chart-type renderer preferences */
  rendererByType?: Partial<Record<ChartType, ChartRenderer>>;
  /** Animation enabled */
  animationEnabled?: boolean;
  /** Default theme */
  theme?: 'light' | 'dark' | 'auto';
}

/**
 * Chart Renderer Compatibility Matrix
 * Defines which chart types are supported by each renderer
 */
export const RENDERER_SUPPORT: Record<ChartRenderer, ChartType[]> = {
  recharts: ['line', 'bar', 'area', 'scatter', 'pie', 'radar'],
  chartjs: ['line', 'bar', 'area', 'scatter', 'pie'],
  plotly: ['line', 'bar', 'area', 'scatter', 'pie'],
};

/**
 * Get chart types supported by a specific renderer
 */
export function getSupportedChartTypes(renderer: ChartRenderer): ChartType[] {
  return RENDERER_SUPPORT[renderer] || [];
}

/**
 * Get renderers that support a specific chart type
 */
export function getSupportedRenderers(chartType: ChartType): ChartRenderer[] {
  return (Object.keys(RENDERER_SUPPORT) as ChartRenderer[]).filter(
    (renderer) => RENDERER_SUPPORT[renderer].includes(chartType)
  );
}

/**
 * Check if a renderer supports a chart type
 */
export function isCompatible(renderer: ChartRenderer, chartType: ChartType): boolean {
  return RENDERER_SUPPORT[renderer]?.includes(chartType) || false;
}

/**
 * Get the first compatible renderer for a chart type
 * Useful for auto-selecting when current renderer is incompatible
 */
export function getFirstCompatibleRenderer(chartType: ChartType): ChartRenderer | null {
  const compatibleRenderers = getSupportedRenderers(chartType);
  return compatibleRenderers.length > 0 ? compatibleRenderers[0] : null;
}

/**
 * Get the first compatible chart type for a renderer
 * Useful for auto-selecting when current chart type is incompatible
 */
export function getFirstCompatibleChartType(renderer: ChartRenderer): ChartType | null {
  const compatibleTypes = getSupportedChartTypes(renderer);
  return compatibleTypes.length > 0 ? compatibleTypes[0] : null;
}
