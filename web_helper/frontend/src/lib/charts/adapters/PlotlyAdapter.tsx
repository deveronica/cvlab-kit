import React from "react";

/**
 * Plotly Adapter
 *
 * Implements ChartAdapter interface for Plotly.js library.
 * Converts universal chart _config to Plotly.js React components.
 */

import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type {
  ChartAdapter,
  ChartType,
  UniversalChartConfig,
  ChartExportOptions,
} from '../types';

/**
 * Color palette for Plotly
 */
const DEFAULT_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
];

/**
 * Get Plotly line dash style
 */
function getLineDash(style?: 'solid' | 'dashed' | 'dotted'): string {
  switch (style) {
    case 'dashed':
      return 'dash';
    case 'dotted':
      return 'dot';
    default:
      return 'solid';
  }
}

/**
 * Plotly Chart Component with memoization
 */
function PlotlyChart({ _config }: { _config: UniversalChartConfig }) {
  const {
    type,
    data,
    series,
    xAxis,
    yAxis,
    legend: legendConfig,
    height = 400,
    colors = DEFAULT_COLORS,
    responsive = true,
  } = _config;

  // Validate data
  if (!data || data.length === 0) {
    return (
      <div className="p-4 text-yellow-600 dark:text-yellow-400 border border-yellow-300 dark:border-yellow-800 rounded bg-yellow-50 dark:bg-yellow-950/20">
        <strong>Plotly Adapter Warning:</strong> No data provided to render chart.
        <br />
        <span className="text-sm">Please provide a valid data array with at least one element.</span>
      </div>
    );
  }

  // Validate series
  if (!series || series.length === 0) {
    console.warn('Plotly Adapter: No series configuration provided. Please provide at least one series with a dataKey.');
    return null;
  }

  // Build Plotly data and _layout (memoized)
  const { _plotlyData, _layout: baseLayout } = useMemo(() => {
    let _plotlyData: any[];
    let _layout: any;

    switch (type) {
      case 'line':
        return buildLineChart(_config, colors);
      case 'bar':
        return buildBarChart(_config, colors);
      case 'area':
        return buildAreaChart(_config, colors);
      case 'scatter':
        return buildScatterChart(_config, colors);
      case 'pie':
        return buildPieChart(_config, colors);
      default:
        return { _plotlyData: [], _layout: {} };
    }
  }, [type, data, series, xAxis, yAxis, colors]);

  // Unsupported chart type
  if (_plotlyData.length === 0) {
    return (
      <div className="p-4 text-red-500 border border-red-300 rounded">
        <strong>Plotly Adapter Error:</strong> Unsupported chart type "{type}".
        Supported types: line, bar, area, scatter, pie.
      </div>
    );
  }

  // Common _layout settings (memoized)
  const finalLayout = useMemo(() => {
    // Map position to Plotly legend coordinates
    const position = legendConfig?.position || 'top';
    let legendY = 1.02;
    let legendYAnchor: 'top' | 'bottom' | 'middle' = 'bottom';
    let legendOrientation: 'v' | 'h' = 'h';

    switch (position) {
      case 'top':
        legendY = 1.02;
        legendYAnchor = 'bottom';
        legendOrientation = 'h';
        break;
      case 'bottom':
        legendY = -0.15;
        legendYAnchor = 'top';
        legendOrientation = 'h';
        break;
      case 'left':
        legendY = 0.5;
        legendYAnchor = 'middle';
        legendOrientation = 'v';
        break;
      case 'right':
        legendY = 0.5;
        legendYAnchor = 'middle';
        legendOrientation = 'v';
        break;
    }

    // Map align to x position
    const align = legendConfig?.align || 'center';
    let legendX = 0.5;
    let legendXAnchor: 'left' | 'center' | 'right' = 'center';

    if (position === 'left') {
      legendX = -0.15;
      legendXAnchor = 'left';
    } else if (position === 'right') {
      legendX = 1.02;
      legendXAnchor = 'left';
    } else {
      // For top/bottom, use align setting
      if (align === 'start') {
        legendX = 0;
        legendXAnchor = 'left';
      } else if (align === 'end') {
        legendX = 1;
        legendXAnchor = 'right';
      } else {
        legendX = 0.5;
        legendXAnchor = 'center';
      }
    }

    return {
      ...baseLayout,
      // Remove fixed height - let container control size
      autosize: true,
      showlegend: legendConfig?.show !== false,
      legend: {
        orientation: legendOrientation,
        x: legendX,
        xanchor: legendXAnchor,
        y: legendY,
        yanchor: legendYAnchor,
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {
        color: 'currentColor',
      },
    };
  }, [baseLayout, height, responsive, legendConfig]);

  // Config for Plotly (memoized)
  const plotlyConfig = useMemo(() => ({
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
  }), [responsive]);

  // Use absolute positioning wrapper to ensure consistent sizing
  const containerStyle: React.CSSProperties = {
    position: 'relative',
    width: '100%',
    height: typeof height === 'number' ? `${height}px` : height || '400px',
  };

  const innerStyle: React.CSSProperties = {
    position: 'absolute',
    inset: 0,
  };

  return (
    <div style={containerStyle}>
      <div style={innerStyle}>
        <Plot
          data={_plotlyData}
          _layout={finalLayout}
          _config={plotlyConfig}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </div>
    </div>
  );
}

/**
 * Build line chart data and layout
 */
function buildLineChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series, xAxis, yAxis } = _config;
    const xKey = xAxis?.dataKey || 'name';

    const _plotlyData = series
      .filter(s => s.visible !== false)
      .map((s, idx) => ({
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: s.name || s.dataKey,
        x: data.map(d => d[xKey]),
        y: data.map(d => d[s.dataKey]),
        line: {
          color: s.color || colors[idx % colors.length],
          dash: getLineDash(s.strokeStyle),
        },
        marker: {
          size: 6,
        },
      }));

    const _layout = {
      xaxis: {
        title: xAxis?.label || '',
        showgrid: xAxis?.showGrid !== false,
        gridcolor: 'rgba(128, 128, 128, 0.2)',
      },
      yaxis: {
        type: yAxis?.scale || 'linear',
        title: yAxis?.label || '',
        showgrid: yAxis?.showGrid !== false,
        gridcolor: 'rgba(128, 128, 128, 0.2)',
      },
      margin: { l: 50, r: 30, t: 30, b: 50 },
    };

    return { _plotlyData, _layout };
  }

/**
 * Build bar chart data and layout
 */
function buildBarChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series, xAxis, yAxis } = _config;
    const xKey = xAxis?.dataKey || 'name';

    const _plotlyData = series
      .filter(s => s.visible !== false)
      .map((s, idx) => ({
        type: 'bar' as const,
        name: s.name || s.dataKey,
        x: data.map(d => d[xKey]),
        y: data.map(d => d[s.dataKey]),
        marker: {
          color: s.color || colors[idx % colors.length],
        },
      }));

    const _layout = {
      xaxis: {
        title: xAxis?.label || '',
      },
      yaxis: {
        type: yAxis?.scale || 'linear',
        title: yAxis?.label || '',
        gridcolor: 'rgba(128, 128, 128, 0.2)',
      },
      margin: { l: 50, r: 30, t: 30, b: 50 },
    };

    return { _plotlyData, _layout };
  }

/**
 * Build area chart data and layout
 */
function buildAreaChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series, xAxis, yAxis } = _config;
    const xKey = xAxis?.dataKey || 'name';

    const _plotlyData = series
      .filter(s => s.visible !== false)
      .map((s, idx) => ({
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: s.name || s.dataKey,
        x: data.map(d => d[xKey]),
        y: data.map(d => d[s.dataKey]),
        fill: 'tonexty' as const,
        fillcolor: `${s.color || colors[idx % colors.length]}40`,
        line: {
          color: s.color || colors[idx % colors.length],
        },
      }));

    const _layout = {
      xaxis: {
        title: xAxis?.label || '',
        showgrid: xAxis?.showGrid !== false,
        gridcolor: 'rgba(128, 128, 128, 0.2)',
      },
      yaxis: {
        type: yAxis?.scale || 'linear',
        title: yAxis?.label || '',
        showgrid: yAxis?.showGrid !== false,
        gridcolor: 'rgba(128, 128, 128, 0.2)',
      },
      margin: { l: 50, r: 30, t: 30, b: 50 },
    };

    return { _plotlyData, _layout };
  }

/**
 * Build scatter chart data and layout
 */
function buildScatterChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series, xAxis, yAxis } = _config;
    const xKey = xAxis?.dataKey || 'x';

    const _plotlyData = series
      .filter(s => s.visible !== false)
      .map((s, idx) => ({
        type: 'scatter' as const,
        mode: 'markers' as const,
        name: s.name || s.dataKey,
        x: data.map(d => d[xKey]),
        y: data.map(d => d[s.dataKey]),
        marker: {
          size: 8,
          color: s.color || colors[idx % colors.length],
        },
      }));

    const _layout = {
      xaxis: {
        title: xAxis?.label || '',
        showgrid: xAxis?.showGrid !== false,
        gridcolor: 'rgba(128, 128, 128, 0.2)',
      },
      yaxis: {
        type: yAxis?.scale || 'linear',
        title: yAxis?.label || '',
        showgrid: yAxis?.showGrid !== false,
        gridcolor: 'rgba(128, 128, 128, 0.2)',
      },
      margin: { l: 50, r: 30, t: 30, b: 50 },
    };

    return { _plotlyData, _layout };
  }

/**
 * Build pie chart data and layout
 */
function buildPieChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series } = _config;
    const dataKey = series[0]?.dataKey || 'value';

    const _plotlyData = [
      {
        type: 'pie' as const,
        labels: data.map(d => String(d.name || '')),
        values: data.map(d => d[dataKey]),
        marker: {
          colors: colors,
        },
      },
    ];

    const _layout = {
      margin: { l: 20, r: 20, t: 20, b: 20 },
    };

    return { _plotlyData, _layout };
  }

export class PlotlyAdapter implements ChartAdapter {
  readonly name = 'plotly' as const;
  readonly supportedTypes: ChartType[] = ['line', 'bar', 'area', 'scatter', 'pie'];

  supportsType(type: ChartType): boolean {
    return this.supportedTypes.includes(type);
  }

  render(_config: UniversalChartConfig): React.ReactElement {
    return <PlotlyChart _config={_config} />;
  }

  async export(_config: UniversalChartConfig, _options: ChartExportOptions): Promise<void> {
    // Plotly has built-in export via modebar
    throw new Error('Use Plotly modebar for exports (camera icon in chart)');
  }
}

// Export singleton instance
export const plotlyAdapter = new PlotlyAdapter();
