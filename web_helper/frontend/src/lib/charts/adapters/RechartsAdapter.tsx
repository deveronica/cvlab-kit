import React from "react";

/**
 * Recharts Adapter
 *
 * Implements ChartAdapter interface for Recharts library.
 * Converts universal chart _config to Recharts components.
 */

import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type {
  ChartAdapter,
  ChartType,
  UniversalChartConfig,
  ChartExportOptions,
} from '../types';

/**
 * Color palette for Recharts
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
 * Get stroke dash array for line styles
 */
function getStrokeDashArray(style?: 'solid' | 'dashed' | 'dotted'): string | undefined {
  switch (style) {
    case 'dashed':
      return '5 5';
    case 'dotted':
      return '1 4';
    default:
      return undefined;
  }
}

/**
 * Format axis tick values
 */
function formatTick(value: any, format?: (v: any) => string): string {
  if (format) {
    return format(value);
  }
  if (typeof value === 'number') {
    // Format large numbers with K/M suffix
    if (Math.abs(value) >= 1e6) {
      return `${(value / 1e6).toFixed(1)}M`;
    }
    if (Math.abs(value) >= 1e3) {
      return `${(value / 1e3).toFixed(1)}K`;
    }
    // Format decimals
    if (value !== Math.floor(value)) {
      return value.toFixed(2);
    }
  }
  return String(value);
}

export class RechartsAdapter implements ChartAdapter {
  readonly name = 'recharts' as const;
  readonly supportedTypes: ChartType[] = ['line', 'bar', 'area', 'scatter', 'pie', 'radar'];

  supportsType(type: ChartType): boolean {
    return this.supportedTypes.includes(type);
  }

  render(_config: UniversalChartConfig): React.ReactElement {
    const {
      type,
      data,
      series,
      xAxis,
      yAxis,
      legend: _legendConfig,
      tooltip: _tooltipConfig,
      _interactions,
      height = 400,
      _theme = 'light',
      colors = DEFAULT_COLORS,
      animation = true,
      responsive = true,
    } = _config;

    // Validate data
    if (!data || data.length === 0) {
      return (
        <div className="p-4 text-yellow-600 dark:text-yellow-400 border border-yellow-300 dark:border-yellow-800 rounded bg-yellow-50 dark:bg-yellow-950/20">
          <strong>Recharts Adapter Warning:</strong> No data provided to render chart.
          <br />
          <span className="text-sm">Please provide a valid data array with at least one element.</span>
        </div>
      );
    }

    // Validate series
    if (!series || series.length === 0) {
      console.warn('Recharts Adapter: No series configuration provided. Please provide at least one series with a dataKey.', {
        type,
        dataLength: data?.length || 0,
        series,
        stackTrace: new Error().stack
      });
      return null;
    }

    // Common props for all chart types
    const commonProps = {
      data,
      margin: { top: 20, right: 30, left: 20, bottom: 60 },
    };

    // Render functions for each chart type
    const renderChart = () => {
      switch (type) {
        case 'line':
          return this.renderLineChart(_config, commonProps, colors, animation);
        case 'bar':
          return this.renderBarChart(_config, commonProps, colors, animation);
        case 'area':
          return this.renderAreaChart(_config, commonProps, colors, animation);
        case 'scatter':
          return this.renderScatterChart(_config, commonProps, colors);
        case 'pie':
          return this.renderPieChart(_config, colors);
        case 'radar':
          return this.renderRadarChart(_config, colors);
        default:
          return <div className="text-red-500">Unsupported chart type: {type}</div>;
      }
    };

    // Wrap in ResponsiveContainer if responsive
    if (responsive) {
      return (
        <ResponsiveContainer width="100%" height={height}>
          {renderChart()}
        </ResponsiveContainer>
      );
    }

    return renderChart();
  }

  private renderLineChart(
    _config: UniversalChartConfig,
    commonProps: any,
    colors: string[],
    animation: boolean
  ): React.ReactElement {
    const { series, xAxis, yAxis, legend, tooltip, _interactions, libraryOptions } = _config;

    // Extract library-specific options
    const strokeWidth = libraryOptions?.strokeWidth || 2;
    const showDots = libraryOptions?.showDots !== false;
    const curveType = libraryOptions?.curveType || 'monotone';

    return (
      <LineChart {...commonProps}>
        {(xAxis?.showGrid !== false || yAxis?.showGrid !== false) && (
          <CartesianGrid
            strokeDasharray="3 3"
            horizontal={yAxis?.showGrid !== false}
            vertical={xAxis?.showGrid !== false}
          />
        )}
        <XAxis
          dataKey={xAxis?.dataKey || 'name'}
          label={xAxis?.label ? { value: xAxis.label, position: 'bottom', offset: 0 } : undefined}
          tick={{ fill: 'currentColor', fontSize: 12 }}
          tickFormatter={(v) => formatTick(v, xAxis?.tickFormat)}
          height={60}
        />
        <YAxis
          label={yAxis?.label ? { value: yAxis.label, angle: -90, position: 'insideLeft', offset: 10 } : undefined}
          tick={{ fill: 'currentColor' }}
          domain={yAxis?.domain}
          scale={yAxis?.scale || 'linear'}
          tickFormatter={(v) => formatTick(v, yAxis?.tickFormat)}
          width={80}
        />
        {tooltip?.show !== false && (
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
            }}
            formatter={(value: any, name: string) => {
              const seriesConfig = series.find(s => s.dataKey === name);
              return [value, seriesConfig?.name || name];
            }}
          />
        )}
        {legend?.show !== false && (
          <Legend
            layout={legend?.position === 'left' || legend?.position === 'right' ? 'vertical' : 'horizontal'}
            verticalAlign={
              legend?.position === 'left' || legend?.position === 'right' ? 'middle'
              : legend?.position === 'bottom' ? 'bottom'
              : 'top'
            }
            align={
              legend?.position === 'left' ? 'left'
              : legend?.position === 'right' ? 'right'
              : legend?.align === 'start' ? 'left'
              : legend?.align === 'end' ? 'right'
              : 'center'
            }
            formatter={(value) => {
              const seriesConfig = series.find(s => s.dataKey === value);
              return seriesConfig?.name || value;
            }}
          />
        )}
        {series
          .filter(s => s.visible !== false)
          .map((s, idx) => (
            <Line
              key={s.dataKey}
              type={curveType as any}
              dataKey={s.dataKey}
              name={s.name || s.dataKey}
              stroke={s.color || colors[idx % colors.length]}
              strokeWidth={strokeWidth}
              strokeDasharray={getStrokeDashArray(s.strokeStyle)}
              dot={showDots ? { r: 3 } : false}
              activeDot={showDots ? { r: 5 } : false}
              isAnimationActive={animation}
              onClick={(data: any) => _interactions?.onClick?.(data, s.dataKey)}
            />
          ))}
      </LineChart>
    );
  }

  private renderBarChart(
    _config: UniversalChartConfig,
    commonProps: any,
    colors: string[],
    animation: boolean
  ): React.ReactElement {
    const { series, xAxis, yAxis, legend, tooltip, _interactions, libraryOptions } = _config;
    const stackMode = libraryOptions?.stackMode || 'none';
    const stackId = stackMode !== 'none' ? 'stack' : undefined;

    return (
      <BarChart {...commonProps} stackOffset={stackMode === 'percent' ? 'expand' : undefined}>
        {(xAxis?.showGrid !== false || yAxis?.showGrid !== false) && (
          <CartesianGrid
            strokeDasharray="3 3"
            horizontal={yAxis?.showGrid !== false}
            vertical={xAxis?.showGrid !== false}
          />
        )}
        <XAxis
          dataKey={xAxis?.dataKey || 'name'}
          label={xAxis?.label ? { value: xAxis.label, position: 'insideBottom', offset: -10 } : undefined}
          tick={{ fill: 'currentColor', fontSize: 12 }}
          angle={-45}
          textAnchor="end"
          height={80}
          tickFormatter={(v) => formatTick(v, xAxis?.tickFormat)}
        />
        <YAxis
          label={yAxis?.label ? { value: yAxis.label, angle: -90, position: 'insideLeft', offset: 10 } : undefined}
          tick={{ fill: 'currentColor' }}
          domain={yAxis?.domain}
          scale={yAxis?.scale || 'linear'}
          tickFormatter={(v) => formatTick(v, yAxis?.tickFormat)}
          width={80}
        />
        {tooltip?.show !== false && (
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
            }}
          />
        )}
        {legend?.show !== false && (
          <Legend
            layout={legend?.position === 'left' || legend?.position === 'right' ? 'vertical' : 'horizontal'}
            verticalAlign={
              legend?.position === 'left' || legend?.position === 'right' ? 'middle'
              : legend?.position === 'bottom' ? 'bottom'
              : 'top'
            }
            align={
              legend?.position === 'left' ? 'left'
              : legend?.position === 'right' ? 'right'
              : legend?.align === 'start' ? 'left'
              : legend?.align === 'end' ? 'right'
              : 'center'
            }
            wrapperStyle={
              legend?.position === 'bottom'
                ? { paddingTop: '20px' }
                : undefined
            }
          />
        )}
        {series
          .filter(s => s.visible !== false)
          .map((s, idx) => (
            <Bar
              key={s.dataKey}
              dataKey={s.dataKey}
              name={s.name || s.dataKey}
              fill={s.color || colors[idx % colors.length]}
              isAnimationActive={animation}
              onClick={(data: any) => _interactions?.onClick?.(data, s.dataKey)}
              stackId={stackId}
            />
          ))}
      </BarChart>
    );
  }

  private renderAreaChart(
    _config: UniversalChartConfig,
    commonProps: any,
    colors: string[],
    animation: boolean
  ): React.ReactElement {
    const { series, xAxis, yAxis, legend, tooltip, libraryOptions } = _config;

    // Extract library-specific options
    const fillOpacity = libraryOptions?.fillOpacity !== undefined ? libraryOptions.fillOpacity : 0.6;
    const curveType = libraryOptions?.curveType || 'monotone';
    const strokeWidth = libraryOptions?.strokeWidth || 2;
    const stackMode = libraryOptions?.stackMode || 'none';
    const stackId = stackMode !== 'none' ? 'stack' : undefined;

    return (
      <AreaChart {...commonProps} stackOffset={stackMode === 'percent' ? 'expand' : undefined}>
        {(xAxis?.showGrid !== false || yAxis?.showGrid !== false) && (
          <CartesianGrid
            strokeDasharray="3 3"
            horizontal={yAxis?.showGrid !== false}
            vertical={xAxis?.showGrid !== false}
          />
        )}
        <XAxis
          dataKey={xAxis?.dataKey || 'name'}
          label={xAxis?.label ? { value: xAxis.label, position: 'bottom', offset: 0 } : undefined}
          tick={{ fill: 'currentColor' }}
          tickFormatter={(v) => formatTick(v, xAxis?.tickFormat)}
          height={60}
        />
        <YAxis
          label={yAxis?.label ? { value: yAxis.label, angle: -90, position: 'insideLeft', offset: 10 } : undefined}
          tick={{ fill: 'currentColor' }}
          domain={yAxis?.domain}
          scale={yAxis?.scale || 'linear'}
          tickFormatter={(v) => formatTick(v, yAxis?.tickFormat)}
          width={80}
        />
        {tooltip?.show !== false && <Tooltip />}
        {legend?.show !== false && (
          <Legend
            layout={legend?.position === 'left' || legend?.position === 'right' ? 'vertical' : 'horizontal'}
            verticalAlign={
              legend?.position === 'left' || legend?.position === 'right' ? 'middle'
              : legend?.position === 'bottom' ? 'bottom'
              : 'top'
            }
            align={
              legend?.position === 'left' ? 'left'
              : legend?.position === 'right' ? 'right'
              : legend?.align === 'start' ? 'left'
              : legend?.align === 'end' ? 'right'
              : 'center'
            }
          />
        )}
        {series
          .filter(s => s.visible !== false)
          .map((s, idx) => (
            <Area
              key={s.dataKey}
              type={curveType as any}
              dataKey={s.dataKey}
              name={s.name || s.dataKey}
              stroke={s.color || colors[idx % colors.length]}
              strokeWidth={strokeWidth}
              fill={s.color || colors[idx % colors.length]}
              fillOpacity={fillOpacity}
              isAnimationActive={animation}
              stackId={stackId}
            />
          ))}
      </AreaChart>
    );
  }

  private renderScatterChart(
    _config: UniversalChartConfig,
    commonProps: any,
    colors: string[]
  ): React.ReactElement {
    const { series, xAxis, yAxis, legend, tooltip, libraryOptions } = _config;
    const dotSize = libraryOptions?.dotSize || 4;

    return (
      <ScatterChart {...commonProps}>
        {(xAxis?.showGrid !== false || yAxis?.showGrid !== false) && (
          <CartesianGrid
            horizontal={yAxis?.showGrid !== false}
            vertical={xAxis?.showGrid !== false}
          />
        )}
        <XAxis
          type="number"
          dataKey={xAxis?.dataKey || 'x'}
          name={xAxis?.label}
          label={xAxis?.label ? { value: xAxis.label, position: 'insideBottom', offset: -5 } : undefined}
          tick={{ fill: 'currentColor' }}
          tickFormatter={(v) => formatTick(v, xAxis?.tickFormat)}
          height={60}
        />
        <YAxis
          type="number"
          dataKey={yAxis?.dataKey || 'y'}
          name={yAxis?.label}
          label={yAxis?.label ? { value: yAxis.label, angle: -90, position: 'insideLeft', offset: 10 } : undefined}
          tick={{ fill: 'currentColor' }}
          domain={yAxis?.domain}
          scale={yAxis?.scale || 'linear'}
          tickFormatter={(v) => formatTick(v, yAxis?.tickFormat)}
          width={80}
        />
        {tooltip?.show !== false && <Tooltip cursor={{ strokeDasharray: '3 3' }} />}
        {legend?.show !== false && (
          <Legend
            layout={legend?.position === 'left' || legend?.position === 'right' ? 'vertical' : 'horizontal'}
            verticalAlign={
              legend?.position === 'left' || legend?.position === 'right' ? 'middle'
              : legend?.position === 'bottom' ? 'bottom'
              : 'top'
            }
            align={
              legend?.position === 'left' ? 'left'
              : legend?.position === 'right' ? 'right'
              : legend?.align === 'start' ? 'left'
              : legend?.align === 'end' ? 'right'
              : 'center'
            }
          />
        )}
        {series
          .filter(s => s.visible !== false)
          .map((s, idx) => (
            <Scatter
              key={s.dataKey}
              name={s.name || s.dataKey}
              data={_config.data}
              fill={s.color || colors[idx % colors.length]}
              shape="circle"
              r={dotSize}
            />
          ))}
      </ScatterChart>
    );
  }

  private renderPieChart(_config: UniversalChartConfig, colors: string[]): React.ReactElement {
    const { data, series, legend, tooltip } = _config;
    const dataKey = series[0]?.dataKey || 'value';
    const nameKey = 'name';

    return (
      <PieChart>
        {tooltip?.show !== false && <Tooltip />}
        {legend?.show !== false && (
          <Legend
            layout={legend?.position === 'left' || legend?.position === 'right' ? 'vertical' : 'horizontal'}
            verticalAlign={
              legend?.position === 'left' || legend?.position === 'right' ? 'middle'
              : legend?.position === 'bottom' ? 'bottom'
              : 'top'
            }
            align={
              legend?.position === 'left' ? 'left'
              : legend?.position === 'right' ? 'right'
              : legend?.align === 'start' ? 'left'
              : legend?.align === 'end' ? 'right'
              : 'center'
            }
          />
        )}
        <Pie
          data={data}
          dataKey={dataKey}
          nameKey={nameKey}
          cx="50%"
          cy="50%"
          outerRadius={80}
          label
        >
          {data.map((_entry, index) => (
            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
          ))}
        </Pie>
      </PieChart>
    );
  }

  private renderRadarChart(_config: UniversalChartConfig, colors: string[]): React.ReactElement {
    const { series, legend, tooltip } = _config;

    return (
      <RadarChart cx="50%" cy="50%" outerRadius="80%" data={_config.data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="subject" />
        <PolarRadiusAxis />
        {tooltip?.show !== false && <Tooltip />}
        {legend?.show !== false && (
          <Legend
            layout={legend?.position === 'left' || legend?.position === 'right' ? 'vertical' : 'horizontal'}
            verticalAlign={
              legend?.position === 'left' || legend?.position === 'right' ? 'middle'
              : legend?.position === 'bottom' ? 'bottom'
              : 'top'
            }
            align={
              legend?.position === 'left' ? 'left'
              : legend?.position === 'right' ? 'right'
              : legend?.align === 'start' ? 'left'
              : legend?.align === 'end' ? 'right'
              : 'center'
            }
          />
        )}
        {series
          .filter(s => s.visible !== false)
          .map((s, idx) => (
            <Radar
              key={s.dataKey}
              name={s.name || s.dataKey}
              dataKey={s.dataKey}
              stroke={s.color || colors[idx % colors.length]}
              fill={s.color || colors[idx % colors.length]}
              fillOpacity={0.6}
            />
          ))}
      </RadarChart>
    );
  }

  async export(_config: UniversalChartConfig, _options: ChartExportOptions): Promise<void> {
    // Export implementation would go here
    // Can reuse existing chart-export.ts functions
    throw new Error('Export not yet implemented for Recharts adapter');
  }
}

// Export singleton instance
export const rechartsAdapter = new RechartsAdapter();
