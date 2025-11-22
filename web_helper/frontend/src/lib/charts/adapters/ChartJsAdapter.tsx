import React from "react";

/**
 * Chart.js Adapter
 *
 * Implements ChartAdapter interface for Chart.js library via react-chartjs-2.
 * Converts universal chart _config to Chart.js components.
 */

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { Line, Bar, Pie, Scatter } from 'react-chartjs-2';
import type {
  ChartAdapter,
  ChartType,
  UniversalChartConfig,
  ChartExportOptions,
} from '../types';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

/**
 * Color palette for Chart.js
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
 * Get line dash pattern
 */
function getLineDash(style?: 'solid' | 'dashed' | 'dotted'): number[] {
  switch (style) {
    case 'dashed':
      return [5, 5];
    case 'dotted':
      return [2, 2];
    default:
      return [];
  }
}

export class ChartJsAdapter implements ChartAdapter {
  readonly name = 'chartjs' as const;
  readonly supportedTypes: ChartType[] = ['line', 'bar', 'area', 'scatter', 'pie'];

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
      legend: legendConfig,
      height = 400,
      colors = DEFAULT_COLORS,
      responsive = true,
      animation = true,
    } = _config;

    // Validate data
    if (!data || data.length === 0) {
      return (
        <div className="p-4 text-yellow-600 dark:text-yellow-400 border border-yellow-300 dark:border-yellow-800 rounded bg-yellow-50 dark:bg-yellow-950/20">
          <strong>Chart.js Adapter Warning:</strong> No data provided to render chart.
          <br />
          <span className="text-sm">Please provide a valid data array with at least one element.</span>
        </div>
      );
    }

    // Validate series
    if (!series || series.length === 0) {
      console.warn('Chart.js Adapter: No series configuration provided. Please provide at least one series with a dataKey.');
      return null;
    }

    // Build Chart.js data and options
    let chartData: any;
    let _options: ChartOptions<any>;

    switch (type) {
      case 'line':
        ({ data: chartData, _options } = this.buildLineChart(_config, colors));
        break;
      case 'bar':
        ({ data: chartData, _options } = this.buildBarChart(_config, colors));
        break;
      case 'area':
        ({ data: chartData, _options } = this.buildAreaChart(_config, colors));
        break;
      case 'scatter':
        ({ data: chartData, _options } = this.buildScatterChart(_config, colors));
        break;
      case 'pie':
        ({ data: chartData, _options } = this.buildPieChart(_config, colors));
        break;
      default:
        return (
          <div className="p-4 text-red-500 border border-red-300 rounded">
            <strong>Chart.js Adapter Error:</strong> Unsupported chart type "{type}".
            Supported types: {this.supportedTypes.join(', ')}.
          </div>
        );
    }

    // Common options
    // For Chart.js with absolute positioning, we need to disable responsive mode
    // and let the container control the size
    const commonOptions: ChartOptions<any> = {
      ..._options,
      responsive: true,
      maintainAspectRatio: false,
      animation: animation ? undefined : false,
      plugins: {
        ..._options.plugins,
        legend: {
          display: legendConfig?.show !== false,
          position: legendConfig?.position || 'top',
          align: legendConfig?.align || 'center',
        },
      },
    };

    // Render based on chart type
    // Chart.js needs special handling: absolute positioning with explicit dimensions
    const containerStyle: React.CSSProperties = {
      position: 'relative',
      width: '100%',
      height: typeof height === 'number' ? `${height}px` : height,
    };

    // Chart.js canvas must have explicit dimensions on parent
    const canvasWrapperStyle: React.CSSProperties = {
      position: 'absolute',
      inset: 0,
      width: '100%',
      height: '100%',
    };

    switch (type) {
      case 'line':
      case 'area':
        return (
          <div style={containerStyle}>
            <div style={canvasWrapperStyle}>
              <Line data={chartData} options={commonOptions} />
            </div>
          </div>
        );
      case 'bar':
        return (
          <div style={containerStyle}>
            <div style={canvasWrapperStyle}>
              <Bar data={chartData} options={commonOptions} />
            </div>
          </div>
        );
      case 'scatter':
        return (
          <div style={containerStyle}>
            <div style={canvasWrapperStyle}>
              <Scatter data={chartData} options={commonOptions} />
            </div>
          </div>
        );
      case 'pie':
        return (
          <div style={containerStyle}>
            <div style={canvasWrapperStyle}>
              <Pie data={chartData} options={commonOptions} />
            </div>
          </div>
        );
      default:
        return <div>Unsupported chart type</div>;
    }
  }

  private buildLineChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series, xAxis, yAxis, libraryOptions } = _config;
    const xKey = xAxis?.dataKey || 'name';

    // Extract library-specific options
    const strokeWidth = libraryOptions?.strokeWidth || 2;
    const curveType = libraryOptions?.curveType || 'monotone';
    const tension = curveType === 'linear' ? 0 : curveType === 'step' ? 0 : 0.4; // linear=0, smooth=0.4
    const showDots = libraryOptions?.showDots !== false;

    const chartData = {
      labels: data.map(d => String(d[xKey] || '')),
      datasets: series
        .filter(s => s.visible !== false)
        .map((s, idx) => ({
          label: s.name || s.dataKey,
          data: data.map(d => d[s.dataKey]),
          borderColor: s.color || colors[idx % colors.length],
          backgroundColor: s.color || colors[idx % colors.length],
          borderWidth: strokeWidth,
          borderDash: getLineDash(s.strokeStyle),
          tension: tension,
          pointRadius: showDots ? 3 : 0,
          pointHoverRadius: showDots ? 5 : 0,
        })),
    };

    const _options: ChartOptions<'line'> = {
      plugins: {
        title: {
          display: false,
        },
      },
      scales: {
        x: {
          title: {
            display: !!xAxis?.label,
            text: xAxis?.label || '',
          },
          grid: {
            display: xAxis?.showGrid !== false,
          },
        },
        y: {
          type: yAxis?.scale === 'log' ? 'logarithmic' : 'linear',
          title: {
            display: !!yAxis?.label,
            text: yAxis?.label || '',
          },
          grid: {
            display: yAxis?.showGrid !== false,
          },
        },
      },
    };

    return { data: chartData, _options };
  }

  private buildBarChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series, xAxis, yAxis } = _config;
    const xKey = xAxis?.dataKey || 'name';

    const chartData = {
      labels: data.map(d => String(d[xKey] || '')),
      datasets: series
        .filter(s => s.visible !== false)
        .map((s, idx) => ({
          label: s.name || s.dataKey,
          data: data.map(d => d[s.dataKey]),
          backgroundColor: s.color || colors[idx % colors.length],
          borderColor: s.color || colors[idx % colors.length],
          borderWidth: 1,
        })),
    };

    const _options: ChartOptions<'bar'> = {
      plugins: {
        title: {
          display: false,
        },
      },
      scales: {
        x: {
          title: {
            display: !!xAxis?.label,
            text: xAxis?.label || '',
          },
        },
        y: {
          type: yAxis?.scale === 'log' ? 'logarithmic' : 'linear',
          title: {
            display: !!yAxis?.label,
            text: yAxis?.label || '',
          },
          grid: {
            display: yAxis?.showGrid !== false,
          },
        },
      },
    };

    return { data: chartData, _options };
  }

  private buildAreaChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series, xAxis, yAxis, libraryOptions } = _config;
    const xKey = xAxis?.dataKey || 'name';

    // Extract library-specific options
    const strokeWidth = libraryOptions?.strokeWidth || 2;
    const curveType = libraryOptions?.curveType || 'monotone';
    const tension = curveType === 'linear' ? 0 : curveType === 'step' ? 0 : 0.4;
    const fillOpacity = libraryOptions?.fillOpacity !== undefined ? libraryOptions.fillOpacity : 0.3;
    // Convert opacity (0-1) to hex (00-FF)
    const opacityHex = Math.round(fillOpacity * 255).toString(16).padStart(2, '0');

    const chartData = {
      labels: data.map(d => String(d[xKey] || '')),
      datasets: series
        .filter(s => s.visible !== false)
        .map((s, idx) => {
          const color = s.color || colors[idx % colors.length];
          return {
            label: s.name || s.dataKey,
            data: data.map(d => d[s.dataKey]),
            borderColor: color,
            borderWidth: strokeWidth,
            backgroundColor: `${color}${opacityHex}`, // Dynamic transparency
            fill: true,
            tension: tension,
          };
        }),
    };

    const _options: ChartOptions<'line'> = {
      plugins: {
        title: {
          display: false,
        },
      },
      scales: {
        x: {
          title: {
            display: !!xAxis?.label,
            text: xAxis?.label || '',
          },
          grid: {
            display: xAxis?.showGrid !== false,
          },
        },
        y: {
          type: yAxis?.scale === 'log' ? 'logarithmic' : 'linear',
          title: {
            display: !!yAxis?.label,
            text: yAxis?.label || '',
          },
          grid: {
            display: yAxis?.showGrid !== false,
          },
        },
      },
    };

    return { data: chartData, _options };
  }

  private buildScatterChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series, xAxis, yAxis } = _config;
    const xKey = xAxis?.dataKey || 'x';

    const chartData = {
      datasets: series
        .filter(s => s.visible !== false)
        .map((s, idx) => ({
          label: s.name || s.dataKey,
          data: data.map(d => ({
            x: d[xKey] as number,
            y: d[s.dataKey] as number,
          })),
          backgroundColor: s.color || colors[idx % colors.length],
        })),
    };

    const _options: ChartOptions<'scatter'> = {
      plugins: {
        title: {
          display: false,
        },
      },
      scales: {
        x: {
          title: {
            display: !!xAxis?.label,
            text: xAxis?.label || '',
          },
          grid: {
            display: xAxis?.showGrid !== false,
          },
        },
        y: {
          type: yAxis?.scale === 'log' ? 'logarithmic' : 'linear',
          title: {
            display: !!yAxis?.label,
            text: yAxis?.label || '',
          },
          grid: {
            display: yAxis?.showGrid !== false,
          },
        },
      },
    };

    return { data: chartData, _options };
  }

  private buildPieChart(_config: UniversalChartConfig, colors: string[]) {
    const { data, series } = _config;
    const dataKey = series[0]?.dataKey || 'value';

    const chartData = {
      labels: data.map(d => String(d.name || '')),
      datasets: [
        {
          data: data.map(d => d[dataKey]),
          backgroundColor: colors.slice(0, data.length),
          borderColor: colors.slice(0, data.length),
          borderWidth: 1,
        },
      ],
    };

    const _options: ChartOptions<'pie'> = {
      plugins: {
        title: {
          display: false,
        },
      },
    };

    return { data: chartData, _options };
  }

  async export(_config: UniversalChartConfig, _options: ChartExportOptions): Promise<void> {
    throw new Error('Export not yet implemented for Chart.js adapter. Use browser right-click save.');
  }
}

// Export singleton instance
export const chartjsAdapter = new ChartJsAdapter();
