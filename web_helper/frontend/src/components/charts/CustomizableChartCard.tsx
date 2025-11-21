import React from "react";

/**
 * Customizable Chart Card
 *
 * Fully customizable chart component with all controls:
 * - Chart Type Selector
 * - Renderer Selector
 * - Data Series Selector
 * - Settings Panel
 * - Export Button
 */

import { useState, useMemo } from 'react';
import { ChartCard } from '@/components/ui/chart-card';
import { UniversalChart } from '@/lib/charts/UniversalChart';
import { ChartRendererSelector } from '@/components/ui/chart-renderer-selector';
import { DataSeriesSelector } from './DataSeriesSelector';
import { ChartSettingsPanel, ChartSettings } from './ChartSettingsPanel';
import { ExportFormat } from '@/components/ui/export-menu';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { LineChart, BarChart3, AreaChart, ScatterChart } from 'lucide-react';
import type {
  ChartType,
  ChartRenderer,
  DataPoint,
  SeriesConfig,
  UniversalChartConfig,
} from '@/lib/charts/types';
import {
  getSupportedRenderers,
  getSupportedChartTypes,
  isCompatible,
  getFirstCompatibleRenderer,
  getFirstCompatibleChartType,
} from '@/lib/charts/types';

export interface CustomizableChartCardProps {
  /** Chart title */
  title: string;
  /** Chart description */
  description?: string;
  /** Raw data */
  data: DataPoint[];
  /** Initial _series configuration */
  initialSeries: SeriesConfig[];
  /** Chart type (can be changed by user) */
  chartType?: ChartType;
  /** Supported chart types (defaults to line, bar, area, scatter) */
  supportedChartTypes?: ChartType[];
  /** Initial renderer */
  initialRenderer?: ChartRenderer;
  /** Initial settings */
  initialSettings?: Partial<ChartSettings>;
  /** X-axis data key */
  xAxisKey?: string;
  /** className */
  className?: string;
  /** Compact mode (less padding) */
  compact?: boolean;

  // Feature visibility controls
  /** Show chart type selector (default: true) */
  showChartTypeSelector?: boolean;
  /** Show renderer selector (default: true) */
  showRendererSelector?: boolean;
  /** Show data _series selector (default: true) */
  showSeriesSelector?: boolean;
  /** Show settings panel (default: true) */
  showSettings?: boolean;
  /** Show fullscreen button (default: true) */
  showFullscreen?: boolean;
  /** Show export button (default: true) */
  showExport?: boolean;
}

/**
 * Default chart settings
 */
const DEFAULT_SETTINGS: ChartSettings = {
  animation: true,
  height: 400,
  legend: {
    show: true,
    position: 'top',
    align: 'center',
  },
  xAxis: {
    showGrid: true,
  },
  yAxis: {
    showGrid: true,
    scale: 'linear',
  },
  strokeWidth: 2,
  showDots: false,
  curveType: 'monotone',
  fillOpacity: 0.3,
  stackMode: 'none',
  dotSize: 4,
};

export function CustomizableChartCard({
  title,
  description,
  data,
  initialSeries,
  chartType: initialChartType = 'bar',
  supportedChartTypes = ['line', 'bar', 'area', 'scatter'],
  initialRenderer = 'recharts',
  initialSettings,
  xAxisKey = 'name',
  className,
  compact = false,
  showChartTypeSelector = true,
  showRendererSelector = true,
  showSeriesSelector = true,
  showSettings = true,
  showFullscreen = true,
  showExport = true,
}: CustomizableChartCardProps) {
  // State management
  const [chartType, setChartType] = useState<ChartType>(initialChartType);
  const [renderer, setRenderer] = useState<ChartRenderer>(initialRenderer);
  const [_series, setSeries] = useState<SeriesConfig[]>(initialSeries);
  const [settings, setSettings] = useState<ChartSettings>({
    ...DEFAULT_SETTINGS,
    ...initialSettings,
  });

  // Update settings when initialSettings changes (deep comparison via JSON)
  React.useEffect(() => {
    setSettings({
      ...DEFAULT_SETTINGS,
      ...initialSettings,
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(initialSettings)]);

  // Get available renderers for the current chart type
  const availableRenderers = useMemo((): ChartRenderer[] => {
    return getSupportedRenderers(chartType);
  }, [chartType]);

  // Get available chart types for the current renderer
  const _availableChartTypes = useMemo(() => {
    return getSupportedChartTypes(renderer);
  }, [renderer]);

  // Auto-switch renderer if current renderer doesn't support the new chart type
  React.useEffect(() => {
    if (!isCompatible(renderer, chartType)) {
      const compatibleRenderer = getFirstCompatibleRenderer(chartType);
      if (compatibleRenderer) {
        setRenderer(compatibleRenderer);
      }
    }
  }, [chartType, renderer]);

  // Auto-switch chart type if current chart type is not supported by the new renderer
  React.useEffect(() => {
    if (!isCompatible(renderer, chartType)) {
      const compatibleType = getFirstCompatibleChartType(renderer);
      if (compatibleType) {
        setChartType(compatibleType);
      }
    }
  }, [renderer, chartType]);

  // Build chart config
  const chartConfig: UniversalChartConfig = useMemo(
    () => ({
      type: chartType,
      data,
      series: _series,
      xAxis: {
        dataKey: xAxisKey,
        label: settings.xAxis?.label,
        showGrid: settings.xAxis?.showGrid,
      },
      yAxis: {
        label: settings.yAxis?.label,
        showGrid: settings.yAxis?.showGrid,
        scale: settings.yAxis?.scale,
      },
      legend: settings.legend,
      height: settings.height,
      animation: settings.animation,
      // Pass chart-specific settings through libraryOptions
      libraryOptions: {
        strokeWidth: settings.strokeWidth,
        showDots: settings.showDots,
        curveType: settings.curveType,
        fillOpacity: settings.fillOpacity,
        stackMode: settings.stackMode,
        dotSize: settings.dotSize,
      },
    }),
    [chartType, data, _series, xAxisKey, settings]
  );

  // Export handler
  const handleExport = (format: ExportFormat) => {
    if (format === 'csv') {
      // Export data as CSV
      const csv = convertToCSV(data, _series);
      downloadFile(csv, `chart-${title.toLowerCase().replace(/\s+/g, '-')}.csv`, 'text/csv');
    } else if (format === 'json') {
      // Export data as JSON
      const json = JSON.stringify({ data, _series, settings }, null, 2);
      downloadFile(json, `chart-${title.toLowerCase().replace(/\s+/g, '-')}.json`, 'application/json');
    }
  };

  return (
    <ChartCard
      title={title}
      description={description}
      chartType={chartType as any}
      enableSettings={false}
      enableFullscreen={showFullscreen}
      enableExport={showExport}
      onExport={showExport ? handleExport : undefined}
      height={settings.height}
      className={className}
      variant={compact ? 'compact' : 'default'}
      customControls={
        <>
          {/* Chart Type Selector */}
          {showChartTypeSelector && (
            <ToggleGroup
              type="single"
              value={chartType}
              onValueChange={(v) => v && setChartType(v as ChartType)}
              className="h-8"
            >
              {supportedChartTypes.includes('line') && (
                <ToggleGroupItem value="line" aria-label="Line Chart" className="h-8 px-2">
                  <LineChart className="h-4 w-4" />
                </ToggleGroupItem>
              )}
              {supportedChartTypes.includes('bar') && (
                <ToggleGroupItem value="bar" aria-label="Bar Chart" className="h-8 px-2">
                  <BarChart3 className="h-4 w-4" />
                </ToggleGroupItem>
              )}
              {supportedChartTypes.includes('area') && (
                <ToggleGroupItem value="area" aria-label="Area Chart" className="h-8 px-2">
                  <AreaChart className="h-4 w-4" />
                </ToggleGroupItem>
              )}
              {supportedChartTypes.includes('scatter') && (
                <ToggleGroupItem value="scatter" aria-label="Scatter Chart" className="h-8 px-2">
                  <ScatterChart className="h-4 w-4" />
                </ToggleGroupItem>
              )}
            </ToggleGroup>
          )}

          {/* Chart Renderer Selector - only show if enabled and multiple renderers available */}
          {showRendererSelector && availableRenderers.length > 1 && (
            <ChartRendererSelector
              value={renderer}
              onChange={setRenderer}
              availableRenderers={availableRenderers}
              currentChartType={chartType}
              size="sm"
              showIcon={false}
            />
          )}

          {/* Data Series Selector */}
          {showSeriesSelector && (
            <DataSeriesSelector _series={_series} onSeriesChange={setSeries} compact />
          )}

          {/* Chart Settings Button */}
          {showSettings && (
            <ChartSettingsPanel
              settings={settings}
              onSettingsChange={setSettings}
              chartType={chartType}
              renderer={renderer}
            />
          )}
        </>
      }
    >
      <UniversalChart renderer={renderer} {...chartConfig} />
    </ChartCard>
  );
}

/**
 * Helper: Convert data to CSV
 */
function convertToCSV(data: DataPoint[], _series: SeriesConfig[]): string {
  if (data.length === 0) return '';

  // Get all keys from first data point + _series keys
  const keys = Object.keys(data[0]);
  const header = keys.join(',');

  const rows = data.map(row =>
    keys.map(key => {
      const value = row[key];
      // Escape commas and quotes
      if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
        return `"${value.replace(/"/g, '""')}"`;
      }
      return value;
    }).join(',')
  );

  return [header, ...rows].join('\n');
}

/**
 * Helper: Download file
 */
function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
