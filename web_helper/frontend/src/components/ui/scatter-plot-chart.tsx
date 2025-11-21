import React from "react";
import { useMemo, useState } from 'react';
import { ChartCard } from './chart-card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './select';
import { Label } from './label';
import { UniversalChart } from '../../lib/charts/UniversalChart';
import { ChartRendererSelector } from './chart-renderer-selector';
import { ChartSettingsPanel, ChartSettings } from '../charts/ChartSettingsPanel';
import { ExportFormat } from './export-menu';
import type { ChartRenderer } from '../../lib/charts/types';

interface ScatterPlotChartProps {
  data: Array<Record<string, any>>;
  xOptions: string[];
  yOptions: string[];
  title?: string;
  description?: string;
  className?: string;
  defaultX?: string;
  defaultY?: string;
  renderer?: ChartRenderer;
  variant?: 'default' | 'compact';

  // Feature visibility controls
  /** Show renderer selector (default: true) */
  showRendererSelector?: boolean;
  /** Show settings panel (default: true) */
  showSettings?: boolean;
  /** Show fullscreen button (default: true) */
  showFullscreen?: boolean;
  /** Show export button (default: true) */
  showExport?: boolean;
}

const DEFAULT_SETTINGS: ChartSettings = {
  animation: true,
  height: 400,
  legend: { show: false },
  xAxis: { showGrid: true },
  yAxis: { showGrid: true, scale: 'linear' },
};

export function ScatterPlotChart({
  data,
  xOptions,
  yOptions,
  title = 'Scatter Plot Analysis',
  description,
  className,
  defaultX,
  defaultY,
  renderer: initialRenderer = 'recharts',
  variant = 'default',
  showRendererSelector = true,
  showSettings = true,
  showFullscreen = true,
  showExport = true,
}: ScatterPlotChartProps) {
  const [xField, setXField] = useState(defaultX || xOptions[0] || '');
  const [yField, setYField] = useState(defaultY || yOptions[0] || '');
  const [renderer, setRenderer] = useState<ChartRenderer>(initialRenderer);
  const [settings, setSettings] = useState<ChartSettings>(DEFAULT_SETTINGS);
  const [_isFullscreen, _setIsFullscreen] = useState(false);

  const chartData = useMemo(() => {
    return data
      .map((d, idx) => {
        const xVal = typeof d[xField] === 'number' ? d[xField] : parseFloat(d[xField]);
        const yVal = typeof d[yField] === 'number' ? d[yField] : parseFloat(d[yField]);

        if (isNaN(xVal) || isNaN(yVal)) return null;

        return {
          x: xVal,
          y: yVal,
          name: d.run_name || `Run ${idx + 1}`,
        };
      })
      .filter((d) => d !== null) as Array<{ x: number; y: number; name: string }>;
  }, [data, xField, yField]);

  const correlation = useMemo(() => {
    if (chartData.length < 2) return null;

    const xValues = chartData.map((d) => d.x);
    const yValues = chartData.map((d) => d.y);

    const n = xValues.length;
    const xMean = xValues.reduce((a, b) => a + b, 0) / n;
    const yMean = yValues.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let xDenominator = 0;
    let yDenominator = 0;

    for (let i = 0; i < n; i++) {
      const xDiff = xValues[i] - xMean;
      const yDiff = yValues[i] - yMean;
      numerator += xDiff * yDiff;
      xDenominator += xDiff * xDiff;
      yDenominator += yDiff * yDiff;
    }

    const r = numerator / Math.sqrt(xDenominator * yDenominator);
    return isNaN(r) ? null : r;
  }, [chartData]);

  const getCorrelationStrength = (r: number | null) => {
    if (r === null) return { text: 'N/A', color: 'text-gray-500' };
    const abs = Math.abs(r);
    if (abs >= 0.7) return { text: 'Strong', color: 'text-green-600 dark:text-green-400' };
    if (abs >= 0.4) return { text: 'Moderate', color: 'text-yellow-600 dark:text-yellow-400' };
    return { text: 'Weak', color: 'text-red-600 dark:text-red-400' };
  };

  const correlationStrength = getCorrelationStrength(correlation);

  // Available renderers for scatter charts
  const availableRenderers: ChartRenderer[] = ['recharts', 'chartjs', 'plotly'];

  // Export handler
  const handleExport = (format: ExportFormat) => {
    if (format === 'csv') {
      const csv = convertToCSV(chartData);
      downloadFile(csv, `scatter-plot-${xField}-vs-${yField}.csv`, 'text/csv');
    } else if (format === 'json') {
      const json = JSON.stringify({ data: chartData, xField, yField, correlation }, null, 2);
      downloadFile(json, `scatter-plot-${xField}-vs-${yField}.json`, 'application/json');
    }
  };

  if (chartData.length === 0) {
    return (
      <ChartCard
        title={title}
        description={description}
        isEmpty
        emptyMessage="No valid data points for scatter plot"
        chartType="scatter"
        enableSettings={false}
        enableFullscreen={false}
        enableExport={false}
        height={400}
        className={className}
        variant={variant}
      >
        <div />
      </ChartCard>
    );
  }

  return (
    <ChartCard
      title={title}
      description={description}
      chartType="scatter"
      enableSettings={false}
      enableFullscreen={showFullscreen}
      enableExport={showExport}
      onExport={showExport ? handleExport : undefined}
      height={settings.height || 400}
      className={className}
      variant={variant}
      customControls={
        <>
          {/* Axis Selectors */}
          <div className="flex items-center gap-2">
            <Label htmlFor="x-axis" className="text-sm font-medium whitespace-nowrap">
              X:
            </Label>
            <Select value={xField} onValueChange={setXField}>
              <SelectTrigger id="x-axis" className="h-8 w-[180px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {xOptions.map((opt) => (
                  <SelectItem key={opt} value={opt}>
                    {opt}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <Label htmlFor="y-axis" className="text-sm font-medium whitespace-nowrap">
              Y:
            </Label>
            <Select value={yField} onValueChange={setYField}>
              <SelectTrigger id="y-axis" className="h-8 w-[180px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {yOptions.map((opt) => (
                  <SelectItem key={opt} value={opt}>
                    {opt}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Chart Renderer Selector */}
          {showRendererSelector && availableRenderers.length > 1 && (
            <ChartRendererSelector
              value={renderer}
              onChange={setRenderer}
              availableRenderers={availableRenderers}
              currentChartType="scatter"
              size="sm"
              showIcon={false}
            />
          )}

          {/* Chart Settings */}
          {showSettings && (
            <ChartSettingsPanel
              settings={settings}
              onSettingsChange={setSettings}
              chartType="scatter"
              renderer={renderer}
            />
          )}
        </>
      }
    >
      <div className="space-y-4">
        {/* Chart */}
        <UniversalChart
          type="scatter"
          renderer={renderer}
          data={chartData}
          series={[
            {
              dataKey: 'y',
              name: 'Runs',
              color: 'hsl(var(--primary))',
              visible: true,
            }
          ]}
          xAxis={{
            dataKey: 'x',
            label: settings.xAxis?.label || xField,
            showGrid: settings.xAxis?.showGrid,
          }}
          yAxis={{
            dataKey: 'y',
            label: settings.yAxis?.label || yField,
            showGrid: settings.yAxis?.showGrid,
            scale: settings.yAxis?.scale,
          }}
          height={settings.height || 400}
          tooltip={{ show: true }}
          legend={settings.legend}
          animation={settings.animation}
          libraryOptions={{
            dotSize: settings.dotSize,
          }}
        />

        {/* Correlation Statistics */}
        {correlation !== null && (
          <div className="grid grid-cols-3 gap-4 pt-4 border-t">
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">
                Correlation (r)
              </div>
              <div className="text-lg font-bold">{correlation.toFixed(4)}</div>
            </div>
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Strength</div>
              <div className={`text-lg font-bold ${correlationStrength.color}`}>
                {correlationStrength.text}
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Data Points</div>
              <div className="text-lg font-bold">{chartData.length}</div>
            </div>
          </div>
        )}

        {/* Interpretation */}
        <div className="text-xs text-muted-foreground bg-muted/50 p-3 rounded-md">
          <strong>Interpretation:</strong>{' '}
          {correlation !== null && Math.abs(correlation) >= 0.4
            ? `There is a ${correlationStrength.text.toLowerCase()} ${correlation > 0 ? 'positive' : 'negative'} correlation between ${xField} and ${yField}.`
            : `There is little to no linear correlation between ${xField} and ${yField}.`}
        </div>
      </div>
    </ChartCard>
  );
}

/**
 * Helper: Convert data to CSV
 */
function convertToCSV(data: Array<{ x: number; y: number; name: string }>): string {
  if (data.length === 0) return '';

  const header = 'x,y,name';
  const rows = data.map(row => `${row.x},${row.y},"${row.name}"`);

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
