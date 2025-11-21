import React from "react";
import { useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';
import { UniversalChart } from '../../lib/charts/UniversalChart';
import { ChartRendererSelector } from './chart-renderer-selector';
import { ChartSettingsPanel, ChartSettings } from '../charts/ChartSettingsPanel';
import { ChartExportButton } from '../charts/ChartExportButton';
import type { ChartRenderer, ChartExportOptions } from '../../lib/charts/types';

interface DistributionChartProps {
  data: Array<Record<string, any>>;
  field: string;
  title?: string;
  description?: string;
  bins?: number;
  className?: string;
  renderer?: ChartRenderer;
}

const DEFAULT_SETTINGS: ChartSettings = {
  animation: true,
  height: 300,
  legend: { show: false },
  xAxis: { showGrid: true },
  yAxis: { showGrid: true, scale: 'linear' },
};

export function DistributionChart({
  data,
  field,
  title,
  description,
  bins = 10,
  className,
  renderer: initialRenderer = 'recharts',
}: DistributionChartProps) {
  const [renderer, setRenderer] = useState<ChartRenderer>(initialRenderer);
  const [settings, setSettings] = useState<ChartSettings>(DEFAULT_SETTINGS);
  const chartData = useMemo(() => {
    // Extract values for the field
    const values = data
      .map((d) => {
        const value = d[field];
        return typeof value === 'number' ? value : parseFloat(value);
      })
      .filter((v) => !isNaN(v));

    if (values.length === 0) return [];

    // Calculate histogram
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;

    // Handle case where all values are the same
    if (range === 0) {
      return [{
        range: min.toFixed(2),
        count: values.length,
        binStart: min,
        binEnd: min,
      }];
    }

    const binWidth = range / bins;

    // Initialize bins
    const histogram = Array.from({ length: bins }, (_, i) => ({
      range: `${(min + i * binWidth).toFixed(2)} - ${(min + (i + 1) * binWidth).toFixed(2)}`,
      count: 0,
      binStart: min + i * binWidth,
      binEnd: min + (i + 1) * binWidth,
    }));

    // Fill histogram
    values.forEach((value) => {
      const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
      if (histogram[binIndex]) {
        histogram[binIndex].count++;
      }
    });

    return histogram;
  }, [data, field, bins]);

  const stats = useMemo(() => {
    const values = data
      .map((d) => {
        const value = d[field];
        return typeof value === 'number' ? value : parseFloat(value);
      })
      .filter((v) => !isNaN(v));

    if (values.length === 0) return null;

    const sorted = [...values].sort((a, b) => a - b);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const median = sorted[Math.floor(sorted.length / 2)];
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];

    // Standard deviation
    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);

    return { mean, median, min, max, q1, q3, std, count: values.length };
  }, [data, field]);

  // Available renderers for bar charts
  const availableRenderers: ChartRenderer[] = ['recharts', 'chartjs', 'plotly'];

  // Export handler
  const handleExport = async (options: ChartExportOptions) => {
    if (options.format === 'csv') {
      const csv = convertToCSV(chartData);
      downloadFile(csv, `${options.filename}.csv`, 'text/csv');
    } else if (options.format === 'json') {
      const json = JSON.stringify({ data: chartData, stats, field }, null, 2);
      downloadFile(json, `${options.filename}.json`, 'application/json');
    } else {
      console.warn('Image export not yet implemented. Use browser screenshot or right-click save.');
    }
  };

  if (chartData.length === 0 || !stats) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{title || `${field} Distribution`}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-muted-foreground">
            No numeric data available for {field}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <CardTitle>{title || `${field} Distribution`}</CardTitle>
            {description && <CardDescription className="mt-1">{description}</CardDescription>}
          </div>

          {/* Right-aligned controls */}
          <div className="flex items-center gap-2">
            {/* Chart Renderer Selector */}
            {availableRenderers.length > 1 && (
              <ChartRendererSelector
                value={renderer}
                onChange={setRenderer}
                availableRenderers={availableRenderers}
                size="sm"
                showIcon={false}
              />
            )}

            {/* Chart Settings Button */}
            <ChartSettingsPanel
              settings={settings}
              onSettingsChange={setSettings}
            />

            {/* Export Button */}
            <ChartExportButton onExport={handleExport} size="sm" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Chart */}
          <UniversalChart
            type="bar"
            renderer={renderer}
            data={chartData}
            series={[
              {
                dataKey: 'count',
                name: 'Count',
                color: 'hsl(var(--primary))',
                visible: true,
              }
            ]}
            xAxis={{
              dataKey: 'range',
              label: settings.xAxis?.label || field,
              showGrid: settings.xAxis?.showGrid,
            }}
            yAxis={{
              label: settings.yAxis?.label || 'Count',
              showGrid: settings.yAxis?.showGrid,
              scale: settings.yAxis?.scale,
            }}
            height={settings.height || 300}
            tooltip={{ show: true }}
            legend={settings.legend}
            animation={settings.animation}
          />

          {/* Statistics Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Mean</div>
              <div className="text-sm font-semibold">{stats.mean.toFixed(4)}</div>
            </div>
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Median</div>
              <div className="text-sm font-semibold">{stats.median.toFixed(4)}</div>
            </div>
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Std Dev</div>
              <div className="text-sm font-semibold">{stats.std.toFixed(4)}</div>
            </div>
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Range</div>
              <div className="text-sm font-semibold">
                {stats.min.toFixed(2)} - {stats.max.toFixed(2)}
              </div>
            </div>
          </div>

          {/* Box Plot Summary */}
          <div className="space-y-2 pt-2 border-t">
            <div className="text-xs font-medium text-muted-foreground">Box Plot Summary</div>
            <div className="flex items-center gap-2 text-xs">
              <span className="font-medium">Min:</span> {stats.min.toFixed(4)}
              <span className="text-muted-foreground">|</span>
              <span className="font-medium">Q1:</span> {stats.q1.toFixed(4)}
              <span className="text-muted-foreground">|</span>
              <span className="font-medium">Q2:</span> {stats.median.toFixed(4)}
              <span className="text-muted-foreground">|</span>
              <span className="font-medium">Q3:</span> {stats.q3.toFixed(4)}
              <span className="text-muted-foreground">|</span>
              <span className="font-medium">Max:</span> {stats.max.toFixed(4)}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Helper: Convert data to CSV
 */
function convertToCSV(data: Array<{ range: string; count: number; binStart: number; binEnd: number }>): string {
  if (data.length === 0) return '';

  const header = 'range,count,binStart,binEnd';
  const rows = data.map(row => `"${row.range}",${row.count},${row.binStart},${row.binEnd}`);

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
