import React from "react";

/**
 * Distribution Chart
 *
 * Visualizes metric distribution across runs using:
 * - Histogram (frequency distribution)
 * - Box plot overlay (optional)
 * - Statistical summary (mean, median, quartiles, std)
 * - Outlier highlighting
 */

import { useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts';
import { ChartCard } from '../ui/chart-card';
import { ChartSettingsPanel, ChartSettings } from './ChartSettingsPanel';
import type { Run } from '../../lib/types';
import { transformForDistribution } from '../../lib/chartDataTransformers';
import { useChartSettings, HistogramSettings, getColorScheme } from '../../lib/stores/chart-settings';
import { ExportFormat } from '../ui/export-menu';

interface DistributionChartProps {
  runs: Run[];
  metricKey: string;
  title?: string;
  description?: string;
  height?: number;
  binCount?: number;
  showBoxPlot?: boolean;
  highlightRange?: [number, number];
  variant?: 'default' | 'compact';
  badge?: {
    label: string;
    variant?: 'default' | 'destructive' | 'outline' | 'secondary';
  };
  customControls?: React.ReactNode;
  showSettings?: boolean;
  showFullscreen?: boolean;
  showExport?: boolean;
}

const DEFAULT_SETTINGS: ChartSettings = {
  animation: true,
  height: 400,
  legend: { show: false },
  xAxis: { showGrid: true },
  yAxis: { showGrid: true, scale: 'linear' },
};

export function DistributionChart({
  runs,
  metricKey,
  title = 'Distribution',
  description,
  height = 400,
  binCount,
  showBoxPlot = true,
  highlightRange,
  variant = 'default',
  badge,
  customControls,
  showSettings = true,
  showFullscreen = true,
  showExport = true,
}: DistributionChartProps) {
  // Local settings state
  const [settings, setSettings] = useState<ChartSettings>(DEFAULT_SETTINGS);

  // Get chart settings from store for defaults
  const { getSettings: getStoreSettings } = useChartSettings();
  const storeSettings = getStoreSettings<HistogramSettings>('histogram');

  // Merge settings with priority: ChartSettings.histogram > prop > store > default
  const actualBinCount = settings.histogram?.binCount ?? binCount ?? storeSettings.binCount ?? 20;
  const actualShowMean = settings.histogram?.showMean ?? storeSettings.showMean ?? true;
  const actualShowMedian = settings.histogram?.showMedian ?? storeSettings.showMedian ?? false;
  const actualShowGrid = settings.xAxis?.showGrid ?? storeSettings.showGrid ?? true;
  const actualShowLegend = settings.legend?.show ?? storeSettings.showLegend ?? false;
  const actualFontSize = storeSettings.fontSize ?? 'sm';
  const actualColorScheme = storeSettings.colorScheme ?? 'nord';

  // Font size mapping
  const fontSizeMap = { xs: 10, sm: 11, base: 12, lg: 14 };
  const tickFontSize = fontSizeMap[actualFontSize];
  const labelFontSize = fontSizeMap[actualFontSize] + 1;

  // Get color scheme
  const colors = getColorScheme(actualColorScheme);

  // Transform data
  const distributionData = useMemo(() => {
    return transformForDistribution(runs, metricKey, actualBinCount);
  }, [runs, metricKey, actualBinCount]);

  // Prepare histogram data for Recharts
  const histogramData = useMemo(() => {
    return distributionData.bins.map((bin, index) => ({
      binIndex: index,
      binLabel: `${bin.x0.toFixed(2)}-${bin.x1.toFixed(2)}`,
      count: bin.length,
      rangeStart: bin.x0,
      rangeEnd: bin.x1,
    }));
  }, [distributionData.bins]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;

    return (
      <div className="bg-background border border-border rounded-lg shadow-lg p-3">
        <div className="space-y-1 text-sm">
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Range:</span>
            <span className="font-medium font-mono">{data.binLabel}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Count:</span>
            <span className="font-medium">{data.count} runs</span>
          </div>
        </div>
      </div>
    );
  };

  // Determine if a bin is in the highlight range
  const isInHighlightRange = (rangeStart: number, rangeEnd: number): boolean => {
    if (!highlightRange) return false;
    const [min, max] = highlightRange;
    return rangeStart <= max && rangeEnd >= min;
  };

  // Export handler
  const handleExport = (format: ExportFormat) => {
    if (format === 'csv') {
      const csv = convertToCSV(histogramData, distributionData.stats);
      downloadFile(csv, `distribution-${metricKey}.csv`, 'text/csv');
    } else if (format === 'json') {
      const json = JSON.stringify({ data: histogramData, stats: distributionData.stats, metricKey }, null, 2);
      downloadFile(json, `distribution-${metricKey}.json`, 'application/json');
    }
  };

  if (distributionData.bins.length === 0) {
    return (
      <ChartCard
        title={title}
        description={description}
        isEmpty
        emptyMessage="No valid data for distribution analysis"
        chartType="histogram"
        enableSettings={false}
        enableFullscreen={false}
        enableExport={false}
        height={height}
      >
        <div />
      </ChartCard>
    );
  }

  const { stats } = distributionData;

  return (
    <ChartCard
      title={title}
      description={description}
      badge={badge}
      customControls={
        <>
          {customControls}
          {showSettings && (
            <ChartSettingsPanel
              settings={settings}
              onSettingsChange={setSettings}
              chartType="histogram"
              renderer="recharts"
            />
          )}
        </>
      }
      enableSettings={false}
      enableFullscreen={showFullscreen}
      enableExport={showExport}
      onExport={showExport ? handleExport : undefined}
      height={settings.height || height}
      variant={variant}
    >
      <div className="space-y-2">
        <ResponsiveContainer width="100%" height={height}>
          <BarChart
            data={histogramData}
            margin={{ top: 20, right: 30, left: 10, bottom: 10 }}
          >
            {actualShowGrid && <CartesianGrid strokeDasharray="3 3" className="opacity-30" />}

            <XAxis
              dataKey="binLabel"
              tick={{ fontSize: tickFontSize }}
              angle={-45}
              textAnchor="end"
              height={80}
              label={{ value: metricKey, position: 'insideBottom', offset: -15, fontSize: labelFontSize }}
            />

            <YAxis
              tick={{ fontSize: tickFontSize }}
              label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fontSize: labelFontSize }}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Box plot reference lines */}
            {showBoxPlot && (
              <>
                {/* Median */}
                {actualShowMedian && (
                  <ReferenceLine
                    x={histogramData.findIndex(
                      d => d.rangeStart <= stats.median && d.rangeEnd >= stats.median
                    )}
                    stroke={colors.primary}
                    strokeWidth={2}
                    label={{
                      value: `Median: ${stats.median.toFixed(3)}`,
                      position: 'top',
                      fontSize: tickFontSize,
                      fill: colors.primary,
                    }}
                  />
                )}

                {/* Mean */}
                {actualShowMean && (
                  <ReferenceLine
                    x={histogramData.findIndex(
                      d => d.rangeStart <= stats.mean && d.rangeEnd >= stats.mean
                    )}
                    stroke={colors.error || colors.accent}
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    label={{
                      value: `Mean: ${stats.mean.toFixed(3)}`,
                      position: 'bottom',
                      fontSize: tickFontSize,
                      fill: colors.error || colors.accent,
                    }}
                  />
                )}

                {/* Q1 */}
                <ReferenceLine
                  x={histogramData.findIndex(
                    d => d.rangeStart <= stats.q1 && d.rangeEnd >= stats.q1
                  )}
                  stroke={colors.muted}
                  strokeWidth={1}
                  strokeDasharray="3 3"
                />

                {/* Q3 */}
                <ReferenceLine
                  x={histogramData.findIndex(
                    d => d.rangeStart <= stats.q3 && d.rangeEnd >= stats.q3
                  )}
                  stroke={colors.muted}
                  strokeWidth={1}
                  strokeDasharray="3 3"
                />
              </>
            )}

            {/* Histogram bars */}
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {histogramData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={
                    isInHighlightRange(entry.rangeStart, entry.rangeEnd)
                      ? colors.accent
                      : colors.primary
                  }
                  opacity={
                    isInHighlightRange(entry.rangeStart, entry.rangeEnd) ? 1 : 0.7
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Statistics summary */}
        <div className="mt-2 grid grid-cols-3 gap-3 text-sm border-t pt-2">
          <div>
            <h4 className="font-medium mb-1">Central Tendency</h4>
            <div className="space-y-0.5 text-xs text-muted-foreground">
              <div className="flex justify-between">
                <span>Mean:</span>
                <span className="font-mono font-medium text-foreground">
                  {stats.mean.toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Median:</span>
                <span className="font-mono font-medium text-foreground">
                  {stats.median.toFixed(4)}
                </span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-1">Spread</h4>
            <div className="space-y-0.5 text-xs text-muted-foreground">
              <div className="flex justify-between">
                <span>Std Dev:</span>
                <span className="font-mono font-medium text-foreground">
                  {stats.std.toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>IQR:</span>
                <span className="font-mono font-medium text-foreground">
                  {(stats.q3 - stats.q1).toFixed(4)}
                </span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-1">Range</h4>
            <div className="space-y-0.5 text-xs text-muted-foreground">
              <div className="flex justify-between">
                <span>Min:</span>
                <span className="font-mono font-medium text-foreground">
                  {stats.min.toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Max:</span>
                <span className="font-mono font-medium text-foreground">
                  {stats.max.toFixed(4)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Box plot legend */}
        {showBoxPlot && actualShowLegend && (
          <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground border-t pt-2">
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-primary" />
              <span>Median</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-destructive border-dashed" />
              <span>Mean</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-muted-foreground border-dashed" />
              <span>Q1/Q3</span>
            </div>
          </div>
        )}
      </div>
    </ChartCard>
  );
}

/**
 * Helper: Convert data to CSV
 */
function convertToCSV(data: any[], stats: any): string {
  if (data.length === 0) return '';

  const header = 'binLabel,count,rangeStart,rangeEnd';
  const rows = data.map(row => `"${row.binLabel}",${row.count},${row.rangeStart},${row.rangeEnd}`);
  const statsRow = `\nStatistics:,,,\nMean,${stats.mean},Median,${stats.median}\nStd Dev,${stats.std},IQR,${stats.q3 - stats.q1}\nMin,${stats.min},Max,${stats.max}`;

  return [header, ...rows, statsRow].join('\n');
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
