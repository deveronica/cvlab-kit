import React from "react";
/**
 * Hyperparameter Correlation Chart
 *
 * Visualizes the relationship between a hyperparameter and a metric using:
 * - Scatter plot with customizable settings
 * - Linear regression trend line
 * - Pearson correlation coefficient
 * - Outlier detection and highlighting
 * - Fullscreen support with adaptive layout
 */

import { useMemo, useCallback, useState } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  _Legend,
  _TooltipProps,
} from 'recharts';
import { ChartCard } from '../ui/chart-card';
import { ChartSettingsPanel, ChartSettings } from './ChartSettingsPanel';
import { Badge } from '../ui/badge';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { ExportFormat } from '../ui/export-menu';
import type { Run } from '../../lib/types';
import { transformForCorrelation } from '../../lib/chartDataTransformers';
import { useChartExport } from '../../hooks/useChartExport';
import { useChartSettings, ScatterSettings, getColorScheme } from '../../lib/stores/chart-settings';

interface HyperparamCorrelationChartProps {
  runs: Run[];
  hyperparamKey: string;
  metricKey: string;
  metricDisplayMode?: 'final' | 'max' | 'min';
  title?: string;
  height?: number;
  showTrendLine?: boolean;
  showOutliers?: boolean;
  onPointClick?: (runName: string) => void;
  variant?: 'default' | 'compact';
}

const DEFAULT_SETTINGS: ChartSettings = {
  animation: true,
  height: 400,
  legend: { show: false },
  xAxis: { showGrid: true },
  yAxis: { showGrid: true, scale: 'linear' },
};

export function HyperparamCorrelationChart({
  runs,
  hyperparamKey,
  metricKey,
  metricDisplayMode = 'final',
  title,
  height = 400,
  showTrendLine = true,
  showOutliers = true,
  onPointClick,
  variant = 'default',
}: HyperparamCorrelationChartProps) {
  // Local settings state
  const [settings, setSettings] = useState<ChartSettings>(DEFAULT_SETTINGS);

  // Get chart settings from store for defaults
  const { getSettings: getStoreSettings } = useChartSettings();
  const storeSettings = getStoreSettings<ScatterSettings>('scatter');

  // Use settings with prop fallbacks
  const actualShowTrendLine = showTrendLine ?? storeSettings.showTrendLine ?? false;
  const actualShowGrid = settings.xAxis?.showGrid ?? storeSettings.showGrid ?? true;
  const actualShowCorrelation = storeSettings.showCorrelation ?? true;
  const _actualPointSize = storeSettings.pointSize ?? 4;
  const actualOpacity = storeSettings.opacity ?? 0.7;
  const actualFontSize = storeSettings.fontSize ?? 'sm';
  const actualColorScheme = storeSettings.colorScheme ?? 'nord';

  // Font size mapping
  const fontSizeMap = { xs: 10, sm: 12, base: 13, lg: 15 };
  const tickFontSize = fontSizeMap[actualFontSize];
  const labelFontSize = fontSizeMap[actualFontSize] + 1;

  // Get color scheme
  const colors = getColorScheme(actualColorScheme);

  // Transform data
  const correlationData = useMemo(() => {
    return transformForCorrelation(runs, hyperparamKey, metricKey, metricDisplayMode);
  }, [runs, hyperparamKey, metricKey, metricDisplayMode]);

  // Calculate trend line points
  const trendLine = useMemo(() => {
    if (!actualShowTrendLine || correlationData.points.length === 0) return [];

    const { slope, intercept } = correlationData.regression;
    const xMin = correlationData.stats.x.min;
    const xMax = correlationData.stats.x.max;

    return [
      { x: xMin, y: slope * xMin + intercept },
      { x: xMax, y: slope * xMax + intercept },
    ];
  }, [correlationData, actualShowTrendLine]);

  // Separate normal points and outliers
  const normalPoints = useMemo(() =>
    correlationData.points.filter(p => !p.isOutlier),
    [correlationData.points]
  );

  const outlierPoints = useMemo(() =>
    correlationData.points.filter(p => p.isOutlier),
    [correlationData.points]
  );

  // Use unified export hook
  const csvData = useMemo(() =>
    correlationData.points.map(point => ({
      run_name: point.runName,
      [hyperparamKey]: point.x,
      [metricKey]: point.y,
      is_outlier: point.isOutlier
    })),
    [correlationData.points, hyperparamKey, metricKey]
  );

  const { chartContainerRef, exportToPNG, exportToSVG, exportToCSV } = useChartExport({
    filename: `${hyperparamKey}_vs_${metricKey}`,
    data: csvData,
  });

  // Export handler
  const handleExport = useCallback(
    (format: ExportFormat) => {
      if (format === 'png') exportToPNG();
      else if (format === 'svg') exportToSVG();
      else if (format === 'csv') exportToCSV();
    },
    [exportToPNG, exportToSVG, exportToCSV]
  );

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;

    return (
      <div className="bg-background border border-border rounded-lg shadow-lg p-3">
        <p className="font-mono text-sm font-medium mb-2">{data.runName}</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">{hyperparamKey}:</span>
            <span className="font-medium">{data.x.toFixed(4)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">{metricKey}:</span>
            <span className="font-medium">{data.y.toFixed(4)}</span>
          </div>
          {data.isOutlier && (
            <Badge variant="destructive" className="text-xs mt-1">
              Outlier
            </Badge>
          )}
        </div>
      </div>
    );
  };

  // Determine correlation strength
  const getCorrelationStrength = (r: number): { label: string; color: string } => {
    const absR = Math.abs(r);
    if (absR >= 0.7) return { label: 'Strong', color: 'text-green-600 dark:text-green-400' };
    if (absR >= 0.4) return { label: 'Moderate', color: 'text-yellow-600 dark:text-yellow-400' };
    return { label: 'Weak', color: 'text-red-600 dark:text-red-400' };
  };

  const correlationStrength = getCorrelationStrength(correlationData.pearsonR);

  // Chart content (for normal and fullscreen)
  const renderChart = (isFullscreen: boolean = false) => {
    const chartHeight = isFullscreen ? '100%' : (variant === 'compact' ? height - 60 : height);

    return (
      <div ref={isFullscreen ? undefined : chartContainerRef} className={isFullscreen ? 'flex flex-col h-full p-6' : ''}>
        {/* Correlation info header */}
        {actualShowCorrelation && (
          <div className={`flex ${variant === 'compact' && !isFullscreen ? 'flex-row items-center justify-between' : 'flex-col items-start'} gap-2 mb-2 pb-2 border-b ${isFullscreen ? 'flex-shrink-0' : ''}`}>
            {/* Correlation coefficient */}
            <div className="flex items-center gap-2">
              {correlationData.pearsonR > 0.1 ? (
                <TrendingUp className="h-4 w-4 text-green-600 dark:text-green-400" />
              ) : correlationData.pearsonR < -0.1 ? (
                <TrendingDown className="h-4 w-4 text-red-600 dark:text-red-400" />
              ) : (
                <Minus className="h-4 w-4 text-gray-600 dark:text-gray-400" />
              )}
              <span className="text-sm font-mono font-medium">
                r = {correlationData.pearsonR.toFixed(3)}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <Badge variant="outline" className={correlationStrength.color}>
                {correlationStrength.label}
              </Badge>

              {/* R² */}
              {actualShowTrendLine && (
                <span className="text-xs text-muted-foreground">
                  R² = {correlationData.regression.rSquared.toFixed(3)}
                </span>
              )}
            </div>
          </div>
        )}

        <div className={isFullscreen ? 'flex-1 min-h-0' : ''}>
          <ResponsiveContainer width="100%" height={chartHeight}>
        <ScatterChart margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
          {actualShowGrid && <CartesianGrid strokeDasharray="3 3" className="opacity-30" />}

          <XAxis
            type="number"
            dataKey="x"
            name={hyperparamKey}
            tick={{ fontSize: tickFontSize }}
            label={{ value: hyperparamKey, position: 'insideBottom', offset: -5, fontSize: labelFontSize }}
          />

          <YAxis
            type="number"
            dataKey="y"
            name={metricKey}
            tick={{ fontSize: tickFontSize }}
            label={{ value: metricKey, angle: -90, position: 'insideLeft', fontSize: labelFontSize }}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Trend line */}
          {actualShowTrendLine && trendLine.length === 2 && (
            <ReferenceLine
              segment={[
                { x: trendLine[0].x, y: trendLine[0].y },
                { x: trendLine[1].x, y: trendLine[1].y },
              ]}
              stroke={colors.primary}
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{
                value: `y = ${correlationData.regression.slope.toFixed(3)}x + ${correlationData.regression.intercept.toFixed(3)}`,
                position: 'top',
                fontSize: tickFontSize,
                fill: colors.muted,
              }}
            />
          )}

          {/* Normal points */}
          <Scatter
            name="Runs"
            data={normalPoints}
            fill={colors.primary}
            fillOpacity={actualOpacity}
            onClick={(data: any) => {
              if (onPointClick && data.runName) {
                onPointClick(data.runName);
              }
            }}
            cursor="pointer"
          />

          {/* Outlier points */}
          {showOutliers && outlierPoints.length > 0 && (
            <Scatter
              name="Outliers"
              data={outlierPoints}
              fill={colors.error || colors.accent}
              fillOpacity={actualOpacity}
              shape="cross"
              onClick={(data: any) => {
                if (onPointClick && data.runName) {
                  onPointClick(data.runName);
                }
              }}
              cursor="pointer"
            />
          )}
        </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Statistics summary - hide in compact mode when not fullscreen */}
        {(variant !== 'compact' || isFullscreen) && (
          <div className={`mt-2 grid grid-cols-2 gap-3 text-sm border-t pt-2 ${isFullscreen ? 'flex-shrink-0' : ''}`}>
            <div>
              <h4 className="font-medium mb-1">{hyperparamKey} Statistics</h4>
              <div className="space-y-0.5 text-xs text-muted-foreground">
                <div className="flex justify-between">
                  <span>Range:</span>
                  <span className="font-mono">
                    [{correlationData.stats.x.min.toFixed(3)}, {correlationData.stats.x.max.toFixed(3)}]
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Mean:</span>
                  <span className="font-mono">{correlationData.stats.x.mean.toFixed(3)}</span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-1">{metricKey} Statistics</h4>
              <div className="space-y-0.5 text-xs text-muted-foreground">
                <div className="flex justify-between">
                  <span>Range:</span>
                  <span className="font-mono">
                    [{correlationData.stats.y.min.toFixed(3)}, {correlationData.stats.y.max.toFixed(3)}]
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Mean:</span>
                  <span className="font-mono">{correlationData.stats.y.mean.toFixed(3)}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <ChartCard
      title={title || `${hyperparamKey} vs ${metricKey}`}
      description={`${correlationData.points.length} runs analyzed`}
      isEmpty={correlationData.points.length === 0}
      emptyMessage="No valid data points for correlation analysis"
      variant={variant}
      enableSettings={false}
      enableFullscreen={true}
      enableExport={true}
      exportFormats={['png', 'svg', 'csv']}
      onExport={handleExport}
      exportRef={chartContainerRef}
      height={settings.height || height}
      customControls={
        <ChartSettingsPanel
          settings={settings}
          onSettingsChange={setSettings}
          chartType="scatter"
          renderer="recharts"
        />
      }
      fullscreenChildren={renderChart(true)}
    >
      {renderChart(false)}
    </ChartCard>
  );
}
