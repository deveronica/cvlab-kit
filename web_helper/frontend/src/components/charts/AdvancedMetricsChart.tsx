import React from "react";

/**
 * Advanced Metrics Visualization Component
 * Production-level charting with granular controls for ML experiment metrics
 *
 * Features:
 * - Multiple chart types (Line, Area, Bar, Scatter)
 * - Zoom & Pan controls
 * - Brush for time range selection
 * - Export to PNG/SVG
 * - Data smoothing & downsampling
 * - Custom styling & theming
 * - Real-time data updates
 * - Responsive design
 */

import { useState, useMemo, useCallback } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  _ReferenceLine,
  _ReferenceArea,
  _TooltipProps,
} from 'recharts';
import { ChartCard } from '../ui/chart-card';
import { Button } from '../ui/button';
import {
  Settings,
  TrendingUp,
  BarChart as BarChartIcon,
  Activity,
  Circle,
} from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from '../ui/popover';
import { Switch } from '../ui/switch';
import { Label } from '../ui/label';
import { ToggleGroup, ToggleGroupItem } from '../ui/toggle-group';
import { type ExportFormat } from '../ui/export-menu';
import type { Run } from '../../lib/types';
import { generateChartFilename } from '../../lib/chart-utils';
import { devError, devWarn } from '../../lib/dev-utils';

type ChartType = 'line' | 'area' | 'bar' | 'scatter';
type SmoothingType = 'none' | 'moving-average' | 'exponential';

interface AdvancedMetricsChartProps {
  runs: Run[];
  metricKey: string;
  title?: string;
  height?: number;
  defaultChartType?: ChartType;
  _enableZoom?: boolean;
  enableBrush?: boolean;
  enableSmoothing?: boolean;
  enableExport?: boolean;
  showControls?: boolean;
  colors?: string[];
  variant?: 'default' | 'compact';
  customControls?: React.ReactNode;
}

interface ChartDataPoint {
  step: number;
  [runId: string]: number | string;
}

interface ChartControls {
  chartType: ChartType;
  smoothing: SmoothingType;
  smoothingWindow: number;
  showGrid: boolean;
  showLegend: boolean;
  showDots: boolean;
  strokeWidth: number;
  zoomDomain: [number, number] | null;
}

const DEFAULT_COLORS = [
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#f97316', // orange
  '#84cc16', // lime
  '#ec4899', // pink
  '#6366f1', // indigo
  '#14b8a6', // teal
];

const CHART_TYPE_ICONS = {
  line: TrendingUp,
  area: Activity,
  bar: BarChartIcon,
  scatter: Circle,
};

export function AdvancedMetricsChart({
  runs,
  metricKey,
  title,
  height = 400,
  defaultChartType = 'line',
  _enableZoom = true,
  enableBrush = true,
  enableSmoothing = true,
  enableExport = true,
  showControls = true,
  colors = DEFAULT_COLORS,
  variant = 'default',
  customControls,
}: AdvancedMetricsChartProps) {
  const [controls, setControls] = useState<ChartControls>({
    chartType: defaultChartType,
    smoothing: 'none',
    smoothingWindow: 5,
    showGrid: true,
    showLegend: true,
    showDots: false,
    strokeWidth: 2,
    zoomDomain: null,
  });
  const chartContainerRef = React.useRef<HTMLDivElement>(null);

  // Smoothing functions
  const applySmoothing = useCallback((data: number[], windowSize: number, type: SmoothingType): number[] => {
    if (type === 'none' || data.length === 0) return data;

    if (type === 'moving-average') {
      return data.map((_, idx) => {
        const start = Math.max(0, idx - Math.floor(windowSize / 2));
        const end = Math.min(data.length, idx + Math.ceil(windowSize / 2));
        const window = data.slice(start, end);
        return window.reduce((sum, val) => sum + val, 0) / window.length;
      });
    }

    if (type === 'exponential') {
      const alpha = 2 / (windowSize + 1);
      const smoothed: number[] = [data[0]];
      for (let i = 1; i < data.length; i++) {
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1];
      }
      return smoothed;
    }

    return data;
  }, []);

  // Process chart data
  const chartData = useMemo(() => {
    if (!runs.length) return [];

    // Collect all unique steps
    const allSteps = new Set<number>();
    runs.forEach(run => {
      if (run.metrics?.timeseries) {
        run.metrics.timeseries.forEach(point => {
          allSteps.add(point.step);
        });
      }
    });

    const sortedSteps = Array.from(allSteps).sort((a, b) => a - b);

    // Create data points with smoothing
    return sortedSteps.map(step => {
      const dataPoint: ChartDataPoint = { step };

      runs.forEach(run => {
        const timeseriesPoint = run.metrics?.timeseries?.find(t => t.step === step);
        const value = timeseriesPoint?.values[metricKey];

        if (value !== undefined) {
          dataPoint[run.run_name] = value;
        }
      });

      return dataPoint;
    });
  }, [runs, metricKey]);

  // Apply smoothing to chart data
  const smoothedData = useMemo(() => {
    if (controls.smoothing === 'none') return chartData;

    return chartData.map((point, idx) => {
      const smoothedPoint = { ...point };

      runs.forEach(run => {
        const values = chartData.map(p => p[run.run_name] as number).filter(v => v !== undefined);
        if (values.length > 0) {
          const smoothed = applySmoothing(values, controls.smoothingWindow, controls.smoothing);
          smoothedPoint[run.run_name] = smoothed[idx];
        }
      });

      return smoothedPoint;
    });
  }, [chartData, controls.smoothing, controls.smoothingWindow, runs, applySmoothing]);

  // Apply zoom domain
  const displayData = useMemo(() => {
    if (!controls.zoomDomain) return smoothedData;

    const [start, end] = controls.zoomDomain;
    return smoothedData.filter(point => point.step >= start && point.step <= end);
  }, [smoothedData, controls.zoomDomain]);

  // Export handler
  const handleExport = useCallback(
    (format: ExportFormat) => {
      if (format === 'csv') {
        const runNames = runs.map(r => r.run_name);
        const headers = ['step', ...runNames.map(name => `${name}_${metricKey}`)];
        const csvRows = [headers.join(',')];

        displayData.forEach(point => {
          const row = [
            point.step,
            ...runNames.map(name => point[name] !== undefined ? point[name] : '')
          ];
          csvRows.push(row.join(','));
        });

        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = generateChartFilename(`${metricKey}_data`, 'csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        return;
      }

      // PNG and SVG export
      if (!chartContainerRef.current) {
        devWarn('Chart container ref not available');
        return;
      }

      const responsiveContainer = chartContainerRef.current.querySelector('.recharts-responsive-container');
      if (!responsiveContainer) {
        devWarn('No recharts container found for export');
        return;
      }

      const svgElement = responsiveContainer.querySelector('svg.recharts-surface') as SVGSVGElement;
      if (!svgElement) {
        devError('Could not find chart SVG for export', { metricKey });
        alert('차트를 찾을 수 없습니다. 페이지를 새로고침 후 다시 시도해주세요.');
        return;
      }

      if (format === 'svg') {
        const svgData = new XMLSerializer().serializeToString(svgElement);
        const blob = new Blob([svgData], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = generateChartFilename(`${metricKey}_chart`, 'svg');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        return;
      }

      if (format === 'png') {
        const width = svgElement.width.baseVal.value || svgElement.clientWidth || 800;
        const height = svgElement.height.baseVal.value || svgElement.clientHeight || 400;

        const clonedSvg = svgElement.cloneNode(true) as SVGSVGElement;
        clonedSvg.setAttribute('width', width.toString());
        clonedSvg.setAttribute('height', height.toString());

        const svgData = new XMLSerializer().serializeToString(clonedSvg);
        const canvas = document.createElement('canvas');

        const scale = 2;
        canvas.width = width * scale;
        canvas.height = height * scale;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const img = new Image();
        img.onload = () => {
          ctx.fillStyle = 'white';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.scale(scale, scale);
          ctx.drawImage(img, 0, 0);

          canvas.toBlob(blob => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = generateChartFilename(`${metricKey}_chart`, 'png');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
          }, 'image/png');
        };

        img.onerror = (err) => {
          devError('Failed to export chart as PNG:', err);
        };

        const encodedData = encodeURIComponent(svgData);
        img.src = 'data:image/svg+xml;charset=utf-8,' + encodedData;
      }
    },
    [metricKey, displayData, runs]
  );

  // Reset zoom
  const _resetZoom = useCallback(() => {
    setControls(prev => ({ ...prev, zoomDomain: null }));
  }, []);

  // Render chart based on type
  const renderChart = () => {
    const commonProps = {
      data: displayData,
      margin: { top: 10, right: 30, left: 0, bottom: 0 },
    };

    const commonAxisProps = {
      xAxis: {
        dataKey: "step",
        tick: { fontSize: 12 },
        label: { value: 'Step', position: 'insideBottom', offset: -5 },
      },
      yAxis: {
        tick: { fontSize: 12 },
        label: { value: metricKey, angle: -90, position: 'insideLeft' },
      },
    };

    const commonElements = (
      <>
        {controls.showGrid && <CartesianGrid strokeDasharray="3 3" className="opacity-30" />}
        <XAxis {...commonAxisProps.xAxis} />
        <YAxis {...commonAxisProps.yAxis} />
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--background))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '6px',
            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
          }}
          labelStyle={{ color: 'hsl(var(--foreground))' }}
        />
        {controls.showLegend && (
          <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="line" />
        )}
        {enableBrush && !controls.zoomDomain && (
          <Brush dataKey="step" height={30} stroke="hsl(var(--primary))" />
        )}
      </>
    );

    switch (controls.chartType) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            {commonElements}
            {runs.map((run, index) => (
              <Area
                key={run.run_name}
                type="monotone"
                dataKey={run.run_name}
                stroke={colors[index % colors.length]}
                fill={colors[index % colors.length]}
                fillOpacity={0.3}
                strokeWidth={controls.strokeWidth}
                dot={controls.showDots ? { r: 4, strokeWidth: 0, fill: colors[index % colors.length] } : false}
                activeDot={controls.showDots ? { r: 6, strokeWidth: 2, stroke: '#fff' } : false}
                name={`${run.run_name.substring(0, 12)}...`}
                connectNulls={false}
              />
            ))}
          </AreaChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            {commonElements}
            {runs.map((run, index) => (
              <Bar
                key={run.run_name}
                dataKey={run.run_name}
                fill={colors[index % colors.length]}
                name={`${run.run_name.substring(0, 12)}...`}
              />
            ))}
          </BarChart>
        );

      case 'scatter':
        return (
          <ScatterChart {...commonProps}>
            {commonElements}
            {runs.map((run, index) => (
              <Scatter
                key={run.run_name}
                dataKey={run.run_name}
                fill={colors[index % colors.length]}
                name={`${run.run_name.substring(0, 12)}...`}
              />
            ))}
          </ScatterChart>
        );

      case 'line':
      default:
        return (
          <LineChart {...commonProps}>
            {commonElements}
            {runs.map((run, index) => (
              <Line
                key={run.run_name}
                type="monotone"
                dataKey={run.run_name}
                stroke={colors[index % colors.length]}
                strokeWidth={controls.strokeWidth}
                dot={controls.showDots ? { r: 4, strokeWidth: 0, fill: colors[index % colors.length] } : false}
                activeDot={{ r: 6, strokeWidth: 2, stroke: '#fff' }}
                name={`${run.run_name.substring(0, 12)}...`}
                connectNulls={false}
              />
            ))}
          </LineChart>
        );
    }
  };

  // Chart type selector component
  const chartTypeSelector = showControls ? (
    <ToggleGroup
      type="single"
      value={controls.chartType}
      onValueChange={(value) => {
        if (value) setControls(prev => ({ ...prev, chartType: value as ChartType }));
      }}
      variant="outline"
      size="sm"
    >
      {Object.entries(CHART_TYPE_ICONS).map(([type, Icon]) => (
        <ToggleGroupItem
          key={type}
          value={type}
          aria-label={`${type} chart`}
          className="h-8 w-8 p-0"
        >
          <Icon className="h-4 w-4" />
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  ) : null;

  // Settings popover component
  const settingsPopover = showControls ? (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="h-8 w-8 p-0">
          <Settings className="h-3.5 w-3.5" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" align="end">
        <div className="space-y-4">
          <h4 className="font-medium text-sm">Chart Settings</h4>

          {/* Smoothing */}
          {enableSmoothing && (
            <div className="space-y-2">
              <Label className="text-sm">Smoothing</Label>
              <select
                className="w-full rounded border border-input bg-background px-3 py-1 text-sm"
                value={controls.smoothing}
                onChange={(e) => setControls(prev => ({ ...prev, smoothing: e.target.value as SmoothingType }))}
              >
                <option value="none">None</option>
                <option value="moving-average">Moving Average</option>
                <option value="exponential">Exponential</option>
              </select>

              {controls.smoothing !== 'none' && (
                <div className="flex items-center gap-2">
                  <Label className="text-sm min-w-fit">Window: {controls.smoothingWindow}</Label>
                  <input
                    type="range"
                    min="2"
                    max="20"
                    value={controls.smoothingWindow}
                    onChange={(e) => setControls(prev => ({ ...prev, smoothingWindow: parseInt(e.target.value) }))}
                    className="flex-1"
                  />
                </div>
              )}
            </div>
          )}

          {/* Visual Options */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Show Grid</Label>
              <Switch
                checked={controls.showGrid}
                onCheckedChange={(checked) => setControls(prev => ({ ...prev, showGrid: checked }))}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-sm">Show Legend</Label>
              <Switch
                checked={controls.showLegend}
                onCheckedChange={(checked) => setControls(prev => ({ ...prev, showLegend: checked }))}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-sm">Show Data Points</Label>
              <Switch
                checked={controls.showDots}
                onCheckedChange={(checked) => setControls(prev => ({ ...prev, showDots: checked }))}
              />
            </div>
          </div>

          {/* Line Width */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Label className="text-sm min-w-fit">Line Width: {controls.strokeWidth}px</Label>
              <input
                type="range"
                min="1"
                max="5"
                value={controls.strokeWidth}
                onChange={(e) => setControls(prev => ({ ...prev, strokeWidth: parseInt(e.target.value) }))}
                className="flex-1"
              />
            </div>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  ) : null;

  // Chart content
  const chartContent = (
    <div ref={chartContainerRef}>
      <ResponsiveContainer width="100%" height={height}>
        {renderChart()}
      </ResponsiveContainer>

      {/* Zoom Status */}
      {controls.zoomDomain && (
        <div className="mt-2 text-sm text-muted-foreground text-center">
          Showing steps {controls.zoomDomain[0]} to {controls.zoomDomain[1]}
        </div>
      )}
    </div>
  );

  return (
    <ChartCard
      title={title || `${metricKey} over time`}
      isEmpty={!chartData.length}
      emptyMessage={`No data available for ${metricKey}`}
      variant={variant}
      customControls={
        <>
          {customControls}
          {chartTypeSelector}
          {settingsPopover}
        </>
      }
      enableSettings={false}
      enableFullscreen={true}
      enableExport={enableExport}
      exportFormats={['png', 'svg', 'csv']}
      onExport={handleExport}
      exportRef={chartContainerRef}
      height={height}
    >
      {chartContent}
    </ChartCard>
  );
}
