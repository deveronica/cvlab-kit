import React from "react";

/**
 * Unified Metrics Visualization Component
 * Advanced multi-metric charting for ML experiment comparison
 *
 * Features:
 * - Multiple metrics in one chart
 * - Normalization options (none, 0-1, z-score)
 * - Dual Y-axis support
 * - Metric selection (multi-select)
 * - Smoothing & filtering
 * - Export to PNG/SVG/CSV
 * - Tableau-grade customization
 * - Synchronized tooltips
 */

import { useState, useMemo, useCallback, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  _ReferenceLine,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Checkbox } from '../ui/checkbox';
import {
  Settings,
  TrendingUp,
  Expand,
  Download,
  _X,
  _ChevronDown,
  _ChevronUp,
} from 'lucide-react';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '../ui/popover';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '../ui/dropdown-menu';
import { Switch } from '../ui/switch';
import { Label } from '../ui/label';
import { ScrollArea } from '../ui/scroll-area';
import { Input } from '../ui/input';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import type { Run } from '../../lib/types';
import { generateChartFilename } from '../../lib/chart-utils';

type NormalizationType = 'none' | 'min-max' | 'z-score';
type SmoothingType = 'none' | 'moving-average' | 'exponential';

interface UnifiedMetricsChartProps {
  runs: Run[];
  availableMetrics: string[];
  title?: string;
  height?: number;
  defaultSelectedMetrics?: string[];
  enableExport?: boolean;
}

interface ChartDataPoint {
  step: number;
  [key: string]: number | string; // run_name_metric: value
}

interface ChartControls {
  selectedMetrics: Set<string>;
  normalization: NormalizationType;
  smoothing: SmoothingType;
  smoothingWindow: number;
  showGrid: boolean;
  showLegend: boolean;
  showDots: boolean;
  strokeWidth: number;
  showBrush: boolean;
}

const DEFAULT_COLORS = [
  '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444',
  '#f97316', '#84cc16', '#ec4899', '#6366f1', '#14b8a6',
  '#a855f7', '#3b82f6', '#22c55e', '#eab308', '#dc2626',
];

export function UnifiedMetricsChart({
  runs,
  availableMetrics,
  title = 'Training Metrics',
  height = 500,
  defaultSelectedMetrics,
  enableExport = true,
}: UnifiedMetricsChartProps) {
  const [controls, setControls] = useState<ChartControls>({
    selectedMetrics: new Set(defaultSelectedMetrics || availableMetrics.slice(0, 3)),
    normalization: 'none',
    smoothing: 'none',
    smoothingWindow: 5,
    showGrid: true,
    showLegend: true,
    showDots: false,
    strokeWidth: 2,
    showBrush: true,
  });

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [metricSearch, setMetricSearch] = useState('');
  const chartContainerRef = useRef<HTMLDivElement>(null);

  // Smoothing function
  const applySmoothing = useCallback(
    (data: number[], windowSize: number, type: SmoothingType): number[] => {
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
    },
    []
  );

  // Normalization functions
  const normalizeData = useCallback(
    (data: number[], type: NormalizationType): number[] => {
      if (type === 'none') return data;

      const values = data.filter((v) => !isNaN(v) && isFinite(v));
      if (values.length === 0) return data;

      if (type === 'min-max') {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min;
        if (range === 0) return data.map(() => 0);
        return data.map((v) => (isNaN(v) || !isFinite(v) ? NaN : (v - min) / range));
      }

      if (type === 'z-score') {
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const variance =
          values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        if (std === 0) return data.map(() => 0);
        return data.map((v) => (isNaN(v) || !isFinite(v) ? NaN : (v - mean) / std));
      }

      return data;
    },
    []
  );

  // Process chart data
  const chartData = useMemo(() => {
    if (!runs.length || controls.selectedMetrics.size === 0) return [];

    // Collect all unique steps
    const allSteps = new Set<number>();
    runs.forEach((run) => {
      if (run.metrics?.timeseries) {
        run.metrics.timeseries.forEach((point) => {
          allSteps.add(point.step);
        });
      }
    });

    const sortedSteps = Array.from(allSteps).sort((a, b) => a - b);

    // Collect raw data for each run-metric combination
    const rawDataMap = new Map<string, number[]>();

    runs.forEach((run) => {
      Array.from(controls.selectedMetrics).forEach((metric) => {
        const key = `${run.run_name}_${metric}`;
        const values: number[] = [];

        sortedSteps.forEach((step) => {
          const timeseriesPoint = run.metrics?.timeseries?.find((t) => t.step === step);
          const value = timeseriesPoint?.values[metric];
          values.push(value !== undefined ? value : NaN);
        });

        rawDataMap.set(key, values);
      });
    });

    // Apply smoothing
    const smoothedDataMap = new Map<string, number[]>();
    rawDataMap.forEach((values, key) => {
      const smoothed = applySmoothing(values, controls.smoothingWindow, controls.smoothing);
      smoothedDataMap.set(key, smoothed);
    });

    // Apply normalization
    const normalizedDataMap = new Map<string, number[]>();
    smoothedDataMap.forEach((values, key) => {
      const normalized = normalizeData(values, controls.normalization);
      normalizedDataMap.set(key, normalized);
    });

    // Build chart data points
    return sortedSteps.map((step, idx) => {
      const dataPoint: ChartDataPoint = { step };

      normalizedDataMap.forEach((values, key) => {
        const value = values[idx];
        if (!isNaN(value) && isFinite(value)) {
          dataPoint[key] = value;
        }
      });

      return dataPoint;
    });
  }, [runs, controls.selectedMetrics, controls.smoothing, controls.smoothingWindow, controls.normalization, applySmoothing, normalizeData]);

  // Toggle metric selection
  const toggleMetric = (metric: string) => {
    setControls((prev) => {
      const newSelected = new Set(prev.selectedMetrics);
      if (newSelected.has(metric)) {
        newSelected.delete(metric);
      } else {
        newSelected.add(metric);
      }
      return { ...prev, selectedMetrics: newSelected };
    });
  };

  // Select/deselect all metrics
  const selectAllMetrics = () => {
    setControls((prev) => ({
      ...prev,
      selectedMetrics: new Set(availableMetrics),
    }));
  };

  const deselectAllMetrics = () => {
    setControls((prev) => ({
      ...prev,
      selectedMetrics: new Set(),
    }));
  };

  // Filtered metrics for search
  const filteredMetrics = useMemo(() => {
    if (!metricSearch) return availableMetrics;
    return availableMetrics.filter((m) =>
      m.toLowerCase().includes(metricSearch.toLowerCase())
    );
  }, [availableMetrics, metricSearch]);

  // Export to PNG
  const exportToPNG = useCallback(() => {
    if (!chartContainerRef.current) return;

    const responsiveContainer = chartContainerRef.current.querySelector(
      '.recharts-responsive-container'
    );
    if (!responsiveContainer) return;

    const svgElement = responsiveContainer.querySelector(
      'svg.recharts-surface'
    ) as SVGSVGElement;
    if (!svgElement) return;

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

      canvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = generateChartFilename('unified_metrics', 'png');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      }, 'image/png');
    };

    const encodedData = encodeURIComponent(svgData);
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodedData;
  }, []);

  // Export to SVG
  const exportToSVG = useCallback(() => {
    if (!chartContainerRef.current) return;

    const responsiveContainer = chartContainerRef.current.querySelector(
      '.recharts-responsive-container'
    );
    if (!responsiveContainer) return;

    const svgElement = responsiveContainer.querySelector('svg.recharts-surface');
    if (!svgElement) return;

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const blob = new Blob([svgData], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = generateChartFilename('unified_metrics', 'svg');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  // Export to CSV
  const exportToCSV = useCallback(() => {
    const headers = ['step'];
    runs.forEach((run) => {
      Array.from(controls.selectedMetrics).forEach((metric) => {
        headers.push(`${run.run_name}_${metric}`);
      });
    });

    const csvRows = [headers.join(',')];

    chartData.forEach((point) => {
      const row = [point.step.toString()];
      runs.forEach((run) => {
        Array.from(controls.selectedMetrics).forEach((metric) => {
          const key = `${run.run_name}_${metric}`;
          row.push(point[key] !== undefined ? point[key].toString() : '');
        });
      });
      csvRows.push(row.join(','));
    });

    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = generateChartFilename('unified_metrics_data', 'csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [chartData, controls.selectedMetrics, runs]);

  // Render chart
  const renderChart = () => {
    const lines: React.ReactElement[] = [];
    let colorIndex = 0;

    runs.forEach((run) => {
      Array.from(controls.selectedMetrics).forEach((metric) => {
        const key = `${run.run_name}_${metric}`;
        const color = DEFAULT_COLORS[colorIndex % DEFAULT_COLORS.length];
        colorIndex++;

        lines.push(
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={color}
            strokeWidth={controls.strokeWidth}
            dot={controls.showDots ? { r: 3 } : false}
            activeDot={{ r: 5 }}
            name={`${run.run_name.substring(0, 12)}... | ${metric}`}
            connectNulls={false}
          />
        );
      });
    });

    return (
      <LineChart
        data={chartData}
        margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
      >
        {controls.showGrid && (
          <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
        )}
        <XAxis
          dataKey="step"
          tick={{ fontSize: 12 }}
          label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          label={{
            value: controls.normalization === 'none' ? 'Value' : 'Normalized Value',
            angle: -90,
            position: 'insideLeft',
          }}
        />
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
        {controls.showBrush && <Brush dataKey="step" height={30} stroke="hsl(var(--primary))" />}
        {lines}
      </LineChart>
    );
  };

  if (availableMetrics.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            No metrics available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <CardTitle className="truncate mr-2 flex-shrink min-w-0">{title}</CardTitle>

          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Metric Selector */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="h-9">
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Metrics ({controls.selectedMetrics.size}/{availableMetrics.length})
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-80" align="end">
                <DropdownMenuLabel>Select Metrics</DropdownMenuLabel>
                <div className="px-2 pb-2">
                  <Input
                    type="text"
                    placeholder="Search metrics..."
                    value={metricSearch}
                    onChange={(e) => setMetricSearch(e.target.value)}
                    className="h-8 text-sm"
                  />
                </div>
                <div className="flex gap-2 px-2 pb-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 h-7 text-xs"
                    onClick={selectAllMetrics}
                  >
                    Select All
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 h-7 text-xs"
                    onClick={deselectAllMetrics}
                  >
                    Clear
                  </Button>
                </div>
                <DropdownMenuSeparator />
                <ScrollArea className="h-64">
                  {filteredMetrics.map((metric) => (
                    <DropdownMenuCheckboxItem
                      key={metric}
                      checked={controls.selectedMetrics.has(metric)}
                      onCheckedChange={() => toggleMetric(metric)}
                    >
                      {metric}
                    </DropdownMenuCheckboxItem>
                  ))}
                </ScrollArea>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Fullscreen */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsFullscreen(true)}
              className="h-9"
              title="Fullscreen"
            >
              <Expand className="h-4 w-4" />
            </Button>

            {/* Export */}
            {enableExport && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="h-9">
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuCheckboxItem onClick={exportToPNG}>
                    Export as PNG
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuCheckboxItem onClick={exportToSVG}>
                    Export as SVG
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuCheckboxItem onClick={exportToCSV}>
                    Export as CSV
                  </DropdownMenuCheckboxItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}

            {/* Settings Popover */}
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" size="sm" className="h-9">
                  <Settings className="h-4 w-4" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-80" align="end">
                <div className="space-y-4">
                  <h4 className="font-medium text-sm">Chart Settings</h4>

                  {/* Normalization */}
                  <div className="space-y-2">
                    <Label className="text-sm">Normalization</Label>
                    <select
                      className="w-full rounded border border-input bg-background px-3 py-1 text-sm"
                      value={controls.normalization}
                      onChange={(e) =>
                        setControls((prev) => ({
                          ...prev,
                          normalization: e.target.value as NormalizationType,
                        }))
                      }
                    >
                      <option value="none">None</option>
                      <option value="min-max">Min-Max (0-1)</option>
                      <option value="z-score">Z-Score</option>
                    </select>
                  </div>

                  {/* Smoothing */}
                  <div className="space-y-2">
                    <Label className="text-sm">Smoothing</Label>
                    <select
                      className="w-full rounded border border-input bg-background px-3 py-1 text-sm"
                      value={controls.smoothing}
                      onChange={(e) =>
                        setControls((prev) => ({
                          ...prev,
                          smoothing: e.target.value as SmoothingType,
                        }))
                      }
                    >
                      <option value="none">None</option>
                      <option value="moving-average">Moving Average</option>
                      <option value="exponential">Exponential</option>
                    </select>

                    {controls.smoothing !== 'none' && (
                      <div className="flex items-center gap-2">
                        <Label className="text-sm min-w-fit">
                          Window: {controls.smoothingWindow}
                        </Label>
                        <input
                          type="range"
                          min="2"
                          max="20"
                          value={controls.smoothingWindow}
                          onChange={(e) =>
                            setControls((prev) => ({
                              ...prev,
                              smoothingWindow: parseInt(e.target.value),
                            }))
                          }
                          className="flex-1"
                        />
                      </div>
                    )}
                  </div>

                  {/* Visual Options */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Show Grid</Label>
                      <Switch
                        checked={controls.showGrid}
                        onCheckedChange={(checked) =>
                          setControls((prev) => ({ ...prev, showGrid: checked }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Show Legend</Label>
                      <Switch
                        checked={controls.showLegend}
                        onCheckedChange={(checked) =>
                          setControls((prev) => ({ ...prev, showLegend: checked }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Show Data Points</Label>
                      <Switch
                        checked={controls.showDots}
                        onCheckedChange={(checked) =>
                          setControls((prev) => ({ ...prev, showDots: checked }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Show Brush</Label>
                      <Switch
                        checked={controls.showBrush}
                        onCheckedChange={(checked) =>
                          setControls((prev) => ({ ...prev, showBrush: checked }))
                        }
                      />
                    </div>
                  </div>

                  {/* Line Width */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Label className="text-sm min-w-fit">
                        Line Width: {controls.strokeWidth}px
                      </Label>
                      <input
                        type="range"
                        min="1"
                        max="5"
                        value={controls.strokeWidth}
                        onChange={(e) =>
                          setControls((prev) => ({
                            ...prev,
                            strokeWidth: parseInt(e.target.value),
                          }))
                        }
                        className="flex-1"
                      />
                    </div>
                  </div>
                </div>
              </PopoverContent>
            </Popover>
          </div>
        </CardHeader>

        <CardContent ref={chartContainerRef}>
          {controls.selectedMetrics.size === 0 ? (
            <div className="flex items-center justify-center h-64 text-muted-foreground">
              Please select at least one metric
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={height}>
              {renderChart()}
            </ResponsiveContainer>
          )}

          {/* Status Info */}
          {controls.selectedMetrics.size > 0 && (
            <div className="mt-4 flex flex-wrap gap-2 text-xs text-muted-foreground">
              {controls.normalization !== 'none' && (
                <Badge variant="outline">
                  Normalization: {controls.normalization}
                </Badge>
              )}
              {controls.smoothing !== 'none' && (
                <Badge variant="outline">
                  Smoothing: {controls.smoothing} (window: {controls.smoothingWindow})
                </Badge>
              )}
              <Badge variant="outline">
                {runs.length} run{runs.length > 1 ? 's' : ''} × {controls.selectedMetrics.size}{' '}
                metric{controls.selectedMetrics.size > 1 ? 's' : ''}
              </Badge>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Fullscreen Modal */}
      <Dialog open={isFullscreen} onOpenChange={setIsFullscreen}>
        <DialogContent className="max-w-[95vw] max-h-[95vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle>{title}</DialogTitle>
          </DialogHeader>
          <div className="flex-1 min-h-[70vh] relative">
            {controls.selectedMetrics.size === 0 ? (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                Please select at least one metric
              </div>
            ) : (
              <div style={{ position: 'absolute', inset: 0 }}>
                <ResponsiveContainer width="100%" height="100%">
                  {renderChart()}
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
