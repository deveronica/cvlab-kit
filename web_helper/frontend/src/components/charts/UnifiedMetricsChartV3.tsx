import React from "react";

/**
 * Unified Metrics Chart V3
 * Simple, stable, and working version with smart two-level legend
 */

import { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Brush,
} from 'recharts';
import { ChartCard } from '../ui/chart-card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { TrendingUp, Settings, Activity, BarChart3, Search } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '../ui/dropdown-menu';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '../ui/popover';
import { Switch } from '../ui/switch';
import { Label } from '../ui/label';
import { ScrollArea } from '../ui/scroll-area';
import { ToggleGroup, ToggleGroupItem } from '../ui/toggle-group';
import { Input } from '../ui/input';
import type { Run } from '../../lib/types';

type ChartType = 'line' | 'area' | 'stacked-area';
type LineStyleMode = 'by-run' | 'by-metric' | 'none';

interface UnifiedMetricsChartV3Props {
  runs: Run[];
  availableMetrics: string[];
  title?: string;
  height?: number;
  defaultSelectedMetrics?: string[];
}

const COLORS = [
  '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444',
  '#f97316', '#84cc16', '#ec4899', '#6366f1', '#14b8a6',
  '#0ea5e9', '#22c55e', '#eab308', '#a855f7', '#f43f5e',
];

// Line dash patterns for differentiation
const LINE_STYLES = [
  '',           // solid
  '5 5',        // dashed
  '2 3',        // dotted
  '10 5',       // long dash
  '5 2 2 2',    // dash-dot
];

// Line widths for differentiation
const LINE_WIDTHS = [2, 2.5, 3, 1.5, 2];

// Custom Two-Level Legend Component
interface TwoLevelLegendProps {
  runs: Run[];
  metrics: string[];
  colors: Map<string, string>;
  lineStyles: Map<string, string>;
  hiddenItems: Set<string>;
  onToggleItem: (key: string) => void;
  compact?: boolean;
}

function TwoLevelLegend({
  runs,
  metrics,
  colors,
  lineStyles,
  hiddenItems,
  onToggleItem,
  compact = false,
}: TwoLevelLegendProps) {
  const [searchQuery, setSearchQuery] = useState('');

  // Filter runs and metrics based on search
  const filteredRuns = useMemo(() => {
    if (!searchQuery) return runs;
    const query = searchQuery.toLowerCase();
    return runs.filter(run => run.run_name.toLowerCase().includes(query));
  }, [runs, searchQuery]);

  const filteredMetrics = useMemo(() => {
    if (!searchQuery) return metrics;
    const query = searchQuery.toLowerCase();
    return metrics.filter(metric => metric.toLowerCase().includes(query));
  }, [metrics, searchQuery]);

  if (compact) {
    // Compact mode: Show runs and metrics separately
    return (
      <div className="flex flex-col gap-3 p-3 bg-muted/30 rounded-md border">
        {/* Search */}
        {(runs.length > 5 || metrics.length > 5) && (
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
            <Input
              placeholder="Search runs or metrics..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-7 pl-7 text-xs"
            />
          </div>
        )}

        <div className="grid grid-cols-2 gap-4">
          {/* Runs Section */}
          <div className="space-y-2">
            <div className="text-xs font-semibold text-muted-foreground">
              Runs ({filteredRuns.length})
            </div>
            <ScrollArea className="h-32">
              <div className="space-y-1.5">
                {filteredRuns.map((run, idx) => {
                  const colorKey = `run_${idx}`;
                  const color = colors.get(colorKey) || COLORS[idx % COLORS.length];
                  const lineStyle = lineStyles.get(colorKey) || '';

                  return (
                    <button
                      key={run.run_name}
                      onClick={() => onToggleItem(`run_${run.run_name}`)}
                      className="flex items-center gap-2 w-full text-left hover:bg-muted/50 rounded px-1.5 py-1 transition-colors"
                    >
                      <svg width="20" height="2" className="flex-shrink-0">
                        <line
                          x1="0"
                          y1="1"
                          x2="20"
                          y2="1"
                          stroke={color}
                          strokeWidth="2"
                          strokeDasharray={lineStyle}
                          opacity={hiddenItems.has(`run_${run.run_name}`) ? 0.3 : 1}
                        />
                      </svg>
                      <span
                        className="text-xs truncate flex-1"
                        style={{
                          opacity: hiddenItems.has(`run_${run.run_name}`) ? 0.4 : 1,
                          textDecoration: hiddenItems.has(`run_${run.run_name}`) ? 'line-through' : 'none'
                        }}
                        title={run.run_name}
                      >
                        {run.run_name}
                      </span>
                    </button>
                  );
                })}
              </div>
            </ScrollArea>
          </div>

          {/* Metrics Section */}
          <div className="space-y-2">
            <div className="text-xs font-semibold text-muted-foreground">
              Metrics ({filteredMetrics.length})
            </div>
            <ScrollArea className="h-32">
              <div className="space-y-1.5">
                {filteredMetrics.map((metric, idx) => {
                  const colorKey = `metric_${idx}`;
                  const color = colors.get(colorKey) || COLORS[idx % COLORS.length];
                  const lineStyle = lineStyles.get(colorKey) || '';

                  return (
                    <button
                      key={metric}
                      onClick={() => onToggleItem(`metric_${metric}`)}
                      className="flex items-center gap-2 w-full text-left hover:bg-muted/50 rounded px-1.5 py-1 transition-colors"
                    >
                      <svg width="20" height="2" className="flex-shrink-0">
                        <line
                          x1="0"
                          y1="1"
                          x2="20"
                          y2="1"
                          stroke={color}
                          strokeWidth="2"
                          strokeDasharray={lineStyle}
                          opacity={hiddenItems.has(`metric_${metric}`) ? 0.3 : 1}
                        />
                      </svg>
                      <span
                        className="text-xs truncate flex-1"
                        style={{
                          opacity: hiddenItems.has(`metric_${metric}`) ? 0.4 : 1,
                          textDecoration: hiddenItems.has(`metric_${metric}`) ? 'line-through' : 'none'
                        }}
                        title={metric}
                      >
                        {metric}
                      </span>
                    </button>
                  );
                })}
              </div>
            </ScrollArea>
          </div>
        </div>
      </div>
    );
  }

  // Standard mode: Show all run × metric combinations
  const allItems = useMemo(() => {
    const items: Array<{ key: string; runName: string; metric: string; color: string; lineStyle: string }> = [];
    runs.forEach((run) => {
      metrics.forEach((metric) => {
        const key = `${run.run_name}_${metric}`;
        const colorKey = `${run.run_name}_${metric}`;
        const color = colors.get(colorKey) || '#8b5cf6';
        const lineStyle = lineStyles.get(key) || '';

        if (!searchQuery ||
            run.run_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            metric.toLowerCase().includes(searchQuery.toLowerCase())) {
          items.push({ key, runName: run.run_name, metric, color, lineStyle });
        }
      });
    });
    return items;
  }, [runs, metrics, colors, lineStyles, searchQuery]);

  return (
    <div className="flex flex-col gap-2 p-2 bg-muted/30 rounded-md border">
      {/* Search */}
      {allItems.length > 10 && (
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
          <Input
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="h-7 pl-7 text-xs"
          />
        </div>
      )}

      {/* Items */}
      <ScrollArea className="h-24">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {allItems.map(({ key, runName, metric, color, lineStyle }) => (
            <button
              key={key}
              onClick={() => onToggleItem(key)}
              className="flex items-center gap-1.5 text-left hover:bg-muted/50 rounded px-1 py-0.5 transition-colors"
            >
              <svg width="16" height="2" className="flex-shrink-0">
                <line
                  x1="0"
                  y1="1"
                  x2="16"
                  y2="1"
                  stroke={color}
                  strokeWidth="2"
                  strokeDasharray={lineStyle}
                  opacity={hiddenItems.has(key) ? 0.3 : 1}
                />
              </svg>
              <span
                className="text-[10px] truncate flex-1"
                style={{
                  opacity: hiddenItems.has(key) ? 0.4 : 1,
                  textDecoration: hiddenItems.has(key) ? 'line-through' : 'none'
                }}
                title={`${runName} | ${metric}`}
              >
                {runName.substring(0, 15)}... | {metric}
              </span>
            </button>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}

export function UnifiedMetricsChartV3({
  runs,
  availableMetrics,
  title = 'Training Metrics',
  height = 550,
  defaultSelectedMetrics,
}: UnifiedMetricsChartV3Props) {
  const [selectedMetrics, setSelectedMetrics] = useState<Set<string>>(
    () => new Set(defaultSelectedMetrics || availableMetrics.slice(0, 5))
  );
  const [chartType, setChartType] = useState<ChartType>('line');
  const [showGrid, setShowGrid] = useState(true);
  const [showLegend, setShowLegend] = useState(true);
  const [showBrush, setShowBrush] = useState(true);
  const [lineStyleMode, setLineStyleMode] = useState<LineStyleMode>('by-run');
  const [useVariableWidth, setUseVariableWidth] = useState(false);
  const [compactLegend, setCompactLegend] = useState(true);
  const [hiddenItems, setHiddenItems] = useState<Set<string>>(new Set());

  // Build color and line style maps for legend
  const { colorMap, lineStyleMap } = useMemo(() => {
    const colors = new Map<string, string>();
    const styles = new Map<string, string>();
    let colorIndex = 0;

    runs.forEach((run, runIndex) => {
      Array.from(selectedMetrics).forEach((metric, metricIndex) => {
        const key = `${run.run_name}_${metric}`;
        const color = COLORS[colorIndex % COLORS.length];
        colors.set(key, color);

        // Run-based colors for compact mode
        colors.set(`run_${runIndex}`, COLORS[runIndex % COLORS.length]);

        // Metric-based colors for compact mode
        colors.set(`metric_${metricIndex}`, COLORS[metricIndex % COLORS.length]);

        // Determine line style based on mode
        let strokeDasharray = '';
        if (lineStyleMode === 'by-run') {
          strokeDasharray = LINE_STYLES[runIndex % LINE_STYLES.length];
          styles.set(`run_${runIndex}`, strokeDasharray);
        } else if (lineStyleMode === 'by-metric') {
          strokeDasharray = LINE_STYLES[metricIndex % LINE_STYLES.length];
          styles.set(`metric_${metricIndex}`, strokeDasharray);
        }
        styles.set(key, strokeDasharray);

        colorIndex++;
      });
    });

    return { colorMap: colors, lineStyleMap: styles };
  }, [runs, selectedMetrics, lineStyleMode]);

  // Build chart data
  const chartData = useMemo(() => {
    if (!runs.length || selectedMetrics.size === 0) return [];

    const allSteps = new Set<number>();
    runs.forEach((run) => {
      run.metrics?.timeseries?.forEach((point) => allSteps.add(point.step));
    });

    const sortedSteps = Array.from(allSteps).sort((a, b) => a - b);

    return sortedSteps.map((step) => {
      const dataPoint: any = { step };

      runs.forEach((run) => {
        selectedMetrics.forEach((metric) => {
          const point = run.metrics?.timeseries?.find((t) => t.step === step);
          const value = point?.values[metric];
          if (value !== undefined && !isNaN(value) && isFinite(value)) {
            dataPoint[`${run.run_name}_${metric}`] = value;
          }
        });
      });

      return dataPoint;
    });
  }, [runs, selectedMetrics]);

  // Build chart elements (lines or areas)
  const chartElements = useMemo(() => {
    const result: React.ReactElement[] = [];

    runs.forEach((run, runIndex) => {
      Array.from(selectedMetrics).forEach((metric, metricIndex) => {
        const key = `${run.run_name}_${metric}`;
        const color = colorMap.get(key) || COLORS[0];
        const name = `${run.run_name.substring(0, 20)}... | ${metric}`;

        // Check if item is hidden (either run or metric is hidden, or specific combo is hidden)
        const isHidden = hiddenItems.has(key) ||
                        hiddenItems.has(`run_${run.run_name}`) ||
                        hiddenItems.has(`metric_${metric}`);

        // Skip if hidden
        if (isHidden) return;

        // Determine line style based on mode
        let strokeDasharray = '';
        if (lineStyleMode === 'by-run') {
          strokeDasharray = LINE_STYLES[runIndex % LINE_STYLES.length];
        } else if (lineStyleMode === 'by-metric') {
          strokeDasharray = LINE_STYLES[metricIndex % LINE_STYLES.length];
        }

        // Determine stroke width
        const strokeWidth = useVariableWidth
          ? LINE_WIDTHS[metricIndex % LINE_WIDTHS.length]
          : 2;

        if (chartType === 'line') {
          result.push(
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={color}
              strokeWidth={strokeWidth}
              strokeDasharray={strokeDasharray}
              dot={false}
              activeDot={{ r: 5 }}
              name={name}
              connectNulls={false}
            />
          );
        } else if (chartType === 'area') {
          result.push(
            <Area
              key={key}
              type="monotone"
              dataKey={key}
              stroke={color}
              fill={color}
              fillOpacity={0.3}
              strokeWidth={strokeWidth}
              strokeDasharray={strokeDasharray}
              name={name}
              connectNulls={false}
            />
          );
        } else if (chartType === 'stacked-area') {
          result.push(
            <Area
              key={key}
              type="monotone"
              dataKey={key}
              stroke={color}
              fill={color}
              fillOpacity={0.6}
              strokeWidth={1}
              name={name}
              stackId="1"
              connectNulls={false}
            />
          );
        }
      });
    });

    return result;
  }, [runs, selectedMetrics, chartType, lineStyleMode, useVariableWidth, colorMap, hiddenItems]);

  // Toggle metric
  const toggleMetric = (metric: string) => {
    setSelectedMetrics((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(metric)) {
        newSet.delete(metric);
      } else {
        newSet.add(metric);
      }
      return newSet;
    });
  };

  // Toggle legend item visibility
  const toggleLegendItem = (key: string) => {
    setHiddenItems((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  };

  return (
    <ChartCard
      title={title}
      height={height}
      enableSettings={false}
      enableFullscreen={true}
      enableExport={false}
      contentClassName="p-0"
      customControls={
        <>
          {/* Chart Type Selector */}
          <ToggleGroup type="single" value={chartType} onValueChange={(v) => v && setChartType(v as ChartType)} className="h-8">
            <ToggleGroupItem value="line" aria-label="Line Chart" className="h-8 px-2">
              <Activity className="h-4 w-4" />
            </ToggleGroupItem>
            <ToggleGroupItem value="area" aria-label="Area Chart" className="h-8 px-2">
              <BarChart3 className="h-4 w-4" />
            </ToggleGroupItem>
            <ToggleGroupItem value="stacked-area" aria-label="Stacked Area" className="h-8 px-2">
              <BarChart3 className="h-4 w-4" />
            </ToggleGroupItem>
          </ToggleGroup>

          {/* Metric Selector */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="h-8">
                <TrendingUp className="h-4 w-4 mr-1" />
                Metrics ({selectedMetrics.size}/{availableMetrics.length})
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-80" align="end">
              <DropdownMenuLabel>Select Metrics</DropdownMenuLabel>
              <div className="flex gap-2 px-2 pb-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={() => setSelectedMetrics(new Set(availableMetrics))}
                >
                  All
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={() => setSelectedMetrics(new Set())}
                >
                  Clear
                </Button>
              </div>
              <DropdownMenuSeparator />
              <ScrollArea className="h-64">
                {availableMetrics.map((metric) => (
                  <DropdownMenuCheckboxItem
                    key={metric}
                    checked={selectedMetrics.has(metric)}
                    onCheckedChange={() => toggleMetric(metric)}
                  >
                    {metric}
                  </DropdownMenuCheckboxItem>
                ))}
              </ScrollArea>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Settings */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 w-8 p-0">
                <Settings className="h-4 w-4" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-72" align="end">
              <div className="space-y-4">
                <div>
                  <Label className="text-sm font-medium mb-2 block">Display Options</Label>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Show Grid</Label>
                      <Switch checked={showGrid} onCheckedChange={setShowGrid} />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Show Legend</Label>
                      <Switch checked={showLegend} onCheckedChange={setShowLegend} />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Compact Legend</Label>
                      <Switch checked={compactLegend} onCheckedChange={setCompactLegend} />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Show Brush</Label>
                      <Switch checked={showBrush} onCheckedChange={setShowBrush} />
                    </div>
                  </div>
                </div>

                <div className="border-t pt-3">
                  <Label className="text-sm font-medium mb-2 block">Line Differentiation</Label>
                  <div className="space-y-3">
                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">Line Style Mode</Label>
                      <ToggleGroup
                        type="single"
                        value={lineStyleMode}
                        onValueChange={(v) => v && setLineStyleMode(v as LineStyleMode)}
                        className="grid grid-cols-3 gap-1"
                      >
                        <ToggleGroupItem value="none" className="text-xs px-2 h-7">
                          None
                        </ToggleGroupItem>
                        <ToggleGroupItem value="by-run" className="text-xs px-2 h-7">
                          By Run
                        </ToggleGroupItem>
                        <ToggleGroupItem value="by-metric" className="text-xs px-2 h-7">
                          By Metric
                        </ToggleGroupItem>
                      </ToggleGroup>
                    </div>
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Variable Width</Label>
                      <Switch checked={useVariableWidth} onCheckedChange={setUseVariableWidth} />
                    </div>
                  </div>
                </div>
              </div>
            </PopoverContent>
          </Popover>
        </>
      }
    >
      <div
        style={{
          width: '100%',
          height: height || 400,
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Chart */}
        <div style={{
          flex: 1,
          minHeight: 0,
          marginBottom: '8px',
          position: 'relative',
          width: '100%',
        }}>
          {selectedMetrics.size === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
              Please select at least one metric
            </div>
          ) : (
            <div style={{ position: 'absolute', inset: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
              {chartType === 'line' ? (
                <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                  {showGrid && <CartesianGrid strokeDasharray="3 3" opacity={0.3} />}
                  <XAxis
                    dataKey="step"
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Step', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis
                    domain={['auto', 'auto']}
                    tick={{ fontSize: 12 }}
                    tickCount={8}
                    label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--background))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '6px',
                    }}
                  />
                  {showBrush && <Brush dataKey="step" height={30} stroke="hsl(var(--primary))" />}
                  {chartElements}
                </LineChart>
              ) : (
                <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                  {showGrid && <CartesianGrid strokeDasharray="3 3" opacity={0.3} />}
                  <XAxis
                    dataKey="step"
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Step', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis
                    domain={['auto', 'auto']}
                    tick={{ fontSize: 12 }}
                    tickCount={8}
                    label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--background))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '6px',
                    }}
                  />
                  {showBrush && <Brush dataKey="step" height={30} stroke="hsl(var(--primary))" />}
                  {chartElements}
                </AreaChart>
              )}
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Custom Two-Level Legend */}
        {showLegend && (
          <div style={{ flexShrink: 0, marginBottom: '8px' }}>
            <TwoLevelLegend
              runs={runs}
              metrics={Array.from(selectedMetrics)}
              colors={colorMap}
              lineStyles={lineStyleMap}
              hiddenItems={hiddenItems}
              onToggleItem={toggleLegendItem}
              compact={compactLegend}
            />
          </div>
        )}

        {/* Status */}
        <div
          className="flex gap-2 text-xs text-muted-foreground overflow-x-auto"
          style={{ flexShrink: 0, height: '32px', alignItems: 'center' }}
        >
          <Badge variant="outline" className="flex-shrink-0">
            {chartType === 'line' ? 'Line' : chartType === 'area' ? 'Area' : 'Stacked Area'}
          </Badge>
          <Badge variant="outline" className="flex-shrink-0">
            {runs.length} run{runs.length > 1 ? 's' : ''}
          </Badge>
          <Badge variant="outline" className="flex-shrink-0">
            {selectedMetrics.size} metric{selectedMetrics.size > 1 ? 's' : ''}
          </Badge>
        </div>
      </div>
    </ChartCard>
  );
}
