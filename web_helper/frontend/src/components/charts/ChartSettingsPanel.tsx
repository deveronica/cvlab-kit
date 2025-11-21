import React from "react";
/**
 * Chart Settings Panel
 *
 * UI component for configuring chart appearance (colors, legend, axes, etc.)
 */

import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Settings2 } from 'lucide-react';
import type { LegendConfig, AxisConfig, ChartRenderer } from '@/lib/charts/types';

export interface ChartSettings {
  /** Legend configuration */
  legend?: LegendConfig;
  /** X-axis configuration */
  xAxis?: AxisConfig;
  /** Y-axis configuration */
  yAxis?: AxisConfig;
  /** Enable animations */
  animation?: boolean;
  /** Chart height (responsive is always enabled) */
  height?: number;
  /** Line/Area chart: stroke width */
  strokeWidth?: number;
  /** Line/Area/Scatter chart: show data points */
  showDots?: boolean;
  /** Line chart: curve type */
  curveType?: 'linear' | 'monotone' | 'step';
  /** Area chart: fill opacity */
  fillOpacity?: number;
  /** Bar/Area chart: stacking mode */
  stackMode?: 'none' | 'normal' | 'percent';
  /** Scatter chart: dot size */
  dotSize?: number;
  /** Heatmap-specific settings */
  heatmap?: {
    colorPalette?: 'blue-red' | 'viridis' | 'coolwarm' | 'spectral';
    showValues?: boolean;
    minCorrelation?: number;
    maxCorrelation?: number;
  };
  /** Histogram-specific settings */
  histogram?: {
    binCount?: number;
    showMean?: boolean;
    showMedian?: boolean;
  };
}

interface ChartSettingsPanelProps {
  /** Current settings */
  settings: ChartSettings;
  /** Callback when settings change */
  onSettingsChange: (settings: ChartSettings) => void;
  /** Chart type for type-specific settings */
  chartType?: 'line' | 'area' | 'bar' | 'scatter' | 'pie' | 'radar' | 'heatmap' | 'histogram';
  /** Chart renderer for renderer-specific settings */
  renderer?: ChartRenderer;
  /** className for container */
  className?: string;
}

/**
 * Feature support matrix by renderer
 * Only includes features that are CURRENTLY IMPLEMENTED and working in adapters
 *
 * IMPORTANT: Setting to 'true' means the option is actually implemented in the adapter
 * and will work when changed in Settings Panel. Don't set to true unless it's implemented!
 */
const RENDERER_FEATURES = {
  recharts: {
    animation: true,        // ✓ Implemented: isAnimationActive prop
    strokeWidth: true,      // ✓ Implemented: Line/Area strokeWidth
    showDots: true,         // ✓ Implemented: Line dot visibility
    curveType: true,        // ✓ Implemented: Line/Area type prop
    fillOpacity: true,      // ✓ Implemented: Area fillOpacity
    stackMode: true,        // ✓ Implemented: stackId prop (Phase 2)
    dotSize: true,          // ✓ Implemented: Scatter r prop (Phase 2)
    yAxisScale: true,       // ✓ Implemented: YAxis scale prop (linear/log/sqrt)
  },
  chartjs: {
    animation: true,        // ✓ Implemented: animation config in options
    strokeWidth: true,      // ✓ Implemented: borderWidth for Line/Area
    showDots: true,         // ✓ Implemented: pointRadius for Line charts
    curveType: true,        // ✓ Implemented: tension mapping for Line/Area
    fillOpacity: true,      // ✓ Implemented: backgroundColor opacity for Area
    stackMode: false,       // ✗ NOT implemented (requires stacked config)
    dotSize: false,         // ✗ NOT easily configurable
    yAxisScale: true,       // ✓ Implemented: scales.y.type (linear/logarithmic)
  },
  plotly: {
    animation: false,       // ✗ Plotly manages its own animation
    strokeWidth: false,     // ✗ Plotly manages via line.width (internal)
    showDots: false,        // ✗ Plotly manages via mode (internal)
    curveType: false,       // ✗ Plotly manages via line.shape (internal)
    fillOpacity: false,     // ✗ Plotly manages via fillcolor alpha (internal)
    stackMode: false,       // ✗ Plotly manages via stackgroup (internal)
    dotSize: false,         // ✗ Plotly manages via marker.size (internal)
    yAxisScale: true,       // ✓ Implemented: yaxis.type (linear/log/sqrt)
  },
} as const;

export function ChartSettingsPanel({
  settings,
  onSettingsChange,
  chartType,
  renderer = 'recharts',
  className,
}: ChartSettingsPanelProps) {

  const features = RENDERER_FEATURES[renderer] || RENDERER_FEATURES.recharts;

  const updateSetting = <K extends keyof ChartSettings>(
    key: K,
    value: ChartSettings[K]
  ) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  const updateLegend = (updates: Partial<LegendConfig>) => {
    updateSetting('legend', { ...settings.legend, ...updates });
  };

  const updateXAxis = (updates: Partial<AxisConfig>) => {
    updateSetting('xAxis', { ...settings.xAxis, ...updates });
  };

  const updateYAxis = (updates: Partial<AxisConfig>) => {
    updateSetting('yAxis', { ...settings.yAxis, ...updates });
  };

  const updateHeatmap = (updates: Partial<NonNullable<ChartSettings['heatmap']>>) => {
    updateSetting('heatmap', { ...settings.heatmap, ...updates });
  };

  const updateHistogram = (updates: Partial<NonNullable<ChartSettings['histogram']>>) => {
    updateSetting('histogram', { ...settings.histogram, ...updates });
  };

  // Determine which tabs to show based on chart type
  const showLegendTab = chartType !== 'heatmap' && chartType !== 'histogram';
  const showAxesTab = chartType !== 'heatmap';
  const showHeatmapTab = chartType === 'heatmap';
  const showHistogramTab = chartType === 'histogram';

  const tabCount = 1 + (showLegendTab ? 1 : 0) + (showAxesTab ? 1 : 0) + (showHeatmapTab ? 1 : 0) + (showHistogramTab ? 1 : 0);

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8"
          title="Chart settings"
        >
          <Settings2 className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" align="end">
        <Tabs defaultValue="general" className="w-full">
          <TabsList className={`grid w-full mb-3`} style={{ gridTemplateColumns: `repeat(${tabCount}, 1fr)` }}>
            <TabsTrigger value="general" className="text-xs">General</TabsTrigger>
            {showLegendTab && <TabsTrigger value="legend" className="text-xs">Legend</TabsTrigger>}
            {showAxesTab && <TabsTrigger value="axes" className="text-xs">Axes</TabsTrigger>}
            {showHeatmapTab && <TabsTrigger value="heatmap" className="text-xs">Heatmap</TabsTrigger>}
            {showHistogramTab && <TabsTrigger value="histogram" className="text-xs">Histogram</TabsTrigger>}
          </TabsList>

          {/* General Tab */}
          <TabsContent value="general" className="space-y-4 mt-0">
            <div className="space-y-3">
              {features.animation && (
                <div className="flex items-center justify-between">
                  <Label htmlFor="animation" className="text-sm">
                    Animation
                  </Label>
                  <Switch
                    id="animation"
                    checked={settings.animation !== false}
                    onCheckedChange={(checked) => updateSetting('animation', checked)}
                  />
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="height" className="text-sm">
                  Height (px)
                </Label>
                <Input
                  id="height"
                  type="number"
                  min={200}
                  max={1000}
                  step={50}
                  value={settings.height || 400}
                  onChange={(e) => updateSetting('height', parseInt(e.target.value) || 400)}
                  className="h-8"
                />
              </div>

              {/* Chart Type Specific Settings */}
              {(chartType === 'line' || chartType === 'area') && (
                <>
                  <div className="border-t pt-3 space-y-3">
                    <Label className="text-sm font-semibold">
                      {chartType === 'line' ? 'Line' : 'Area'} Chart Settings
                    </Label>

                    {features.strokeWidth && (
                      <div className="space-y-2">
                        <Label htmlFor="strokeWidth" className="text-sm">
                          Stroke Width: {settings.strokeWidth || 2}px
                        </Label>
                        <input
                          id="strokeWidth"
                          type="range"
                          min="1"
                          max="5"
                          value={settings.strokeWidth || 2}
                          onChange={(e) => updateSetting('strokeWidth', parseInt(e.target.value))}
                          className="w-full"
                        />
                      </div>
                    )}

                    {features.showDots && (
                      <div className="flex items-center justify-between">
                        <Label htmlFor="showDots" className="text-sm">
                          Show Data Points
                        </Label>
                        <Switch
                          id="showDots"
                          checked={settings.showDots || false}
                          onCheckedChange={(checked) => updateSetting('showDots', checked)}
                        />
                      </div>
                    )}

                    {chartType === 'line' && features.curveType && (
                      <div className="space-y-2">
                        <Label htmlFor="curveType" className="text-sm">
                          Curve Type
                        </Label>
                        <Select
                          value={settings.curveType || 'monotone'}
                          onValueChange={(value: any) => updateSetting('curveType', value)}
                        >
                          <SelectTrigger id="curveType" className="h-8">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="linear">Linear</SelectItem>
                            <SelectItem value="monotone">Smooth</SelectItem>
                            <SelectItem value="step">Step</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    )}

                    {chartType === 'area' && features.fillOpacity && (
                      <div className="space-y-2">
                        <Label htmlFor="fillOpacity" className="text-sm">
                          Fill Opacity: {Math.round((settings.fillOpacity || 0.3) * 100)}%
                        </Label>
                        <input
                          id="fillOpacity"
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={settings.fillOpacity || 0.3}
                          onChange={(e) => updateSetting('fillOpacity', parseFloat(e.target.value))}
                          className="w-full"
                        />
                      </div>
                    )}
                  </div>
                </>
              )}

              {chartType === 'scatter' && features.dotSize && (
                <div className="border-t pt-3 space-y-3">
                  <Label className="text-sm font-semibold">Scatter Chart Settings</Label>

                  <div className="space-y-2">
                    <Label htmlFor="dotSize" className="text-sm">
                      Dot Size: {settings.dotSize || 4}
                    </Label>
                    <input
                      id="dotSize"
                      type="range"
                      min="2"
                      max="10"
                      value={settings.dotSize || 4}
                      onChange={(e) => updateSetting('dotSize', parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              )}

              {(chartType === 'bar' || chartType === 'area') && features.stackMode && (
                <div className="border-t pt-3 space-y-3">
                  <Label className="text-sm font-semibold">Stacking</Label>

                  <div className="space-y-2">
                    <Label htmlFor="stackMode" className="text-sm">
                      Stack Mode
                    </Label>
                    <Select
                      value={settings.stackMode || 'none'}
                      onValueChange={(value: any) => updateSetting('stackMode', value)}
                    >
                      <SelectTrigger id="stackMode" className="h-8">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">None</SelectItem>
                        <SelectItem value="normal">Normal</SelectItem>
                        <SelectItem value="percent">Percent</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}
            </div>

            {/* Reset Button */}
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                onSettingsChange({
                  animation: true,
                  height: 400,
                  legend: { show: true, position: 'top', align: 'center' },
                  xAxis: { showGrid: true },
                  yAxis: { showGrid: true, scale: 'linear' },
                  strokeWidth: 2,
                  showDots: false,
                  curveType: 'monotone',
                  fillOpacity: 0.3,
                  stackMode: 'none',
                  dotSize: 4,
                })
              }
              className="w-full"
            >
              Reset to Defaults
            </Button>
          </TabsContent>

          {/* Legend Tab */}
          <TabsContent value="legend" className="space-y-4 mt-0">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="legend-show" className="text-sm">
                  Show Legend
                </Label>
                <Switch
                  id="legend-show"
                  checked={settings.legend?.show !== false}
                  onCheckedChange={(checked) => updateLegend({ show: checked })}
                />
              </div>

              {settings.legend?.show !== false && (
                <>
                  <div className="space-y-2">
                    <Label htmlFor="legend-position" className="text-sm">
                      Position
                    </Label>
                    <Select
                      value={settings.legend?.position || 'top'}
                      onValueChange={(value: any) => updateLegend({ position: value })}
                    >
                      <SelectTrigger id="legend-position" className="h-8">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="top">Top</SelectItem>
                        <SelectItem value="right">Right</SelectItem>
                        <SelectItem value="bottom">Bottom</SelectItem>
                        <SelectItem value="left">Left</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="legend-align" className="text-sm">
                      Alignment
                    </Label>
                    <Select
                      value={settings.legend?.align || 'center'}
                      onValueChange={(value: any) => updateLegend({ align: value })}
                    >
                      <SelectTrigger id="legend-align" className="h-8">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="start">Start</SelectItem>
                        <SelectItem value="center">Center</SelectItem>
                        <SelectItem value="end">End</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </>
              )}
            </div>
          </TabsContent>

          {/* Axes Tab */}
          <TabsContent value="axes" className="space-y-4 mt-0">
            <div className="space-y-4">
              {/* X-Axis Section */}
              <div className="space-y-3">
                <Label className="text-sm font-semibold">X-Axis</Label>

                <div className="space-y-2">
                  <Label htmlFor="xaxis-label" className="text-sm">
                    Label
                  </Label>
                  <Input
                    id="xaxis-label"
                    placeholder="e.g., Epoch"
                    value={settings.xAxis?.label || ''}
                    onChange={(e) => updateXAxis({ label: e.target.value })}
                    className="h-8"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="xaxis-grid" className="text-sm">
                    Show Grid
                  </Label>
                  <Switch
                    id="xaxis-grid"
                    checked={settings.xAxis?.showGrid !== false}
                    onCheckedChange={(checked) => updateXAxis({ showGrid: checked })}
                  />
                </div>
              </div>

              {/* Y-Axis Section */}
              <div className="space-y-3 border-t border-border pt-3">
                <Label className="text-sm font-semibold">Y-Axis</Label>

                <div className="space-y-2">
                  <Label htmlFor="yaxis-label" className="text-sm">
                    Label
                  </Label>
                  <Input
                    id="yaxis-label"
                    placeholder="e.g., Loss"
                    value={settings.yAxis?.label || ''}
                    onChange={(e) => updateYAxis({ label: e.target.value })}
                    className="h-8"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="yaxis-grid" className="text-sm">
                    Show Grid
                  </Label>
                  <Switch
                    id="yaxis-grid"
                    checked={settings.yAxis?.showGrid !== false}
                    onCheckedChange={(checked) => updateYAxis({ showGrid: checked })}
                  />
                </div>

                {features.yAxisScale && (
                  <div className="space-y-2">
                    <Label htmlFor="yaxis-scale" className="text-sm">
                      Scale
                    </Label>
                    <Select
                      value={settings.yAxis?.scale || 'linear'}
                      onValueChange={(value: any) => updateYAxis({ scale: value })}
                    >
                      <SelectTrigger id="yaxis-scale" className="h-8">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="linear">Linear</SelectItem>
                        <SelectItem value="log">Logarithmic</SelectItem>
                        <SelectItem value="sqrt">Square Root</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          {/* Heatmap Tab */}
          {showHeatmapTab && (
            <TabsContent value="heatmap" className="space-y-4 mt-0">
              <div className="space-y-3">
                <div className="space-y-2">
                  <Label htmlFor="color-palette" className="text-sm">
                    Color Palette
                  </Label>
                  <Select
                    value={settings.heatmap?.colorPalette || 'blue-red'}
                    onValueChange={(value: any) => updateHeatmap({ colorPalette: value })}
                  >
                    <SelectTrigger id="color-palette" className="h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="blue-red">Blue-Red</SelectItem>
                      <SelectItem value="viridis">Viridis</SelectItem>
                      <SelectItem value="coolwarm">Cool-Warm</SelectItem>
                      <SelectItem value="spectral">Spectral</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="show-values" className="text-sm">
                    Show Values
                  </Label>
                  <Switch
                    id="show-values"
                    checked={settings.heatmap?.showValues !== false}
                    onCheckedChange={(checked) => updateHeatmap({ showValues: checked })}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="min-correlation" className="text-sm">
                    Min Correlation: {settings.heatmap?.minCorrelation ?? -1}
                  </Label>
                  <Input
                    id="min-correlation"
                    type="number"
                    min={-1}
                    max={1}
                    step={0.1}
                    value={settings.heatmap?.minCorrelation ?? -1}
                    onChange={(e) => updateHeatmap({ minCorrelation: parseFloat(e.target.value) })}
                    className="h-8"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="max-correlation" className="text-sm">
                    Max Correlation: {settings.heatmap?.maxCorrelation ?? 1}
                  </Label>
                  <Input
                    id="max-correlation"
                    type="number"
                    min={-1}
                    max={1}
                    step={0.1}
                    value={settings.heatmap?.maxCorrelation ?? 1}
                    onChange={(e) => updateHeatmap({ maxCorrelation: parseFloat(e.target.value) })}
                    className="h-8"
                  />
                </div>
              </div>
            </TabsContent>
          )}

          {/* Histogram Tab */}
          {showHistogramTab && (
            <TabsContent value="histogram" className="space-y-4 mt-0">
              <div className="space-y-3">
                <div className="space-y-2">
                  <Label htmlFor="bin-count" className="text-sm">
                    Bin Count: {settings.histogram?.binCount || 20}
                  </Label>
                  <Input
                    id="bin-count"
                    type="number"
                    min={5}
                    max={50}
                    value={settings.histogram?.binCount || 20}
                    onChange={(e) => updateHistogram({ binCount: parseInt(e.target.value) })}
                    className="h-8"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="show-mean" className="text-sm">
                    Show Mean Line
                  </Label>
                  <Switch
                    id="show-mean"
                    checked={settings.histogram?.showMean === true}
                    onCheckedChange={(checked) => updateHistogram({ showMean: checked })}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="show-median" className="text-sm">
                    Show Median Line
                  </Label>
                  <Switch
                    id="show-median"
                    checked={settings.histogram?.showMedian === true}
                    onCheckedChange={(checked) => updateHistogram({ showMedian: checked })}
                  />
                </div>
              </div>
            </TabsContent>
          )}
        </Tabs>
      </PopoverContent>
    </Popover>
  );
}
