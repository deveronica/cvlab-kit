import React from "react";
/**
 * Chart Settings Menu
 *
 * Reusable settings menu component for all chart types
 * Provides a dropdown menu with chart-specific configuration options
 */

import { Settings, RotateCcw } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
} from './dropdown-menu';
import { Button } from './button';
import { Slider } from './slider';
import { Label } from './label';
import {
  useChartSettings,
  ChartType,
  ChartSettings,
  HistogramSettings,
  ScatterSettings,
  LineChartSettings,
  BarChartSettings,
  HeatmapSettings,
  BoxPlotSettings,
} from '@/lib/stores/chart-settings';

interface ChartSettingsMenuProps {
  chartType: ChartType;
  onSettingsChange?: (settings: ChartSettings) => void;
}

export function ChartSettingsMenu({ chartType, onSettingsChange }: ChartSettingsMenuProps) {
  const { getSettings, updateSettings, resetSettings } = useChartSettings();
  const settings = getSettings(chartType);

  const handleUpdate = (newSettings: Partial<ChartSettings>) => {
    updateSettings(chartType, newSettings);
    if (onSettingsChange) {
      onSettingsChange({ ...settings, ...newSettings });
    }
  };

  const handleReset = () => {
    resetSettings(chartType);
    if (onSettingsChange) {
      onSettingsChange(getSettings(chartType));
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          aria-label="Chart settings"
        >
          <Settings className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64 max-h-[600px] overflow-y-auto">
        <DropdownMenuLabel>Chart Settings</DropdownMenuLabel>
        <DropdownMenuSeparator />

        {/* Common Settings */}
        <div className="px-2 py-2 space-y-3">
          {/* Color Scheme */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Color Scheme</Label>
            <DropdownMenuRadioGroup
              value={settings.colorScheme || 'nord'}
              onValueChange={(value) => handleUpdate({ colorScheme: value as any })}
            >
              <DropdownMenuRadioItem value="nord" className="text-xs">
                Nord (Default)
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="warm" className="text-xs">
                Warm
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="cool" className="text-xs">
                Cool
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="monochrome" className="text-xs">
                Monochrome
              </DropdownMenuRadioItem>
            </DropdownMenuRadioGroup>
          </div>

          {/* Font Size */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Font Size</Label>
            <DropdownMenuRadioGroup
              value={settings.fontSize || 'sm'}
              onValueChange={(value) => handleUpdate({ fontSize: value as any })}
            >
              <DropdownMenuRadioItem value="xs" className="text-xs">
                Extra Small
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="sm" className="text-xs">
                Small
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="base" className="text-xs">
                Base
              </DropdownMenuRadioItem>
              <DropdownMenuRadioItem value="lg" className="text-xs">
                Large
              </DropdownMenuRadioItem>
            </DropdownMenuRadioGroup>
          </div>
        </div>

        <DropdownMenuSeparator />

        {/* Common Toggles */}
        <DropdownMenuCheckboxItem
          checked={settings.showLegend ?? true}
          onCheckedChange={(checked) => handleUpdate({ showLegend: checked })}
        >
          Show Legend
        </DropdownMenuCheckboxItem>
        <DropdownMenuCheckboxItem
          checked={settings.showGrid ?? true}
          onCheckedChange={(checked) => handleUpdate({ showGrid: checked })}
        >
          Show Grid
        </DropdownMenuCheckboxItem>

        <DropdownMenuSeparator />

        {/* Chart-Specific Settings */}
        {renderChartSpecificSettings(chartType, settings, handleUpdate)}

        <DropdownMenuSeparator />

        {/* Reset Button */}
        <DropdownMenuItem onClick={handleReset} className="text-destructive">
          <RotateCcw className="h-3.5 w-3.5 mr-2" />
          Reset to Defaults
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/**
 * Render chart-specific settings based on chart type
 */
function renderChartSpecificSettings(
  chartType: ChartType,
  settings: ChartSettings,
  handleUpdate: (newSettings: Partial<ChartSettings>) => void
) {
  switch (chartType) {
    case 'histogram':
      return renderHistogramSettings(settings as HistogramSettings, handleUpdate);
    case 'scatter':
      return renderScatterSettings(settings as ScatterSettings, handleUpdate);
    case 'line':
      return renderLineChartSettings(settings as LineChartSettings, handleUpdate);
    case 'bar':
      return renderBarChartSettings(settings as BarChartSettings, handleUpdate);
    case 'heatmap':
      return renderHeatmapSettings(settings as HeatmapSettings, handleUpdate);
    case 'boxplot':
      return renderBoxPlotSettings(settings as BoxPlotSettings, handleUpdate);
    default:
      return null;
  }
}

function renderHistogramSettings(
  settings: HistogramSettings,
  handleUpdate: (newSettings: Partial<ChartSettings>) => void
) {
  return (
    <>
      <div className="px-2 py-2 space-y-2">
        <Label className="text-xs font-medium">Bin Count: {settings.binCount || 20}</Label>
        <Slider
          value={[settings.binCount || 20]}
          onValueChange={([value]) => handleUpdate({ binCount: value })}
          min={5}
          max={100}
          step={5}
          className="w-full"
        />
      </div>
      <DropdownMenuCheckboxItem
        checked={settings.showDensity ?? false}
        onCheckedChange={(checked) => handleUpdate({ showDensity: checked })}
      >
        Show Density Curve
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={settings.showMean ?? true}
        onCheckedChange={(checked) => handleUpdate({ showMean: checked })}
      >
        Show Mean Line
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={settings.showMedian ?? false}
        onCheckedChange={(checked) => handleUpdate({ showMedian: checked })}
      >
        Show Median Line
      </DropdownMenuCheckboxItem>
    </>
  );
}

function renderScatterSettings(
  settings: ScatterSettings,
  handleUpdate: (newSettings: Partial<ChartSettings>) => void
) {
  return (
    <>
      <div className="px-2 py-2 space-y-2">
        <Label className="text-xs font-medium">Point Size: {settings.pointSize || 4}</Label>
        <Slider
          value={[settings.pointSize || 4]}
          onValueChange={([value]) => handleUpdate({ pointSize: value })}
          min={2}
          max={12}
          step={1}
          className="w-full"
        />
      </div>
      <div className="px-2 py-2 space-y-2">
        <Label className="text-xs font-medium">
          Opacity: {((settings.opacity || 0.7) * 100).toFixed(0)}%
        </Label>
        <Slider
          value={[(settings.opacity || 0.7) * 100]}
          onValueChange={([value]) => handleUpdate({ opacity: value / 100 })}
          min={10}
          max={100}
          step={10}
          className="w-full"
        />
      </div>
      <DropdownMenuCheckboxItem
        checked={settings.showTrendLine ?? false}
        onCheckedChange={(checked) => handleUpdate({ showTrendLine: checked })}
      >
        Show Trend Line
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={settings.showCorrelation ?? true}
        onCheckedChange={(checked) => handleUpdate({ showCorrelation: checked })}
      >
        Show Correlation
      </DropdownMenuCheckboxItem>
    </>
  );
}

function renderLineChartSettings(
  settings: LineChartSettings,
  handleUpdate: (newSettings: Partial<ChartSettings>) => void
) {
  return (
    <>
      <div className="px-2 py-2 space-y-2">
        <Label className="text-xs font-medium">Line Width: {settings.lineWidth || 2}</Label>
        <Slider
          value={[settings.lineWidth || 2]}
          onValueChange={([value]) => handleUpdate({ lineWidth: value })}
          min={1}
          max={5}
          step={0.5}
          className="w-full"
        />
      </div>
      <DropdownMenuCheckboxItem
        checked={settings.showPoints ?? false}
        onCheckedChange={(checked) => handleUpdate({ showPoints: checked })}
      >
        Show Data Points
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={settings.smoothing ?? false}
        onCheckedChange={(checked) => handleUpdate({ smoothing: checked })}
      >
        Smooth Lines
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={settings.fillArea ?? false}
        onCheckedChange={(checked) => handleUpdate({ fillArea: checked })}
      >
        Fill Area
      </DropdownMenuCheckboxItem>
    </>
  );
}

function renderBarChartSettings(
  settings: BarChartSettings,
  handleUpdate: (newSettings: Partial<ChartSettings>) => void
) {
  return (
    <>
      <div className="px-2 py-2 space-y-1.5">
        <Label className="text-xs font-medium">Orientation</Label>
        <DropdownMenuRadioGroup
          value={settings.orientation || 'vertical'}
          onValueChange={(value) => handleUpdate({ orientation: value as any })}
        >
          <DropdownMenuRadioItem value="vertical" className="text-xs">
            Vertical
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="horizontal" className="text-xs">
            Horizontal
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </div>
      <div className="px-2 py-2 space-y-2">
        <Label className="text-xs font-medium">Bar Width: {settings.barWidth || 20}</Label>
        <Slider
          value={[settings.barWidth || 20]}
          onValueChange={([value]) => handleUpdate({ barWidth: value })}
          min={10}
          max={50}
          step={5}
          className="w-full"
        />
      </div>
      <DropdownMenuCheckboxItem
        checked={settings.stacked ?? false}
        onCheckedChange={(checked) => handleUpdate({ stacked: checked })}
      >
        Stacked Bars
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={settings.showValues ?? false}
        onCheckedChange={(checked) => handleUpdate({ showValues: checked })}
      >
        Show Values
      </DropdownMenuCheckboxItem>
    </>
  );
}

function renderHeatmapSettings(
  settings: HeatmapSettings,
  handleUpdate: (newSettings: Partial<ChartSettings>) => void
) {
  return (
    <>
      <div className="px-2 py-2 space-y-1.5">
        <Label className="text-xs font-medium">Interpolation</Label>
        <DropdownMenuRadioGroup
          value={settings.interpolation || 'linear'}
          onValueChange={(value) => handleUpdate({ interpolation: value as any })}
        >
          <DropdownMenuRadioItem value="linear" className="text-xs">
            Linear
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="step" className="text-xs">
            Step
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="basis" className="text-xs">
            Basis
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </div>
      <DropdownMenuCheckboxItem
        checked={settings.showValues ?? false}
        onCheckedChange={(checked) => handleUpdate({ showValues: checked })}
      >
        Show Cell Values
      </DropdownMenuCheckboxItem>
    </>
  );
}

function renderBoxPlotSettings(
  settings: BoxPlotSettings,
  handleUpdate: (newSettings: Partial<ChartSettings>) => void
) {
  return (
    <>
      <div className="px-2 py-2 space-y-1.5">
        <Label className="text-xs font-medium">Orientation</Label>
        <DropdownMenuRadioGroup
          value={settings.orientation || 'vertical'}
          onValueChange={(value) => handleUpdate({ orientation: value as any })}
        >
          <DropdownMenuRadioItem value="vertical" className="text-xs">
            Vertical
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="horizontal" className="text-xs">
            Horizontal
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </div>
      <DropdownMenuCheckboxItem
        checked={settings.showOutliers ?? true}
        onCheckedChange={(checked) => handleUpdate({ showOutliers: checked })}
      >
        Show Outliers
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={settings.showMean ?? true}
        onCheckedChange={(checked) => handleUpdate({ showMean: checked })}
      >
        Show Mean
      </DropdownMenuCheckboxItem>
      <DropdownMenuCheckboxItem
        checked={settings.notched ?? false}
        onCheckedChange={(checked) => handleUpdate({ notched: checked })}
      >
        Notched Boxes
      </DropdownMenuCheckboxItem>
    </>
  );
}
