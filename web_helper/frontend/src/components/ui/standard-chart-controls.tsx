import React from "react";

/**
 * Standard Chart Controls
 * Unified UX pattern for all charts across the application
 *
 * Usage Pattern:
 * 1. Chart Type Selector (ToggleGroup) - if multiple types supported
 * 2. Data/Metric Selector (DropdownMenu)
 * 3. Settings (Popover)
 * 4. ChartCard provides: Fullscreen, Export
 */

import { Button } from './button';
import { Settings } from 'lucide-react';
import {
  ToggleGroup,
  ToggleGroupItem,
} from './toggle-group';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from './dropdown-menu';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from './popover';
import { Switch } from './switch';
import { Label } from './label';
import { ScrollArea } from './scroll-area';

// Chart Type Selector
interface ChartTypeSelectorProps {
  value: string;
  onValueChange: (value: string) => void;
  types: Array<{
    value: string;
    label: string;
    icon: React.ReactNode;
  }>;
}

export function ChartTypeSelector({ value, onValueChange, types }: ChartTypeSelectorProps) {
  return (
    <ToggleGroup
      type="single"
      value={value}
      onValueChange={(v) => v && onValueChange(v)}
      className="h-8"
    >
      {types.map((type) => (
        <ToggleGroupItem
          key={type.value}
          value={type.value}
          aria-label={type.label}
          className="h-8 px-2"
          title={type.label}
        >
          {type.icon}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}

// Metric Selector
interface MetricSelectorProps {
  selectedMetrics: Set<string>;
  availableMetrics: string[];
  onToggle: (metric: string) => void;
  onSelectAll: () => void;
  onClearAll: () => void;
  label?: string;
  icon?: React.ReactNode;
}

export function MetricSelector({
  selectedMetrics,
  availableMetrics,
  onToggle,
  onSelectAll,
  onClearAll,
  label = 'Metrics',
  icon,
}: MetricSelectorProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="h-8">
          {icon && <span className="mr-1">{icon}</span>}
          {label} ({selectedMetrics.size}/{availableMetrics.length})
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-80" align="end">
        <DropdownMenuLabel>Select {label}</DropdownMenuLabel>
        <div className="flex gap-2 px-2 pb-2">
          <Button variant="outline" size="sm" className="flex-1" onClick={onSelectAll}>
            All
          </Button>
          <Button variant="outline" size="sm" className="flex-1" onClick={onClearAll}>
            Clear
          </Button>
        </div>
        <DropdownMenuSeparator />
        <ScrollArea className="h-64">
          {availableMetrics.map((metric) => (
            <DropdownMenuCheckboxItem
              key={metric}
              checked={selectedMetrics.has(metric)}
              onCheckedChange={() => onToggle(metric)}
            >
              {metric}
            </DropdownMenuCheckboxItem>
          ))}
        </ScrollArea>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

// Settings Popover
interface SettingItem {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}

interface ChartSettingsProps {
  settings: SettingItem[];
}

export function ChartSettings({ settings }: ChartSettingsProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="h-8 w-8 p-0" aria-label="Settings">
          <Settings className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-64" align="end">
        <div className="space-y-3">
          {settings.map((setting, idx) => (
            <div key={idx} className="flex items-center justify-between">
              <Label className="text-sm">{setting.label}</Label>
              <Switch checked={setting.value} onCheckedChange={setting.onChange} />
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}

// Combined Standard Controls
interface StandardChartControlsProps {
  // Chart Type
  chartType?: {
    value: string;
    onValueChange: (value: string) => void;
    types: Array<{
      value: string;
      label: string;
      icon: React.ReactNode;
    }>;
  };

  // Metric Selector
  metricSelector?: {
    selectedMetrics: Set<string>;
    availableMetrics: string[];
    onToggle: (metric: string) => void;
    onSelectAll: () => void;
    onClearAll: () => void;
    label?: string;
    icon?: React.ReactNode;
  };

  // Settings
  settings?: SettingItem[];

  // Additional custom controls
  customControls?: React.ReactNode;
}

export function StandardChartControls({
  chartType,
  metricSelector,
  settings,
  customControls,
}: StandardChartControlsProps) {
  return (
    <>
      {chartType && <ChartTypeSelector {...chartType} />}
      {metricSelector && <MetricSelector {...metricSelector} />}
      {customControls}
      {settings && settings.length > 0 && <ChartSettings settings={settings} />}
    </>
  );
}
