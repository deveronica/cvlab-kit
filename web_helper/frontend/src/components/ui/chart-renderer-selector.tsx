import React from "react";
/**
 * Chart Renderer Selector
 *
 * UI component for selecting chart rendering library.
 * Supports Recharts, Victory, Nivo, ECharts, etc.
 */

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './select';
import { BarChart3 } from 'lucide-react';
import type { ChartRenderer, ChartType } from '@/lib/charts/types';
import { getAvailableRenderers } from '@/lib/charts/UniversalChart';
import { isCompatible } from '@/lib/charts/types';

interface ChartRendererSelectorProps {
  /** Currently selected renderer */
  value: ChartRenderer;
  /** Change handler */
  onChange: (renderer: ChartRenderer) => void;
  /** Available renderers (filters options based on chart type support) */
  availableRenderers?: ChartRenderer[];
  /** Current chart type (for compatibility checking) */
  currentChartType?: ChartType;
  /** Disabled state */
  disabled?: boolean;
  /** Show icon */
  showIcon?: boolean;
  /** Size variant */
  size?: 'sm' | 'default' | 'lg';
  /** className for container */
  className?: string;
}

/**
 * Renderer display names and descriptions
 * Top 3 most popular React chart libraries (2025)
 */
const RENDERER_INFO: Record<ChartRenderer, { name: string; description: string }> = {
  recharts: {
    name: 'Recharts',
    description: 'React + D3 - Most popular',
  },
  chartjs: {
    name: 'Chart.js',
    description: 'Stable & performant',
  },
  plotly: {
    name: 'Plotly.js',
    description: 'Scientific & 3D charts',
  },
  echarts: {
    name: 'ECharts',
    description: 'Apache ECharts - Advanced',
  },
};

export function ChartRendererSelector({
  value,
  onChange,
  availableRenderers: propAvailableRenderers,
  currentChartType,
  disabled = false,
  showIcon = true,
  size = 'default',
  className,
}: ChartRendererSelectorProps) {
  // Use provided availableRenderers or fall back to all available renderers
  const availableRenderers = propAvailableRenderers || getAvailableRenderers();
  const selectedInfo = RENDERER_INFO[value];

  return (
    <div className={`flex items-center gap-2 ${className || ''}`}>
      {showIcon && <BarChart3 className="h-4 w-4 text-muted-foreground" />}
      <Select value={value} onValueChange={(v) => onChange(v as ChartRenderer)} disabled={disabled}>
        <SelectTrigger className={`w-[180px] ${size === 'sm' ? 'h-8' : ''}`}>
          <SelectValue placeholder="Select renderer">
            <div className="flex items-center gap-2 truncate">
              <span className="font-medium truncate">{selectedInfo?.name || 'Select'}</span>
            </div>
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {Object.entries(RENDERER_INFO).map(([key, info]) => {
            const renderer = key as ChartRenderer;
            const isAvailable = availableRenderers.includes(renderer);
            // Check if this renderer is compatible with the current chart type
            const isIncompatible = currentChartType ? !isCompatible(renderer, currentChartType) : false;
            const isDisabled = !isAvailable || isIncompatible;

            return (
              <SelectItem key={key} value={key} disabled={isDisabled}>
                <div className="flex items-center justify-between w-full">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">{info.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {info.description}
                    </span>
                  </div>
                </div>
              </SelectItem>
            );
          })}
        </SelectContent>
      </Select>
    </div>
  );
}

/**
 * Compact version for toolbars
 */
export function ChartRendererSelectorCompact({
  value,
  onChange,
  disabled = false,
}: Omit<ChartRendererSelectorProps, 'showIcon' | 'size' | 'className'>) {
  return (
    <ChartRendererSelector
      value={value}
      onChange={onChange}
      disabled={disabled}
      showIcon={false}
      size="sm"
    />
  );
}
