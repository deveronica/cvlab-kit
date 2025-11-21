import React from "react";

/**
 * Chart Type Selector
 *
 * UI component for selecting chart type (line, bar, area, scatter, pie)
 */

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { BarChart3, LineChart, PieChart, ScatterChart, AreaChart } from 'lucide-react';
import type { ChartType, ChartRenderer } from '@/lib/charts/types';
import { isCompatible } from '@/lib/charts/types';

interface ChartTypeSelectorProps {
  /** Currently selected chart type */
  value: ChartType;
  /** Change handler */
  onChange: (type: ChartType) => void;
  /** Supported chart types (filters available options) */
  supportedTypes?: ChartType[];
  /** Current renderer (for compatibility checking) */
  currentRenderer?: ChartRenderer;
  /** Disabled state */
  disabled?: boolean;
  /** Size variant */
  size?: 'sm' | 'default';
  /** className for container */
  className?: string;
}

/**
 * Chart type information
 */
const CHART_TYPE_INFO: Record<ChartType, { name: string; icon: React.ElementType; description: string }> = {
  line: {
    name: 'Line',
    icon: LineChart,
    description: 'Show trends over time',
  },
  bar: {
    name: 'Bar',
    icon: BarChart3,
    description: 'Compare values across categories',
  },
  area: {
    name: 'Area',
    icon: AreaChart,
    description: 'Show cumulative totals',
  },
  scatter: {
    name: 'Scatter',
    icon: ScatterChart,
    description: 'Show correlations',
  },
  pie: {
    name: 'Pie',
    icon: PieChart,
    description: 'Show proportions',
  },
  radar: {
    name: 'Radar',
    icon: BarChart3,
    description: 'Multi-variate comparison',
  },
  heatmap: {
    name: 'Heatmap',
    icon: BarChart3,
    description: 'Show data density',
  },
};

// Default supported chart types (most common)
const DEFAULT_SUPPORTED_TYPES: ChartType[] = ['line', 'bar', 'area', 'scatter', 'pie'];

export function ChartTypeSelector({
  value,
  onChange,
  supportedTypes = DEFAULT_SUPPORTED_TYPES,
  currentRenderer,
  disabled = false,
  size = 'default',
  className,
}: ChartTypeSelectorProps) {
  const selectedInfo = CHART_TYPE_INFO[value];
  const Icon = selectedInfo?.icon || BarChart3;

  return (
    <div className={`flex items-center gap-1.5 h-8 ${className || ''}`}>
      <Icon className="h-4 w-4 text-muted-foreground flex-shrink-0" />
      <Select value={value} onValueChange={(v) => onChange(v as ChartType)} disabled={disabled}>
        <SelectTrigger className={`${size === 'sm' ? 'h-8 w-[110px] text-xs py-1' : 'h-8 w-[140px] py-1'}`}>
          <SelectValue placeholder="Select type">
            <span className="font-medium">{selectedInfo?.name || 'Select'}</span>
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {supportedTypes.map((type) => {
            const info = CHART_TYPE_INFO[type];
            const TypeIcon = info.icon;
            // Check if this chart type is compatible with the current renderer
            const isDisabled = currentRenderer ? !isCompatible(currentRenderer, type) : false;

            return (
              <SelectItem key={type} value={type} disabled={isDisabled}>
                <div className="flex items-center gap-2 w-full">
                  <TypeIcon className="h-4 w-4" />
                  <div className="flex flex-col items-start">
                    <span className="font-medium">{info.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {info.description}
                      {isDisabled && ' (Not supported by current renderer)'}
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
export function ChartTypeSelectorCompact(props: Omit<ChartTypeSelectorProps, 'size'>) {
  return <ChartTypeSelector {...props} size="sm" />;
}
