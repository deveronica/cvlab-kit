import React from "react";
/**
 * Data Series Selector
 *
 * UI component for toggling visibility of data series in charts
 */

import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Eye, EyeOff, ListFilter } from 'lucide-react';
import type { SeriesConfig } from '@/lib/charts/types';

interface DataSeriesSelectorProps {
  /** Array of series configurations */
  series: SeriesConfig[];
  /** Callback when series visibility changes */
  onSeriesChange: (updatedSeries: SeriesConfig[]) => void;
  /** className for container */
  className?: string;
  /** Compact mode (popover button) */
  compact?: boolean;
}

export function DataSeriesSelector({
  series = [],
  onSeriesChange,
  className,
  compact = false,
}: DataSeriesSelectorProps) {
  const toggleSeries = (index: number) => {
    const updated = [...series];
    updated[index] = {
      ...updated[index],
      visible: updated[index].visible !== false ? false : true,
    };
    onSeriesChange(updated);
  };

  const toggleAll = (visible: boolean) => {
    const updated = series.map(s => ({ ...s, visible }));
    onSeriesChange(updated);
  };

  const visibleCount = series.filter(s => s.visible !== false).length;
  const allVisible = visibleCount === series.length;
  const noneVisible = visibleCount === 0;

  if (compact) {
    return (
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 relative"
            title={`Data series (${visibleCount}/${series.length} visible)`}
          >
            <ListFilter className="h-4 w-4" />
            {visibleCount < series.length && (
              <span className="absolute -top-1 -right-1 h-3.5 w-3.5 rounded-full bg-primary text-[9px] text-primary-foreground flex items-center justify-center">
                {visibleCount}
              </span>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-64" align="end">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">Data Series</Label>
              <button
                onClick={() => toggleAll(!allVisible)}
                className="text-xs text-primary hover:underline"
                disabled={series.length === 0}
              >
                {allVisible ? 'Hide All' : 'Show All'}
              </button>
            </div>

            <div className="space-y-1 max-h-[300px] overflow-y-auto">
              {series.map((s, idx) => {
                const isVisible = s.visible !== false;
                const id = `series-compact-${s.dataKey}-${idx}`;

                return (
                  <div
                    key={s.dataKey}
                    className="flex items-center space-x-2 p-2 rounded hover:bg-accent/50 transition-colors duration-200"
                  >
                    <Checkbox
                      id={id}
                      checked={isVisible}
                      onCheckedChange={() => toggleSeries(idx)}
                    />
                    <Label
                      htmlFor={id}
                      className="flex-1 flex items-center gap-2 cursor-pointer text-sm"
                    >
                      {s.color && (
                        <div
                          className="w-3 h-3 rounded-sm border border-border"
                          style={{ backgroundColor: s.color }}
                        />
                      )}
                      <span>{s.name || s.dataKey}</span>
                    </Label>
                    {isVisible ? (
                      <Eye className="h-3.5 w-3.5 text-muted-foreground" />
                    ) : (
                      <EyeOff className="h-3.5 w-3.5 text-muted-foreground" />
                    )}
                  </div>
                );
              })}
            </div>

            {noneVisible && (
              <div className="p-2 text-xs text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-300 dark:border-yellow-800 rounded">
                ⚠️ Select at least one series
              </div>
            )}
          </div>
        </PopoverContent>
      </Popover>
    );
  }

  return (
    <div className={`space-y-3 ${className || ''}`}>
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">Data Series</Label>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            {visibleCount} / {series.length} shown
          </Badge>
          <button
            onClick={() => toggleAll(!allVisible)}
            className="text-xs text-primary hover:underline"
            disabled={series.length === 0}
          >
            {allVisible ? 'Hide All' : 'Show All'}
          </button>
        </div>
      </div>

      <div className="space-y-2">
        {series.map((s, idx) => {
          const isVisible = s.visible !== false;
          const id = `series-${s.dataKey}-${idx}`;

          return (
            <div
              key={s.dataKey}
              className="flex items-center space-x-2 p-2 rounded hover:bg-accent/50 transition-colors duration-200"
            >
              <Checkbox
                id={id}
                checked={isVisible}
                onCheckedChange={() => toggleSeries(idx)}
              />
              <Label
                htmlFor={id}
                className="flex-1 flex items-center gap-2 cursor-pointer"
              >
                {s.color && (
                  <div
                    className="w-3 h-3 rounded-sm border border-border"
                    style={{ backgroundColor: s.color }}
                  />
                )}
                <span className="font-medium text-sm">{s.name || s.dataKey}</span>
                {s.type && (
                  <Badge variant="outline" className="text-[9px] h-4 px-1">
                    {s.type}
                  </Badge>
                )}
              </Label>
              {isVisible ? (
                <Eye className="h-4 w-4 text-muted-foreground" />
              ) : (
                <EyeOff className="h-4 w-4 text-muted-foreground" />
              )}
            </div>
          );
        })}
      </div>

      {noneVisible && (
        <div className="p-3 text-sm text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-300 dark:border-yellow-800 rounded">
          ⚠️ No series are visible. Select at least one to display the chart.
        </div>
      )}
    </div>
  );
}
