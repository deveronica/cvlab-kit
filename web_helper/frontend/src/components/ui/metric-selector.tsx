import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './select';
import { Button } from './button';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { Card, CardContent } from './card';

interface MetricSelectorProps {
  metrics: string[];
  selectedMetric: string;
  onMetricChange: (metric: string) => void;
  direction: 'maximize' | 'minimize';
  onDirectionChange: (direction: 'maximize' | 'minimize') => void;
  className?: string;
}

export function MetricSelector({
  metrics,
  selectedMetric,
  onMetricChange,
  direction,
  onDirectionChange,
  className,
}: MetricSelectorProps) {
  if (metrics.length === 0) return null;

  return (
    <Card className={className}>
      <CardContent className="pt-6">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-muted-foreground whitespace-nowrap">
              Target Metric:
            </label>
            <Select value={selectedMetric} onValueChange={onMetricChange}>
              <SelectTrigger className="w-[200px]">
                <SelectValue placeholder="Select metric" />
              </SelectTrigger>
              <SelectContent>
                {metrics.map((metric) => (
                  <SelectItem key={metric} value={metric}>
                    {metric}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-muted-foreground whitespace-nowrap">
              Optimize:
            </label>
            <div className="flex gap-1 border rounded-md p-1">
              <Button
                variant={direction === 'maximize' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => onDirectionChange('maximize')}
                className="gap-1"
              >
                <TrendingUp className="h-4 w-4" />
                Maximize
              </Button>
              <Button
                variant={direction === 'minimize' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => onDirectionChange('minimize')}
                className="gap-1"
              >
                <TrendingDown className="h-4 w-4" />
                Minimize
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
