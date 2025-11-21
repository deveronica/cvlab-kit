import React from "react";
import { useState, useMemo } from 'react';
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
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';
import { Button } from './button';
import { Checkbox } from './checkbox';
import { Label } from './label';
import { Download, RotateCcw } from 'lucide-react';
import { devLog } from '@/lib/dev-utils';

interface TrainingCurveChartProps {
  data: Array<Record<string, any>>;
  metrics: string[];
  title?: string;
  description?: string;
  xAxisKey?: string;
  className?: string;
  onExport?: (format: 'png' | 'svg') => void;
}

const COLORS = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#10b981', // green
  '#f59e0b', // amber
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
];

export function TrainingCurveChart({
  data,
  metrics,
  title = 'Training Curves',
  description,
  xAxisKey = 'step',
  className,
  onExport,
}: TrainingCurveChartProps) {
  const [visibleMetrics, setVisibleMetrics] = useState<Set<string>>(
    new Set(metrics.slice(0, 4)) // Show first 4 metrics by default
  );
  const [brushDomain, setBrushDomain] = useState<[number, number] | null>(null);

  const toggleMetric = (metric: string) => {
    setVisibleMetrics((prev) => {
      const next = new Set(prev);
      if (next.has(metric)) {
        next.delete(metric);
      } else {
        next.add(metric);
      }
      return next;
    });
  };

  const resetZoom = () => {
    setBrushDomain(null);
  };

  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];

    // Ensure all data points have numeric values
    return data.map(point => {
      const cleanPoint: Record<string, any> = { ...point };
      metrics.forEach(metric => {
        if (cleanPoint[metric] !== undefined) {
          const value = parseFloat(cleanPoint[metric]);
          cleanPoint[metric] = isNaN(value) ? null : value;
        }
      });
      return cleanPoint;
    });
  }, [data, metrics]);

  const handleExport = (format: 'png' | 'svg') => {
    if (onExport) {
      onExport(format);
    } else {
      // Default export behavior - could be enhanced with html2canvas
      devLog(`Export to ${format} requested`);
    }
  };

  if (!filteredData || filteredData.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-muted-foreground">
            No training data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={resetZoom} disabled={!brushDomain}>
              <RotateCcw className="h-3 w-3 mr-1" />
              Reset Zoom
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleExport('png')}
            >
              <Download className="h-3 w-3 mr-1" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Metric Selection */}
          <div className="flex flex-wrap gap-4 pb-2 border-b">
            {metrics.map((metric, _idx) => (
              <div key={metric} className="flex items-center space-x-2">
                <Checkbox
                  id={`metric-${metric}`}
                  checked={visibleMetrics.has(metric)}
                  onCheckedChange={() => toggleMetric(metric)}
                />
                <Label
                  htmlFor={`metric-${metric}`}
                  className="text-sm font-medium cursor-pointer flex items-center gap-2"
                >
                  <div
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: COLORS[_idx % COLORS.length] }}
                  />
                  {metric}
                </Label>
              </div>
            ))}
          </div>

          {/* Chart */}
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={filteredData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey={xAxisKey}
                label={{ value: xAxisKey, position: 'insideBottom', offset: -5 }}
                domain={brushDomain || ['auto', 'auto']}
              />
              <YAxis label={{ value: 'Value', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--background))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '0.5rem',
                }}
              />
              <Legend wrapperStyle={{ paddingTop: '1rem' }} />

              {Array.from(visibleMetrics).map((metric, _idx) => (
                <Line
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  stroke={COLORS[metrics.indexOf(metric) % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                  connectNulls
                />
              ))}

              <Brush
                dataKey={xAxisKey}
                height={30}
                stroke="hsl(var(--primary))"
                onChange={(domain: any) => {
                  if (domain && domain.startIndex !== undefined && domain.endIndex !== undefined) {
                    const start = filteredData[domain.startIndex][xAxisKey];
                    const end = filteredData[domain.endIndex][xAxisKey];
                    setBrushDomain([start, end]);
                  }
                }}
              />
            </LineChart>
          </ResponsiveContainer>

          {/* Stats Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2 border-t">
            {Array.from(visibleMetrics).slice(0, 4).map((metric) => {
              const values = filteredData
                .map((d) => d[metric])
                .filter((v) => v !== null && !isNaN(v));

              if (values.length === 0) return null;

              const latest = values[values.length - 1];
              const best = Math.max(...values);
              const worst = Math.min(...values);

              return (
                <div key={metric} className="space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">{metric}</div>
                  <div className="text-sm">
                    <span className="font-semibold">Latest:</span> {latest.toFixed(4)}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Best: {best.toFixed(4)} | Worst: {worst.toFixed(4)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
