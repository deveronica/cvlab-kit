import React from "react";

import { useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Expand } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { ExportMenu } from '../ui/export-menu';
import type { Run } from '../../lib/types';
import { useChartExport } from '../../hooks/useChartExport';

interface MetricAreaChartProps {
  runs: Run[];
  metricKey: string;
  title?: string;
  height?: number;
  showLegend?: boolean;
  stacked?: boolean;
}

interface ChartDataPoint {
  step: number;
  [runId: string]: number | string;
}

const CHART_COLORS = [
  '#8b5cf6', // violet-500
  '#06b6d4', // cyan-500
  '#10b981', // emerald-500
  '#f59e0b', // amber-500
  '#ef4444', // red-500
  '#f97316', // orange-500
  '#84cc16', // lime-500
  '#ec4899', // pink-500
  '#6366f1', // indigo-500
  '#14b8a6', // teal-500
];

export function MetricAreaChart({
  runs,
  metricKey,
  title,
  height = 400,
  showLegend = true,
  stacked = false
}: MetricAreaChartProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Process data to create chart format
  const chartData = React.useMemo(() => {
    if (!runs.length) return [];

    // Get all unique steps across all runs
    const allSteps = new Set<number>();
    runs.forEach(run => {
      if (run.metrics?.timeseries) {
        run.metrics.timeseries.forEach(point => {
          allSteps.add(point.step);
        });
      }
    });

    const sortedSteps = Array.from(allSteps).sort((a, b) => a - b);

    // Create data points for each step
    return sortedSteps.map(step => {
      const dataPoint: ChartDataPoint = { step };

      runs.forEach(run => {
        const timeseriesPoint = run.metrics?.timeseries?.find(t => t.step === step);
        const value = timeseriesPoint?.values[metricKey];

        if (value !== undefined) {
          dataPoint[run.run_name] = value;
        }
      });

      return dataPoint;
    });
  }, [runs, metricKey]);

  // Use unified export hook
  const { chartContainerRef, exportToPNG, exportToSVG, exportToCSV } = useChartExport({
    filename: `${metricKey}_area_chart`,
    data: chartData,
  });

  if (!chartData.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title || `${metricKey} progress`}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            No data available for {metricKey}
          </div>
        </CardContent>
      </Card>
    );
  }

  const renderChart = () => (
    <AreaChart
      data={chartData}
      margin={{
        top: 10,
        right: 30,
        left: 20,
        bottom: 5,
      }}
    >
      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
      <XAxis
        dataKey="step"
        tick={{ fontSize: 12 }}
        className="text-muted-foreground"
      />
      <YAxis
        tick={{ fontSize: 12 }}
        className="text-muted-foreground"
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
      {showLegend && (
        <Legend
          wrapperStyle={{ paddingTop: '20px' }}
        />
      )}
      {runs.map((run, index) => (
        <Area
          key={run.run_name}
          type="monotone"
          dataKey={run.run_name}
          stackId={stacked ? "1" : undefined}
          stroke={CHART_COLORS[index % CHART_COLORS.length]}
          fill={CHART_COLORS[index % CHART_COLORS.length]}
          fillOpacity={stacked ? 0.8 : 0.3}
          strokeWidth={2}
          name={`${run.run_name.substring(0, 8)}...`}
          connectNulls={false}
        />
      ))}
    </AreaChart>
  );

  return (
    <>
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <CardTitle>{title || `${metricKey} progress`}</CardTitle>

          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => setIsFullscreen(true)} className="h-7">
              <Expand className="h-4 w-4" />
            </Button>

            <ExportMenu
              onExport={(format) => {
                if (format === 'png') exportToPNG();
                else if (format === 'svg') exportToSVG();
                else if (format === 'csv') exportToCSV();
              }}
              formats={['png', 'svg', 'csv']}
              showLabel={false}
              size="sm"
            />
          </div>
        </div>
      </CardHeader>
      <CardContent ref={chartContainerRef}>
        <ResponsiveContainer width="100%" height={height}>
          {renderChart()}
        </ResponsiveContainer>
      </CardContent>
    </Card>

      <Dialog open={isFullscreen} onOpenChange={setIsFullscreen}>
        <DialogContent className="max-w-[95vw] max-h-[95vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle>{title || `${metricKey} progress`}</DialogTitle>
          </DialogHeader>
          <div className="flex-1 min-h-[70vh]">
            <ResponsiveContainer width="100%" height="100%">
              {renderChart()}
            </ResponsiveContainer>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}