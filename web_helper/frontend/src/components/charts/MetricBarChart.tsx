import React from "react";

import { useState } from 'react';
import {
  BarChart,
  Bar,
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

interface MetricBarChartProps {
  runs: Run[];
  metricKeys: string[];
  title?: string;
  height?: number;
  showLegend?: boolean;
  _orientation?: 'horizontal' | 'vertical';
}

interface ChartDataPoint {
  runId: string;
  runName: string;
  [metricKey: string]: number | string;
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

export function MetricBarChart({
  runs,
  metricKeys,
  title,
  height = 400,
  showLegend = true,
  _orientation = 'vertical'
}: MetricBarChartProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Process data for bar chart
  const chartData = React.useMemo(() => {
    return runs.map(run => {
      const dataPoint: ChartDataPoint = {
        runId: run.run_name,
        runName: run.run_name.substring(0, 12) + '...',
      };

      metricKeys.forEach(metricKey => {
        // Get final value for the metric
        const value = run.metrics?.final?.[metricKey] ||
                     run.metrics?.max?.[metricKey] ||
                     run.metrics?.mean?.[metricKey];

        if (value !== undefined) {
          dataPoint[metricKey] = value;
        }
      });

      return dataPoint;
    });
  }, [runs, metricKeys]);

  // Use unified export hook
  const { chartContainerRef, exportToPNG, exportToSVG, exportToCSV } = useChartExport({
    filename: 'metric_comparison',
    data: chartData,
  });

  if (!chartData.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title || 'Metric Comparison'}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            No data available
          </div>
        </CardContent>
      </Card>
    );
  }

  const renderChart = () => (
    <BarChart
      data={chartData}
      margin={{
        top: 20,
        right: 30,
        left: 20,
        bottom: 5,
      }}
    >
      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
      <XAxis
        dataKey="runName"
        tick={{ fontSize: 12 }}
        className="text-muted-foreground"
        angle={-45}
        textAnchor="end"
        height={80}
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
        formatter={(value: any, name: string) => [
          typeof value === 'number' ? value.toFixed(4) : value,
          name
        ]}
      />
      {showLegend && (
        <Legend
          wrapperStyle={{ paddingTop: '20px' }}
        />
      )}
      {metricKeys.map((metricKey, index) => (
        <Bar
          key={metricKey}
          dataKey={metricKey}
          fill={CHART_COLORS[index % CHART_COLORS.length]}
          name={metricKey}
          radius={[2, 2, 0, 0]}
        />
      ))}
    </BarChart>
  );

  return (
    <>
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <CardTitle>{title || 'Metric Comparison'}</CardTitle>

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
            <DialogTitle>{title || 'Metric Comparison'}</DialogTitle>
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