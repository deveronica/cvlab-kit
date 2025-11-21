import React from "react";
/**
 * Learning Rate Schedule Chart
 * Visualizes how learning rate changes over training epochs/steps
 */

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { ChartCard } from '../ui/chart-card';
import type { Run } from '../../lib/types';

interface LearningRateScheduleChartProps {
  runs: Run[];
  height?: number;
  variant?: 'default' | 'compact';
}

interface ChartDataPoint {
  step: number;
  [runId: string]: number | string;
}

const COLORS = [
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#f59e0b', // amber
  '#ef4444', // red
  '#10b981', // green
  '#ec4899', // pink
];

export function LearningRateScheduleChart({
  runs,
  height = 300,
  variant = 'default',
}: LearningRateScheduleChartProps) {
  const { chartData, hasData, runColors } = useMemo(() => {
    const dataMap = new Map<number, Record<string, number | string>>();
    const colors: Record<string, string> = {};
    let hasLRData = false;

    runs.forEach((run, idx) => {
      const runId = run.run_name || `Run ${idx + 1}`;
      colors[runId] = COLORS[idx % COLORS.length];

      // Try to find learning rate in metrics timeseries
      const timeseries = run.metrics?.timeseries || [];

      if (timeseries.length > 0) {
        // Check if any timeseries point has learning rate
        const firstPoint = timeseries[0];
        const lrKeys = Object.keys(firstPoint.values || {}).filter(key =>
          key.toLowerCase().includes('lr') ||
          key.toLowerCase().includes('learning_rate')
        );

        if (lrKeys.length > 0) {
          hasLRData = true;
          const lrKey = lrKeys[0];

          timeseries.forEach(point => {
            const value = point.values[lrKey];
            if (typeof value === 'number' && !isNaN(value)) {
              if (!dataMap.has(point.step)) {
                dataMap.set(point.step, { step: point.step });
              }
              dataMap.get(point.step)![runId] = value;
            }
          });
        }
      }
    });

    const data = Array.from(dataMap.values()).sort((a, b) => (a.step as number) - (b.step as number));

    return {
      chartData: data,
      hasData: hasLRData && data.length > 0,
      runColors: colors,
    };
  }, [runs]);

  return (
    <ChartCard
      title="Learning Rate Schedule"
      description={
        hasData
          ? `Learning rate changes over training steps (${runs.length} run${runs.length > 1 ? 's' : ''})`
          : "No learning rate data available"
      }
      isEmpty={!hasData}
      emptyMessage="Learning rate data not found in metrics history"
      variant={variant}
      enableSettings={false}
      enableFullscreen={true}
      enableExport={false}
      height={height}
    >
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="step"
            label={{ value: 'Training Step', position: 'insideBottom', offset: -5 }}
            className="text-xs"
          />
          <YAxis
            label={{ value: 'Learning Rate', angle: -90, position: 'insideLeft' }}
            className="text-xs"
            scale="log"
            domain={['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
            }}
            formatter={(value: number) => value.toExponential(2)}
          />
          <Legend />
          {runs.map((run, idx) => {
            const runId = run.run_name || `Run ${idx + 1}`;
            return (
              <Line
                key={runId}
                type="monotone"
                dataKey={runId}
                stroke={runColors[runId]}
                strokeWidth={2}
                dot={false}
                name={runId}
                connectNulls
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
