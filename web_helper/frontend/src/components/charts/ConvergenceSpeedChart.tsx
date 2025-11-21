import React from "react";

/**
 * Convergence Speed Comparison Chart
 * Analyzes and compares how quickly different runs converge to their final performance
 */

import { useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { ChartCard } from '../ui/chart-card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Zap, TrendingUp, Clock } from 'lucide-react';
import type { Run } from '../../lib/types';

interface ConvergenceSpeedChartProps {
  runs: Run[];
  availableMetrics: string[];
  height?: number;
  targetThreshold?: number; // e.g., 90% of final value
  customControls?: React.ReactNode;
}

interface ConvergenceMetrics {
  runName: string;
  stepsToConvergence: number;
  convergenceRate: number; // improvement per step
  finalValue: number;
  earlyValue: number; // value at 10% of training
  improvement: number;
  efficiency: number; // improvement per step
  color: string;
}

const COLORS = [
  '#10b981', // green
  '#3b82f6', // blue
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#06b6d4', // cyan
];

export function ConvergenceSpeedChart({
  runs,
  availableMetrics,
  height = 400,
  targetThreshold = 0.95,
}: ConvergenceSpeedChartProps) {
  const [selectedMetric, setSelectedMetric] = useState<string>(availableMetrics[0] || '');

  const convergenceData = useMemo(() => {
    const metricKey = selectedMetric;
    const data: ConvergenceMetrics[] = [];

    runs.forEach((run, _idx) => {
      const timeseries = run.metrics?.timeseries || [];

      if (timeseries.length === 0) {
        return;
      }

      // Extract values for the specific metric
      const validValues = timeseries
        .map(point => point.values[metricKey])
        .filter((v): v is number => typeof v === 'number' && !isNaN(v));

      if (validValues.length < 10) {
        return;
      }

      const finalValue = validValues[validValues.length - 1];
      const earlyValue = validValues[Math.floor(validValues.length * 0.1)];
      const improvement = Math.abs(finalValue - earlyValue);

      // Determine if we're maximizing or minimizing based on metric name
      const isLoss = metricKey.toLowerCase().includes('loss') ||
                    metricKey.toLowerCase().includes('error');
      const targetValue = isLoss
        ? earlyValue - improvement * targetThreshold
        : earlyValue + improvement * targetThreshold;

      // Find convergence point - where we reach target threshold
      let convergenceStep = validValues.length;
      for (let i = 0; i < validValues.length; i++) {
        const value = validValues[i];
        const reached = isLoss
          ? value <= targetValue
          : value >= targetValue;

        if (reached) {
          convergenceStep = i;
          break;
        }
      }

      const convergenceRate = improvement / convergenceStep;
      const efficiency = improvement / validValues.length;

      data.push({
        runName: run.run_name || `Run ${_idx + 1}`,
        stepsToConvergence: convergenceStep,
        convergenceRate,
        finalValue,
        earlyValue,
        improvement,
        efficiency,
        color: COLORS[_idx % COLORS.length],
      });
    });

    // Sort by convergence speed (faster first)
    return data.sort((a, b) => a.stepsToConvergence - b.stepsToConvergence);
  }, [runs, selectedMetric, targetThreshold]);

  const fastestRun = convergenceData.length > 0 ? convergenceData[0] : null;
  const slowestRun = convergenceData.length > 0 ? convergenceData[convergenceData.length - 1] : null;

  const chartContent = (
    <div className="space-y-4">
      {/* Metric Selector */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Metric:</label>
        <Select value={selectedMetric} onValueChange={setSelectedMetric}>
          <SelectTrigger>
            <SelectValue placeholder="Select metric" />
          </SelectTrigger>
          <SelectContent>
            {availableMetrics.map(metric => (
              <SelectItem key={metric} value={metric}>
                {metric}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Summary Statistics */}
      {fastestRun && slowestRun && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pb-4 border-b">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Zap className="h-3 w-3" />
              Fastest Convergence
            </div>
            <div className="text-sm font-medium text-green-600 dark:text-green-400">
              {fastestRun.runName}
            </div>
            <div className="text-xs text-muted-foreground">
              {fastestRun.stepsToConvergence} steps
            </div>
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Clock className="h-3 w-3" />
              Slowest Convergence
            </div>
            <div className="text-sm font-medium text-red-600 dark:text-red-400">
              {slowestRun.runName}
            </div>
            <div className="text-xs text-muted-foreground">
              {slowestRun.stepsToConvergence} steps
            </div>
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <TrendingUp className="h-3 w-3" />
              Speed Ratio
            </div>
            <div className="text-sm font-medium">
              {(slowestRun.stepsToConvergence / fastestRun.stepsToConvergence).toFixed(2)}x
            </div>
            <div className="text-xs text-muted-foreground">
              slower than fastest
            </div>
          </div>
        </div>
      )}

      {/* Bar Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={convergenceData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            type="number"
            label={{ value: 'Steps to Convergence', position: 'insideBottom', offset: -5 }}
            className="text-xs"
          />
          <YAxis
            type="category"
            dataKey="runName"
            width={100}
            className="text-xs"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
            }}
            formatter={(value: number, name: string) => {
              if (name === 'stepsToConvergence') return [value, 'Steps'];
              return [value, name];
            }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              const data = payload[0].payload as ConvergenceMetrics;
              return (
                <div className="bg-card border border-border rounded-lg p-3 shadow-lg space-y-2">
                  <div className="font-semibold text-sm">{data.runName}</div>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Steps:</span>
                      <span className="font-medium">{data.stepsToConvergence}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Rate:</span>
                      <span className="font-medium">{data.convergenceRate.toFixed(6)}/step</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Efficiency:</span>
                      <span className="font-medium">{data.efficiency.toFixed(6)}/step</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Improvement:</span>
                      <span className="font-medium">{data.improvement.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              );
            }}
          />
          <Bar dataKey="stepsToConvergence" radius={[0, 4, 4, 0]}>
            {convergenceData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Efficiency Table */}
      <div className="border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-muted">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Run</th>
              <th className="px-3 py-2 text-right font-medium">Steps</th>
              <th className="px-3 py-2 text-right font-medium">Rate</th>
              <th className="px-3 py-2 text-right font-medium">Efficiency</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {convergenceData.map((data, _idx) => (
              <tr key={data.runName} className="hover:bg-muted/50 transition-colors duration-200">
                <td className="px-3 py-2 font-medium">{data.runName}</td>
                <td className="px-3 py-2 text-right">{data.stepsToConvergence}</td>
                <td className="px-3 py-2 text-right">{data.convergenceRate.toFixed(6)}</td>
                <td className="px-3 py-2 text-right">{data.efficiency.toFixed(6)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <ChartCard
      title="Convergence Speed Analysis"
      description={`Steps required to reach ${(targetThreshold * 100).toFixed(0)}% of final performance`}
      isEmpty={convergenceData.length === 0}
      emptyMessage={`Insufficient data to analyze convergence speed for ${selectedMetric}`}
      badge={
        convergenceData.length > 0
          ? { label: `${convergenceData.length} Runs`, variant: 'secondary' }
          : undefined
      }
      enableSettings={false}
      enableFullscreen={true}
      enableExport={false}
    >
      {chartContent}
    </ChartCard>
  );
}
