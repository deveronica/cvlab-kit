import React from "react";
/**
 * Training vs Validation Comparison Chart
 * Compares training and validation metrics to detect overfitting/underfitting
 */

import { useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { ChartCard } from '../ui/chart-card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Badge } from '../ui/badge';
import { TrendingUp, TrendingDown } from 'lucide-react';
import type { Run } from '../../lib/types';

interface TrainingValidationComparisonChartProps {
  runs: Run[];
  availableMetrics: string[];
  height?: number;
}

interface ChartDataPoint {
  step: number;
  [key: string]: number | string;
}

interface OverfittingAnalysis {
  isOverfitting: boolean;
  divergencePoint: number | null;
  maxGap: number;
}

const COLORS = {
  training: '#10b981', // green
  validation: '#ef4444', // red
};

export function TrainingValidationComparisonChart({
  runs,
  availableMetrics,
  height = 400,
}: TrainingValidationComparisonChartProps) {
  const [selectedMetric, setSelectedMetric] = useState<string>(availableMetrics[0] || '');
  const [selectedRun, setSelectedRun] = useState<string>(runs[0]?.run_name || '');

  const { chartData, overfittingAnalysis, hasData } = useMemo(() => {
    if (!selectedMetric || !selectedRun) {
      return { chartData: [], overfittingAnalysis: null, hasData: false };
    }

    const run = runs.find(r => r.run_name === selectedRun);
    if (!run) {
      return { chartData: [], overfittingAnalysis: null, hasData: false };
    }

    const timeseries = run.metrics?.timeseries || [];

    if (timeseries.length === 0) {
      return { chartData: [], overfittingAnalysis: null, hasData: false };
    }

    // Find training and validation versions of the metric
    const firstPoint = timeseries[0];
    const allKeys = Object.keys(firstPoint.values || {});

    const baseName = selectedMetric.replace('val_', '').replace('train_', '');
    const trainKey = allKeys.find(k =>
      k.includes('train') && k.includes(baseName)
    );
    const valKey = allKeys.find(k =>
      k.includes('val') && k.includes(baseName)
    );

    if (!trainKey && !valKey) {
      return { chartData: [], overfittingAnalysis: null, hasData: false };
    }

    const data: ChartDataPoint[] = [];
    let maxGap = 0;
    let divergencePoint: number | null = null;
    let consecutiveDiverge = 0;

    timeseries.forEach((point, i) => {
      const trainVal = trainKey ? point.values[trainKey] : undefined;
      const valVal = valKey ? point.values[valKey] : undefined;

      const dataPoint: ChartDataPoint = { step: point.step };

      if (typeof trainVal === 'number' && !isNaN(trainVal)) {
        dataPoint.training = trainVal;
      }
      if (typeof valVal === 'number' && !isNaN(valVal)) {
        dataPoint.validation = valVal;
      }

      data.push(dataPoint);

      // Overfitting detection: validation gets worse while training improves
      if (typeof trainVal === 'number' && typeof valVal === 'number') {
        const gap = Math.abs(valVal - trainVal);
        if (gap > maxGap) {
          maxGap = gap;
        }

        // Check if validation is getting worse (for loss) or not improving (for accuracy)
        if (i > 10) { // Start checking after warmup period
          const isLoss = selectedMetric.toLowerCase().includes('loss') ||
                        selectedMetric.toLowerCase().includes('error');

          if (isLoss) {
            // For loss: overfitting if val_loss > train_loss and gap increasing
            if (valVal > trainVal && gap > 0.1) {
              consecutiveDiverge++;
              if (consecutiveDiverge >= 5 && divergencePoint === null) {
                divergencePoint = i - 4;
              }
            } else {
              consecutiveDiverge = 0;
            }
          } else {
            // For accuracy: overfitting if train_acc > val_acc and gap increasing
            if (trainVal > valVal && gap > 0.05) {
              consecutiveDiverge++;
              if (consecutiveDiverge >= 5 && divergencePoint === null) {
                divergencePoint = i - 4;
              }
            } else {
              consecutiveDiverge = 0;
            }
          }
        }
      }
    });

    const analysis: OverfittingAnalysis = {
      isOverfitting: divergencePoint !== null,
      divergencePoint,
      maxGap,
    };

    return { chartData: data, overfittingAnalysis: analysis, hasData: data.length > 0 };
  }, [runs, selectedMetric, selectedRun]);

  // Chart content
  const chartContent = (
    <div className="space-y-4">
      {/* Selectors */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
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
        <div className="space-y-2">
          <label className="text-sm font-medium">Run:</label>
          <Select value={selectedRun} onValueChange={setSelectedRun}>
            <SelectTrigger>
              <SelectValue placeholder="Select run" />
            </SelectTrigger>
            <SelectContent>
              {runs.map(run => (
                <SelectItem key={run.run_name} value={run.run_name || ''}>
                  {run.run_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Chart */}
      {hasData ? (
        <>
          <ResponsiveContainer width="100%" height={height - 120}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="step"
                label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
                className="text-xs"
              />
              <YAxis
                label={{ value: selectedMetric, angle: -90, position: 'insideLeft' }}
                className="text-xs"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                }}
              />
              <Legend />

              {overfittingAnalysis?.divergencePoint && (
                <ReferenceLine
                  x={overfittingAnalysis.divergencePoint}
                  stroke="#ef4444"
                  strokeDasharray="3 3"
                  label={{
                    value: 'Overfitting starts',
                    position: 'top',
                    fill: '#ef4444',
                    fontSize: 12,
                  }}
                />
              )}

              <Line
                type="monotone"
                dataKey="training"
                stroke={COLORS.training}
                strokeWidth={2}
                dot={false}
                name="Training"
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="validation"
                stroke={COLORS.validation}
                strokeWidth={2}
                dot={false}
                name="Validation"
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>

          {/* Analysis Summary */}
          {overfittingAnalysis && (
            <div className="space-y-3 pt-2 border-t">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">Status</div>
                  <div className="flex items-center gap-2">
                    {overfittingAnalysis.isOverfitting ? (
                      <>
                        <TrendingDown className="h-4 w-4 text-red-600 dark:text-red-400" />
                        <span className="text-sm font-medium text-red-600 dark:text-red-400">
                          Overfitting
                        </span>
                      </>
                    ) : (
                      <>
                        <TrendingUp className="h-4 w-4 text-green-600 dark:text-green-400" />
                        <span className="text-sm font-medium text-green-600 dark:text-green-400">
                          Healthy
                        </span>
                      </>
                    )}
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">Divergence Point</div>
                  <div className="text-sm font-medium">
                    {overfittingAnalysis.divergencePoint !== null
                      ? `Step ${overfittingAnalysis.divergencePoint}`
                      : 'N/A'}
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">Max Gap</div>
                  <div className="text-sm font-medium">
                    {overfittingAnalysis.maxGap.toFixed(4)}
                  </div>
                </div>
              </div>
              {/* Overfitting Detection Criteria */}
              <div className="text-xs text-muted-foreground bg-muted/30 rounded p-2">
                <strong>Detection Criteria:</strong> Loss metrics: val &gt; train AND gap &gt; 0.1 for 5+ consecutive steps.
                Accuracy metrics: train &gt; val AND gap &gt; 0.05 for 5+ consecutive steps. Analysis starts after 10-step warmup.
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="flex items-center justify-center h-[200px] text-muted-foreground text-sm">
          No training/validation data found for selected metric and run
        </div>
      )}
    </div>
  );

  return (
    <ChartCard
      title="Training vs Validation Comparison"
      description="Compare training and validation metrics to detect overfitting"
      isEmpty={availableMetrics.length === 0 || runs.length === 0}
      emptyMessage="No training/validation metrics found"
      badge={
        overfittingAnalysis?.isOverfitting
          ? { label: 'Overfitting Detected', variant: 'destructive' }
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
