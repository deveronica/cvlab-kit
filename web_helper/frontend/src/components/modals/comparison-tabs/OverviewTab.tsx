import React from "react";
/**
 * Overview Tab
 *
 * Provides high-level summary of compared runs:
 * - Metric selector dropdown
 * - Highest/Lowest/Final mini tabs for metric views
 * - Best/worst run identification
 * - Quick statistics overview
 * - Manual direction button group
 */

import { useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardDescription } from '../../ui/card';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';
import { TrendingUp, TrendingDown, Award, AlertTriangle } from 'lucide-react';
import { InlineEmptyState } from '../../charts/EmptyState';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../ui/select';
import type { Run } from '../../../lib/types';

interface OverviewTabProps {
  runs: Run[];
  availableMetrics: string[];
}

export function OverviewTab({ runs, availableMetrics }: OverviewTabProps) {
  // Selected metric for focused view
  const [selectedMetric, setSelectedMetric] = useState<string>(availableMetrics[0] || '');

  // Mini tab for view mode (Best/Worst/All)
  const [viewMode, setViewMode] = useState<'best' | 'worst' | 'all'>('best');

  // Calculate metric values for selected metric only
  const metricData = useMemo(() => {
    if (!selectedMetric) return null;

    const values = runs
      .map(run => ({
        runName: run.run_name,
        value: run.metrics?.final?.[selectedMetric],
      }))
      .filter(item => typeof item.value === 'number') as { runName: string; value: number }[];

    if (values.length === 0) return null;

    // Auto-detect direction based on metric name
    const lowerIsBetter = ['loss', 'error', 'mse', 'mae', 'rmse']
      .some(keyword => selectedMetric.toLowerCase().includes(keyword));

    const sorted = [...values].sort((a, b) =>
      lowerIsBetter ? a.value - b.value : b.value - a.value
    );

    const best = sorted[0];
    const worst = sorted[sorted.length - 1];
    const avg = values.reduce((sum, item) => sum + item.value, 0) / values.length;
    const std = Math.sqrt(
      values.reduce((sum, item) => sum + Math.pow(item.value - avg, 2), 0) / values.length
    );

    return {
      metricKey: selectedMetric,
      values,
      sorted,
      best,
      worst,
      avg,
      std,
      lowerIsBetter,
      count: values.length,
    };
  }, [runs, selectedMetric]);

  // Get runs to display based on view mode
  const displayRuns = useMemo(() => {
    if (!metricData) return [];

    if (viewMode === 'best') {
      // Show top 5 best performing runs
      return metricData.sorted.slice(0, Math.min(5, metricData.sorted.length));
    } else if (viewMode === 'worst') {
      // Show top 5 worst performing runs
      return metricData.sorted.slice(Math.max(0, metricData.sorted.length - 5)).reverse();
    } else {
      // Show all runs with final values
      return metricData.values;
    }
  }, [metricData, viewMode]);

  if (availableMetrics.length === 0) {
    return <InlineEmptyState message="No metrics available for comparison" />;
  }

  if (!metricData) {
    return <InlineEmptyState message="No data available for selected metric" />;
  }

  return (
    <div className="space-y-6">
      {/* Header: Metric Selector + View Mode Tabs */}
      <div className="space-y-3">
        {/* Metric Info Row */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 flex-1 min-w-0 overflow-visible">
            <label className="text-sm font-medium text-muted-foreground shrink-0">Metric:</label>
            <Select value={selectedMetric} onValueChange={setSelectedMetric}>
              <SelectTrigger className="h-9 max-w-xs">
                <SelectValue />
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
          <Badge variant="outline" className="shrink-0">
            {metricData.lowerIsBetter ? (
              <><TrendingDown className="h-3 w-3 mr-1" />Lower is Better</>
            ) : (
              <><TrendingUp className="h-3 w-3 mr-1" />Higher is Better</>
            )}
          </Badge>
        </div>

        {/* View Mode Tabs */}
        <div className="flex bg-muted rounded-lg p-1 gap-1">
          <button
            onClick={() => setViewMode('best')}
            className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
              viewMode === 'best'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
            }`}
          >
            <Award className="h-4 w-4" />
            Best
          </button>
          <button
            onClick={() => setViewMode('worst')}
            className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
              viewMode === 'worst'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
            }`}
          >
            <AlertTriangle className="h-4 w-4" />
            Worst
          </button>
          <button
            onClick={() => setViewMode('all')}
            className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
              viewMode === 'all'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
            }`}
          >
            <TrendingUp className="h-4 w-4" />
            All Runs
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-4 gap-3">
        <Card variant="compact">
          <CardContent variant="compact" className="p-4">
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Best</div>
              <div className="text-2xl font-bold font-mono">{metricData.best.value.toFixed(4)}</div>
            </div>
          </CardContent>
        </Card>
        <Card variant="compact">
          <CardContent variant="compact" className="p-4">
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Worst</div>
              <div className="text-2xl font-bold font-mono">{metricData.worst.value.toFixed(4)}</div>
            </div>
          </CardContent>
        </Card>
        <Card variant="compact">
          <CardContent variant="compact" className="p-4">
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Average</div>
              <div className="text-2xl font-bold font-mono">{metricData.avg.toFixed(4)}</div>
            </div>
          </CardContent>
        </Card>
        <Card variant="compact">
          <CardContent variant="compact" className="p-4">
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">Std Dev</div>
              <div className="text-2xl font-bold font-mono">{metricData.std.toFixed(4)}</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Runs List */}
      <Card variant="compact">
        <CardHeader variant="compact">
          <CardDescription>
            {viewMode === 'best' && 'Top 5 best performing runs'}
            {viewMode === 'worst' && 'Top 5 worst performing runs'}
            {viewMode === 'all' && `All ${metricData.count} runs`}
          </CardDescription>
        </CardHeader>
        <CardContent variant="compact" className="p-0">
          <div className="divide-y">
            {displayRuns.map((item, idx) => {
              const isBest = item.runName === metricData.best.runName;
              const isWorst = item.runName === metricData.worst.runName;

              return (
                <div
                  key={item.runName}
                  className={`flex items-center justify-between px-4 py-3 hover:bg-muted/50 transition-colors ${
                    isBest ? 'bg-primary/5' : ''
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0" style={{ flex: '1 1 0%' }}>
                    <div className="text-sm font-medium text-muted-foreground w-6 text-center shrink-0">
                      #{idx + 1}
                    </div>
                    <div className="text-sm font-mono truncate" style={{ flex: '1 1 0%', minWidth: 0 }} title={item.runName}>
                      {item.runName}
                    </div>
                  </div>
                  <div className="flex items-center gap-3 shrink-0 ml-4">
                    {isBest && (
                      <Badge variant="default" className="text-xs">
                        <Award className="h-3 w-3 mr-1" />
                        Best
                      </Badge>
                    )}
                    {isWorst && (
                      <Badge variant="outline" className="text-xs">
                        <AlertTriangle className="h-3 w-3 mr-1" />
                        Worst
                      </Badge>
                    )}
                    <div className="text-lg font-bold font-mono">
                      {item.value.toFixed(4)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
