import React from "react";
/**
 * Metric Statistics Dialog
 *
 * Shows detailed statistics for a metric across all training steps:
 * - Summary statistics (min, max, mean, std, median, percentiles)
 * - Best value and step
 * - Trend analysis
 * - Mini line chart visualization
 */

import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from './dialog';
import { Badge } from './badge';
import { Button } from './button';
import { Loader2, TrendingUp, TrendingDown, Minus, BarChart3, Download } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

interface MetricStatisticsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  project: string;
  runName: string;
  metricName: string;
}

interface MetricStats {
  metric_name: string;
  count: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  median: number;
  q25: number;
  q75: number;
  best: number;
  best_step: number;
  latest: number;
  is_lower_better: boolean;
  trend: 'improving' | 'degrading' | 'stable';
  improvement_pct: number;
  series: Array<{ step: number; value: number }>;
}

export function MetricStatisticsDialog({
  isOpen,
  onClose,
  project,
  runName,
  metricName,
}: MetricStatisticsDialogProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<MetricStats | null>(null);

  useEffect(() => {
    if (isOpen && project && runName && metricName) {
      fetchStatistics();
    }
  }, [isOpen, project, runName, metricName]);

  const fetchStatistics = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/metrics/statistics/${encodeURIComponent(project)}/${encodeURIComponent(runName)}?metric_name=${encodeURIComponent(metricName)}`
      );
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.detail || 'Failed to fetch statistics');
      }

      setStats(data.data);
    } catch (err: any) {
      setError(err.message || 'Failed to load metric statistics');
    } finally {
      setLoading(false);
    }
  };

  const formatValue = (value: number) => {
    if (Math.abs(value) < 0.01 || Math.abs(value) > 1000) {
      return value.toExponential(3);
    }
    return value.toFixed(4);
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving':
        return <TrendingUp className="h-4 w-4 text-green-600 dark:text-green-400" />;
      case 'degrading':
        return <TrendingDown className="h-4 w-4 text-red-600 dark:text-red-400" />;
      default:
        return <Minus className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getTrendBadge = (trend: string) => {
    const variants = {
      improving: 'bg-green-50 text-green-700 border-green-200 dark:bg-green-950/50 dark:text-green-400 dark:border-green-800/30',
      degrading: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/50 dark:text-red-400 dark:border-red-800/30',
      stable: 'bg-gray-50 text-gray-700 border-gray-200 dark:bg-gray-950/50 dark:text-gray-400 dark:border-gray-800/30',
    };
    return (
      <Badge className={variants[trend as keyof typeof variants] || variants.stable}>
        <div className="flex items-center gap-1">
          {getTrendIcon(trend)}
          <span className="capitalize">{trend}</span>
        </div>
      </Badge>
    );
  };

  const handleExportCSV = () => {
    if (!stats) return;

    const csv = [
      ['Metric', stats.metric_name],
      ['Run', runName],
      ['Project', project],
      [''],
      ['Statistic', 'Value'],
      ['Count', stats.count],
      ['Min', stats.min],
      ['Max', stats.max],
      ['Mean', stats.mean],
      ['Std Dev', stats.std],
      ['Median', stats.median],
      ['Q25', stats.q25],
      ['Q75', stats.q75],
      ['Best', stats.best],
      ['Best Step', stats.best_step],
      ['Latest', stats.latest],
      ['Trend', stats.trend],
      ['Improvement %', stats.improvement_pct],
    ]
      .map((row) => row.join(','))
      .join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${runName}_${metricName}_statistics.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <DialogTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Metric Statistics
              </DialogTitle>
              <DialogDescription className="mt-1">
                Step-by-step analysis for <span className="font-mono font-medium">{metricName}</span>
              </DialogDescription>
            </div>
            {stats && (
              <Button variant="outline" size="sm" onClick={handleExportCSV}>
                <Download className="h-3.5 w-3.5 mr-1.5" />
                Export CSV
              </Button>
            )}
          </div>
        </DialogHeader>

        <div className="space-y-4">
          {/* Loading State */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Loader2 className="h-8 w-8 animate-spin mb-3" />
              <p className="text-sm">Calculating statistics...</p>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="bg-destructive/10 border border-destructive/20 rounded-md p-4 text-center">
              <p className="text-sm text-destructive font-medium">{error}</p>
            </div>
          )}

          {/* Statistics Display */}
          {stats && !loading && !error && (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Card variant="compact">
                  <CardHeader variant="compact">
                    <CardTitle size="sm" className="text-muted-foreground">Best Value</CardTitle>
                  </CardHeader>
                  <CardContent variant="compact">
                    <div className="space-y-1">
                      <p className="text-lg font-semibold">{formatValue(stats.best)}</p>
                      <p className="text-xs text-muted-foreground">at step {stats.best_step}</p>
                    </div>
                  </CardContent>
                </Card>

                <Card variant="compact">
                  <CardHeader variant="compact">
                    <CardTitle size="sm" className="text-muted-foreground">Latest</CardTitle>
                  </CardHeader>
                  <CardContent variant="compact">
                    <p className="text-lg font-semibold">{formatValue(stats.latest)}</p>
                  </CardContent>
                </Card>

                <Card variant="compact">
                  <CardHeader variant="compact">
                    <CardTitle size="sm" className="text-muted-foreground">Mean ± Std</CardTitle>
                  </CardHeader>
                  <CardContent variant="compact">
                    <p className="text-lg font-semibold">
                      {formatValue(stats.mean)}
                    </p>
                    <p className="text-xs text-muted-foreground">± {formatValue(stats.std)}</p>
                  </CardContent>
                </Card>

                <Card variant="compact">
                  <CardHeader variant="compact">
                    <CardTitle size="sm" className="text-muted-foreground">Trend</CardTitle>
                  </CardHeader>
                  <CardContent variant="compact">
                    <div className="space-y-1">
                      {getTrendBadge(stats.trend)}
                      <p className="text-xs text-muted-foreground">
                        {stats.improvement_pct > 0 ? '+' : ''}
                        {stats.improvement_pct.toFixed(1)}%
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Detailed Statistics Table */}
              <Card>
                <CardHeader>
                  <CardTitle size="sm">Distribution Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Count:</span>
                      <span className="font-mono font-medium">{stats.count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Min:</span>
                      <span className="font-mono font-medium">{formatValue(stats.min)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Max:</span>
                      <span className="font-mono font-medium">{formatValue(stats.max)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Median:</span>
                      <span className="font-mono font-medium">{formatValue(stats.median)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Q25:</span>
                      <span className="font-mono font-medium">{formatValue(stats.q25)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Q75:</span>
                      <span className="font-mono font-medium">{formatValue(stats.q75)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Visualization */}
              <Card>
                <CardHeader>
                  <CardTitle size="sm">Value Over Steps</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={stats.series}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                        <XAxis
                          dataKey="step"
                          label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
                          tick={{ fontSize: 12 }}
                        />
                        <YAxis
                          label={{ value: stats.metric_name, angle: -90, position: 'insideLeft' }}
                          tick={{ fontSize: 12 }}
                          domain={['auto', 'auto']}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'hsl(var(--background))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px',
                            fontSize: '12px',
                          }}
                          formatter={(value: number) => [formatValue(value), stats.metric_name]}
                          labelFormatter={(label) => `Step: ${label}`}
                        />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="hsl(var(--primary))"
                          strokeWidth={2}
                          dot={false}
                          activeDot={{ r: 4 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Meta Information */}
              <div className="text-xs text-muted-foreground space-y-1 border-t border-border pt-3">
                <p>
                  <span className="font-medium">Run:</span> {runName}
                </p>
                <p>
                  <span className="font-medium">Project:</span> {project}
                </p>
                <p>
                  <span className="font-medium">Optimization:</span>{' '}
                  {stats.is_lower_better ? 'Lower is better' : 'Higher is better'}
                </p>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
