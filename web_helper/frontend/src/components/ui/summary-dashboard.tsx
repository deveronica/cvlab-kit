import React from "react";
import { useMemo } from 'react';
import { ProjectStatsCard } from './project-stats-card';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip} from 'recharts';
import { Activity, CheckCircle, TrendingUp, Timer } from 'lucide-react';
import { RunStatusBadge } from './run-status-badge';

interface RunData {
  run_name: string;
  status: string;
  started_at?: string;
  finished_at?: string;
  final_metrics?: Record<string, any>;
  max_metrics?: Record<string, any>;
  min_metrics?: Record<string, any>;
}

interface SummaryDashboardProps {
  runs: RunData[];
  keyMetric?: string;
  metricDirection?: 'maximize' | 'minimize';
  className?: string;
}

export function SummaryDashboard({ runs, keyMetric = 'val/acc', metricDirection = 'maximize', className }: SummaryDashboardProps) {
  const stats = useMemo(() => {
    const totalRuns = runs.length;
    const completedRuns = runs.filter(r => r.status === 'completed');
    const runningRuns = runs.filter(r => r.status === 'running');
    const failedRuns = runs.filter(r => r.status === 'failed');
    const successRate = totalRuns > 0 ? (completedRuns.length / totalRuns) * 100 : 0;

    // Calculate average duration
    const durations = completedRuns
      .filter(r => r.started_at && r.finished_at)
      .map(r => {
        const start = new Date(r.started_at!).getTime();
        const end = new Date(r.finished_at!).getTime();
        return end - start;
      });

    const avgDurationMs = durations.length > 0
      ? durations.reduce((a, b) => a + b, 0) / durations.length
      : 0;

    const avgDuration = avgDurationMs > 0
      ? `${Math.round(avgDurationMs / (1000 * 60))}m`
      : '-';

    // Find best metric value
    let bestMetric: { value: number; run_name: string } | null = null;
    const metricsSource = metricDirection === 'maximize' ? 'max_metrics' : 'min_metrics';

    completedRuns.forEach(run => {
      const metrics = run[metricsSource];
      if (metrics && metrics[keyMetric] !== undefined) {
        const value = parseFloat(metrics[keyMetric]);
        if (!isNaN(value)) {
          const isBetter = bestMetric === null || (
            metricDirection === 'maximize' ? value > bestMetric.value : value < bestMetric.value
          );
          if (isBetter) {
            bestMetric = { value, run_name: run.run_name };
          }
        }
      }
    });

    // Calculate distribution data for pie chart
    const distributionData = [
      { name: 'Completed', value: completedRuns.length, color: '#10b981' },
      { name: 'Running', value: runningRuns.length, color: '#3b82f6' },
      { name: 'Failed', value: failedRuns.length, color: '#ef4444' },
    ].filter(d => d.value > 0);

    return {
      totalRuns,
      completedCount: completedRuns.length,
      runningCount: runningRuns.length,
      failedCount: failedRuns.length,
      successRate: successRate.toFixed(1),
      avgDuration,
      bestMetric,
      distributionData,
    };
  }, [runs, keyMetric, metricDirection]);

  // Get recent runs (last 5)
  const recentRuns = useMemo(() => {
    return [...runs]
      .sort((a, b) => {
        const dateA = a.started_at ? new Date(a.started_at).getTime() : 0;
        const dateB = b.started_at ? new Date(b.started_at).getTime() : 0;
        return dateB - dateA;
      })
      .slice(0, 5);
  }, [runs]);

  return (
    <div className={className}>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
        <ProjectStatsCard
          title="Total Runs"
          value={stats.totalRuns}
          subtitle={`${stats.completedCount} completed`}
          icon={Activity}
        />

        <ProjectStatsCard
          title="Success Rate"
          value={`${stats.successRate}%`}
          subtitle={`${stats.failedCount} failed`}
          icon={CheckCircle}
          trend={
            stats.totalRuns > 10 ? { value: parseFloat(stats.successRate) - 85 } : undefined
          }
        />

        <ProjectStatsCard
          title="Avg Duration"
          value={stats.avgDuration}
          subtitle="per run"
          icon={Timer}
        />

        <ProjectStatsCard
          title={`Best ${keyMetric}`}
          value={stats.bestMetric ? stats.bestMetric.value.toFixed(4) : '-'}
          subtitle={stats.bestMetric?.run_name ? stats.bestMetric.run_name.substring(0, 20) : undefined}
          icon={TrendingUp}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-2">
        {/* Status Distribution Chart */}
        <Card variant="compact" className="overflow-visible">
          <CardHeader variant="compact">
            <CardTitle size="sm">Run Status Distribution</CardTitle>
          </CardHeader>
          <CardContent variant="compact" className="overflow-visible py-4">
            <div className="w-full" style={{ height: '200px', overflow: 'visible' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                  <Pie
                    data={stats.distributionData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={65}
                    label={({ name, value }) => `${name}: ${value}`}
                    labelLine={true}
                  >
                    {stats.distributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Recent Runs */}
        <Card variant="compact" className="lg:col-span-2">
          <CardHeader variant="compact">
            <CardTitle size="sm">Recent Runs</CardTitle>
          </CardHeader>
          <CardContent variant="compact">
            <div className="space-y-1">
              {recentRuns.map((run, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-1 rounded-md hover:bg-muted/50 transition-colors duration-200"
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <RunStatusBadge status={run.status as any} showIcon={true} />
                    <span className="text-sm font-medium truncate">{run.run_name}</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {run.started_at
                      ? new Date(run.started_at).toLocaleDateString(undefined, {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })
                      : '-'}
                  </div>
                </div>
              ))}
              {recentRuns.length === 0 && (
                <div className="text-center text-muted-foreground py-6 text-sm">
                  No runs available
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
