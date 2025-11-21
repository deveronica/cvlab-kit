import React from "react";
/**
 * Top Runs Comparison Table
 *
 * Displays top 10 runs ranked by the selected metric
 */

import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Trophy, Medal, Award } from 'lucide-react';
import type { Run } from '../../lib/types';

export interface TopRunsTableProps {
  runs: Run[];
  keyMetric: string;
  metricDirection: 'maximize' | 'minimize';
  secondaryMetrics?: string[];
  className?: string;
  onRunClick?: (runName: string) => void;
}

export function TopRunsTable({
  runs,
  keyMetric,
  metricDirection,
  secondaryMetrics = [],
  className,
  onRunClick,
}: TopRunsTableProps) {
  const topRuns = useMemo(() => {
    // Filter completed runs with the key metric
    const validRuns = runs.filter(run => {
      const metricValue = run.metrics?.final?.[keyMetric];
      return run.status === 'completed' && typeof metricValue === 'number';
    });

    // Sort by key metric
    const sorted = [...validRuns].sort((a, b) => {
      const aValue = a.metrics?.final?.[keyMetric] as number;
      const bValue = b.metrics?.final?.[keyMetric] as number;
      return metricDirection === 'maximize' ? bValue - aValue : aValue - bValue;
    });

    // Take top 10
    return sorted.slice(0, 10);
  }, [runs, keyMetric, metricDirection]);

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return <Trophy className="h-4 w-4 text-yellow-600 dark:text-yellow-500" />;
      case 2:
        return <Medal className="h-4 w-4 text-gray-400 dark:text-gray-500" />;
      case 3:
        return <Medal className="h-4 w-4 text-orange-600 dark:text-orange-500" />;
      default:
        return <Award className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const formatDuration = (startedAt?: string, finishedAt?: string) => {
    if (!startedAt || !finishedAt) return '-';
    const duration = new Date(finishedAt).getTime() - new Date(startedAt).getTime();
    const minutes = Math.round(duration / (1000 * 60));
    return minutes > 0 ? `${minutes}m` : '-';
  };

  const displayMetrics = useMemo(() => {
    return [keyMetric, ...secondaryMetrics.slice(0, 2)];
  }, [keyMetric, secondaryMetrics]);

  if (topRuns.length === 0) {
    return (
      <Card variant="compact" className={className} style={{ minHeight: '320px' }}>
        <CardHeader variant="compact">
          <CardTitle size="sm">Top Runs</CardTitle>
        </CardHeader>
        <CardContent variant="compact" className="flex items-center justify-center" style={{ minHeight: '260px' }}>
          <div className="text-center text-muted-foreground text-sm">
            No completed runs{keyMetric ? ` with ${keyMetric}` : ''} available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="compact" className={`w-full h-full !flex !flex-col ${className || ''}`}>
      <CardHeader variant="compact">
        <CardTitle size="sm">Top 10 Runs{keyMetric ? ` by ${keyMetric.replace(/_/g, ' ')}` : ''}</CardTitle>
      </CardHeader>
      <CardContent variant="compact" className="flex-1 flex flex-col">
        <div className="overflow-auto flex-1">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1 pr-1.5 font-medium text-muted-foreground text-xs">#</th>
                <th className="text-left py-1 px-1.5 font-medium text-muted-foreground text-xs">Run Name</th>
                {displayMetrics.map(metric => (
                  <th key={metric} className="text-right py-1 px-1.5 font-medium text-muted-foreground text-xs">
                    {metric.replace(/_/g, ' ')}
                  </th>
                ))}
                <th className="text-right py-1 pl-1.5 font-medium text-muted-foreground text-xs">Time</th>
              </tr>
            </thead>
            <tbody>
              {topRuns.map((run, index) => {
                const rank = index + 1;
                return (
                  <tr
                    key={run.run_name}
                    className={`border-b last:border-b-0 hover:bg-muted/50 transition-colors duration-200 ${
                      onRunClick ? 'cursor-pointer' : ''
                    }`}
                    onClick={() => onRunClick?.(run.run_name)}
                  >
                    <td className="py-1 pr-1.5">
                      <div className="flex items-center gap-0.5">
                        {getRankIcon(rank)}
                        <span className="text-xs font-medium">{rank}</span>
                      </div>
                    </td>
                    <td className="py-1 px-1.5">
                      <div className="font-mono text-xs truncate max-w-[200px]" title={run.run_name}>
                        {run.run_name}
                      </div>
                    </td>
                    {displayMetrics.map(metric => {
                      const value = run.metrics?.final?.[metric];
                      const display = typeof value === 'number' ? value.toFixed(4) : 'N/A';
                      const isKeyMetric = metric === keyMetric;
                      return (
                        <td key={metric} className="py-1 px-1.5 text-right">
                          <span className={`font-mono text-xs ${isKeyMetric ? 'font-semibold text-primary' : ''}`}>
                            {display}
                          </span>
                        </td>
                      );
                    })}
                    <td className="py-1 pl-1.5 text-right">
                      <span className="text-xs text-muted-foreground">
                        {formatDuration(run.started_at, run.finished_at)}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
