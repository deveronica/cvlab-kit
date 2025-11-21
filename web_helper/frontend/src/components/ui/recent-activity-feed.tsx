import React from "react";

/**
 * Recent Activity Feed
 *
 * Shows recent notable events and achievements
 */

import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import {
  Activity,
  CheckCircle,
  XCircle,
  TrendingUp,
  _TrendingDown,
  Clock,
  Zap,
} from 'lucide-react';
import type { Run } from '../../lib/types';

export interface RecentActivityFeedProps {
  runs: Run[];
  keyMetric: string;
  metricDirection: 'maximize' | 'minimize';
  className?: string;
}

interface ActivityItem {
  icon: React.ElementType;
  iconColor: string;
  text: string;
  time?: string;
}

export function RecentActivityFeed({
  runs,
  keyMetric,
  metricDirection,
  className,
}: RecentActivityFeedProps) {
  const activities = useMemo(() => {
    const items: ActivityItem[] = [];
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

    // Count runs completed today
    const completedToday = runs.filter(run => {
      if (run.status !== 'completed' || !run.finished_at) return false;
      const finishedDate = new Date(run.finished_at);
      return finishedDate >= today;
    });

    if (completedToday.length > 0) {
      items.push({
        icon: CheckCircle,
        iconColor: 'text-green-600 dark:text-green-400',
        text: `${completedToday.length} run${completedToday.length > 1 ? 's' : ''} completed today`,
      });
    }

    // Recent failures
    const recentFailed = runs
      .filter(run => run.status === 'failed')
      .sort((a, b) => {
        const aTime = a.finished_at || a.started_at || '';
        const bTime = b.finished_at || b.started_at || '';
        return bTime.localeCompare(aTime);
      })
      .slice(0, 1);

    if (recentFailed.length > 0) {
      const failedRun = recentFailed[0];
      const failedTime = failedRun.finished_at || failedRun.started_at;
      items.push({
        icon: XCircle,
        iconColor: 'text-red-600 dark:text-red-400',
        text: `Run failed: ${failedRun.run_name ? (failedRun.run_name.substring(0, 25) + (failedRun.run_name.length > 25 ? '...' : '')) : 'Unknown'}`,
        time: failedTime,
      });
    }

    // Find best metric achievement
    const completedRuns = runs.filter(run => {
      const metricValue = run.metrics?.final?.[keyMetric];
      return run.status === 'completed' && typeof metricValue === 'number';
    });

    if (completedRuns.length > 0) {
      const sorted = [...completedRuns].sort((a, b) => {
        const aValue = a.metrics?.final?.[keyMetric] as number;
        const bValue = b.metrics?.final?.[keyMetric] as number;
        return metricDirection === 'maximize' ? bValue - aValue : aValue - bValue;
      });

      const best = sorted[0];
      const bestValue = best.metrics?.final?.[keyMetric] as number;
      const bestTime = best.finished_at;

      // Check if this is recent (within last week)
      if (bestTime && new Date(bestTime) >= lastWeek) {
        items.push({
          icon: TrendingUp,
          iconColor: 'text-blue-600 dark:text-blue-400',
          text: `New best${keyMetric ? ` ${keyMetric.replace(/_/g, ' ')}` : ' metric'}: ${bestValue.toFixed(4)}`,
          time: bestTime,
        });
      }
    }

    // Calculate average duration trend
    const recentCompleted = runs
      .filter(run => {
        if (run.status !== 'completed' || !run.started_at || !run.finished_at) return false;
        const finishedDate = new Date(run.finished_at);
        return finishedDate >= lastWeek;
      })
      .sort((a, b) => {
        const aTime = a.finished_at || '';
        const bTime = b.finished_at || '';
        return bTime.localeCompare(aTime);
      });

    if (recentCompleted.length >= 4) {
      // Compare first half vs second half
      const midpoint = Math.floor(recentCompleted.length / 2);
      const recent = recentCompleted.slice(0, midpoint);
      const older = recentCompleted.slice(midpoint);

      const calcAvgDuration = (runs: typeof recentCompleted) => {
        const durations = runs.map(run => {
          const start = new Date(run.started_at!).getTime();
          const end = new Date(run.finished_at!).getTime();
          return end - start;
        });
        return durations.reduce((a, b) => a + b, 0) / durations.length;
      };

      const recentAvg = calcAvgDuration(recent);
      const olderAvg = calcAvgDuration(older);
      const improvement = ((olderAvg - recentAvg) / olderAvg) * 100;

      if (Math.abs(improvement) > 5) {
        items.push({
          icon: improvement > 0 ? Zap : Clock,
          iconColor: improvement > 0
            ? 'text-green-600 dark:text-green-400'
            : 'text-yellow-600 dark:text-yellow-400',
          text: `Avg runtime ${improvement > 0 ? 'improved' : 'increased'} by ${Math.abs(improvement).toFixed(0)}%`,
        });
      }
    }

    // Currently running
    const running = runs.filter(run => run.status === 'running');
    if (running.length > 0) {
      items.push({
        icon: Activity,
        iconColor: 'text-blue-600 dark:text-blue-400',
        text: `${running.length} run${running.length > 1 ? 's' : ''} currently running`,
      });
    }

    return items;
  }, [runs, keyMetric, metricDirection]);

  const formatTimeAgo = (time?: string) => {
    if (!time) return '';
    const date = new Date(time);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  if (activities.length === 0) {
    return (
      <Card variant="compact" className={className}>
        <CardHeader variant="compact">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" />
            <CardTitle size="sm">Recent Activity</CardTitle>
          </div>
        </CardHeader>
        <CardContent variant="compact">
          <div className="text-center py-6 text-muted-foreground text-sm">
            No recent activity
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="compact" className={className}>
      <CardHeader variant="compact">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          <CardTitle size="sm">Recent Activity</CardTitle>
        </div>
      </CardHeader>
      <CardContent variant="compact">
        <div className="space-y-2">
          {activities.map((activity, index) => {
            const Icon = activity.icon;
            return (
              <div
                key={index}
                className="flex items-start gap-2 p-2 rounded-lg hover:bg-muted/50 transition-colors"
              >
                <div className="flex-shrink-0 mt-0.5">
                  <Icon className={`h-4 w-4 ${activity.iconColor}`} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm">{activity.text}</p>
                  {activity.time && (
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {formatTimeAgo(activity.time)}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
