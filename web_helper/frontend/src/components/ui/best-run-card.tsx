import React from "react";
/**
 * Best Run Highlight Card
 *
 * Displays the best performing run based on a key metric
 */

import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Badge } from './badge';
import { Button } from './button';
import { Trophy, TrendingUp, Calendar } from 'lucide-react';
import type { Run } from '../../lib/types';

export interface BestRunCardProps {
  run: Run;
  keyMetric: string;
  secondaryMetrics?: string[];
  onOpenDetail?: (runId: string) => void;
}

export function BestRunCard({
  run,
  keyMetric,
  secondaryMetrics = [],
  onOpenDetail,
}: BestRunCardProps) {
  // Get key metric value
  const keyMetricValue = run.metrics?.final?.[keyMetric];
  const keyMetricDisplay = typeof keyMetricValue === 'number'
    ? keyMetricValue.toFixed(4)
    : 'N/A';

  return (
    <Card
      variant="compact"
      className="w-full h-full !flex !flex-col border-2 border-primary/30 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent shadow-lg shadow-primary/10 dark:shadow-primary/5 hover:shadow-xl hover:shadow-primary/15 transition-all duration-200"
    >
      <CardHeader variant="compact" className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Trophy className="h-5 w-5 text-yellow-600 dark:text-yellow-500 drop-shadow-md" />
              <div className="absolute inset-0 blur-sm bg-yellow-500/30 rounded-full" />
            </div>
            <CardTitle size="sm" className="text-base">Best Overall Run</CardTitle>
          </div>
          <Badge variant="default" className="text-xs font-semibold shadow-sm">Top Performer</Badge>
        </div>
      </CardHeader>
      <CardContent variant="compact" className="flex-1 flex flex-col">
        {/* Run Name - Compact */}
        <div className="mb-2">
          <div className="text-xs text-muted-foreground">Run Name</div>
          <div className="font-mono text-xs font-medium truncate" title={run.run_name}>
            {run.run_name}
          </div>
        </div>

        {/* Key Metric - Compact Display */}
        <div className="px-2 py-1.5 rounded-lg bg-primary/10 dark:bg-primary/5 border border-primary/20 mb-2">
          <div className="text-[10px] text-muted-foreground flex items-center gap-1 font-medium">
            <TrendingUp className="h-3 w-3" />
            {keyMetric.replace(/_/g, ' ').toUpperCase()}
          </div>
          <div className="text-2xl font-bold text-primary tracking-tight">
            {keyMetricDisplay}
          </div>
        </div>

        {/* Secondary Metrics - Compact */}
        {secondaryMetrics.length > 0 && (
          <div className="grid grid-cols-2 gap-1.5 py-1.5 border-t">
            {secondaryMetrics.slice(0, 4).map(metric => {
              const value = run.metrics?.final?.[metric];
              const display = typeof value === 'number'
                ? value.toFixed(4)
                : 'N/A';

              return (
                <div key={metric}>
                  <div className="text-[10px] text-muted-foreground truncate" title={metric}>
                    {metric.replace(/_/g, ' ')}
                  </div>
                  <div className="text-xs font-semibold">{display}</div>
                </div>
              );
            })}
          </div>
        )}

        {/* Footer with Date and Button - Compact */}
        <div className="mt-auto pt-1.5 border-t space-y-1.5">
          {run.started_at && (
            <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <Calendar className="h-3 w-3" />
              <span>
                {new Date(run.started_at).toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                })}
              </span>
            </div>
          )}

          {onOpenDetail && (
            <Button
              variant="outline"
              size="sm"
              className="w-full h-7 text-xs"
              onClick={() => onOpenDetail(run.run_name)}
            >
              View Details
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
