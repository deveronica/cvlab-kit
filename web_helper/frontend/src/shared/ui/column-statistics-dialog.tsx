import React from "react";
/**
 * Column Statistics Dialog
 *
 * Shows simple statistics for a metric column across all visible runs in the table.
 */

import { useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from './dialog';
import { Card, CardContent } from './card';
import { BarChart3 } from 'lucide-react';

interface ColumnStatisticsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  metricName: string;
  values: Array<{ runName: string; value: number | null }>;
}

export function ColumnStatisticsDialog({
  isOpen,
  onClose,
  metricName,
  values,
}: ColumnStatisticsDialogProps) {
  const stats = useMemo(() => {
    // Filter out null/undefined values
    const validValues = values
      .filter(v => v.value !== null && v.value !== undefined && !isNaN(v.value))
      .map(v => ({ runName: v.runName, value: v.value as number }));

    if (validValues.length === 0) {
      return null;
    }

    const nums = validValues.map(v => v.value).sort((a, b) => a - b);
    const count = nums.length;
    const sum = nums.reduce((acc, val) => acc + val, 0);
    const mean = sum / count;

    const variance = nums.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / count;
    const std = Math.sqrt(variance);

    const min = nums[0];
    const max = nums[nums.length - 1];
    const median = count % 2 === 0
      ? (nums[count / 2 - 1] + nums[count / 2]) / 2
      : nums[Math.floor(count / 2)];

    // Find best and worst runs
    const bestRun = validValues.find(v => v.value === max);
    const worstRun = validValues.find(v => v.value === min);

    return {
      count,
      min,
      max,
      mean,
      std,
      median,
      bestRun: bestRun?.runName || 'N/A',
      worstRun: worstRun?.runName || 'N/A',
    };
  }, [values]);

  const formatValue = (value: number) => {
    if (Math.abs(value) < 0.01 || Math.abs(value) > 1000) {
      return value.toExponential(3);
    }
    return value.toFixed(4);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Column Statistics
          </DialogTitle>
          <DialogDescription>
            Statistics for <span className="font-mono font-medium">{metricName}</span> across all visible runs
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {!stats ? (
            <div className="text-center text-muted-foreground py-8 text-sm">
              No valid data available for this metric
            </div>
          ) : (
            <>
              {/* Summary Grid */}
              <div className="grid grid-cols-2 gap-3">
                <Card variant="compact">
                  <CardContent variant="compact" className="pt-3">
                    <div className="text-xs text-muted-foreground mb-1">Count</div>
                    <div className="text-2xl font-bold">{stats.count}</div>
                  </CardContent>
                </Card>

                <Card variant="compact">
                  <CardContent variant="compact" className="pt-3">
                    <div className="text-xs text-muted-foreground mb-1">Mean</div>
                    <div className="text-2xl font-bold">{formatValue(stats.mean)}</div>
                    <div className="text-xs text-muted-foreground">± {formatValue(stats.std)}</div>
                  </CardContent>
                </Card>

                <Card variant="compact">
                  <CardContent variant="compact" className="pt-3">
                    <div className="text-xs text-muted-foreground mb-1">Min</div>
                    <div className="text-lg font-semibold">{formatValue(stats.min)}</div>
                    <div className="text-xs text-muted-foreground truncate" title={stats.worstRun}>
                      {stats.worstRun}
                    </div>
                  </CardContent>
                </Card>

                <Card variant="compact">
                  <CardContent variant="compact" className="pt-3">
                    <div className="text-xs text-muted-foreground mb-1">Max</div>
                    <div className="text-lg font-semibold">{formatValue(stats.max)}</div>
                    <div className="text-xs text-muted-foreground truncate" title={stats.bestRun}>
                      {stats.bestRun}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Additional Stats */}
              <Card>
                <CardContent className="pt-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Median:</span>
                      <span className="font-mono font-medium">{formatValue(stats.median)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Range:</span>
                      <span className="font-mono font-medium">{formatValue(stats.max - stats.min)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
