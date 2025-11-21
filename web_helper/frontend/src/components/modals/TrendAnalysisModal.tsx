import React from "react";
import { BarChart3, TrendingUp, TrendingDown, Minus, X } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type {
  TrendAnalysisResponse,
  TrendDirection,
  TrendStrength,
} from '@/lib/api/trends';

interface TrendAnalysisModalProps {
  open: boolean;
  onClose: () => void;
  data: TrendAnalysisResponse;
}

export function TrendAnalysisModal({
  open,
  onClose,
  data,
}: TrendAnalysisModalProps) {
  const getTrendIcon = (direction: TrendDirection, size: string = 'h-4 w-4') => {
    switch (direction) {
      case 'improving':
        return <TrendingUp className={`${size} text-green-600 dark:text-green-400`} />;
      case 'degrading':
        return <TrendingDown className={`${size} text-red-600 dark:text-red-400`} />;
      default:
        return <Minus className={`${size} text-gray-600 dark:text-gray-400`} />;
    }
  };

  const getStrengthColor = (strength: TrendStrength) => {
    switch (strength) {
      case 'strong':
        return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200';
      case 'moderate':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'weak':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
      default:
        return 'bg-gray-50 text-gray-600 dark:bg-gray-900 dark:text-gray-400';
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent hideCloseButton className="max-w-4xl h-[90vh] flex flex-col">
        <div className="flex flex-col space-y-1.5 text-left">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-blue-500 flex-shrink-0" />
              <DialogTitle>Trend Analysis - Detailed Results</DialogTitle>
            </div>
            <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close">
              <X className="h-4 w-4" />
            </Button>
          </div>
          <DialogDescription>
            Performance trends and statistical analysis across {data.analyzed_metrics} metrics
          </DialogDescription>
        </div>

        <div className="flex-1 overflow-y-auto space-y-3">
          {Object.entries(data.trends).map(([metric, trend]) => (
            <div key={metric} className="p-4 border rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getTrendIcon(trend.direction, 'h-5 w-5')}
                  <span className="text-base font-medium font-mono">
                    {metric}
                  </span>
                  <Badge
                    variant="outline"
                    className={getStrengthColor(trend.strength)}
                  >
                    {trend.strength}
                  </Badge>
                </div>
                {trend.is_significant && (
                  <Badge variant="default">
                    p &lt; 0.05
                  </Badge>
                )}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-2 border rounded">
                  <div className="text-xs text-muted-foreground">R² Score</div>
                  <div className="text-lg font-mono font-bold">
                    {trend.r_squared.toFixed(3)}
                  </div>
                </div>
                <div className="p-2 border rounded">
                  <div className="text-xs text-muted-foreground">Slope</div>
                  <div className="text-lg font-mono font-bold">
                    {trend.slope.toFixed(4)}
                  </div>
                </div>
                <div className="p-2 border rounded">
                  <div className="text-xs text-muted-foreground">Improvement Rate</div>
                  <div className="text-lg font-mono font-bold">
                    {trend.improvement_rate > 0 ? '+' : ''}
                    {trend.improvement_rate.toFixed(2)}%
                  </div>
                </div>
                <div className="p-2 border rounded">
                  <div className="text-xs text-muted-foreground">Data Points</div>
                  <div className="text-lg font-mono font-bold">
                    {trend.data_points}
                  </div>
                </div>
              </div>

              {trend.prediction && (
                <div className="p-3 border rounded-lg bg-muted/30">
                  <div className="text-sm font-medium mb-2">Prediction (Next Run)</div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Predicted Value:</span>
                      <span className="text-base font-mono font-bold">
                        {trend.prediction.value.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">95% Confidence Interval:</span>
                      <span className="text-sm font-mono">
                        [{trend.prediction.confidence_interval[0].toFixed(4)}, {trend.prediction.confidence_interval[1].toFixed(4)}]
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
