import React from "react";
/**
 * Hyperparameter Insights
 *
 * Shows the most successful hyperparameter configurations
 */

import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Sparkles, TrendingUp, TrendingDown, Info } from 'lucide-react';
import type { Run } from '../../lib/types';
import { flattenObject } from '../../lib/table-columns';
import { Badge } from './badge';
import { spearmanr } from '../../lib/statistics';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './tooltip';

export interface HyperparameterInsightsProps {
  runs: Run[];
  keyMetric: string;
  metricDirection: 'maximize' | 'minimize';
  hyperparamColumns?: string[];
  maxInsights?: number;
  className?: string;
}

interface HyperparamInsight {
  param: string;
  correlation: number;
  direction: 'positive' | 'negative';
}

export function HyperparameterInsights({
  runs,
  keyMetric,
  metricDirection,
  hyperparamColumns = [],
  maxInsights = 5,
  className,
}: HyperparameterInsightsProps) {
  const insights = useMemo(() => {
    // Filter completed runs with the key metric
    const validRuns = runs.filter(run => {
      const metricValue = run.metrics?.final?.[keyMetric];
      return run.status === 'completed' && typeof metricValue === 'number';
    });

    if (validRuns.length < 3 || hyperparamColumns.length === 0) {
      return [];
    }

    // Get metric values
    const metricValues = validRuns.map(run => {
      const val = run.metrics?.final?.[keyMetric];
      return typeof val === 'number' ? val : NaN;
    });

    const allInsights: HyperparamInsight[] = [];

    // Calculate correlation for each numeric hyperparameter
    for (const param of hyperparamColumns) {
      // Filter out metadata fields (common patterns)
      const paramLower = param.toLowerCase();
      if (
        paramLower.includes('date') ||
        paramLower.includes('time') ||
        paramLower.includes('author') ||
        paramLower.includes('user') ||
        paramLower.includes('name') ||
        paramLower.includes('path') ||
        paramLower === 'project' ||
        paramLower === 'run_name'
      ) {
        continue;
      }

      // Get parameter values
      const paramValues = validRuns.map(run => {
        const flatParams = flattenObject(run.hyperparameters || {});
        const val = flatParams[param];
        return typeof val === 'number' ? val : NaN;
      });

      // Check if we have enough numeric values
      const numericCount = paramValues.filter(v => !isNaN(v)).length;
      if (numericCount < 3) {
        continue;
      }

      // Skip hyperparameters with only one unique value (no variance)
      const uniqueValues = new Set(paramValues.filter(v => !isNaN(v)));
      if (uniqueValues.size === 1) {
        continue;
      }

      // Calculate Spearman correlation
      const result = spearmanr(paramValues, metricValues);

      if (!isNaN(result.correlation) && Math.abs(result.correlation) > 0.1) {
        allInsights.push({
          param,
          correlation: result.correlation,
          direction: result.correlation > 0 ? 'positive' : 'negative',
        });
      }
    }

    // Sort by absolute correlation strength
    return allInsights
      .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
      .slice(0, maxInsights);
  }, [runs, keyMetric, hyperparamColumns, maxInsights]);

  const getCorrelationStrength = (corr: number): { text: string; color: string } => {
    const abs = Math.abs(corr);
    if (abs >= 0.7) return { text: 'Strong', color: 'text-green-600 dark:text-green-400' };
    if (abs >= 0.4) return { text: 'Moderate', color: 'text-yellow-600 dark:text-yellow-400' };
    return { text: 'Weak', color: 'text-muted-foreground' };
  };

  if (insights.length === 0) {
    return (
      <Card variant="compact" className={className}>
        <CardHeader variant="compact">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <CardTitle size="sm">Best Hyperparameters</CardTitle>
          </div>
        </CardHeader>
        <CardContent variant="compact">
          <div className="text-center py-6 text-muted-foreground text-sm">
            No completed runs available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="compact" className={className}>
      <CardHeader variant="compact">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <CardTitle size="sm">Best Hyperparameters{keyMetric ? ` for ${keyMetric.replace(/_/g, ' ')}` : ''}</CardTitle>
            <Badge variant="secondary" className="text-[10px] h-4 px-1">
              ✨ AI
            </Badge>
          </div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="text-muted-foreground hover:text-foreground transition-colors duration-200">
                  <Info className="h-3.5 w-3.5" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-xs">
                <div className="space-y-1.5">
                  <p className="font-medium text-xs">Correlation Analysis</p>
                  <p className="text-xs text-muted-foreground">
                    Uses Spearman correlation to measure the strength of relationship between hyperparameters and {keyMetric || 'metric'}.
                    Positive correlation means higher values lead to {metricDirection === 'maximize' ? 'better' : 'worse'} performance.
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    <strong>Use this to:</strong> Identify which hyperparameters have the strongest impact on your target metric.
                  </p>
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </CardHeader>
      <CardContent variant="compact">
        <div className="space-y-1.5">
          {insights.map((insight, index) => {
            const strength = getCorrelationStrength(insight.correlation);
            return (
              <div
                key={insight.param}
                className="flex items-center justify-between p-1.5 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors duration-200"
              >
                <div className="flex items-center gap-1.5 flex-1 min-w-0">
                  <div className="flex-shrink-0 w-4 h-4 rounded-full bg-primary/10 flex items-center justify-center">
                    <span className="text-[10px] font-semibold text-primary">{index + 1}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="font-medium text-xs">{insight.param.replace(/_/g, ' ')}</span>
                      <Badge variant="outline" className={`text-[9px] h-3.5 px-1 ${strength.color}`}>
                        {strength.text}
                      </Badge>
                    </div>
                    <div className="text-[10px] text-muted-foreground">
                      {insight.direction === 'positive' ? 'Higher is better' : 'Lower is better'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-0.5 flex-shrink-0">
                  {insight.direction === 'positive' ? (
                    <TrendingUp className="h-3 w-3 text-green-600 dark:text-green-400" />
                  ) : (
                    <TrendingDown className="h-3 w-3 text-red-600 dark:text-red-400" />
                  )}
                  <span className="font-mono text-xs font-semibold">
                    {insight.correlation.toFixed(3)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
        <div className="mt-2 pt-1.5 border-t text-[10px] text-muted-foreground">
          Showing hyperparameters with strongest correlation to {keyMetric ? keyMetric.replace(/_/g, ' ') : 'metric'}
        </div>
      </CardContent>
    </Card>
  );
}
