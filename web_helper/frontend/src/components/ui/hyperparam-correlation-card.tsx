import React from "react";
/**
 * Hyperparameter Correlation Card
 *
 * Displays correlation analysis between hyperparameters and metrics
 * Uses statistical correlation (Pearson, Spearman, Point-Biserial, ANOVA)
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Badge } from './badge';
import { Button } from './button';
import { TrendingUp, TrendingDown, AlertCircle, Loader2, BarChart3 } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './tooltip';

interface CorrelationResult {
  hyperparam_name: string;
  metric_name: string;
  correlation: number;
  p_value: number;
  method: string;
  sample_size: number;
  hyperparam_type: string;
}

interface CorrelationAnalysisResponse {
  project: string;
  metric: string;
  correlations: CorrelationResult[];
  total_runs: number;
  min_sample_threshold: number;
}

interface HyperparamCorrelationCardProps {
  project: string;
  metric: string;
  useMax?: boolean;
  maxDisplay?: number;
  className?: string;
}

const METHOD_LABELS: Record<string, string> = {
  'pearson': 'Pearson',
  'spearman': 'Spearman',
  'point_biserial': 'Point-Biserial',
  'anova': 'ANOVA (η²)',
};

const TYPE_LABELS: Record<string, string> = {
  'numeric': 'Numeric',
  'categorical': 'Categorical',
  'boolean': 'Boolean',
};

export function HyperparamCorrelationCard({
  project,
  metric,
  useMax = true,
  maxDisplay = 5,
  className,
}: HyperparamCorrelationCardProps) {
  const [data, setData] = useState<CorrelationAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchCorrelations = async () => {
      if (!project || !metric) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const response = await fetch(
          `/api/correlations/${project}?metric=${encodeURIComponent(metric)}&use_max=${useMax}`
        );

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch correlations' }));
          // Silently handle "insufficient data" errors (expected when < 10 runs)
          if (errorData.detail?.includes('Insufficient data')) {
            setError(errorData.detail);
            setLoading(false);
            return;
          }
          throw new Error(errorData.detail || 'Failed to fetch correlations');
        }

        const result: CorrelationAnalysisResponse = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchCorrelations();
  }, [project, metric, useMax]);

  if (loading) {
    return (
      <Card variant="compact" className={className}>
        <CardHeader variant="compact">
          <CardTitle size="sm" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Best Hyperparameters
          </CardTitle>
        </CardHeader>
        <CardContent variant="compact" className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="compact" className={className}>
        <CardHeader variant="compact">
          <CardTitle size="sm" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Best Hyperparameters
          </CardTitle>
        </CardHeader>
        <CardContent variant="compact">
          <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data || data.correlations.length === 0) {
    return (
      <Card variant="compact" className={className}>
        <CardHeader variant="compact">
          <CardTitle size="sm" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Best Hyperparameters
          </CardTitle>
        </CardHeader>
        <CardContent variant="compact">
          <div className="text-sm text-muted-foreground py-4 text-center">
            No significant correlations found. Need at least {data?.min_sample_threshold || 10} runs.
          </div>
        </CardContent>
      </Card>
    );
  }

  const topCorrelations = data.correlations.slice(0, maxDisplay);

  // Get correlation strength label
  const getStrengthLabel = (corr: number) => {
    const absCorr = Math.abs(corr);
    if (absCorr >= 0.7) return 'Strong';
    if (absCorr >= 0.4) return 'Moderate';
    return 'Weak';
  };

  // Get correlation color
  const getCorrelationColor = (corr: number) => {
    const absCorr = Math.abs(corr);
    if (absCorr >= 0.7) return corr > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
    if (absCorr >= 0.4) return corr > 0 ? 'text-blue-600 dark:text-blue-400' : 'text-orange-600 dark:text-orange-400';
    return 'text-muted-foreground';
  };

  return (
    <Card variant="compact" className={className}>
      <CardHeader variant="compact">
        <div className="flex items-center justify-between">
          <CardTitle size="sm" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Best Hyperparameters
          </CardTitle>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge variant="outline" className="text-xs">
                  {data.total_runs} runs
                </Badge>
              </TooltipTrigger>
              <TooltipContent>
                <p>Analysis based on {data.total_runs} experiment runs</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Top hyperparameters correlated with {metric}
        </p>
      </CardHeader>
      <CardContent variant="compact" className="space-y-2">
        {topCorrelations.map((item, idx) => {
          const absCorr = Math.abs(item.correlation);
          const isPositive = item.correlation > 0;
          const strength = getStrengthLabel(item.correlation);
          const colorClass = getCorrelationColor(item.correlation);

          return (
            <div
              key={`${item.hyperparam_name}-${idx}`}
              className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50 transition-colors duration-200"
            >
              {/* Rank */}
              <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-semibold flex items-center justify-center">
                {idx + 1}
              </div>

              {/* Hyperparam Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium truncate" title={item.hyperparam_name}>
                    {item.hyperparam_name.replace(/_/g, ' ')}
                  </span>
                  <Badge variant="outline" className="text-[10px] px-1 py-0">
                    {TYPE_LABELS[item.hyperparam_type] || item.hyperparam_type}
                  </Badge>
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-xs text-muted-foreground">
                    {METHOD_LABELS[item.method] || item.method}
                  </span>
                  {item.p_value < 0.05 && (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Badge variant="default" className="text-[10px] px-1 py-0">
                            p &lt; 0.05
                          </Badge>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Statistically significant (p = {item.p_value.toFixed(4)})</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                </div>
              </div>

              {/* Correlation Value */}
              <div className="flex-shrink-0 text-right">
                <div className={`flex items-center gap-1 ${colorClass}`}>
                  {isPositive ? (
                    <TrendingUp className="h-3.5 w-3.5" />
                  ) : (
                    <TrendingDown className="h-3.5 w-3.5" />
                  )}
                  <span className="text-sm font-bold">
                    {absCorr.toFixed(3)}
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {strength}
                </div>
              </div>
            </div>
          );
        })}

        {/* Show More Button */}
        {data.correlations.length > maxDisplay && (
          <div className="pt-2 border-t border-border/50">
            <Button
              variant="ghost"
              size="sm"
              className="w-full text-xs"
              onClick={() => {
                // TODO: Open detailed view or modal
                console.log('Show all correlations');
              }}
            >
              Show {data.correlations.length - maxDisplay} more correlations
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
