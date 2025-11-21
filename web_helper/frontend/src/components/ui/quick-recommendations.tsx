import React from "react";
/**
 * Quick Recommendations
 *
 * Summary version of Best Run Recommendations from Advanced tab
 * Shows top 1-2 Pareto optimal runs
 */

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Target, Trophy, ArrowRight, Sparkles, Info } from 'lucide-react';
import { Badge } from './badge';
import { Button } from './button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './tooltip';
import {
  getRecommendationSummary,
  type RecommendationSummaryResponse,
} from '@/lib/api/recommendations';
import { devError } from '@/lib/dev-utils';

export interface QuickRecommendationsProps {
  project: string;
  className?: string;
  onViewMore?: () => void;
  onRunClick?: (runName: string) => void;
}

export function QuickRecommendations({
  project,
  className,
  onViewMore,
  onRunClick,
}: QuickRecommendationsProps) {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<RecommendationSummaryResponse | null>(null);

  useEffect(() => {
    loadRecommendations();
  }, [project]);

  const loadRecommendations = async () => {
    if (!project) return;

    setLoading(true);
    try {
      // Fetch only top 2 recommendations for overview
      const result = await getRecommendationSummary(project, 2);
      setSummary(result);
    } catch (error) {
      devError('Failed to load recommendations:', error);
      setSummary(null);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card variant="compact" className={className}>
        <CardHeader variant="compact">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <CardTitle size="sm">AI Recommendations</CardTitle>
          </div>
        </CardHeader>
        <CardContent variant="compact">
          <div className="flex items-center justify-center py-6">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!summary || summary.message || summary.top_recommendations.length === 0) {
    return (
      <Card variant="compact" className={className}>
        <CardHeader variant="compact">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <CardTitle size="sm">AI Recommendations</CardTitle>
          </div>
        </CardHeader>
        <CardContent variant="compact">
          <div className="text-center py-6 text-muted-foreground text-sm">
            {summary?.message || 'No recommendations available'}
          </div>
        </CardContent>
      </Card>
    );
  }

  const topRec = summary.top_recommendations[0];
  const hasSecond = summary.top_recommendations.length > 1;

  return (
    <Card variant="compact" className={className}>
      <CardHeader variant="compact">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <CardTitle size="sm">AI Recommendations</CardTitle>
            <Badge variant="secondary" className="text-[10px] h-4 px-1">
              ✨ AI
            </Badge>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Target className="h-3 w-3" />
              <span>{summary.pareto_optimal_count} Pareto-optimal</span>
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
                    <p className="font-medium text-xs">Pareto-Optimal Recommendations</p>
                    <p className="text-xs text-muted-foreground">
                      AI automatically identifies runs that are not dominated by any other run across multiple objectives.
                      A run is Pareto-optimal if improving one metric would require degrading another.
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      <strong>Use this to:</strong> Find the best trade-offs when optimizing multiple conflicting metrics simultaneously.
                    </p>
                    <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                      Click "View All in Advanced Tab" for full analysis →
                    </p>
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </CardHeader>

      <CardContent variant="compact" className="space-y-2">
        {/* Top Recommendation */}
        <div
          className="p-2 border-2 border-green-300 dark:border-green-800 bg-green-50/30 dark:bg-green-950/20 rounded-lg cursor-pointer hover:shadow-md transition-all duration-200"
          onClick={() => onRunClick?.(topRec.run_name)}
        >
          <div className="flex items-start justify-between gap-1.5 mb-1.5">
            <div className="flex items-center gap-1.5">
              <Trophy className="h-3.5 w-3.5 text-yellow-600 dark:text-yellow-500 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="font-mono text-xs truncate" title={topRec.run_name}>
                  {topRec.run_name}
                </div>
              </div>
            </div>
            <Badge variant="default" className="bg-green-600 text-white flex-shrink-0 text-[10px] h-5 px-1.5">
              #1 Pareto
            </Badge>
          </div>

          {/* Objective values */}
          <div className="grid grid-cols-2 gap-1.5 text-[10px]">
            {Object.entries(topRec.objective_values)
              .slice(0, 4)
              .map(([obj, value]) => (
                <div key={obj} className="flex justify-between">
                  <span className="text-muted-foreground truncate">{obj}:</span>
                  <span className="font-mono ml-1 font-medium">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </span>
                </div>
              ))}
          </div>

          {topRec.dominates_count > 0 && (
            <div className="mt-1.5 text-[10px] text-muted-foreground">
              Dominates {topRec.dominates_count} other run{topRec.dominates_count > 1 ? 's' : ''}
            </div>
          )}
        </div>

        {/* Second recommendation if available */}
        {hasSecond && summary.top_recommendations[1] && (
          <div
            className="p-1.5 border rounded-lg cursor-pointer hover:bg-muted/50 transition-colors duration-200"
            onClick={() => onRunClick?.(summary.top_recommendations[1].run_name)}
          >
            <div className="flex items-center justify-between gap-1.5 mb-1">
              <div className="font-mono text-[10px] truncate" title={summary.top_recommendations[1].run_name}>
                {summary.top_recommendations[1].run_name}
              </div>
              <Badge variant="outline" className="text-[10px] h-4 px-1 flex-shrink-0">
                Rank #2
              </Badge>
            </div>
            <div className="grid grid-cols-2 gap-1 text-[10px]">
              {Object.entries(summary.top_recommendations[1].objective_values)
                .slice(0, 2)
                .map(([obj, value]) => (
                  <div key={obj} className="flex justify-between">
                    <span className="text-muted-foreground truncate">{obj}:</span>
                    <span className="font-mono ml-1">
                      {typeof value === 'number' ? value.toFixed(4) : value}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* View More Button */}
        {onViewMore && (
          <Button
            variant="ghost"
            size="sm"
            className="w-full h-6 text-[10px]"
            onClick={onViewMore}
          >
            <span>View All in Advanced Tab</span>
            <ArrowRight className="h-3 w-3 ml-1" />
          </Button>
        )}

        {/* Objectives Summary */}
        <div className="pt-1.5 border-t">
          <div className="text-[10px] text-muted-foreground mb-1">Optimizing for:</div>
          <div className="flex flex-wrap gap-1">
            {summary.auto_selected_objectives.slice(0, 4).map((obj) => (
              <Badge key={obj} variant="outline" className="text-[10px] h-4 px-1">
                {obj}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
