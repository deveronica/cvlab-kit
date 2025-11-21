import React from "react";
/**
 * TrendAnalysisCard - Displays performance trend analysis and predictions.
 */

import { useEffect, useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Info,
  BarChart3,
} from 'lucide-react';
import { TrendAnalysisModal } from '@/components/modals/TrendAnalysisModal';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  getTrendSummary,
  analyzeTrends,
  type TrendSummaryResponse,
  type TrendAnalysisResponse,
  type TrendDirection,
  type TrendStrength,
} from '@/lib/api/trends';
import { devError } from '@/lib/dev-utils';

interface TrendAnalysisCardProps {
  project: string;
  metrics?: string[];
}

export function TrendAnalysisCard({ project, metrics }: TrendAnalysisCardProps) {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<TrendSummaryResponse | null>(null);
  const [detailedResults, setDetailedResults] =
    useState<TrendAnalysisResponse | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    loadTrendSummary();
  }, [project]);

  const loadTrendSummary = async () => {
    if (!project) return;

    setLoading(true);
    try {
      const result = await getTrendSummary(project);
      setSummary(result);
    } catch (error) {
      devError('Failed to load trend summary:', error);
      setSummary(null);
    } finally {
      setLoading(false);
    }
  };

  const loadDetailedAnalysis = async () => {
    if (!project) return;

    setLoading(true);
    try {
      const result = await analyzeTrends({
        project,
        metrics,
      });
      setDetailedResults(result);
      setIsModalOpen(true);
    } catch (error) {
      devError('Failed to load detailed analysis:', error);
    } finally {
      setLoading(false);
    }
  };

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

  const getHealthScoreColor = (score: number) => {
    if (score > 50) return 'text-green-600 dark:text-green-400';
    if (score > 0) return 'text-yellow-600 dark:text-yellow-400';
    if (score > -50) return 'text-orange-600 dark:text-orange-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <Card variant="compact">
      <CardHeader variant="compact">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-blue-500 flex-shrink-0" />
            <div>
              <CardTitle size="sm">Trend Analysis</CardTitle>
              <CardDescription className="mt-0.5 text-xs">
                Performance trends and predictions over time
              </CardDescription>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={loadTrendSummary}
            disabled={loading}
            className="h-6 px-2"
          >
            <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>

      <CardContent variant="compact" className="space-y-3">
        {/* Summary View */}
        {summary && !summary.message && (
          <div className="space-y-3">
            {/* Health Score */}
            <div className="p-3 border rounded-lg bg-muted/30">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Performance Health</span>
                <div className="flex items-center gap-2">
                  {summary.health_score > 0 ? (
                    <TrendingUp className="h-4 w-4 text-green-600 dark:text-green-400" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-red-600 dark:text-red-400" />
                  )}
                  <span
                    className={`text-xl font-bold ${getHealthScoreColor(summary.health_score)}`}
                  >
                    {summary.health_score > 0 ? '+' : ''}
                    {summary.health_score.toFixed(0)}
                  </span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                Based on {summary.analyzed_metrics} metrics across{' '}
                {summary.total_runs} runs
              </p>
            </div>

            {/* Trend Categories */}
            <div className="grid grid-cols-3 gap-2">
              <div className="p-2 border rounded-lg border-green-200 dark:border-green-900">
                <div className="flex items-center gap-2 mb-0.5">
                  <TrendingUp className="h-3 w-3 text-green-600 dark:text-green-400" />
                  <span className="text-xs font-medium">Improving</span>
                </div>
                <div className="text-xl font-bold text-green-600 dark:text-green-400">
                  {summary.summary.improving.length}
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5">
                  metrics trending up
                </div>
              </div>

              <div className="p-2 border rounded-lg">
                <div className="flex items-center gap-2 mb-0.5">
                  <Minus className="h-3 w-3 text-gray-600 dark:text-gray-400" />
                  <span className="text-xs font-medium">Stable</span>
                </div>
                <div className="text-xl font-bold">
                  {summary.summary.stable.length}
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5">
                  metrics steady
                </div>
              </div>

              <div className="p-2 border rounded-lg border-red-200 dark:border-red-900">
                <div className="flex items-center gap-2 mb-0.5">
                  <TrendingDown className="h-3 w-3 text-red-600 dark:text-red-400" />
                  <span className="text-xs font-medium">Degrading</span>
                </div>
                <div className="text-xl font-bold text-red-600 dark:text-red-400">
                  {summary.summary.degrading.length}
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5">
                  metrics trending down
                </div>
              </div>
            </div>

            {/* Significant Trends */}
            {summary.summary.significant_trends.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Info className="h-4 w-4 text-muted-foreground" />
                  <h4 className="text-sm font-medium">Significant Trends</h4>
                </div>
                <div className="space-y-1">
                  {summary.summary.significant_trends.slice(0, 5).map((trend) => (
                    <div
                      key={trend.metric}
                      className="flex items-center justify-between p-2 border rounded hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        {getTrendIcon(trend.direction)}
                        <span className="text-sm font-mono">{trend.metric}</span>
                        <Badge
                          variant="outline"
                          className={getStrengthColor(trend.strength)}
                        >
                          {trend.strength}
                        </Badge>
                      </div>
                      <span className="text-sm font-medium">
                        {trend.improvement_rate > 0 ? '+' : ''}
                        {trend.improvement_rate.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Load Detailed Analysis Button */}
            <Button
              onClick={loadDetailedAnalysis}
              disabled={loading || !summary}
              className="w-full"
              variant="outline"
            >
              {loading ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <BarChart3 className="h-4 w-4 mr-2" />
                  View Detailed Analysis
                </>
              )}
            </Button>
          </div>
        )}

        {/* Insufficient Data Message */}
        {summary && summary.message && (
          <div className="text-center py-8 text-muted-foreground">
            <Info className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">{summary.message}</p>
          </div>
        )}

        {/* No Data State */}
        {!loading && !summary && (
          <div className="text-center py-8 text-muted-foreground">
            <BarChart3 className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No trend data available</p>
          </div>
        )}
      </CardContent>

      {/* Modal */}
      {detailedResults && (
        <TrendAnalysisModal
          open={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          data={detailedResults}
        />
      )}
    </Card>
  );
}
