import React from "react";
/**
 * OutlierDetectionCard - Displays outlier detection results and analysis.
 */

import { useEffect, useState } from 'react';
import {
  AlertCircle,
  TrendingUp,
  Info,
  RefreshCw,
  Settings,
} from 'lucide-react';
import { OutlierDetectionModal } from '@/components/modals/OutlierDetectionModal';
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import {
  detectOutliers,
  getOutlierSummary,
  OutlierMethod,
  OutlierSummaryResponse,
  OutlierDetectionResponse,
} from '@/lib/api/outliers';
import { devError } from '@/lib/dev-utils';

interface OutlierDetectionCardProps {
  project: string;
  runNames?: string[];
  hyperparamColumns?: string[];
  metricColumns?: string[];
  onOutlierSelect?: (runName: string) => void;
}

export function OutlierDetectionCard({
  project,
  runNames,
  hyperparamColumns,
  metricColumns,
  onOutlierSelect,
}: OutlierDetectionCardProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [method, setMethod] = useState<OutlierMethod>('iqr');
  const [threshold, setThreshold] = useState(1.5);
  const [summary, setSummary] = useState<OutlierSummaryResponse | null>(null);
  const [detailedResults, setDetailedResults] =
    useState<OutlierDetectionResponse | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const methodLabels: Record<OutlierMethod, string> = {
    iqr: 'IQR (Interquartile Range)',
    zscore: 'Z-Score',
    modified_zscore: 'Modified Z-Score (MAD)',
  };

  const methodDescriptions: Record<OutlierMethod, string> = {
    iqr: 'Robust method based on quartiles. Good for skewed distributions.',
    zscore: 'Parametric method assuming normal distribution.',
    modified_zscore:
      'Robust variant using median absolute deviation. Best for heavy tails.',
  };

  useEffect(() => {
    loadOutlierSummary();
  }, [project, method, threshold]);

  const loadOutlierSummary = async () => {
    if (!project) return;

    setLoading(true);
    setError(null);
    try {
      const result = await getOutlierSummary(project, method, threshold);
      setSummary(result);
    } catch (error) {
      devError('Failed to load outlier summary:', error);
      setError(error instanceof Error ? error.message : 'Failed to load outlier data');
      setSummary(null);
    } finally {
      setLoading(false);
    }
  };

  const loadDetailedResults = async () => {
    if (!project) return;

    setLoading(true);
    setError(null);
    try {
      const result = await detectOutliers({
        project,
        run_names: runNames,
        hyperparam_columns: hyperparamColumns,
        metric_columns: metricColumns,
        method,
        threshold,
      });
      setDetailedResults(result);
      setIsModalOpen(true);
    } catch (error) {
      devError('Failed to load detailed results:', error);
      setError(error instanceof Error ? error.message : 'Failed to load detailed analysis');
    } finally {
      setLoading(false);
    }
  };

  const getThresholdLabel = () => {
    switch (method) {
      case 'iqr':
        return `IQR Multiplier: ${threshold.toFixed(1)}`;
      case 'zscore':
        return `Sigma: ${threshold.toFixed(1)}σ`;
      case 'modified_zscore':
        return `MAD Threshold: ${threshold.toFixed(1)}`;
    }
  };

  return (
    <Card variant="compact">
      <CardHeader variant="compact">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-orange-500 flex-shrink-0" />
            <div>
              <CardTitle size="sm">Outlier Detection</CardTitle>
              <CardDescription className="mt-0.5 text-xs">
                Identify anomalous runs using statistical methods
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSettings(!showSettings)}
              className="h-6 px-2"
            >
              <Settings className="h-3 w-3" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={loadOutlierSummary}
              disabled={loading}
              className="h-6 px-2"
            >
              <RefreshCw
                className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`}
              />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent variant="compact" className="space-y-3">
        {/* Settings Panel */}
        {showSettings && (
          <div className="p-3 border rounded-lg bg-muted/30 space-y-3">
            <div className="space-y-2">
              <label className="text-sm font-medium">Detection Method</label>
              <Select
                value={method}
                onValueChange={(v) => setMethod(v as OutlierMethod)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(methodLabels).map(([value, label]) => (
                    <SelectItem key={value} value={value}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {methodDescriptions[method]}
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Threshold</label>
                <span className="text-xs font-mono">{getThresholdLabel()}</span>
              </div>
              <Slider
                value={[threshold]}
                onValueChange={([v]) => setThreshold(v)}
                min={method === 'iqr' ? 1.0 : 2.0}
                max={method === 'iqr' ? 3.0 : 5.0}
                step={0.1}
                className="py-2"
              />
              <p className="text-xs text-muted-foreground">
                {method === 'iqr'
                  ? 'Higher = fewer outliers detected'
                  : 'Higher = stricter outlier detection'}
              </p>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="p-3 border border-red-200 dark:border-red-900 rounded-lg bg-red-50 dark:bg-red-950/20">
            <div className="flex items-center gap-2 text-red-800 dark:text-red-200">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium">Error loading outlier data</p>
                <p className="text-xs mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Summary View */}
        {!error && summary && (
          <div className="space-y-3">
            {/* Stats Grid */}
            <div className="grid grid-cols-3 gap-2">
              <div className="p-2 border rounded-lg">
                <div className="text-xs text-muted-foreground">Total Runs</div>
                <div className="text-xl font-bold">{summary.total_runs}</div>
              </div>
              <div className="p-2 border rounded-lg border-orange-200 dark:border-orange-900">
                <div className="text-xs text-muted-foreground">Outliers</div>
                <div className="text-xl font-bold text-orange-600 dark:text-orange-400">
                  {summary.total_outlier_runs}
                </div>
              </div>
              <div className="p-2 border rounded-lg">
                <div className="text-xs text-muted-foreground">Percentage</div>
                <div className="text-xl font-bold">
                  {summary.outlier_percentage.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Top Outlier Runs */}
            {summary.top_outlier_runs.length > 0 ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  <h4 className="text-sm font-medium">
                    Most Anomalous Runs
                  </h4>
                </div>
                <div className="space-y-1">
                  {summary.top_outlier_runs.slice(0, 5).map((item) => (
                    <div
                      key={item.run_name}
                      className="flex items-center justify-between p-2 border rounded hover:bg-muted/50 cursor-pointer transition-colors"
                      onClick={() => onOutlierSelect?.(item.run_name)}
                    >
                      <span className="text-sm font-mono truncate flex-1">
                        {item.run_name}
                      </span>
                      <Badge variant="destructive" className="text-xs">
                        {item.outlier_count} columns
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            ) : summary.total_outlier_runs === 0 ? (
              <div className="p-3 border rounded-lg bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-900">
                <div className="flex items-center gap-2 text-green-800 dark:text-green-200">
                  <Info className="h-4 w-4 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium">No outliers detected</p>
                    <p className="text-xs mt-1">All runs appear normal with current threshold settings</p>
                  </div>
                </div>
              </div>
            ) : null}

            {/* Columns with Outliers */}
            {Object.keys(summary.columns_with_outliers).length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Info className="h-4 w-4 text-muted-foreground" />
                  <h4 className="text-sm font-medium">
                    Columns with Outliers
                  </h4>
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(summary.columns_with_outliers)
                    .slice(0, 10)
                    .map(([col, count]) => (
                      <Badge key={col} variant="outline" className="text-xs">
                        {col.replace('hyperparam.', 'HP: ').replace('metric.', 'M: ')}
                        <span className="ml-1 text-orange-600 dark:text-orange-400">
                          ({count})
                        </span>
                      </Badge>
                    ))}
                </div>
              </div>
            )}

            {/* Load Detailed Results Button */}
            <Button
              onClick={loadDetailedResults}
              disabled={loading}
              className="w-full"
              variant="outline"
            >
              {loading ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Loading...
                </>
              ) : (
                'View Detailed Analysis'
              )}
            </Button>
          </div>
        )}

        {/* No Data State */}
        {!loading && !error && !summary && (
          <div className="text-center py-8 text-muted-foreground">
            <Info className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm font-medium">No outlier data available</p>
            <p className="text-xs mt-1">Run outlier detection to analyze your experiments</p>
          </div>
        )}
      </CardContent>

      {/* Modal */}
      {detailedResults && (
        <OutlierDetectionModal
          open={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          data={detailedResults}
          onOutlierSelect={onOutlierSelect}
        />
      )}
    </Card>
  );
}
