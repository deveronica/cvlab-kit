import React from "react";
/**
 * Advanced View - Statistical Analysis & Smart Features
 *
 * Dedicated view for advanced statistical analysis and AI-powered insights.
 * Separated from ChartsView which focuses on exploratory visualization.
 *
 * Features:
 * - Correlation Heatmap (statistical)
 * - Parallel Coordinates (multi-dimensional)
 * - Outlier Detection (Phase 3.1)
 * - Trend Analysis (Phase 3.2)
 * - Best Run Recommendation (Phase 3.3)
 */

import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/card';
import { adaptExperiments } from '../../lib/dataAdapter';
import { flattenObject } from '../../lib/table-columns';
import ParallelCoordinatesChart from '../charts/ParallelCoordinatesChart';
import { spearmanr } from '../../lib/statistics';
import { AlertCircle, Sparkles } from 'lucide-react';
import { Badge } from '../ui/badge';
import { OutlierDetectionCard } from '../cards/OutlierDetectionCard';
import { TrendAnalysisCard } from '../cards/TrendAnalysisCard';
import { BestRunRecommendationCard } from '../cards/BestRunRecommendationCard';

interface AdvancedViewProps {
  experimentsData: any[];
  selectedExperiments: string[];
  hyperparamColumns: string[];
  metricColumns: string[];
  activeProject: string | null;
  onHighlightRun?: (runName: string) => void;
}

function hparamMetricCorr(runs: any[], metric: string, hyperparamColumns: string[]) {
  const rho: { [key: string]: number } = {};
  const y = runs.map(r => {
    const m = flattenObject(r.metrics?.final || {});
    const val = m[metric];
    return typeof val === 'number' ? val : NaN;
  });

  for (const k of hyperparamColumns) {
    const x = runs.map(r => {
      const p = flattenObject(r.hyperparameters || {});
      const val = p[k];
      return typeof val === 'number' ? val : NaN;
    });
    rho[k] = spearmanr(x, y).correlation;
  }
  return rho;
}

export function AdvancedView({
  experimentsData,
  selectedExperiments,
  hyperparamColumns,
  metricColumns,
  activeProject,
  onHighlightRun,
}: AdvancedViewProps) {
  const adaptedData = useMemo(() => {
    if (selectedExperiments.length === 0) {
      return adaptExperiments(experimentsData, activeProject || '');
    }
    const selectedExps = experimentsData.filter(exp =>
      selectedExperiments.includes(exp.run_name)
    );
    return adaptExperiments(selectedExps, activeProject || '');
  }, [experimentsData, selectedExperiments, activeProject]);

  const _correlationData = useMemo(() => {
    return metricColumns.map((metric, i) => {
      const correlations = hparamMetricCorr(adaptedData, metric, hyperparamColumns);
      return hyperparamColumns.map((param, j) => {
        return [j, i, correlations[param]];
      });
    }).flat();
  }, [adaptedData, metricColumns, hyperparamColumns]);

  const parallelCoordsData = useMemo(() => {
    return adaptedData.map(exp => {
      const flatParams = flattenObject(exp.hyperparameters || {});
      const flatMetrics = flattenObject(exp.metrics?.final || {});
      const row = { ...flatParams, ...flatMetrics };
      return Object.values(row);
    });
  }, [adaptedData]);

  const parallelCoordsDimensions = useMemo(() => {
    return [
      ...hyperparamColumns.map(p => ({ key: p, name: p })),
      ...metricColumns.map(m => ({ key: m, name: m }))
    ];
  }, [hyperparamColumns, metricColumns]);

  const parallelCoordsRunNames = useMemo(() => {
    return adaptedData.map(exp => exp.run_name);
  }, [adaptedData]);

  // Check if we have data
  if (adaptedData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Advanced Analysis</CardTitle>
          <CardDescription>No data available for analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-muted-foreground">
            <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="font-medium">No experiments found</p>
            <p className="text-sm mt-2">Run some experiments to see advanced analysis here.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-3 pb-4">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Advanced Analysis</h2>
        <p className="text-muted-foreground mt-1">
          {selectedExperiments.length === 0
            ? `Statistical analysis of all ${experimentsData.length} runs`
            : `Statistical analysis of ${selectedExperiments.length} selected runs`}
        </p>
        <p className="text-xs text-muted-foreground mt-0.5">
          Deep statistical insights and AI-powered recommendations.
        </p>
      </div>


      {/* Parallel Coordinates Chart */}
      {(hyperparamColumns.length > 0 || metricColumns.length > 0) && parallelCoordsData.length > 0 && (
        <Card variant="compact">
          <CardHeader variant="compact">
            <CardTitle size="base">Parallel Coordinates</CardTitle>
            <CardDescription>
              Multi-dimensional view showing relationships across all hyperparameters and metrics.
              <span className="text-muted-foreground"> Hover over lines to see exact values.</span>
              <span className="text-green-600 dark:text-green-400 font-medium"> Green axes</span> represent target metrics.
            </CardDescription>
          </CardHeader>
          <CardContent variant="compact">
            <ParallelCoordinatesChart
              data={parallelCoordsData}
              dimensions={parallelCoordsDimensions}
              hyperparamCount={hyperparamColumns.length}
              runNames={parallelCoordsRunNames}
              onRunSelect={onHighlightRun}
            />
          </CardContent>
        </Card>
      )}

      {/* Analysis Cards Grid - 2 columns on large screens */}
      {activeProject && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {/* Outlier Detection - Phase 3.1 IMPLEMENTED */}
          <OutlierDetectionCard
            project={activeProject}
            runNames={selectedExperiments.length > 0 ? selectedExperiments : undefined}
            hyperparamColumns={hyperparamColumns}
            metricColumns={metricColumns}
            onOutlierSelect={(runName) => {
              if (onHighlightRun) {
                onHighlightRun(runName);
              }
            }}
          />

          {/* Trend Analysis - Phase 3.2 IMPLEMENTED */}
          <TrendAnalysisCard
            project={activeProject}
            metrics={metricColumns}
          />
        </div>
      )}

      {/* Best Run Recommendation - Phase 3.3 IMPLEMENTED - Full width for prominence */}
      {activeProject && (
        <BestRunRecommendationCard
          project={activeProject}
          availableMetrics={metricColumns}
          onRunSelect={(runName) => {
            if (onHighlightRun) {
              onHighlightRun(runName);
            }
          }}
        />
      )}

      {/* All Features Complete Banner */}
      <Card className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 border-green-200 dark:border-green-800">
        <CardContent className="p-2">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-green-600 dark:text-green-400 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-sm text-green-900 dark:text-green-100">
                All Advanced Features Complete! ✅
              </h3>
              <p className="text-xs text-green-700 dark:text-green-300 mt-0.5">
                Intelligent outlier detection, automatic trend discovery, and Pareto-optimal
                hyperparameter recommendations are now available!
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
