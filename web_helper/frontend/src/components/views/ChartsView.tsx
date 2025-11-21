import React from "react";
import { useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/card';
import { adaptExperiments } from '../../lib/dataAdapter';
import { flattenObject } from '../../lib/table-columns';
import { DistributionChart as RichDistributionChart } from '../charts/DistributionChart';
import { ScatterPlotChart } from '../ui/scatter-plot-chart';
import { CustomizableChartCard } from '../charts/CustomizableChartCard';
import CorrelationHeatmap from '../charts/CorrelationHeatmap';
import { spearmanr } from '../../lib/statistics';
import type { SeriesConfig } from '../../lib/charts/types';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import type { Run } from '../../lib/types';
import { transformForDistribution } from '../../lib/chartDataTransformers';

/**
 * Compute correlation between hyperparameters and a metric
 */
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

interface ChartsViewProps {
  experimentsData: any[];
  selectedExperiments: string[];
  hyperparamColumns: string[];
  metricColumns: string[];
  activeProject: string | null;
}

/**
 * Check if a column contains numeric values
 * Returns true if all non-null values are numbers
 */
function isNumericColumn(runs: any[], field: string, isHyperparam: boolean): boolean {
  const values = runs
    .map(r => {
      const source = isHyperparam
        ? flattenObject(r.hyperparameters || {})
        : flattenObject(r.metrics?.final || {});
      return source[field];
    })
    .filter(v => v !== undefined && v !== null);

  if (values.length === 0) return false;

  // Check if all values are numbers
  return values.every(v => typeof v === 'number');
}

/**
 * Check if a hyperparameter has meaningful variance across runs
 * Returns true if the parameter varies (not all identical values)
 */
function hasMeaningfulVariance(runs: any[], param: string): boolean {
  const values = runs
    .map(r => {
      const p = flattenObject(r.hyperparameters || {});
      return p[param];
    })
    .filter(v => v !== undefined && v !== null);

  if (values.length === 0) return false;

  // For numeric values, check if all are identical
  if (typeof values[0] === 'number') {
    const firstValue = values[0];
    const hasVariance = values.some(v => Math.abs(v - firstValue) > 1e-10);
    return hasVariance;
  }

  // For string/categorical values, check if all are identical
  const firstValue = values[0];
  return values.some(v => v !== firstValue);
}

export function ChartsView({ experimentsData, selectedExperiments, hyperparamColumns, metricColumns, activeProject }: ChartsViewProps) {
  const [selectedHyperparam, setSelectedHyperparam] = useState<string | null>(null);

  const getSelectedExperimentData = () => {
    const selectedExps = experimentsData.filter(exp => selectedExperiments.includes(exp.run_name));
    return adaptExperiments(selectedExps, activeProject || '');
  };

  const adaptedData = selectedExperiments.length === 0
    ? adaptExperiments(experimentsData, activeProject || '')
    : getSelectedExperimentData();

  // Filter to only numeric columns for chart display
  const numericMetricColumns = useMemo(() => {
    return metricColumns.filter(metric => isNumericColumn(adaptedData, metric, false));
  }, [adaptedData, metricColumns]);

  const numericHyperparamColumns = useMemo(() => {
    return hyperparamColumns.filter(param => isNumericColumn(adaptedData, param, true));
  }, [adaptedData, hyperparamColumns]);

  // Filter hyperparameters to only include those with meaningful variance (from numeric columns)
  const { meaningfulHyperparams, filteredCount } = useMemo(() => {
    const meaningful = numericHyperparamColumns.filter(param => hasMeaningfulVariance(adaptedData, param));
    return {
      meaningfulHyperparams: meaningful,
      filteredCount: numericHyperparamColumns.length - meaningful.length,
    };
  }, [adaptedData, numericHyperparamColumns]);

  // Prepare data for advanced visualizations
  const correlationData = useMemo(() => {
    if (numericMetricColumns.length === 0 || numericHyperparamColumns.length === 0) return [];
    return numericMetricColumns.map((metric, i) => {
      const correlations = hparamMetricCorr(adaptedData, metric, numericHyperparamColumns);
      return numericHyperparamColumns.map((param, j) => {
        return [j, i, correlations[param]];
      });
    }).flat();
  }, [adaptedData, numericMetricColumns, numericHyperparamColumns]);

  // Check if we have data
  if (adaptedData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Charts</CardTitle>
          <CardDescription>No data available for visualization</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-muted-foreground">
            No experiments found. Run some experiments to see visualizations here.
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Charts & Visualizations</h2>
        <p className="text-muted-foreground mt-1">
          {selectedExperiments.length === 0
            ? `Exploring all ${experimentsData.length} runs`
            : `Exploring ${selectedExperiments.length} selected runs`}
        </p>
        <p className="text-xs text-muted-foreground mt-1">
          Interactive exploration charts. Each chart can be customized independently.
        </p>
      </div>

      {/* Metrics Comparison Chart - Customizable */}
      {numericMetricColumns.length > 0 && useMemo(() => {
        // Prepare data for chart
        const metricsChartData = adaptedData.map(exp => {
          const metrics = flattenObject(exp.metrics?.final || {});
          return {
            name: exp.run_name?.substring(0, 15) || 'Unknown',
            fullName: exp.run_name,
            ...numericMetricColumns.slice(0, 6).reduce((acc, metric) => {
              acc[metric] = typeof metrics[metric] === 'number' ? metrics[metric] : 0;
              return acc;
            }, {} as Record<string, number>)
          };
        });

        // Build series configuration
        const series: SeriesConfig[] = numericMetricColumns.slice(0, 6).map((metric, idx) => ({
          dataKey: metric,
          name: metric.replace(/_/g, ' '),
          color: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'][idx % 6],
          visible: idx < 4, // Show first 4 by default
        }));

        return (
          <CustomizableChartCard
            key="metrics-comparison-chart"
            title="Metrics Comparison"
            description={`Compare final metric values across runs (top ${Math.min(6, numericMetricColumns.length)} numeric metrics)`}
            data={metricsChartData}
            initialSeries={series}
            chartType="bar"
            supportedChartTypes={['line', 'bar', 'area']}
            initialRenderer="recharts"
            xAxisKey="name"
            compact={true}
            initialSettings={{
              height: 400,
              legend: { show: true, position: 'right', align: 'center' },
              xAxis: { label: 'Runs', showGrid: true },
              yAxis: { label: 'Metric Values', showGrid: true, scale: 'linear' },
            }}
          />
        );
      }, [adaptedData, numericMetricColumns])}

      {/* Scatter Plot Analysis - Most Important Chart */}
      {numericHyperparamColumns.length > 0 && numericMetricColumns.length > 0 && (
        <ScatterPlotChart
          data={adaptedData.map(exp => ({
            ...flattenObject(exp.hyperparameters || {}),
            ...flattenObject(exp.metrics?.final || {}),
            run_name: exp.run_name
          }))}
          xOptions={numericHyperparamColumns}
          yOptions={numericMetricColumns}
          title="Hyperparameter vs Metric Analysis"
          description="Explore relationships between numeric hyperparameters and metrics"
          defaultX={numericHyperparamColumns[0]}
          defaultY={numericMetricColumns[0]}
          variant="compact"
        />
      )}


      {/* Hyperparameter Distributions - Rich Visualization with Selector */}
      {numericHyperparamColumns.length > 0 && (() => {
        if (meaningfulHyperparams.length === 0) {
          return (
            <RichDistributionChart
              runs={[]}
              metricKey=""
              title="Hyperparameter Distribution"
              description="No hyperparameters with variance detected"
              height={300}
              showSettings={false}
              showFullscreen={false}
              showExport={false}
            />
          );
        }

        // Use selected hyperparameter or default to first one
        const currentParam = selectedHyperparam || meaningfulHyperparams[0];

        // Adapt data: move hyperparameter values to metrics.final for DistributionChart
        const adaptedRuns: Run[] = adaptedData.map(exp => ({
          ...exp,
          metrics: {
            ...exp.metrics,
            final: {
              ...(exp.metrics?.final || {}),
              [currentParam]: flattenObject(exp.hyperparameters || {})[currentParam]
            }
          }
        })) as Run[];

        // Get distribution data to check for outliers
        const distributionPreview = transformForDistribution(adaptedRuns, currentParam, 20);

        return (
          <RichDistributionChart
            runs={adaptedRuns}
            metricKey={currentParam}
            title="Hyperparameter Distribution"
            description={`${adaptedData.length} runs analyzed, ${distributionPreview.bins.length} bins${filteredCount > 0 ? ` (${filteredCount} params without variance hidden)` : ''}`}
            badge={
              distributionPreview.outliers.length > 0
                ? { label: `${distributionPreview.outliers.length} Outliers`, variant: 'destructive' }
                : undefined
            }
            height={400}
            showBoxPlot={true}
            variant="compact"
            customControls={
              <Select
                value={currentParam}
                onValueChange={setSelectedHyperparam}
              >
                <SelectTrigger className="w-[250px] h-8">
                  <SelectValue placeholder="Select hyperparameter" />
                </SelectTrigger>
                <SelectContent>
                  {meaningfulHyperparams.map(param => (
                    <SelectItem key={param} value={param}>
                      {param.replace(/_/g, ' ')}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
          />
        );
      })()}

      {/* Correlation Heatmap */}
      {numericHyperparamColumns.length > 0 && numericMetricColumns.length > 0 && correlationData.length > 0 && (
        <CorrelationHeatmap
          data={correlationData}
          xAxis={numericHyperparamColumns}
          yAxis={numericMetricColumns}
          title="Hyperparameter-Metric Correlations"
          description="Heatmap showing correlations between hyperparameters and metrics. Darker colors indicate stronger correlations."
          height={450}
          variant="compact"
        />
      )}

    </div>
  );
}
