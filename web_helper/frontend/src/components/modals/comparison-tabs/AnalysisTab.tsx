import React from "react";
import { useState, useMemo } from 'react';
import { DistributionChart } from '../../charts/DistributionChart';
import ParallelCoordinatesChart from '../../charts/ParallelCoordinatesChart';
import { ScatterMatrixChart } from '../../charts/ScatterMatrixChart';
import { EmptyState } from '../../charts/EmptyState';
import { BarChart3, Layers, Grid3x3 } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../ui/select';
import type { Run } from '../../../lib/types';
import { transformForParallel } from '../../../lib/chartDataTransformers';
import { devLog, devError } from '../../../lib/dev-utils';

interface AnalysisTabProps {
  runs: Run[];
  availableMetrics: string[];
  hyperparamKeys: string[];
  _onRunSelect?: (runName: string) => void;
}

export function AnalysisTab({
  runs,
  availableMetrics,
  hyperparamKeys,
  _onRunSelect,
}: AnalysisTabProps) {
  const [activeSubTab, setActiveSubTab] = useState<'distribution' | 'parallel' | 'matrix'>('distribution');
  const [error, setError] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);

  if (availableMetrics.length === 0) {
    return <EmptyState variant="no-data" />;
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <div className="text-destructive text-sm font-medium">Error loading analysis</div>
        <div className="text-xs text-muted-foreground max-w-md text-center">{error}</div>
        <button
          onClick={() => setError(null)}
          className="text-xs text-primary hover:underline"
        >
          Try again
        </button>
      </div>
    );
  }

  return (
    <div className="w-full overflow-hidden">
      <div className="flex bg-muted rounded-lg p-1 gap-1">
        <button
          onClick={() => setActiveSubTab('distribution')}
          className={`
            flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
            ${activeSubTab === 'distribution'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
            }
          `}
        >
          <BarChart3 className="h-4 w-4" />
          Distribution
        </button>
        <button
          onClick={() => setActiveSubTab('parallel')}
          className={`
            flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
            ${activeSubTab === 'parallel'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
            }
          `}
        >
          <Layers className="h-4 w-4" />
          Parallel
        </button>
        <button
          onClick={() => setActiveSubTab('matrix')}
          className={`
            flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
            ${activeSubTab === 'matrix'
              ? 'bg-background text-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
            }
          `}
        >
          <Grid3x3 className="h-4 w-4" />
          Matrix
        </button>
      </div>

      <div className="mt-4">
        {activeSubTab === 'distribution' && (
          <div className="space-y-4">
            {(() => {
              try {
                // Validate runs data
                if (!runs || runs.length === 0) {
                  return <EmptyState variant="no-data" />;
                }

                // Use selected metric or default to first one
                const currentMetric = selectedMetric || availableMetrics[0];

                return (
                  <div className="space-y-4">
                    {/* Metric Selector */}
                    <div className="flex items-center justify-between">
                      <div className="text-sm text-muted-foreground">
                        Select a metric to view its distribution across {runs.length} runs
                      </div>
                      <Select
                        value={currentMetric}
                        onValueChange={setSelectedMetric}
                      >
                        <SelectTrigger className="w-[250px]">
                          <SelectValue placeholder="Select metric" />
                        </SelectTrigger>
                        <SelectContent>
                          {availableMetrics.map(metric => (
                            <SelectItem key={metric} value={metric}>
                              {metric.replace(/_/g, ' ')}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Single Distribution Chart */}
                    <DistributionChart
                      runs={runs}
                      metricKey={currentMetric}
                      title={`Distribution of ${currentMetric}`}
                      height={400}
                      showBoxPlot={true}
                      variant="default"
                    />
                  </div>
                );
              } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to render distribution chart');
                return null;
              }
            })()}
          </div>
        )}

        {activeSubTab === 'parallel' && (
          <ParallelCoordinatesSection
            runs={runs}
            hyperparamKeys={hyperparamKeys}
            availableMetrics={availableMetrics}
            onError={setError}
          />
        )}

        {activeSubTab === 'matrix' && (
          <div className="space-y-4">
            {(() => {
              try {
                if (availableMetrics.length < 2) {
                  return <EmptyState variant="no-metric" />;
                }

                // Validate runs data
                if (!runs || runs.length === 0) {
                  return <EmptyState variant="no-data" />;
                }

                return (
                  <ScatterMatrixChart
                    runs={runs}
                    metricKeys={availableMetrics.slice(0, 4)}
                    title="Metric Correlation Matrix"
                    height={600}
                    showCorrelation={true}
                    onCellClick={(metricX, metricY) => {
                      devLog('Clicked cell:', metricX, metricY);
                    }}
                    variant="compact"
                  />
                );
              } catch (err) {
                devError('Error rendering scatter matrix chart:', err);
                setError(err instanceof Error ? err.message : 'Failed to render scatter matrix chart');
                return null;
              }
            })()}
          </div>
        )}
      </div>
    </div>
  );
}

// Separate component to properly memoize parallel coordinates data
interface ParallelCoordinatesSectionProps {
  runs: Run[];
  hyperparamKeys: string[];
  availableMetrics: string[];
  onError: (error: string) => void;
}

function ParallelCoordinatesSection({
  runs,
  hyperparamKeys,
  availableMetrics,
  onError,
}: ParallelCoordinatesSectionProps) {
  // Memoize the transformation to prevent unnecessary recalculations
  const parallelData = useMemo(() => {
    try {
      if (hyperparamKeys.length === 0 || availableMetrics.length === 0) {
        return null;
      }

      if (!runs || runs.length === 0) {
        return null;
      }

      const { runs: transformedRuns, dimensions } = transformForParallel(
        runs.slice(0, 10),
        hyperparamKeys.slice(0, 5),
        availableMetrics.slice(0, 3)
      );

      if (!transformedRuns || transformedRuns.length === 0 || !dimensions || dimensions.length === 0) {
        return null;
      }

      const data = transformedRuns.map(r => Object.values(r.values));

      return { data, dimensions };
    } catch (err) {
      devError('Error transforming data for parallel coordinates:', err);
      onError(err instanceof Error ? err.message : 'Failed to transform data');
      return null;
    }
  }, [runs, hyperparamKeys, availableMetrics, onError]);

  if (!parallelData) {
    return (
      <div className="space-y-4">
        <EmptyState variant="no-data" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <ParallelCoordinatesChart
        data={parallelData.data}
        dimensions={parallelData.dimensions}
      />
    </div>
  );
}
