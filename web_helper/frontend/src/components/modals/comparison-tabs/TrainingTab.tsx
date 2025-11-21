import React from "react";
/**
 * Training Tab (Enhanced)
 *
 * Visualizes training progression with multiple view modes:
 * - Unified View: All metrics in one chart (Tableau-grade)
 * - Individual View: Single metric with detailed controls
 * - Hyperparameter Correlation: Correlation analysis
 * - Learning Rate: Learning rate schedule visualization
 * - Train vs Val: Training/Validation comparison with overfitting detection
 * - Convergence: Convergence speed analysis
 */

import { useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../../ui/tabs';
import { UnifiedMetricsChartV3 } from '../../charts/UnifiedMetricsChartV3';
import { AdvancedMetricsChart } from '../../charts/AdvancedMetricsChart';
import { HyperparamCorrelationChart } from '../../charts/HyperparamCorrelationChart';
import { LearningRateScheduleChart } from '../../charts/LearningRateScheduleChart';
import { TrainingValidationComparisonChart } from '../../charts/TrainingValidationComparisonChart';
import { ConvergenceSpeedChart } from '../../charts/ConvergenceSpeedChart';
import { EmptyState, InlineEmptyState } from '../../charts/EmptyState';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../ui/select';
import { TrendingUp, GitCompare, Layers, Zap, GitMerge, Activity } from 'lucide-react';
import type { Run } from '../../../lib/types';

interface TrainingTabProps {
  runs: Run[];
  availableMetrics: string[];
  hyperparamKeys: string[];
  onPointClick?: (runName: string) => void;
}

export function TrainingTab({
  runs,
  availableMetrics,
  hyperparamKeys,
  onPointClick,
}: TrainingTabProps) {
  const [selectedMetric, setSelectedMetric] = useState<string>(availableMetrics[0] || '');
  const [metricDisplayMode, setMetricDisplayMode] = useState<'final' | 'max' | 'min'>('final');
  const [selectedHyperparam, setSelectedHyperparam] = useState<string>(hyperparamKeys[0] || '');

  if (availableMetrics.length === 0) {
    return <EmptyState variant="no-data" />;
  }

  return (
    <Tabs defaultValue="unified" className="w-full">
      <TabsList className="grid w-full grid-cols-6">
        <TabsTrigger value="unified" className="flex items-center gap-1.5">
          <Layers className="h-4 w-4" />
          <span>Unified</span>
        </TabsTrigger>
        <TabsTrigger value="individual" className="flex items-center gap-1.5">
          <TrendingUp className="h-4 w-4" />
          <span>Individual</span>
        </TabsTrigger>
        <TabsTrigger value="correlation" className="flex items-center gap-1.5">
          <GitCompare className="h-4 w-4" />
          <span>Correlation</span>
        </TabsTrigger>
        <TabsTrigger value="learning-rate" className="flex items-center gap-1.5">
          <Activity className="h-4 w-4" />
          <span>LR Schedule</span>
        </TabsTrigger>
        <TabsTrigger value="train-val" className="flex items-center gap-1.5">
          <GitMerge className="h-4 w-4" />
          <span>Train/Val</span>
        </TabsTrigger>
        <TabsTrigger value="convergence" className="flex items-center gap-1.5">
          <Zap className="h-4 w-4" />
          <span>Convergence</span>
        </TabsTrigger>
      </TabsList>

      <TabsContent value="unified" className="mt-2">
        <div className="space-y-4">
          {/* Chart */}
          <UnifiedMetricsChartV3
            runs={runs}
            availableMetrics={availableMetrics}
            title="Training Metrics - Unified View"
            height={480}
            defaultSelectedMetrics={availableMetrics.slice(0, Math.min(5, availableMetrics.length))}
          />
        </div>
      </TabsContent>

      <TabsContent value="individual" className="mt-2">
        {selectedMetric ? (
          <AdvancedMetricsChart
            runs={runs}
            metricKey={selectedMetric}
            title={selectedMetric}
            height={420}
            enableZoom
            enableBrush
            enableSmoothing
            showControls
            customControls={
              <Select value={selectedMetric} onValueChange={setSelectedMetric}>
                <SelectTrigger className="w-[240px] h-8">
                  <SelectValue placeholder="Select Metric" />
                </SelectTrigger>
                <SelectContent>
                  {availableMetrics.map(metric => (
                    <SelectItem key={metric} value={metric}>
                      {metric}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
          />
        ) : (
          <div className="flex items-center justify-center h-[420px] text-muted-foreground text-sm">
            Please select a metric to display
          </div>
        )}
      </TabsContent>

      <TabsContent value="correlation" className="mt-2">
        <div className="space-y-4">
          {/* Control Panel */}
          {hyperparamKeys.length > 0 && availableMetrics.length > 0 ? (
            <>
              <div className="flex items-center gap-4 pb-2 border-b">
                {/* Metric Display Mode Selector */}
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium">Metric Value:</label>
                  <Select value={metricDisplayMode} onValueChange={(v) => setMetricDisplayMode(v as 'final' | 'max' | 'min')}>
                    <SelectTrigger className="w-[140px] h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="final">Last (Final)</SelectItem>
                      <SelectItem value="max">Highest (Max)</SelectItem>
                      <SelectItem value="min">Lowest (Min)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Hyperparameter Selector */}
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium">Hyperparameter:</label>
                  <Select value={selectedHyperparam} onValueChange={setSelectedHyperparam}>
                    <SelectTrigger className="w-[200px] h-8">
                      <SelectValue placeholder="Select hyperparameter" />
                    </SelectTrigger>
                    <SelectContent>
                      {hyperparamKeys.map(key => (
                        <SelectItem key={key} value={key}>
                          {key}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Charts Grid */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                {availableMetrics.slice(0, 6).map(metricKey => (
                  <HyperparamCorrelationChart
                    key={`${selectedHyperparam}-${metricKey}-${metricDisplayMode}`}
                    runs={runs}
                    hyperparamKey={selectedHyperparam}
                    metricKey={metricKey}
                    metricDisplayMode={metricDisplayMode}
                    title={`${selectedHyperparam} vs ${metricKey}`}
                    height={300}
                    showTrendLine={true}
                    showOutliers={true}
                    onPointClick={onPointClick}
                    variant="compact"
                  />
                ))}
              </div>
            </>
          ) : (
            <InlineEmptyState message="No numeric hyperparameters available for correlation analysis" />
          )}
        </div>
      </TabsContent>

      <TabsContent value="learning-rate" className="mt-2">
        <div className="space-y-4">
          {/* Chart */}
          <LearningRateScheduleChart runs={runs} height={420} />
        </div>
      </TabsContent>

      <TabsContent value="train-val" className="mt-2">
        <div className="space-y-4">
          {/* Chart */}
          <TrainingValidationComparisonChart
            runs={runs}
            availableMetrics={availableMetrics}
            height={450}
          />
        </div>
      </TabsContent>

      <TabsContent value="convergence" className="mt-2">
        <div className="space-y-4">
          <ConvergenceSpeedChart
            runs={runs}
            availableMetrics={availableMetrics}
            height={420}
            targetThreshold={0.95}
          />
        </div>
      </TabsContent>
    </Tabs>
  );
}
