import React from "react";
/**
 * Run Detail Modal
 *
 * Comprehensive drill-down view for a single experiment run.
 * Provides detailed analysis and visualization of run data.
 *
 * Features:
 * - Run metadata and hyperparameters
 * - Final metrics summary
 * - Training curves for all metrics
 * - Hyperparameter-metric correlation analysis
 * - Metric distribution comparison
 * - Export functionality
 */

import { useState, useMemo, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import {
  Dialog,
  DialogContent,
  _DialogHeader,
  DialogTitle,
  DialogDescription,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/card';
import { Download, ExternalLink, FileCode, X } from 'lucide-react';
import type { Run } from '../../lib/types';
import type { SeriesConfig } from '../../lib/charts/types';
import { InlineEmptyState } from '../charts/EmptyState';
import { ConfigViewerModal } from './ConfigViewerModal';
import { NotesTagsEditor } from '../ui/notes-tags-editor';
import { ArtifactBrowser, type Artifact } from '../ui/artifact-browser';
import { CustomizableChartCard } from '../charts/CustomizableChartCard';
import { apiClient } from '../../lib/api-client';
import { RunStatusBadge, type RunStatus } from '../ui/run-status-badge';
import { devError } from '../../lib/dev-utils';

interface RunDetailModalProps {
  open: boolean;
  onClose: () => void;
  run: Run | null;
  _allRuns?: Run[]; // For comparison and distribution
  onCompareClick?: (runName: string) => void;
}

export function RunDetailModal({
  open,
  onClose,
  run,
  _allRuns = [],
  onCompareClick,
}: RunDetailModalProps) {
  const [activeTab, setActiveTab] = useState('summary');
  const [trainingView, setTrainingView] = useState<'unified' | 'individual'>('unified');
  const [configViewerOpen, setConfigViewerOpen] = useState(false);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [isLoadingArtifacts, setIsLoadingArtifacts] = useState(false);
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null);

  // Handle notes/tags save - refresh the run data if needed
  const handleNotesTagsSave = (notes: string, tags: string[]) => {
    // Update the run object in-place to reflect changes
    if (run) {
      run.notes = notes;
      run.tags = tags;
    }
  };

  // Fetch artifacts when modal opens
  useEffect(() => {
    if (open && run) {
      const fetchArtifacts = async () => {
        setIsLoadingArtifacts(true);
        try {
          const response = await apiClient.getRunArtifacts(run.project, run.run_name);
          setArtifacts(response.artifacts);
        } catch (error) {
          devError('Failed to fetch artifacts:', error);
          setArtifacts([]);
        } finally {
          setIsLoadingArtifacts(false);
        }
      };
      fetchArtifacts();
    }
  }, [open, run]);

  // Handle artifact download
  const handleDownloadArtifact = async (artifact: Artifact) => {
    if (!run) return;
    setDownloadingFile(artifact.path);
    try {
      await apiClient.downloadArtifact(run.project, run.run_name, artifact.path);
    } catch (error) {
      devError('Failed to download artifact:', error);
      alert(`Failed to download ${artifact.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setDownloadingFile(null);
    }
  };

  // Extract metric keys from timeseries data (for Training tab)
  const metricKeys = useMemo(() => {
    if (!run?.metrics?.timeseries || run.metrics.timeseries.length === 0) return [];
    // Get keys from first timeseries entry
    const firstEntry = run.metrics.timeseries[0];
    return Object.keys(firstEntry.values || {});
  }, [run?.metrics?.timeseries]);

  // Extract hyperparameter keys
  const _hyperparamKeys = useMemo(() => {
    if (!run?.hyperparameters) return [];
    return Object.keys(run.hyperparameters).filter(key => {
      const value = run.hyperparameters[key];
      return typeof value === 'number';
    });
  }, [run?.hyperparameters]);

  // Memoize unified chart data
  const unifiedChartData = useMemo(() => {
    return run?.metrics?.timeseries?.map((point) => ({
      step: point.step,
      ...point.values,
    })) || [];
  }, [run?.metrics?.timeseries]);

  // Memoize unified chart series
  const unifiedChartSeries = useMemo(() => {
    const topMetrics = metricKeys.slice(0, Math.min(6, metricKeys.length));
    return topMetrics.map((metric, idx) => ({
      dataKey: metric,
      name: metric.replace(/_/g, ' '),
      color: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'][idx % 6],
      visible: idx < 4, // Show first 4 by default
    }));
  }, [metricKeys]);

  // Export run data as JSON
  const exportRunData = () => {
    if (!run) return;

    const dataStr = JSON.stringify(run, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${run.run_name}_data.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (!run) {
    return null;
  }

  return (
    <>
      <Dialog open={open} onOpenChange={onClose}>
        <DialogContent className="!w-[90vw] !max-w-[90vw] !h-[90vh] overflow-hidden flex flex-col gap-4" hideCloseButton>
        <div className="flex flex-col space-y-1.5 text-left">
          <div className="flex items-start justify-between gap-4">
            <DialogTitle>{run.run_name}</DialogTitle>
            {/* Action Buttons with X */}
            <div className="flex items-center gap-2 flex-shrink-0">
              {onCompareClick && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onCompareClick(run.run_name)}
                >
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Compare
                </Button>
              )}
              <Button variant="outline" size="sm" onClick={exportRunData}>
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                aria-label="Close"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <DialogDescription>
            Detailed analysis and metrics for this experiment run
          </DialogDescription>
        </div>

        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Simple Tab Navigation - Reduced to 3 tabs */}
          <div className="flex bg-muted rounded-lg p-1 gap-1">
            <button
              onClick={() => setActiveTab('summary')}
              className={`
                flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
                ${activeTab === 'summary'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
                }
              `}
            >
              Summary
            </button>
            <button
              onClick={() => setActiveTab('training')}
              className={`
                flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
                ${activeTab === 'training'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
                }
              `}
            >
              Training
            </button>
            <button
              onClick={() => setActiveTab('diagnostics')}
              className={`
                flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
                ${activeTab === 'diagnostics'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
                }
              `}
            >
              Diagnostics
            </button>
          </div>

          <div className="flex-1 mt-4 overflow-y-auto px-1">
            {/* Summary Tab - Overview + Final Metrics Only */}
            {activeTab === 'summary' && (
              <div className="space-y-4">
              {/* Metadata */}
              <Card variant="compact">
                <CardHeader variant="compact">
                  <CardTitle size="base">Run Information</CardTitle>
                </CardHeader>
                <CardContent variant="compact">
                  <div className="grid grid-cols-2 gap-3 md:gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Run Name:</span>
                      <p className="font-mono mt-1">{run.run_name}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Project:</span>
                      <p className="font-mono mt-1">{run.project}</p>
                    </div>
                    {run.started_at && (
                      <div>
                        <span className="text-muted-foreground">Started:</span>
                        <p className="font-mono mt-1">
                          {new Date(run.started_at).toLocaleString()}
                        </p>
                      </div>
                    )}
                    {run.status && (
                      <div>
                        <span className="text-muted-foreground">Status:</span>
                        <div className="mt-1">
                          <RunStatusBadge status={run.status as RunStatus} />
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Notes & Tags Editor */}
              <NotesTagsEditor
                project={run.project}
                runName={run.run_name}
                initialNotes={run.notes || ''}
                initialTags={run.tags || []}
                onSave={handleNotesTagsSave}
              />

              {/* Final Metrics */}
              {run.metrics?.final && (
                <Card variant="compact">
                  <CardHeader variant="compact">
                    <CardTitle size="base">Final Metrics</CardTitle>
                    <CardDescription>
                      Final performance measurements from this run
                    </CardDescription>
                  </CardHeader>
                  <CardContent variant="compact">
                    <div className="grid grid-cols-3 gap-3 md:gap-4">
                      {Object.entries(run.metrics.final)
                        .filter(([key]) => key !== 'step' && key !== 'epoch')
                        .map(([key, value]) => (
                          <div key={key} className="border rounded-lg p-3">
                            <span className="text-xs text-muted-foreground">{key}</span>
                            <p className="text-2xl font-bold font-mono mt-1">
                              {typeof value === 'number' ? value.toFixed(4) : value}
                            </p>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
            )}

            {/* Training Tab - Metrics over time */}
            {activeTab === 'training' && (
              <div className="space-y-4">
                {/* View Toggle */}
                <div className="flex bg-muted rounded-lg p-1 gap-1 w-fit">
                  <button
                    onClick={() => setTrainingView('unified')}
                    className={`
                      px-3 py-1.5 rounded-md text-xs font-medium transition-colors duration-200
                      ${trainingView === 'unified'
                        ? 'bg-background text-foreground shadow-sm'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
                      }
                    `}
                  >
                    Unified View
                  </button>
                  <button
                    onClick={() => setTrainingView('individual')}
                    className={`
                      px-3 py-1.5 rounded-md text-xs font-medium transition-colors duration-200
                      ${trainingView === 'individual'
                        ? 'bg-background text-foreground shadow-sm'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
                      }
                    `}
                  >
                    Individual Metrics
                  </button>
                </div>

                {metricKeys.length === 0 ? (
                  <InlineEmptyState message="No timeseries metrics available for this run" />
                ) : (
                  <>
                    {/* Unified View */}
                    {trainingView === 'unified' && (
                      <CustomizableChartCard
                        key={`unified-chart-${run.run_name}`}
                        title="Training Metrics Over Time"
                        description={`Monitoring ${unifiedChartSeries.length} key metrics across training steps`}
                        data={unifiedChartData}
                        initialSeries={unifiedChartSeries}
                        chartType="line"
                        supportedChartTypes={['line', 'area', 'bar', 'scatter']}
                        xAxisKey="step"
                        initialSettings={{
                          height: 350,
                          legend: { show: true, position: 'bottom', align: 'center' },
                          xAxis: { label: 'Step', showGrid: true },
                          yAxis: { label: 'Value', showGrid: true, scale: 'linear' },
                        }}
                        compact
                      />
                    )}

                    {/* Individual View */}
                    {trainingView === 'individual' && (
                      <>
                        {metricKeys.map(metricKey => {
                          // Prepare individual metric chart data
                          const chartData = run.metrics?.timeseries?.map((point) => ({
                            step: point.step,
                            value: point.values[metricKey],
                          })) || [];

                          const series: SeriesConfig[] = [{
                            dataKey: 'value',
                            name: metricKey.replace(/_/g, ' '),
                            color: '#3b82f6',
                            visible: true,
                          }];

                          return (
                            <CustomizableChartCard
                              key={`individual-chart-${run.run_name}-${metricKey}`}
                              title={metricKey.replace(/_/g, ' ')}
                              description={`${metricKey} progression over training steps`}
                              data={chartData}
                              initialSeries={series}
                              chartType="line"
                              supportedChartTypes={['line', 'area', 'bar', 'scatter']}
                              xAxisKey="step"
                              initialSettings={{
                                height: 250,
                                legend: { show: false, position: 'top', align: 'center' },
                                xAxis: { label: 'Step', showGrid: true },
                                yAxis: { label: metricKey, showGrid: true, scale: 'linear' },
                              }}
                              compact
                              showSeriesSelector={false}
                            />
                          );
                        })}
                      </>
                    )}
                  </>
                )}
              </div>
            )}

            {/* Diagnostics Tab - Detailed Configuration & System Info */}
            {activeTab === 'diagnostics' && (
              <div className="space-y-4">
                {/* System Information */}
                <Card variant="compact">
                  <CardHeader variant="compact">
                    <CardTitle size="base">System Information</CardTitle>
                    <CardDescription>
                      Runtime environment and execution details
                    </CardDescription>
                  </CardHeader>
                  <CardContent variant="compact">
                    <div className="grid grid-cols-2 gap-3 md:gap-4 text-sm">
                      <div className="border-b pb-2">
                        <span className="text-muted-foreground">Run Name:</span>
                        <p className="font-mono text-xs mt-1">{run.run_name}</p>
                      </div>
                      <div className="border-b pb-2">
                        <span className="text-muted-foreground">Project:</span>
                        <p className="font-mono text-xs mt-1">{run.project}</p>
                      </div>
                      <div className="border-b pb-2">
                        <span className="text-muted-foreground">Started At:</span>
                        <p className="font-mono text-xs mt-1">
                          {run.started_at ? new Date(run.started_at).toISOString() : 'N/A'}
                        </p>
                      </div>
                      <div className="border-b pb-2">
                        <span className="text-muted-foreground">Status:</span>
                        <p className="font-mono text-xs mt-1">{run.status || 'unknown'}</p>
                      </div>
                      {run.metrics && Object.keys(run.metrics).length > 0 && (
                        <>
                          <div className="border-b pb-2">
                            <span className="text-muted-foreground">Metrics Count:</span>
                            <p className="font-mono text-xs mt-1">
                              {Object.keys(run.metrics.final || {}).length} final metrics tracked
                            </p>
                          </div>
                          <div className="border-b pb-2">
                            <span className="text-muted-foreground">Hyperparameters Count:</span>
                            <p className="font-mono text-xs mt-1">
                              {Object.keys(run.hyperparameters || {}).length} parameters configured
                            </p>
                          </div>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Run Configuration - Tabs for Hyperparameters vs Metadata */}
                <Card variant="compact">
                  <CardHeader variant="compact">
                    <CardTitle size="base">Configuration</CardTitle>
                    <CardDescription>
                      Experiment hyperparameters and run metadata
                    </CardDescription>
                  </CardHeader>
                  <CardContent variant="compact">
                    <Tabs defaultValue="hyperparams" className="w-full">
                      <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="hyperparams">Hyperparameters</TabsTrigger>
                        <TabsTrigger value="metadata">Metadata</TabsTrigger>
                      </TabsList>

                      <TabsContent value="hyperparams" className="mt-3">
                        {run.hyperparameters && Object.keys(run.hyperparameters).length > 0 ? (
                          <div className="grid grid-cols-2 gap-3 text-sm max-h-64 overflow-y-auto">
                            {Object.entries(run.hyperparameters).map(([key, value]) => (
                              <div key={key} className="border-b pb-2">
                                <span className="text-muted-foreground">{key}:</span>
                                <p className="font-mono text-xs mt-1 break-all">
                                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                </p>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="text-center py-6 text-muted-foreground text-sm">
                            No hyperparameters available
                          </div>
                        )}
                        <div className="mt-3 flex justify-end">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setConfigViewerOpen(true)}
                          >
                            <FileCode className="h-4 w-4 mr-2" />
                            View Full YAML
                          </Button>
                        </div>
                      </TabsContent>

                      <TabsContent value="metadata" className="mt-3">
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div className="border-b pb-2">
                            <span className="text-muted-foreground">Run Name:</span>
                            <p className="font-mono text-xs mt-1 break-all">{run.run_name}</p>
                          </div>
                          <div className="border-b pb-2">
                            <span className="text-muted-foreground">Project:</span>
                            <p className="font-mono text-xs mt-1">{run.project}</p>
                          </div>
                          <div className="border-b pb-2">
                            <span className="text-muted-foreground">Status:</span>
                            <p className="font-mono text-xs mt-1 capitalize">{run.status || 'unknown'}</p>
                          </div>
                          <div className="border-b pb-2">
                            <span className="text-muted-foreground">Started:</span>
                            <p className="font-mono text-xs mt-1">
                              {run.started_at ? new Date(run.started_at).toLocaleString() : 'N/A'}
                            </p>
                          </div>
                          <div className="border-b pb-2">
                            <span className="text-muted-foreground">Finished:</span>
                            <p className="font-mono text-xs mt-1">
                              {run.finished_at ? new Date(run.finished_at).toLocaleString() : 'N/A'}
                            </p>
                          </div>
                          <div className="border-b pb-2">
                            <span className="text-muted-foreground">Total Steps:</span>
                            <p className="font-mono text-xs mt-1">
                              {run.metrics?.timeseries?.[run.metrics.timeseries.length - 1]?.step ?? 'N/A'}
                            </p>
                          </div>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>

                {/* Artifacts Section */}
                <Card variant="compact">
                  <CardHeader variant="compact">
                    <CardTitle size="base">Artifacts</CardTitle>
                    <CardDescription>
                      Download experiment artifacts including configs, checkpoints, and logs
                    </CardDescription>
                  </CardHeader>
                  <CardContent variant="compact">
                    <ArtifactBrowser
                      artifacts={artifacts}
                      isLoading={isLoadingArtifacts}
                      onDownload={handleDownloadArtifact}
                      downloadingFile={downloadingFile}
                    />
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>

      {/* Config Viewer Modal */}
      <ConfigViewerModal
        open={configViewerOpen}
        onClose={() => setConfigViewerOpen(false)}
        project={run.project}
        runName={run.run_name}
      />
    </>
  );
}
