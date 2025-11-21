import React from "react";
/**
 * Metrics Comparison Modal (Enhanced)
 * Advanced modal for comparing multiple experiment runs
 *
 * Features:
 * - Side-by-side metric comparison
 * - Multiple chart views (training curves, correlation, distribution)
 * - Advanced visualizations (Parallel Coordinates, Scatter Matrix)
 * - Statistical summaries with drill-down
 * - Export comparison reports
 * - Save comparison as preset
 * - Interactive highlighting and filtering
 */

import { useState, useMemo, useEffect} from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { RunDetailModal } from './RunDetailModal';
import { useRunDetail } from '../../contexts/ChartInteractionContext';
import { saveView, updateView, getProjectSavedViews, type SavedView } from '../../lib/saved-views';
import { useMetricsTimeseries } from '../../hooks/useMetricsTimeseries';
import { EmptyState } from '../charts/EmptyState';
import { ScrollArea } from '../ui/scroll-area';

// Direct imports (not lazy) to avoid dynamic import issues
import { OverviewTab } from './comparison-tabs/OverviewTab';
import { TrainingTab } from './comparison-tabs/TrainingTab';
import { AnalysisTab } from './comparison-tabs/AnalysisTab';
import { DetailsTab } from './comparison-tabs/DetailsTab';
import {
  Save,
  Eye,
  TrendingUp,
  BarChart3,
  X,
  Loader2,
  Table,
  Download,
  Share,
} from 'lucide-react';
import type { Run } from '../../lib/types';

interface MetricsComparisonModalProps {
  open: boolean;
  onClose: () => void;
  runs: Run[];
  projectName: string;
}

export function MetricsComparisonModal({
  open,
  onClose,
  runs,
  projectName,
}: MetricsComparisonModalProps) {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [comparisonName, setComparisonName] = useState<string>('');
  const [currentSavedView, setCurrentSavedView] = useState<SavedView | null>(null);
  const [originalViewName, setOriginalViewName] = useState<string>('');
  const [hasUserEditedName, setHasUserEditedName] = useState<boolean>(false);

  // Use run detail hook for drill-down
  const { selectedRun, openRunDetail, closeRunDetail } = useRunDetail(runs);

  // Detect if current runs match any saved view (only on initial open)
  useEffect(() => {
    if (!open || runs.length === 0) return;

    const savedViews = getProjectSavedViews(projectName);
    const currentRunNames = runs.map(r => r.run_name).sort().join(',');

    // Find matching saved view
    const matchingView = savedViews.find(view => {
      const viewRunNames = view.state.selectedRuns.sort().join(',');
      return viewRunNames === currentRunNames;
    });

    if (matchingView) {
      setCurrentSavedView(matchingView);
      // Only set name if user hasn't edited it
      if (!hasUserEditedName) {
        setComparisonName(matchingView.name);
        setOriginalViewName(matchingView.name);
      }
    } else {
      setCurrentSavedView(null);
      if (!hasUserEditedName) {
        setComparisonName('');
        setOriginalViewName('');
      }
    }
  }, [open, runs, projectName]);

  // Reset user edit flag when modal opens
  useEffect(() => {
    if (open) {
      setHasUserEditedName(false);
    }
  }, [open]);

  // Load timeseries data for all runs
  const {
    runs: runsWithTimeseries,
    isLoading: isLoadingTimeseries,
  } = useMetricsTimeseries(runs, {
    downsample: 500,
    enabled: open && runs.length > 0,
  });

  // Extract all available metrics
  const availableMetrics = useMemo(() => {
    const metrics = new Set<string>();

    runs.forEach(run => {
      if (run.metrics?.final) {
        Object.keys(run.metrics.final).forEach(key => {
          // Filter out non-metric fields
          if (key !== 'step' && key !== 'epoch') {
            metrics.add(key);
          }
        });
      }
    });

    return Array.from(metrics);
  }, [runs]);

  // Extract numeric hyperparameter keys
  const hyperparamKeys = useMemo(() => {
    const keys = new Set<string>();

    runs.forEach(run => {
      if (run.hyperparameters) {
        Object.entries(run.hyperparameters).forEach(([key, value]) => {
          if (typeof value === 'number') {
            keys.add(key);
          }
        });
      }
    });

    return Array.from(keys);
  }, [runs]);

  // Handle save comparison
  const handleSaveComparison = () => {
    if (!comparisonName.trim()) return;

    const viewState = {
      selectedRuns: runs.map(r => r.run_name),
      flatten: false,
      diffOnly: false,
      visibleHyperparams: [],
      visibleMetrics: [],
      pinnedLeftHyperparams: [],
      pinnedRightMetrics: [],
    };

    if (currentSavedView && comparisonName === originalViewName) {
      // Update existing view
      updateView(currentSavedView.id, {
        name: comparisonName,
        projectName,
        state: viewState,
      });
    } else {
      // Create new view
      const newView = saveView({
        name: comparisonName,
        projectName,
        state: viewState,
      });
      setCurrentSavedView(newView);
      setOriginalViewName(comparisonName);
    }

    // Trigger storage event to update other components
    window.dispatchEvent(new Event('storage'));
  };

  // Handle empty state
  if (runs.length === 0) {
    return (
      <Dialog open={open} onOpenChange={onClose}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Compare Runs</DialogTitle>
          </DialogHeader>
          <EmptyState variant="no-selection" onAction={onClose} actionLabel="Close" />
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="w-[90vw] h-[90vh] overflow-hidden flex flex-col p-6" hideCloseButton>
        {/* Header Section */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-4">
            <div>
              <DialogTitle>Compare Runs</DialogTitle>
              <DialogDescription className="mt-1">
                Comparing {runs.length} runs across {availableMetrics.length} metrics
                {isLoadingTimeseries && (
                  <span className="ml-2 inline-flex items-center gap-1 text-xs">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Loading timeseries...
                  </span>
                )}
              </DialogDescription>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              {/* Save Group */}
              <div className="flex items-center gap-0 border border-border rounded-md overflow-hidden h-8">
                <input
                  type="text"
                  placeholder="Comparison name..."
                  value={comparisonName}
                  onChange={(e) => {
                    setComparisonName(e.target.value);
                    setHasUserEditedName(true);
                  }}
                  className="h-full w-[200px] border-0 border-r rounded-none bg-transparent px-3 py-0 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0"
                  style={{ height: '32px', lineHeight: '32px', margin: 0, padding: '0 12px' }}
                />
                <Button
                  variant="ghost"
                  onClick={handleSaveComparison}
                  disabled={!comparisonName.trim() || (currentSavedView !== null && comparisonName === originalViewName)}
                  className="h-8 w-8 p-0 rounded-none hover:bg-accent"
                  title={
                    currentSavedView && comparisonName === originalViewName
                      ? "Already saved"
                      : !comparisonName.trim()
                      ? "Enter a name to save"
                      : "Save comparison"
                  }
                >
                  <Save className="h-4 w-4" />
                </Button>
              </div>

              <Button
                variant="outline"
                onClick={() => {
                  // Export functionality placeholder
                  console.log('Export comparison');
                }}
                className="h-8 w-8 p-0"
              >
                <Download className="h-4 w-4" />
              </Button>

              <Button
                variant="outline"
                onClick={() => {
                  if (navigator.share) {
                    navigator.share({
                      title: 'Experiment Comparison',
                      text: `Comparing ${runs.length} runs`,
                      url: window.location.href,
                    }).catch(() => {});
                  }
                }}
                className="h-8 w-8 p-0"
              >
                <Share className="h-4 w-4" />
              </Button>

              <Button variant="ghost" onClick={onClose} aria-label="Close" className="h-8 w-8 p-0">
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex bg-muted rounded-lg p-1 gap-1">
          <button
            onClick={() => setActiveTab('overview')}
            className={`
              flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
              ${activeTab === 'overview'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
              }
            `}
          >
            <Eye className="h-4 w-4" />
            Overview
          </button>
          <button
            onClick={() => setActiveTab('details')}
            className={`
              flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
              ${activeTab === 'details'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
              }
            `}
          >
            <Table className="h-4 w-4" />
            Details
          </button>
          <button
            onClick={() => setActiveTab('training')}
            className={`
              flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
              ${activeTab === 'training'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
              }
            `}
          >
            <TrendingUp className="h-4 w-4" />
            Training
          </button>
          <button
            onClick={() => setActiveTab('analysis')}
            className={`
              flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200
              ${activeTab === 'analysis'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
              }
            `}
          >
            <BarChart3 className="h-4 w-4" />
            Analysis
          </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 min-h-0 overflow-hidden">
          {/* Details Tab - No ScrollArea to allow horizontal scroll */}
          {activeTab === 'details' ? (
            <div className="h-full overflow-y-auto">
              <div className="px-6 pt-2 pb-6">
                <DetailsTab runs={runsWithTimeseries} availableMetrics={availableMetrics} />
              </div>
            </div>
          ) : (
            <ScrollArea className="h-full w-full">
              <div className="px-6 pt-2 pb-6">
                {/* Overview Tab */}
                {activeTab === 'overview' && (
                  <OverviewTab runs={runsWithTimeseries} availableMetrics={availableMetrics} />
                )}

                {/* Training Tab */}
                {activeTab === 'training' && (
                  <TrainingTab
                    runs={runsWithTimeseries}
                    availableMetrics={availableMetrics}
                    hyperparamKeys={hyperparamKeys}
                    onPointClick={openRunDetail}
                  />
                )}

                {/* Analysis Tab */}
                {activeTab === 'analysis' && (
                  <AnalysisTab
                    runs={runsWithTimeseries}
                    availableMetrics={availableMetrics}
                    hyperparamKeys={hyperparamKeys}
                    onRunSelect={openRunDetail}
                  />
                )}
              </div>
            </ScrollArea>
          )}
        </div>
      </DialogContent>

      {/* Run Detail Modal for drill-down */}
      <RunDetailModal
        open={!!selectedRun}
        onClose={closeRunDetail}
        run={selectedRun}
        allRuns={runs}
      />
    </Dialog>
  );
}
