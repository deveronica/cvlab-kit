import React from "react";
import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useProjectsView } from '../../hooks/useProjectsView';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Switch } from '../ui/switch';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/tabs';
import { ExportMenu } from '../ui/export-menu';
import { AdvancedFilter } from '../ui/advanced-filter';
import { MetricsComparisonModal } from '../modals/MetricsComparisonModal';
import { RunDetailModal } from '../modals/RunDetailModal';
import { BestRunCard } from '../ui/best-run-card';
import { SavedViewsMenu } from '../ui/saved-views-menu';
import { SummaryDashboard } from '../ui/summary-dashboard';
import { RunTimeline } from '../ui/run-timeline';
import { RunStatusBadge } from '../ui/run-status-badge';
import { ProjectStatsCard } from '../ui/project-stats-card';
import { TopRunsTable } from '../ui/top-runs-table';
import { HyperparameterInsights } from '../ui/hyperparameter-insights';
import { HyperparamCorrelationCard } from '../ui/hyperparam-correlation-card';
import { QuickRecommendations } from '../ui/quick-recommendations';
import type { SavedView } from '../../lib/saved-views';
import { Input } from '../ui/input';
import { ChartsView } from './ChartsView';
import { AdvancedView } from './AdvancedView';
import { MetricSelector } from '../ui/metric-selector';
import { AGGridProjectsTable } from '../tables/AGGridProjectsTable';
import { ColumnSuggestionBanner } from '../ui/column-suggestion-banner';
import { ColumnMappingDialog } from '../dialogs/ColumnMappingDialog';
import { ColumnManager } from '../ui/column-manager';
import { ErrorBoundary } from '../ui/error-boundary';
import { BarChart3, Layers, Search, X, Clock, CheckCircle, Activity, Sparkles, RefreshCw, Download, AlignJustify, Settings2, Minus, Equal } from 'lucide-react';
import { adaptExperiments } from '../../lib/dataAdapter';

export function ProjectsView() {
  const { projectName: activeProject } = useParams<{ projectName: string }>();
  const navigate = useNavigate();
  const {
    _projects,
    projectSummaries,
    experimentsData,
    hyperparamColumns,
    metricColumns,
    getBestRun,
    projectsLoading,
    flattenParams, setFlattenParams,
    diffMode, setDiffMode,
    selectedExperiments, setSelectedExperiments,
    activeView, setActiveView,
    isComparisonModalOpen, setIsComparisonModalOpen,
    visibleHyperparams, setVisibleHyperparams,
    visibleMetrics, setVisibleMetrics,
    pinnedLeftHyperparams, setPinnedLeftHyperparams,
    pinnedRightMetrics, setPinnedRightMetrics,
    hyperparamOrder, setHyperparamOrder,
    metricOrder, setMetricOrder,
    clearSelection,
    exportRuns,
    getSelectedExperimentData,
    fetchRunDetails,
    chartsRef,
    _filters, _setFilters,
    selectedMetric, setSelectedMetric,
    metricDirection, setMetricDirection,
    selectedRunDetail, setSelectedRunDetail,
    // Column mapping
    columnMappings,
    columnSuggestions,
    isColumnMappingDialogOpen,
    setIsColumnMappingDialogOpen,
    isLoadingColumnMappings,
    _isGeneratingSuggestions,
    handleAcceptSuggestion,
    handleRejectSuggestion,
    handleCreateMapping,
    handleUpdateMapping,
    handleDeleteMapping,
    // Outlier detection
    cellOutliers,
    highlightedRun,
    setHighlightedRun,
  } = useProjectsView();

  // Local state for table controls
  const [quickFilterText, setQuickFilterText] = useState('');
  const [rowMode, setRowMode] = useState<'1-line' | '2-line' | '3-line'>('2-line');
  const [isPinningEnabled, setIsPinningEnabled] = useState(true);
  const gridRef = useRef<any>(null);

  // Scroll to top on tab change
  useEffect(() => {
    window.scrollTo({
      top: 0,
      left: 0,
      behavior: 'smooth',
    });
  }, [activeView]);

  // Table control handlers - Cycle through row modes: 1-line → 2-line → 3-line → 1-line
  const handleToggleMaximize = useCallback(() => {
    setRowMode(prev => {
      if (prev === '1-line') return '2-line';
      if (prev === '2-line') return '3-line';
      return '1-line';
    });
  }, []);

  const handleResetFilters = useCallback(() => {
    setQuickFilterText('');
    if (gridRef.current?.api) {
      gridRef.current.api.setFilterModel(null);
      const allColumns = gridRef.current.api.getColumns();
      if (allColumns) {
        gridRef.current.api.applyColumnState({
          state: allColumns.map((col: any) => ({ colId: col.getId(), sort: null })),
          defaultState: { sort: null },
        });
      }
    }
  }, []);

  const handleExportCSV = useCallback(() => {
    if (gridRef.current?.api) {
      gridRef.current.api.exportDataAsCsv({
        fileName: `experiments_${new Date().toISOString().split('T')[0]}.csv`,
        columnKeys: undefined,
      });
    }
  }, []);

  // Adapt experiments data for components that expect Run format
  const adaptedRuns = useMemo(() => {
    return adaptExperiments(experimentsData, activeProject || '');
  }, [experimentsData, activeProject]);

  if (!activeProject) {
    return (
      <div className="space-y-4 pb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Projects</h1>
            <p className="text-muted-foreground mt-1">Browse and manage your experiment _projects</p>
          </div>
        </div>
        {projectsLoading ? (
          <div className="text-center py-12">Loading _projects...</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {projectSummaries.map((project) => (
              <Card
                key={project.name}
                expandable
                onExpand={() => navigate(`/projects/${project.name}`)}
              >
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <Layers className="h-5 w-5 text-primary" />
                      <CardTitle className="text-xl">{project.name}</CardTitle>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-0.5">
                        <div className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                          <Activity className="h-3 w-3" />
                          Total Runs
                        </div>
                        <div className="text-2xl font-bold">{project.run_count}</div>
                      </div>
                      <div className="space-y-0.5">
                        <div className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                          <CheckCircle className="h-3 w-3" />
                          Completed
                        </div>
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {project.completed_count}
                        </div>
                      </div>
                    </div>
                    <div className="pt-2 border-t space-y-1.5">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          Avg Duration
                        </span>
                        <span className="font-medium">{project.avg_duration}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Last Activity</span>
                        <span className="font-medium">{project.last_activity}</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      <div className="space-y-4 pb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" onClick={() => navigate('/_projects')}>
              ← Back to Projects
            </Button>
            <div>
              <h1 className="text-3xl font-bold">{activeProject}</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <SavedViewsMenu
              projectName={activeProject}
              currentState={{ selectedRuns: selectedExperiments, flatten: flattenParams, diffOnly: diffMode, visibleHyperparams: Array.from(visibleHyperparams), visibleMetrics: Array.from(visibleMetrics), pinnedLeftHyperparams: Array.from(pinnedLeftHyperparams), pinnedRightMetrics: Array.from(pinnedRightMetrics) }}
              onLoadView={(view: SavedView) => {
                setSelectedExperiments(view.state.selectedRuns);
                setFlattenParams(view.state.flatten);
                setDiffMode(view.state.diffOnly);
                setVisibleHyperparams(new Set(view.state.visibleHyperparams));
                setVisibleMetrics(new Set(view.state.visibleMetrics));
                setPinnedLeftHyperparams(new Set(view.state.pinnedLeftHyperparams));
                setPinnedRightMetrics(new Set(view.state.pinnedRightMetrics));

                // Auto-open comparison modal if 2+ runs are selected
                if (view.state.selectedRuns.length >= 2) {
                  setTimeout(() => setIsComparisonModalOpen(true), 100);
                }
              }}
            />
            <ExportMenu onExport={(format) => exportRuns(format)} formats={['json', 'markdown', 'png']} />
          </div>
        </div>

        {/* Tab Navigation */}
        <Tabs value={activeView} onValueChange={setActiveView}>
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="table">Table</TabsTrigger>
            <TabsTrigger value="charts">Charts</TabsTrigger>
            <TabsTrigger value="advanced" className="gap-1.5">
              <Sparkles className="h-4 w-4" />
              Advanced
              <Badge variant="secondary" className="text-[9px] h-4 px-1 ml-0.5">
                AI
              </Badge>
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-2">
            <ErrorBoundary level="section">
              {/* Run Timeline - Execution Schedule at Top */}
              <ErrorBoundary level="component">
                <RunTimeline
                  runs={adaptedRuns}
                  onRunClick={(runName) => {
                    const run = experimentsData.find(r => r.run_name === runName);
                    if (run) setSelectedRunDetail(run);
                  }}
                />
              </ErrorBoundary>

              {/* Metric Selection */}
              {metricColumns.length > 0 && (
                <MetricSelector
                  metrics={metricColumns}
                  selectedMetric={selectedMetric}
                  onMetricChange={setSelectedMetric}
                  direction={metricDirection}
                  onDirectionChange={setMetricDirection}
                />
              )}

              {/* Summary Dashboard */}
              <SummaryDashboard
                runs={experimentsData}
                keyMetric={selectedMetric}
                metricDirection={metricDirection}
              />

              {/* Best Run & Top Runs Grid - Matching SummaryDashboard layout (1/3 + 2/3) */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-2 lg:items-stretch">
                {/* Best Run Card - 1/3 width (matching Run Status Distribution) */}
                {getBestRun && (
                  <ErrorBoundary level="component">
                    <BestRunCard
                      run={getBestRun}
                      keyMetric={selectedMetric}
                      secondaryMetrics={metricColumns.filter(m => m !== selectedMetric).slice(0, 3)}
                      onOpenDetail={(runId) => {
                        const run = experimentsData.find(exp => exp.run_name === runId);
                        if (run) fetchRunDetails(run);
                      }}
                    />
                  </ErrorBoundary>
                )}

                {/* Top Runs Comparison Table - 2/3 width (matching Recent Runs) */}
                <div className={getBestRun ? "lg:col-span-2" : "lg:col-span-3"}>
                  <TopRunsTable
                    runs={adaptedRuns}
                    keyMetric={selectedMetric}
                    metricDirection={metricDirection}
                    secondaryMetrics={metricColumns.filter(m => m !== selectedMetric).slice(0, 2)}
                    onRunClick={(runName) => {
                      const run = experimentsData.find(exp => exp.run_name === runName);
                      if (run) fetchRunDetails(run);
                    }}
                  />
                </div>
              </div>

              {/* AI Insights Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
                {/* Quick Recommendations */}
                <ErrorBoundary level="component">
                  <QuickRecommendations
                    project={activeProject}
                    onViewMore={() => setActiveView('advanced')}
                    onRunClick={(runName) => {
                      const run = experimentsData.find(exp => exp.run_name === runName);
                      if (run) fetchRunDetails(run);
                    }}
                  />
                </ErrorBoundary>

                {/* Hyperparameter Correlation Analysis */}
                <ErrorBoundary level="component">
                  <HyperparamCorrelationCard
                    project={activeProject || ''}
                    metric={selectedMetric}
                    useMax={metricDirection === 'maximize'}
                    maxDisplay={5}
                  />
                </ErrorBoundary>
              </div>
            </ErrorBoundary>
          </TabsContent>

          {/* Table Tab */}
          <TabsContent value="table" className="space-y-2">
            <ErrorBoundary level="section">
              {/* Column Suggestion Banner */}
              {columnSuggestions.length > 0 && (
                <ColumnSuggestionBanner
                  suggestions={columnSuggestions}
                  onAccept={handleAcceptSuggestion}
                  onReject={handleRejectSuggestion}
                  onOpenEditor={() => setIsColumnMappingDialogOpen(true)}
                  isLoading={isLoadingColumnMappings}
                />
              )}

              <Card className="p-0">
              <CardHeader className="pt-4 px-4 pb-2">
                {/* Single-Line Header Layout */}
                <div className="flex items-center gap-1.5 flex-wrap-none">
                  {/* Selection Badge */}
                  <Badge variant="outline" className="text-[10px] h-6 px-1.5">{selectedExperiments.length}</Badge>

                  {/* Compare Button */}
                  {selectedExperiments.length >= 2 && (
                    <Button variant="default" size="sm" className="h-6 text-[10px] px-2" onClick={() => setIsComparisonModalOpen(true)}>
                      <BarChart3 className="h-3 w-3 mr-1" />
                      Compare
                    </Button>
                  )}

                  {/* Clear Selection Button */}
                  {selectedExperiments.length > 0 && (
                    <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={clearSelection} aria-label="Clear selection">
                      <X className="h-3 w-3" />
                    </Button>
                  )}

                  {/* Highlighted Run Badge with Clear */}
                  {highlightedRun && (
                    <>
                      <div className="h-4 w-px bg-border" />
                      <Badge variant="default" className="text-[10px] h-6 px-1.5 bg-blue-600 text-white">
                        Highlighted: {highlightedRun.substring(0, 15)}{highlightedRun.length > 15 ? '...' : ''}
                      </Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => setHighlightedRun(null)}
                        aria-label="Clear highlight"
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </>
                  )}

                  <div className="h-4 w-px bg-border" />

                  {/* Flatten/Diff Toggles */}
                  <div className="flex items-center gap-0.5">
                    <label htmlFor="flatten-toggle" className="text-[9px] font-medium whitespace-nowrap">Flat</label>
                    <Switch id="flatten-toggle" checked={flattenParams} onCheckedChange={setFlattenParams} className="scale-75" />
                  </div>
                  <div className="flex items-center gap-0.5">
                    <label htmlFor="diff-toggle" className="text-[9px] font-medium">Diff</label>
                    <Switch id="diff-toggle" checked={diffMode} onCheckedChange={setDiffMode} className="scale-75" />
                  </div>

                  <div className="h-4 w-px bg-border" />

                  {/* Column Management */}
                  <ColumnManager
                    hyperparamColumns={hyperparamOrder.length > 0 ? hyperparamOrder : hyperparamColumns}
                    metricColumns={metricOrder.length > 0 ? metricOrder : metricColumns}
                    visibleHyperparams={visibleHyperparams}
                    visibleMetrics={visibleMetrics}
                    pinnedLeftHyperparams={pinnedLeftHyperparams}
                    pinnedRightMetrics={pinnedRightMetrics}
                    onVisibilityChange={(hyperparams, metrics) => {
                      setVisibleHyperparams(hyperparams);
                      setVisibleMetrics(metrics);
                    }}
                    onPinnedChange={(pinnedLeft, pinnedRight) => {
                      setPinnedLeftHyperparams(pinnedLeft);
                      setPinnedRightMetrics(pinnedRight);
                    }}
                    onPinningEnabledChange={setIsPinningEnabled}
                    onColumnOrderChange={(hyperparams, metrics) => {
                      setHyperparamOrder(hyperparams);
                      setMetricOrder(metrics);
                    }}
                  />

                  {/* Column Mapping Button */}
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-6 text-[10px] px-2"
                    onClick={() => setIsColumnMappingDialogOpen(true)}
                    title="Configure column mappings"
                  >
                    <Settings2 className="h-3 w-3 mr-0.5" />
                    Mapping
                  </Button>

                  <div className="h-4 w-px bg-border" />

                  {/* Table Action Buttons */}
                  <Button variant="outline" size="sm" className="h-6 text-[10px] px-2" onClick={handleToggleMaximize}>
                    {rowMode === '1-line' && <Minus className="h-3 w-3 mr-0.5" />}
                    {rowMode === '2-line' && <Equal className="h-3 w-3 mr-0.5" />}
                    {rowMode === '3-line' && <AlignJustify className="h-3 w-3 mr-0.5" />}
                    {rowMode === '1-line' ? '1-Line' : rowMode === '2-line' ? '2-Line' : '3-Line'}
                  </Button>
                  <Button variant="outline" size="sm" className="h-6 text-[10px] px-2" onClick={handleResetFilters}>
                    <RefreshCw className="h-3 w-3 mr-0.5" />
                    Reset
                  </Button>
                  <Button variant="outline" size="sm" className="h-6 text-[10px] px-2" onClick={handleExportCSV}>
                    <Download className="h-3 w-3 mr-0.5" />
                    CSV
                  </Button>

                  {/* Quick Search - Right aligned */}
                  <div className="flex items-center gap-1 flex-1 min-w-[100px] ml-auto">
                    <Search className="h-3 w-3 text-muted-foreground" />
                    <Input
                      type="text"
                      placeholder="Quick search..."
                      value={quickFilterText}
                      onChange={(e) => setQuickFilterText(e.target.value)}
                      className="h-6 text-[10px] px-2 flex-1"
                    />
                  </div>
                </div>
              </CardHeader>
              <CardContent className="px-3 pb-3">
              <AGGridProjectsTable
                data={experimentsData}
                hyperparamColumns={hyperparamOrder.length > 0 ? hyperparamOrder : hyperparamColumns}
                metricColumns={metricOrder.length > 0 ? metricOrder : metricColumns}
                flattenParams={flattenParams}
                activeProject={activeProject}
                diffMode={diffMode}
                visibleHyperparams={visibleHyperparams}
                visibleMetrics={visibleMetrics}
                pinnedLeftHyperparams={pinnedLeftHyperparams}
                pinnedRightMetrics={pinnedRightMetrics}
                isPinningEnabled={isPinningEnabled}
                selectedRows={selectedExperiments}
                onRowSelectionChange={(rows) => setSelectedExperiments(rows.map((r: any) => r.run_name))}
                onRowClick={(row) => fetchRunDetails(row)}
                cellOutliers={cellOutliers}
                highlightedRun={highlightedRun}
                quickFilterText={quickFilterText}
                onQuickFilterChange={setQuickFilterText}
                rowMode={rowMode}
                onToggleMaximize={handleToggleMaximize}
                onResetFilters={handleResetFilters}
                onExportCSV={handleExportCSV}
                gridRef={gridRef}
                onColumnStateChange={(pinnedLeft, pinnedRight) => {
                  setPinnedLeftHyperparams(pinnedLeft);
                  setPinnedRightMetrics(pinnedRight);
                }}
                onColumnOrderChange={(hyperparams, metrics) => {
                  setHyperparamOrder(hyperparams);
                  setMetricOrder(metrics);
                }}
              />
            </CardContent>
          </Card>
        </ErrorBoundary>
      </TabsContent>

      {/* Charts Tab */}
      <TabsContent value="charts">
        <ErrorBoundary level="section">
          <div ref={chartsRef}>
            <ChartsView experimentsData={experimentsData} selectedExperiments={selectedExperiments} hyperparamColumns={hyperparamColumns} metricColumns={metricColumns} activeProject={activeProject} />
          </div>
        </ErrorBoundary>
      </TabsContent>

      {/* Advanced Tab */}
      <TabsContent value="advanced">
        <ErrorBoundary level="section">
          <AdvancedView
            experimentsData={experimentsData}
            selectedExperiments={selectedExperiments}
            hyperparamColumns={hyperparamColumns}
            metricColumns={metricColumns}
            activeProject={activeProject}
            onHighlightRun={(runName) => {
              // Find the run data and open detail modal
              const run = experimentsData.find(exp => exp.run_name === runName);
              if (run) {
                setSelectedRunDetail(run);
              }
            }}
          />
        </ErrorBoundary>
      </TabsContent>
        </Tabs>
      </div>

      <MetricsComparisonModal
        open={isComparisonModalOpen}
        onClose={() => setIsComparisonModalOpen(false)}
        runs={getSelectedExperimentData()}
        projectName={activeProject}
      />

      <RunDetailModal
        open={!!selectedRunDetail}
        onClose={() => setSelectedRunDetail(null)}
        run={selectedRunDetail as any}
        allRuns={experimentsData}
        onCompareClick={(runName) => {
          const runToAdd = experimentsData.find(exp => exp.run_name === runName);
          if (runToAdd && !selectedExperiments.includes(runName)) {
            setSelectedExperiments([...selectedExperiments, runName]);
          }
          setIsComparisonModalOpen(true);
          setSelectedRunDetail(null);
        }}
      />

      {/* Column Mapping Dialog */}
      <ColumnMappingDialog
        open={isColumnMappingDialogOpen}
        onClose={() => setIsColumnMappingDialogOpen(false)}
        project={activeProject}
        suggestions={columnSuggestions}
        existingMappings={columnMappings}
        onAcceptSuggestion={handleAcceptSuggestion}
        onCreateMapping={handleCreateMapping}
        onUpdateMapping={handleUpdateMapping}
        onDeleteMapping={handleDeleteMapping}
        isLoading={isLoadingColumnMappings}
      />
    </>
  );
}
