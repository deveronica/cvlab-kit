/**
 * ExperimentsView - Unified Queue + Results View
 *
 * Single view for all experiment management:
 * - Live queue status (pending, running)
 * - Completed/failed results with full details
 * - Real-time updates via SSE
 * - Configuration viewer, terminal logs, errors
 */

import React, { useState, useMemo, memo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useExperimentsStore, useExperimentStats, useConnectionHealth, useExperimentsByStatus } from '@/entities/experiment/model/experimentsStore';
import { useExperiments, useExperiment } from '@/shared/model/useExperiments';
import { useLogStream } from '@/shared/model/useLogStream';
import { useTheme } from '@/app/ui/ThemeContext';
import { useNavigationStore } from '@/shared/model/navigationStore';
import { Card, CardContent, CardHeader, CardTitle } from '@/shared/ui/card';
import { Badge } from '@/shared/ui/badge';
import { Button } from '@/shared/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/shared/ui/tabs';
import { Dialog, DialogContent, DialogTitle, DialogDescription } from '@/shared/ui/dialog';
import { AdvancedDataTable } from '@/shared/ui/advanced-data-table';
import { type ColumnDef, type ColumnSizingState } from '@tanstack/react-table';
import CodeMirror from '@uiw/react-codemirror';
import { yaml } from '@codemirror/lang-yaml';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import {
  AlertCircle,
  RefreshCw,
  Play,
  Pause,
  XCircle,
  CheckCircle,
  Clock,
  Loader2,
  WifiOff,
  X,
  Copy,
  Download,
} from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { devLog, devError } from '@/shared/lib/utils';

// =============================================================================
// Types
// =============================================================================

interface Experiment {
  experiment_uid: string;
  project: string | null;
  config_path: string;
  log_path?: string;
  error_log_path?: string;
  status: string;
  priority: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  assigned_device?: string | null;
  exit_code?: number | null;
  error_message?: string | null;
  last_indexed?: string;
  metadata?: {
    has_config?: boolean;
    [key: string]: unknown;
  };
}

// =============================================================================
// Status Configuration
// =============================================================================

const STATUS_CONFIG: Record<string, { emoji: string; color: string; label: string; animate?: boolean }> = {
  pending: { emoji: '🟡', color: 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/40 dark:text-yellow-100 dark:border-yellow-800', label: 'Pending' },
  queued: { emoji: '🔵', color: 'bg-indigo-100 text-indigo-800 border-indigo-200 dark:bg-indigo-900/40 dark:text-indigo-100 dark:border-indigo-800', label: 'Queued' },
  running: { emoji: '🟢', color: 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/40 dark:text-blue-100 dark:border-blue-800', label: 'Running', animate: true },
  paused: { emoji: '💤', color: 'bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-900/40 dark:text-purple-100 dark:border-purple-800', label: 'Paused' },
  completed: { emoji: '✅', color: 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/40 dark:text-green-100 dark:border-green-800', label: 'Completed' },
  failed: { emoji: '🔴', color: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/40 dark:text-red-100 dark:border-red-800', label: 'Failed' },
  cancelled: { emoji: '🛑', color: 'bg-red-200 text-red-900 border-red-300 dark:bg-red-900/60 dark:text-red-100 dark:border-red-800', label: 'Cancelled' },
};

// Status arrays as constants to prevent re-creation on each render
const QUEUE_STATUSES = ['pending', 'queued', 'running', 'paused'] as const;
const RESULTS_STATUSES = ['completed', 'failed', 'cancelled'] as const;

// =============================================================================
// API Functions
// =============================================================================

const fetchJobConfig = async (experimentUid: string): Promise<string> => {
  const response = await fetch(`/api/queue/experiments/${experimentUid}/config`);
  if (!response.ok) return 'Config not available';
  const data = await response.json();
  return data.data?.content || data.content || '';
};

// =============================================================================
// Utility Functions
// =============================================================================

const formatDuration = (started: string | null | undefined, finished: string | null | undefined): string => {
  if (!started) return '-';
  const start = new Date(started).getTime();
  const end = finished ? new Date(finished).getTime() : new Date().getTime();
  const durationMs = end - start;
  const seconds = Math.floor(durationMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
};

const formatDateTime = (dateTime: string | null | undefined): string => {
  if (!dateTime) return '-';
  const date = new Date(dateTime);
  return `${date.toLocaleDateString()} ${date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit'
  })}`;
};

const calculateTextWidth = (text: string, font: string = '14px system-ui'): number => {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return text.length * 8;
  context.font = font;
  return context.measureText(text).width;
};

const calculateOptimalColumnWidths = (
  data: Experiment[],
  columns: { id: string; header: string; minSize: number; maxSize: number; font?: string }[]
): ColumnSizingState => {
  const sizing: ColumnSizingState = {};

  columns.forEach((col) => {
    let maxWidth = calculateTextWidth(col.header, 'bold 14px system-ui') + 40;

    data.forEach((row) => {
      let cellValue = '';

      switch (col.id) {
        case 'experiment_uid':
          cellValue = row.experiment_uid;
          break;
        case 'project':
          cellValue = row.project || 'Not Logged';
          break;
        case 'status':
          cellValue = row.status;
          break;
        case 'priority':
          cellValue = row.priority || 'normal';
          break;
        case 'started_at':
          cellValue = formatDateTime(row.started_at);
          break;
        case 'completed_at':
          cellValue = formatDateTime(row.completed_at);
          break;
        case 'duration':
          cellValue = formatDuration(row.started_at, row.completed_at);
          break;
        case 'assigned_device':
          cellValue = row.assigned_device || '-';
          break;
      }

      const cellWidth = calculateTextWidth(cellValue, col.font || '14px system-ui') + 32;
      maxWidth = Math.max(maxWidth, cellWidth);
    });

    const optimalWidth = Math.max(col.minSize, Math.min(maxWidth, col.maxSize));
    sizing[col.id] = optimalWidth;
  });

  return sizing;
};

// =============================================================================
// Sub-Components
// =============================================================================

const StatusBadge = memo(({ status }: { status: string }) => {
  const config = STATUS_CONFIG[status.toLowerCase()] || STATUS_CONFIG.failed;
  return (
    <Badge className={`${config.color} flex items-center gap-1`}>
      {config.animate ? (
        <Loader2 className="w-3 h-3 animate-spin" />
      ) : (
        <span>{config.emoji}</span>
      )}
      {config.label}
    </Badge>
  );
});
StatusBadge.displayName = 'StatusBadge';

interface ConnectionStatusProps {
  onRefresh: () => void;
}

const ConnectionStatus = memo(({ onRefresh }: ConnectionStatusProps) => {
  const { isHealthy, isRetrying, hasError, error, retryCount, canRetry } = useConnectionHealth();

  if (isHealthy) return null;

  return (
    <div className={cn(
      'flex items-center gap-2 px-3 py-2 rounded-lg text-sm',
      hasError ? 'bg-red-100 text-red-900 dark:bg-red-950/40 dark:text-red-100' : 'bg-yellow-100 text-yellow-900 dark:bg-yellow-950/40 dark:text-yellow-100'
    )}>
      <WifiOff className="w-4 h-4" />
      <span>
        {hasError
          ? `Connection failed: ${error}`
          : `Reconnecting... (${retryCount}/3)`}
      </span>
      {hasError && canRetry && (
        <Button variant="ghost" size="sm" onClick={onRefresh}>
          <RefreshCw className="w-3 h-3 mr-1" />
          Retry
        </Button>
      )}
    </div>
  );
});
ConnectionStatus.displayName = 'ConnectionStatus';

const StatsCard = memo(({ label, value, color }: { label: string; value: number; color?: string }) => (
  <Card className="flex-1 min-w-[100px]">
    <CardContent className="p-4 text-center">
      <div className={cn('text-2xl font-bold', color)}>{value}</div>
    <div className="text-xs text-foreground">{label}</div>
    </CardContent>
  </Card>
));
StatsCard.displayName = 'StatsCard';

// =============================================================================
// Experiment Detail Modal
// =============================================================================

interface ExperimentDetailModalProps {
  experiment: Experiment | null;
  onClose: () => void;
  onCancel?: (uid: string) => void;
  onPause?: (uid: string) => void;
  onResume?: (uid: string) => void;
}

const ExperimentDetailModal = memo(({
  experiment,
  onClose,
  onCancel,
  onPause,
  onResume,
}: ExperimentDetailModalProps) => {
  const [activeTab, setActiveTab] = useState('configuration');
  const { theme } = useTheme();
  const { navigateToProject } = useNavigationStore();

  const isRunning = experiment?.status?.toLowerCase() === 'running';

  // Config query
  const { data: jobConfig, isLoading: isLoadingConfig } = useQuery({
    queryKey: ['jobConfig', experiment?.experiment_uid],
    queryFn: () => fetchJobConfig(experiment!.experiment_uid),
    enabled: !!experiment && activeTab === 'configuration',
  });

  // Real-time log streaming via WebSocket (only for running jobs)
  const { logs: streamLogs, isConnected } = useLogStream(
    experiment && activeTab === 'logs' && isRunning ? experiment.experiment_uid : null
  );

  // Static logs for completed/failed jobs
  const { data: staticLogs, isLoading: isLoadingLogs } = useQuery({
    queryKey: ['jobLogs', experiment?.experiment_uid],
    queryFn: async () => {
      const response = await fetch(`/api/queue/experiments/${experiment!.experiment_uid}/logs`);
      if (!response.ok) return [];
      const data = await response.json();
      return (data.data?.content || data.content || '').split('\n');
    },
    enabled: !!experiment && activeTab === 'logs' && !isRunning,
  });

  // Static error logs (stderr only)
  const { data: staticErrorLogs, isLoading: isLoadingErrorLogs } = useQuery({
    queryKey: ['jobErrorLogs', experiment?.experiment_uid],
    queryFn: async () => {
      const response = await fetch(`/api/queue/experiments/${experiment!.experiment_uid}/stderr`);
      if (!response.ok) return [];
      const data = await response.json();
      return (data.data?.content || data.content || '').split('\n');
    },
    enabled: !!experiment && activeTab === 'errors',
    retry: 1,
    staleTime: 0,
  });

  const displayLogs = isRunning ? streamLogs : (staticLogs || []);
  const displayErrorLogs = staticErrorLogs || [];

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const downloadFile = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!experiment) return null;

  return (
    <Dialog open={!!experiment} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="w-full max-w-4xl h-[90vh] flex flex-col p-0" hideCloseButton>
        {/* Fixed Header */}
        <div className="flex-shrink-0 border-b p-4 sm:p-6">
          <div className="flex items-start justify-between gap-4 mb-2">
            <DialogTitle className="text-lg sm:text-xl">Experiment Details</DialogTitle>
            <div className="flex items-center gap-2">
              {/* Action buttons for active experiments */}
              {(experiment.status === 'running' || experiment.status === 'pending' || experiment.status === 'queued' || experiment.status === 'paused') && (
                <>
                  {experiment.status === 'running' && onPause && (
                    <Button variant="outline" size="sm" onClick={() => onPause(experiment.experiment_uid)}>
                      <Pause className="w-4 h-4 mr-1" />
                      Pause
                    </Button>
                  )}
                  {experiment.status === 'paused' && onResume && (
                    <Button variant="outline" size="sm" onClick={() => onResume(experiment.experiment_uid)}>
                      <Play className="w-4 h-4 mr-1" />
                      Resume
                    </Button>
                  )}
                  {onCancel && (
                    <Button variant="outline" size="sm" className="text-red-600" onClick={() => onCancel(experiment.experiment_uid)}>
                      <XCircle className="w-4 h-4 mr-1" />
                      Cancel
                    </Button>
                  )}
                </>
              )}
              <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close">
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <DialogDescription className="text-xs sm:text-sm flex items-center gap-2">
            <span className="font-mono">{experiment.experiment_uid}</span>
            <span>•</span>
            <StatusBadge status={experiment.status} />
            {experiment.project && (
              <>
                <span>•</span>
                <Badge
                  variant="outline"
                  className="text-xs cursor-pointer hover:bg-primary/10"
                  onClick={() => navigateToProject(experiment.project!)}
                >
                  {experiment.project}
                </Badge>
              </>
            )}
          </DialogDescription>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
          <TabsList className="flex-shrink-0 mx-4 sm:mx-6 mt-4">
            <TabsTrigger value="configuration">Configuration</TabsTrigger>
            <TabsTrigger value="logs">Terminal Logs</TabsTrigger>
            <TabsTrigger value="errors">Terminal Errors</TabsTrigger>
          </TabsList>

          {/* Configuration Tab */}
          <TabsContent value="configuration" className="flex-1 overflow-y-auto min-h-0 mt-0">
            <div className="p-4 sm:p-6">
              <div className="border rounded-lg overflow-hidden">
                <div className="bg-background border-b p-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Configuration Used</h3>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => copyToClipboard(jobConfig || '')}
                        disabled={!jobConfig || isLoadingConfig}
                      >
                        <Copy className="h-4 w-4 mr-1" />
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadFile(jobConfig || '', `${experiment.experiment_uid}_config.yaml`)}
                        disabled={!jobConfig || isLoadingConfig}
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </Button>
                    </div>
                  </div>
                </div>
                <div className="h-[calc(90vh-400px)] rounded-b-lg p-3 overflow-hidden">
                  {isLoadingConfig ? (
                    <div className="flex items-center justify-center h-full">
                      <div className="text-center">
                        <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2" />
                <p className="text-foreground">Loading configuration...</p>
                      </div>
                    </div>
                  ) : jobConfig ? (
                    <div className="h-full border rounded-lg">
                      <CodeMirror
                        value={jobConfig}
                        extensions={[yaml()]}
                        editable={false}
                        height="100%"
                        className="h-full rounded-lg overflow-auto"
                        theme={theme === 'dark' ? vscodeDark : 'light'}
                        basicSetup={{
                          lineNumbers: true,
                          highlightActiveLineGutter: false,
                          highlightActiveLine: false,
                          foldGutter: false,
                        }}
                      />
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full">
              <p className="text-foreground">Configuration not available</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Logs Tab */}
          <TabsContent value="logs" className="flex-1 overflow-y-auto min-h-0 mt-0">
            <div className="p-4 sm:p-6">
              <div className="border rounded-lg overflow-hidden">
                <div className="bg-background border-b p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <h3 className="text-lg font-semibold">Terminal Logs</h3>
                      {isRunning && isConnected && (
                        <span className="flex items-center gap-1.5 text-xs text-green-600">
                          <span className="inline-block w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                          Live
                        </span>
                      )}
                      {!isRunning && (
                        <span className="flex items-center gap-1.5 text-xs text-gray-500">
                          <span className="inline-block w-2 h-2 bg-gray-400 rounded-full" />
                          Static
                        </span>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => copyToClipboard(displayLogs.join('\n'))}
                        disabled={displayLogs.length === 0}
                      >
                        <Copy className="h-4 w-4 mr-1" />
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadFile(displayLogs.join('\n'), `${experiment.experiment_uid}_logs.txt`)}
                        disabled={displayLogs.length === 0}
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </Button>
                    </div>
                  </div>
                </div>
                <div className="h-[calc(90vh-400px)] bg-black dark:bg-gray-950 rounded-b-lg p-3 overflow-y-auto">
                  <div className="text-green-400 font-mono text-xs leading-relaxed">
                    {isLoadingLogs && !isRunning ? (
                      <div className="text-gray-500 flex items-center justify-center h-full">
                        <div className="text-center">
                          <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2 text-green-500" />
                          Loading logs...
                        </div>
                      </div>
                    ) : displayLogs.length === 0 ? (
                      <div className="text-gray-500 flex items-center justify-center h-full">
                        {isRunning ? (isConnected ? 'Waiting for logs...' : 'Connecting...') : 'No logs available'}
                      </div>
                    ) : (
                      displayLogs.map((line: string, index: number) => (
                        <div key={index} className="whitespace-pre-wrap">{line}</div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Errors Tab */}
          <TabsContent value="errors" className="flex-1 overflow-y-auto min-h-0 mt-0">
            <div className="p-4 sm:p-6">
              <div className="border rounded-lg overflow-hidden">
                <div className="bg-background border-b p-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Terminal Errors (STDERR)</h3>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => copyToClipboard(displayErrorLogs.join('\n'))}
                        disabled={displayErrorLogs.length === 0}
                      >
                        <Copy className="h-4 w-4 mr-1" />
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadFile(displayErrorLogs.join('\n'), `${experiment.experiment_uid}_errors.txt`)}
                        disabled={displayErrorLogs.length === 0}
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </Button>
                    </div>
                  </div>
                </div>
                <div className="h-[calc(90vh-400px)] bg-black dark:bg-gray-950 rounded-b-lg p-3 overflow-y-auto">
                  <div className="text-red-400 font-mono text-xs leading-relaxed">
                    {isLoadingErrorLogs ? (
                      <div className="text-gray-500 flex items-center justify-center h-full">
                        <div className="text-center">
                          <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2 text-red-500" />
                          Loading error logs...
                        </div>
                      </div>
                    ) : displayErrorLogs.length === 0 ? (
                      <div className="text-gray-500 flex items-center justify-center h-full">
                        No error logs available
                      </div>
                    ) : (
                      displayErrorLogs.map((line: string, index: number) => (
                        <div key={index} className="whitespace-pre-wrap">{line}</div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
});
ExperimentDetailModal.displayName = 'ExperimentDetailModal';

// =============================================================================
// Main Component
// =============================================================================

export const ExperimentsView = memo(() => {
  const [activeTab, setActiveTab] = useState<'queue' | 'results'>('queue');
  const [selectedUid, setSelectedUid] = useState<string | null>(null);
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>({});
  const { theme } = useTheme();
  const { navigateToProject } = useNavigationStore();

  // Unified experiments hook (handles all fetching, SSE, retry)
  const { cancelExperiment, pauseExperiment, resumeExperiment, refresh } = useExperiments();

  // Get selected experiment from store
  const selectedExperiment = useExperiment(selectedUid);

  // Stats from store
  const stats = useExperimentStats();

  // Filtered experiments
  const queueExperiments = useExperimentsByStatus(QUEUE_STATUSES);
  const resultsExperiments = useExperimentsByStatus(RESULTS_STATUSES);

  // Loading state
  const isLoading = useExperimentsStore((state) => state.loadingStates.list);

  // Column metadata for auto-sizing
  const columnMetadata = useMemo(() => [
    { id: 'experiment_uid', header: 'Experiment ID', minSize: 130, maxSize: 200, font: 'mono 14px system-ui' },
    { id: 'project', header: 'Project', minSize: 100, maxSize: 180 },
    { id: 'status', header: 'Status', minSize: 110, maxSize: 140 },
    { id: 'priority', header: 'Priority', minSize: 80, maxSize: 100 },
    { id: 'started_at', header: 'Started At', minSize: 150, maxSize: 180 },
    { id: 'duration', header: 'Duration', minSize: 70, maxSize: 100 },
    { id: 'assigned_device', header: 'Device', minSize: 150, maxSize: 200, font: 'mono 14px system-ui' },
  ], []);

  // Calculate optimal column widths when data changes
  useEffect(() => {
    const allExperiments = [...queueExperiments, ...resultsExperiments];
    if (allExperiments.length > 0) {
      const optimalSizing = calculateOptimalColumnWidths(allExperiments as Experiment[], columnMetadata);
      setColumnSizing(optimalSizing);
    }
  }, [queueExperiments, resultsExperiments, columnMetadata]);

  // Table columns
  const columns = useMemo<ColumnDef<Experiment>[]>(() => [
    {
      accessorKey: 'experiment_uid',
      header: 'Experiment ID',
      minSize: 130,
      maxSize: 200,
      enableResizing: true,
      cell: ({ getValue, row }) => (
        <div
          className="font-mono text-sm cursor-pointer hover:underline text-blue-600 whitespace-nowrap"
          onClick={(e) => {
            e.stopPropagation();
            setSelectedUid(row.original.experiment_uid);
          }}
        >
          {getValue<string>()}
        </div>
      ),
    },
    {
      accessorKey: 'project',
      header: 'Project',
      minSize: 100,
      maxSize: 180,
      enableResizing: true,
      cell: ({ getValue, row }) => {
        const project = getValue<string>();
        const hasConfig = row.original.metadata?.has_config;

        if (hasConfig === false) {
          return (
            <Badge variant="secondary" className="text-xs cursor-default opacity-60">
              Legacy Job
            </Badge>
          );
        }

        const isProjectMissing = !project || project === 'default_project' || project.trim() === '';

        if (isProjectMissing) {
          return (
            <Badge variant="secondary" className="text-xs cursor-default opacity-60">
              Not Logged
            </Badge>
          );
        }

        return (
          <Badge
            variant="outline"
            className="text-xs cursor-pointer hover:bg-primary/10"
            onClick={(e) => {
              e.stopPropagation();
              navigateToProject(project);
            }}
          >
            {project}
          </Badge>
        );
      },
    },
    {
      accessorKey: 'status',
      header: 'Status',
      minSize: 110,
      maxSize: 300,
      enableResizing: true,
      cell: ({ getValue, row }) => {
        const status = getValue<string>();
        const errorMessage = row.original.error_message;
        return (
          <div className="flex flex-col gap-1">
            <StatusBadge status={status} />
            {errorMessage && status?.toLowerCase() === 'failed' && (
              <span className="text-xs text-red-600 truncate max-w-[250px]" title={errorMessage}>
                {errorMessage}
              </span>
            )}
          </div>
        );
      },
    },
    {
      accessorKey: 'priority',
      header: 'Priority',
      minSize: 80,
      maxSize: 100,
      enableResizing: true,
      cell: ({ getValue }) => {
        const priority = getValue<string>()?.toLowerCase();
        const colors: Record<string, string> = {
          urgent: 'bg-red-100 text-red-800',
          high: 'bg-orange-100 text-orange-800',
          normal: 'bg-blue-100 text-blue-800',
          low: 'bg-gray-100 text-gray-800',
        };
        return (
          <Badge className={`${colors[priority] || colors.normal} text-xs whitespace-nowrap`}>
            {getValue<string>()}
          </Badge>
        );
      },
    },
    {
      accessorKey: 'started_at',
      header: 'Started At',
      minSize: 150,
      maxSize: 180,
      enableResizing: true,
      cell: ({ getValue }) => (
        <span className="text-sm whitespace-nowrap">{formatDateTime(getValue<string | null>())}</span>
      ),
    },
    {
      id: 'duration',
      header: 'Duration',
      minSize: 70,
      maxSize: 100,
      enableResizing: true,
      accessorFn: (row) => formatDuration(row.started_at, row.completed_at),
      cell: ({ getValue }) => <span className="text-sm whitespace-nowrap">{getValue<string>()}</span>,
    },
    {
      accessorKey: 'assigned_device',
      header: 'Device',
      minSize: 150,
      maxSize: 200,
      enableResizing: true,
      cell: ({ getValue }) => (
        <span className="text-sm font-mono whitespace-nowrap">{getValue<string>() || '-'}</span>
      ),
    },
    {
      id: 'actions',
      header: '',
      size: 100,
      enableResizing: false,
      cell: ({ row }) => {
        const exp = row.original;
        const isActive = ['running', 'pending', 'queued', 'paused'].includes(exp.status);

        if (!isActive) return null;

        return (
          <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
            {exp.status === 'running' && (
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => pauseExperiment(exp.experiment_uid)}>
                <Pause className="w-3 h-3" />
              </Button>
            )}
            {exp.status === 'paused' && (
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => resumeExperiment(exp.experiment_uid)}>
                <Play className="w-3 h-3" />
              </Button>
            )}
            <Button variant="ghost" size="icon" className="h-7 w-7 text-red-600" onClick={() => cancelExperiment(exp.experiment_uid)}>
              <XCircle className="w-3 h-3" />
            </Button>
          </div>
        );
      },
    },
  ], [navigateToProject, pauseExperiment, resumeExperiment, cancelExperiment]);

  // Sorted lists
  const sortedQueue = useMemo(() =>
    [...queueExperiments].sort((a, b) => {
      if (a.status === 'running' && b.status !== 'running') return -1;
      if (b.status === 'running' && a.status !== 'running') return 1;
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    }) as Experiment[],
    [queueExperiments]
  );

  const sortedResults = useMemo(() =>
    [...resultsExperiments].sort((a, b) =>
      new Date(b.completed_at || b.created_at).getTime() -
      new Date(a.completed_at || a.created_at).getTime()
    ) as Experiment[],
    [resultsExperiments]
  );

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Experiments</h1>
          <p className="text-sm text-foreground">
            Queue management and experiment results
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refresh} disabled={isLoading}>
          <RefreshCw className={cn('w-4 h-4 mr-1', isLoading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {/* Connection Status Banner */}
      <ConnectionStatus onRefresh={refresh} />

      {/* Stats */}
      <div className="flex gap-2 flex-wrap">
          <StatsCard label="Running" value={stats.running} color="text-foreground" />
          <StatsCard label="Pending" value={stats.pending + stats.queued} color="text-foreground" />
          <StatsCard label="Completed" value={stats.completed} color="text-foreground" />
          <StatsCard label="Failed" value={stats.failed} color="text-foreground" />
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'queue' | 'results')}>
        <TabsList>
          <TabsTrigger value="queue" className="gap-1">
            Queue
            {sortedQueue.length > 0 && (
              <Badge variant="secondary" className="ml-1">{sortedQueue.length}</Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="results" className="gap-1">
            Results
            {sortedResults.length > 0 && (
              <Badge variant="secondary" className="ml-1">{sortedResults.length}</Badge>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="queue" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Active Queue</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading && sortedQueue.length === 0 ? (
                <div className="flex items-center justify-center py-12 text-foreground">
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                  Loading...
                </div>
              ) : sortedQueue.length === 0 ? (
                <div className="text-center py-12 text-foreground">
                  <Clock className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No experiments in queue</p>
                  <p className="text-sm">Add experiments from the Execute tab</p>
                </div>
              ) : (
                <AdvancedDataTable
                  columns={columns}
                  data={sortedQueue}
                  enableSorting
                  enableFiltering
                  enablePagination
                  pageSize={10}
                  initialSizing={columnSizing}
                  tableId="queue-experiments-table"
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Execution Timeline</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading && sortedResults.length === 0 ? (
                <div className="flex items-center justify-center py-12 text-foreground">
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                  Loading...
                </div>
              ) : sortedResults.length === 0 ? (
                <div className="text-center py-12 text-foreground">
                  <CheckCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No completed experiments yet</p>
                </div>
              ) : (
                <AdvancedDataTable
                  columns={columns}
                  data={sortedResults}
                  enableSorting
                  enableFiltering
                  enablePagination
                  pageSize={20}
                  initialSizing={columnSizing}
                  tableId="results-experiments-table"
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Experiment Detail Modal */}
      <ExperimentDetailModal
        experiment={selectedExperiment as Experiment | null}
        onClose={() => setSelectedUid(null)}
        onCancel={cancelExperiment}
        onPause={pauseExperiment}
        onResume={resumeExperiment}
      />
    </div>
  );
});

ExperimentsView.displayName = 'ExperimentsView';

export default ExperimentsView;
