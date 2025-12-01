import React from "react";
import { useState, useMemo, useEffect} from 'react';
import { useQuery } from '@tanstack/react-query';
import { queryKeys } from '@/lib/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Dialog, DialogContent, DialogTitle, DialogDescription } from '../ui/dialog';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/tabs';
import { Copy, Download, X } from 'lucide-react';
import CodeMirror from '@uiw/react-codemirror';
import { yaml } from '@codemirror/lang-yaml';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import { useTheme } from '../../contexts/ThemeContext';
import { AdvancedDataTable } from '../ui/advanced-data-table';
import { type ColumnDef, type ColumnSizingState } from '@tanstack/react-table';
import { useNavigationStore } from '@/store/navigationStore';
import { useLogStream } from '@/hooks/useLogStream';
import { devLog, devError } from '@/lib/dev-utils';

interface QueueJob {
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
  last_indexed?: string;
  metadata?: {
    has_config?: boolean;
    [key: string]: unknown;
  };
}

const fetchAllJobs = async (): Promise<QueueJob[]> => {
  const response = await fetch('/api/queue/experiments?limit=1000');
  if (!response.ok) throw new Error('Failed to fetch experiments');
  const data = await response.json();
  return data.data?.experiments || [];
};

const fetchJobConfig = async (experimentUid: string): Promise<string> => {
  const response = await fetch(`/api/queue/experiments/${experimentUid}/config`);
  if (!response.ok) return 'Config not available';
  const data = await response.json();
  return data.data?.content || data.content || '';
};

const statusConfig: Record<string, { emoji: string; color: string; label: string }> = {
  pending: { emoji: '🟡', color: 'bg-yellow-500/10 text-yellow-700 border-yellow-200', label: 'Pending' },
  queued: { emoji: '🔵', color: 'bg-indigo-500/10 text-indigo-700 border-indigo-200', label: 'Queued' },
  running: { emoji: '🟢', color: 'bg-blue-500/10 text-blue-700 border-blue-200', label: 'Running' },
  completed: { emoji: '✅', color: 'bg-green-500/10 text-green-700 border-green-200', label: 'Completed' },
  failed: { emoji: '🔴', color: 'bg-red-500/10 text-red-700 border-red-200', label: 'Failed' },
  cancelled: { emoji: '🛑', color: 'bg-red-600/10 text-red-800 border-red-300', label: 'Cancelled' },
  paused: { emoji: '💤', color: 'bg-purple-500/10 text-purple-700 border-purple-200', label: 'Paused' },
};

const StatusBadge = ({ status }: { status: string }) => {
  const config = statusConfig[status.toLowerCase()] || statusConfig.failed;
  return (
    <Badge className={`${config.color} flex items-center gap-1`}>
      <span>{config.emoji}</span>
      {config.label}
    </Badge>
  );
};

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

/**
 * Calculate optimal column width based on content
 * Uses a canvas to measure actual text width
 */
const calculateTextWidth = (text: string, font: string = '14px system-ui'): number => {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return text.length * 8; // Fallback
  context.font = font;
  return context.measureText(text).width;
};

/**
 * Calculate optimal widths for all columns based on data content
 */
const calculateOptimalColumnWidths = (
  data: QueueJob[],
  columns: { id: string; header: string; minSize: number; maxSize: number; font?: string }[]
): ColumnSizingState => {
  const sizing: ColumnSizingState = {};

  columns.forEach((col) => {
    let maxWidth = calculateTextWidth(col.header, 'bold 14px system-ui') + 40; // Header + padding

    // Measure content width for each row
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

      const cellWidth = calculateTextWidth(cellValue, col.font || '14px system-ui') + 32; // Content + padding
      maxWidth = Math.max(maxWidth, cellWidth);
    });

    // Clamp between min and max
    const optimalWidth = Math.max(col.minSize, Math.min(maxWidth, col.maxSize));
    sizing[col.id] = optimalWidth;
  });

  return sizing;
};

export function ResultsView() {
  const [selectedJob, setSelectedJob] = useState<QueueJob | null>(null);
  const [activeModalTab, setActiveModalTab] = useState('configuration');
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>({});
  const { theme } = useTheme();
  const { navigateToProject } = useNavigationStore();

  const { data: jobs = [], isLoading, error } = useQuery<QueueJob[]>({
    queryKey: queryKeys.queueJobs,
    queryFn: fetchAllJobs,
    refetchInterval: 10000,
  });

  const { data: jobConfig, isLoading: isLoadingConfig } = useQuery({
    queryKey: ['jobConfig', selectedJob?.experiment_uid],
    queryFn: () => fetchJobConfig(selectedJob!.experiment_uid),
    enabled: !!selectedJob && activeModalTab === 'configuration',
  });

  // Real-time log streaming via WebSocket (only for running jobs)
  const isRunning = selectedJob?.status?.toLowerCase() === 'running';
  const { logs: streamLogs, isConnected } = useLogStream(
    selectedJob && activeModalTab === 'logs' && isRunning ? selectedJob.experiment_uid : null
  );

  // Static logs for completed/failed jobs
  const { data: staticLogs, isLoading: isLoadingLogs } = useQuery({
    queryKey: ['jobLogs', selectedJob?.experiment_uid],
    queryFn: async () => {
      const response = await fetch(`/api/queue/experiments/${selectedJob!.experiment_uid}/logs`);
      if (!response.ok) return '';
      const data = await response.json();
      return (data.data?.content || data.content || '').split('\n');
    },
    enabled: !!selectedJob && activeModalTab === 'logs' && !isRunning,
  });

  // Static error logs (stderr only)
  const { data: staticErrorLogs, isLoading: isLoadingErrorLogs, error: _errorLogsError } = useQuery({
    queryKey: ['jobErrorLogs', selectedJob?.experiment_uid],
    queryFn: async () => {
      devLog('Fetching stderr for:', selectedJob!.experiment_uid);
      const response = await fetch(`/api/queue/experiments/${selectedJob!.experiment_uid}/stderr`);
      devLog('Stderr response status:', response.status);

      if (!response.ok) {
        devError('Failed to fetch stderr:', response.status);
        return [];
      }

      const data = await response.json();
      devLog('Stderr data:', data);

      const content = data.data?.content || data.content || '';
      const lines = content.split('\n');
      devLog('Stderr lines:', lines.length);

      return lines;
    },
    enabled: !!selectedJob && activeModalTab === 'errors',
    retry: 1,
    staleTime: 0,
  });

  // Use stream logs for running, static logs for others
  const displayLogs = isRunning ? streamLogs : (staticLogs || []);
  const displayErrorLogs = staticErrorLogs || [];

  // Column metadata for auto-sizing
  const columnMetadata = useMemo(() => [
    { id: 'experiment_uid', header: 'Experiment ID', minSize: 130, maxSize: 200, font: 'mono 14px system-ui' },
    { id: 'project', header: 'Project', minSize: 100, maxSize: 180 },
    { id: 'status', header: 'Status', minSize: 110, maxSize: 140 },
    { id: 'priority', header: 'Priority', minSize: 80, maxSize: 100 },
    { id: 'started_at', header: 'Started At', minSize: 150, maxSize: 180 },
    { id: 'completed_at', header: 'Completed At', minSize: 150, maxSize: 180 },
    { id: 'duration', header: 'Duration', minSize: 70, maxSize: 100 },
    { id: 'assigned_device', header: 'Device', minSize: 150, maxSize: 200, font: 'mono 14px system-ui' },
  ], []);

  // Calculate optimal column widths when data changes
  useEffect(() => {
    if (jobs.length > 0) {
      const optimalSizing = calculateOptimalColumnWidths(jobs, columnMetadata);
      setColumnSizing(optimalSizing);
    }
  }, [jobs, columnMetadata]);

  const columns = useMemo<ColumnDef<QueueJob>[]>(() => [
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
            setSelectedJob(row.original);
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

        // If no config.yaml exists, show "Legacy Job"
        if (hasConfig === false) {
          return (
            <Badge variant="secondary" className="text-xs cursor-default opacity-60">
              Legacy Job
            </Badge>
          );
        }

        // If config exists but project is null/undefined/empty or default_project, show "Not Logged"
        const isProjectMissing = !project || project === 'default_project' || project.trim() === '';

        if (isProjectMissing) {
          return (
            <Badge variant="secondary" className="text-xs cursor-default opacity-60">
              Not Logged
            </Badge>
          );
        }

        // Valid project with config - clickable badge
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
      maxSize: 140,
      enableResizing: true,
      cell: ({ getValue }) => <StatusBadge status={getValue<string>()} />,
    },
    {
      accessorKey: 'priority',
      header: 'Priority',
      minSize: 80,
      maxSize: 100,
      enableResizing: true,
      cell: ({ getValue }) => {
        const priority = getValue<string>()?.toLowerCase();
        const colors = {
          urgent: 'bg-red-100 text-red-800',
          high: 'bg-orange-100 text-orange-800',
          normal: 'bg-blue-100 text-blue-800',
          low: 'bg-gray-100 text-gray-800',
        };
        return (
          <Badge className={`${colors[priority as keyof typeof colors] || colors.normal} text-xs whitespace-nowrap`}>
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
      accessorKey: 'completed_at',
      header: 'Completed At',
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
  ], []);

  const sortedJobs = useMemo(() => {
    return [...jobs].sort((a, b) => {
      const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
      const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
      return bTime - aTime;
    });
  }, [jobs]);

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

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Experiment Results</h1>
          <p className="text-muted-foreground">
            Execution timeline and logs for all experiments
          </p>
        </div>
        <div className="text-center p-8">
          <div className="text-red-600 mb-4">⚠️ Error loading experiments</div>
          <p className="text-muted-foreground">{(error as Error)?.message || 'Unknown error occurred'}</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Experiment Results</h1>
          <p className="text-muted-foreground">
            Execution timeline and logs for all experiments
          </p>
        </div>
        <div className="text-center p-8">
          <div className="animate-spin h-8 w-8 border-b-2 border-primary mx-auto" />
          <p className="text-muted-foreground mt-2">Loading experiments...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Experiment Results</h1>
        <p className="text-muted-foreground">
          Execution timeline and logs for all experiments
        </p>
      </div>

      {/* Results Table */}
      <Card>
        <CardHeader>
          <CardTitle>Execution Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          {jobs.length === 0 ? (
            <div className="text-center p-8">
              <p className="text-muted-foreground">No experiments found.</p>
            </div>
          ) : (
            <AdvancedDataTable
              columns={columns}
              data={sortedJobs}
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

      {/* Complete Experiment Details Modal */}
      <Dialog open={!!selectedJob} onOpenChange={(open) => !open && setSelectedJob(null)}>
        <DialogContent className="w-full max-w-4xl h-[90vh] flex flex-col p-0" hideCloseButton>
          {selectedJob && (
            <>
              {/* Fixed Header */}
              <div className="flex-shrink-0 border-b p-4 sm:p-6">
                <div className="flex items-start justify-between gap-4 mb-2">
                  <DialogTitle className="text-lg sm:text-xl">Experiment Details</DialogTitle>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedJob(null)} aria-label="Close">
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                <DialogDescription className="text-xs sm:text-sm">
                  <span className="font-mono">{selectedJob.experiment_uid}</span> • {selectedJob.project || 'No Project'}
                </DialogDescription>
              </div>

              {/* Tabs */}
              <Tabs value={activeModalTab} onValueChange={setActiveModalTab} className="flex-1 flex flex-col min-h-0">
                <TabsList className="flex-shrink-0 mx-4 sm:mx-6 mt-4">
                  <TabsTrigger value="configuration">Configuration</TabsTrigger>
                  <TabsTrigger value="logs">Terminal Logs</TabsTrigger>
                  <TabsTrigger value="errors">Terminal Errors</TabsTrigger>
                </TabsList>

                {/* Configuration Tab Content */}
                <TabsContent value="configuration" className="flex-1 overflow-y-auto min-h-0 mt-0">
                  <div className="p-4 sm:p-6">
                    <div className="animate-in fade-in duration-300">
                      <div className="border rounded-lg overflow-hidden">
                        {/* Configuration Header */}
                        <div className="bg-background border-b p-4">
                          <div className="flex items-center justify-between">
                            <h3 className="text-lg font-semibold">Configuration Used</h3>
                            <div className="flex gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => copyToClipboard(jobConfig || '')}
                                disabled={!jobConfig || isLoadingConfig}
                                className="flex items-center gap-2"
                              >
                                <Copy className="h-4 w-4" />
                                Copy
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => downloadFile(jobConfig || '', `${selectedJob.experiment_uid}_config.yaml`)}
                                disabled={!jobConfig || isLoadingConfig}
                                className="flex items-center gap-2"
                              >
                                <Download className="h-4 w-4" />
                                Download
                              </Button>
                            </div>
                          </div>
                        </div>
                        {/* Configuration Content */}
                        <div className="h-[calc(90vh-400px)] rounded-b-lg p-3 overflow-hidden">
                          {isLoadingConfig ? (
                            <div className="flex items-center justify-center h-full">
                              <div className="text-center">
                                <div className="animate-spin h-6 w-6 border-b-2 border-primary mx-auto mb-2" />
                                <p className="text-muted-foreground">Loading configuration...</p>
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
                              <p className="text-muted-foreground">Configuration not available</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                {/* Logs Tab Content */}
                <TabsContent value="logs" className="flex-1 overflow-y-auto min-h-0 mt-0">
                  <div className="p-4 sm:p-6">
                    <div className="animate-in fade-in duration-300">
                      <div className="border rounded-lg overflow-hidden">
                        {/* Terminal Header */}
                        <div className="bg-background border-b p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <h3 className="text-lg font-semibold">Terminal Logs</h3>
                              {isRunning && isConnected && (
                                <span className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
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
                                onClick={() => {
                                  const logText = displayLogs.join('\n');
                                  navigator.clipboard.writeText(logText);
                                }}
                                disabled={displayLogs.length === 0}
                                className="flex items-center gap-2"
                              >
                                <Copy className="h-4 w-4" />
                                Copy
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  const blob = new Blob([displayLogs.join('\n')], { type: 'text/plain' });
                                  const url = URL.createObjectURL(blob);
                                  const a = document.createElement('a');
                                  a.href = url;
                                  a.download = `${selectedJob.experiment_uid}_logs.txt`;
                                  document.body.appendChild(a);
                                  a.click();
                                  document.body.removeChild(a);
                                  URL.revokeObjectURL(url);
                                }}
                                disabled={displayLogs.length === 0}
                                className="flex items-center gap-2"
                              >
                                <Download className="h-4 w-4" />
                                Download
                              </Button>
                            </div>
                          </div>
                        </div>
                        {/* Terminal Content */}
                        <div className="h-[calc(90vh-400px)] bg-black dark:bg-gray-950 rounded-b-lg p-3 overflow-y-auto">
                          <div className="text-green-400 font-mono text-xs leading-relaxed">
                            {isLoadingLogs && !isRunning ? (
                              <div className="text-gray-500 flex items-center justify-center h-full">
                                <div className="text-center">
                                  <div className="animate-spin h-6 w-6 border-b-2 border-green-500 mx-auto mb-2" />
                                  Loading logs...
                                </div>
                              </div>
                            ) : displayLogs.length === 0 ? (
                              <div className="text-gray-500 flex items-center justify-center h-full">
                                {isRunning ? (isConnected ? 'Waiting for logs...' : 'Connecting...') : 'No logs available'}
                              </div>
                            ) : (
                              displayLogs.map((line, index) => (
                                <div key={index} className="whitespace-pre-wrap">
                                  {line}
                                </div>
                              ))
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                {/* Errors Tab Content */}
                <TabsContent value="errors" className="flex-1 overflow-y-auto min-h-0 mt-0">
                  <div className="p-4 sm:p-6">
                    <div className="animate-in fade-in duration-300">
                      <div className="border rounded-lg overflow-hidden">
                        {/* Error Terminal Header */}
                        <div className="bg-background border-b p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <h3 className="text-lg font-semibold">Terminal Errors (STDERR)</h3>
                            </div>
                            <div className="flex gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  const errorText = displayErrorLogs.join('\n');
                                  navigator.clipboard.writeText(errorText);
                                }}
                                disabled={displayErrorLogs.length === 0}
                                className="flex items-center gap-2"
                              >
                                <Copy className="h-4 w-4" />
                                Copy
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  const blob = new Blob([displayErrorLogs.join('\n')], { type: 'text/plain' });
                                  const url = URL.createObjectURL(blob);
                                  const a = document.createElement('a');
                                  a.href = url;
                                  a.download = `${selectedJob.experiment_uid}_errors.txt`;
                                  document.body.appendChild(a);
                                  a.click();
                                  document.body.removeChild(a);
                                  URL.revokeObjectURL(url);
                                }}
                                disabled={displayErrorLogs.length === 0}
                                className="flex items-center gap-2"
                              >
                                <Download className="h-4 w-4" />
                                Download
                              </Button>
                            </div>
                          </div>
                        </div>
                        {/* Error Terminal Content */}
                        <div className="h-[calc(90vh-400px)] bg-black dark:bg-gray-950 rounded-b-lg p-3 overflow-y-auto">
                          <div className="text-red-400 font-mono text-xs leading-relaxed">
                            {isLoadingErrorLogs ? (
                              <div className="text-gray-500 flex items-center justify-center h-full">
                                <div className="text-center">
                                  <div className="animate-spin h-6 w-6 border-b-2 border-red-500 mx-auto mb-2" />
                                  Loading error logs...
                                </div>
                              </div>
                            ) : displayErrorLogs.length === 0 ? (
                              <div className="text-gray-500 flex items-center justify-center h-full">
                                No error logs available
                              </div>
                            ) : (
                              displayErrorLogs.map((line, index) => (
                                <div key={index} className="whitespace-pre-wrap">
                                  {line}
                                </div>
                              ))
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
