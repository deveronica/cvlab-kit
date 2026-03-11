/**
 * Runs table with real-time updates and inline editing
 */

import { useMemo, useState } from 'react';
import { type ColumnDef } from '@tanstack/react-table';
import { Badge } from '@shared/ui/badge';
import { AdvancedDataTable } from '@shared/ui/advanced-data-table';
import { useRuns } from '@/shared/model/useProjects';
import { RunStatusBadge, type RunStatus } from '@shared/ui/run-status-badge';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@shared/ui/dialog';
import { ScrollArea } from '@shared/ui/scroll-area';
import { Loader2 } from 'lucide-react';

interface Run {
  run_name: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  project: string;
}

interface RunsTableProps {
  projectName?: string;
  onRunClick?: (run: Run) => void;
}

// Format duration
function formatDuration(startTime: string | null, endTime: string | null): string {
  if (!startTime) return '-';

  const start = new Date(startTime);
  const end = endTime ? new Date(endTime) : new Date();
  const durationMs = end.getTime() - start.getTime();

  if (durationMs < 0) return '-';

  const seconds = Math.floor(durationMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

// Format timestamp
function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) return '-';
  return new Date(timestamp).toLocaleString();
}

export function RunsTable({ projectName, onRunClick }: RunsTableProps) {
  const { data: runs = [], isLoading } = useRuns(projectName);

  // Dialog states
  const [metricsDialogOpen, setMetricsDialogOpen] = useState(false);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [dialogLoading, setDialogLoading] = useState(false);
  const [metricsData, setMetricsData] = useState<Record<string, unknown>[] | null>(null);
  const [configData, setConfigData] = useState<string | null>(null);
  const [dialogError, setDialogError] = useState<string | null>(null);

  // Fetch metrics for a run
  const handleViewMetrics = async (run: Run) => {
    setSelectedRun(run);
    setMetricsDialogOpen(true);
    setDialogLoading(true);
    setDialogError(null);
    setMetricsData(null);

    try {
      const response = await fetch(`/api/runs/${run.project}/${run.run_name}/metrics?downsample=100`);
      if (!response.ok) {
        throw new Error('Failed to fetch metrics');
      }
      const result = await response.json();
      setMetricsData(result.data?.data || []);
    } catch (error) {
      setDialogError(error instanceof Error ? error.message : 'Failed to load metrics');
    } finally {
      setDialogLoading(false);
    }
  };

  // Fetch config for a run
  const handleViewConfig = async (run: Run) => {
    setSelectedRun(run);
    setConfigDialogOpen(true);
    setDialogLoading(true);
    setDialogError(null);
    setConfigData(null);

    try {
      const response = await fetch(`/api/runs/${run.project}/${run.run_name}/config`);
      if (!response.ok) {
        throw new Error('Failed to fetch config');
      }
      const result = await response.json();
      setConfigData(result.data?.content || '');
    } catch (error) {
      setDialogError(error instanceof Error ? error.message : 'Failed to load config');
    } finally {
      setDialogLoading(false);
    }
  };

  const columns = useMemo<ColumnDef<Run>[]>(() => [
    {
      accessorKey: 'run_name',
      header: 'Run ID',
      cell: ({ getValue }) => (
        <span className="font-mono text-sm font-medium">{getValue<string>()}</span>
      ),
    },
    ...(projectName ? [] : [{
      accessorKey: 'project' as const,
      header: 'Project',
      cell: ({ getValue }: any) => (
        <Badge variant="outline">{getValue()}</Badge>
      ),
    }]),
    {
      accessorKey: 'status',
      header: 'Status',
      cell: ({ getValue }) => <RunStatusBadge status={getValue<string>() as RunStatus} />,
    },
    {
      accessorKey: 'started_at',
      header: 'Started',
      cell: ({ getValue }) => (
        <span className="text-sm text-muted-foreground">
          {formatTimestamp(getValue<string>())}
        </span>
      ),
    },
    {
      accessorKey: 'finished_at',
      header: 'Finished',
      cell: ({ getValue }) => (
        <span className="text-sm text-muted-foreground">
          {formatTimestamp(getValue<string>())}
        </span>
      ),
    },
    {
      id: 'duration',
      header: 'Duration',
      cell: ({ row }) => {
        const duration = formatDuration(row.original.started_at, row.original.finished_at);
        return (
          <span className="text-sm font-mono">
            {duration}
            {row.original.status === 'running' && (
              <span className="ml-1 text-green-600 animate-pulse">●</span>
            )}
          </span>
        );
      },
    },
    {
      id: 'actions',
      header: 'Actions',
      cell: ({ row }) => (
        <div className="flex items-center space-x-2">
          <button
            onClick={() => onRunClick?.(row.original)}
            className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
          >
            View
          </button>
          <button
            onClick={() => handleViewMetrics(row.original)}
            className="text-sm text-green-600 hover:text-green-800 hover:underline"
          >
            Metrics
          </button>
          <button
            onClick={() => handleViewConfig(row.original)}
            className="text-sm text-purple-600 hover:text-purple-800 hover:underline"
          >
            Config
          </button>
        </div>
      ),
    },
  ], [projectName, onRunClick]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">
          {projectName ? `${projectName} Runs` : 'All Runs'} ({runs.length})
        </h3>
        <Badge variant="outline">
          Real-time updates enabled
        </Badge>
      </div>

      <AdvancedDataTable
        columns={columns}
        data={runs}
        enableSorting
        enableFiltering
        enablePagination
        pageSize={15}
        onRowClick={onRunClick}
        className="border rounded-lg"
        emptyStateComponent={(
          <tr>
            <td colSpan={columns.length} className="h-24 text-center text-muted-foreground">
              No runs have been recorded for this project yet. You can start a new run from the 'Execute' tab.
            </td>
          </tr>
        )}
      />

      {/* Metrics Dialog */}
      <Dialog open={metricsDialogOpen} onOpenChange={setMetricsDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle>
              Metrics: {selectedRun?.run_name}
            </DialogTitle>
          </DialogHeader>
          <ScrollArea className="h-[60vh]">
            {dialogLoading ? (
              <div className="flex items-center justify-center h-32">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : dialogError ? (
              <div className="text-center text-destructive p-4">
                {dialogError}
              </div>
            ) : metricsData && metricsData.length > 0 ? (
              <div className="space-y-4">
                {/* Summary of latest metrics */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {Object.entries(metricsData[metricsData.length - 1] || {}).map(([key, value]) => (
                    <div key={key} className="p-3 bg-muted rounded-lg">
                      <div className="text-xs text-muted-foreground uppercase">{key}</div>
                      <div className="text-lg font-mono font-semibold">
                        {typeof value === 'number' ? value.toFixed(4) : String(value)}
                      </div>
                    </div>
                  ))}
                </div>
                {/* Full data table */}
                <div className="border rounded-lg overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-muted">
                      <tr>
                        {Object.keys(metricsData[0] || {}).map(key => (
                          <th key={key} className="px-3 py-2 text-left font-medium">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {metricsData.slice(-20).map((row, idx) => (
                        <tr key={idx} className="border-t">
                          {Object.values(row).map((value, vIdx) => (
                            <td key={vIdx} className="px-3 py-2 font-mono">
                              {typeof value === 'number' ? value.toFixed(4) : String(value)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {metricsData.length > 20 && (
                    <div className="px-3 py-2 text-xs text-muted-foreground bg-muted">
                      Showing last 20 of {metricsData.length} rows
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground p-4">
                No metrics data available
              </div>
            )}
          </ScrollArea>
        </DialogContent>
      </Dialog>

      {/* Config Dialog */}
      <Dialog open={configDialogOpen} onOpenChange={setConfigDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle>
              Config: {selectedRun?.run_name}
            </DialogTitle>
          </DialogHeader>
          <ScrollArea className="h-[60vh]">
            {dialogLoading ? (
              <div className="flex items-center justify-center h-32">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : dialogError ? (
              <div className="text-center text-destructive p-4">
                {dialogError}
              </div>
            ) : configData ? (
              <pre className="p-4 bg-muted rounded-lg overflow-auto text-sm font-mono whitespace-pre-wrap">
                {configData}
              </pre>
            ) : (
              <div className="text-center text-muted-foreground p-4">
                No config data available
              </div>
            )}
          </ScrollArea>
        </DialogContent>
      </Dialog>
    </div>
  );
}
