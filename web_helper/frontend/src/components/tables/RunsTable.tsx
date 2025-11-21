import React from "react";
/**
 * Runs table with real-time updates and inline editing
 */

import { useMemo } from 'react';
import { type ColumnDef } from '@tanstack/react-table';
import { Badge } from '../ui/badge';
import { AdvancedDataTable } from '../ui/advanced-data-table';
import { useRuns } from '../../hooks/useProjects';
import { RunStatusBadge, type RunStatus } from '../ui/run-status-badge';
import { devLog } from '../../lib/dev-utils';

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
            onClick={() => {
              // TODO: Implement metrics view
              devLog('View metrics for:', row.original.run_name);
            }}
            className="text-sm text-green-600 hover:text-green-800 hover:underline"
          >
            Metrics
          </button>
          <button
            onClick={() => {
              // TODO: Implement config view
              devLog('View config for:', row.original.run_name);
            }}
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
    </div>
  );
}