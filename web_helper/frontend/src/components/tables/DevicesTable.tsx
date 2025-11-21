import React from "react";
/**
 * Devices table with real-time updates and inline editing
 */

import { useMemo } from 'react';
import { type ColumnDef } from '@tanstack/react-table';
import { Badge } from '../ui/badge';
import { AdvancedDataTable } from '../ui/advanced-data-table';
import { useDevices, useHeartbeatMutation } from '../../hooks/useDevices';
import type { Device } from '../../lib/types';

// Device status badge component
function DeviceStatusBadge({ status }: { status: string }) {
  const variant = status === 'online' ? 'default' : status === 'stale' ? 'secondary' : 'destructive';
  const label = status === 'online' ? '🟢 Online' : status === 'stale' ? '🟡 Stale' : '🔴 Offline';

  return <Badge variant={variant}>{label}</Badge>;
}

// Format memory/storage values
function formatBytes(bytes: number | null): string {
  if (!bytes) return '-';
  const gb = bytes / (1024 * 1024 * 1024);
  return `${gb.toFixed(1)} GB`;
}

// Format percentage
function formatPercent(value: number | null): string {
  if (value === null || value === undefined) return '-';
  return `${value.toFixed(1)}%`;
}

// Format timestamp
function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) return '-';
  return new Date(timestamp).toLocaleString();
}

export function DevicesTable() {
  const { data: devices = [], isLoading } = useDevices();
  const _heartbeatMutation = useHeartbeatMutation();

  const columns = useMemo<ColumnDef<Device>[]>(() => [
    {
      accessorKey: 'host_id',
      header: 'Host ID',
      cell: ({ getValue }) => (
        <span className="font-mono text-sm">{getValue<string>()}</span>
      ),
    },
    {
      accessorKey: 'status',
      header: 'Status',
      cell: ({ getValue }) => <DeviceStatusBadge status={getValue<string>()} />,
    },
    {
      accessorKey: 'gpu_util',
      header: 'GPU Usage',
      cell: ({ getValue, _row }) => (
        <div className="flex items-center space-x-2">
          <span>{formatPercent(getValue<number>())}</span>
          {getValue<number>() !== null && (
            <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${Math.min(getValue<number>() || 0, 100)}%` }}
              />
            </div>
          )}
        </div>
      ),
    },
    {
      accessorKey: 'cpu_util',
      header: 'CPU Usage',
      cell: ({ getValue }) => (
        <div className="flex items-center space-x-2">
          <span>{formatPercent(getValue<number>())}</span>
          {getValue<number>() !== null && (
            <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500 transition-all duration-300"
                style={{ width: `${Math.min(getValue<number>() || 0, 100)}%` }}
              />
            </div>
          )}
        </div>
      ),
    },
    {
      accessorKey: 'memory_used',
      header: 'Memory',
      cell: ({ _row }) => {
        const used = _row.original.memory_used;
        const total = _row.original.memory_total;
        if (!used || !total) return '-';

        const percent = (used / total) * 100;
        return (
          <div className="flex items-center space-x-2">
            <span className="text-sm">
              {formatBytes(used)} / {formatBytes(total)}
            </span>
            <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-orange-500 transition-all duration-300"
                style={{ width: `${Math.min(percent, 100)}%` }}
              />
            </div>
          </div>
        );
      },
    },
    {
      accessorKey: 'vram_used',
      header: 'VRAM',
      cell: ({ _row }) => {
        const used = _row.original.vram_used;
        const total = _row.original.vram_total;
        if (!used || !total) return '-';

        const percent = (used / total) * 100;
        return (
          <div className="flex items-center space-x-2">
            <span className="text-sm">
              {formatBytes(used)} / {formatBytes(total)}
            </span>
            <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 transition-all duration-300"
                style={{ width: `${Math.min(percent, 100)}%` }}
              />
            </div>
          </div>
        );
      },
    },
    {
      accessorKey: 'torch_version',
      header: 'PyTorch',
      cell: ({ getValue }) => (
        <span className="font-mono text-xs">{getValue<string>() || '-'}</span>
      ),
    },
    {
      accessorKey: 'cuda_version',
      header: 'CUDA',
      cell: ({ getValue }) => (
        <span className="font-mono text-xs">{getValue<string>() || '-'}</span>
      ),
    },
    {
      accessorKey: 'last_heartbeat',
      header: 'Last Seen',
      cell: ({ getValue }) => (
        <span className="text-xs text-muted-foreground">
          {formatTimestamp(getValue<string>())}
        </span>
      ),
    },
  ], []);

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
        <h3 className="text-lg font-semibold">Devices ({devices.length})</h3>
        <Badge variant="outline">
          Real-time updates enabled
        </Badge>
      </div>

      <AdvancedDataTable
        columns={columns}
        data={devices}
        enableSorting
        enableFiltering
        enablePagination
        pageSize={20}
        className="border rounded-lg"
      />
    </div>
  );
}