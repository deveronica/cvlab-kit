import React from "react";

import { Badge } from './badge';
import { CheckCircle2, Clock, XCircle, AlertCircle, Archive } from 'lucide-react';
import { cn } from '../../lib/utils';

export type RunStatus = 'completed' | 'running' | 'failed' | 'pending' | 'orphaned' | 'archived';

interface RunStatusBadgeProps {
  status: RunStatus;
  className?: string;
  showIcon?: boolean;
}

const statusConfig: Record<
  RunStatus,
  {
    label: string;
    variant: 'default' | 'secondary' | 'destructive' | 'outline';
    icon: React.ComponentType<{ className?: string }>;
    className: string;
  }
> = {
  completed: {
    label: 'Completed',
    variant: 'default',
    icon: CheckCircle2,
    className: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100 border-green-200 dark:border-green-800',
  },
  running: {
    label: 'Running',
    variant: 'default',
    icon: Clock,
    className: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-100 border-blue-200 dark:border-blue-800',
  },
  failed: {
    label: 'Failed',
    variant: 'destructive',
    icon: XCircle,
    className: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100 border-red-200 dark:border-red-800',
  },
  pending: {
    label: 'Pending',
    variant: 'secondary',
    icon: Clock,
    className: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100 border-yellow-200 dark:border-yellow-800',
  },
  orphaned: {
    label: 'Orphaned',
    variant: 'outline',
    icon: AlertCircle,
    className: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-100 border-orange-200 dark:border-orange-800',
  },
  archived: {
    label: 'Archived',
    variant: 'outline',
    icon: Archive,
    className: 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-100 border-gray-200 dark:border-gray-800',
  },
};

export function RunStatusBadge({ status, className, showIcon = true }: RunStatusBadgeProps) {
  const config = statusConfig[status] || statusConfig.completed;
  const Icon = config.icon;

  return (
    <Badge
      variant={config.variant}
      className={cn(config.className, 'flex items-center gap-1.5 font-medium', className)}
    >
      {showIcon && <Icon className="h-3 w-3" />}
      <span>{config.label}</span>
    </Badge>
  );
}

export function getStatusColor(status: RunStatus): string {
  const config = statusConfig[status] || statusConfig.completed;
  return config.className;
}

export function getStatusIcon(status: RunStatus) {
  const config = statusConfig[status] || statusConfig.completed;
  return config.icon;
}
