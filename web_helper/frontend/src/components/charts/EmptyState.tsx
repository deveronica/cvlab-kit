import React from "react";

/**
 * EmptyState Component
 *
 * Displays contextual empty states for charts and data visualizations.
 * Provides clear guidance on what action to take.
 */

import { AlertCircle, BarChart3, Database, Search } from 'lucide-react';
import { Card, CardContent } from '../ui/card';
import { Button } from '../ui/button';

export type EmptyStateVariant = 'no-selection' | 'no-data' | 'no-metric' | 'loading-error';

interface EmptyStateProps {
  variant: EmptyStateVariant;
  metricKey?: string;
  onAction?: () => void;
  actionLabel?: string;
  className?: string;
}

const VARIANT_CONFIG: Record<
  EmptyStateVariant,
  {
    icon: React.ComponentType<{ className?: string }>;
    title: string;
    description: string;
    defaultAction?: string;
  }
> = {
  'no-selection': {
    icon: Search,
    title: 'No runs selected',
    description: 'Select one or more runs from the table to visualize metrics and compare results.',
    defaultAction: 'Select runs',
  },
  'no-data': {
    icon: Database,
    title: 'No data available',
    description: 'The selected runs do not contain any metric data. Check if the experiment completed successfully.',
  },
  'no-metric': {
    icon: BarChart3,
    title: 'Metric not found',
    description: 'The selected runs do not contain this metric. Try selecting a different metric or run.',
  },
  'loading-error': {
    icon: AlertCircle,
    title: 'Failed to load data',
    description: 'An error occurred while loading metric data. Please try again or check your connection.',
    defaultAction: 'Retry',
  },
};

export function EmptyState({
  variant,
  metricKey,
  onAction,
  actionLabel,
  className,
}: EmptyStateProps) {
  const config = VARIANT_CONFIG[variant];
  const Icon = config.icon;

  return (
    <Card className={className}>
      <CardContent className="flex flex-col items-center justify-center py-12 px-6">
        <div className="flex flex-col items-center text-center max-w-md space-y-4">
          {/* Icon */}
          <div className="rounded-full bg-muted p-3">
            <Icon className="h-6 w-6 text-muted-foreground" />
          </div>

          {/* Title */}
          <h3 className="text-lg font-semibold text-foreground">
            {config.title}
          </h3>

          {/* Description */}
          <p className="text-sm text-muted-foreground">
            {variant === 'no-metric' && metricKey
              ? `The metric "${metricKey}" is not available in the selected runs.`
              : config.description}
          </p>

          {/* Action Button */}
          {(onAction || config.defaultAction) && (
            <Button
              onClick={onAction}
              variant={variant === 'loading-error' ? 'default' : 'outline'}
              className="mt-4"
            >
              {actionLabel || config.defaultAction}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Minimal inline empty state for smaller components
 */
export function InlineEmptyState({
  message,
  className,
}: {
  message: string;
  className?: string;
}) {
  return (
    <div
      className={`flex items-center justify-center h-64 text-sm text-muted-foreground ${className || ''}`}
    >
      {message}
    </div>
  );
}
