import React from "react";

/**
 * Chart Card Component
 *
 * Unified card wrapper for all chart types with:
 * - Standardized header layout
 * - Loading states
 * - Error boundaries
 * - Fullscreen support
 * - Export functionality
 * - Settings menu integration
 * - Consistent styling
 */

import { useState, forwardRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';
import { Button } from './button';
import { Badge } from './badge';
import { Loader2, Expand, AlertCircle, X } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  _DialogHeader,
  DialogTitle,
} from './dialog';
import { ExportMenu, ExportFormat } from './export-menu';
import { ChartSettingsMenu } from './chart-settings-menu';
import { ChartType } from '@/lib/stores/chart-settings';
import { cn } from '@/lib/utils';

interface ChartCardProps {
  // Content
  children: React.ReactNode;
  fullscreenChildren?: React.ReactNode; // Optional different content for fullscreen

  // Header
  title: string;
  description?: string;
  badge?: {
    label: string;
    variant?: 'default' | 'destructive' | 'outline' | 'secondary';
  };
  customControls?: React.ReactNode; // Custom controls to show before standard buttons

  // Features
  chartType?: ChartType; // For settings menu
  enableSettings?: boolean;
  enableFullscreen?: boolean;
  enableExport?: boolean;
  exportFormats?: ExportFormat[];
  onExport?: (format: ExportFormat) => void;

  // States
  isLoading?: boolean;
  error?: string | null;
  isEmpty?: boolean;
  emptyMessage?: string;

  // Styling
  variant?: 'default' | 'compact';
  className?: string;
  contentClassName?: string;
  height?: number | string;

  // Refs
  exportRef?: React.RefObject<HTMLDivElement>;
}

export const ChartCard = forwardRef<HTMLDivElement, ChartCardProps>(
  (
    {
      children,
      fullscreenChildren,
      title,
      description,
      badge,
      customControls,
      chartType,
      enableSettings = true,
      enableFullscreen = true,
      enableExport = true,
      exportFormats = ['png', 'svg', 'csv'],
      onExport,
      isLoading = false,
      error = null,
      isEmpty = false,
      emptyMessage = 'No data available',
      variant = 'default',
      className,
      contentClassName,
      height,
      exportRef,
    },
    ref
  ) => {
    const [isFullscreen, setIsFullscreen] = useState(false);

    // Loading state
    if (isLoading) {
      return (
        <Card variant={variant} className={className} ref={ref}>
          <CardHeader variant={variant}>
            <CardTitle size={variant === 'compact' ? 'sm' : 'base'}>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
          <CardContent
            className={cn(
              variant === 'default' ? 'p-6 pt-0' : 'p-3 pt-0',
              'flex items-center justify-center',
              contentClassName
            )}
            style={{ height: typeof height === 'number' ? `${height}px` : (height || '400px') }}
          >
            <div className="flex flex-col items-center gap-3 text-muted-foreground">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="text-sm">Loading chart data...</span>
            </div>
          </CardContent>
        </Card>
      );
    }

    // Error state
    if (error) {
      return (
        <Card variant={variant} className={className} ref={ref}>
          <CardHeader variant={variant}>
            <CardTitle size={variant === 'compact' ? 'sm' : 'base'}>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
          <CardContent
            className={cn(
              variant === 'default' ? 'p-6 pt-0' : 'p-3 pt-0',
              'flex items-center justify-center',
              contentClassName
            )}
            style={{ height: typeof height === 'number' ? `${height}px` : (height || '400px') }}
          >
            <div className="flex flex-col items-center gap-3 text-destructive">
              <AlertCircle className="h-8 w-8" />
              <div className="text-center">
                <p className="text-sm font-medium">Failed to load chart</p>
                <p className="text-xs text-muted-foreground mt-1">{error}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      );
    }

    // Empty state
    if (isEmpty) {
      return (
        <Card variant={variant} className={className} ref={ref}>
          <CardHeader variant={variant}>
            <CardTitle size={variant === 'compact' ? 'sm' : 'base'}>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
          <CardContent
            className={cn(
              variant === 'default' ? 'p-6 pt-0' : 'p-3 pt-0',
              'flex items-center justify-center',
              contentClassName
            )}
            style={{ height: typeof height === 'number' ? `${height}px` : (height || '400px') }}
          >
            <div className="text-sm text-muted-foreground text-center">
              {emptyMessage}
            </div>
          </CardContent>
        </Card>
      );
    }

    // Normal state
    return (
      <>
        <Card variant={variant} className={className} ref={ref}>
          <CardHeader variant={variant}>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <CardTitle size={variant === 'compact' ? 'sm' : 'base'}>
                  {title}
                </CardTitle>
                {description && (
                  <CardDescription className="mt-1">{description}</CardDescription>
                )}
              </div>

              <div className="flex items-center gap-2">
                {/* Badge */}
                {badge && (
                  <Badge variant={badge.variant || 'default'}>
                    {badge.label}
                  </Badge>
                )}

                {/* Custom Controls */}
                {customControls}

                {/* Settings */}
                {enableSettings && chartType && (
                  <ChartSettingsMenu chartType={chartType} />
                )}

                {/* Fullscreen */}
                {enableFullscreen && (
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setIsFullscreen(true)}
                    className="h-8 w-8"
                    aria-label="Fullscreen"
                  >
                    <Expand className="h-4 w-4" />
                  </Button>
                )}

                {/* Export */}
                {enableExport && onExport && (
                  <ExportMenu
                    onExport={onExport}
                    formats={exportFormats}
                    showLabel={false}
                    size="sm"
                    className="h-8 w-8 p-0"
                  />
                )}
              </div>
            </div>
          </CardHeader>

          <CardContent
            className={cn(
              variant === 'default' ? 'p-6 pt-0' : 'p-3 pt-0',
              contentClassName
            )}
            ref={exportRef}
            style={height ? { height: typeof height === 'number' ? `${height}px` : height } : undefined}
          >
            {children}
          </CardContent>
        </Card>

        {/* Fullscreen Dialog */}
        {enableFullscreen && (
          <Dialog open={isFullscreen} onOpenChange={setIsFullscreen}>
            <DialogContent hideCloseButton className="w-screen h-screen max-w-none p-0 gap-0 overflow-hidden flex flex-col">
              {/* Header with all controls */}
              <div className="flex flex-col space-y-1.5 p-6 pb-4 flex-shrink-0 border-b">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <DialogTitle className="font-semibold leading-none tracking-tight text-base">
                      {title}
                    </DialogTitle>
                    {description && (
                      <p className="text-sm text-muted-foreground mt-1">{description}</p>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Badge */}
                    {badge && (
                      <Badge variant={badge.variant || 'default'}>
                        {badge.label}
                      </Badge>
                    )}

                    {/* Custom Controls */}
                    {customControls}

                    {/* Settings */}
                    {enableSettings && chartType && (
                      <ChartSettingsMenu chartType={chartType} />
                    )}

                    {/* Export */}
                    {enableExport && onExport && (
                      <ExportMenu
                        onExport={onExport}
                        formats={exportFormats}
                        showLabel={false}
                        size="sm"
                        className="h-8 w-8 p-0"
                      />
                    )}

                    {/* Close */}
                    <Button
                      variant="ghost"
                      onClick={() => setIsFullscreen(false)}
                      aria-label="Close"
                      className="h-8 w-8 p-0"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>

              {/* Content - Canvas only, no card */}
              <div className="flex-1 min-h-0 p-6">
                {fullscreenChildren || children}
              </div>
            </DialogContent>
          </Dialog>
        )}
      </>
    );
  }
);

ChartCard.displayName = 'ChartCard';
