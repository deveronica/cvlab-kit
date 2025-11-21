import React from "react";

/**
 * TimelineCard - Unified wrapper for timeline/gantt components
 *
 * Features:
 * - Consistent header layout (Title + Description + Controls)
 * - Loading, error, empty states
 * - Time range selector
 * - Status filter (show/hide by status)
 * - View mode selector (compact/standard/detailed)
 * - Zoom controls
 * - Export functionality (CSV, JSON, PNG)
 * - Fullscreen mode
 * - Settings panel
 */

import { useState, forwardRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './dialog';
import { Button } from './button';
import { Badge } from './badge';
import {
  Loader2,
  AlertCircle,
  Maximize2,
  Download,
  Settings,
  Filter,
  ZoomIn,
  ZoomOut,
  Layers,
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
  DropdownMenuLabel,
} from './dropdown-menu';
import { cn } from '@/lib/utils';

export interface TimelineCardProps {
  // Content
  title: string;
  description?: string;
  badge?: React.ReactNode;
  children: React.ReactNode;
  fullscreenChildren?: React.ReactNode; // Optional different content for fullscreen

  // State
  isLoading?: boolean;
  error?: string | null;
  isEmpty?: boolean;
  emptyMessage?: string;

  // Features
  enableFullscreen?: boolean;
  enableExport?: boolean;
  enableSettings?: boolean;
  enableStatusFilter?: boolean;
  enableZoomControls?: boolean;
  enableViewModeControl?: boolean;

  // Status Filter
  availableStatuses?: string[];
  visibleStatuses?: Set<string>;
  onStatusFilterChange?: (statuses: Set<string>) => void;

  // Zoom
  zoomLevel?: number;
  onZoomIn?: () => void;
  onZoomOut?: () => void;

  // View Mode
  viewMode?: 'compact' | 'standard' | 'detailed';
  onViewModeChange?: (mode: 'compact' | 'standard' | 'detailed') => void;

  // Export
  onExportCSV?: () => void;
  onExportJSON?: () => void;
  onExportImage?: () => void;

  // Settings
  onSettingsOpen?: () => void;

  // Styling
  className?: string;
  variant?: 'default' | 'compact';
  height?: number;

  // Callbacks
  onFullscreenChange?: (isFullscreen: boolean) => void;
}

export const TimelineCard = forwardRef<HTMLDivElement, TimelineCardProps>(
  (
    {
      // Content
      title,
      description,
      badge,
      children,
      fullscreenChildren,

      // State
      isLoading = false,
      error = null,
      isEmpty = false,
      emptyMessage = 'No timeline data available',

      // Features
      enableFullscreen = true,
      enableExport = true,
      enableSettings = false,
      enableStatusFilter = false,
      enableZoomControls = false,
      enableViewModeControl = false,

      // Status Filter
      availableStatuses = ['completed', 'running', 'failed', 'pending'],
      visibleStatuses = new Set(availableStatuses),
      onStatusFilterChange,

      // Zoom
      zoomLevel = 1,
      onZoomIn,
      onZoomOut,

      // View Mode
      viewMode = 'standard',
      onViewModeChange,

      // Export
      onExportCSV,
      onExportJSON,
      onExportImage,

      // Settings
      onSettingsOpen,

      // Styling
      className,
      variant = 'default',
      height,

      // Callbacks
      onFullscreenChange,
    },
    ref
  ) => {
    const [isFullscreen, setIsFullscreen] = useState(false);

    const handleFullscreenToggle = () => {
      const newState = !isFullscreen;
      setIsFullscreen(newState);
      onFullscreenChange?.(newState);
    };

    const handleStatusToggle = (status: string) => {
      if (!onStatusFilterChange) return;
      const newStatuses = new Set(visibleStatuses);
      if (newStatuses.has(status)) {
        newStatuses.delete(status);
      } else {
        newStatuses.add(status);
      }
      onStatusFilterChange(newStatuses);
    };

    // Loading state
    if (isLoading) {
      return (
        <Card ref={ref} className={cn('overflow-hidden', className)}>
          <CardHeader className={variant === 'compact' ? 'p-4 pb-2' : undefined}>
            <CardTitle size={variant === 'compact' ? 'sm' : 'base'}>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
          <CardContent
            className={cn(
              'flex items-center justify-center',
              variant === 'compact' ? 'p-4 pt-2' : undefined
            )}
            style={{ height: height ? `${height}px` : undefined }}
          >
            <div className="flex flex-col items-center gap-3 text-muted-foreground py-12">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="text-sm">Loading timeline...</span>
            </div>
          </CardContent>
        </Card>
      );
    }

    // Error state
    if (error) {
      return (
        <Card ref={ref} className={cn('overflow-hidden', className)}>
          <CardHeader className={variant === 'compact' ? 'p-4 pb-2' : undefined}>
            <CardTitle size={variant === 'compact' ? 'sm' : 'base'}>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
          <CardContent
            className={cn(
              'flex items-center justify-center',
              variant === 'compact' ? 'p-4 pt-2' : undefined
            )}
            style={{ height: height ? `${height}px` : undefined }}
          >
            <div className="flex flex-col items-center gap-3 text-destructive py-12">
              <AlertCircle className="h-8 w-8" />
              <div className="text-center">
                <p className="text-sm font-medium">Error loading timeline</p>
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
        <Card ref={ref} className={cn('overflow-hidden', className)}>
          <CardHeader className={variant === 'compact' ? 'p-4 pb-2' : undefined}>
            <CardTitle size={variant === 'compact' ? 'sm' : 'base'}>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
          <CardContent
            className={cn(
              'flex items-center justify-center',
              variant === 'compact' ? 'p-4 pt-2' : undefined
            )}
            style={{ height: height ? `${height}px` : undefined }}
          >
            <div className="text-center text-muted-foreground py-12">
              <p className="text-sm">{emptyMessage}</p>
            </div>
          </CardContent>
        </Card>
      );
    }

    // Normal state with full controls
    const renderContent = (isFullscreenMode: boolean) => (
      <Card ref={!isFullscreenMode ? ref : undefined} className={cn('overflow-hidden', !isFullscreenMode && className)}>
        <CardHeader className={variant === 'compact' ? 'p-4 pb-2' : undefined}>
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
            {/* Left: Title, Description, Badge */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <CardTitle size={variant === 'compact' ? 'sm' : 'base'}>{title}</CardTitle>
                {badge && <div className="flex-shrink-0">{badge}</div>}
              </div>
              {description && <CardDescription className="mt-1">{description}</CardDescription>}
            </div>

            {/* Right: Controls */}
            <div className="flex items-center gap-1.5 flex-shrink-0">
              {/* Status Filter */}
              {enableStatusFilter && onStatusFilterChange && availableStatuses.length > 0 && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      title="Filter by status"
                    >
                      <Filter className="h-3.5 w-3.5" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-48">
                    <DropdownMenuLabel>Filter by Status</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {availableStatuses.map((status) => (
                      <DropdownMenuCheckboxItem
                        key={status}
                        checked={visibleStatuses.has(status)}
                        onCheckedChange={() => handleStatusToggle(status)}
                      >
                        <span className="capitalize">{status}</span>
                      </DropdownMenuCheckboxItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              )}

              {/* Zoom Controls */}
              {enableZoomControls && (onZoomIn || onZoomOut) && (
                <div className="flex items-center gap-1">
                  {onZoomOut && (
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      onClick={onZoomOut}
                      disabled={zoomLevel <= 0.5}
                      title="Zoom out"
                    >
                      <ZoomOut className="h-3.5 w-3.5" />
                    </Button>
                  )}
                  {onZoomIn && (
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      onClick={onZoomIn}
                      disabled={zoomLevel >= 2}
                      title="Zoom in"
                    >
                      <ZoomIn className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </div>
              )}

              {/* View Mode Control */}
              {enableViewModeControl && onViewModeChange && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      title="View mode"
                    >
                      <Layers className="h-3.5 w-3.5" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      onClick={() => onViewModeChange('compact')}
                      className={viewMode === 'compact' ? 'bg-accent' : ''}
                    >
                      Compact
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => onViewModeChange('standard')}
                      className={viewMode === 'standard' ? 'bg-accent' : ''}
                    >
                      Standard
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => onViewModeChange('detailed')}
                      className={viewMode === 'detailed' ? 'bg-accent' : ''}
                    >
                      Detailed
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}

              {/* Settings */}
              {enableSettings && onSettingsOpen && (
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8"
                  onClick={onSettingsOpen}
                  title="Timeline settings"
                >
                  <Settings className="h-3.5 w-3.5" />
                </Button>
              )}

              {/* Export */}
              {enableExport && (onExportCSV || onExportJSON || onExportImage) && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      title="Export timeline"
                    >
                      <Download className="h-3.5 w-3.5" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    {onExportCSV && (
                      <DropdownMenuItem onClick={onExportCSV}>
                        Export as CSV
                      </DropdownMenuItem>
                    )}
                    {onExportJSON && (
                      <DropdownMenuItem onClick={onExportJSON}>
                        Export as JSON
                      </DropdownMenuItem>
                    )}
                    {onExportImage && (
                      <>
                        {(onExportCSV || onExportJSON) && <DropdownMenuSeparator />}
                        <DropdownMenuItem onClick={onExportImage}>
                          Export as Image
                        </DropdownMenuItem>
                      </>
                    )}
                  </DropdownMenuContent>
                </DropdownMenu>
              )}

              {/* Fullscreen */}
              {enableFullscreen && !isFullscreenMode && (
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8"
                  onClick={handleFullscreenToggle}
                  title="Fullscreen"
                >
                  <Maximize2 className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
          </div>
        </CardHeader>

        <CardContent
          className={variant === 'compact' ? 'p-4 pt-2' : undefined}
          style={{ height: height ? `${height}px` : undefined }}
        >
          {children}
        </CardContent>
      </Card>
    );

    return (
      <>
        {renderContent(false)}

        {/* Fullscreen Dialog */}
        {enableFullscreen && (
          <Dialog open={isFullscreen} onOpenChange={handleFullscreenToggle}>
            <DialogContent className="max-w-[95vw] w-full h-[90vh] flex flex-col">
              <DialogHeader>
                <DialogTitle>{title}</DialogTitle>
              </DialogHeader>
              <div className="flex-1 overflow-hidden">
                {fullscreenChildren || children}
              </div>
            </DialogContent>
          </Dialog>
        )}
      </>
    );
  }
);

TimelineCard.displayName = 'TimelineCard';
