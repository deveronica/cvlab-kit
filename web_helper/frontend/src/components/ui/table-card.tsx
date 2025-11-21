import React from "react";

/**
 * TableCard - Unified wrapper for all table components
 *
 * Features:
 * - Consistent header layout (Title + Description + Controls)
 * - Loading, error, empty states
 * - Column visibility toggle
 * - Quick search/filter
 * - Pagination controls
 * - Density selector (compact/standard/comfortable)
 * - Export functionality (CSV, JSON)
 * - Fullscreen mode
 * - Settings panel
 */

import { useState, forwardRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './dialog';
import { Button } from './button';
import { Input } from './input';
import { Badge } from './badge';
import {
  Loader2,
  AlertCircle,
  Maximize2,
  Download,
  Settings,
  Search,
  Columns3,
  SlidersHorizontal,
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  _DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './dropdown-menu';
import { cn } from '@/lib/utils';

export interface TableCardProps {
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
  enableColumnToggle?: boolean;
  enableQuickFilter?: boolean;
  enablePagination?: boolean;
  enableDensityControl?: boolean;

  // Quick Filter
  quickFilterValue?: string;
  quickFilterPlaceholder?: string;
  onQuickFilterChange?: (value: string) => void;

  // Pagination
  currentPage?: number;
  totalPages?: number;
  pageSize?: number;
  totalRows?: number;
  onPageChange?: (page: number) => void;
  onPageSizeChange?: (size: number) => void;

  // Density
  density?: 'compact' | 'standard' | 'comfortable';
  onDensityChange?: (density: 'compact' | 'standard' | 'comfortable') => void;

  // Export
  onExportCSV?: () => void;
  onExportJSON?: () => void;

  // Column Toggle
  onColumnToggle?: () => void;

  // Settings
  onSettingsOpen?: () => void;

  // Styling
  className?: string;
  variant?: 'default' | 'compact';
  height?: number;

  // Callbacks
  onFullscreenChange?: (isFullscreen: boolean) => void;
}

export const TableCard = forwardRef<HTMLDivElement, TableCardProps>(
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
      emptyMessage = 'No data available',

      // Features
      enableFullscreen = true,
      enableExport = true,
      enableSettings = false,
      enableColumnToggle = false,
      enableQuickFilter = false,
      enablePagination = false,
      enableDensityControl = false,

      // Quick Filter
      quickFilterValue = '',
      quickFilterPlaceholder = 'Search...',
      onQuickFilterChange,

      // Pagination
      currentPage = 1,
      totalPages = 1,
      pageSize = 10,
      totalRows = 0,
      onPageChange,
      onPageSizeChange,

      // Density
      density = 'standard',
      onDensityChange,

      // Export
      onExportCSV,
      onExportJSON,

      // Column Toggle
      onColumnToggle,

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
              <span className="text-sm">Loading data...</span>
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
                <p className="text-sm font-medium">Error loading data</p>
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
              {/* Quick Filter */}
              {enableQuickFilter && onQuickFilterChange && !isFullscreenMode && (
                <div className="relative">
                  <Search className="absolute left-2.5 top-1/2 transform -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                  <Input
                    type="text"
                    placeholder={quickFilterPlaceholder}
                    value={quickFilterValue}
                    onChange={(e) => onQuickFilterChange(e.target.value)}
                    className="h-8 w-48 pl-8 text-sm"
                  />
                </div>
              )}

              {/* Column Toggle */}
              {enableColumnToggle && onColumnToggle && (
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8"
                  onClick={onColumnToggle}
                  title="Toggle columns"
                >
                  <Columns3 className="h-3.5 w-3.5" />
                </Button>
              )}

              {/* Density Control */}
              {enableDensityControl && onDensityChange && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      title="Table density"
                    >
                      <SlidersHorizontal className="h-3.5 w-3.5" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      onClick={() => onDensityChange('compact')}
                      className={density === 'compact' ? 'bg-accent' : ''}
                    >
                      Compact
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => onDensityChange('standard')}
                      className={density === 'standard' ? 'bg-accent' : ''}
                    >
                      Standard
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => onDensityChange('comfortable')}
                      className={density === 'comfortable' ? 'bg-accent' : ''}
                    >
                      Comfortable
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
                  title="Table settings"
                >
                  <Settings className="h-3.5 w-3.5" />
                </Button>
              )}

              {/* Export */}
              {enableExport && (onExportCSV || onExportJSON) && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      className="h-8 w-8"
                      title="Export data"
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

        {/* Pagination Footer */}
        {enablePagination && totalPages > 1 && (
          <div className="border-t border-border px-4 py-3 flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              Showing {(currentPage - 1) * pageSize + 1} to {Math.min(currentPage * pageSize, totalRows)} of {totalRows} rows
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={currentPage === 1}
                onClick={() => onPageChange?.(currentPage - 1)}
              >
                Previous
              </Button>
              <span className="text-sm text-muted-foreground">
                Page {currentPage} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={currentPage === totalPages}
                onClick={() => onPageChange?.(currentPage + 1)}
              >
                Next
              </Button>
              {onPageSizeChange && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm">
                      {pageSize} rows
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    {[10, 25, 50, 100].map((size) => (
                      <DropdownMenuItem
                        key={size}
                        onClick={() => onPageSizeChange(size)}
                        className={pageSize === size ? 'bg-accent' : ''}
                      >
                        {size} rows
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </div>
          </div>
        )}
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

TableCard.displayName = 'TableCard';
