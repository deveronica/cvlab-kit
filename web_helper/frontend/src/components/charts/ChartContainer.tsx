import React from "react";

/**
 * Chart Container
 *
 * Common wrapper for all chart components providing:
 * - Consistent card styling and layout
 * - Loading and error states
 * - Export functionality (PNG, SVG, CSV)
 * - Chart actions toolbar
 * - Responsive sizing
 */

import { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/card';
import { Button } from '../ui/button';
import { Maximize2, Settings } from 'lucide-react';
import { ExportMenu } from '../ui/export-menu';
import { Skeleton } from '../ui/skeleton';
import { devWarn, devError } from '../../lib/dev-utils';

interface ChartContainerProps {
  title: string;
  description?: string;
  isLoading?: boolean;
  error?: Error | null;
  children: React.ReactNode;
  actions?: React.ReactNode;
  enableExport?: boolean;
  enableFullscreen?: boolean;
  onSettingsClick?: () => void;
  className?: string;
}

export function ChartContainer({
  title,
  description,
  isLoading = false,
  error = null,
  children,
  actions,
  enableExport = true,
  enableFullscreen = false,
  onSettingsClick,
  className = '',
}: ChartContainerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Export to PNG
  const exportToPNG = useCallback(() => {
    const svgElement = document.querySelector('.recharts-wrapper svg') as SVGElement;
    if (!svgElement) {
      devWarn('No chart SVG found for export');
      return;
    }

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      canvas.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${title.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      });
    };
    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
  }, [title]);

  // Export to SVG
  const exportToSVG = useCallback(() => {
    const svgElement = document.querySelector('.recharts-wrapper svg') as SVGElement;
    if (!svgElement) {
      devWarn('No chart SVG found for export');
      return;
    }

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const blob = new Blob([svgData], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${title.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [title]);

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  // Render loading state
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <Skeleton className="h-64 w-full" />
            <div className="flex gap-2">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-4 w-28" />
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Render error state
  if (error) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <p className="text-destructive font-medium mb-2">Error loading chart</p>
            <p className="text-sm text-muted-foreground">{error.message}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      className={`${className} ${isFullscreen ? 'fixed inset-4 z-50 overflow-auto' : ''}`}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div>
          <CardTitle className={isFullscreen ? 'text-xl' : 'text-base'}>
            {title}
          </CardTitle>
          {description && <CardDescription className="mt-1">{description}</CardDescription>}
        </div>

        <div className="flex items-center gap-2">
          {/* Custom actions */}
          {actions}

          {/* Export dropdown */}
          {enableExport && (
            <ExportMenu
              onExport={(format) => {
                if (format === 'png') exportToPNG();
                else if (format === 'svg') exportToSVG();
              }}
              formats={['png', 'svg']}
              showLabel={false}
              size="sm"
            />
          )}

          {/* Fullscreen toggle */}
          {enableFullscreen && (
            <Button
              variant="outline"
              size="sm"
              onClick={toggleFullscreen}
              className="h-7"
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
          )}

          {/* Settings */}
          {onSettingsClick && (
            <Button
              variant="outline"
              size="sm"
              onClick={onSettingsClick}
              className="h-7"
            >
              <Settings className="h-4 w-4" />
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className={isFullscreen ? 'h-[calc(100%-5rem)]' : ''}>
        {children}
      </CardContent>
    </Card>
  );
}

/**
 * Chart Loading Skeleton
 * Standalone loading component for charts
 */
export function ChartLoadingSkeleton({
  title,
  description,
}: {
  title: string;
  description?: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <Skeleton className="h-64 w-full" />
          <div className="flex gap-2">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-4 w-28" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Chart Error Boundary
 * Error boundary wrapper for charts
 */
interface ChartErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface ChartErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ChartErrorBoundary extends React.Component<
  ChartErrorBoundaryProps,
  ChartErrorBoundaryState
> {
  constructor(props: ChartErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ChartErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    devError('Chart Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Card>
          <CardContent className="py-8">
            <div className="flex flex-col items-center justify-center text-center">
              <p className="text-destructive font-medium mb-2">
                Chart rendering error
              </p>
              <p className="text-sm text-muted-foreground">
                {this.state.error?.message || 'An unexpected error occurred'}
              </p>
            </div>
          </CardContent>
        </Card>
      );
    }

    return this.props.children;
  }
}
