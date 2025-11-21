import React from "react";
/**
 * Artifact Browser
 *
 * File browser UI for experiment artifacts with type-specific icons and metadata
 */

import { Button } from './button';
import { Badge } from './badge';
import { Skeleton } from './skeleton';
import {
  FileCode,
  Database,
  FileText,
  Table,
  Download,
  Loader2,
  File,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export interface Artifact {
  name: string;
  type: string;
  size: number;
  path: string;
  last_modified: number;
}

interface ArtifactBrowserProps {
  artifacts: Artifact[];
  isLoading?: boolean;
  onDownload: (artifact: Artifact) => void;
  downloadingFile?: string | null;
  className?: string;
}

/**
 * Get icon component based on artifact type
 */
function getArtifactIcon(type: string) {
  const iconClass = 'h-4 w-4 flex-shrink-0';

  switch (type.toLowerCase()) {
    case 'config':
      return <FileCode className={cn(iconClass, 'text-blue-600 dark:text-blue-400')} />;
    case 'metrics':
      return <Table className={cn(iconClass, 'text-green-600 dark:text-green-400')} />;
    case 'checkpoint':
      return <Database className={cn(iconClass, 'text-purple-600 dark:text-purple-400')} />;
    case 'log':
      return <FileText className={cn(iconClass, 'text-yellow-600 dark:text-yellow-400')} />;
    default:
      return <File className={cn(iconClass, 'text-muted-foreground')} />;
  }
}

/**
 * Get type badge variant based on artifact type
 */
function _getTypeBadgeVariant(type: string): 'default' | 'secondary' | 'outline' {
  switch (type.toLowerCase()) {
    case 'config':
      return 'default';
    case 'metrics':
      return 'secondary';
    case 'checkpoint':
    case 'log':
      return 'outline';
    default:
      return 'outline';
  }
}

/**
 * Format file size to human-readable string
 */
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Format timestamp to relative time or date
 */
function formatLastModified(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
  });
}

/**
 * Get file extension from filename
 */
function getFileExtension(filename: string): string {
  const parts = filename.split('.');
  return parts.length > 1 ? parts[parts.length - 1].toUpperCase() : '';
}

/**
 * Loading skeleton
 */
function ArtifactBrowserSkeleton() {
  return (
    <div className="space-y-2">
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="flex items-center gap-3 p-3 border rounded-lg"
        >
          <Skeleton className="h-4 w-4 rounded" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-3 w-32" />
          </div>
          <Skeleton className="h-8 w-20" />
        </div>
      ))}
    </div>
  );
}

/**
 * Empty state
 */
function ArtifactBrowserEmpty() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <File className="h-12 w-12 text-muted-foreground/50 mb-3" />
      <p className="text-sm text-muted-foreground">No artifacts found</p>
      <p className="text-xs text-muted-foreground mt-1">
        Artifacts are generated during experiment execution
      </p>
    </div>
  );
}

/**
 * Artifact Browser Component
 */
export function ArtifactBrowser({
  artifacts,
  isLoading = false,
  onDownload,
  downloadingFile,
  className,
}: ArtifactBrowserProps) {
  // Loading state
  if (isLoading) {
    return (
      <div className={className}>
        <ArtifactBrowserSkeleton />
      </div>
    );
  }

  // Empty state
  if (artifacts.length === 0) {
    return (
      <div className={className}>
        <ArtifactBrowserEmpty />
      </div>
    );
  }

  // Group artifacts by type
  const groupedArtifacts = artifacts.reduce((acc, artifact) => {
    const type = artifact.type.toLowerCase();
    if (!acc[type]) {
      acc[type] = [];
    }
    acc[type].push(artifact);
    return acc;
  }, {} as Record<string, Artifact[]>);

  // Sort groups by priority
  const typeOrder = ['config', 'checkpoint', 'metrics', 'log'];
  const sortedGroups = Object.entries(groupedArtifacts).sort((a, b) => {
    const indexA = typeOrder.indexOf(a[0]);
    const indexB = typeOrder.indexOf(b[0]);
    if (indexA === -1 && indexB === -1) return a[0].localeCompare(b[0]);
    if (indexA === -1) return 1;
    if (indexB === -1) return -1;
    return indexA - indexB;
  });

  return (
    <div className={cn('space-y-4', className)}>
      {sortedGroups.map(([type, typeArtifacts]) => (
        <div key={type}>
          {/* Type Header */}
          <div className="flex items-center gap-2 mb-2">
            {getArtifactIcon(type)}
            <h4 className="text-sm font-semibold capitalize">{type} Files</h4>
            <Badge variant="secondary" className="text-xs">
              {typeArtifacts.length}
            </Badge>
          </div>

          {/* Artifact List */}
          <div className="space-y-1.5">
            {typeArtifacts.map((artifact, idx) => (
              <div
                key={idx}
                className="group flex items-center gap-3 p-3 border rounded-lg hover:bg-accent/50 transition-colors duration-200"
              >
                {/* Icon */}
                <div className="flex-shrink-0">
                  {getArtifactIcon(artifact.type)}
                </div>

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium truncate">
                      {artifact.name}
                    </p>
                    <Badge
                      variant="outline"
                      className="text-xs font-mono px-1.5 py-0"
                    >
                      {getFileExtension(artifact.name)}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                    <span className="font-mono">
                      {formatFileSize(artifact.size)}
                    </span>
                    <span>•</span>
                    <span>
                      {formatLastModified(artifact.last_modified)}
                    </span>
                  </div>
                </div>

                {/* Download Button */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onDownload(artifact)}
                  disabled={downloadingFile === artifact.path}
                  className="opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                  aria-label={`Download ${artifact.name}`}
                >
                  {downloadingFile === artifact.path ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                </Button>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
