import React from "react";
/**
 * File Preview Component
 *
 * Display file previews for different file types:
 * - Images: Base64 display
 * - Text: Syntax-highlighted text
 * - JSON: Formatted JSON viewer
 * - CSV: Table view
 * - Binary: Metadata display
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Button } from './button';
import { Badge } from './badge';
import { Loader2, Download, AlertCircle, FileText, Image as ImageIcon, Code, Database } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FilePreviewProps {
  filePath: string | null;
  className?: string;
}

interface FilePreviewData {
  path: string;
  name: string;
  size: number;
  mime_type: string;
  extension: string;
  preview_type: 'image' | 'text' | 'json' | 'csv' | 'binary';
  content: string | null;
  metadata: Record<string, any> | null;
}

/**
 * Format file size
 */
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export function FilePreview({ filePath, className }: FilePreviewProps) {
  const [preview, setPreview] = useState<FilePreviewData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!filePath) {
      setPreview(null);
      return;
    }

    const fetchPreview = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(`/api/artifacts/preview?path=${encodeURIComponent(filePath)}`);

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to load preview');
        }

        const data: FilePreviewData = await response.json();
        setPreview(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    };

    fetchPreview();
  }, [filePath]);

  const handleDownload = () => {
    if (!filePath) return;
    window.open(`/api/artifacts/download/${encodeURIComponent(filePath)}`, '_blank');
  };

  if (!filePath) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-sm text-muted-foreground">Select a file to preview</div>
        </CardContent>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="flex flex-col items-center gap-3 text-muted-foreground">
            <Loader2 className="h-8 w-8 animate-spin" />
            <span className="text-sm">Loading preview...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="flex flex-col items-center gap-3 text-destructive">
            <AlertCircle className="h-8 w-8" />
            <div className="text-center">
              <p className="text-sm font-medium">Failed to load preview</p>
              <p className="text-xs text-muted-foreground mt-1">{error}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!preview) {
    return null;
  }

  // Icon based on preview type
  const PreviewIcon = {
    image: ImageIcon,
    text: FileText,
    json: Code,
    csv: Database,
    binary: FileText,
  }[preview.preview_type];

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <PreviewIcon className="h-5 w-5 text-primary flex-shrink-0" />
            <CardTitle size="base" className="truncate">{preview.name}</CardTitle>
            <Badge variant="outline" className="text-xs">
              {preview.preview_type}
            </Badge>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleDownload}
            className="ml-2"
          >
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>

        {/* Metadata */}
        <div className="flex items-center gap-4 text-xs text-muted-foreground mt-2">
          <span className="font-mono">{formatFileSize(preview.size)}</span>
          <span>{preview.extension}</span>
          {preview.metadata && Object.entries(preview.metadata).map(([key, value]) => (
            <span key={key} className="flex items-center gap-1">
              <span className="font-medium">{key}:</span>
              <span>{String(value)}</span>
            </span>
          ))}
        </div>
      </CardHeader>

      <CardContent>
        <div className="rounded-lg border border-border bg-muted/20 overflow-hidden">
          {/* Image preview */}
          {preview.preview_type === 'image' && preview.content && (
            <div className="p-4 flex items-center justify-center bg-checkered">
              <img
                src={`data:${preview.mime_type};base64,${preview.content}`}
                alt={preview.name}
                className="max-w-full max-h-[600px] object-contain"
              />
            </div>
          )}

          {/* Text preview */}
          {preview.preview_type === 'text' && preview.content && (
            <pre className="p-4 text-xs font-mono overflow-auto max-h-[600px] whitespace-pre-wrap break-words">
              {preview.content}
            </pre>
          )}

          {/* JSON preview */}
          {preview.preview_type === 'json' && preview.content && (
            <pre className="p-4 text-xs font-mono overflow-auto max-h-[600px] bg-slate-950 text-green-400">
              {preview.content}
            </pre>
          )}

          {/* CSV preview */}
          {preview.preview_type === 'csv' && preview.content && (
            <div className="overflow-auto max-h-[600px]">
              <table className="w-full text-xs border-collapse">
                <tbody>
                  {preview.content.split('\n').map((row, rowIndex) => {
                    const cells = row.split(',');
                    const isHeader = rowIndex === 0;

                    return (
                      <tr
                        key={rowIndex}
                        className={cn(
                          'border-b border-border',
                          isHeader && 'bg-muted font-medium'
                        )}
                      >
                        {cells.map((cell, cellIndex) => (
                          <td
                            key={cellIndex}
                            className="px-3 py-2 border-r border-border last:border-r-0"
                          >
                            {cell}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Binary preview (metadata only) */}
          {preview.preview_type === 'binary' && (
            <div className="p-6 flex flex-col items-center gap-4 text-center">
              <FileText className="h-12 w-12 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium">Binary file</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Preview not available for this file type
                </p>
              </div>
              {preview.metadata && (
                <div className="mt-4 p-4 bg-muted rounded-lg text-left w-full max-w-md">
                  <h4 className="text-xs font-semibold mb-2">File Information</h4>
                  <dl className="space-y-1 text-xs">
                    {Object.entries(preview.metadata).map(([key, value]) => (
                      <div key={key} className="flex justify-between gap-4">
                        <dt className="text-muted-foreground">{key}:</dt>
                        <dd className="font-mono">{String(value)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
