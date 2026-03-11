/**
 * CodeViewerPanel - Resizable side panel for viewing source code
 *
 * Features:
 * - Line numbers
 * - Syntax highlighting (basic)
 * - Selected range highlighting (yellow)
 * - Uncovered lines indicator (red)
 * - Scroll to selected range
 */

import { memo, useEffect, useRef } from 'react';
import { X, FileCode, Loader2, AlertCircle } from 'lucide-react';
import { Button } from '@/shared/ui/button';
import { useCodeViewerStore } from '@/features/builder/model/codeViewerStore';
import { cn } from '@/shared/lib/utils';

interface CodeViewerPanelProps {
  className?: string;
}

export const CodeViewerPanel = memo(({ className }: CodeViewerPanelProps) => {
  const {
    isOpen,
    filePath,
    content,
    isLoading,
    error,
    highlightedLines,
    selectedRange,
    closeViewer,
    setContent,
    setError,
  } = useCodeViewerStore();

  const contentRef = useRef<HTMLDivElement>(null);
  const selectedRef = useRef<HTMLDivElement>(null);

  // Fetch content when filePath changes
  useEffect(() => {
    if (!filePath || !isOpen) return;

    const fetchContent = async () => {
      try {
        const response = await fetch(`/api/files/content?path=${encodeURIComponent(filePath)}`);

        if (!response.ok) {
          throw new Error(`Failed to load: ${response.statusText}`);
        }

        const data = await response.json();
        setContent(data.content);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load file');
      }
    };

    fetchContent();
  }, [filePath, isOpen, setContent, setError]);

  // Scroll to selected range
  useEffect(() => {
    if (selectedRef.current && content) {
      selectedRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  }, [selectedRange, content]);

  if (!isOpen) return null;

  const fileName = filePath?.split('/').pop() || 'Unknown';
  const lines = content?.split('\n') || [];

  return (
    <div className={cn(
      "w-[400px] border-l flex flex-col bg-background",
      className
    )}>
      {/* Header */}
      <div className="px-3 py-2 border-b flex items-center justify-between bg-muted/30">
        <div className="flex items-center gap-2 min-w-0">
          <FileCode className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium truncate" title={filePath || ''}>
            {fileName}
          </span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={closeViewer}
          className="h-6 w-6 p-0"
          aria-label="Close code viewer"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <div ref={contentRef} className="flex-1 overflow-auto">
        {isLoading && (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-destructive">
            <AlertCircle className="h-6 w-6" />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {content && (
          <div className="font-mono text-xs leading-relaxed">
            {lines.map((line, idx) => {
              const lineNum = idx + 1;
              const isSelected = selectedRange &&
                lineNum >= selectedRange.start &&
                lineNum <= selectedRange.end;
              const isCovered = highlightedLines.covered.includes(lineNum);
              const isUncovered = highlightedLines.uncovered.includes(lineNum);
              const isFirstSelected = selectedRange && lineNum === selectedRange.start;

              return (
                <div
                  key={lineNum}
                  ref={isFirstSelected ? selectedRef : undefined}
                  className={cn(
                    "flex hover:bg-muted/50",
                    isSelected && "bg-yellow-100 dark:bg-yellow-900/30",
                    !isSelected && isUncovered && "bg-red-50 dark:bg-red-950/20",
                    !isSelected && isCovered && "bg-green-50/30 dark:bg-green-950/10"
                  )}
                >
                  {/* Line number */}
                  <span className={cn(
                    "w-10 pr-2 text-right select-none flex-shrink-0",
                    "text-muted-foreground border-r border-border/50",
                    isSelected && "text-yellow-700 dark:text-yellow-400 font-medium"
                  )}>
                    {lineNum}
                  </span>

                  {/* Code */}
                  <span className="flex-1 pl-2 whitespace-pre overflow-x-auto">
                    {line || ' '}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {!isLoading && !error && !content && (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <span className="text-sm">Select a node to view source code</span>
          </div>
        )}
      </div>

      {/* Footer with file info */}
      {content && (
        <div className="px-3 py-1.5 border-t bg-muted/30 text-[10px] text-muted-foreground flex items-center justify-between">
          <span>{lines.length} lines</span>
          {selectedRange && (
            <span>
              Lines {selectedRange.start}
              {selectedRange.end !== selectedRange.start && `-${selectedRange.end}`}
            </span>
          )}
        </div>
      )}
    </div>
  );
});

CodeViewerPanel.displayName = 'CodeViewerPanel';
