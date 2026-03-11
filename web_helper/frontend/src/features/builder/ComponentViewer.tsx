/**
 * ComponentViewer - Modal to view component source code
 *
 * Features:
 * - View forward() method implementation
 * - Syntax highlighted code view
 * - Copy to clipboard
 * - Navigate to component file
 */

import { useEffect, useState, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { X, Copy, Code, Loader2, Check, ChevronRight } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { Button } from '@/shared/ui/button';
import { Badge } from '@/shared/ui/badge';
import { ScrollArea } from '@/shared/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/shared/ui/dialog';
import { useAgentBuilder, ComponentViewerState } from '@/entities/node-system/model/AgentBuilderContext';
import { getCategoryTheme } from '@/shared/config/node-themes';

interface ComponentGraphData {
  id: string;
  label: string;
  source_code?: string;
  nodes?: Array<{ id: string; label: string; type: string }>;
}

// Fetch component graph (includes source code)
async function fetchComponentSource(
  category: string,
  name: string
): Promise<ComponentGraphData | null> {
  try {
    const response = await fetch(
      `/api/nodes/component/${encodeURIComponent(category)}/${encodeURIComponent(name)}`
    );
    if (!response.ok) return null;

    const result = await response.json();
    return result.data;
  } catch (error) {
    console.error('Failed to fetch component source:', error);
    return null;
  }
}

// Simple syntax highlighting for Python
function highlightPython(code: string): string {
  // Keywords
  const keywords =
    /\b(def|class|return|if|elif|else|for|while|try|except|finally|with|as|import|from|raise|pass|break|continue|yield|lambda|and|or|not|in|is|None|True|False|self)\b/g;
  // Built-in functions
  const builtins =
    /\b(print|len|range|enumerate|zip|map|filter|sum|max|min|abs|round|sorted|reversed|list|dict|set|tuple|str|int|float|bool|type|super|isinstance|hasattr|getattr|setattr)\b/g;
  // Decorators
  const decorators = /@\w+/g;
  // Strings
  const strings = /(["'])(?:(?=(\\?))\2.)*?\1/g;
  // Comments
  const comments = /#.*$/gm;
  // Numbers
  const numbers = /\b\d+\.?\d*\b/g;

  let highlighted = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  highlighted = highlighted
    .replace(comments, '<span class="text-muted-foreground italic">$&</span>')
    .replace(strings, '<span class="text-amber-600 dark:text-amber-400">$&</span>')
    .replace(decorators, '<span class="text-cyan-600 dark:text-cyan-400">$&</span>')
    .replace(keywords, '<span class="text-purple-600 dark:text-purple-400 font-medium">$&</span>')
    .replace(builtins, '<span class="text-blue-600 dark:text-blue-400">$&</span>')
    .replace(numbers, '<span class="text-green-600 dark:text-green-400">$&</span>');

  return highlighted;
}

// Code viewer with line numbers
function CodeView({ code, className }: { code: string; className?: string }) {
  const lines = code.split('\n');
  const highlighted = highlightPython(code);
  const highlightedLines = highlighted.split('\n');

  return (
    <div className={cn('font-mono text-xs', className)}>
      <table className="w-full border-collapse">
        <tbody>
          {lines.map((_, idx) => (
            <tr key={idx} className="hover:bg-accent/30">
              <td className="w-10 pr-3 text-right text-muted-foreground/50 select-none border-r border-border">
                {idx + 1}
              </td>
              <td className="pl-3">
                <pre
                  className="whitespace-pre"
                  dangerouslySetInnerHTML={{ __html: highlightedLines[idx] || '' }}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ComponentViewer() {
  const { componentViewer, closeComponentViewer } = useAgentBuilder();
  const { isOpen, category, name, implementation } = componentViewer;

  const [copied, setCopied] = useState(false);

  // Derive component name from implementation or use provided name
  const componentName = implementation
    ? implementation.split('(')[0].toLowerCase().replace(/_/g, '_')
    : name;

  // Fetch component source
  const { data, isLoading, error } = useQuery({
    queryKey: ['component-source', category, componentName],
    queryFn: () => fetchComponentSource(category, componentName),
    enabled: isOpen && !!category && !!componentName,
    staleTime: 60000,
  });

  // Copy to clipboard
  const handleCopy = useCallback(async () => {
    if (!data?.source_code) return;

    try {
      await navigator.clipboard.writeText(data.source_code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error('Failed to copy:', e);
    }
  }, [data?.source_code]);

  // Reset copied state when dialog closes
  useEffect(() => {
    if (!isOpen) setCopied(false);
  }, [isOpen]);

  const categoryTheme = category ? getCategoryTheme(category) : null;

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && closeComponentViewer()}>
      <DialogContent className="max-w-4xl h-[80vh] flex flex-col p-0" hideCloseButton>
        {/* Header */}
        <DialogHeader className="px-4 py-3 border-b flex-shrink-0">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <DialogTitle className="flex items-center gap-2">
                <Code className="h-4 w-4" />
                <span>Component Source</span>
                {categoryTheme && (
                  <Badge variant="outline" className={cn('text-xs', categoryTheme.badge)}>
                    {category}
                  </Badge>
                )}
              </DialogTitle>
              <DialogDescription className="flex items-center gap-1 text-xs">
                <span className="text-muted-foreground">cvlabkit/component/{category}/</span>
                <span className="font-medium">{componentName}.py</span>
                <ChevronRight className="h-3 w-3 mx-1 text-muted-foreground" />
                <span className="font-mono">forward()</span>
              </DialogDescription>
            </div>

            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={handleCopy}
                disabled={!data?.source_code}
              >
                {copied ? (
                  <Check className="h-3.5 w-3.5 text-green-500" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </Button>
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={closeComponentViewer}>
                <X className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        </DialogHeader>

        {/* Content */}
        <div className="flex-1 min-h-0">
          {isLoading ? (
            <div className="h-full flex items-center justify-center">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : error || !data ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
              <Code className="h-12 w-12 opacity-20 mb-3" />
              <p className="text-sm font-medium">Unable to load component</p>
              <p className="text-xs mt-1">
                {category}/{componentName}.py may not exist
              </p>
            </div>
          ) : !data.source_code ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
              <Code className="h-12 w-12 opacity-20 mb-3" />
              <p className="text-sm font-medium">No source code available</p>
              <p className="text-xs mt-1">forward() method not found in this component</p>

              {/* Show internal structure if available */}
              {data.nodes && data.nodes.length > 0 && (
                <div className="mt-4 p-3 bg-muted/50 rounded-lg max-w-md">
                  <p className="text-xs font-medium mb-2">Internal Structure:</p>
                  <div className="flex flex-wrap gap-1">
                    {data.nodes.slice(0, 10).map((node) => (
                      <Badge key={node.id} variant="outline" className="text-[10px]">
                        {node.label}
                      </Badge>
                    ))}
                    {data.nodes.length > 10 && (
                      <Badge variant="secondary" className="text-[10px]">
                        +{data.nodes.length - 10} more
                      </Badge>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <ScrollArea className="h-full">
              <CodeView code={data.source_code} className="p-4" />
            </ScrollArea>
          )}
        </div>

        {/* Footer */}
        <div className="flex-shrink-0 px-4 py-2 border-t bg-muted/30 flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            {data?.nodes?.length && (
              <span>
                {data.nodes.length} operations in forward()
              </span>
            )}
          </div>
          <div className="text-xs text-muted-foreground">
            cvlabkit/component/{category}/{componentName}.py
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
