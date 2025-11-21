import React from "react";
/**
 * Config Viewer Modal
 *
 * Full-screen YAML configuration viewer with syntax highlighting.
 * Displays the complete experiment configuration file.
 *
 * Features:
 * - YAML syntax highlighting (CodeMirror)
 * - Copy to clipboard
 * - Download YAML file
 * - Read-only view
 */

import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Copy, Download, Check, AlertCircle, Loader2, X } from 'lucide-react';
import CodeMirror from '@uiw/react-codemirror';
import { yaml } from '@codemirror/lang-yaml';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import { Card, CardContent } from '../ui/card';
import { ScrollArea } from '../ui/scroll-area';
import { devError } from '../../lib/dev-utils';

interface ConfigViewerModalProps {
  open: boolean;
  onClose: () => void;
  project: string;
  runName: string;
}

export function ConfigViewerModal({
  open,
  onClose,
  project,
  runName,
}: ConfigViewerModalProps) {
  const [configContent, setConfigContent] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Fetch config when modal opens
  useEffect(() => {
    if (!open || !project || !runName) {
      return;
    }

    const fetchConfig = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`/api/runs/${project}/${runName}/config`);

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail?.detail || 'Failed to load configuration');
        }

        const data = await response.json();
        setConfigContent(data.data.content || '# No configuration content available');
      } catch (err) {
        devError('Error fetching config:', err);
        setError(err instanceof Error ? err.message : 'Failed to load configuration');
        setConfigContent('# Error loading configuration');
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, [open, project, runName]);

  // Copy to clipboard
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(configContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      devError('Failed to copy:', err);
    }
  };

  // Download as YAML file
  const handleDownload = () => {
    const blob = new Blob([configContent], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${runName}_config.yaml`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="w-full max-w-6xl h-[90vh] overflow-hidden flex flex-col gap-4" hideCloseButton>
        <div className="flex flex-col space-y-1.5 text-left">
          <div className="flex items-start justify-between gap-4">
            <DialogTitle>Configuration Viewer</DialogTitle>
            <div className="flex items-center gap-2 flex-shrink-0">
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopy}
                disabled={loading || !!error}
              >
                {copied ? (
                  <>
                    <Check className="h-4 w-4 mr-2 text-green-600" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4 mr-2" />
                    Copy
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownload}
                disabled={loading || !!error}
              >
                <Download className="h-4 w-4 mr-2" />
                Download
              </Button>
              <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close">
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <DialogDescription>
            <span className="font-mono text-sm">{project} / {runName}</span>
          </DialogDescription>
        </div>

        <div className="flex-1 overflow-hidden flex flex-col">
          {loading && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">Loading configuration...</p>
              </div>
            </div>
          )}

          {error && (
            <Card className="border-red-500 bg-red-50 dark:bg-red-950/20">
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 text-red-800 dark:text-red-200">
                  <AlertCircle className="h-4 w-4 flex-shrink-0" />
                  <p className="text-sm">{error}</p>
                </div>
              </CardContent>
            </Card>
          )}

          {!loading && !error && (
            <ScrollArea className="flex-1 border rounded-md">
              <CodeMirror
                value={configContent}
                extensions={[yaml()]}
                theme={vscodeDark}
                readOnly
                basicSetup={{
                  lineNumbers: true,
                  highlightActiveLineGutter: true,
                  highlightActiveLine: true,
                  foldGutter: true,
                  dropCursor: false,
                  allowMultipleSelections: false,
                  indentOnInput: false,
                }}
                height="600px"
                style={{ fontSize: '13px' }}
              />
            </ScrollArea>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
