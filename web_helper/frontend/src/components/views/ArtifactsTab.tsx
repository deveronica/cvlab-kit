import React from "react";
/**
 * Artifacts Tab
 *
 * Main artifacts management interface:
 * - File browser (left panel)
 * - File preview (right panel)
 * - Split pane layout
 */

import { useState, useEffect } from 'react';
import { FileBrowser } from '../ui/file-browser';
import { FilePreview } from '../ui/file-preview';
import { AlertCircle } from 'lucide-react';

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  modified?: number;
  extension?: string;
  mime_type?: string;
  children?: FileNode[];
}

interface ArtifactsTabProps {
  project?: string | null;
}

export function ArtifactsTab({ project }: ArtifactsTabProps) {
  const [tree, setTree] = useState<FileNode | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<FileNode | null>(null);

  useEffect(() => {
    const fetchTree = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const url = project
          ? `/api/artifacts/tree?project=${encodeURIComponent(project)}`
          : '/api/artifacts/tree';

        const response = await fetch(url);

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to load artifacts');
        }

        const data: FileNode = await response.json();
        setTree(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    };

    fetchTree();
  }, [project]);

  const handleFileSelect = (file: FileNode) => {
    if (file.type === 'file') {
      setSelectedFile(file);
    }
  };

  if (error) {
    return (
      <div className="flex items-center justify-center h-[600px]">
        <div className="flex flex-col items-center gap-3 text-destructive">
          <AlertCircle className="h-8 w-8" />
          <div className="text-center">
            <p className="text-sm font-medium">Failed to load artifacts</p>
            <p className="text-xs text-muted-foreground mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left Panel: File Browser */}
      <div className="lg:col-span-1">
        <FileBrowser
          tree={tree}
          isLoading={isLoading}
          onFileSelect={handleFileSelect}
          selectedPath={selectedFile?.path}
        />
      </div>

      {/* Right Panel: File Preview */}
      <div className="lg:col-span-1">
        <FilePreview filePath={selectedFile?.path || null} />
      </div>
    </div>
  );
}
