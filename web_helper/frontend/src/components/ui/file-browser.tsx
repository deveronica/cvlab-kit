import React from "react";
/**
 * File Browser Component
 *
 * Interactive file tree browser with:
 * - Collapsible directory structure
 * - File type icons
 * - Size and date display
 * - Click to preview
 * - Download support
 * - Search/filter
 */

import { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Input } from './input';
import { Button } from './button';
import { Badge } from './badge';
import { Loader2, Search, FileText, Image, Code, Database, Archive, File, Folder, FolderOpen, ChevronRight, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

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

interface FileBrowserProps {
  tree: FileNode | null;
  isLoading?: boolean;
  onFileSelect?: (file: FileNode) => void;
  selectedPath?: string;
  className?: string;
}

/**
 * Get icon for file type
 */
function getFileIcon(node: FileNode) {
  if (node.type === 'directory') {
    return Folder;
  }

  const ext = node.extension?.toLowerCase();

  // Images
  if (['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'].includes(ext || '')) {
    return Image;
  }

  // Code/Config
  if (['.yaml', '.yml', '.json', '.md', '.txt'].includes(ext || '')) {
    return Code;
  }

  // Data
  if (['.csv'].includes(ext || '')) {
    return Database;
  }

  // Models
  if (['.pt', '.pth', '.ckpt', '.safetensors'].includes(ext || '')) {
    return Database;
  }

  // Archives
  if (['.zip', '.tar', '.gz'].includes(ext || '')) {
    return Archive;
  }

  // Logs
  if (['.log'].includes(ext || '')) {
    return FileText;
  }

  return File;
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

/**
 * Format modified time
 */
function formatModifiedTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  // Less than 1 day
  if (diff < 24 * 60 * 60 * 1000) {
    const hours = Math.floor(diff / (60 * 60 * 1000));
    if (hours < 1) {
      const minutes = Math.floor(diff / (60 * 1000));
      return `${minutes}m ago`;
    }
    return `${hours}h ago`;
  }

  // Less than 7 days
  if (diff < 7 * 24 * 60 * 60 * 1000) {
    const days = Math.floor(diff / (24 * 60 * 60 * 1000));
    return `${days}d ago`;
  }

  // Format as date
  return date.toLocaleDateString();
}

/**
 * Tree node component
 */
function TreeNode({
  node,
  level,
  onSelect,
  selectedPath,
  searchQuery,
}: {
  node: FileNode;
  level: number;
  onSelect: (file: FileNode) => void;
  selectedPath?: string;
  searchQuery: string;
}) {
  const [isExpanded, setIsExpanded] = useState(level === 0);

  const hasChildren = node.children && node.children.length > 0;
  const isDirectory = node.type === 'directory';
  const isSelected = selectedPath === node.path;

  const Icon = isDirectory
    ? isExpanded
      ? FolderOpen
      : Folder
    : getFileIcon(node);

  // Filter children based on search
  const filteredChildren = useMemo(() => {
    if (!searchQuery || !hasChildren) return node.children || [];

    return node.children!.filter((child) => {
      const matchesName = child.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesPath = child.path.toLowerCase().includes(searchQuery.toLowerCase());
      const hasMatchingChildren =
        child.children?.some((gc) =>
          gc.name.toLowerCase().includes(searchQuery.toLowerCase())
        ) || false;

      return matchesName || matchesPath || hasMatchingChildren;
    });
  }, [node.children, searchQuery]);

  // Auto-expand if search matches
  const shouldShow = useMemo(() => {
    if (!searchQuery) return true;

    const matches =
      node.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      node.path.toLowerCase().includes(searchQuery.toLowerCase());

    if (matches && hasChildren) {
      setIsExpanded(true);
    }

    return matches || filteredChildren.length > 0;
  }, [node, searchQuery, filteredChildren]);

  if (!shouldShow) return null;

  const handleClick = () => {
    if (isDirectory) {
      setIsExpanded(!isExpanded);
    } else {
      onSelect(node);
    }
  };

  return (
    <div>
      <div
        className={cn(
          'flex items-center gap-2 py-1.5 px-2 rounded-md cursor-pointer transition-colors duration-200 group',
          isSelected && 'bg-primary/10 border border-primary/30',
          !isSelected && 'hover:bg-muted/50'
        )}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={handleClick}
      >
        {/* Expand/Collapse */}
        {isDirectory ? (
          <div className="flex-shrink-0">
            {isExpanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
            )}
          </div>
        ) : (
          <div className="w-4" />
        )}

        {/* Icon */}
        <Icon className={cn('h-4 w-4', isDirectory ? 'text-yellow-600 dark:text-yellow-500' : 'text-muted-foreground')} />

        {/* Name */}
        <span className={cn('flex-1 text-sm truncate', isSelected && 'font-medium')}>
          {node.name}
        </span>

        {/* Size */}
        {node.size !== undefined && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-mono">
            {formatFileSize(node.size)}
          </Badge>
        )}

        {/* Modified */}
        {node.modified !== undefined && (
          <span className="text-xs text-muted-foreground">
            {formatModifiedTime(node.modified)}
          </span>
        )}
      </div>

      {/* Children */}
      {isDirectory && isExpanded && hasChildren && (
        <div className="mt-0.5">
          {filteredChildren.map((child, index) => (
            <TreeNode
              key={`${child.path}-${index}`}
              node={child}
              level={level + 1}
              onSelect={onSelect}
              selectedPath={selectedPath}
              searchQuery={searchQuery}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Main FileBrowser component
 */
export function FileBrowser({
  tree,
  isLoading,
  onFileSelect,
  selectedPath,
  className,
}: FileBrowserProps) {
  const [searchQuery, setSearchQuery] = useState('');

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="flex flex-col items-center gap-3 text-muted-foreground">
            <Loader2 className="h-8 w-8 animate-spin" />
            <span className="text-sm">Loading artifacts...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!tree) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-sm text-muted-foreground">No artifacts found</div>
        </CardContent>
      </Card>
    );
  }

  const handleFileSelect = (file: FileNode) => {
    if (onFileSelect) {
      onFileSelect(file);
    }
  };

  // Count total files
  const totalFiles = useMemo(() => {
    const countFiles = (node: FileNode): number => {
      if (node.type === 'file') return 1;
      if (!node.children) return 0;
      return node.children.reduce((count, child) => count + countFiles(child), 0);
    };
    return countFiles(tree);
  }, [tree]);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Folder className="h-5 w-5 text-yellow-600 dark:text-yellow-500" />
            <CardTitle size="base">Artifacts</CardTitle>
            <Badge variant="secondary" className="text-xs">
              {totalFiles} files
            </Badge>
          </div>
        </div>

        {/* Search */}
        <div className="relative mt-3">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search files..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-9 text-sm"
          />
        </div>
      </CardHeader>

      <CardContent>
        <div className="max-h-[600px] overflow-y-auto space-y-0.5 pr-2">
          {tree.children && tree.children.length > 0 ? (
            tree.children.map((node, index) => (
              <TreeNode
                key={`${node.path}-${index}`}
                node={node}
                level={0}
                onSelect={handleFileSelect}
                selectedPath={selectedPath}
                searchQuery={searchQuery}
              />
            ))
          ) : (
            <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
              No files found
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
