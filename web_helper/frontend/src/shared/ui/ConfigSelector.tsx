/**
 * ConfigSelector - Searchable Config File Selector
 *
 * Improved UX for selecting config files:
 * - Search/filter functionality
 * - Shows file name prominently, path as subtitle
 * - Groups by folder structure
 */

import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { Check, ChevronsUpDown, File, Folder, Search } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { Button } from '@shared/ui/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@shared/ui/popover';
import { Input } from '@shared/ui/input';
import { ScrollArea } from '@shared/ui/scroll-area';

interface ConfigSelectorProps {
  value: string;
  onValueChange: (value: string) => void;
  configFiles: string[];
  isLoading?: boolean;
  error?: Error | null;
  className?: string;
}

// Parse file path into folder and filename
function parseFilePath(filePath: string): { folder: string; fileName: string } {
  const parts = filePath.split('/');
  const fileName = parts.pop() || filePath;
  const folder = parts.join('/') || 'root';
  return { folder, fileName };
}

// Group files by folder
function groupByFolder(files: string[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();

  for (const file of files) {
    const { folder } = parseFilePath(file);
    const existing = groups.get(folder) || [];
    existing.push(file);
    groups.set(folder, existing);
  }

  // Sort folders: 'config' first, then alphabetically
  const sortedGroups = new Map<string, string[]>();
  const folders = Array.from(groups.keys()).sort((a, b) => {
    if (a === 'config') return -1;
    if (b === 'config') return 1;
    return a.localeCompare(b);
  });

  for (const folder of folders) {
    const files = groups.get(folder) || [];
    // Sort files within folder
    files.sort((a, b) => {
      const aName = parseFilePath(a).fileName;
      const bName = parseFilePath(b).fileName;
      return aName.localeCompare(bName);
    });
    sortedGroups.set(folder, files);
  }

  return sortedGroups;
}

export function ConfigSelector({
  value,
  onValueChange,
  configFiles,
  isLoading = false,
  error = null,
  className,
}: ConfigSelectorProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when popover opens
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 100);
    } else {
      setSearch('');
    }
  }, [open]);

  // Filter files by search term
  const filteredFiles = useMemo(() => {
    if (!search.trim()) return configFiles;

    const searchLower = search.toLowerCase();
    return configFiles.filter((file) => {
      const { fileName, folder } = parseFilePath(file);
      return (
        fileName.toLowerCase().includes(searchLower) ||
        folder.toLowerCase().includes(searchLower)
      );
    });
  }, [configFiles, search]);

  // Group filtered files
  const groupedFiles = useMemo(() => groupByFolder(filteredFiles), [filteredFiles]);

  // Handle selection
  const handleSelect = useCallback(
    (file: string) => {
      onValueChange(file);
      setOpen(false);
    },
    [onValueChange]
  );

  // Display value
  const displayValue = useMemo(() => {
    if (!value) return null;
    const { fileName } = parseFilePath(value);
    return fileName;
  }, [value]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          aria-label="Select configuration file"
          className={cn('w-[320px] justify-between h-9', className)}
          disabled={isLoading}
        >
          {isLoading ? (
            <span className="text-muted-foreground">Loading...</span>
          ) : displayValue ? (
            <span className="flex items-center gap-2 truncate">
              <File className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
              <span className="truncate">{displayValue}</span>
            </span>
          ) : (
            <span className="text-muted-foreground">Select config file...</span>
          )}
          <ChevronsUpDown className="h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[400px] p-0" align="start">
        {/* Search Input */}
        <div className="flex items-center border-b px-3 py-2">
          <Search className="h-4 w-4 text-muted-foreground mr-2" />
          <Input
            ref={inputRef}
            placeholder="Search config files..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 border-0 p-0 focus-visible:ring-0 placeholder:text-muted-foreground"
          />
        </div>

        {/* File List */}
        <ScrollArea className="max-h-[300px]">
          {error ? (
            <div className="p-4 text-center text-sm text-destructive">
              Error loading configs
            </div>
          ) : filteredFiles.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted-foreground">
              {search ? 'No matching files found' : 'No config files available'}
            </div>
          ) : (
            <div className="p-1">
              {Array.from(groupedFiles.entries()).map(([folder, files]) => (
                <div key={folder}>
                  {/* Folder Header (only show if multiple folders) */}
                  {groupedFiles.size > 1 && (
                    <div className="flex items-center gap-1.5 px-2 py-1.5 text-xs font-medium text-muted-foreground">
                      <Folder className="h-3 w-3" />
                      {folder}
                    </div>
                  )}

                  {/* Files in folder */}
                  {files.map((file) => {
                    const { fileName } = parseFilePath(file);
                    const isSelected = value === file;

                    return (
                      <button
                        key={file}
                        onClick={() => handleSelect(file)}
                        className={cn(
                          'flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors',
                          'hover:bg-accent hover:text-accent-foreground',
                          'focus:bg-accent focus:text-accent-foreground focus:outline-none',
                          isSelected && 'bg-accent text-accent-foreground'
                        )}
                      >
                        <File className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="font-medium truncate">{fileName}</div>
                          {groupedFiles.size === 1 && folder !== 'root' && (
                            <div className="text-xs text-muted-foreground truncate">
                              {folder}
                            </div>
                          )}
                        </div>
                        {isSelected && (
                          <Check className="h-4 w-4 text-primary flex-shrink-0" />
                        )}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
}
