import React, { useState, useEffect } from 'react';
import { Card } from '@/shared/ui/card';
import { Button } from '@/shared/ui/button';
import { ScrollArea } from '@/shared/ui/scroll-area';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/shared/ui/tooltip';
import {
  Folder,
  FolderOpen,
  FileCode2,
  ChevronRight,
  RefreshCw,
  ListTree,
  PanelLeftClose,
} from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { useBuilderStore } from '@/shared/model/builderStore';

const CATEGORIES = [
  'agent', 'model', 'dataset', 'dataloader', 'transform',
  'optimizer', 'loss', 'metric', 'scheduler',
];

const STORAGE_KEY = 'builder-panel-state'; // Re-using the key from the store

export function ComponentExplorerPanel() {
  const {
    leftPanelOpen,
    toggleLeftPanel,
    components,
    setComponents,
    selectedComponent,
    setSelectedComponent,
    setBreadcrumb,
    isComponentsLoaded,
    setComponentsLoaded,
  } = useBuilderStore();

  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(CATEGORIES));
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isComponentsLoaded) {
      fetchComponents();
    }
  }, [isComponentsLoaded]);

  const fetchComponents = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/ui/versions');
      const result = await response.json();
      if (result.data) {
        setComponents(result.data);
        setComponentsLoaded(true);
      }
    } catch (error) {
      console.error('Failed to fetch components:', error);
    }
    setLoading(false);
  };

  const handleScanComponents = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/ui/scan', { method: 'POST' });
      const result = await response.json();
      if (result.data) fetchComponents(); // Re-fetch after scan
    } catch (error) {
      console.error('Failed to scan components:', error);
    }
    setLoading(false);
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) next.delete(category);
      else next.add(category);
      return next;
    });
  };

  const groupedComponents = CATEGORIES.reduce((acc, category) => {
    acc[category] = components.filter((c) => c.category === category);
    return acc;
  }, {} as Record<string, any[]>); // Use 'any[]' for now, will refine type later

  return (
    <div className={cn(
      "flex-shrink-0 transition-all duration-300 ease-in-out overflow-hidden",
      leftPanelOpen ? "w-56 opacity-100" : "w-0 opacity-0"
    )}>
      <Card className="h-full flex flex-col w-56">
        <div className="px-3 py-2 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <ListTree className="h-4 w-4" />
            <span>Explorer</span>
          </div>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleScanComponents} disabled={loading}>
                <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Scan local</TooltipContent>
          </Tooltip>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-1.5">
            {CATEGORIES.map((category) => {
              const items = groupedComponents[category] || [];
              const isExpanded = expandedCategories.has(category);
              if (items.length === 0) return null;
              return (
                <div key={category} className="mb-0.5">
                  <button
                    onClick={() => toggleCategory(category)}
                    className="w-full flex items-center gap-1.5 px-2 py-1 rounded hover:bg-accent/50 text-sm transition-colors duration-200"
                  >
                    <ChevronRight className={cn(
                      "h-3.5 w-3.5 text-muted-foreground transition-transform duration-200",
                      isExpanded && "rotate-90"
                    )} />
                    {isExpanded ? <FolderOpen className="h-4 w-4 text-amber-500" /> : <Folder className="h-4 w-4 text-amber-500" />}
                    <span className="font-medium">{category}</span>
                    <span className="ml-auto text-xs text-muted-foreground">{items.length}</span>
                  </button>
                  <div className={cn(
                    "ml-3 pl-2 border-l border-border/50 overflow-hidden transition-all duration-200",
                    isExpanded ? "max-h-[1000px] opacity-100" : "max-h-0 opacity-0"
                  )}>
                    {items.map((item) => (
                      <button
                        key={item.path}
                        onClick={() => {
                          setSelectedComponent(item);
                          setBreadcrumb([
                            { level: 'root', label: 'cvlabkit', path: '' },
                            { level: 'package', label: item.category, path: item.category },
                            { level: 'file', label: `${item.name}.py`, path: item.path },
                          ]);
                        }}
                        className={cn(
                          "w-full flex items-center gap-1.5 px-2 py-1 rounded text-sm transition-colors duration-200",
                          selectedComponent?.path === item.path ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
                        )}
                      >
                        <FileCode2 className="h-3.5 w-3.5 text-blue-500 flex-shrink-0" />
                        <span className="truncate">{item.name}.py</span>
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </Card>
    </div>
  );
}
