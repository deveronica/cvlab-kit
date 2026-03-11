/**
 * ComponentPalette - Draggable component list for adding nodes
 *
 * Features:
 * - Category-based component listing
 * - Search/filter
 * - Drag-to-add or click-to-add
 * - Component type icons
 */

import { useState, useMemo, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Database,
  Layers,
  Zap,
  Target,
  Activity,
  Settings,
  TrendingUp,
  Search,
  GripVertical,
  Plus,
  X,
  GitBranch,
  Repeat,
  MessageSquare,
  Workflow,
} from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { Input } from '@/shared/ui/input';
import { Button } from '@/shared/ui/button';
import { Badge } from '@/shared/ui/badge';
import { ScrollArea } from '@/shared/ui/scroll-area';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/shared/ui/sheet';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/shared/ui/tooltip';
import { ComponentCategory } from '@/shared/model/node-graph';

// Category metadata
const CATEGORIES = [
  { key: 'model', label: 'Model', icon: Box, color: 'bg-blue-500' },
  { key: 'dataset', label: 'Dataset', icon: Database, color: 'bg-emerald-500' },
  { key: 'dataloader', label: 'Dataloader', icon: Layers, color: 'bg-teal-500' },
  { key: 'transform', label: 'Transform', icon: Zap, color: 'bg-violet-500' },
  { key: 'optimizer', label: 'Optimizer', icon: TrendingUp, color: 'bg-amber-500' },
  { key: 'loss', label: 'Loss', icon: Target, color: 'bg-rose-500' },
  { key: 'metric', label: 'Metric', icon: Activity, color: 'bg-cyan-500' },
  { key: 'scheduler', label: 'Scheduler', icon: Settings, color: 'bg-orange-500' },
] as const;

// Control Flow blocks (static, not from API)
const CONTROL_FLOW_ITEMS = [
  {
    key: 'if',
    label: 'If',
    icon: GitBranch,
    color: 'bg-orange-500',
    description: 'Conditional branching - executes True or False path',
    nodeType: 'if',
  },
  {
    key: 'loop',
    label: 'For Loop',
    icon: Repeat,
    color: 'bg-cyan-500',
    description: 'Iteration block - loops over items',
    nodeType: 'loop',
  },
  {
    key: 'comment',
    label: 'Comment',
    icon: MessageSquare,
    color: 'bg-yellow-500',
    description: 'Annotation block for documentation',
    nodeType: 'comment',
  },
] as const;

interface ComponentItem {
  name: string;
  category: ComponentCategory;
  description?: string;
  parameters?: string[];
  nodeType?: string; // For control flow items
}

// Control Flow item (static block)
export interface ControlFlowItem {
  key: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  description: string;
  nodeType: string;
}

interface ComponentPaletteProps {
  onAddComponent?: (component: ComponentItem) => void;
  onAddControlFlow?: (item: ControlFlowItem) => void;
  className?: string;
}

// Fetch components from API
// API returns: {data: [{category: "model", components: [{name: "resnet18", ...}]}]}
async function fetchComponents(): Promise<ComponentItem[]> {
  const results: ComponentItem[] = [];

  for (const cat of CATEGORIES) {
    try {
      const response = await fetch(`/api/components?category=${cat.key}`);
      if (!response.ok) continue;

      const data = await response.json();
      // Extract components from the nested structure
      const categoryData = data.data?.[0];
      const components = categoryData?.components || [];

      for (const comp of components) {
        results.push({
          name: comp.name,
          category: cat.key as ComponentCategory,
          description: comp.description || comp.docstring,
          parameters: comp.parameters ? Object.keys(comp.parameters) : [],
        });
      }
    } catch (e) {
      console.error(`Failed to fetch ${cat.key} components:`, e);
    }
  }

  return results;
}

// Component item in the palette
function ComponentItemCard({
  component,
  onAdd,
}: {
  component: ComponentItem;
  onAdd: () => void;
}) {
  const category = CATEGORIES.find((c) => c.key === component.category);
  const Icon = category?.icon || Box;

  return (
    <div
      className={cn(
        'group flex items-center gap-2 px-2 py-1.5 rounded-md',
        'hover:bg-accent cursor-pointer transition-colors',
        'border border-transparent hover:border-border'
      )}
      onClick={onAdd}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData('application/json', JSON.stringify(component));
        e.dataTransfer.effectAllowed = 'copy';
      }}
    >
      {/* Drag handle */}
      <GripVertical className="h-3 w-3 text-muted-foreground/50 group-hover:text-muted-foreground" />

      {/* Category indicator */}
      <div className={cn('w-2 h-2 rounded-full flex-shrink-0', category?.color)} />

      {/* Icon */}
      <Icon className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />

      {/* Name */}
      <span className="text-xs font-medium truncate flex-1">{component.name}</span>

      {/* Add button (visible on hover) */}
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={(e) => {
              e.stopPropagation();
              onAdd();
            }}
          >
            <Plus className="h-3 w-3" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Add to graph</TooltipContent>
      </Tooltip>
    </div>
  );
}

// Control Flow section
function ControlFlowSection({
  onAdd,
  expanded,
  onToggle,
}: {
  onAdd: (item: ControlFlowItem) => void;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="mb-2">
      <button
        onClick={onToggle}
        className={cn(
          'w-full flex items-center gap-2 px-2 py-1.5 rounded-md',
          'text-xs font-medium text-muted-foreground',
          'hover:bg-accent hover:text-foreground transition-colors'
        )}
      >
        <div className="w-2 h-2 rounded-full bg-gray-500" />
        <Workflow className="h-3.5 w-3.5" />
        <span className="flex-1 text-left">Control Flow</span>
        <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
          {CONTROL_FLOW_ITEMS.length}
        </Badge>
      </button>

      {expanded && (
        <div className="ml-2 mt-1 space-y-0.5">
          {CONTROL_FLOW_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.key}
                className={cn(
                  'group flex items-center gap-2 px-2 py-1.5 rounded-md',
                  'hover:bg-accent cursor-pointer transition-colors',
                  'border border-transparent hover:border-border'
                )}
                onClick={() => onAdd(item as ControlFlowItem)}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData(
                    'application/json',
                    JSON.stringify({ nodeType: item.nodeType, isControlFlow: true })
                  );
                  e.dataTransfer.effectAllowed = 'copy';
                }}
              >
                {/* Drag handle */}
                <GripVertical className="h-3 w-3 text-muted-foreground/50 group-hover:text-muted-foreground" />

                {/* Category indicator */}
                <div className={cn('w-2 h-2 rounded-full flex-shrink-0', item.color)} />

                {/* Icon */}
                <Icon className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />

                {/* Name */}
                <span className="text-xs font-medium truncate flex-1">{item.label}</span>

                {/* Add button (visible on hover) */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={(e) => {
                        e.stopPropagation();
                        onAdd(item as ControlFlowItem);
                      }}
                    >
                      <Plus className="h-3 w-3" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{item.description}</TooltipContent>
                </Tooltip>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Category section
function CategorySection({
  category,
  components,
  onAdd,
  expanded,
  onToggle,
}: {
  category: (typeof CATEGORIES)[number];
  components: ComponentItem[];
  onAdd: (component: ComponentItem) => void;
  expanded: boolean;
  onToggle: () => void;
}) {
  const Icon = category.icon;

  if (components.length === 0) return null;

  return (
    <div className="mb-2">
      <button
        onClick={onToggle}
        className={cn(
          'w-full flex items-center gap-2 px-2 py-1.5 rounded-md',
          'text-xs font-medium text-muted-foreground',
          'hover:bg-accent hover:text-foreground transition-colors'
        )}
      >
        <div className={cn('w-2 h-2 rounded-full', category.color)} />
        <Icon className="h-3.5 w-3.5" />
        <span className="flex-1 text-left">{category.label}</span>
        <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
          {components.length}
        </Badge>
      </button>

      {expanded && (
        <div className="ml-2 mt-1 space-y-0.5">
          {components.map((comp) => (
            <ComponentItemCard
              key={comp.name}
              component={comp}
              onAdd={() => onAdd(comp)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function ComponentPalette({ onAddComponent, onAddControlFlow, className }: ComponentPaletteProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['control-flow', ...CATEGORIES.map((c) => c.key)])
  );

  // Fetch components
  const { data: components = [], isLoading } = useQuery({
    queryKey: ['builder-components-palette'],
    queryFn: fetchComponents,
    staleTime: 60000,
  });

  // Filter components
  const filteredComponents = useMemo(() => {
    if (!search.trim()) return components;

    const lower = search.toLowerCase();
    return components.filter(
      (c) =>
        c.name.toLowerCase().includes(lower) ||
        c.category.toLowerCase().includes(lower) ||
        c.description?.toLowerCase().includes(lower)
    );
  }, [components, search]);

  // Group by category
  const groupedComponents = useMemo(() => {
    const groups: Record<string, ComponentItem[]> = {};
    for (const cat of CATEGORIES) {
      groups[cat.key] = filteredComponents.filter((c) => c.category === cat.key);
    }
    return groups;
  }, [filteredComponents]);

  // Handle add component
  const handleAdd = useCallback(
    (component: ComponentItem) => {
      onAddComponent?.(component);
      // Don't close sheet to allow adding multiple
    },
    [onAddComponent]
  );

  // Handle add control flow
  const handleAddControlFlow = useCallback(
    (item: ControlFlowItem) => {
      onAddControlFlow?.(item);
    },
    [onAddControlFlow]
  );

  // Toggle category
  const toggleCategory = useCallback((key: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  return (
    <TooltipProvider>
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetTrigger asChild>
          <Button variant="outline" size="sm" className={cn('gap-1.5', className)}>
            <Plus className="h-3.5 w-3.5" />
            Add Component
          </Button>
        </SheetTrigger>
        <SheetContent side="right" className="w-80 p-0">
          <SheetHeader className="px-4 py-3 border-b">
            <SheetTitle className="text-base">Component Palette</SheetTitle>
            <SheetDescription className="text-xs">
              Click or drag components to add them to the graph
            </SheetDescription>
          </SheetHeader>

          {/* Search */}
          <div className="px-4 py-2 border-b">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                placeholder="Search components..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="h-8 pl-8 text-xs"
              />
              {search && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute right-1 top-1/2 -translate-y-1/2 h-6 w-6"
                  onClick={() => setSearch('')}
                >
                  <X className="h-3 w-3" />
                </Button>
              )}
            </div>
          </div>

          {/* Component list */}
          <ScrollArea className="flex-1 h-[calc(100vh-160px)]">
            <div className="p-2">
              {/* Control Flow section (always shown) */}
              <ControlFlowSection
                onAdd={handleAddControlFlow}
                expanded={expandedCategories.has('control-flow')}
                onToggle={() => toggleCategory('control-flow')}
              />

              {/* Divider */}
              <div className="my-2 border-t border-border" />

              {/* Component categories */}
              {isLoading ? (
                <div className="text-center py-8 text-muted-foreground">
                  <div className="animate-pulse">Loading components...</div>
                </div>
              ) : filteredComponents.length === 0 && search ? (
                <div className="text-center py-8 text-muted-foreground">
                  <p className="text-xs">No components found</p>
                </div>
              ) : (
                CATEGORIES.map((cat) => (
                  <CategorySection
                    key={cat.key}
                    category={cat}
                    components={groupedComponents[cat.key] || []}
                    onAdd={handleAdd}
                    expanded={expandedCategories.has(cat.key)}
                    onToggle={() => toggleCategory(cat.key)}
                  />
                ))
              )}
            </div>
          </ScrollArea>

          {/* Footer */}
          <div className="px-4 py-2 border-t bg-muted/30">
            <p className="text-[10px] text-muted-foreground">
              {filteredComponents.length} components available
            </p>
          </div>
        </SheetContent>
      </Sheet>
    </TooltipProvider>
  );
}

// Inline palette (for embedding in BuilderEditPane)
export function InlineComponentPalette({
  onAddComponent,
  onAddControlFlow,
  className,
}: ComponentPaletteProps) {
  const [search, setSearch] = useState('');

  // Filter control flow items
  const filteredControlFlow = useMemo(() => {
    if (!search.trim()) return CONTROL_FLOW_ITEMS;
    const lower = search.toLowerCase();
    return CONTROL_FLOW_ITEMS.filter(
      (item) =>
        item.label.toLowerCase().includes(lower) ||
        item.description.toLowerCase().includes(lower)
    );
  }, [search]);

  const handleAdd = useCallback(
    (category: (typeof CATEGORIES)[number]) => {
      onAddComponent?.({
        name: `new_${category.key}`,
        category: category.key as ComponentCategory,
        description: `Template ${category.label} component`,
      });
    },
    [onAddComponent]
  );

  const handleAddControlFlow = useCallback(
    (item: ControlFlowItem) => {
      onAddControlFlow?.(item);
    },
    [onAddControlFlow]
  );

  return (
    <div className={cn('space-y-2', className)}>
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
        <Input
          placeholder="Search items..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-7 pl-7 text-xs"
        />
      </div>

      {/* Control Flow items */}
      <div className="text-[10px] font-medium text-muted-foreground px-1 uppercase tracking-wider">
        Logic Blocks
      </div>
      <div className="space-y-0.5">
        {filteredControlFlow.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={`control-${item.key}`}
              onClick={() => handleAddControlFlow(item as ControlFlowItem)}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData(
                  'application/json',
                  JSON.stringify({ nodeType: item.nodeType, isControlFlow: true })
                );
                e.dataTransfer.effectAllowed = 'copy';
              }}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded hover:bg-accent transition-colors text-left border border-transparent hover:border-border group"
            >
              <div className={cn('w-1 h-6 rounded-full flex-shrink-0', item.color)} />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium truncate">{item.label}</div>
              </div>
              <Plus className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100" />
            </button>
          );
        })}
      </div>

      <div className="border-t border-border my-2" />

      {/* Abstract Component Templates */}
      <div className="text-[10px] font-medium text-muted-foreground px-1 uppercase tracking-wider">
        Component Templates
      </div>
      <div className="space-y-0.5">
        {CATEGORIES.map((cat) => {
          const Icon = cat.icon;
          return (
            <button
              key={`template-${cat.key}`}
              onClick={() => handleAdd(cat)}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData(
                  'application/json',
                  JSON.stringify({
                    name: `new_${cat.key}`,
                    category: cat.key,
                  })
                );
                e.dataTransfer.effectAllowed = 'copy';
              }}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded hover:bg-accent transition-colors text-left border border-transparent hover:border-border group"
            >
              <div className={cn('w-1 h-6 rounded-full flex-shrink-0', cat.color)} />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium truncate">{cat.label}</div>
              </div>
              <Plus className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100" />
            </button>
          );
        })}
      </div>
    </div>
  );
}
