/**
 * HierarchyBreadcrumb - Simulink-style Navigation Breadcrumb
 *
 * Displays the logical hierarchy path for drill-down navigation:
 * Agent → train_step → Model → Layer3
 *
 * Features:
 * - Click to navigate to any level
 * - Type-based icons (agent, method, component, layer, operation)
 * - Current level highlighting
 * - Collapsible for deep hierarchies
 */

import React, { memo } from 'react';
import {
  ChevronRight,
  Home,
  Play,
  Box,
  Layers,
  Cpu,
  MoreHorizontal,
} from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { BreadcrumbItem } from '@/shared/model/node-graph';
import {
  Breadcrumb,
  BreadcrumbItem as BreadcrumbItemUI,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/shared/ui/breadcrumb';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/shared/ui/dropdown-menu';
import { Button } from '@/shared/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/shared/ui/tooltip';

interface HierarchyBreadcrumbProps {
  /** Breadcrumb items from the current graph hierarchy */
  items: BreadcrumbItem[];
  /** Callback when a breadcrumb item is clicked */
  onNavigate: (item: BreadcrumbItem) => void;
  /** Maximum visible items before collapsing */
  maxVisible?: number;
  /** Current depth (for highlighting) */
  currentDepth?: number;
}

// Icon mapping for breadcrumb types
const typeIcons: Record<string, React.ElementType> = {
  agent: Play,
  method: Cpu,
  component: Box,
  layer: Layers,
  operation: Cpu,
};

// Color mapping for breadcrumb types
const typeColors: Record<string, string> = {
  agent: 'text-purple-500',
  method: 'text-blue-500',
  component: 'text-green-500',
  layer: 'text-orange-500',
  operation: 'text-cyan-500',
};

/**
 * Single breadcrumb item with icon
 */
const BreadcrumbItemContent = memo(
  ({
    item,
    isLast,
    onClick,
  }: {
    item: BreadcrumbItem;
    isLast: boolean;
    onClick?: () => void;
  }) => {
    const Icon = typeIcons[item.type] || Box;
    const colorClass = typeColors[item.type] || 'text-muted-foreground';

    const content = (
      <div className="flex items-center gap-1.5">
        <Icon className={cn('h-3.5 w-3.5', colorClass)} />
        <span className="truncate max-w-[120px]">{item.label}</span>
      </div>
    );

    if (isLast) {
      return (
        <BreadcrumbPage className="font-medium">{content}</BreadcrumbPage>
      );
    }

    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <BreadcrumbLink
              onClick={onClick}
              className="cursor-pointer hover:text-foreground transition-colors duration-200"
            >
              {content}
            </BreadcrumbLink>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <span className="text-xs">
              Navigate to {item.type}: {item.label}
            </span>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }
);
BreadcrumbItemContent.displayName = 'BreadcrumbItemContent';

/**
 * Collapsed items dropdown
 */
const CollapsedItems = memo(
  ({
    items,
    onNavigate,
  }: {
    items: BreadcrumbItem[];
    onNavigate: (item: BreadcrumbItem) => void;
  }) => {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0 hover:bg-muted"
          >
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {items.map((item) => {
            const Icon = typeIcons[item.type] || Box;
            const colorClass = typeColors[item.type] || 'text-muted-foreground';
            return (
              <DropdownMenuItem
                key={item.node_id}
                onClick={() => onNavigate(item)}
                className="flex items-center gap-2"
              >
                <Icon className={cn('h-4 w-4', colorClass)} />
                <span>{item.label}</span>
                <span className="text-xs text-muted-foreground ml-auto">
                  {item.type}
                </span>
              </DropdownMenuItem>
            );
          })}
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }
);
CollapsedItems.displayName = 'CollapsedItems';

/**
 * Main HierarchyBreadcrumb Component
 */
export const HierarchyBreadcrumb = memo(
  ({ items, onNavigate, maxVisible = 4 }: HierarchyBreadcrumbProps) => {
    // If no items, show root
    if (!items || items.length === 0) {
      return (
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItemUI>
              <BreadcrumbPage className="flex items-center gap-1.5 text-muted-foreground">
                <Home className="h-3.5 w-3.5" />
                <span>Root</span>
              </BreadcrumbPage>
            </BreadcrumbItemUI>
          </BreadcrumbList>
        </Breadcrumb>
      );
    }

    // Determine which items to show and which to collapse
    const shouldCollapse = items.length > maxVisible;
    const visibleItems = shouldCollapse
      ? [...items.slice(0, 1), ...items.slice(-Math.max(maxVisible - 2, 1))]
      : items;
    const collapsedItems = shouldCollapse
      ? items.slice(1, -Math.max(maxVisible - 2, 1))
      : [];

    return (
      <Breadcrumb className="flex-1">
        <BreadcrumbList className="flex-wrap">
          {/* Root/Home */}
          <BreadcrumbItemUI>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <BreadcrumbLink
                    onClick={() =>
                      onNavigate({
                        type: 'root',
                        label: 'Root',
                        node_id: '',
                        graph_id: '',
                      })
                    }
                    className="cursor-pointer hover:text-foreground transition-colors duration-200"
                  >
                    <Home className="h-3.5 w-3.5" />
                  </BreadcrumbLink>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  <span className="text-xs">Return to root</span>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </BreadcrumbItemUI>

          <BreadcrumbSeparator>
            <ChevronRight className="h-3.5 w-3.5" />
          </BreadcrumbSeparator>

          {/* First item (always visible) */}
          <BreadcrumbItemUI>
            <BreadcrumbItemContent
              item={visibleItems[0]}
              isLast={visibleItems.length === 1}
              onClick={() => onNavigate(visibleItems[0])}
            />
          </BreadcrumbItemUI>

          {/* Collapsed items dropdown */}
          {shouldCollapse && collapsedItems.length > 0 && (
            <>
              <BreadcrumbSeparator>
                <ChevronRight className="h-3.5 w-3.5" />
              </BreadcrumbSeparator>
              <BreadcrumbItemUI>
                <CollapsedItems items={collapsedItems} onNavigate={onNavigate} />
              </BreadcrumbItemUI>
            </>
          )}

          {/* Remaining visible items */}
          {visibleItems.slice(1).map((item, index) => (
            <React.Fragment key={item.node_id}>
              <BreadcrumbSeparator>
                <ChevronRight className="h-3.5 w-3.5" />
              </BreadcrumbSeparator>
              <BreadcrumbItemUI>
                <BreadcrumbItemContent
                  item={item}
                  isLast={index === visibleItems.length - 2}
                  onClick={() => onNavigate(item)}
                />
              </BreadcrumbItemUI>
            </React.Fragment>
          ))}
        </BreadcrumbList>
      </Breadcrumb>
    );
  }
);

HierarchyBreadcrumb.displayName = 'HierarchyBreadcrumb';

export default HierarchyBreadcrumb;
