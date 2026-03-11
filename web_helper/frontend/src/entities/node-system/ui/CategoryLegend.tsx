/**
 * CategoryLegend - compact category legend used across node-system UIs
 *
 * Keeps icon + label styling consistent between Execute/Builder/Editor.
 */

import { memo } from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/shared/ui/tooltip';
import { cn } from '@/shared/lib/utils';
import { getCategoryIcon } from '@/shared/lib/category-registry';
import { getCategoryTheme } from '@/entities/node-system/config/themes';

export interface CategoryLegendItem {
  key: string;
  label: string;
}

export interface CategoryLegendProps {
  items: CategoryLegendItem[];
  className?: string;
  /** Show shortened label (e.g. "Optimizer" -> "Opt") */
  abbreviated?: boolean;
}

export const CategoryLegend = memo(function CategoryLegend({
  items,
  className,
  abbreviated = true,
}: CategoryLegendProps) {
  return (
    <div className={cn('flex items-center gap-1', className)}>
      {items.map(({ key, label }) => {
        const Icon = getCategoryIcon(key);
        const theme = getCategoryTheme(key);
        const shortLabel = abbreviated ? label.slice(0, 3) : label;

        return (
          <Tooltip key={key}>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-muted/50 cursor-default">
                <Icon className={cn('h-3 w-3', theme.icon)} />
                <span className="text-[10px] text-muted-foreground">{shortLabel}</span>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <span className="text-xs">{label}</span>
            </TooltipContent>
          </Tooltip>
        );
      })}
    </div>
  );
});

CategoryLegend.displayName = 'CategoryLegend';
