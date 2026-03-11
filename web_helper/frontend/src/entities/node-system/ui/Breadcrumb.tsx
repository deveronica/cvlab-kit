/**
 * Breadcrumb - Simulink-style hierarchical navigation
 *
 * Allows drill-down into subsystems and navigation back to parent levels.
 * Example: Agent > setup() > model (internals)
 */

import { memo } from 'react';
import { ChevronRight, Home } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import type { BreadcrumbItem } from '@/entities/node-system/model/types';

interface BreadcrumbProps {
  items: BreadcrumbItem[];
  onNavigate: (index: number) => void;
  className?: string;
}

export const Breadcrumb = memo(function Breadcrumb({
  items,
  onNavigate,
  className,
}: BreadcrumbProps) {
  if (items.length === 0) return null;

  return (
    <nav
      className={cn(
        'flex items-center gap-1 px-3 py-2 bg-background border rounded-lg shadow-sm text-sm',
        className
      )}
      aria-label="Breadcrumb navigation"
    >
      {items.map((item, index) => {
        const isLast = index === items.length - 1;
        const isFirst = index === 0;

        return (
          <div key={`${item.path}-${index}`} className="flex items-center">
            {/* Separator */}
            {!isFirst && (
              <ChevronRight className="h-4 w-4 text-muted-foreground mx-1 flex-shrink-0" />
            )}

            {/* Breadcrumb item */}
            {isLast ? (
              // Current location (not clickable)
              <span className="font-medium text-foreground flex items-center gap-1.5">
                {isFirst && <Home className="h-3.5 w-3.5" />}
                {item.label}
              </span>
            ) : (
              // Parent location (clickable)
              <button
                onClick={() => onNavigate(index)}
                className={cn(
                  'text-muted-foreground hover:text-foreground transition-colors',
                  'flex items-center gap-1.5 hover:underline'
                )}
              >
                {isFirst && <Home className="h-3.5 w-3.5" />}
                {item.label}
              </button>
            )}
          </div>
        );
      })}
    </nav>
  );
});

Breadcrumb.displayName = 'Breadcrumb';
