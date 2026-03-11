import React, { Fragment } from 'react';
import { ChevronRight } from 'lucide-react';
import { Button } from '@/shared/ui/button';
import { cn } from '@/shared/lib/utils';

export type BreadcrumbLevel = 'root' | 'package' | 'subpackage' | 'file' | 'symbol' | 'method' | 'node';

export interface BreadcrumbItem {
  level: BreadcrumbLevel;
  label: string;
  path: string;
}

interface BreadCrumbProps {
  items: BreadcrumbItem[];
  onNavigate: (item: BreadcrumbItem, index: number) => void;
}

export function BreadCrumb({ items, onNavigate }: BreadCrumbProps) {
  return (
    <nav className="flex items-center gap-2 px-6 py-3 bg-muted/50 border-b border-border">
      {items.map((item, index) => (
        <Fragment key={`${item.path}-${index}`}>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onNavigate(item, index)}
            className={cn(
              "h-7 px-2 text-sm font-medium transition-colors duration-200",
              index === items.length - 1
                ? "text-foreground cursor-default"
                : "text-muted-foreground hover:text-foreground"
            )}
            disabled={index === items.length - 1}
          >
            {item.label}
          </Button>
          {index < items.length - 1 && (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
        </Fragment>
      ))}
    </nav>
  );
}
