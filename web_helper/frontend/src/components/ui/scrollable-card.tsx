import React from "react";

/**
 * Standardized Scrollable Card Component
 *
 * Card with proper padding that contains scrollable content.
 * Ensures padding is outside the scroll area.
 */

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';
import { cn } from '../../lib/utils';

export interface ScrollableCardProps {
  title?: string;
  description?: string;
  children: React.ReactNode;
  variant?: 'default' | 'compact';
  _maxHeight?: string;
  /** Custom header content (overrides title/description) */
  headerContent?: React.ReactNode;
  /** Additional actions in header (e.g., buttons) */
  headerActions?: React.ReactNode;
  className?: string;
  contentClassName?: string;
}

/**
 * Card with scrollable content area while maintaining padding
 *
 * @example
 * ```tsx
 * <ScrollableCard
 *   title="Metrics Summary"
 *   description="Statistical summary of metrics"
 *   _maxHeight="500px"
 * >
 *   <DataTableContainer>
 *     <table>...</table>
 *   </DataTableContainer>
 * </ScrollableCard>
 * ```
 */
export function ScrollableCard({
  title,
  description,
  children,
  variant = 'default',
  _maxHeight = '500px',
  headerContent,
  headerActions,
  className,
  contentClassName,
}: ScrollableCardProps) {
  return (
    <Card variant={variant} className={className}>
      {(title || description || headerContent || headerActions) && (
        <CardHeader variant={variant}>
          {headerContent ? (
            headerContent
          ) : (
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                {title && <CardTitle size={variant === 'compact' ? 'base' : 'default'}>{title}</CardTitle>}
                {description && <CardDescription>{description}</CardDescription>}
              </div>
              {headerActions && (
                <div className="flex items-center gap-2 flex-shrink-0 ml-4">
                  {headerActions}
                </div>
              )}
            </div>
          )}
        </CardHeader>
      )}
      <CardContent variant={variant} className={contentClassName}>
        {children}
      </CardContent>
    </Card>
  );
}

/**
 * Scrollable section inside a card
 * Use when you need scrolling within CardContent but want to keep padding
 */
export function ScrollableSection({
  children,
  _maxHeight = '400px',
  className,
  horizontal = false,
  vertical = true,
}: {
  children: React.ReactNode;
  _maxHeight?: string;
  className?: string;
  horizontal?: boolean;
  vertical?: boolean;
}) {
  return (
    <div
      className={cn(
        'relative',
        horizontal && 'overflow-x-auto',
        vertical && 'overflow-y-auto',
        className
      )}
      style={{ _maxHeight }}
    >
      {children}
    </div>
  );
}
