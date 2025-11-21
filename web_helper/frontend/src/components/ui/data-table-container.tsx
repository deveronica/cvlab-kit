import React from "react";

/**
 * Standardized Data Table Container
 *
 * Provides consistent spacing, scrolling, and sticky headers for all tables.
 * Use this instead of manually creating table wrappers.
 */

import { cn } from '../../lib/utils';

export interface DataTableContainerProps {
  children: React.ReactNode;
  maxHeight?: string;
  className?: string;
  /** Enable horizontal scrolling */
  horizontalScroll?: boolean;
  /** Enable vertical scrolling */
  verticalScroll?: boolean;
}

/**
 * Standard table container with consistent spacing and scrolling
 *
 * @example
 * ```tsx
 * <DataTableContainer maxHeight="500px">
 *   <table className="w-full">
 *     <thead className="sticky top-0 bg-muted">
 *       <tr>
 *         <th>Column</th>
 *       </tr>
 *     </thead>
 *     <tbody>
 *       <tr><td>Data</td></tr>
 *     </tbody>
 *   </table>
 * </DataTableContainer>
 * ```
 */
export function DataTableContainer({
  children,
  maxHeight,
  className,
  horizontalScroll = true,
  verticalScroll = true,
}: DataTableContainerProps) {
  return (
    <div className={cn('rounded-md border', className)}>
      <div
        className={cn(
          'relative',
          horizontalScroll && 'overflow-x-auto',
          verticalScroll && 'overflow-y-auto'
        )}
        style={maxHeight ? { maxHeight } : undefined}
      >
        {children}
      </div>
    </div>
  );
}

/**
 * Standard table with consistent styling
 * Use inside DataTableContainer
 */
export function StandardTable({
  children,
  className,
  minWidth = '600px',
}: {
  children: React.ReactNode;
  className?: string;
  minWidth?: string;
}) {
  return (
    <table
      className={cn('w-full text-sm', className)}
      style={{ minWidth }}
    >
      {children}
    </table>
  );
}

/**
 * Sticky table header with consistent styling
 */
export function StickyTableHeader({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <thead className={cn('sticky top-0 z-10 bg-muted', className)}>
      {children}
    </thead>
  );
}

/**
 * Sticky first column (for row labels)
 */
export function StickyTableCell({
  children,
  className,
  isHeader = false,
  rowSpan,
}: {
  children: React.ReactNode;
  className?: string;
  isHeader?: boolean;
  rowSpan?: number;
}) {
  const Tag = isHeader ? 'th' : 'td';

  return (
    <Tag
      className={cn(
        'p-3 sticky left-0 border-r',
        isHeader ? 'bg-muted font-medium z-30' : 'bg-background z-20',
        className
      )}
      rowSpan={rowSpan}
    >
      {children}
    </Tag>
  );
}
