import React from "react";

/**
 * Virtualized TanStack Table for large datasets
 */

import { useMemo } from 'react';
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnFiltersState,
  type SortingState,
  type VisibilityState,
} from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '../../lib/utils';

interface VirtualizedDataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  height?: number;
  itemHeight?: number;
  overscan?: number;
  enableSorting?: boolean;
  enableFiltering?: boolean;
  className?: string;
  headerClassName?: string;
  rowClassName?: string | ((row: TData, index: number) => string);
  loading?: boolean;
  emptyMessage?: string;
  onRowClick?: (row: TData) => void;
}

export function VirtualizedDataTable<TData, TValue>({
  columns,
  data,
  height = 600,
  itemHeight = 50,
  overscan = 10,
  enableSorting = true,
  enableFiltering = true,
  className,
  headerClassName,
  rowClassName,
  loading = false,
  emptyMessage = 'No results found.',
  onRowClick,
}: VirtualizedDataTableProps<TData, TValue>) {
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([]);
  const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({});

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: enableSorting ? getSortedRowModel() : undefined,
    getFilteredRowModel: enableFiltering ? getFilteredRowModel() : undefined,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
    },
  });

  const { rows } = table.getRowModel();

  const parentRef = React.useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => itemHeight,
    overscan: overscan,
  });

  const virtualItems = virtualizer.getVirtualItems();

  const totalSize = virtualizer.getTotalSize();
  const paddingTop = virtualItems.length > 0 ? virtualItems[0]?.start || 0 : 0;
  const paddingBottom =
    virtualItems.length > 0
      ? totalSize - (virtualItems[virtualItems.length - 1]?.end || 0)
      : 0;

  const getRowClassName = (row: TData, index: number) => {
    const baseClass = 'flex border-b transition-colors hover:bg-muted/50';
    if (typeof rowClassName === 'function') {
      return `${baseClass} ${rowClassName(row, index)}`;
    }
    return `${baseClass} ${rowClassName}`;
  };

  if (loading) {
    return (
      <div className={cn('rounded-md border', className)} style={{ height }}>
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin h-8 w-8 border-b-2 border-primary rounded-full" />
          <span className="ml-2 text-muted-foreground">Loading...</span>
        </div>
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className={cn('rounded-md border', className)} style={{ height }}>
        <div className="flex items-center justify-center h-full text-muted-foreground">
          {emptyMessage}
        </div>
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {enableFiltering && (
        <div className="flex items-center space-x-2">
          <input
            placeholder="Filter rows..."
            value={
              (table.getColumn('run_name')?.getFilterValue() as string) ??
              (table.getColumn('host_id')?.getFilterValue() as string) ??
              (table.getColumn('project')?.getFilterValue() as string) ??
              ''
            }
            onChange={(event) => {
              const firstTextColumn = table.getAllColumns().find(
                (column) => column.getCanFilter()
              );
              if (firstTextColumn) {
                firstTextColumn.setFilterValue(event.target.value);
              }
            }}
            className="max-w-sm px-3 py-2 border border-input rounded-md"
          />
          <span className="text-sm text-muted-foreground">
            {table.getFilteredRowModel().rows.length} rows
          </span>
        </div>
      )}

      <div className="rounded-md border">
        <div className={cn('border-b bg-muted/50', headerClassName)}>
          {table.getHeaderGroups().map((headerGroup) => (
            <div key={headerGroup.id} className="flex">
              {headerGroup.headers.map((header) => (
                <div
                  key={header.id}
                  className="flex items-center h-12 px-4 font-medium text-left border-r last:border-r-0"
                  style={{ width: header.getSize() }}
                >
                  {header.isPlaceholder ? null : (
                    <div
                      className={cn(
                        'flex items-center space-x-2',
                        enableSorting && header.column.getCanSort() && 'cursor-pointer select-none'
                      )}
                      onClick={
                        enableSorting && header.column.getCanSort()
                          ? header.column.getToggleSortingHandler()
                          : undefined
                      }
                    >
                      <span>
                        {flexRender(header.column.columnDef.header, header.getContext())}
                      </span>
                      {enableSorting && header.column.getCanSort() && (
                        <div className="flex flex-col">
                          <ChevronUp
                            className={cn(
                              'h-3 w-3',
                              header.column.getIsSorted() === 'asc'
                                ? 'text-foreground'
                                : 'text-muted-foreground/30'
                            )}
                          />
                          <ChevronDown
                            className={cn(
                              'h-3 w-3 -mt-1',
                              header.column.getIsSorted() === 'desc'
                                ? 'text-foreground'
                                : 'text-muted-foreground/30'
                            )}
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>

        <div
          ref={parentRef}
          className="overflow-auto"
          style={{ height: `${height}px` }}
        >
          <div style={{ height: `${totalSize}px`, position: 'relative' }}>
            {paddingTop > 0 && <div style={{ height: `${paddingTop}px` }} />}

            {virtualItems.map((virtualItem) => {
              const row = rows[virtualItem.index];
              return (
                <div
                  key={row.id}
                  data-index={virtualItem.index}
                  ref={virtualizer.measureElement}
                  className={getRowClassName(row.original, virtualItem.index)}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    transform: `translateY(${virtualItem.start}px)`,
                  }}
                  onClick={onRowClick ? () => onRowClick(row.original) : undefined}
                >
                  {row.getVisibleCells().map((cell) => (
                    <div
                      key={cell.id}
                      className="flex items-center px-4 py-2 border-r last:border-r-0"
                      style={{ width: cell.column.getSize() }}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </div>
                  ))}
                </div>
              );
            })}

            {paddingBottom > 0 && <div style={{ height: `${paddingBottom}px` }} />}
          </div>
        </div>

        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Showing {virtualItems.length} of {rows.length} rows</span>
          <span>Virtual scrolling enabled for performance</span>
        </div>
      </div>
    </div>
  );
}