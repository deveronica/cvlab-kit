import React from "react";

import { memo} from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { flexRender, getCoreRowModel, useReactTable, ColumnDef, Row } from '@tanstack/react-table';

interface VirtualTableProps<T> {
  data: T[];
  columns: ColumnDef<T, any>[];
  height?: number;
  itemHeight?: number;
  overscan?: number;
  onRowClick?: (row: T) => void;
  className?: string;
  headerClassName?: string;
  rowClassName?: string | ((row: T, index: number) => string);
  loading?: boolean;
  emptyMessage?: string;
}

function VirtualTable<T>({
  data,
  columns,
  height = 500,
  itemHeight = 48,
  overscan = 5,
  onRowClick,
  className = '',
  headerClassName = '',
  rowClassName = '',
  loading = false,
  emptyMessage = 'No data available'
}: VirtualTableProps<T>) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  const { rows } = table.getRowModel();

  const parentRef = React.useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => itemHeight,
    overscan,
  });

  const items = virtualizer.getVirtualItems();

  const getRowClassName = (row: T, index: number) => {
    const baseClass = 'flex items-center border-b border-border transition-colors hover:bg-muted/50';
    if (typeof rowClassName === 'function') {
      return `${baseClass} ${rowClassName(row, index)}`;
    }
    return `${baseClass} ${rowClassName}`;
  };

  if (loading) {
    return (
      <div className={`border rounded-lg ${className}`} style={{ height }}>
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin h-8 w-8 border-b-2 border-primary rounded-full" />
          <span className="ml-2 text-muted-foreground">Loading...</span>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className={`border rounded-lg ${className}`} style={{ height }}>
        <div className="flex items-center justify-center h-full">
          <span className="text-muted-foreground">{emptyMessage}</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`border rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className={`flex bg-muted/50 border-b border-border ${headerClassName}`}>
        {table.getHeaderGroups().map(headerGroup =>
          headerGroup.headers.map(header => (
            <div
              key={header.id}
              className="flex items-center px-4 py-3 text-sm font-medium text-left"
              style={{ width: header.getSize() }}
            >
              {header.isPlaceholder
                ? null
                : flexRender(header.column.columnDef.header, header.getContext())}
            </div>
          ))
        )}
      </div>

      {/* Virtual Scrolling Body */}
      <div
        ref={parentRef}
        className="overflow-auto"
        style={{ height }}
      >
        <div
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            width: '100%',
            position: 'relative',
          }}
        >
          {items.map((virtualItem) => {
            const row = rows[virtualItem.index] as Row<T>;
            return (
              <div
                key={virtualItem.key}
                className={getRowClassName(row.original, virtualItem.index)}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: `${virtualItem.size}px`,
                  transform: `translateY(${virtualItem.start}px)`,
                  cursor: onRowClick ? 'pointer' : 'default'
                }}
                onClick={() => onRowClick?.(row.original)}
              >
                {row.getVisibleCells().map(cell => (
                  <div
                    key={cell.id}
                    className="flex items-center px-4 py-3 text-sm"
                    style={{ width: cell.column.getSize() }}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer Info */}
      <div className="flex items-center justify-between px-4 py-2 bg-muted/30 border-t border-border text-xs text-muted-foreground">
        <span>
          Showing {Math.min(items.length, data.length)} of {data.length} rows
        </span>
        <span>
          Virtual scrolling active • {virtualizer.getVirtualItems().length} rendered
        </span>
      </div>
    </div>
  );
}

const MemoizedVirtualTable = memo(VirtualTable) as typeof VirtualTable;

export { MemoizedVirtualTable as VirtualTable };