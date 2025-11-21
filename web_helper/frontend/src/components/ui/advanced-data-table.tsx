import React from "react";

/**
 * Advanced TanStack Table with column pinning, global search, and enhanced features
 * Designed for Projects View with complex hyperparameter/metrics structure
 */

import { useState, useEffect, useCallback } from 'react';
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnFiltersState,
  type SortingState,
  type VisibilityState,
  type ColumnPinningState,
  type ColumnSizingState,
} from '@tanstack/react-table';
import { ChevronDown, ChevronUp, Search, Settings2, Download, Maximize2 } from 'lucide-react';
import { cn } from '../../lib/utils';
import { Input } from './input';
import { Button } from './button';
import { Popover, PopoverContent, PopoverTrigger } from './popover';
import { devWarn } from '../../lib/dev-utils';

interface AdvancedDataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  // Table features
  enableSorting?: boolean;
  enableFiltering?: boolean;
  enablePagination?: boolean;
  enableRowSelection?: boolean;
  enableGlobalFilter?: boolean;
  enableColumnPinning?: boolean;
  // Styling
  className?: string;
  // Pagination
  pageSize?: number;
  // Initial state
  initialPinning?: ColumnPinningState;
  initialVisibility?: VisibilityState;
  initialSizing?: ColumnSizingState;
  // Table ID for localStorage
  tableId?: string;
  // Callbacks
  onRowClick?: (row: TData) => void;
  onRowSelectionChange?: (selectedRows: TData[]) => void;
  onColumnVisibilityChange?: (visibility: VisibilityState) => void;
  onRowEdit?: (row: TData, field: string, value: any) => void;
  emptyStateComponent?: React.ReactNode;
}

export function AdvancedDataTable<TData, TValue>({
  columns,
  data,
  enableSorting = true,
  enableFiltering = true,
  enablePagination = false,
  enableRowSelection = false,
  enableGlobalFilter = true,
  enableColumnPinning = false,
  className,
  pageSize = 50,
  initialPinning,
  initialVisibility,
  initialSizing,
  tableId = 'default-table',
  onRowClick,
  onRowSelectionChange,
  onColumnVisibilityChange,
  onRowEdit,
  emptyStateComponent,
}: AdvancedDataTableProps<TData, TValue>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(initialVisibility || {});
  const [rowSelection, setRowSelection] = useState({});
  const [columnPinning, setColumnPinning] = useState<ColumnPinningState>(initialPinning || {});
  const [globalFilter, setGlobalFilter] = useState('');
  const [columnSizing, setColumnSizing] = useState<ColumnSizingState>(() => {
    // Try to load from localStorage first
    if (tableId) {
      try {
        const saved = localStorage.getItem(`table-sizing-${tableId}`);
        if (saved) {
          return { ...initialSizing, ...JSON.parse(saved) };
        }
      } catch (e) {
        devWarn('Failed to load column sizing from localStorage', e);
      }
    }
    return initialSizing || {};
  });

  // Save column sizing to localStorage
  useEffect(() => {
    if (tableId && Object.keys(columnSizing).length > 0) {
      try {
        localStorage.setItem(`table-sizing-${tableId}`, JSON.stringify(columnSizing));
      } catch (e) {
        devWarn('Failed to save column sizing to localStorage', e);
      }
    }
  }, [columnSizing, tableId]);

  // Reset column sizing to default
  const resetColumnSizing = useCallback(() => {
    setColumnSizing(initialSizing || {});
    if (tableId) {
      try {
        localStorage.removeItem(`table-sizing-${tableId}`);
      } catch (e) {
        devWarn('Failed to remove column sizing from localStorage', e);
      }
    }
  }, [initialSizing, tableId]);

  // Export functions
  const exportToCSV = () => {
    const visibleColumns = table.getVisibleLeafColumns();
    const rows = table.getFilteredRowModel().rows;

    // Create CSV header
    const headers = visibleColumns.map(col => {
      const header = col.columnDef.header;
      return typeof header === 'string' ? header : col.id;
    });

    // Create CSV rows
    const csvRows = rows.map(row => {
      return visibleColumns.map(col => {
        const cellValue = row.getValue(col.id);
        // Escape quotes and wrap in quotes if contains comma
        const stringValue = String(cellValue ?? '');
        return stringValue.includes(',') ? `"${stringValue.replace(/"/g, '""')}"` : stringValue;
      });
    });

    // Combine header and rows
    const csv = [headers.join(','), ...csvRows.map(row => row.join(','))].join('\n');

    // Download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `export_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const exportToJSON = () => {
    const visibleColumns = table.getVisibleLeafColumns();
    const rows = table.getFilteredRowModel().rows;

    const jsonData = rows.map(row => {
      const rowData: Record<string, any> = {};
      visibleColumns.forEach(col => {
        const header = col.columnDef.header;
        const key = typeof header === 'string' ? header : col.id;
        rowData[key] = row.getValue(col.id);
      });
      return rowData;
    });

    const json = JSON.stringify(jsonData, null, 2);

    // Download
    const blob = new Blob([json], { type: 'application/json;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `export_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: enablePagination ? getPaginationRowModel() : undefined,
    getSortedRowModel: enableSorting ? getSortedRowModel() : undefined,
    getFilteredRowModel: enableFiltering ? getFilteredRowModel() : undefined,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: (updater) => {
      setColumnVisibility(updater);
      if (onColumnVisibilityChange) {
        const newVisibility = typeof updater === 'function' ? updater(columnVisibility) : updater;
        onColumnVisibilityChange(newVisibility);
      }
    },
    onRowSelectionChange: (updater) => {
      setRowSelection(updater);
      if (onRowSelectionChange) {
        const newSelection = typeof updater === 'function' ? updater(rowSelection) : updater;
        const selectedRowIds = Object.keys(newSelection).filter(key => newSelection[key]);
        const selectedData = selectedRowIds.map(id => data[parseInt(id)]).filter(Boolean);
        onRowSelectionChange(selectedData);
      }
    },
    onColumnPinningChange: setColumnPinning,
    onGlobalFilterChange: setGlobalFilter,
    onColumnSizingChange: setColumnSizing,
    enableRowSelection,
    enableColumnPinning,
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    globalFilterFn: 'includesString',
    initialState: {
      pagination: {
        pageSize,
      },
    },
    state: {
      sorting,
      columnFilters,
      columnVisibility,
      rowSelection,
      columnPinning,
      globalFilter,
      columnSizing,
    },
    meta: {
      onRowEdit,
    },
  });

  return (
    <div className={cn('space-y-4', className)}>
      {/* Toolbar: Global Search + Column Visibility */}
      <div className="flex items-center justify-between gap-4">
        {/* Global Search */}
        {enableGlobalFilter && (
          <div className="flex items-center gap-2 flex-1 max-w-md">
            <Search className="h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search all columns..."
              value={globalFilter ?? ''}
              onChange={(e) => setGlobalFilter(e.target.value)}
            />
          </div>
        )}

        {/* Export & Column Visibility Controls */}
        <div className="flex items-center gap-2">
          {/* Export Dropdown */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-[150px]">
              <div className="space-y-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start"
                  onClick={exportToCSV}
                >
                  Export as CSV
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start"
                  onClick={exportToJSON}
                >
                  Export as JSON
                </Button>
              </div>
            </PopoverContent>
          </Popover>

          {/* Reset Column Width */}
          <Button
            variant="outline"
            size="sm"
            onClick={resetColumnSizing}
            title="Reset column widths to default"
          >
            <Maximize2 className="h-4 w-4 mr-2" />
            Reset Width
          </Button>

          {/* Column Visibility Control */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm">
                <Settings2 className="h-4 w-4 mr-2" />
                Columns
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-[200px]">
              <div className="space-y-2">
                <h4 className="font-medium text-sm mb-2">Toggle columns</h4>
                <div className="max-h-[300px] overflow-y-auto space-y-2">
                  {table.getAllLeafColumns()
                    .filter(column => column.getCanHide())
                    .map(column => (
                      <div key={column.id} className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          className="rounded border-gray-300"
                          checked={column.getIsVisible()}
                          onChange={column.getToggleVisibilityHandler()}
                        />
                        <label className="text-sm cursor-pointer flex-1" onClick={() => column.toggleVisibility()}>
                          {typeof column.columnDef.header === 'string'
                            ? column.columnDef.header
                            : column.id}
                        </label>
                      </div>
                    ))}
                </div>
              </div>
            </PopoverContent>
          </Popover>
        </div>
      </div>

      {/* Table */}
      <div className="rounded-md border overflow-auto relative">
        <table className="w-full caption-bottom text-sm border-collapse" style={{ tableLayout: 'fixed' }}>
          <thead className="[&_tr]:border-b">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b transition-colors hover:bg-muted/50">
                {headerGroup.headers.map((header) => {
                  const isPinned = header.column.getIsPinned();
                  return (
                    <th
                      key={header.id}
                      className={cn(
                        "h-10 px-2 text-left align-middle font-medium text-muted-foreground relative",
                        isPinned && "sticky z-10 bg-background shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]",
                        (header.column.columnDef.meta as any)?.stickyLeft && "sticky z-20 bg-background shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]",
                        isPinned === 'right' && "right-0",
                        (header.column.columnDef.meta as any)?.className
                      )}
                      style={{
                        ...(isPinned === 'left' && { left: `${header.getStart()}px` }),
                        ...(isPinned === 'right' && { right: 0 }),
                        ...((header.column.columnDef.meta as any)?.stickyLeft && { left: `${header.getStart()}px` }),
                        width: header.getSize(),
                        minWidth: header.column.columnDef.minSize,
                        maxWidth: header.column.columnDef.maxSize,
                      }}
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
                      {header.column.getCanResize() && (
                        <div
                          onMouseDown={header.getResizeHandler()}
                          onTouchStart={header.getResizeHandler()}
                          onDoubleClick={() => header.column.resetSize()}
                          className={cn(
                            "absolute right-0 top-0 h-full w-1.5 cursor-col-resize select-none touch-none group",
                            "hover:bg-primary/50 active:bg-primary",
                            header.column.getIsResizing() && "bg-primary",
                            // Add a wider invisible hit area for easier grabbing
                            "before:absolute before:right-[-4px] before:top-0 before:h-full before:w-[9px] before:content-['']"
                          )}
                          title="Drag to resize, double-click to auto-fit"
                        >
                          {/* Visual indicator for resizing */}
                          <div className={cn(
                            "absolute right-0 top-0 h-full w-0.5 transition-colors",
                            "group-hover:bg-primary/70",
                            header.column.getIsResizing() && "bg-primary"
                          )} />
                        </div>
                      )}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody className="[&_tr:last-child]:border-0">
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  data-state={row.getIsSelected() && 'selected'}
                  className={cn(
                    'border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted',
                    onRowClick && 'cursor-pointer'
                  )}
                  onClick={onRowClick ? () => onRowClick(row.original) : undefined}
                >
                  {row.getVisibleCells().map((cell) => {
                    const isPinned = cell.column.getIsPinned();
                    return (
                      <td
                        key={cell.id}
                        className={cn(
                          "p-2 align-middle",
                          isPinned && "sticky z-10 bg-background shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]",
                          (cell.column.columnDef.meta as any)?.stickyLeft && "sticky z-20 bg-background shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]",
                          isPinned === 'right' && "right-0",
                          (cell.column.columnDef.meta as any)?.className
                        )}
                        style={{
                          ...(isPinned === 'left' && { left: `${cell.column.getStart()}px` }),
                          ...(isPinned === 'right' && { right: 0 }),
                          ...((cell.column.columnDef.meta as any)?.stickyLeft && { left: `${cell.column.getStart()}px` }),
                          width: cell.column.getSize(),
                          minWidth: cell.column.columnDef.minSize,
                          maxWidth: cell.column.columnDef.maxSize,
                        }}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    );
                  })}
                </tr>
              ))
            ) : (
              emptyStateComponent ? (
                emptyStateComponent
              ) : (
                <tr>
                  <td colSpan={columns.length} className="h-24 text-center">
                    No results.
                  </td>
                </tr>
              )
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {enablePagination && (
        <div className="flex items-center justify-between px-2">
          <div className="flex-1 text-sm text-muted-foreground">
            {enableRowSelection && (
              <>
                {table.getFilteredSelectedRowModel().rows.length} of{' '}
                {table.getFilteredRowModel().rows.length} row(s) selected.
              </>
            )}
          </div>
          <div className="flex items-center space-x-6 lg:space-x-8">
            <div className="flex items-center space-x-2">
              <p className="text-sm font-medium">Rows per page</p>
              <select
                className="h-8 w-[70px] rounded border border-input bg-background px-3 py-1 text-sm"
                value={table.getState().pagination.pageSize}
                onChange={(e) => {
                  table.setPageSize(Number(e.target.value));
                }}
              >
                {[10, 20, 30, 40, 50, 100].map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex w-[100px] items-center justify-center text-sm font-medium">
              Page {table.getState().pagination.pageIndex + 1} of{' '}
              {table.getPageCount()}
            </div>
            <div className="flex items-center space-x-2">
              <button
                className="h-8 w-8 p-0 border rounded bg-background hover:bg-muted disabled:opacity-50"
                onClick={() => table.setPageIndex(0)}
                disabled={!table.getCanPreviousPage()}
              >
                «
              </button>
              <button
                className="h-8 w-8 p-0 border rounded bg-background hover:bg-muted disabled:opacity-50"
                onClick={() => table.previousPage()}
                disabled={!table.getCanPreviousPage()}
              >
                ‹
              </button>
              <button
                className="h-8 w-8 p-0 border rounded bg-background hover:bg-muted disabled:opacity-50"
                onClick={() => table.nextPage()}
                disabled={!table.getCanNextPage()}
              >
                ›
              </button>
              <button
                className="h-8 w-8 p-0 border rounded bg-background hover:bg-muted disabled:opacity-50"
                onClick={() => table.setPageIndex(table.getPageCount() - 1)}
                disabled={!table.getCanNextPage()}
              >
                »
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
