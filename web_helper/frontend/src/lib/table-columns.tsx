import React from "react";
/**
 * TanStack Table Column Definitions for Projects View
 * Generates dynamic columns for hyperparameters and metrics
 */

import { type ColumnDef } from '@tanstack/react-table';

interface ExperimentRow {
  run_name: string;
  hyperparameters?: Record<string, any>;
  final_metrics?: Record<string, any>;
  [key: string]: any;
}

/**
 * Helper to flatten nested objects
 */
export const flattenObject = (obj: any, prefix = ''): Record<string, any> => {
  if (!obj || typeof obj !== 'object') return {};
  const result: Record<string, any> = {};
  for (const [key, value] of Object.entries(obj)) {
    const newKey = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      Object.assign(result, flattenObject(value, newKey));
    } else {
      result[newKey] = value;
    }
  }
  return result;
};

/**
 * Extract columns from nested/flattened objects
 * In flatten mode, includes all keys (objects will be stringified)
 * In nested mode, filters out object values
 */
export const extractColumns = (obj: any, flatten: boolean): Set<string> => {
  const columns = new Set<string>();

  if (flatten) {
    // Flatten mode: extract all nested keys as dot-separated strings
    // Include all keys, even if value is still an object (will be stringified in display)
    const flattened = flattenObject(obj);
    Object.keys(flattened).forEach(key => {
      columns.add(key);
    });
  } else {
    // Nested mode: extract top-level keys (including objects - they'll be stringified)
    if (obj && typeof obj === 'object') {
      Object.keys(obj).forEach(key => {
        columns.add(key);
      });
    }
  }

  return columns;
};

/**
 * Get value from object based on flatten mode
 */
export const getValue = (obj: any, key: string, flatten: boolean): any => {
  if (flatten) {
    // In flatten mode, key might be dot-separated (e.g., "optimizer.lr")
    const flattened = flattenObject(obj);
    return flattened[key];
  } else {
    // In nested mode, key is top-level only
    return obj?.[key];
  }
};

/**
 * Generate column definitions for Projects View with column groups
 */
export const generateProjectColumns = (
  hyperparamColumns: string[],
  metricColumns: string[],
  flattenParams: boolean,
  data: ExperimentRow[] = [],
  _onRowClick?: (row: ExperimentRow) => void
): ColumnDef<ExperimentRow>[] => {
  const columns: ColumnDef<ExperimentRow>[] = [];

  // Checkbox column (for selection)
  columns.push({
    id: 'select',
    header: ({ table }) => (
      <input
        type="checkbox"
        className="rounded border-gray-300"
        checked={table.getIsAllRowsSelected()}
        onChange={table.getToggleAllRowsSelectedHandler()}
      />
    ),
    cell: ({ row }) => (
      <input
        type="checkbox"
        className="rounded border-gray-300"
        checked={row.getIsSelected()}
        onChange={row.getToggleSelectedHandler()}
        onClick={(e) => e.stopPropagation()}
      />
    ),
    enableSorting: false,
    enableColumnFilter: false,
    enablePinning: true,
    enableResizing: false,
    size: 40,
    minSize: 40,
    maxSize: 40,
    meta: {
      stickyLeft: true,
    },
  });

  // Run ID column (always visible, pinnable)
  const runNameSize = calculateColumnSize(
    data,
    (row) => row.run_name,
    'Run ID',
    120,
    200
  );

  columns.push({
    accessorKey: 'run_name',
    id: 'run_name',
    header: 'Run ID',
    cell: ({ getValue }) => {
      const value = getValue<string>();
      return (
        <div
          className="font-mono text-sm overflow-hidden"
          title={value}
          style={{
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            wordBreak: 'break-word',
            overflowWrap: 'break-word',
          }}
        >
          {value}
        </div>
      );
    },
    enableSorting: true,
    enableColumnFilter: true,
    enablePinning: true,
    size: runNameSize,
    minSize: 80,
    maxSize: 600,
    meta: {
      stickyLeft: true,
    },
  });

  // Hyperparameter columns
  hyperparamColumns.forEach((param) => {
    const accessor = (row: ExperimentRow) => getValue(row.hyperparameters || {}, param, flattenParams);
    const columnSize = calculateColumnSize(data, accessor, param, 80, 250);

    columns.push({
      id: `hp_${param}`,
      accessorFn: accessor,
      header: param, // Original key name without transformation
      cell: ({ getValue }) => {
        const value = getValue();
        const isMissing = value === undefined || value === null || value === '' ||
          (typeof value === 'number' && isNaN(value));

        if (isMissing) {
          return <span className="text-muted-foreground italic">N/A</span>;
        }

        // Handle objects and arrays
        if (typeof value === 'object' && value !== null) {
          const jsonStr = JSON.stringify(value);
          return (
            <div
              className="text-xs text-muted-foreground italic font-mono overflow-hidden"
              title={jsonStr}
              style={{
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                wordBreak: 'break-word',
                overflowWrap: 'break-word',
              }}
            >
              {jsonStr}
            </div>
          );
        }

        return (
          <div
            className="text-sm overflow-hidden"
            title={String(value)}
            style={{
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              wordBreak: 'break-word',
              overflowWrap: 'break-word',
            }}
          >
            {String(value)}
          </div>
        );
      },
      enableSorting: true,
      enableColumnFilter: true,
      enablePinning: true,
      size: columnSize,
      minSize: 60,
      maxSize: 600,
      meta: {
        className: 'bg-blue-50/50 dark:bg-blue-950/10 border-l border-blue-200/50 dark:border-blue-800/50',
        categoryColor: 'blue',
        category: 'Hyperparameters',
      },
    });
  });

  // Metric columns
  metricColumns.forEach((metric, index) => {
    const accessor = (row: ExperimentRow) => getValue(row.final_metrics || {}, metric, flattenParams);
    const columnSize = calculateColumnSize(data, accessor, metric, 100, 180);

    columns.push({
      id: `metric_${metric}`,
      accessorFn: accessor,
      header: metric, // Original key name without transformation
      cell: ({ getValue }) => {
        const value = getValue();
        const isMissing = value === undefined || value === null || value === '' ||
          (typeof value === 'number' && isNaN(value));

        if (isMissing) {
          return <span className="text-muted-foreground italic">N/A</span>;
        }

        const displayValue = typeof value === 'number' ? Number(value).toFixed(4) : String(value);
        const className = typeof value === 'number' ? 'text-sm font-mono overflow-hidden' : 'text-sm overflow-hidden';

        return (
          <div
            className={className}
            title={String(value)}
            style={{
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              wordBreak: 'break-word',
              overflowWrap: 'break-word',
            }}
          >
            {displayValue}
          </div>
        );
      },
      enableSorting: true,
      enableColumnFilter: true,
      enablePinning: true,
      sortingFn: 'basic',
      size: columnSize,
      minSize: 80,
      maxSize: 600,
      meta: {
        className: `bg-green-50/50 dark:bg-green-950/10 ${index === 0 ? 'border-l border-green-200/50 dark:border-green-800/50' : ''}`,
        categoryColor: 'green',
        category: 'Metrics',
      },
    });
  });

  return columns;
};

/**
 * Convert pinning state to TanStack ColumnPinningState
 */
export const convertPinningState = (
  pinnedLeftHyperparams: Set<string>,
  pinnedRightMetrics: Set<string>,
  isPinningEnabled: boolean = true
): { left?: string[]; right?: string[] } => {
  const pinningState: { left?: string[]; right?: string[] } = {};

  // Only pin select and run_name if global pinning is enabled
  const leftPins: string[] = [];
  if (isPinningEnabled) {
    leftPins.push('select', 'run_name');
  }

  // Add pinned hyperparameters (only if pinning is enabled)
  if (isPinningEnabled) {
    pinnedLeftHyperparams.forEach(param => {
      leftPins.push(`hp_${param}`);
    });
  }

  if (leftPins.length > 0) {
    pinningState.left = leftPins;
  }

  // Add pinned metrics to right (only if pinning is enabled)
  const rightPins: string[] = [];
  if (isPinningEnabled) {
    pinnedRightMetrics.forEach(metric => {
      rightPins.push(`metric_${metric}`);
    });
  }

  if (rightPins.length > 0) {
    pinningState.right = rightPins;
  }

  return pinningState;
};

/**
 * Calculate optimal column size based on content
 * Improved algorithm with better font metrics and type-aware sizing
 */
export const calculateColumnSize = (
  data: any[],
  accessor: (row: any) => any,
  headerText: string,
  minSize = 80,
  maxSize = 400
): number => {
  if (data.length === 0) return minSize;

  // Sample first 100 rows for better accuracy (was 50)
  const sampleSize = Math.min(100, data.length);
  const sample = data.slice(0, sampleSize);

  // Calculate content widths with improved metrics
  const contentWidths = sample.map(row => {
    const value = accessor(row);
    if (value === undefined || value === null) return 0;

    // Handle objects/arrays
    if (typeof value === 'object') {
      const jsonStr = JSON.stringify(value);
      // Use monospace metrics for JSON (9px per char for font-mono)
      return jsonStr.length * 9;
    }

    const strValue = String(value);

    // Type-aware character width estimation
    if (typeof value === 'number') {
      // Numbers: use monospace metrics (9px per char)
      return strValue.length * 9;
    }

    // Strings: average character width in default font
    // Use more accurate 7.5px for mixed characters
    return strValue.length * 7.5;
  });

  // Header width calculation
  // Account for sorting icons (24px) and padding (16px each side)
  const headerWidth = headerText.length * 8 + 24 + 32;

  // Get max width from content and header
  const maxContentWidth = Math.max(...contentWidths, 0);
  const estimatedWidth = Math.max(maxContentWidth + 32, headerWidth);

  // Apply min/max constraints
  const finalWidth = Math.max(minSize, Math.min(maxSize, estimatedWidth));

  // Round to nearest 10px for cleaner layout
  return Math.round(finalWidth / 10) * 10;
};

/**
 * Calculate auto-fit width for a column (for double-click auto-fit)
 */
export const calculateAutoFitWidth = (
  _columnId: string,
  data: any[],
  accessor: (row: any) => any,
  headerText: string
): number => {
  // Use all data for auto-fit (not sampled)
  const contentWidths = data.map(row => {
    const value = accessor(row);
    if (value === undefined || value === null) return 0;

    if (typeof value === 'object') {
      return JSON.stringify(value).length * 9;
    }

    const strValue = String(value);
    if (typeof value === 'number') {
      return strValue.length * 9;
    }

    return strValue.length * 7.5;
  });

  const headerWidth = headerText.length * 8 + 24 + 32;
  const maxContentWidth = Math.max(...contentWidths, 0);

  return Math.max(maxContentWidth + 32, headerWidth, 80);
};

/**
 * Convert visibility Sets to TanStack VisibilityState
 */
export const convertVisibilityState = (
  hyperparamColumns: string[],
  metricColumns: string[],
  visibleHyperparams: Set<string>,
  visibleMetrics: Set<string>
): Record<string, boolean> => {
  const visibilityState: Record<string, boolean> = {};

  // If Sets are empty, show all columns by default
  const showAllHp = visibleHyperparams.size === 0;
  const showAllMetrics = visibleMetrics.size === 0;

  // Hyperparameter columns
  hyperparamColumns.forEach(param => {
    const _columnId = `hp_${param}`;
    visibilityState[_columnId] = showAllHp || visibleHyperparams.has(param);
  });

  // Metric columns
  metricColumns.forEach(metric => {
    const _columnId = `metric_${metric}`;
    visibilityState[_columnId] = showAllMetrics || visibleMetrics.has(metric);
  });

  return visibilityState;
};
