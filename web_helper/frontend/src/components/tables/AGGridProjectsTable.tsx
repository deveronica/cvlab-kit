import React from "react";

/**
 * AG Grid Projects Table Component
 * Comparison implementation using AG Grid Community
 *
 * Features:
 * - Column resizing, reordering, pinning
 * - Row selection with checkboxes
 * - Sorting (multi-column)
 * - Filtering (text, number)
 * - Quick search (global filter)
 * - CSV export
 * - Pagination
 * - Auto-size columns
 * - Column visibility toggle
 * - State persistence
 */

import { useMemo, useCallback, useRef, useState, useEffect } from 'react';
import { AgGridReact } from 'ag-grid-react';
import type { ColDef, GridReadyEvent, SelectionChangedEvent, GridApi, IHeaderParams } from 'ag-grid-community';
import { ModuleRegistry, AllCommunityModule, themeQuartz } from 'ag-grid-community';
import { getValue } from '../../lib/table-columns';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { AlertCircle, BarChart3, ChevronDown, TrendingUp, TrendingDown, Clock, Calculator } from 'lucide-react';
import { Badge } from '../ui/badge';
import { Checkbox } from '../ui/checkbox';
import { ColumnStatisticsDialog } from '../ui/column-statistics-dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from '../ui/dropdown-menu';

// Register AG Grid modules
ModuleRegistry.registerModules([AllCommunityModule]);

interface ExperimentRow {
  run_name: string;
  hyperparameters?: Record<string, any>;
  final_metrics?: Record<string, any>;
  [key: string]: any;
}

interface AGGridProjectsTableProps {
  data: ExperimentRow[];
  hyperparamColumns: string[];
  metricColumns: string[];
  flattenParams: boolean;
  _activeProject?: string | null; // Active project name for statistics
  diffMode?: boolean; // When true, only show columns with variance
  visibleHyperparams?: Set<string>; // Visible hyperparameters
  visibleMetrics?: Set<string>; // Visible metrics
  pinnedLeftHyperparams?: Set<string>; // Hyperparameters to pin left
  pinnedRightMetrics?: Set<string>; // Metrics to pin right
  isPinningEnabled?: boolean; // Global pinning toggle state
  onRowSelectionChange?: (selectedRows: ExperimentRow[]) => void;
  onRowClick?: (row: ExperimentRow) => void;
  cellOutliers?: Set<string>; // Cell-level outliers in format "run_name|column"
  selectedRows?: string[]; // Array of run_names that should be selected
  highlightedRun?: string | null; // Single run to highlight (e.g., from outlier/recommendation click)
  quickFilterText?: string; // External filter text
  _onQuickFilterChange?: (text: string) => void; // Filter change handler
  rowMode?: '1-line' | '2-line' | '3-line'; // Row display _mode (1-line, 2-line, or 3-line)
  _onToggleMaximize?: () => void; // Toggle row _mode handler
  _onResetFilters?: () => void; // Reset filters handler (called from parent)
  _onExportCSV?: () => void; // Export CSV handler (called from parent)
  gridRef?: React.MutableRefObject<any>; // Grid ref to expose API
  onColumnStateChange?: (pinnedLeft: Set<string>, pinnedRight: Set<string>) => void; // Column pinning state feedback
  onColumnOrderChange?: (hyperparamOrder: string[], metricOrder: string[]) => void; // Column order change feedback
}

/**
 * Calculate text width using Canvas for optimal column sizing
 */
const calculateTextWidth = (text: string, font: string = '14px system-ui'): number => {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return text.length * 8; // Fallback
  context.font = font;
  return context.measureText(text).width;
};

export function AGGridProjectsTable({
  data,
  hyperparamColumns,
  metricColumns,
  flattenParams,
  _activeProject,
  diffMode = false,
  visibleHyperparams,
  visibleMetrics,
  pinnedLeftHyperparams = new Set(),
  pinnedRightMetrics = new Set(),
  isPinningEnabled = true,
  onRowSelectionChange,
  onRowClick,
  cellOutliers = new Set(),
  selectedRows = [],
  highlightedRun = null,
  quickFilterText = '',
  _onQuickFilterChange,
  rowMode = '2-line',
  _onToggleMaximize,
  _onResetFilters,
  _onExportCSV,
  gridRef: externalGridRef,
  onColumnStateChange,
  onColumnOrderChange,
}: AGGridProjectsTableProps) {
  const internalGridRef = useRef<AgGridReact>(null);
  const gridRef = externalGridRef || internalGridRef;

  // Helper variables
  const lineClampClass = rowMode === '3-line' ? 'line-clamp-3' : rowMode === '2-line' ? 'line-clamp-2' : 'line-clamp-1';

  // Dynamic column widths for 2-Line mode
  const [dynamicWidths, setDynamicWidths] = useState<Record<string, { min: number; default: number; max: number }>>({});

  // Column statistics dialog state
  const [columnStatsDialog, setColumnStatsDialog] = useState<{
    isOpen: boolean;
    metricName: string;
    values: Array<{ runName: string; value: number | null }>;
  } | null>(null);

  // Metric display _mode per column
  type MetricDisplayMode = 'best-higher' | 'best-lower' | 'latest' | 'mean' | 'median';
  const [metricDisplayModes, setMetricDisplayModes] = useState<Record<string, MetricDisplayMode>>(() => {
    try {
      const saved = localStorage.getItem('metricDisplayModes');
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  // Use ref to avoid stale closures in valueGetter
  // This prevents columnDefs from recreating when display modes change
  const metricDisplayModesRef = useRef(metricDisplayModes);

  // Save metric display modes to localStorage and update ref
  useEffect(() => {
    metricDisplayModesRef.current = metricDisplayModes;
    localStorage.setItem('metricDisplayModes', JSON.stringify(metricDisplayModes));
  }, [metricDisplayModes]);

  // Refresh metric cells and headers when display _mode changes (without recreating columns)
  useEffect(() => {
    if (!gridRef.current?.api) return;

    // Force re-evaluation of all cells without recreating columns
    // This will trigger valueGetter to run again with new display modes
    gridRef.current.api.redrawRows();

    // Also refresh headers to update the check mark in dropdown
    gridRef.current.api.refreshHeader();
  }, [metricDisplayModes]); // Only depend on metricDisplayModes, not metricColumns

  // Get display _mode for a metric (with auto-detection fallback)
  const getMetricDisplayMode = useCallback((metricName: string): MetricDisplayMode => {
    if (metricDisplayModes[metricName]) {
      return metricDisplayModes[metricName];
    }
    // Auto-detect based on metric name
    const direction = getMetricDirection(metricName);
    if (direction === 'higher') return 'best-higher';
    if (direction === 'lower') return 'best-lower';
    return 'latest';
  }, [metricDisplayModes]);

  // Set display _mode for a metric
  const setMetricDisplayMode = useCallback((metricName: string, _mode: MetricDisplayMode) => {
    setMetricDisplayModes(prev => ({ ...prev, [metricName]: _mode }));
  }, []);

  // Track previous column order to avoid redundant callbacks
  const prevOrderRef = useRef<{ hyperparams: string[]; metrics: string[] }>({ hyperparams: [], metrics: [] });

  // Manual row selection state (replaces AG Grid's built-in row selection)
  const [selectedRunNames, setSelectedRunNames] = useState<Set<string>>(new Set(selectedRows || []));

  // Sync external selectedRows prop to internal state (only if actually different)
  useEffect(() => {
    const newSet = new Set(selectedRows || []);
    const currentArray = Array.from(selectedRunNames).sort();
    const newArray = Array.from(newSet).sort();

    // Only update if the sets are actually different
    if (currentArray.length !== newArray.length ||
        currentArray.some((val, idx) => val !== newArray[idx])) {
      setSelectedRunNames(newSet);
    }
  }, [selectedRows]); // Remove selectedRunNames from deps to prevent loop

  // Notify parent when selection changes (debounced to prevent rapid updates)
  useEffect(() => {
    if (!onRowSelectionChange) return;
    const selected = data.filter(row => selectedRunNames.has(row.run_name));

    // Use setTimeout to debounce and prevent immediate re-render loops
    const timeoutId = setTimeout(() => {
      onRowSelectionChange(selected);
    }, 0);

    return () => clearTimeout(timeoutId);
  }, [selectedRunNames, data, onRowSelectionChange]);

  // Toggle single row selection
  const toggleRowSelection = useCallback((runName: string) => {
    setSelectedRunNames(prev => {
      const newSet = new Set(prev);
      if (newSet.has(runName)) {
        newSet.delete(runName);
      } else {
        newSet.add(runName);
      }
      return newSet;
    });
  }, []);

  // Toggle all rows selection (header checkbox)
  const toggleAllRowsSelection = useCallback(() => {
    const allDataRows = data.filter(row => !row.isStatisticsRow);
    const allRunNames = allDataRows.map(row => row.run_name);

    // If all selected, deselect all. Otherwise, select all.
    const allSelected = allRunNames.every(name => selectedRunNames.has(name));

    if (allSelected) {
      setSelectedRunNames(new Set());
    } else {
      setSelectedRunNames(new Set(allRunNames));
    }
  }, [data, selectedRunNames]);

  // Read CSS variables from :root (shared with index.css)
  const getCSSVariable = (varName: string): string => {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  };

  const fontSize = parseFloat(getCSSVariable('--ag-cell-font-size')) || 14;
  const lineHeight = parseFloat(getCSSVariable('--ag-cell-line-height')) || 1.5;
  const paddingVertical = (parseFloat(getCSSVariable('--ag-cell-padding-vertical')) || 8) * 2;

  // Dynamic row height calculation based on line-height and padding
  // Reusable for 2-line, 3-line modes
  const calculateRowHeight = (numLines: number): number => {
    return Math.ceil(fontSize * lineHeight * numLines + paddingVertical);
  };

  const getRowHeight = (_mode: '1-line' | '2-line' | '3-line'): number => {
    if (_mode === '3-line') return calculateRowHeight(3);
    if (_mode === '2-line') return calculateRowHeight(2);
    return calculateRowHeight(1); // 1-line
  };

  const getHeaderHeight = (_mode: '1-line' | '2-line' | '3-line'): number => {
    // Header height = font-size * line-height + padding-vertical + border
    // Font size: 14px, Line height: 1.5, Very compact padding: 2px top + 2px bottom
    const fontSize = 14;
    const lineHeight = 1.5;
    const paddingVertical = 4; // 2px top + 2px bottom (very compact)
    const borderAndMargin = 2; // Border

    // Always 1 line for header text, but height adjusts for cell padding/margins
    const textHeight = fontSize * lineHeight;
    const totalHeight = Math.ceil(textHeight + paddingVertical + borderAndMargin);

    // Return consistent height regardless of _mode (header is always 1 line)
    return totalHeight; // ~27px
  };

  // Calculate column width based on header name and data content
  const _calculateColumnWidth = (headerName: string, dataValues: any[], targetLines: number = 2): { min: number; default: number; max: number } => {
    const CHAR_WIDTH_AVG = 8; // Average character width in pixels
    const PADDING_HORIZONTAL = 24; // Left + right padding
    const cellHeight = getRowHeight(rowMode);

    // 최소 너비: 컬럼명이 모두 보이는 너비
    const minWidth = Math.max(
      headerName.length * CHAR_WIDTH_AVG + PADDING_HORIZONTAL,
      80 // Absolute minimum
    );

    // 최대 너비: 셀 높이 × 12
    const maxWidth = cellHeight * 12;

    // Find longest data value
    const maxDataLength = dataValues.reduce((max, val) => {
      if (val === null || val === undefined) return max;
      const str = String(val);
      return Math.max(max, str.length);
    }, 0);

    // 기본 너비: 컬럼명 + 텍스트가 n줄을 유지하는 최소 너비 (단, 최대 너비 이하)
    // targetLines에 맞춰 데이터가 표시되도록 계산
    const widthForData = Math.ceil(maxDataLength / targetLines) * CHAR_WIDTH_AVG + PADDING_HORIZONTAL;

    const defaultWidth = Math.min(
      Math.max(minWidth, widthForData),
      maxWidth // 최대 너비를 넘지 않음
    );

    return {
      min: Math.ceil(minWidth),
      default: Math.ceil(defaultWidth),
      max: Math.ceil(maxWidth)
    };
  };

  // Helper function to determine metric optimization direction
  const getMetricDirection = (metricName: string): 'higher' | 'lower' | 'latest' => {
    const lower = metricName.toLowerCase();
    // Lower is better: loss, error, cost, distance, etc.
    if (lower.includes('loss') || lower.includes('error') || lower.includes('cost') ||
        lower.includes('distance') || lower.includes('mse') || lower.includes('mae') ||
        lower.includes('rmse')) {
      return 'lower';
    }
    // Higher is better: accuracy, precision, recall, f1, score, etc.
    if (lower.includes('acc') || lower.includes('precision') || lower.includes('recall') ||
        lower.includes('f1') || lower.includes('score') || lower.includes('iou') ||
        lower.includes('dice') || lower.includes('auc')) {
      return 'higher';
    }
    // Default: latest value
    return 'latest';
  };

  // Calculate the display value for a metric based on the selected mode
  const getMetricDisplayValue = useCallback((value: any, metricName: string): number | null => {
    const _mode = getMetricDisplayMode(metricName);

    // Handle null/undefined
    if (value === null || value === undefined) {
      return null;
    }

    // If value is a simple number, return it directly
    if (typeof value === 'number') {
      return value;
    }

    // If value is an array, calculate statistics
    if (Array.isArray(value)) {
      const nums = value.filter((v: any) => typeof v === 'number' && !isNaN(v));
      if (nums.length === 0) return null;

      switch (_mode) {
        case 'best-higher':
          return Math.max(...nums);
        case 'best-lower':
          return Math.min(...nums);
        case 'latest':
          return nums[nums.length - 1];
        case 'mean': {
          const sum = nums.reduce((acc: number, val: number) => acc + val, 0);
          return sum / nums.length;
        }
        case 'median': {
          const sorted = [...nums].sort((a, b) => a - b);
          return sorted.length % 2 === 0
            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
            : sorted[Math.floor(sorted.length / 2)];
        }
        default:
          return nums[nums.length - 1];
      }
    }

    // If value is an object (but not array), try to extract a numeric value
    if (typeof value === 'object') {
      // Try common patterns: { value: X }, { latest: X }, { final: X }
      if ('value' in value && typeof value.value === 'number') {
        return value.value;
      }
      if ('latest' in value && typeof value.latest === 'number') {
        return value.latest;
      }
      if ('final' in value && typeof value.final === 'number') {
        return value.final;
      }
      // If object has history array, recurse with the array
      if ('history' in value && Array.isArray(value.history)) {
        return getMetricDisplayValue(value.history, metricName);
      }
    }

    // If value is a string that looks like a number, parse it
    if (typeof value === 'string') {
      const parsed = parseFloat(value);
      if (!isNaN(parsed)) {
        return parsed;
      }
    }

    return null;
  }, [getMetricDisplayMode]);

  // Helper function to check if a column has variance (different values across rows)
  const hasVariance = useCallback((field: string, isHyperparam: boolean) => {
    if (data.length <= 1) return true; // Single row always shows

    const values = data.map(row => {
      const source = isHyperparam ? row.hyperparameters : row.final_metrics;
      return getValue(source || {}, field, flattenParams);
    }).filter(v => v !== undefined && v !== null);

    if (values.length === 0) return false;

    // Check if all values are identical
    const firstValue = values[0];
    if (typeof firstValue === 'number') {
      return values.some(v => Math.abs((v as number) - (firstValue as number)) > 1e-10);
    }
    return values.some(v => JSON.stringify(v) !== JSON.stringify(firstValue));
  }, [data, flattenParams]);

  // Calculate optimal column widths for all modes (1-line, 2-line, 3-line)
  // Memoize data length and first run name to avoid recalculating on every refetch
  const dataKey = useMemo(() =>
    `${data.length}-${data[0]?.run_name || ''}`,
    [data.length, data[0]?.run_name]
  );

  useEffect(() => {
    if (data.length === 0) {
      setDynamicWidths({});
      return;
    }

    const widths: Record<string, { min: number; default: number; max: number }> = {};
    const padding = 26; // Cell padding (horizontal padding)
    const safetyMargin = 1.05; // 5% safety margin for accurate line wrapping
    const cellHeight = getRowHeight(rowMode);
    const maxColumnWidth = cellHeight * 12; // 최대 너비 = 셀 높이 × 12

    // Target lines based on rowMode
    const targetLines = rowMode === '3-line' ? 3 : rowMode === '2-line' ? 2 : 1;

    // Calculate Run ID width (header: "Run ID")
    const runIdHeaderWidth = calculateTextWidth('Run ID', '14px system-ui');
    const maxRunNameWidth = data.reduce((max, row) => {
      const width = calculateTextWidth(row.run_name, '14px monospace');
      return Math.max(max, width);
    }, 0);
    // Calculate width to fit content in targetLines
    const runIdContentWidth = Math.ceil((maxRunNameWidth / targetLines) * safetyMargin + padding);
    // Min width: only ensure header is visible (with 2% margin)
    const runIdMinWidth = Math.ceil(runIdHeaderWidth * 1.02) + padding;
    const runIdDefaultWidth = Math.max(runIdMinWidth, Math.min(runIdContentWidth, maxColumnWidth));
    widths['run_name'] = {
      min: runIdMinWidth,
      default: runIdDefaultWidth,
      max: Math.ceil(runIdDefaultWidth * 1.2)
    };

    // Calculate Status width (header: "Status")
    const statusHeaderWidth = calculateTextWidth('Status', '14px system-ui');
    const maxStatusWidth = data.reduce((max, row) => {
      const status = row.status || 'unknown';
      const width = calculateTextWidth(status, '14px system-ui');
      return Math.max(max, width);
    }, 0);
    // Calculate width to fit content in targetLines
    const statusContentWidth = Math.ceil((maxStatusWidth / targetLines) * safetyMargin + padding) + 30;
    // Min width: only ensure header is visible (with 2% margin)
    const statusMinWidth = Math.ceil(statusHeaderWidth * 1.02) + padding;
    const statusDefaultWidth = Math.max(statusMinWidth, statusContentWidth);
    widths['status'] = {
      min: statusMinWidth,
      default: statusDefaultWidth,
      max: Math.ceil(statusDefaultWidth * 1.2)
    };

    // Calculate Tags width (header: "Tags")
    const tagsHeaderWidth = calculateTextWidth('Tags', '14px system-ui');
    const maxTagsWidth = data.reduce((max, row) => {
      if (!row.tags || row.tags.length === 0) return max;
      const text = row.tags.join(', ');
      const width = calculateTextWidth(text, '12px system-ui');
      return Math.max(max, width);
    }, 0);
    // Calculate width to fit content in targetLines
    const tagsContentWidth = Math.ceil((maxTagsWidth / targetLines) * safetyMargin + padding);
    // Min width: only ensure header is visible (with 2% margin)
    const tagsMinWidth = Math.ceil(tagsHeaderWidth * 1.02) + padding;
    const tagsDefaultWidth = Math.max(tagsMinWidth, tagsContentWidth);
    widths['tags'] = {
      min: tagsMinWidth,
      default: tagsDefaultWidth,
      max: Math.ceil(tagsDefaultWidth * 1.2)
    };

    // Calculate Hyperparam widths
    hyperparamColumns.forEach(col => {
      if (['Run ID', 'Status', 'Tags'].includes(col)) return;

      const headerWidth = calculateTextWidth(col, '14px system-ui');
      const maxContentWidth = data.reduce((max, row) => {
        const value = getValue(row.hyperparameters || {}, col, flattenParams);
        if (value === undefined || value === null) return max;
        const text = typeof value === 'object' ? JSON.stringify(value) : String(value);
        const width = calculateTextWidth(text, '14px system-ui');
        return Math.max(max, width);
      }, 0);
      // Calculate width to fit content in targetLines
      const contentWidth = Math.ceil((maxContentWidth / targetLines) * safetyMargin + padding);
      // Min width: only ensure header is visible (with 2% margin)
      const minWidth = Math.ceil(headerWidth * 1.02) + padding;
      const defaultWidth = Math.max(minWidth, contentWidth);

      widths[`hp_${col}`] = {
        min: minWidth,
        default: defaultWidth,
        max: Math.ceil(defaultWidth * 1.2)
      };
    });

    // Calculate Metric widths
    metricColumns.forEach(col => {
      const headerWidth = calculateTextWidth(col, '14px system-ui');
      const maxContentWidth = data.reduce((max, row) => {
        const value = getValue(row.final_metrics || {}, col, flattenParams);
        if (value === undefined || value === null) return max;
        const text = typeof value === 'number' ? value.toFixed(4) : String(value);
        const width = calculateTextWidth(text, '14px monospace');
        return Math.max(max, width);
      }, 0);
      // Calculate width to fit content in targetLines
      const contentWidth = Math.ceil((maxContentWidth / targetLines) * safetyMargin + padding);
      // Min width: only ensure header is visible (metrics have dropdown + stats button, with 2% margin)
      const minWidth = Math.ceil(headerWidth * 1.02) + padding + 60; // +60 for header icons (dropdown + stats)
      const defaultWidth = Math.max(minWidth, contentWidth);

      widths[`metric_${col}`] = {
        min: minWidth,
        default: defaultWidth,
        max: Math.ceil(defaultWidth * 1.2)
      };
    });

    setDynamicWidths(widths);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataKey, hyperparamColumns, metricColumns, flattenParams, rowMode]); // Use dataKey instead of data to avoid recalc on refetch

  // Filter columns based on diffMode and visibility
  // IMPORTANT: hyperparamColumns is already ordered (from hyperparamOrder prop)
  // We must preserve the order while filtering
  const filteredHyperparamColumns = useMemo(() => {
    // hyperparamColumns is ALREADY ordered from hyperparamOrder prop
    // Just filter, preserving the order
    return hyperparamColumns
      .filter(col => col !== 'Run ID' && col !== 'Status' && col !== 'Tags')  // Exclude always-visible columns
      .filter(param => {
        // Apply diff _mode filter
        if (diffMode && !hasVariance(param, true)) return false;
        // Apply visibility filter
        if (visibleHyperparams && visibleHyperparams.size > 0) {
          return visibleHyperparams.has(param);
        }
        return true;
      });
  }, [diffMode, hyperparamColumns, hasVariance, visibleHyperparams]);

  // IMPORTANT: metricColumns is already ordered (from metricOrder prop)
  // We must preserve the order while filtering
  const filteredMetricColumns = useMemo(() => {
    // metricColumns is ALREADY ordered from metricOrder prop
    // Just filter, preserving the order
    return metricColumns.filter(metric => {
      // Apply diff _mode filter
      if (diffMode && !hasVariance(metric, false)) return false;
      // Apply visibility filter
      if (visibleMetrics && visibleMetrics.size > 0) {
        return visibleMetrics.has(metric);
      }
      return true;
    });
  }, [diffMode, metricColumns, hasVariance, visibleMetrics]);

  // Statistics row disabled
  const pinnedBottomRowData = useMemo(() => [], []);

  // Custom header component for metric columns with statistics button
  const MetricHeaderComponent = (_props: IHeaderParams) => {
    const metricName = _props.column.getColDef().headerName || '';
    const _colId = _props.column.getColId();
    // Get the actual metric name from headerComponentParams
    const metric = (_props.column.getColDef() as any).headerComponentParams?.metricName || metricName;

    // Read from ref to get the latest display _mode (avoids stale closure)
    const currentModeFromRef = metricDisplayModesRef.current[metric];
    const currentMode: MetricDisplayMode = currentModeFromRef || (() => {
      const direction = getMetricDirection(metric);
      if (direction === 'higher') return 'best-higher';
      if (direction === 'lower') return 'best-lower';
      return 'latest';
    })();

    const handleStatisticsClick = (e: React.MouseEvent) => {
      e.stopPropagation(); // Prevent column sort

      if (data.length === 0) {
        console.error('Cannot open statistics: No data available');
        return;
      }

      // Collect all values for this metric column from the current table data
      // Use the same logic as the column's valueGetter - read from the appropriate metrics based on display mode
      const values = data.map(row => {
        // Get current display _mode from ref
        const _mode = currentMode;
        let rawValue: any;

        // Select the appropriate data field based on display _mode (same logic as valueGetter)
        switch (_mode) {
          case 'best-higher':
            rawValue = getValue(row.max_metrics || {}, metric, flattenParams);
            break;
          case 'best-lower':
            rawValue = getValue(row.min_metrics || {}, metric, flattenParams);
            break;
          case 'mean':
            rawValue = getValue(row.mean_metrics || {}, metric, flattenParams);
            break;
          case 'median':
            rawValue = getValue(row.median_metrics || {}, metric, flattenParams);
            break;
          case 'latest':
          default:
            rawValue = getValue(row.final_metrics || {}, metric, flattenParams);
            break;
        }

        const processedValue = getMetricDisplayValue(rawValue, metric);
        return {
          runName: row.run_name || 'Unknown',
          value: processedValue,
        };
      });

      setColumnStatsDialog({
        isOpen: true,
        metricName: metricName,
        values: values,
      });
    };

    type DisplayModeOption = {
      _mode: MetricDisplayMode;
      label: string;
      icon: React.ReactNode;
    };

    const displayModeOptions: DisplayModeOption[] = [
      { _mode: 'best-higher', label: 'Best (Higher is Better)', icon: <TrendingUp className="h-3.5 w-3.5" /> },
      { _mode: 'best-lower', label: 'Best (Lower is Better)', icon: <TrendingDown className="h-3.5 w-3.5" /> },
      { _mode: 'latest', label: 'Latest Value', icon: <Clock className="h-3.5 w-3.5" /> },
      { _mode: 'mean', label: 'Mean (Average)', icon: <Calculator className="h-3.5 w-3.5" /> },
      { _mode: 'median', label: 'Median', icon: <Calculator className="h-3.5 w-3.5" /> },
    ];

    const currentModeLabel = displayModeOptions.find(opt => opt._mode === currentMode)?.label || 'Latest Value';
    const _currentModeIcon = displayModeOptions.find(opt => opt._mode === currentMode)?.icon;

    return (
      <div className="ag-header-cell-label flex items-center justify-between w-full gap-1">
        <span className="ag-header-cell-text flex-1 truncate">{_props.displayName}</span>
        <div className="flex items-center gap-1 flex-shrink-0">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className="flex items-center gap-0.5 p-0.5 rounded hover:bg-accent transition-colors duration-200"
              onClick={(e) => e.stopPropagation()}
              title={`Display _mode: ${currentModeLabel}`}
              aria-label={`Change display _mode for ${metricName}`}
            >
              <ChevronDown className="h-3.5 w-3.5" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56" onClick={(e) => e.stopPropagation()}>
            <DropdownMenuLabel className="text-xs">Display Mode</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {displayModeOptions.map((option) => (
              <DropdownMenuItem
                key={option._mode}
                onClick={(e) => {
                  e.stopPropagation();
                  setMetricDisplayMode(metric, option._mode);
                }}
                className={`flex items-center gap-2 cursor-pointer ${
                  currentMode === option._mode ? 'bg-accent' : ''
                }`}
              >
                {option.icon}
                <span className="text-sm">{option.label}</span>
                {currentMode === option._mode && (
                  <span className="ml-auto text-xs text-primary">✓</span>
                )}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <button
          className="flex-shrink-0 p-0.5 rounded hover:bg-accent transition-colors duration-200"
          onClick={handleStatisticsClick}
          title={`View statistics for ${metricName}`}
          aria-label={`Statistics for ${metricName}`}
        >
          <BarChart3 className="h-3.5 w-3.5" />
        </button>
      </div>
      </div>
    );
  };

  // Generate column definitions
  // Checkbox, Run ID, Status are always pinned left (order follows hyperparamColumns)
  const columnDefs = useMemo<ColDef[]>(() => {
    const cols: ColDef[] = [];

    // Custom checkbox column (always pinned left, independent of isPinningEnabled)
    const checkboxColDef: ColDef = {
      field: 'checkbox',
      headerName: '',
      pinned: 'left', // Always pinned left, regardless of isPinningEnabled
      lockPosition: true,
      suppressMovable: true,
      width: 50,
      minWidth: 50,
      maxWidth: 50,
      resizable: false,
      sortable: false,
      filter: false,
      suppressSizeToFit: true,
      suppressAutoSize: true,
      suppressNavigable: true,
      headerComponent: (_props: IHeaderParams) => {
        const allDataRows = data.filter(row => !row.isStatisticsRow);
        const allRunNames = allDataRows.map(row => row.run_name);
        const allSelected = allRunNames.length > 0 && allRunNames.every(name => selectedRunNames.has(name));
        const someSelected = allRunNames.some(name => selectedRunNames.has(name));

        return (
          <div
            className="flex items-center justify-center w-full h-full cursor-pointer"
            onMouseDown={(e) => {
              e.stopPropagation();
              e.preventDefault();
              toggleAllRowsSelection();
            }}
          >
            <Checkbox
              checked={allSelected ? true : someSelected ? 'indeterminate' : false}
              aria-label="Select all rows"
            />
          </div>
        );
      },
      cellRenderer: (_params: any) => {
        if (!_params.data || _params.data.isStatisticsRow) return null;
        const isSelected = selectedRunNames.has(_params.data.run_name);

        return (
          <div
            className="flex items-center justify-center w-full h-full cursor-pointer"
            onMouseDown={(e) => {
              e.stopPropagation();
              e.preventDefault();
              toggleRowSelection(_params.data.run_name);
            }}
          >
            <Checkbox
              checked={isSelected}
              aria-label={`Select row ${_params.data.run_name}`}
            />
          </div>
        );
      },
    };

    cols.push(checkboxColDef);

    // Extract Run ID, Status, Tags order from hyperparamColumns
    const _alwaysVisibleColumns = ['Run ID', 'Status', 'Tags'];
    const _orderedAlwaysVisible = hyperparamColumns.filter(col => _alwaysVisibleColumns.includes(col));

    // Column definitions for Run ID and Status
    const runIdColDef: ColDef = {
      field: 'run_name',
      headerName: 'Run ID',
      pinned: isPinningEnabled ? 'left' : null,
      lockPosition: false,
      suppressMovable: false,
      minWidth: dynamicWidths['run_name']?.min || 120,
      width: dynamicWidths['run_name']?.default,
      maxWidth: dynamicWidths['run_name']?.max,
      wrapText: true,
      cellClass: (_params) => {
        if (!_params.data) return 'font-mono text-sm';

        // Statistics row styling
        if (_params.data.isStatisticsRow) {
          return 'bg-yellow-100 dark:bg-yellow-900/30 font-bold border-t-2 border-yellow-600';
        }

        const isHighlighted = highlightedRun === _params.data.run_name;

        let baseClass = 'font-mono text-sm';

        // User-selected highlighting (blue)
        if (isHighlighted) {
          baseClass += ' border-l-4 border-blue-500 bg-blue-50/50 dark:bg-blue-950/20 ring-2 ring-blue-300 dark:ring-blue-700';
        }

        return lineClampClass ? `${baseClass} ${lineClampClass}` : baseClass;
      },
      cellRenderer: (_params: any) => {
        if (!_params.data) return '';
        if (_params.data.isStatisticsRow) {
          return (
            <div className="flex items-center gap-2 font-bold text-primary">
              <BarChart3 className="h-4 w-4" />
              <span>{_params.data.run_name}</span>
            </div>
          );
        }
        return _params.data.run_name;
      },
    };

    const statusColDef: ColDef = {
      field: 'status',
      headerName: 'Status',
      pinned: isPinningEnabled ? 'left' : null,
      lockPosition: false,
      suppressMovable: false,
      minWidth: dynamicWidths['status']?.min || 110,
      width: dynamicWidths['status']?.default,
      maxWidth: dynamicWidths['status']?.max,
      wrapText: true,
      valueGetter: (_params) => {
        if (!_params.data) return null;
        return _params.data.status || 'unknown';
      },
      cellClass: (_params) => {
        if (!_params.data) return '';
        if (_params.data.isStatisticsRow) {
          return 'bg-yellow-100 dark:bg-yellow-900/30 font-bold border-t-2 border-yellow-600';
        }
        return '!overflow-visible z-10';
      },
      cellRenderer: (_params: any) => {
        if (!_params.data) return '';
        if (_params.data.isStatisticsRow) return '';
        const status = _params.data.status || 'unknown';
        const statusColors: Record<string, string> = {
          completed: 'bg-green-50 text-green-700 border border-green-200 dark:bg-green-950/50 dark:text-green-400 dark:border-green-800',
          running: 'bg-blue-50 text-blue-700 border border-blue-200 dark:bg-blue-950/50 dark:text-blue-400 dark:border-blue-800',
          failed: 'bg-red-50 text-red-700 border border-red-200 dark:bg-red-950/50 dark:text-red-400 dark:border-red-800',
          pending: 'bg-yellow-50 text-yellow-700 border border-yellow-200 dark:bg-yellow-950/50 dark:text-yellow-400 dark:border-yellow-800',
          unknown: 'bg-muted text-muted-foreground border border-border',
        };
        const colorClass = statusColors[status] || statusColors.unknown;
        return (
          <div className="flex items-center justify-center h-full">
            <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium transition-colors duration-200 ${colorClass}`}>
              {status}
            </span>
          </div>
        );
      },
    };

    // Tags column definition
    const tagsColDef: ColDef = {
      field: 'tags',
      headerName: 'Tags',
      pinned: isPinningEnabled ? 'left' : null,
      lockPosition: false,
      suppressMovable: false,
      minWidth: dynamicWidths['tags']?.min || 140,
      width: dynamicWidths['tags']?.default,
      maxWidth: dynamicWidths['tags']?.max,
      wrapText: true,
      cellClass: (_params) => {
        if (!_params.data) return lineClampClass;
        if (_params.data.isStatisticsRow) {
          return 'bg-yellow-100 dark:bg-yellow-900/30 font-bold border-t-2 border-yellow-600';
        }
        return lineClampClass;
      },
      valueGetter: (_params) => {
        if (!_params.data || !_params.data.tags) return null;
        return _params.data.tags;
      },
      valueFormatter: (_params) => {
        const tags = _params.value;
        if (!tags || tags.length === 0) return '';
        return tags.join(', ');
      },
      cellRenderer: (_params: any) => {
        if (!_params.data || !_params.data.tags || _params.data.tags.length === 0) {
          return <span className="text-xs text-muted-foreground italic">No tags</span>;
        }

        const tags = _params.data.tags;
        return (
          <div className="flex flex-wrap gap-1 py-0.5">
            {tags.slice(0, 3).map((tag: string, idx: number) => (
              <Badge
                key={idx}
                variant="secondary"
                className="text-xs px-1 py-0"
              >
                {tag}
              </Badge>
            ))}
            {tags.length > 3 && (
              <Badge
                variant="outline"
                className="text-xs px-1 py-0"
              >
                +{tags.length - 3}
              </Badge>
            )}
          </div>
        );
      },
    };

    // Add ALL hyperparams in hyperparamColumns order (including Run ID, Status, Tags)
    // This preserves the exact order from Column Manager
    hyperparamColumns.forEach((col) => {
      // Check if it's Run ID, Status, or Tags
      if (col === 'Run ID') {
        cols.push(runIdColDef);
        return;
      } else if (col === 'Status') {
        cols.push(statusColDef);
        return;
      } else if (col === 'Tags') {
        cols.push(tagsColDef);
        return;
      }

      // Regular hyperparam - only add if it's visible (filtered by diffMode)
      if (!filteredHyperparamColumns.includes(col)) {
        return;
      }

      const isPinned = pinnedLeftHyperparams.has(col);
      const _colId = `hp_${col}`;
      cols.push({
        field: _colId,
        headerName: col,
        pinned: isPinned ? 'left' : null,
        lockPosition: false,
        valueGetter: (_params) => {
          if (!_params.data) return null;
          // Statistics row returns empty for hyperparameters
          if (_params.data.isStatisticsRow) return '';
          return getValue(_params.data.hyperparameters || {}, col, flattenParams);
        },
        valueFormatter: (_params) => {
          const value = _params.value;
          if (value === undefined || value === null || value === '') {
            return '-';
          }
          // Check for error string
          if (typeof value === 'string' && value.toLowerCase() === 'error') {
            return 'Error';
          }
          if (typeof value === 'object') {
            // Convert to readable key-value format
            try {
              const entries = Object.entries(value);
              return entries.map(([k, v]) => {
                // Display value as-is without JSON.stringify escaping
                const displayVal = typeof v === 'string' ? v : JSON.stringify(v);
                return `${k}: ${displayVal}`;
              }).join(', ');
            } catch (e) {
              return JSON.stringify(value);
            }
          }
          return String(value);
        },
        cellClass: (_params) => {
          // Statistics row styling
          if (_params.data?.isStatisticsRow) {
            return 'bg-yellow-100 dark:bg-yellow-900/30 border-t-2 border-yellow-600';
          }

          // Check if this cell is an outlier
          const cellKey = `${_params.data?.run_name}|hyperparam.${col}`;
          const _isOutlier = cellOutliers.has(cellKey);

          const value = _params.value;
          let baseClass = '';
          if (value === undefined || value === null || value === '') {
            baseClass = 'bg-blue-50/50 dark:bg-blue-950/10 text-muted-foreground italic';
          } else if (typeof value === 'object') {
            baseClass = 'bg-blue-50/50 dark:bg-blue-950/10 text-xs text-muted-foreground italic font-mono';
          } else {
            baseClass = 'bg-blue-50/50 dark:bg-blue-950/10 text-sm';
          }

          return lineClampClass ? `${baseClass} ${lineClampClass}` : baseClass;
        },
        cellRenderer: (_params: any) => {
          const value = _params.value;

          // Handle null/undefined/empty
          if (value === undefined || value === null || value === '') {
            return '-';
          }

          // Handle error
          if (typeof value === 'string' && value.toLowerCase() === 'error') {
            return 'Error';
          }

          // Check if this cell is an outlier
          const cellKey = `${_params.data?.run_name}|hyperparam.${col}`;
          const _isOutlier = cellOutliers.has(cellKey);

          let displayValue: string;
          if (typeof value === 'object') {
            // Convert to readable key-value format
            try {
              const entries = Object.entries(value);
              displayValue = entries.map(([k, v]) => {
                // Display value as-is without JSON.stringify escaping
                const displayVal = typeof v === 'string' ? v : JSON.stringify(v);
                return `${k}: ${displayVal}`;
              }).join(', ');
            } catch (e) {
              displayValue = JSON.stringify(value);
            }
          } else {
            displayValue = String(value);
          }

          // Calculate line clamp based on rowMode
          const lineClamp = rowMode === '3-line' ? 3 : rowMode === '2-line' ? 2 : 1;

          return (
            <div className="relative w-full h-full flex items-start">
              <span
                className="overflow-hidden text-ellipsis"
                style={{
                  display: '-webkit-box',
                  WebkitBoxOrient: 'vertical',
                  WebkitLineClamp: lineClamp,
                  wordBreak: 'break-word',
                }}
                title={displayValue}
              >
                {displayValue}
              </span>
              {_isOutlier && (
                <div className="absolute bottom-0 right-0 pointer-events-none">
                  <AlertCircle className="h-3 w-3 text-orange-600 dark:text-orange-400" />
                </div>
              )}
            </div>
          );
        },
        wrapText: true,
        minWidth: dynamicWidths[_colId]?.min || 100,
        width: dynamicWidths[_colId]?.default,
        maxWidth: dynamicWidths[_colId]?.max,
      });
    });

    // Metric columns (filtered by diffMode if enabled)
    filteredMetricColumns.forEach((metric) => {
      const isPinned = pinnedRightMetrics.has(metric);
      const _colId = `metric_${metric}`;
      cols.push({
        field: _colId,
        headerName: metric,
        pinned: isPinned ? 'right' : null,
        lockPosition: false,
        headerComponent: MetricHeaderComponent,
        headerComponentParams: {
          metricName: metric,
        },
        valueGetter: (_params) => {
          if (!_params.data) return null;
          // Check if this is the statistics row
          if (_params.data.isStatisticsRow) {
            return _params.data[_colId];
          }

          // Get display _mode from ref (avoids stale closure)
          const currentMode = metricDisplayModesRef.current[metric];
          let _mode: MetricDisplayMode;

          if (currentMode) {
            _mode = currentMode;
          } else {
            // Auto-detect based on metric name
            const direction = getMetricDirection(metric);
            if (direction === 'higher') _mode = 'best-higher';
            else if (direction === 'lower') _mode = 'best-lower';
            else _mode = 'latest';
          }

          let rawValue: any;

          // Select the appropriate data field based on display mode
          switch (_mode) {
            case 'best-higher':
              rawValue = getValue(_params.data.max_metrics || {}, metric, flattenParams);
              break;
            case 'best-lower':
              rawValue = getValue(_params.data.min_metrics || {}, metric, flattenParams);
              break;
            case 'mean':
              rawValue = getValue(_params.data.mean_metrics || {}, metric, flattenParams);
              break;
            case 'median':
              rawValue = getValue(_params.data.median_metrics || {}, metric, flattenParams);
              break;
            case 'latest':
            default:
              rawValue = getValue(_params.data.final_metrics || {}, metric, flattenParams);
              break;
          }

          // Process value inline (handle null/undefined, objects with 'value' property, arrays)
          if (rawValue === null || rawValue === undefined) {
            return null;
          }
          if (typeof rawValue === 'number') {
            return rawValue;
          }
          if (typeof rawValue === 'string') {
            const parsed = parseFloat(rawValue);
            return isNaN(parsed) ? null : parsed;
          }
          if (Array.isArray(rawValue)) {
            const numericValues = rawValue.filter(v => typeof v === 'number');
            if (numericValues.length === 0) return null;
            const sum = numericValues.reduce((a, b) => a + b, 0);
            return sum / numericValues.length;
          }
          if (typeof rawValue === 'object') {
            // Try to extract numeric value from common patterns
            if ('value' in rawValue && typeof rawValue.value === 'number') {
              return rawValue.value;
            }
            if ('latest' in rawValue && typeof rawValue.latest === 'number') {
              return rawValue.latest;
            }
            if ('final' in rawValue && typeof rawValue.final === 'number') {
              return rawValue.final;
            }
            // Return the whole object for display (will be stringified in formatter)
            return rawValue;
          }
          return null;
        },
        valueFormatter: (_params) => {
          const value = _params.value;

          // Skip formatting for statistics row (handled by cellRenderer)
          if (_params.data?.isStatisticsRow) {
            return '';
          }

          if (value === undefined || value === null || value === '') {
            return 'N/A';
          }
          if (typeof value === 'number') {
            return Number(value).toFixed(4);
          }
          if (typeof value === 'object') {
            // Convert to readable key-value format
            try {
              const entries = Object.entries(value);
              return entries.map(([k, v]) => {
                // Display value as-is without JSON.stringify escaping
                const displayVal = typeof v === 'string' ? v : JSON.stringify(v);
                return `${k}: ${displayVal}`;
              }).join(', ');
            } catch (e) {
              return JSON.stringify(value);
            }
          }
          return String(value);
        },
        // Only use cellRenderer for statistics rows - omit it entirely for regular cells
        cellRendererSelector: (_params) => {
          // Only use custom renderer for statistics rows
          if (_params.data?.isStatisticsRow && _params.value && typeof _params.value === 'object') {
            return {
              component: (_props: any) => {
                const { count, min, max, mean, median } = _props.value;

                const formatValue = (val: number | null | undefined) => {
                  if (val === null || val === undefined || isNaN(val)) {
                    return '-';
                  }
                  if (Math.abs(val) < 0.01 || Math.abs(val) > 1000) {
                    return val.toExponential(2);
                  }
                  return val.toFixed(4);
                };

                return (
                  <div className="flex flex-col gap-0.5 py-1.5 text-[11px]">
                    <div className="flex items-center justify-between gap-1.5 font-bold">
                      <span className="text-xs text-muted-foreground">Count:</span>
                      <span className="font-mono text-primary">{count}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[10px] text-muted-foreground">
                      <div className="flex justify-between">
                        <span className="font-medium">Min:</span>
                        <span className="font-mono">{formatValue(min)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-medium">Max:</span>
                        <span className="font-mono">{formatValue(max)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-medium">Mean:</span>
                        <span className="font-mono">{formatValue(mean)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-medium">Median:</span>
                        <span className="font-mono">{formatValue(median)}</span>
                      </div>
                    </div>
                  </div>
                );
              }
            };
          }

          // Regular metric cells with badge overlay in bottom-right
          return {
            component: (_props: any) => {
              const value = _props.value;

              // Check if this cell is an outlier
              const cellKey = `${_props.data?.run_name}|metric.${metric}`;
              const _isOutlier = cellOutliers.has(cellKey);

              // Handle different value states: null/undefined/empty -> "-", "Error" -> "Error", otherwise use formatted value
              let formattedValue: string;
              if (value === null || value === undefined || value === '') {
                formattedValue = '-';
              } else if (String(value).toLowerCase() === 'error') {
                formattedValue = 'Error';
              } else {
                formattedValue = _props.valueFormatted || String(value);
              }

              // Get current display mode
              const currentMode = metricDisplayModesRef.current[metric];
              let _mode: MetricDisplayMode;

              if (currentMode) {
                _mode = currentMode;
              } else {
                const direction = getMetricDirection(metric);
                if (direction === 'higher') _mode = 'best-higher';
                else if (direction === 'lower') _mode = 'best-lower';
                else _mode = 'latest';
              }

              const modeBadgeText = {
                'best-higher': 'Max',
                'best-lower': 'Min',
                'latest': 'Last',
                'mean': 'Mean',
                'median': 'Med',
              }[_mode] || 'Last';

              return (
                <div className="relative w-full h-full flex items-center">
                  {/* Main value - matches AG Grid default cell rendering */}
                  <span className="font-mono">{formattedValue}</span>
                  {/* Outlier indicator icon - bottom-left overlay */}
                  {_isOutlier && (
                    <div className="absolute bottom-0 left-0 pointer-events-none">
                      <AlertCircle className="h-3 w-3 text-orange-600 dark:text-orange-400" />
                    </div>
                  )}
                  {/* Tiny overlay badge - sticker in bottom-right corner */}
                  <span
                    className="absolute bottom-0.5 right-0.5 text-[12px] px-0.5 py-0 rounded-sm bg-muted/60 text-muted-foreground/60 border border-border/30 pointer-events-none leading-none font-medium"
                    style={{ fontSize: '12px', lineHeight: '14px' }}
                  >
                    {modeBadgeText}
                  </span>
                </div>
              );
            },
          };
        },
        cellClass: (_params) => {
          // Statistics row styling
          if (_params.data?.isStatisticsRow) {
            return 'bg-yellow-100 dark:bg-yellow-900/30 text-xs font-bold border-t-2 border-yellow-600';
          }

          // Check if this cell is an outlier
          const cellKey = `${_params.data?.run_name}|metric.${metric}`;
          const _isOutlier = cellOutliers.has(cellKey);

          const value = _params.value;
          let baseClass = '';
          if (value === undefined || value === null || value === '') {
            baseClass = 'bg-green-50/50 dark:bg-green-950/10 text-muted-foreground italic';
          } else if (typeof value === 'number') {
            baseClass = 'bg-green-50/50 dark:bg-green-950/10 text-sm font-mono';
          } else {
            baseClass = 'bg-green-50/50 dark:bg-green-950/10 text-sm';
          }

          return lineClampClass ? `${baseClass} ${lineClampClass}` : baseClass;
        },
        wrapText: true,
        minWidth: dynamicWidths[_colId]?.min || 150,
        width: dynamicWidths[_colId]?.default || 150,
        maxWidth: dynamicWidths[_colId]?.max,
      });
    });

    return cols;
  }, [hyperparamColumns, filteredHyperparamColumns, filteredMetricColumns, flattenParams, cellOutliers, highlightedRun, pinnedLeftHyperparams, pinnedRightMetrics, isPinningEnabled, dynamicWidths, setMetricDisplayMode, data, selectedRunNames, toggleRowSelection, toggleAllRowsSelection]);

  // Default column definition with all features enabled
  // autoHeight is conditional: true for Max _mode, false for 2-Line mode
  const defaultColDef = useMemo<ColDef>(() => ({
    sortable: true,
    filter: true,
    resizable: true,
    editable: false,
    floatingFilter: true, // Show filter inputs below headers
    enableRowGroup: false,
    enablePivot: false,
    enableValue: false,
    suppressMenu: false,
    menuTabs: ['filterMenuTab', 'generalMenuTab', 'columnsMenuTab'],
    wrapText: true,
    cellClass: lineClampClass, // Apply line-clamp for all modes: line-clamp-1/2/3
  }), [rowMode]);

  // Update column order and pinning when hyperparamColumns, metricColumns, or pinning state changes
  // This ensures that drag & drop in Column Manager updates the table column order
  // AND that Global Pin toggle properly applies/removes pinning
  useEffect(() => {
    if (!gridRef.current?.api) return;

    const api = gridRef.current.api;

    // Check if current order matches _props (avoid infinite loop)
    const currentColumns = api.getAllDisplayedColumns();
    const currentHyperparams: string[] = [];
    const currentMetrics: string[] = [];

    currentColumns.forEach((col: any) => {
      const _colId = col.getColId();
      if (!_colId) return; // Skip if colId is undefined
      if (_colId === 'run_name') currentHyperparams.push('Run ID');
      else if (_colId === 'status') currentHyperparams.push('Status');
      else if (_colId === 'tags') currentHyperparams.push('Tags');
      else if (_colId.startsWith('hp_')) currentHyperparams.push(_colId.substring(3));
      else if (_colId.startsWith('metric_')) currentMetrics.push(_colId.substring(7));
    });

    // If order already matches, skip applyColumnState
    if (JSON.stringify(currentHyperparams) === JSON.stringify(hyperparamColumns) &&
        JSON.stringify(currentMetrics) === JSON.stringify(metricColumns)) {
      return;
    }

    // Build column state with both order and pinning information
    const columnState: any[] = [];

    // 1. AG Grid auto-generated checkbox column (always pinned left, regardless of Global Pin/Unpin)
    columnState.push({
      _colId: 'ag-Grid-SelectionColumn',
      pinned: 'left',
      lockPinned: true,  // Prevent AG Grid from changing pinning state
    });

    // 2. ALL hyperparams in hyperparamColumns order (including Run ID, Status, Tags, and regular _params)
    // This preserves the exact order from Column Manager
    const _alwaysVisibleColumns = ['Run ID', 'Status', 'Tags'];
    hyperparamColumns.forEach(col => {
      // Map column name to colId
      let _colId = '';
      let shouldPin = false;

      if (col === 'Run ID') {
        _colId = 'run_name';
        shouldPin = true; // Always pinned (but order can change in Column Manager)
      } else if (col === 'Status') {
        _colId = 'status';
        shouldPin = true; // Always pinned
      } else if (col === 'Tags') {
        _colId = 'tags';
        shouldPin = true; // Always pinned
      } else {
        // Regular hyperparam - only add if it's visible (not filtered out)
        if (!filteredHyperparamColumns.includes(col)) {
          return; // Skip invisible columns
        }
        _colId = `hp_${col}`;
        shouldPin = isPinningEnabled && pinnedLeftHyperparams.has(col);
      }

      columnState.push({
        _colId,
        pinned: shouldPin ? 'left' : null,
      });
    });

    // 3. Metrics (in metricColumns order)
    // Pinned only if: Global Pin is ON AND column is in pinnedRightMetrics
    filteredMetricColumns.forEach(metric => {
      const shouldPin = isPinningEnabled && pinnedRightMetrics.has(metric);
      columnState.push({
        _colId: `metric_${metric}`,
        pinned: shouldPin ? 'right' : null,
      });
    });

    // Apply both column order and pinning state to AG Grid
    // When Global Pin is OFF, all pinned values are null, so columns follow the order exactly
    // When Global Pin is ON, pinned columns go to their pinned areas, unpinned follow the order
    api.applyColumnState({
      state: columnState,
      applyOrder: true,
    });
  }, [hyperparamColumns, metricColumns, filteredHyperparamColumns, filteredMetricColumns, isPinningEnabled, pinnedLeftHyperparams, pinnedRightMetrics]);

  // Handle grid ready
  const onGridReady = useCallback((_params: GridReadyEvent) => {
    // NOTE: Checkbox column pinning is handled by applyColumnState in applyColumnOrder
    // All modes (1-line, 2-line, 3-line) use fixed row heights and calculated column widths
  }, []);

  // Sync selection state when selectedRows prop changes
  useEffect(() => {
    if (gridRef.current?.api && data.length > 0) {
      const api = gridRef.current.api;

      // Deselect all first
      api.deselectAll();

      // Select rows that match selectedRows prop
      if (selectedRows.length > 0) {
        api.forEachNode((node) => {
          if (node.data && selectedRows.includes(node.data.run_name)) {
            node.setSelected(true);
          }
        });
      }
    }
  }, [selectedRows, data]);

  // Scroll to highlighted row when it changes
  useEffect(() => {
    if (gridRef.current?.api && highlightedRun) {
      const api = gridRef.current.api;

      // Find the row node
      let targetRowIndex: number | null = null;
      api.forEachNode((node, index) => {
        if (node.data && node.data.run_name === highlightedRun) {
          targetRowIndex = index;
        }
      });

      // Scroll to the row
      if (targetRowIndex !== null) {
        api.ensureIndexVisible(targetRowIndex, 'middle');
      }
    }
  }, [highlightedRun]);

  // Note: Selection change is now handled via useEffect on selectedRunNames state
  // (removed old onSelectionChanged callback that relied on AG Grid's built-in row selection)

  // Handle row click
  const onCellClicked = useCallback((event: any) => {
    // Skip if click is on checkbox column
    if (event.column?._colId === 'checkbox') {
      return;
    }

    // Skip statistics row
    if (event.data?.isStatisticsRow) {
      return;
    }

    if (onRowClick && event.data) {
      onRowClick(event.data);
    }
  }, [onRowClick]);

  // Handle column pinning changes (feedback to Column Manager)
  const onColumnPinned = useCallback((event: any) => {
    if (!onColumnStateChange) return;

    // Extract pinning state from all columns
    const allColumns = event.api.getAllDisplayedColumns();
    const newPinnedLeft = new Set<string>();
    const newPinnedRight = new Set<string>();

    allColumns.forEach((col: any) => {
      const _colId = col.getColId();
      if (!_colId) return; // Skip if colId is undefined
      const pinned = col.getPinned();

      // Extract original column name (remove hp_ or metric_ prefix)
      if (_colId.startsWith('hp_')) {
        const columnName = _colId.substring(3);  // Remove 'hp_' prefix
        if (pinned === 'left') {
          newPinnedLeft.add(columnName);
        }
      } else if (_colId.startsWith('metric_')) {
        const columnName = _colId.substring(7);  // Remove 'metric_' prefix
        if (pinned === 'right') {
          newPinnedRight.add(columnName);
        }
      }
    });

    // Notify parent of pinning state change
    onColumnStateChange(newPinnedLeft, newPinnedRight);
  }, [onColumnStateChange]);

  // Handle column order changes (feedback to Column Manager)
  const onColumnMoved = useCallback((event: any) => {
    if (!onColumnOrderChange) return;

    // Extract current column order from AG Grid
    const allColumns = event.api.getAllDisplayedColumns();
    const newHyperparamOrder: string[] = [];
    const newMetricOrder: string[] = [];

    allColumns.forEach((col: any) => {
      const _colId = col.getColId();
      if (!_colId) return; // Skip if colId is undefined

      // Map _colId back to original column name
      if (_colId === 'run_name') {
        newHyperparamOrder.push('Run ID');
      } else if (_colId === 'status') {
        newHyperparamOrder.push('Status');
      } else if (_colId === 'tags') {
        newHyperparamOrder.push('Tags');
      } else if (_colId.startsWith('hp_')) {
        const columnName = _colId.substring(3);  // Remove 'hp_' prefix
        newHyperparamOrder.push(columnName);
      } else if (_colId.startsWith('metric_')) {
        const columnName = _colId.substring(7);  // Remove 'metric_' prefix
        newMetricOrder.push(columnName);
      }
      // Skip 'checkbox' column
    });

    // Check if order actually changed
    const prevOrder = prevOrderRef.current;
    const hyperparamsChanged = JSON.stringify(newHyperparamOrder) !== JSON.stringify(prevOrder.hyperparams);
    const metricsChanged = JSON.stringify(newMetricOrder) !== JSON.stringify(prevOrder.metrics);

    if (!hyperparamsChanged && !metricsChanged) {
      return; // No change, skip callback
    }

    // Update ref and notify parent
    prevOrderRef.current = { hyperparams: newHyperparamOrder, metrics: newMetricOrder };
    onColumnOrderChange(newHyperparamOrder, newMetricOrder);
  }, [onColumnOrderChange]);

  // Create custom theme based on Quartz with Tailwind colors
  // Reduced padding and spacing for compact display
  const customTheme = useMemo(() =>
    themeQuartz.withParams({
      backgroundColor: 'hsl(var(--background))',
      foregroundColor: 'hsl(var(--foreground))',
      headerBackgroundColor: 'hsl(var(--muted) / 0.3)',
      headerTextColor: 'hsl(var(--muted-foreground))',
      oddRowBackgroundColor: 'hsl(var(--background))',
      rowHoverColor: 'hsl(var(--muted) / 0.3)',
      borderColor: 'hsl(var(--border))',
      selectedRowBackgroundColor: 'hsl(var(--accent) / 0.2)',
      cellHorizontalPadding: 4,
      spacing: 2,
    })
  , []);

  return (
    <div className="w-full">
      {/* AG Grid */}
      <div style={{ height: '600px' }} className="ag-theme-custom">
        <style>{`
          /* AG Grid cell styling - Using CSS variables from :root (index.css) */
          .ag-theme-custom .ag-cell {
            padding: var(--ag-cell-padding-vertical) var(--ag-cell-padding-horizontal) !important;
            line-height: var(--ag-cell-line-height) !important;
            font-size: var(--ag-cell-font-size) !important;
            /* Text wrapping */
            white-space: normal !important;
            word-break: break-all !important;
            overflow-wrap: anywhere !important;
            /* Fix outlier border layout shift */
            box-sizing: border-box !important;
          }
          /* Apply word-break to all child elements inside cells */
          .ag-theme-custom .ag-cell *,
          .ag-theme-custom .ag-cell span,
          .ag-theme-custom .ag-cell div {
            word-break: break-all !important;
            overflow-wrap: anywhere !important;
          }
          /* Override specific line-clamp selectors */
          .ag-theme-custom .ag-cell.line-clamp-2 .ag-cell-wrapper > div:not([class*="flex"]) > span,
          .ag-theme-custom .ag-cell.line-clamp-2 .ag-cell-wrapper > span,
          .ag-theme-custom .ag-cell.line-clamp-1 .ag-cell-wrapper > div:not([class*="flex"]) > span,
          .ag-theme-custom .ag-cell.line-clamp-1 .ag-cell-wrapper > span,
          .ag-theme-custom .ag-cell.line-clamp-3 .ag-cell-wrapper > div:not([class*="flex"]) > span,
          .ag-theme-custom .ag-cell.line-clamp-3 .ag-cell-wrapper > span,
          .ag-theme-custom .ag-pinned-left-cols-container .ag-cell .ag-cell-wrapper > div:not([class*="flex"]) > span,
          .ag-theme-custom .ag-pinned-left-cols-container .ag-cell .ag-cell-wrapper > span,
          .ag-theme-custom .ag-pinned-right-cols-container .ag-cell .ag-cell-wrapper > div:not([class*="flex"]) > span,
          .ag-theme-custom .ag-pinned-right-cols-container .ag-cell .ag-cell-wrapper > span {
            word-break: break-all !important;
            overflow-wrap: anywhere !important;
          }
          .ag-theme-custom .ag-header-cell {
            padding: var(--ag-cell-padding-vertical) var(--ag-cell-padding-horizontal) !important;
            font-size: var(--ag-cell-font-size) !important;
          }
          /* 체크박스 정렬 */
          .ag-theme-custom .ag-cell[col-id="checkbox"],
          .ag-theme-custom .ag-header-cell[col-id="checkbox"] {
            padding: 0 !important;
            overflow: visible !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
          }
          /* Metric 컬럼 상단 정렬 */
          .ag-theme-custom .ag-cell[col-id^="metric_"] {
            align-items: flex-start !important;
          }
          /* Hyperparameter 컬럼 상단 정렬 */
          .ag-theme-custom .ag-cell[col-id^="hp_"] {
            align-items: flex-start !important;
          }
          /* Status 컬럼 뱃지 가시성 */
          .ag-theme-custom .ag-cell[col-id="status"] {
            overflow: visible !important;
            padding: 0 !important;
          }
          /* 스크롤 동기화: 모든 애니메이션 완전 제거 */
          .ag-theme-custom *,
          .ag-theme-custom *::before,
          .ag-theme-custom *::after {
            transition: none !important;
            animation: none !important;
          }
          .ag-theme-custom .ag-header-viewport,
          .ag-theme-custom .ag-header,
          .ag-theme-custom .ag-header-container,
          .ag-theme-custom .ag-header-cell,
          .ag-theme-custom .ag-pinned-left-header,
          .ag-theme-custom .ag-pinned-right-header,
          .ag-theme-custom .ag-header-row,
          .ag-theme-custom .ag-body-viewport,
          .ag-theme-custom .ag-body-horizontal-scroll-viewport {
            transition: none !important;
            animation: none !important;
            transform: translate3d(0, 0, 0) !important;
          }
          /* Remove yellow background from pinned bottom row (STATISTICS) */
          .ag-theme-custom .ag-row-pinned {
            background-color: hsl(var(--muted) / 0.5) !important;
          }
          .ag-theme-custom .ag-row-pinned .ag-cell {
            background-color: transparent !important;
            font-weight: 500 !important;
          }
        `}</style>
        <AgGridReact
          ref={gridRef}
          theme={customTheme}
          rowData={data}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          pinnedBottomRowData={pinnedBottomRowData}
          rowHeight={getRowHeight(rowMode)}
          headerHeight={getHeaderHeight(rowMode)}
          floatingFiltersHeight={30}
          onGridReady={onGridReady}
          onCellClicked={onCellClicked}
          onColumnPinned={onColumnPinned}
          onColumnMoved={onColumnMoved}
          animateRows={false}
          suppressAnimationFrame={true}
          enableCellTextSelection={true}
          suppressColumnVirtualisation={false}
          suppressHorizontalScroll={false}
          alwaysShowHorizontalScroll={true}
          quickFilterText={quickFilterText}
          pagination={true}
          paginationPageSize={50}
          paginationPageSizeSelector={[20, 50, 100, 200]}
          enableBrowserTooltips={true}
          tooltipShowDelay={500}
          suppressMenuHide={false}
          enableRangeSelection={true}
          enableCharts={false}
          multiSortKey="ctrl"
        />
      </div>

      {/* Column Statistics Dialog */}
      {columnStatsDialog && (
        <ColumnStatisticsDialog
          isOpen={columnStatsDialog.isOpen}
          onClose={() => setColumnStatsDialog(null)}
          metricName={columnStatsDialog.metricName}
          values={columnStatsDialog.values}
        />
      )}
    </div>
  );
}
