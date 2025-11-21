import React from "react";
/**
 * Custom hook for chart export functionality
 *
 * Provides unified export handlers for PNG, SVG, and CSV with proper ref handling
 */

import { useCallback, useRef } from 'react';
import { exportChartToPNG, exportChartToSVG, exportChartDataToCSV, ExportOptions } from '../lib/chart-export';
import { devError } from '../lib/dev-utils';

export interface UseChartExportOptions<T = any> {
  /** Base filename for exports (timestamp will be added automatically) */
  filename: string;
  /** Chart data for CSV export */
  data?: T[];
  /** Column names for CSV export (optional, defaults to all object keys) */
  csvColumns?: string[];
  /** PNG export options */
  pngOptions?: ExportOptions;
}

export interface ChartExportHandlers {
  /** Ref to attach to the chart container element */
  chartContainerRef: React.RefObject<HTMLDivElement>;
  /** Export chart as PNG */
  exportToPNG: () => Promise<void>;
  /** Export chart as SVG */
  exportToSVG: () => void;
  /** Export chart data as CSV */
  exportToCSV: () => void;
}

/**
 * Hook for unified chart export functionality
 *
 * @example
 * ```tsx
 * function MyChart({ data, metricKey }) {
 *   const { chartContainerRef, exportToPNG, exportToSVG, exportToCSV } = useChartExport({
 *     filename: metricKey,
 *     data: chartData,
 *   });
 *
 *   return (
 *     <CardContent ref={chartContainerRef}>
 *       <ResponsiveContainer>
 *         <LineChart data={data}>...</LineChart>
 *       </ResponsiveContainer>
 *     </CardContent>
 *   );
 * }
 * ```
 */
export function useChartExport<T extends Record<string, any> = any>(
  options: UseChartExportOptions<T>
): ChartExportHandlers {
  const { filename, data = [], csvColumns, pngOptions = {} } = options;

  const chartContainerRef = useRef<HTMLDivElement>(null);

  const exportToPNG = useCallback(async () => {
    try {
      await exportChartToPNG(chartContainerRef, filename, pngOptions);
    } catch (error) {
      devError('PNG export failed:', error);
      alert('PNG 내보내기에 실패했습니다.');
    }
  }, [filename, pngOptions]);

  const exportToSVG = useCallback(() => {
    try {
      exportChartToSVG(chartContainerRef, filename);
    } catch (error) {
      devError('SVG export failed:', error);
      alert('SVG 내보내기에 실패했습니다.');
    }
  }, [filename]);

  const exportToCSV = useCallback(() => {
    try {
      exportChartDataToCSV(data, filename, csvColumns);
    } catch (error) {
      devError('CSV export failed:', error);
      alert('CSV 내보내기에 실패했습니다.');
    }
  }, [data, filename, csvColumns]);

  return {
    chartContainerRef,
    exportToPNG,
    exportToSVG,
    exportToCSV,
  };
}
