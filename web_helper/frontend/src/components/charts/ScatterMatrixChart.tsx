import React from "react";
/**
 * Scatter Matrix Chart
 *
 * Creates a matrix of scatter plots showing pairwise relationships
 * between multiple metrics.
 *
 * Features:
 * - n×n grid of scatter plots for n metrics
 * - Diagonal shows metric distributions (histograms)
 * - Off-diagonal shows scatter plots with correlation coefficients
 * - Interactive highlighting across all plots
 * - Click to drill down to specific metric pair
 */

import { useMemo, useState, useCallback, useRef } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Expand } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { ExportMenu } from '../ui/export-menu';
import type { Run } from '../../lib/types';
import { transformForScatterMatrix } from '../../lib/chartDataTransformers';

interface ScatterMatrixChartProps {
  runs: Run[];
  metricKeys: string[];
  title?: string;
  height?: number;
  showCorrelation?: boolean;
  onCellClick?: (metricX: string, metricY: string) => void;
  variant?: 'default' | 'compact';
}

export function ScatterMatrixChart({
  runs,
  metricKeys,
  title,
  height = 600,
  showCorrelation = true,
  onCellClick,
  variant = 'default',
}: ScatterMatrixChartProps) {
  const [highlightedRun, setHighlightedRun] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const chartRef = useRef<ReactECharts>(null);

  // Transform data
  const matrixData = useMemo(() => {
    return transformForScatterMatrix(runs, metricKeys);
  }, [runs, metricKeys]);

  // Export functions for ECharts
  const exportToPNG = useCallback(() => {
    if (!chartRef.current) return;
    const echartInstance = chartRef.current.getEchartsInstance();
    const url = echartInstance.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
    const link = document.createElement('a');
    link.href = url;
    link.download = `scatter_matrix_${new Date().toISOString().split('T')[0]}.png`;
    link.click();
  }, []);

  const exportToSVG = useCallback(() => {
    if (!chartRef.current) return;
    const echartInstance = chartRef.current.getEchartsInstance();
    const url = echartInstance.getDataURL({ type: 'svg' });
    const link = document.createElement('a');
    link.href = url;
    link.download = `scatter_matrix_${new Date().toISOString().split('T')[0]}.svg`;
    link.click();
  }, []);

  const exportToCSV = useCallback(() => {
    const headers = ['metric_x', 'metric_y', 'pearson_r'];
    const csvRows = [headers.join(',')];

    matrixData.cells.forEach(row => {
      row.forEach(cell => {
        if (cell.pearsonR !== undefined) {
          const csvRow = [cell.metricX, cell.metricY, cell.pearsonR];
          csvRows.push(csvRow.join(','));
        }
      });
    });

    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `scatter_matrix_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [matrixData]);

  // Prepare ECharts option
  const option = useMemo((): EChartsOption => {
    if (matrixData.metrics.length === 0) {
      return {};
    }

    const n = matrixData.metrics.length;
    const gridSize = 100 / (n + 1); // +1 for spacing

    const grids: any[] = [];
    const xAxes: any[] = [];
    const yAxes: any[] = [];
    const series: any[] = [];

    // Create n×n grid
    matrixData.cells.forEach((row, i) => {
      row.forEach((cell, j) => {
        const gridIndex = i * n + j;

        // Grid configuration
        grids.push({
          left: `${(j + 0.5) * gridSize}%`,
          top: `${(i + 0.5) * gridSize}%`,
          width: `${gridSize * 0.9}%`,
          height: `${gridSize * 0.9}%`,
        });

        // X-axis configuration
        xAxes.push({
          gridIndex,
          type: 'value',
          show: i === n - 1, // Only show for bottom row
          name: j === Math.floor(n / 2) && i === n - 1 ? cell.metricX : '',
          nameLocation: 'middle',
          nameGap: 25,
          nameTextStyle: {
            fontSize: 11,
            fontWeight: 'bold',
          },
          axisLabel: {
            show: i === n - 1,
            fontSize: 9,
            formatter: (value: number) => value.toFixed(2),
          },
          splitLine: { show: false },
        });

        // Y-axis configuration
        yAxes.push({
          gridIndex,
          type: 'value',
          show: j === 0, // Only show for left column
          name: i === Math.floor(n / 2) && j === 0 ? cell.metricY : '',
          nameLocation: 'middle',
          nameGap: 30,
          nameTextStyle: {
            fontSize: 11,
            fontWeight: 'bold',
          },
          axisLabel: {
            show: j === 0,
            fontSize: 9,
            formatter: (value: number) => value.toFixed(2),
          },
          splitLine: { show: false },
        });

        // Series configuration
        if (i === j) {
          // Diagonal: show distribution (histogram simulation using scatter)
          const values = cell.points.map(p => p.y);
          const bins = 10;
          const min = Math.min(...values);
          const max = Math.max(...values);
          const binWidth = (max - min) / bins;

          const histogram: { x: number; y: number }[] = [];
          for (let b = 0; b < bins; b++) {
            const binMin = min + b * binWidth;
            const binMax = min + (b + 1) * binWidth;
            const count = values.filter(v => v >= binMin && v < binMax).length;
            histogram.push({
              x: (binMin + binMax) / 2,
              y: count,
            });
          }

          series.push({
            type: 'bar',
            xAxisIndex: gridIndex,
            yAxisIndex: gridIndex,
            data: histogram.map(h => [h.x, h.y]),
            itemStyle: {
              color: 'hsl(var(--primary))',
              opacity: 0.7,
            },
            barWidth: '80%',
          });
        } else {
          // Off-diagonal: scatter plot
          series.push({
            type: 'scatter',
            xAxisIndex: gridIndex,
            yAxisIndex: gridIndex,
            data: cell.points.map(p => ({
              value: [p.x, p.y],
              name: p.runName,
            })),
            symbolSize: 6,
            itemStyle: {
              color: 'hsl(var(--primary))',
              opacity: 0.6,
            },
            emphasis: {
              itemStyle: {
                color: 'hsl(var(--destructive))',
                opacity: 1,
              },
            },
          });

          // Add correlation coefficient text if enabled
          if (showCorrelation && cell.pearsonR !== undefined) {
            const r = cell.pearsonR;
            const absR = Math.abs(r);
            let color = '#888';
            if (absR >= 0.7) color = '#10b981'; // Strong correlation
            else if (absR >= 0.4) color = '#f59e0b'; // Moderate correlation

            series.push({
              type: 'custom',
              xAxisIndex: gridIndex,
              yAxisIndex: gridIndex,
              renderItem: () => ({
                type: 'text',
                style: {
                  text: `r=${r.toFixed(2)}`,
                  fontSize: 10,
                  fontWeight: 'bold',
                  fill: color,
                  x: '50%',
                  y: '10%',
                  textAlign: 'center',
                },
              }),
              data: [0],
              silent: true,
            });
          }
        }
      });
    });

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (!params.data || !params.data.name) return '';

          const runName = params.data.name;
          const [x, y] = params.data.value;

          const i = Math.floor(params.seriesIndex / n);
          const j = params.seriesIndex % n;
          const cell = matrixData.cells[i][j];

          return `
            <div style="font-weight: bold; margin-bottom: 4px;">${runName}</div>
            <div style="color: #888; font-size: 11px;">
              ${cell.metricX}: ${x.toFixed(4)}<br/>
              ${cell.metricY}: ${y.toFixed(4)}
            </div>
          `;
        },
      },
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      series,
    };
  }, [matrixData, showCorrelation]);

  // Handle chart events
  const onEvents = useMemo(() => {
    return {
      click: (params: any) => {
        const n = matrixData.metrics.length;
        const i = Math.floor(params.seriesIndex / n);
        const j = params.seriesIndex % n;

        if (i !== j && onCellClick) {
          const cell = matrixData.cells[i][j];
          onCellClick(cell.metricX, cell.metricY);
        }

        if (params.data && params.data.name) {
          setHighlightedRun(params.data.name);
        }
      },
      mouseout: () => {
        setHighlightedRun(null);
      },
    };
  }, [matrixData, onCellClick]);

  if (matrixData.metrics.length === 0) {
    return (
      <Card variant={variant}>
        <CardHeader variant={variant}>
          <CardTitle>{title || 'Scatter Matrix'}</CardTitle>
        </CardHeader>
        <CardContent variant={variant}>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            No metrics available for scatter matrix visualization
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
    <Card variant={variant}>
      <CardHeader variant={variant}>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle size="base">
              {title || 'Scatter Matrix'}
            </CardTitle>
            <CardDescription className="mt-1">
              {matrixData.metrics.length}×{matrixData.metrics.length} correlation matrix
            </CardDescription>
          </div>

          <div className="flex items-center gap-2">
            {showCorrelation && (
              <Badge variant="outline" className="text-xs">
                With Correlations
              </Badge>
            )}
            {highlightedRun && (
              <Badge className="text-xs">
                {highlightedRun}
              </Badge>
            )}

            {/* Fullscreen */}
            <Button variant="outline" size="sm" onClick={() => setIsFullscreen(true)} className="h-7">
              <Expand className="h-4 w-4" />
            </Button>

            {/* Export */}
            <ExportMenu
              onExport={(format) => {
                if (format === 'png') exportToPNG();
                else if (format === 'svg') exportToSVG();
                else if (format === 'csv') exportToCSV();
              }}
              formats={['png', 'svg', 'csv']}
              showLabel={false}
              size="sm"
            />
          </div>
        </div>
      </CardHeader>

      <CardContent variant={variant}>
        <ReactECharts
          ref={chartRef}
          option={option}
          style={{ height: `${height}px`, width: '100%' }}
          onEvents={onEvents}
          opts={{ renderer: 'canvas' }}
        />

        {/* Legend */}
        <div className="mt-2 text-xs text-muted-foreground border-t pt-2">
          <div className="grid grid-cols-3 gap-3">
            <div>
              <strong>Diagonal:</strong> Distribution histograms
            </div>
            <div>
              <strong>Off-diagonal:</strong> Scatter plots with correlation
            </div>
            <div>
              <strong>Colors:</strong>{' '}
              <span className="text-green-600 dark:text-green-400">Strong (|r|≥0.7)</span>,{' '}
              <span className="text-yellow-600 dark:text-yellow-400">Moderate (|r|≥0.4)</span>,{' '}
              <span className="text-gray-600 dark:text-gray-400">Weak</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>

      {/* Fullscreen Modal */}
      <Dialog open={isFullscreen} onOpenChange={setIsFullscreen}>
        <DialogContent className="max-w-[95vw] max-h-[95vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle>{title || 'Scatter Matrix'}</DialogTitle>
          </DialogHeader>
          <div className="flex-1 min-h-[70vh]">
            <ReactECharts
              option={option}
              style={{ height: '100%', width: '100%' }}
              onEvents={onEvents}
              opts={{ renderer: 'canvas' }}
            />
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
