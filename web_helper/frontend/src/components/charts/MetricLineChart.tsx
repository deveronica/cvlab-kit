import React from "react";

import { useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Expand } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { ExportMenu } from '../ui/export-menu';
import type { Run } from '../../lib/types';
import { generateChartFilename } from '../../lib/chart-utils';
import { useChartTheme } from '../../contexts/ChartThemeContext';
import { ChartThemeSelector } from '../ui/chart-theme-selector';
import { getThemeColor } from '../../lib/chart-themes';
import { devLog, devError, devWarn } from '../../lib/dev-utils';

interface MetricLineChartProps {
  runs: Run[];
  metricKey: string;
  title?: string;
  height?: number;
  showLegend?: boolean;
}

interface ChartDataPoint {
  step: number;
  [runId: string]: number | string;
}

const CHART_COLORS = [
  '#8b5cf6', // violet-500
  '#06b6d4', // cyan-500
  '#10b981', // emerald-500
  '#f59e0b', // amber-500
  '#ef4444', // red-500
  '#8b5cf6', // violet-500
  '#f97316', // orange-500
  '#84cc16', // lime-500
  '#ec4899', // pink-500
  '#6366f1', // indigo-500
];

export function MetricLineChart({
  runs,
  metricKey,
  title,
  height = 400,
  showLegend = true
}: MetricLineChartProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const chartContainerRef = React.useRef<HTMLDivElement>(null);

  // Process data to create chart format
  const chartData = React.useMemo(() => {
    if (!runs.length) return [];

    // Get all unique steps across all runs
    const allSteps = new Set<number>();
    runs.forEach(run => {
      if (run.metrics?.timeseries) {
        run.metrics.timeseries.forEach(point => {
          allSteps.add(point.step);
        });
      }
    });

    const sortedSteps = Array.from(allSteps).sort((a, b) => a - b);

    // Create data points for each step
    return sortedSteps.map(step => {
      const dataPoint: ChartDataPoint = { step };

      runs.forEach(run => {
        const timeseriesPoint = run.metrics?.timeseries?.find(t => t.step === step);
        const value = timeseriesPoint?.values[metricKey];

        if (value !== undefined) {
          dataPoint[run.run_name] = value;
        }
      });

      return dataPoint;
    });
  }, [runs, metricKey]);

  // Export functions
  const exportToPNG = useCallback((event?: React.MouseEvent) => {
    let svgElement: SVGSVGElement | null = null;

    // Method 1: Try using ref
    if (chartContainerRef.current) {
      const responsiveContainer = chartContainerRef.current.querySelector('.recharts-responsive-container');
      if (responsiveContainer) {
        // Get the main chart SVG (not legend)
        svgElement = responsiveContainer.querySelector('svg.recharts-surface') as SVGSVGElement;
      }
    }

    // Method 2: Try finding from event target (fallback)
    if (!svgElement && event?.currentTarget) {
      const button = event.currentTarget as HTMLElement;
      const card = button.closest('.bg-card');
      if (card) {
        const responsiveContainer = card.querySelector('.recharts-responsive-container');
        if (responsiveContainer) {
          svgElement = responsiveContainer.querySelector('svg.recharts-surface') as SVGSVGElement;
        }
      }
    }

    if (!svgElement) {
      devError('Could not find chart SVG for export', {
        hasRef: !!chartContainerRef.current,
        metricKey
      });
      alert('차트를 찾을 수 없습니다. 페이지를 새로고침 후 다시 시도해주세요.');
      return;
    }

    // Get SVG's native dimensions (NOT getBBox which only captures content area)
    const width = svgElement.width.baseVal.value || svgElement.clientWidth || 800;
    const height = svgElement.height.baseVal.value || svgElement.clientHeight || 400;

    devLog('Found SVG for export:', {
      width,
      height,
      metricKey,
      className: svgElement.className.baseVal,
      viewBox: svgElement.getAttribute('viewBox'),
      childElementCount: svgElement.childElementCount
    });

    // Clone SVG to avoid modifying the original
    const clonedSvg = svgElement.cloneNode(true) as SVGSVGElement;

    // Set explicit dimensions from original SVG
    clonedSvg.setAttribute('width', width.toString());
    clonedSvg.setAttribute('height', height.toString());

    const svgData = new XMLSerializer().serializeToString(clonedSvg);
    const canvas = document.createElement('canvas');
    canvas.width = width * 2; // 2x for better quality
    canvas.height = height * 2;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.scale(2, 2);
      ctx.drawImage(img, 0, 0);

      canvas.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = generateChartFilename(`${metricKey}_line_chart`, 'png');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      }, 'image/png');
    };

    img.onerror = (err) => {
      devError('Failed to export chart as PNG:', err);
    };

    const encodedData = encodeURIComponent(svgData);
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodedData;
  }, [metricKey]);

  const exportToSVG = useCallback(() => {
    if (!chartContainerRef.current) {
      devWarn('Chart container ref not available');
      return;
    }

    const responsiveContainer = chartContainerRef.current.querySelector('.recharts-responsive-container');
    if (!responsiveContainer) {
      devWarn('No recharts container found for export');
      return;
    }

    const svgElement = responsiveContainer.querySelector('svg.recharts-surface');
    if (!svgElement) return;

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const blob = new Blob([svgData], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = generateChartFilename(`${metricKey}_line_chart`, 'svg');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [metricKey]);

  const exportToCSV = useCallback(() => {
    const runNames = runs.map(r => r.run_name);
    const headers = ['step', ...runNames];
    const csvRows = [headers.join(',')];

    chartData.forEach(point => {
      const row = [
        point.step,
        ...runNames.map(name => point[name] !== undefined ? point[name] : '')
      ];
      csvRows.push(row.join(','));
    });

    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = generateChartFilename(`${metricKey}_data`, 'csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [metricKey, chartData, runs]);

  if (!chartData.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title || `${metricKey} over time`}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            No data available for {metricKey}
          </div>
        </CardContent>
      </Card>
    );
  }

  const renderChart = () => (
    <LineChart
      data={chartData}
      margin={{
        top: 5,
        right: 30,
        left: 20,
        bottom: 5,
      }}
    >
      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
      <XAxis
        dataKey="step"
        tick={{ fontSize: 12 }}
        className="text-muted-foreground"
      />
      <YAxis
        tick={{ fontSize: 12 }}
        className="text-muted-foreground"
      />
      <Tooltip
        contentStyle={{
          backgroundColor: 'hsl(var(--background))',
          border: '1px solid hsl(var(--border))',
          borderRadius: '6px',
          boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
        }}
        labelStyle={{ color: 'hsl(var(--foreground))' }}
      />
      {showLegend && (
        <Legend
          wrapperStyle={{ paddingTop: '20px' }}
          iconType="line"
        />
      )}
      {runs.map((run, index) => (
        <Line
          key={run.run_name}
          type="monotone"
          dataKey={run.run_name}
          stroke={CHART_COLORS[index % CHART_COLORS.length]}
          strokeWidth={2}
          dot={{ r: 3 }}
          activeDot={{ r: 5 }}
          name={`${run.run_name.substring(0, 8)}...`}
          connectNulls={false}
        />
      ))}
    </LineChart>
  );

  return (
    <>
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <CardTitle className="truncate mr-2 flex-shrink min-w-0">{title || `${metricKey} over time`}</CardTitle>

          <div className="flex items-center gap-2 flex-shrink-0">
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
      <CardContent ref={chartContainerRef}>
        <ResponsiveContainer width="100%" height={height}>
          {renderChart()}
        </ResponsiveContainer>
      </CardContent>
    </Card>

      {/* Fullscreen Modal */}
      <Dialog open={isFullscreen} onOpenChange={setIsFullscreen}>
        <DialogContent className="max-w-[95vw] max-h-[95vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle>{title || `${metricKey} over time`}</DialogTitle>
          </DialogHeader>
          <div className="flex-1 min-h-[70vh]">
            <ResponsiveContainer width="100%" height="100%">
              {renderChart()}
            </ResponsiveContainer>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}