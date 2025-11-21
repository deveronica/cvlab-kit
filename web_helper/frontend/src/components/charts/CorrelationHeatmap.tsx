import React from "react";

import { useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { ChartCard } from '../ui/chart-card';
import { ChartSettingsPanel, ChartSettings } from './ChartSettingsPanel';
import { ExportFormat } from '../ui/export-menu';

interface CorrelationHeatmapProps {
  data: any[];
  xAxis: string[];
  yAxis: string[];
  title?: string;
  description?: string;
  height?: number;
  showSettings?: boolean;
  showFullscreen?: boolean;
  showExport?: boolean;
  variant?: 'default' | 'compact';
}

const DEFAULT_SETTINGS: ChartSettings = {
  animation: true,
  height: 450,
  legend: { show: false },
  xAxis: { showGrid: false },
  yAxis: { showGrid: false, scale: 'linear' },
};

const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  data,
  xAxis,
  yAxis,
  title = 'Correlation Heatmap',
  description,
  height = 450,
  showSettings = true,
  showFullscreen = true,
  showExport = true,
  variant = 'default',
}) => {
  // Local settings state
  const [settings, setSettings] = useState<ChartSettings>(DEFAULT_SETTINGS);

  const option = useMemo(() => {
    // Color palette mapping
    const colorPalettes = {
      'blue-red': ['#3b82f6', '#f3f4f6', '#ef4444'],
      'viridis': ['#440154', '#31688e', '#35b779', '#fde724'],
      'coolwarm': ['#3b4cc0', '#f7f7f7', '#b40426'],
      'spectral': ['#5e4fa2', '#66c2a5', '#fee08b', '#f46d43', '#9e0142'],
    };

    const colorPalette = settings.heatmap?.colorPalette || 'blue-red';
    const showValues = settings.heatmap?.showValues !== false;
    const minValue = settings.heatmap?.minCorrelation ?? -1;
    const maxValue = settings.heatmap?.maxCorrelation ?? 1;

    return {
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          if (!params.data || params.data.length < 3) return '';
          const [xIdx, yIdx, value] = params.data;
          const xLabel = xAxis[xIdx] || 'Unknown';
          const yLabel = yAxis[yIdx] || 'Unknown';
          const correlation = typeof value === 'number' ? value.toFixed(3) : 'N/A';
          return `
            <div style="padding: 8px;">
              <div style="font-weight: bold; margin-bottom: 4px;">Correlation</div>
              <div style="color: #6b7280; font-size: 12px;">${yLabel} vs ${xLabel}</div>
              <div style="font-size: 18px; font-weight: bold; margin-top: 4px;">${correlation}</div>
            </div>
          `;
        },
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderColor: '#e5e7eb',
        borderWidth: 1,
        textStyle: {
          color: '#374151',
        },
        extraCssText: 'box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);',
      },
      grid: {
        height: '60%',
        top: '8%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: xAxis,
        splitArea: {
          show: true,
        },
        axisLabel: {
          rotate: 45,
          fontSize: 10,
          interval: 0, // Show all labels
        },
      },
      yAxis: {
        type: 'category',
        data: yAxis,
        splitArea: {
          show: true,
        },
        axisLabel: {
          fontSize: 10,
          width: 80,
          overflow: 'truncate',
          ellipsis: '...',
        },
      },
      visualMap: {
        min: minValue,
        max: maxValue,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '15%',
        text: ['High Correlation', 'Low Correlation'],
        textStyle: {
          fontSize: 11,
        },
        inRange: {
          color: colorPalettes[colorPalette],
        },
      },
      series: [
        {
          name: 'Correlation',
          type: 'heatmap',
          data: data,
          label: {
            show: showValues,
            formatter: (params: any) => {
              if (!params.data || params.data.length < 3) return '';
              const value = params.data[2];
              // Format to 2 decimal places, hide decimal if too narrow
              if (typeof value !== 'number') return '';
              // Show sign + 1 decimal for very compact display
              return value.toFixed(1);
            },
            fontSize: 9,
            fontWeight: 'bold',
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
            label: {
              fontSize: 11,
              formatter: (params: any) => {
                if (!params.data || params.data.length < 3) return '';
                const value = params.data[2];
                // Show full precision on hover
                return typeof value === 'number' ? value.toFixed(3) : '';
              },
            },
          },
        },
      ],
    };
  }, [data, xAxis, yAxis, settings]);

  // Export handler
  const handleExport = (format: ExportFormat) => {
    if (format === 'csv') {
      const csv = convertToCSV(data, xAxis, yAxis);
      downloadFile(csv, 'correlation-heatmap.csv', 'text/csv');
    } else if (format === 'json') {
      const json = JSON.stringify({ data, xAxis, yAxis }, null, 2);
      downloadFile(json, 'correlation-heatmap.json', 'application/json');
    }
  };

  return (
    <ChartCard
      title={title}
      description={description}
      chartType="heatmap"
      enableSettings={false}
      enableFullscreen={showFullscreen}
      enableExport={showExport}
      onExport={showExport ? handleExport : undefined}
      height={settings.height || height}
      variant={variant}
      customControls={
        showSettings && (
          <ChartSettingsPanel
            settings={settings}
            onSettingsChange={setSettings}
            chartType="heatmap"
            renderer="echarts"
          />
        )
      }
    >
      <ReactECharts option={option} style={{ height: settings.height || height }} notMerge={true} lazyUpdate={true} />
    </ChartCard>
  );
};

/**
 * Helper: Convert data to CSV
 */
function convertToCSV(data: any[], xAxis: string[], yAxis: string[]): string {
  if (data.length === 0) return '';

  const header = 'X-Axis,Y-Axis,Correlation';
  const rows = data.map((item: any) => {
    const [xIdx, yIdx, value] = item;
    return `"${xAxis[xIdx]}","${yAxis[yIdx]}",${value}`;
  });

  return [header, ...rows].join('\n');
}

/**
 * Helper: Download file
 */
function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export default CorrelationHeatmap;
