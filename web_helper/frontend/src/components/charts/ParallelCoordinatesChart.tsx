import React from "react";

import { useMemo, memo } from 'react';
import ReactECharts from 'echarts-for-react';

interface ParallelCoordinatesChartProps {
  data: any[];
  dimensions: { name: string; key?: string }[];
  hyperparamCount?: number; // Number of hyperparameter dimensions (rest are metrics)
  runNames?: string[]; // Corresponding run names for each data point
  onRunSelect?: (runName: string) => void; // Callback when a run is clicked
}

const ParallelCoordinatesChart: React.FC<ParallelCoordinatesChartProps> = memo(({
  data,
  dimensions,
  hyperparamCount = 0,
  runNames = [],
  onRunSelect,
}) => {
  const option = useMemo(() => {
    // Determine which dimensions are metrics (to highlight them)
    const isMetric = (index: number) => index >= hyperparamCount;

    return {
      tooltip: {
        trigger: 'item',
        triggerOn: 'mousemove|click',
        // Add debounce to prevent continuous updates on hover
        showDelay: 200,
        hideDelay: 150,
        transitionDuration: 0.15,
        enterable: true,
        formatter: (params: any) => {
          if (!params.data) return '';

          // Show only key metrics (last 4 dimensions) for concise view
          const values = params.data;
          const runName = runNames[params.dataIndex] || `Run ${params.dataIndex + 1}`;
          const startIdx = Math.max(0, dimensions.length - 4);
          const lines = dimensions.slice(startIdx).map((dim, i) => {
            const actualIdx = startIdx + i;
            const value = values[actualIdx];
            const formattedValue = typeof value === 'number' ? value.toFixed(3) : value;
            return `<span style="color: #6b7280; font-size: 11px;">${dim.name}:</span> <strong style="font-size: 12px;">${formattedValue}</strong>`;
          });

          return `
            <div style="padding: 6px; font-size: 11px;">
              <div style="font-weight: bold; margin-bottom: 4px; color: #1f2937; font-size: 12px;">${runName}</div>
              <div style="font-weight: bold; margin-bottom: 4px; color: #1f2937; font-size: 12px;">Key Metrics</div>
              ${lines.join('<br/>')}
              <div style="margin-top: 4px; color: #9ca3af; font-size: 10px;">Click to view details</div>
            </div>
          `;
        },
        backgroundColor: 'rgba(255, 255, 255, 0.96)',
        borderColor: '#e5e7eb',
        borderWidth: 1,
        textStyle: {
          color: '#374151',
          fontSize: 11,
        },
        extraCssText: 'box-shadow: 0 2px 4px rgb(0 0 0 / 0.1); border-radius: 4px;',
      },
      parallelAxis: dimensions.map((dim, i) => ({
        dim: i,
        name: dim.name,
        nameLocation: 'end',
        nameGap: 20,
        nameTextStyle: {
          fontSize: 12,
          fontWeight: isMetric(i) ? 'bold' : 'normal',
          color: isMetric(i) ? '#10b981' : '#374151', // Green for metrics, dark gray for hyperparams
        },
        axisLabel: {
          fontSize: 11,
          color: '#1f2937', // Dark color for axis values
          fontWeight: 500,
        },
        axisLine: {
          lineStyle: {
            color: isMetric(i) ? '#10b981' : '#9ca3af',
            width: isMetric(i) ? 2 : 1,
          },
        },
        axisTick: {
          show: true,
          lineStyle: {
            color: isMetric(i) ? '#10b981' : '#9ca3af',
          },
        },
        splitLine: {
          show: true,
          lineStyle: {
            color: '#e5e7eb',
            type: 'dashed',
          },
        },
      })),
      parallel: {
        left: 80,
        right: 80,
        bottom: 50,
        top: 80,
        parallelAxisDefault: {
          type: 'value',
          nameRotate: 90,
        },
      },
      series: {
        type: 'parallel',
        lineStyle: {
          width: 2,
          opacity: 0.6,
        },
        emphasis: {
          lineStyle: {
            width: 4,
            opacity: 1,
          },
        },
        data: data,
      },
    };
  }, [data, dimensions, hyperparamCount, runNames]);

  const onEvents = useMemo(() => ({
    click: (params: any) => {
      if (onRunSelect && runNames[params.dataIndex]) {
        onRunSelect(runNames[params.dataIndex]);
      }
    },
  }), [onRunSelect, runNames]);

  return <ReactECharts option={option} style={{ height: 450 }} notMerge={true} lazyUpdate={true} onEvents={onEvents} />;
}, (prevProps, nextProps) => {
  // Custom comparison function for memo
  // Only re-render if data or dimensions actually changed (deep comparison)
  return (
    JSON.stringify(prevProps.data) === JSON.stringify(nextProps.data) &&
    JSON.stringify(prevProps.dimensions) === JSON.stringify(nextProps.dimensions) &&
    prevProps.hyperparamCount === nextProps.hyperparamCount &&
    JSON.stringify(prevProps.runNames) === JSON.stringify(nextProps.runNames) &&
    prevProps.onRunSelect === nextProps.onRunSelect
  );
});

ParallelCoordinatesChart.displayName = 'ParallelCoordinatesChart';

export default ParallelCoordinatesChart;
