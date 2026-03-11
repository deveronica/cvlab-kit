import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';
import { ChartSettingsPanel, type ChartSettings } from './ChartSettingsPanel';

const initialSettings: ChartSettings = {
  animation: true,
  height: 360,
  legend: { show: true, position: 'top', align: 'center' },
  xAxis: { showGrid: true, label: 'Step' },
  yAxis: { showGrid: true, label: 'Value', scale: 'linear' },
  strokeWidth: 2,
  showDots: true,
  curveType: 'monotone',
  fillOpacity: 0.3,
  stackMode: 'none',
  dotSize: 4,
  heatmap: { colorPalette: 'blue-red', showValues: true, minCorrelation: -1, maxCorrelation: 1 },
  histogram: { binCount: 20, showMean: true, showMedian: false },
};

const meta: Meta<typeof ChartSettingsPanel> = {
  title: 'Charts/ChartSettingsPanel',
  component: ChartSettingsPanel,
  args: {
    chartType: 'line',
    renderer: 'recharts',
  },
  render: (args) => {
    const [settings, setSettings] = useState<ChartSettings>(initialSettings);
    return (
      <div className="p-6">
        <ChartSettingsPanel
          {...args}
          settings={settings}
          onSettingsChange={setSettings}
        />
      </div>
    );
  },
};

export default meta;

type Story = StoryObj<typeof ChartSettingsPanel>;

export const LineChart: Story = {};

export const Heatmap: Story = {
  args: {
    chartType: 'heatmap',
    renderer: 'echarts',
  },
};

export const Histogram: Story = {
  args: {
    chartType: 'histogram',
    renderer: 'recharts',
  },
};
