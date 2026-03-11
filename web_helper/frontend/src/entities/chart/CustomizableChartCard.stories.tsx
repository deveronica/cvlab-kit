import type { Meta, StoryObj } from '@storybook/react';
import { CustomizableChartCard } from './CustomizableChartCard';
import type { SeriesConfig } from '@/shared/lib/charts/types';

const data = [
  { name: 'Run 1', accuracy: 0.82, loss: 0.45 },
  { name: 'Run 2', accuracy: 0.86, loss: 0.39 },
  { name: 'Run 3', accuracy: 0.79, loss: 0.52 },
  { name: 'Run 4', accuracy: 0.88, loss: 0.34 },
];

const series: SeriesConfig[] = [
  { dataKey: 'accuracy', name: 'Accuracy', color: '#3b82f6', visible: true },
  { dataKey: 'loss', name: 'Loss', color: '#ef4444', visible: true, type: 'line' },
];

const meta: Meta<typeof CustomizableChartCard> = {
  title: 'Charts/CustomizableChartCard',
  component: CustomizableChartCard,
  args: {
    title: 'Metrics Overview',
    description: 'Compare accuracy and loss across runs',
    data,
    initialSeries: series,
    chartType: 'bar',
    supportedChartTypes: ['line', 'bar', 'area', 'scatter'],
    initialRenderer: 'recharts',
    xAxisKey: 'name',
  },
};

export default meta;

type Story = StoryObj<typeof CustomizableChartCard>;

export const Default: Story = {};

export const Compact: Story = {
  args: {
    compact: true,
  },
};
