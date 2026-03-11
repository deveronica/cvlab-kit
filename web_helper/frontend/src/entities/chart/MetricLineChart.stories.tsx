import type { Meta, StoryObj } from '@storybook/react';
import { MetricLineChart } from './MetricLineChart';
import { ChartThemeProvider } from '@/app/ui';
import type { Run, TimeseriesMetric } from '@/shared/model/types';

const makeTimeseries = (offset: number): TimeseriesMetric[] =>
  Array.from({ length: 20 }, (_, idx) => ({
    step: idx,
    epoch: idx,
    values: {
      accuracy: Math.min(0.95, 0.6 + idx * 0.015 + offset),
      loss: Math.max(0.2, 1.0 - idx * 0.03 - offset),
    },
  }));

const runs: Run[] = [
  {
    run_name: 'Run A',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.88, loss: 0.32 },
      max: { accuracy: 0.9 },
      min: { loss: 0.3 },
      mean: { accuracy: 0.86 },
      timeseries: makeTimeseries(0.0),
    },
  },
  {
    run_name: 'Run B',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.9, loss: 0.28 },
      max: { accuracy: 0.92 },
      min: { loss: 0.26 },
      mean: { accuracy: 0.88 },
      timeseries: makeTimeseries(0.03),
    },
  },
];

const meta: Meta<typeof MetricLineChart> = {
  title: 'Charts/MetricLineChart',
  component: MetricLineChart,
  args: {
    runs,
    metricKey: 'accuracy',
    title: 'Accuracy Over Time',
    height: 360,
  },
  decorators: [
    (Story) => (
      <ChartThemeProvider>
        <Story />
      </ChartThemeProvider>
    ),
  ],
};

export default meta;

type Story = StoryObj<typeof MetricLineChart>;

export const Default: Story = {};
