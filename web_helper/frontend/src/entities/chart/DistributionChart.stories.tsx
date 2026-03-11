import type { Meta, StoryObj } from '@storybook/react';
import { DistributionChart } from './DistributionChart';
import type { Run } from '@/shared/model/types';

const runs: Run[] = [
  {
    run_name: 'Run A',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.82, loss: 0.45 },
      max: { accuracy: 0.87 },
      min: { loss: 0.4 },
      mean: { accuracy: 0.83 },
      timeseries: [],
    },
    hyperparameters: { lr: 0.01, batch_size: 32 },
  },
  {
    run_name: 'Run B',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.86, loss: 0.38 },
      max: { accuracy: 0.9 },
      min: { loss: 0.35 },
      mean: { accuracy: 0.85 },
      timeseries: [],
    },
    hyperparameters: { lr: 0.005, batch_size: 64 },
  },
  {
    run_name: 'Run C',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.78, loss: 0.52 },
      max: { accuracy: 0.8 },
      min: { loss: 0.49 },
      mean: { accuracy: 0.79 },
      timeseries: [],
    },
    hyperparameters: { lr: 0.02, batch_size: 16 },
  },
  {
    run_name: 'Run D',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.88, loss: 0.33 },
      max: { accuracy: 0.9 },
      min: { loss: 0.32 },
      mean: { accuracy: 0.87 },
      timeseries: [],
    },
    hyperparameters: { lr: 0.008, batch_size: 32 },
  },
];

const meta: Meta<typeof DistributionChart> = {
  title: 'Charts/DistributionChart',
  component: DistributionChart,
  args: {
    runs,
    metricKey: 'accuracy',
    title: 'Accuracy Distribution',
    description: 'Distribution across sample runs',
    height: 360,
    showBoxPlot: true,
  },
};

export default meta;

type Story = StoryObj<typeof DistributionChart>;

export const Default: Story = {};

export const Compact: Story = {
  args: {
    variant: 'compact',
    height: 280,
  },
};
