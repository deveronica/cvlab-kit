import type { Meta, StoryObj } from '@storybook/react';
import { ScatterMatrixChart } from './ScatterMatrixChart';
import type { Run } from '@/shared/model/types';

const runs: Run[] = [
  {
    run_name: 'Run A',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.82, loss: 0.45, f1: 0.78 },
      max: { accuracy: 0.86 },
      min: { loss: 0.42 },
      mean: { accuracy: 0.83 },
      timeseries: [],
    },
  },
  {
    run_name: 'Run B',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.86, loss: 0.38, f1: 0.81 },
      max: { accuracy: 0.9 },
      min: { loss: 0.35 },
      mean: { accuracy: 0.85 },
      timeseries: [],
    },
  },
  {
    run_name: 'Run C',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.78, loss: 0.52, f1: 0.73 },
      max: { accuracy: 0.8 },
      min: { loss: 0.49 },
      mean: { accuracy: 0.79 },
      timeseries: [],
    },
  },
  {
    run_name: 'Run D',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.88, loss: 0.33, f1: 0.85 },
      max: { accuracy: 0.9 },
      min: { loss: 0.31 },
      mean: { accuracy: 0.87 },
      timeseries: [],
    },
  },
];

const meta: Meta<typeof ScatterMatrixChart> = {
  title: 'Charts/ScatterMatrixChart',
  component: ScatterMatrixChart,
  args: {
    runs,
    metricKeys: ['accuracy', 'loss', 'f1'],
    title: 'Metric Relationships',
    height: 520,
  },
};

export default meta;

type Story = StoryObj<typeof ScatterMatrixChart>;

export const Default: Story = {};

export const Compact: Story = {
  args: {
    variant: 'compact',
    height: 420,
  },
};
