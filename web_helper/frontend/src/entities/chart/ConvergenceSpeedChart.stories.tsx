import type { Meta, StoryObj } from '@storybook/react';
import { ConvergenceSpeedChart } from './ConvergenceSpeedChart';
import type { Run, TimeseriesMetric } from '@/shared/model/types';

const makeTimeseries = (start: number, delta: number): TimeseriesMetric[] =>
  Array.from({ length: 12 }, (_, idx) => ({
    step: idx,
    epoch: idx,
    values: {
      accuracy: Math.min(0.95, start + idx * delta),
      loss: Math.max(0.2, 1.1 - idx * delta),
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
      final: { accuracy: 0.86, loss: 0.45 },
      max: { accuracy: 0.88 },
      min: { loss: 0.4 },
      mean: { accuracy: 0.84 },
      timeseries: makeTimeseries(0.6, 0.02),
    },
  },
  {
    run_name: 'Run B',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.9, loss: 0.36 },
      max: { accuracy: 0.92 },
      min: { loss: 0.33 },
      mean: { accuracy: 0.88 },
      timeseries: makeTimeseries(0.58, 0.025),
    },
  },
  {
    run_name: 'Run C',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.82, loss: 0.5 },
      max: { accuracy: 0.84 },
      min: { loss: 0.47 },
      mean: { accuracy: 0.81 },
      timeseries: makeTimeseries(0.55, 0.015),
    },
  },
];

const meta: Meta<typeof ConvergenceSpeedChart> = {
  title: 'Charts/ConvergenceSpeedChart',
  component: ConvergenceSpeedChart,
  args: {
    runs,
    availableMetrics: ['accuracy', 'loss'],
    height: 360,
  },
};

export default meta;

type Story = StoryObj<typeof ConvergenceSpeedChart>;

export const Default: Story = {};
