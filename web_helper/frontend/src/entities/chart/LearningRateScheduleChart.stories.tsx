import type { Meta, StoryObj } from '@storybook/react';
import { LearningRateScheduleChart } from './LearningRateScheduleChart';
import type { Run, TimeseriesMetric } from '@/shared/model/types';

const makeTimeseries = (baseLr: number): TimeseriesMetric[] =>
  Array.from({ length: 12 }, (_, idx) => ({
    step: idx,
    epoch: idx,
    values: {
      lr: baseLr * Math.pow(0.9, idx),
      loss: 1.1 / (idx + 1),
      accuracy: 0.6 + idx * 0.02,
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
      final: { accuracy: 0.82, loss: 0.45 },
      max: { accuracy: 0.86 },
      min: { loss: 0.42 },
      mean: { accuracy: 0.83 },
      timeseries: makeTimeseries(0.01),
    },
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
      timeseries: makeTimeseries(0.008),
    },
  },
];

const meta: Meta<typeof LearningRateScheduleChart> = {
  title: 'Charts/LearningRateScheduleChart',
  component: LearningRateScheduleChart,
  args: {
    runs,
    height: 320,
  },
};

export default meta;

type Story = StoryObj<typeof LearningRateScheduleChart>;

export const Default: Story = {};

export const Compact: Story = {
  args: {
    variant: 'compact',
    height: 260,
  },
};
