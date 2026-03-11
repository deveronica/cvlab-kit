import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { MetricAreaChart } from './MetricAreaChart';
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
      final: { accuracy: 0.91, loss: 0.27 },
      max: { accuracy: 0.93 },
      min: { loss: 0.25 },
      mean: { accuracy: 0.89 },
      timeseries: makeTimeseries(0.03),
    },
  },
];

const meta: Meta<typeof MetricAreaChart> = {
  title: 'Charts/MetricAreaChart',
  component: MetricAreaChart,
  args: {
    runs,
    metricKey: 'accuracy',
    title: 'Accuracy Progress',
    height: 360,
    stacked: false,
  },
};

export default meta;

type Story = StoryObj<typeof MetricAreaChart>;

export const Default: Story = {};

export const Stacked: Story = {
  args: {
    stacked: true,
  },
};
