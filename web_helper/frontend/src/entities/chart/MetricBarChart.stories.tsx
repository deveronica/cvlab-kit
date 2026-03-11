import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { MetricBarChart } from './MetricBarChart';
import type { Run } from '@/shared/model/types';

const runs: Run[] = [
  {
    run_name: 'Run A',
    project: 'demo',
    status: 'completed',
    started_at: null,
    finished_at: null,
    metrics: {
      final: { accuracy: 0.88, loss: 0.32, f1: 0.84 },
      max: { accuracy: 0.9 },
      min: { loss: 0.3 },
      mean: { accuracy: 0.86 },
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
      final: { accuracy: 0.91, loss: 0.27, f1: 0.88 },
      max: { accuracy: 0.93 },
      min: { loss: 0.25 },
      mean: { accuracy: 0.89 },
      timeseries: [],
    },
  },
];

const meta: Meta<typeof MetricBarChart> = {
  title: 'Charts/MetricBarChart',
  component: MetricBarChart,
  args: {
    runs,
    metricKeys: ['accuracy', 'loss', 'f1'],
    title: 'Run Metric Comparison',
    height: 360,
  },
};

export default meta;

type Story = StoryObj<typeof MetricBarChart>;

export const Default: Story = {};
