import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { TrainingValidationComparisonChart } from './TrainingValidationComparisonChart';
import type { Run, TimeseriesMetric } from '@/shared/model/types';

const makeTimeseries = (offset: number): TimeseriesMetric[] =>
  Array.from({ length: 20 }, (_, idx) => ({
    step: idx,
    epoch: idx,
    values: {
      train_accuracy: Math.min(0.98, 0.65 + idx * 0.015 + offset),
      val_accuracy: Math.min(0.94, 0.62 + idx * 0.012 + offset),
      train_loss: Math.max(0.2, 1.0 - idx * 0.03 - offset),
      val_loss: Math.max(0.24, 1.05 - idx * 0.028 - offset),
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
      timeseries: makeTimeseries(0.02),
    },
  },
];

const meta: Meta<typeof TrainingValidationComparisonChart> = {
  title: 'Charts/TrainingValidationComparisonChart',
  component: TrainingValidationComparisonChart,
  args: {
    runs,
    availableMetrics: ['accuracy', 'loss'],
    height: 380,
  },
};

export default meta;

type Story = StoryObj<typeof TrainingValidationComparisonChart>;

export const Default: Story = {};
