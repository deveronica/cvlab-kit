import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { ScatterPlotChart } from './scatter-plot-chart';

const data = [
  { run_name: 'Run A', lr: 0.01, accuracy: 0.88, loss: 0.32 },
  { run_name: 'Run B', lr: 0.005, accuracy: 0.91, loss: 0.27 },
  { run_name: 'Run C', lr: 0.02, accuracy: 0.83, loss: 0.41 },
  { run_name: 'Run D', lr: 0.008, accuracy: 0.9, loss: 0.29 },
];

const meta: Meta<typeof ScatterPlotChart> = {
  title: 'Charts/scatter-plot-chart',
  component: ScatterPlotChart,
  args: {
    data,
    xOptions: ['lr', 'loss'],
    yOptions: ['accuracy', 'loss'],
    defaultX: 'lr',
    defaultY: 'accuracy',
    title: 'Hyperparameter Scatter',
  },
};

export default meta;

type Story = StoryObj<typeof ScatterPlotChart>;

export const Default: Story = {};

export const Compact: Story = {
  args: {
    variant: 'compact',
  },
};
