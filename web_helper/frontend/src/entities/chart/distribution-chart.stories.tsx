import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { DistributionChart } from './distribution-chart';

const data = [
  { run_name: 'Run A', accuracy: 0.88, loss: 0.32 },
  { run_name: 'Run B', accuracy: 0.91, loss: 0.27 },
  { run_name: 'Run C', accuracy: 0.83, loss: 0.41 },
  { run_name: 'Run D', accuracy: 0.9, loss: 0.29 },
  { run_name: 'Run E', accuracy: 0.86, loss: 0.35 },
];

const meta: Meta<typeof DistributionChart> = {
  title: 'Charts/distribution-chart',
  component: DistributionChart,
  args: {
    data,
    field: 'accuracy',
    title: 'Accuracy Distribution',
    bins: 8,
  },
};

export default meta;

type Story = StoryObj<typeof DistributionChart>;

export const Default: Story = {};
