import React from "react";
import type { Meta, StoryObj } from '@storybook/react';
import { MetricChart } from './metric-chart';

const data = [
  { time: '00:00', value: 0.62 },
  { time: '00:10', value: 0.68 },
  { time: '00:20', value: 0.73 },
  { time: '00:30', value: 0.79 },
  { time: '00:40', value: 0.84 },
  { time: '00:50', value: 0.88 },
];

const meta: Meta<typeof MetricChart> = {
  title: 'Charts/metric-chart',
  component: MetricChart,
  args: {
    data,
    title: 'Validation Accuracy',
    color: '#3b82f6',
    height: 220,
  },
};

export default meta;

type Story = StoryObj<typeof MetricChart>;

export const Default: Story = {};
