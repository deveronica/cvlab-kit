import type { Meta, StoryObj } from '@storybook/react';
import ParallelCoordinatesChart from './ParallelCoordinatesChart';

const dimensions = [
  { name: 'lr' },
  { name: 'batch_size' },
  { name: 'accuracy' },
  { name: 'loss' },
];

const data = [
  [0.01, 32, 0.82, 0.45],
  [0.005, 64, 0.86, 0.38],
  [0.02, 16, 0.78, 0.52],
  [0.008, 32, 0.88, 0.33],
];

const runNames = ['Run A', 'Run B', 'Run C', 'Run D'];

const meta: Meta<typeof ParallelCoordinatesChart> = {
  title: 'Charts/ParallelCoordinatesChart',
  component: ParallelCoordinatesChart,
  args: {
    data,
    dimensions,
    hyperparamCount: 2,
    runNames,
  },
};

export default meta;

type Story = StoryObj<typeof ParallelCoordinatesChart>;

export const Default: Story = {};
