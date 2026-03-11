import type { Meta, StoryObj } from '@storybook/react';
import CorrelationHeatmap from './CorrelationHeatmap';

const xAxis = ['lr', 'batch_size', 'weight_decay'];
const yAxis = ['accuracy', 'loss', 'f1'];

const data: Array<[number, number, number]> = [
  [0, 0, 0.42],
  [1, 0, -0.18],
  [2, 0, 0.31],
  [0, 1, -0.65],
  [1, 1, 0.22],
  [2, 1, -0.12],
  [0, 2, 0.38],
  [1, 2, 0.05],
  [2, 2, 0.47],
];

const meta: Meta<typeof CorrelationHeatmap> = {
  title: 'Charts/CorrelationHeatmap',
  component: CorrelationHeatmap,
  args: {
    data,
    xAxis,
    yAxis,
    title: 'Hyperparameter Correlations',
    description: 'Sample correlation values',
    height: 420,
  },
};

export default meta;

type Story = StoryObj<typeof CorrelationHeatmap>;

export const Default: Story = {};

export const Compact: Story = {
  args: {
    variant: 'compact',
    height: 320,
  },
};
