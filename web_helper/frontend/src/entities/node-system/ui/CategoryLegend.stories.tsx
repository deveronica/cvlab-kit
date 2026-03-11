import type { Meta, StoryObj } from '@storybook/react';
import { CategoryLegend } from './CategoryLegend';

const meta: Meta<typeof CategoryLegend> = {
  title: 'Node System/CategoryLegend',
  component: CategoryLegend,
  args: {
    items: [
      { key: 'model', label: 'Model' },
      { key: 'loss', label: 'Loss' },
      { key: 'optimizer', label: 'Optimizer' },
      { key: 'dataset', label: 'Dataset' },
    ],
  },
};

export default meta;

type Story = StoryObj<typeof CategoryLegend>;

export const Abbreviated: Story = {
  args: {
    abbreviated: true,
  },
};

export const FullLabels: Story = {
  args: {
    abbreviated: false,
  },
};
