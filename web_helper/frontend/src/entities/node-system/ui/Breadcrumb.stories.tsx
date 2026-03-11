import type { Meta, StoryObj } from '@storybook/react';
import { Breadcrumb } from './Breadcrumb';

const meta: Meta<typeof Breadcrumb> = {
  title: 'Node System/Breadcrumb',
  component: Breadcrumb,
  args: {
    items: [
      { level: 'root', label: 'Agent', path: '' },
      { level: 'method', label: 'train_step()', path: 'train_step' },
      { level: 'subsystem', label: 'model', path: 'train_step/model', nodeId: 'model' },
    ],
    onNavigate: () => {},
  },
};

export default meta;

type Story = StoryObj<typeof Breadcrumb>;

export const Default: Story = {};
