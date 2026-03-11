import type { Meta, StoryObj } from '@storybook/react';
import { FlowPane } from './FlowPane';

const meta: Meta<typeof FlowPane> = {
  title: 'Execute/FlowPane',
  component: FlowPane,
  args: {
    agentName: null,
    className: 'h-[560px]',
  },
};

export default meta;

type Story = StoryObj<typeof FlowPane>;

export const Default: Story = {};
