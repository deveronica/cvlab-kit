import type { Meta, StoryObj } from '@storybook/react';
import { ExecuteFlowPane } from './ExecuteFlowPane';

const meta: Meta<typeof ExecuteFlowPane> = {
  title: 'Node System/ExecuteFlowPane',
  component: ExecuteFlowPane,
  args: {
    agentName: null,
    className: 'h-[560px]',
  },
};

export default meta;

type Story = StoryObj<typeof ExecuteFlowPane>;

export const NoSelection: Story = {};
