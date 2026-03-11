import type { Meta, StoryObj } from '@storybook/react';
import { BuilderEditPane } from './BuilderEditPane';
import { AgentBuilderProvider } from '@/entities/node-system/model/AgentBuilderContext';

const meta: Meta<typeof BuilderEditPane> = {
  title: 'Node System/BuilderEditPane',
  component: BuilderEditPane,
  decorators: [
    (Story) => (
      <AgentBuilderProvider>
        <div className="h-[560px]">
          <Story />
        </div>
      </AgentBuilderProvider>
    ),
  ],
  args: {
    phase: 'initialize',
  },
};

export default meta;

type Story = StoryObj<typeof BuilderEditPane>;

export const Default: Story = {};

export const FlowPhase: Story = {
  args: {
    phase: 'flow',
    method: 'train_step',
  },
};
