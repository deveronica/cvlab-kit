import type { Meta, StoryObj } from '@storybook/react';
import { ReactFlowProvider } from 'reactflow';
import { UnifiedNode } from './UnifiedNode';
import type { UnifiedNodeData } from '@/entities/node-system/model/types';
import { PortType } from '@/entities/node-system/model/port';
import { mockNodeBaseData as baseData } from '@/shared/lib/mocks/nodes';

const meta: Meta<typeof UnifiedNode> = {
  title: 'Node System/UnifiedNode',
  component: UnifiedNode,
  render: (args) => (
    <ReactFlowProvider>
      <UnifiedNode {...args} />
    </ReactFlowProvider>
  ),
  args: {
    id: baseData.id,
    data: baseData,
    selected: false,
    type: 'unified',
    zIndex: 0,
    isConnectable: true,
    dragging: false,
    xPos: 0,
    yPos: 0,
  },
};

export default meta;

type Story = StoryObj<typeof UnifiedNode>;

export const Interactive: Story = {
  args: {
    data: {
      ...baseData,
      onImplementationChange: () => {},
      onParamChange: () => {},
    },
  },
};

export const CodeFooterClick: Story = {
  args: {
    data: {
      ...baseData,
      onCodeClick: () => {},
    },
  },
};

export const BuilderMode: Story = {
  args: {
    data: {
      ...baseData,
      mode: 'builder',
      params: undefined,
      initInputs: undefined,
      selfOutput: undefined,
    },
  },
};

export const FlowMode: Story = {
  args: {
    data: {
      ...baseData,
      mode: 'flow',
      params: undefined,
      initInputs: undefined,
      selfOutput: undefined,
      implementation: undefined,
      availableImplementations: undefined,
    },
  },
};
