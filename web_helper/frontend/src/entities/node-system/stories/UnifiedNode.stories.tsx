import type { Meta, StoryObj } from '@storybook/react';
import { UnifiedNode } from './ui/UnifiedNode';
import { PortType, ComponentCategory, NodeMode } from '@/entities/node-system/model/types';
import { ReactFlowProvider } from 'reactflow';

const meta: Meta<typeof UnifiedNode> = {
  title: 'Features/NodeSystem/UnifiedNode',
  component: UnifiedNode,
  decorators: [
    (Story) => (
      <div className="h-[400px] w-[400px] flex items-center justify-center bg-muted/10 p-10">
        <ReactFlowProvider>
          <Story />
        </ReactFlowProvider>
      </div>
    ),
  ],
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof UnifiedNode>;

/**
 * Actual data from classification.py model node
 */
export const ModelNode: Story = {
  args: {
    id: 'model@28',
    selected: false,
    data: {
      id: 'model@28',
      role: 'model',
      category: 'model', // Literal string
      implementation: 'resnet18',
      mode: 'builder', // Literal string
      inputs: [],
      outputs: [
        { id: 'data:parameters', name: 'parameters', type: 'parameters' as any },
        { id: 'data:out', name: 'out', type: 'tensor' as any },
        { id: 'data:self', name: 'self', type: 'module' as any },
      ],
      executionPins: [
        { id: 'exec:in', name: 'in', type: 'execution' as any },
        { id: 'exec:out', name: 'out', type: 'execution' as any },
      ],
      propertySummary: {
        config_count: 3,
        hardcode_count: 0,
        required_count: 0,
        default_count: 5,
      },
      source: { file: 'classification.py', line: 28 },
    },
  },
};

/**
 * Node with comments from setup() gapped edge
 */
export const LossNode: Story = {
  args: {
    id: 'loss_fn@30',
    data: {
      id: 'loss_fn@30',
      role: 'loss_fn',
      category: 'loss', // Literal string
      implementation: 'cross_entropy',
      mode: 'builder', // Literal string
      inputs: [
        { id: 'data:pred', name: 'pred', type: 'tensor' as any },
        { id: 'data:target', name: 'target', type: 'tensor' as any },
      ],
      outputs: [{ id: 'data:loss', name: 'loss', type: 'tensor' as any }],
      executionPins: [
        { id: 'exec:in', name: 'in', type: 'execution' as any },
        { id: 'exec:out', name: 'out', type: 'execution' as any },
      ],
      source: { file: 'classification.py', line: 30 },
    },
  },
};

export const UncoveredNode: Story = {
  args: {
    ...LossNode.args,
    data: {
      ...LossNode.args!.data!,
      isUncovered: true,
    },
  },
};
