import type { Meta, StoryObj } from '@storybook/react';
import { isEdgeVisibleInTab, FlowType } from '@/entities/node-system/model/edge';
import { toPortType, PortType } from '@/entities/node-system/model/port';

const EdgeVisibilityPanel = () => {
  const edgeBase = {
    id: 'e1',
    source_node: 'n1',
    source_port: 'p1',
    target_node: 'n2',
    target_port: 'p2',
  };

  const cases = [
    { label: 'CONTROL execute', edge: { ...edgeBase, flow_type: FlowType.CONTROL }, tab: 'execute' },
    { label: 'CONTROL builder', edge: { ...edgeBase, flow_type: FlowType.CONTROL }, tab: 'builder' },
    { label: 'GRADIENT execute', edge: { ...edgeBase, flow_type: FlowType.GRADIENT }, tab: 'execute' },
    { label: 'GRADIENT builder', edge: { ...edgeBase, flow_type: FlowType.GRADIENT }, tab: 'builder' },
    { label: 'TENSOR execute', edge: { ...edgeBase, flow_type: FlowType.TENSOR }, tab: 'execute' },
    { label: 'TENSOR builder', edge: { ...edgeBase, flow_type: FlowType.TENSOR }, tab: 'builder' },
    { label: 'CONFIG execute', edge: { ...edgeBase, flow_type: FlowType.CONFIG }, tab: 'execute' },
    { label: 'CONFIG builder', edge: { ...edgeBase, flow_type: FlowType.CONFIG }, tab: 'builder' },
  ];

  const portCases = [
    { label: 'execution', value: toPortType('execution') },
    { label: 'tensor', value: toPortType('tensor') },
    { label: 'torch.Tensor', value: toPortType('torch.Tensor') },
    { label: 'unknown', value: toPortType('unknown_type') },
    { label: 'undefined', value: toPortType(undefined) },
  ];

  return (
    <div className="p-4 space-y-6">
      <div>
        <div className="text-sm font-semibold">Edge Visibility</div>
        <div className="mt-2 space-y-1 text-sm">
          {cases.map((item) => (
            <div key={item.label} className="flex items-center gap-2">
              <span className="w-44">{item.label}</span>
              <span>{String(isEdgeVisibleInTab(item.edge, item.tab as any))}</span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <div className="text-sm font-semibold">Port Type Conversion</div>
        <div className="mt-2 space-y-1 text-sm">
          {portCases.map((item) => (
            <div key={item.label} className="flex items-center gap-2">
              <span className="w-44">{item.label}</span>
              <span>{String(item.value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const meta: Meta<typeof EdgeVisibilityPanel> = {
  title: 'Node System/Edge Visibility',
  component: EdgeVisibilityPanel,
};

export default meta;

type Story = StoryObj<typeof EdgeVisibilityPanel>;

export const Default: Story = {};
