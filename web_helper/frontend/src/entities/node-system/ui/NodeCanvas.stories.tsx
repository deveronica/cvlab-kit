import { useEffect } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { ReactFlowProvider, type Edge, type Node } from 'reactflow';
import { NodeCanvas } from './NodeCanvas';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import type { CodeFlowEdge, NodeMode, TabMode, UnifiedNodeData } from '@/shared/model/types';
import { PortType } from '@/entities/node-system/model/port';

const edges: Edge[] = [
  {
    id: 'e-model-loss',
    source: 'model',
    target: 'loss',
    sourceHandle: 'features',
    targetHandle: 'pred',
    type: 'custom',
    data: { flowType: 'reference' },
  },
];

const codeFlowEdges: CodeFlowEdge[] = [
  {
    id: 'e-model-loss',
    source_node: 'model',
    source_port: 'features',
    target_node: 'loss',
    target_port: 'pred',
    flow_type: 'reference',
  },
];

function makeNodes(mode: NodeMode): Node<UnifiedNodeData>[] {
  return [
    {
      id: 'model',
      type: 'unified',
      position: { x: 120, y: 120 },
      data: {
        id: 'model',
        role: 'model',
        category: 'model',
        mode,
        implementation: 'resnet18',
        availableImplementations: [{ name: 'resnet18' }, { name: 'efficientnet_b0' }],
        inputs: [],
        outputs: [{ id: 'features', name: 'features', type: PortType.TENSOR }],
        initInputs:
          mode === 'execute' ? [{ id: 'num_classes', name: 'num_classes', type: PortType.SCALAR }] : undefined,
        selfOutput:
          mode === 'execute' ? { id: 'self', name: 'self', type: PortType.MODULE } : undefined,
        params:
          mode === 'execute'
            ? [{ name: 'num_classes', type: 'number', value: 10, defaultValue: 10 }]
            : undefined,
      },
    },
    {
      id: 'loss',
      type: 'unified',
      position: { x: 420, y: 120 },
      data: {
        id: 'loss',
        role: 'loss',
        category: 'loss',
        mode,
        implementation: 'cross_entropy',
        availableImplementations: [{ name: 'cross_entropy' }],
        inputs: [{ id: 'pred', name: 'pred', type: PortType.TENSOR }],
        outputs: [{ id: 'value', name: 'value', type: PortType.SCALAR }],
      },
    },
  ];
}

function SeededCanvas({ tab, mode }: { tab: TabMode; mode: NodeMode }) {
  useEffect(() => {
    useNodeStore.setState({
      currentTab: tab,
      nodeMode: mode,
      nodes: makeNodes(mode),
      edges,
      codeFlowEdges,
      currentSubsystemId: null,
      selectedNodeId: null,
      breadcrumb: [{ level: 'root', label: 'Agent', path: '' }],
    });

    return () => {
      useNodeStore.setState({
        nodes: [],
        edges: [],
        codeFlowEdges: [],
        selectedNodeId: null,
        currentSubsystemId: null,
        breadcrumb: [{ level: 'root', label: 'Agent', path: '' }],
      });
    };
  }, [mode, tab]);

  return (
    <div className="h-[520px] w-full">
      <NodeCanvas
        className="h-full"
        editable={false}
        showTabSwitcher={false}
        showBreadcrumb={false}
        showMinimap={true}
        showControls={true}
      />
    </div>
  );
}

const meta: Meta<typeof NodeCanvas> = {
  title: 'Node System/NodeCanvas',
  component: NodeCanvas,
  render: () => (
    <ReactFlowProvider>
      <SeededCanvas tab="builder" mode="builder" />
    </ReactFlowProvider>
  ),
};

export default meta;

type Story = StoryObj<typeof NodeCanvas>;

export const BuilderGraph: Story = {
  render: () => (
    <ReactFlowProvider>
      <SeededCanvas tab="builder" mode="builder" />
    </ReactFlowProvider>
  ),
};

export const ExecuteGraph: Story = {
  render: () => (
    <ReactFlowProvider>
      <SeededCanvas tab="execute" mode="execute" />
    </ReactFlowProvider>
  ),
};
