import type { Meta, StoryObj } from '@storybook/react';
import { NodeCanvas } from './ui/NodeCanvas';
import { ReactFlowProvider } from 'reactflow';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import { useEffect } from 'react';
import { ComponentCategory, NodeMode, FlowType } from '@/entities/node-system/model/types';

const meta: Meta<typeof NodeCanvas> = {
  title: 'Features/NodeSystem/NodeCanvas',
  component: NodeCanvas,
  decorators: [
    (Story) => (
      <div className="h-[800px] w-full border rounded-xl overflow-hidden bg-background">
        <ReactFlowProvider>
          <Story />
        </ReactFlowProvider>
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof NodeCanvas>;

/**
 * Business Logic: Classification Agent Structure
 * Demonstrates Row-based layout and Comment badges
 */
export const ClassificationAgent: Story = {
  render: () => {
    const setNodes = useNodeStore((s) => s.setNodes);
    const setEdges = useNodeStore((s) => s.setEdges);

    useEffect(() => {
      // Mock nodes based on actual classification.py structure
      const mockNodes = [
        {
          id: 'model@28',
          type: 'unified',
          position: { x: 100, y: 100 },
          data: { 
            id: 'model@28', 
            role: 'model', 
            category: 'model', 
            implementation: 'resnet18', 
            mode: 'builder',
            source: { file: 'classification.py', line: 28 }
          }
        },
        {
          id: 'loss_fn@30',
          type: 'unified',
          position: { x: 400, y: 100 },
          data: { 
            id: 'loss_fn@30', 
            role: 'loss_fn', 
            category: 'loss', 
            implementation: 'cross_entropy', 
            mode: 'builder',
            source: { file: 'classification.py', line: 30 }
          }
        },
        {
          id: 'train_dataset@35',
          type: 'unified',
          position: { x: 100, y: 600 }, // Forced to next row (400px gap)
          data: { 
            id: 'train_dataset@35', 
            role: 'train_dataset', 
            category: 'dataset', 
            implementation: 'CIFAR10', 
            mode: 'builder',
            source: { file: 'classification.py', line: 35 }
          }
        }
      ];

      const mockEdges = [
        {
          id: 'exec_model_loss',
          source: 'model@28',
          target: 'loss_fn@30',
          sourceHandle: 'exec:out',
          targetHandle: 'exec:in',
          type: 'custom',
          data: { 
            edgeType: 'execution',
            is_gapped: false
          }
        },
        {
          id: 'exec_loss_dataset',
          source: 'loss_fn@30',
          target: 'train_dataset@35',
          sourceHandle: 'exec:out',
          targetHandle: 'exec:in',
          type: 'custom',
          data: { 
            edgeType: 'execution',
            is_gapped: true,
            comment: 'Create datasets using the named configurations'
          }
        }
      ];

      setNodes(mockNodes as any);
      setEdges(mockEdges as any);
    }, [setNodes, setEdges]);

    return <NodeCanvas editable={true} />;
  }
};
