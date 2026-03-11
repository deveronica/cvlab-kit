import { describe, it, expect, vi } from 'vitest';
import { ReactFlowProvider } from 'reactflow';
import { screen, render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { UnifiedNode } from './ui/UnifiedNode';
import type { UnifiedNodeData } from '@/entities/node-system/model/types';
import { PortType } from '@/entities/node-system/model/port';
import { TooltipProvider } from '@/shared/ui/tooltip';

const renderNode = (data: UnifiedNodeData) =>
  render(
    <ReactFlowProvider>
      <TooltipProvider>
        <UnifiedNode
          id={data.id}
          data={data}
          selected={false}
          type="unified"
          zIndex={0}
          isConnectable={true}
          dragging={false}
          xPos={0}
          yPos={0}
        />
      </TooltipProvider>
    </ReactFlowProvider>
  );

const baseData: UnifiedNodeData = {
  id: 'model',
  role: 'model',
  category: 'model',
  mode: 'execute',
  implementation: 'resnet18',
  availableImplementations: [
    { name: 'resnet18' },
    { name: 'vgg16' },
  ],
  inputs: [],
  outputs: [],
  initInputs: [],
  selfOutput: { id: 'self', name: 'self', type: PortType.ANY },
  params: [
    { name: 'lr', type: 'number', value: 0.1, defaultValue: 0.1 },
    { name: 'use_bias', type: 'boolean', value: false, defaultValue: false },
    { name: 'activation', type: 'select', value: 'relu', defaultValue: 'relu', options: ['relu', 'gelu'] },
    { name: 'note', type: 'string', value: 'alpha' },
  ],
  source: { file: 'cvlabkit/agent/classification.py', line: 42 },
};

describe('UnifiedNode', () => {
  it('fires implementation and param change handlers', async () => {
    const user = userEvent.setup();
    const onImplementationChange = vi.fn();
    const onParamChange = vi.fn();

    renderNode({
      ...baseData,
      onImplementationChange,
      onParamChange,
    });

    await user.click(screen.getByRole('button', { name: 'resnet18' }));
    await user.click(screen.getByRole('menuitem', { name: 'vgg16' }));
    expect(onImplementationChange).toHaveBeenCalledWith('model', 'vgg16');

    const lrInput = screen.getByDisplayValue('0.1');
    await user.clear(lrInput);
    await user.type(lrInput, '0.2');
    await user.tab();
    expect(onParamChange).toHaveBeenCalledWith('model', 'lr', 0.2);

    await user.click(screen.getByRole('button', { name: 'false' }));
    expect(onParamChange).toHaveBeenCalledWith('model', 'use_bias', true);

    await user.click(screen.getByRole('button', { name: 'relu' }));
    await user.click(screen.getByRole('menuitem', { name: 'gelu' }));
    expect(onParamChange).toHaveBeenCalledWith('model', 'activation', 'gelu');
  });

  it('expands params and handles code footer click', async () => {
    const user = userEvent.setup();
    const onCodeClick = vi.fn();

    renderNode({
      ...baseData,
      onCodeClick,
    });

    await user.click(screen.getByRole('button', { name: '+1 more' }));
    expect(screen.getByText('note:')).toBeTruthy();

    await user.click(screen.getByText('classification.py:42'));
    expect(onCodeClick).toHaveBeenCalledWith({
      file: 'cvlabkit/agent/classification.py',
      line: 42,
    });
  });
});
