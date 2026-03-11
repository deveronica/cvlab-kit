/**
 * Port Registry
 *
 * Default ports for each component category based on L3_TSD/node-system/포트-시스템.md
 */

import { ComponentCategory } from '@/shared/model/types';
import { CategoryPorts } from '../model/port';

/**
 * Default ports for each component category
 */
export const DEFAULT_PORTS: Record<ComponentCategory | 'unknown' | 'global', CategoryPorts> = {
  model: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Model instance' },
    executionPins: [
      { name: 'forward', type: 'execution', position: 'left', description: 'Forward pass' },
      { name: 'eval', type: 'execution', position: 'left', description: 'Evaluation mode' },
      { name: 'train', type: 'execution', position: 'left', description: 'Training mode' },
    ],
    dataInputs: [
      { name: 'x', type: 'tensor', position: 'left', description: 'Input tensor' },
    ],
    dataOutputs: [
      { name: 'parameters', type: 'parameters', position: 'right', description: 'Model parameters' },
      { name: 'out', type: 'tensor', position: 'right', description: 'Model output' },
    ],
  },

  optimizer: {
    initInputs: [
      { name: 'parameters', type: 'parameters', position: 'left', description: 'model.parameters()' },
    ],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Optimizer instance' },
    executionPins: [
      { name: 'step', type: 'execution', position: 'left', description: 'Optimizer step' },
      { name: 'zero_grad', type: 'execution', position: 'left', description: 'Zero gradients' },
    ],
    dataInputs: [
      { name: 'grads', type: 'tensor', position: 'left', description: 'Gradients' },
    ],
    dataOutputs: [],
  },

  loss: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Loss function' },
    executionPins: [
      { name: 'forward', type: 'execution', position: 'left', description: 'Calculate loss' },
    ],
    dataInputs: [
      { name: 'pred', type: 'tensor', position: 'left', description: 'Predictions' },
      { name: 'target', type: 'tensor', position: 'left', description: 'Ground truth' },
    ],
    dataOutputs: [
      { name: 'loss', type: 'scalar', position: 'right', description: 'Loss value' },
    ],
  },

  dataset: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Dataset instance' },
    executionPins: [
      { name: '__getitem__', type: 'execution', position: 'left', description: 'Get item' },
      { name: '__len__', type: 'execution', position: 'left', description: 'Get length' },
    ],
    dataInputs: [],
    dataOutputs: [
      { name: 'sample', type: 'tensor', position: 'right', description: 'Data sample' },
    ],
  },

  dataloader: {
    initInputs: [
      { name: 'dataset', type: 'module', position: 'left', description: 'Dataset instance' },
    ],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'DataLoader instance' },
    executionPins: [
      { name: '__iter__', type: 'execution', position: 'left', description: 'Start iteration' },
      { name: '__next__', type: 'execution', position: 'left', description: 'Next batch' },
    ],
    dataInputs: [],
    dataOutputs: [
      { name: 'batch', type: 'tensor', position: 'right', description: 'Batched data' },
    ],
  },

  transform: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Transform instance' },
    executionPins: [],
    dataInputs: [
      { name: 'x', type: 'tensor', position: 'left', description: 'Input data' },
    ],
    dataOutputs: [
      { name: 'x', type: 'tensor', position: 'right', description: 'Transformed data' },
    ],
  },

  metric: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Metric instance' },
    executionPins: [],
    dataInputs: [
      { name: 'pred', type: 'tensor', position: 'left', description: 'Predictions' },
      { name: 'target', type: 'tensor', position: 'left', description: 'Ground truth' },
    ],
    dataOutputs: [
      { name: 'value', type: 'scalar', position: 'right', description: 'Metric value' },
    ],
  },

  scheduler: {
    initInputs: [
      { name: 'optimizer', type: 'module', position: 'left', description: 'Optimizer instance' },
    ],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Scheduler instance' },
    executionPins: [
      { name: 'step', type: 'execution', position: 'left', description: 'Scheduler step' },
      { name: 'get_lr', type: 'execution', position: 'left', description: 'Get learning rate' },
    ],
    dataInputs: [],
    dataOutputs: [
      { name: 'lr', type: 'scalar', position: 'right', description: 'Learning rate' },
    ],
  },

  sampler: {
    initInputs: [
      { name: 'dataset', type: 'module', position: 'left', description: 'Dataset instance' },
    ],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Sampler instance' },
    executionPins: [],
    dataInputs: [],
    dataOutputs: [
      { name: 'indices', type: 'list', position: 'right', description: 'Sample indices' },
    ],
  },

  checkpoint: {
    initInputs: [
      { name: 'model', type: 'module', position: 'left', description: 'Model to save' },
      { name: 'optimizer', type: 'module', position: 'left', description: 'Optimizer to save' },
    ],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Checkpoint manager' },
    executionPins: [],
    dataInputs: [],
    dataOutputs: [],
  },

  callback: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Callback instance' },
    executionPins: [],
    dataInputs: [],
    dataOutputs: [],
  },

  logger: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Logger instance' },
    executionPins: [],
    dataInputs: [],
    dataOutputs: [],
  },

  agent: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Agent instance' },
    executionPins: [],
    dataInputs: [],
    dataOutputs: [],
  },

  global: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Global instance' },
    executionPins: [],
    dataInputs: [],
    dataOutputs: [],
  },

  unknown: {
    initInputs: [],
    selfOutput: { name: 'self', type: 'module', position: 'right', description: 'Component instance' },
    executionPins: [],
    dataInputs: [],
    dataOutputs: [],
  },
};

/**
 * Get default ports for a category
 */
export function getDefaultPorts(category: ComponentCategory): CategoryPorts {
  return (DEFAULT_PORTS as any)[category] || DEFAULT_PORTS.unknown;
}

/**
 * Check if category has __init__ dependencies
 */
export function hasInitInputs(category: ComponentCategory): boolean {
  const ports = (DEFAULT_PORTS as any)[category] || DEFAULT_PORTS.unknown;
  return ports.initInputs.length > 0;
}

/**
 * Get ports for Execute Tab (config graph)
 * Returns initInputs, selfOutput, and executionPins
 */
export function getExecuteTabPorts(category: ComponentCategory): {
  inputs: CategoryPorts['initInputs'];
  output: CategoryPorts['selfOutput'];
  executionPins: CategoryPorts['executionPins'];
} {
  const ports = getDefaultPorts(category);
  return {
    inputs: ports.initInputs,
    output: ports.selfOutput,
    executionPins: ports.executionPins,
  };
}

/**
 * Get ports for Builder Tab (data flow graph)
 * Returns dataInputs, dataOutputs, and executionPins
 */
export function getBuilderTabPorts(category: ComponentCategory): {
  inputs: CategoryPorts['dataInputs'];
  outputs: CategoryPorts['dataOutputs'];
  executionPins: CategoryPorts['executionPins'];
} {
  const ports = getDefaultPorts(category);
  return {
    inputs: ports.dataInputs,
    outputs: ports.dataOutputs,
    executionPins: ports.executionPins,
  };
}
