/**
 * Port System Types
 */

/**
 * Port data types based on PyTorch conventions
 */
export const PortType = {
  // Data types
  TENSOR: 'tensor',
  MODULE: 'module',
  PARAMETERS: 'parameters',
  SCALAR: 'scalar',
  BOOL: 'bool',
  STRING: 'string',
  LIST: 'list',
  DICT: 'dict',
  CONFIG: 'config',
  DEVICE: 'device',
  EXECUTION: 'execution',
  CONTROL_IN: 'control_in',
  CONTROL_OUT: 'control_out',
  ANY: 'any',
} as const;

export type PortType = typeof PortType[keyof typeof PortType];

export type PortPosition = 'left' | 'right' | 'top' | 'bottom';
export type PortZone = 'initInputs' | 'selfOutput' | 'executionPins' | 'dataInputs' | 'dataOutputs';

export interface UnifiedPort {
  id: string;
  name: string;
  type: PortType;
  description?: string;
  optional?: boolean;
}

export interface PortDefinition {
  name: string;
  type: PortType;
  position: PortPosition;
  description?: string;
  optional?: boolean;
}

export interface CategoryPorts {
  initInputs: PortDefinition[];
  selfOutput: PortDefinition;
  executionPins: PortDefinition[];
  dataInputs: PortDefinition[];
  dataOutputs: PortDefinition[];
}

export function toPortType(type?: string): PortType {
  if (!type) return PortType.ANY;
  const normalizedType = type.toLowerCase();
  const typeMap: Record<string, PortType> = {
    'tensor': PortType.TENSOR,
    'torch.tensor': PortType.TENSOR,
    'module': PortType.MODULE,
    'nn.module': PortType.MODULE,
    'params': PortType.PARAMETERS,
    'parameters': PortType.PARAMETERS,
    'scalar': PortType.SCALAR,
    'float': PortType.SCALAR,
    'int': PortType.SCALAR,
    'number': PortType.SCALAR,
    'bool': PortType.BOOL,
    'boolean': PortType.BOOL,
    'string': PortType.STRING,
    'str': PortType.STRING,
    'list': PortType.LIST,
    'array': PortType.LIST,
    'dict': PortType.DICT,
    'dictionary': PortType.DICT,
    'config': PortType.CONFIG,
    'device': PortType.DEVICE,
    'execution': PortType.EXECUTION,
    'control_in': PortType.CONTROL_IN,
    'control_out': PortType.CONTROL_OUT,
  };
  return typeMap[normalizedType] || PortType.ANY;
}

export function createPort(
  name: string,
  type?: string | PortType,
  description?: string
): UnifiedPort {
  return {
    id: name,
    name,
    type: typeof type === 'string' ? toPortType(type) : (type ?? PortType.ANY),
    description,
  };
}
