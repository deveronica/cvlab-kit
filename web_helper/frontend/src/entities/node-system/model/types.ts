/**
 * Node System Internal Types
 */

import { Node, Edge } from 'reactflow';
import { ComponentCategory } from '@/shared/model/types';

export type NodeMode = 'execute' | 'builder' | 'flow';
export type TabMode = 'execute' | 'builder';

export const SpecialNodeType = {
  IF: 'control_if',
  LOOP: 'control_loop',
  MERGE: 'control_merge',
  CODE: 'code_block',
} as const;

export type SpecialNodeType = typeof SpecialNodeType[keyof typeof SpecialNodeType];

export interface UnifiedPort {
  id: string;
  name: string;
  type: string;
  kind?: 'input' | 'output' | 'data' | 'exec';
  value?: any;
}

export interface UnifiedNodeData {
  id: string;
  role: string;
  category: ComponentCategory;
  mode: NodeMode;
  implementation?: string;
  params?: Record<string, any>;
  inputs: UnifiedPort[];
  outputs: UnifiedPort[];
  executionPins?: UnifiedPort[];
  usedConfigKeys?: string[];
  source?: {
    file: string;
    line: number;
    endLine?: number;
  };
  isUncovered?: boolean;
  canDrill?: boolean;
  nodeType?: string;
  parentId?: string;
  [key: string]: any;
}

export type CodeFlowEdge = Edge & {
  data?: {
    edgeType: 'execution' | 'data' | 'ref';
    flowType?: string;
    label?: string;
    sequenceIndex?: number;
    [key: string]: any;
  };
};

export interface NodeGraph {
  nodes: Node<UnifiedNodeData>[];
  edges: CodeFlowEdge[];
}

export interface CategoryPorts {
  initInputs: any[];
  selfOutput: any;
  executionPins: any[];
  dataInputs: any[];
  dataOutputs: any[];
}
