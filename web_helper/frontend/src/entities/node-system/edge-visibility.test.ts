import { describe, it, expect } from 'vitest';
import { isEdgeVisibleInTab, FlowType } from '@/entities/node-system/model/edge';
import { toPortType, PortType } from '@/entities/node-system/model/port';

describe('Edge Visibility', () => {
  it('should show CONTROL edges in Execute tab', () => {
    const edge = {
      id: 'e1',
      source_node: 'n1',
      source_port: 'p1',
      target_node: 'n2',
      target_port: 'p2',
      flow_type: FlowType.CONTROL,
    };
    expect(isEdgeVisibleInTab(edge, 'execute')).toBe(true);
  });

  it('should show CONTROL edges in Builder tab', () => {
    const edge = {
      id: 'e1',
      source_node: 'n1',
      source_port: 'p1',
      target_node: 'n2',
      target_port: 'p2',
      flow_type: FlowType.CONTROL,
    };
    expect(isEdgeVisibleInTab(edge, 'builder')).toBe(true); // flow mode maps to builder tab context
  });

  it('should filter GRADIENT edges', () => {
    const edge = {
      id: 'e1',
      source_node: 'n1',
      source_port: 'p1',
      target_node: 'n2',
      target_port: 'p2',
      flow_type: FlowType.GRADIENT,
    };
    // GRADIENT is in FILTERED_FLOW_TYPES
    expect(isEdgeVisibleInTab(edge, 'execute')).toBe(false);
    expect(isEdgeVisibleInTab(edge, 'builder')).toBe(false);
  });

  it('should show TENSOR edges only in Builder tab', () => {
    const edge = {
      id: 'e1',
      source_node: 'n1',
      source_port: 'p1',
      target_node: 'n2',
      target_port: 'p2',
      flow_type: FlowType.TENSOR,
    };
    expect(isEdgeVisibleInTab(edge, 'execute')).toBe(false);
    expect(isEdgeVisibleInTab(edge, 'builder')).toBe(true);
  });

  it('should show CONFIG edges only in Execute tab', () => {
    const edge = {
      id: 'e1',
      source_node: 'n1',
      source_port: 'p1',
      target_node: 'n2',
      target_port: 'p2',
      flow_type: FlowType.CONFIG,
    };
    expect(isEdgeVisibleInTab(edge, 'execute')).toBe(true);
    expect(isEdgeVisibleInTab(edge, 'builder')).toBe(false);
  });
});

describe('Port Type Conversion', () => {
  it('should convert "execution" to EXECUTION type', () => {
    expect(toPortType('execution')).toBe(PortType.EXECUTION);
  });

  it('should convert "tensor" to TENSOR type', () => {
    expect(toPortType('tensor')).toBe(PortType.TENSOR);
    expect(toPortType('torch.Tensor')).toBe(PortType.TENSOR);
  });

  it('should default to ANY for unknown types', () => {
    expect(toPortType('unknown_type')).toBe(PortType.ANY);
    expect(toPortType(undefined)).toBe(PortType.ANY);
  });
});
