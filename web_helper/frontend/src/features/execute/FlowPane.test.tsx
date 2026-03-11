import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FlowPane } from './FlowPane';

vi.mock('@/entities/node-system', () => ({
  ExecuteFlowPane: ({ agentName }: { agentName?: string | null }) => (
    <div data-testid="execute-flow-pane">{agentName ?? 'none'}</div>
  ),
}));

describe('FlowPane', () => {
  it('renders ExecuteFlowPane wrapper', () => {
    render(<FlowPane agentName="classification" />);
    expect(screen.getByTestId('execute-flow-pane')).toHaveTextContent('classification');
  });
});
