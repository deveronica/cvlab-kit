import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { Breadcrumb } from './ui/Breadcrumb';

describe('Breadcrumb', () => {
  it('renders and navigates to parent item', () => {
    const onNavigate = vi.fn();
    render(
      <Breadcrumb
        items={[
          { level: 'root', label: 'Agent', path: '' },
          { level: 'method', label: 'train_step()', path: 'train_step' },
          { level: 'subsystem', label: 'model', path: 'train_step/model', nodeId: 'model' },
        ]}
        onNavigate={onNavigate}
      />
    );

    fireEvent.click(screen.getByText('Agent'));
    expect(onNavigate).toHaveBeenCalledWith(0);
  });
});
