import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { TabSwitcher } from './ui/TabSwitcher';

describe('TabSwitcher', () => {
  it('calls onTabChange when clicking tabs', () => {
    const onTabChange = vi.fn();
    render(<TabSwitcher currentTab="execute" onTabChange={onTabChange} />);

    fireEvent.click(screen.getByText('Flow'));
    expect(onTabChange).toHaveBeenCalledWith('builder');

    fireEvent.click(screen.getByText('Config'));
    expect(onTabChange).toHaveBeenCalledWith('execute');
  });
});
