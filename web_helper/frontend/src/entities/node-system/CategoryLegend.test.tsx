import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CategoryLegend } from './ui/CategoryLegend';
import { TooltipProvider } from '@/shared/ui/tooltip';

describe('CategoryLegend', () => {
  it('renders abbreviated labels', () => {
    render(
      <TooltipProvider>
        <CategoryLegend
          items={[
            { key: 'model', label: 'Model' },
            { key: 'optimizer', label: 'Optimizer' },
          ]}
          abbreviated={true}
        />
      </TooltipProvider>
    );

    expect(screen.getByText('Mod')).toBeInTheDocument();
    expect(screen.getByText('Opt')).toBeInTheDocument();
  });
});
