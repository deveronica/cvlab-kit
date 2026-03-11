import { memo } from 'react';
import { cn } from '@/shared/lib/utils';

interface PinLegendProps {
  className?: string;
}

export const PinLegend = memo(function PinLegend({ className }: PinLegendProps) {
  return (
    <div className={cn('flex items-center gap-2 text-[10px] text-muted-foreground', className)}>
      <span className="inline-flex items-center gap-1">
        <span
          className="h-2.5 w-2.5 border"
          style={{
            borderColor: 'hsl(var(--foreground))',
            backgroundColor: 'hsl(var(--foreground))',
            clipPath: 'polygon(0 50%, 100% 0, 100% 100%)',
          }}
          aria-hidden="true"
        />
        Exec
      </span>
      <span className="inline-flex items-center gap-1">
        <span
          className="h-2.5 w-2.5 rounded-full border"
          style={{ borderColor: '#06b6d4', backgroundColor: '#06b6d4' }}
          aria-hidden="true"
        />
        Data
      </span>
    </div>
  );
});

PinLegend.displayName = 'PinLegend';
