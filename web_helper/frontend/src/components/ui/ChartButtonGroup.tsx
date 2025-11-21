import React from "react";
import { Button } from './button';

interface ChartButton {
  label: string;
  value: string;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
}

interface ChartButtonGroupProps {
  buttons: ChartButton[];
  className?: string;
}

export function ChartButtonGroup({ buttons, className = '' }: ChartButtonGroupProps) {
  return (
    <div className={`flex gap-2 ${className}`}>
      {buttons.map((btn) => (
        <Button
          key={btn.value}
          size="sm"
          variant={btn.active ? 'default' : 'outline'}
          onClick={btn.onClick}
          disabled={btn.disabled}
          className="h-8 px-3 text-sm transition-colors duration-200"
        >
          {btn.label}
        </Button>
      ))}
    </div>
  );
}
