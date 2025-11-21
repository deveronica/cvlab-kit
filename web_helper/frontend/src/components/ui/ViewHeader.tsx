import React from "react";

import { X } from 'lucide-react';
import { Button } from './button';
import { HEADER_HEIGHTS } from '../../lib/constants/sizes';

interface ViewHeaderProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  onClose?: () => void;
  size?: 'compact' | 'standard' | 'tall';
  className?: string;
}

export function ViewHeader({
  title,
  description,
  actions,
  onClose,
  size = 'standard',
  className = '',
}: ViewHeaderProps) {
  const heightClass = HEADER_HEIGHTS[size];

  return (
    <div
      className={`flex items-center justify-between px-6 border-b bg-background ${heightClass} ${className}`}
    >
      <div className="flex-1 min-w-0">
        <h2 className="text-lg font-semibold truncate">{title}</h2>
        {description && (
          <p className="text-sm text-muted-foreground truncate mt-0.5">
            {description}
          </p>
        )}
      </div>

      <div className="flex items-center gap-2 ml-4 flex-shrink-0">
        {actions}
        {onClose && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close"
            className="h-8 w-8"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
}
