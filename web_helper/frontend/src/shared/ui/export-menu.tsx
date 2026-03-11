import React from "react";

/**
 * Export Menu Component
 *
 * Reusable dropdown menu for exporting data in various formats
 * Consistent design pattern with other UI menu components
 */

import { Button } from './button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './dropdown-menu';
import { Download, FileJson, FileText, Image } from 'lucide-react';

export type ExportFormat = 'json' | 'csv' | 'markdown' | 'png' | 'svg';

export interface ExportOption {
  format: ExportFormat;
  label: string;
  icon?: React.ReactNode;
}

export interface ExportMenuProps {
  onExport: (format: ExportFormat) => void;
  formats?: ExportFormat[];
  customOptions?: ExportOption[];
  label?: string;
  showLabel?: boolean;
  size?: 'sm' | 'default';
  variant?: 'outline' | 'ghost' | 'default';
  className?: string;
  disabled?: boolean;
}

const DEFAULT_EXPORT_OPTIONS: Record<ExportFormat, ExportOption> = {
  json: {
    format: 'json',
    label: 'Export as JSON',
    icon: <FileJson className="mr-2 h-4 w-4" />,
  },
  csv: {
    format: 'csv',
    label: 'Export as CSV',
    icon: <FileText className="mr-2 h-4 w-4" />,
  },
  markdown: {
    format: 'markdown',
    label: 'Export as Markdown',
    icon: <FileText className="mr-2 h-4 w-4" />,
  },
  png: {
    format: 'png',
    label: 'Export as PNG',
    icon: <Image className="mr-2 h-4 w-4" />,
  },
  svg: {
    format: 'svg',
    label: 'Export as SVG',
    icon: <Image className="mr-2 h-4 w-4" />,
  },
};

export function ExportMenu({
  onExport,
  formats = ['json', 'csv', 'png'],
  customOptions,
  label = 'Export',
  showLabel = true,
  size = 'sm',
  variant = 'outline',
  className = '',
  disabled = false,
}: ExportMenuProps) {
  // Use custom options if provided, otherwise use default formats
  const exportOptions = customOptions || formats.map(f => DEFAULT_EXPORT_OPTIONS[f]);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant={variant}
          size={size}
          className={className}
          disabled={disabled}
          title={label}
        >
          <Download className="h-4 w-4" />
          {showLabel && <span className="ml-2">{label}</span>}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuLabel>{label}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {exportOptions.map((option) => (
          <DropdownMenuItem
            key={option.format}
            onClick={() => onExport(option.format)}
          >
            {option.icon}
            {option.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
