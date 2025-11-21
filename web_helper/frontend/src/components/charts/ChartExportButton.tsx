import React from "react";
/**
 * Chart Export Button
 *
 * UI component for exporting charts to various formats (PNG, SVG, CSV, JSON)
 */

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Download, FileImage, FileCode, FileSpreadsheet, FileJson } from 'lucide-react';
import type { ChartExportOptions } from '@/lib/charts/types';

interface ChartExportButtonProps {
  /** Callback to trigger export */
  onExport: (options: ChartExportOptions) => Promise<void>;
  /** Disabled state */
  disabled?: boolean;
  /** Size variant */
  size?: 'sm' | 'default' | 'icon';
  /** Variant */
  variant?: 'default' | 'outline' | 'ghost';
  /** className */
  className?: string;
}

export function ChartExportButton({
  onExport,
  disabled = false,
  size = 'default',
  variant = 'outline',
  className,
}: ChartExportButtonProps) {
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async (format: ChartExportOptions['format']) => {
    setIsExporting(true);
    try {
      await onExport({
        format,
        filename: `chart_${Date.now()}`,
        quality: 0.95,
        scale: 2,
      });
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant={variant}
          size={size === 'sm' ? 'icon' : size}
          disabled={disabled || isExporting}
          className={size === 'sm' ? 'h-8 w-8' : className}
          title={isExporting ? 'Exporting...' : 'Export chart'}
        >
          <Download className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>Export Format</DropdownMenuLabel>
        <DropdownMenuSeparator />

        <DropdownMenuItem onClick={() => handleExport('png')}>
          <FileImage className="h-4 w-4 mr-2" />
          PNG Image
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => handleExport('svg')}>
          <FileCode className="h-4 w-4 mr-2" />
          SVG Vector
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        <DropdownMenuItem onClick={() => handleExport('csv')}>
          <FileSpreadsheet className="h-4 w-4 mr-2" />
          CSV Data
        </DropdownMenuItem>

        <DropdownMenuItem onClick={() => handleExport('json')}>
          <FileJson className="h-4 w-4 mr-2" />
          JSON Data
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
