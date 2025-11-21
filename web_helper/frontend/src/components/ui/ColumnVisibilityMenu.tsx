import React from "react";

import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuCheckboxItem, DropdownMenuLabel, DropdownMenuSeparator } from '../ui/dropdown-menu';
import { Button } from '../ui/button';
import { Columns } from 'lucide-react';

interface ColumnVisibilityMenuProps {
  hyperparamColumns: string[];
  metricColumns: string[];
  visibleHyperparams: Set<string>;
  visibleMetrics: Set<string>;
  toggleHyperparamVisibility: (param: string) => void;
  toggleMetricVisibility: (metric: string) => void;
}

export const ColumnVisibilityMenu = React.forwardRef<HTMLButtonElement, ColumnVisibilityMenuProps>(({ hyperparamColumns, metricColumns, visibleHyperparams, visibleMetrics, toggleHyperparamVisibility, toggleMetricVisibility }, ref) => {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button ref={ref} variant="outline" size="sm" className="flex items-center gap-2">
          <Columns className="h-4 w-4" />
          Columns
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64">
        <DropdownMenuLabel>Toggle Columns</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Hyperparameters</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {hyperparamColumns.map(param => (
          <DropdownMenuCheckboxItem
            key={param}
            checked={visibleHyperparams.has(param)}
            onCheckedChange={() => toggleHyperparamVisibility(param)}
          >
            {param}
          </DropdownMenuCheckboxItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Metrics</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {metricColumns.map(metric => (
          <DropdownMenuCheckboxItem
            key={metric}
            checked={visibleMetrics.has(metric)}
            onCheckedChange={() => toggleMetricVisibility(metric)}
          >
            {metric}
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
});
