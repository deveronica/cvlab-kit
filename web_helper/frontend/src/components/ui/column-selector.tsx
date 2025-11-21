import React from "react";

import { useState } from 'react';
import { Button } from './button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './card';
import { Badge } from './badge';

export interface ColumnSelectorProps {
  availableColumns: string[];
  selectedColumns: string[];
  onSelectionChange: (selected: string[]) => void;
  onClose?: () => void;
}

export function ColumnSelector({
  availableColumns,
  selectedColumns,
  onSelectionChange,
  onClose
}: ColumnSelectorProps) {
  const [localSelection, setLocalSelection] = useState<Set<string>>(
    new Set(selectedColumns)
  );

  const handleToggle = (column: string) => {
    const newSelection = new Set(localSelection);
    if (newSelection.has(column)) {
      newSelection.delete(column);
    } else {
      newSelection.add(column);
    }
    setLocalSelection(newSelection);
  };

  const handleApply = () => {
    onSelectionChange(Array.from(localSelection));
    onClose?.();
  };

  const handleReset = () => {
    setLocalSelection(new Set(availableColumns));
  };

  const handleSelectNone = () => {
    setLocalSelection(new Set());
  };

  const categorizedColumns = React.useMemo(() => {
    const hyperparams: string[] = [];
    const metrics: string[] = [];
    const other: string[] = [];

    availableColumns.forEach(column => {
      if (column.includes('loss') || column.includes('acc') || column.includes('error') ||
          column.includes('f1') || column.includes('precision') || column.includes('recall')) {
        metrics.push(column);
      } else if (column.includes('lr') || column.includes('batch') || column.includes('epoch') ||
                 column.includes('model') || column.includes('optim')) {
        hyperparams.push(column);
      } else {
        other.push(column);
      }
    });

    return { hyperparams, metrics, other };
  }, [availableColumns]);

  const renderColumnGroup = (title: string, columns: string[], color: string) => {
    if (columns.length === 0) return null;

    return (
      <div className="space-y-2">
        <h4 className="font-medium text-sm text-muted-foreground">{title}</h4>
        <div className="flex flex-wrap gap-1">
          {columns.map(column => (
            <Badge
              key={column}
              variant={localSelection.has(column) ? 'default' : 'secondary'}
              className={`cursor-pointer hover:opacity-80 ${color}`}
              onClick={() => handleToggle(column)}
            >
              {column}
            </Badge>
          ))}
        </div>
      </div>
    );
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Column Selector</CardTitle>
        <CardDescription>
          Choose which hyperparameters and metrics to display in the table
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Quick actions */}
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleReset}>
            Select All
          </Button>
          <Button variant="outline" size="sm" onClick={handleSelectNone}>
            Select None
          </Button>
          <div className="flex-1" />
          <Badge variant="outline">
            {localSelection.size} of {availableColumns.length} selected
          </Badge>
        </div>

        {/* Column categories */}
        <div className="space-y-4">
          {renderColumnGroup(
            'Hyperparameters',
            categorizedColumns.hyperparams,
            'bg-blue-500/10 text-blue-700 hover:bg-blue-500/20'
          )}
          {renderColumnGroup(
            'Metrics',
            categorizedColumns.metrics,
            'bg-green-500/10 text-green-700 hover:bg-green-500/20'
          )}
          {renderColumnGroup(
            'Other',
            categorizedColumns.other,
            'bg-gray-500/10 text-gray-700 hover:bg-gray-500/20'
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 pt-4 border-t">
          <Button onClick={handleApply}>
            Apply Selection ({localSelection.size})
          </Button>
          {onClose && (
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}