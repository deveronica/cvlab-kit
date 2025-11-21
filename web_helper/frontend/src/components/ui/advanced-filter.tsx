import React from "react";
/**
 * Advanced Filter Panel
 *
 * Provides advanced filtering controls for experiments table
 */

import { useState } from 'react';
import { Card, CardContent } from './card';
import { Button } from './button';
import { Input } from './input';
import { Badge } from './badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './select';
import { Filter, X, Plus, ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '../../lib/utils';

export interface FilterRule {
  id: string;
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'lt' | 'gte' | 'lte' | 'contains' | 'in';
  value: string | number;
}

interface AdvancedFilterProps {
  fields: Array<{ value: string; label: string; type: 'string' | 'number' }>;
  filters: FilterRule[];
  onFiltersChange: (filters: FilterRule[]) => void;
  className?: string;
}

const operators = {
  string: [
    { value: 'eq', label: 'equals' },
    { value: 'ne', label: 'not equals' },
    { value: 'contains', label: 'contains' },
  ],
  number: [
    { value: 'eq', label: '=' },
    { value: 'ne', label: '≠' },
    { value: 'gt', label: '>' },
    { value: 'lt', label: '<' },
    { value: 'gte', label: '≥' },
    { value: 'lte', label: '≤' },
  ],
};

export function AdvancedFilter({
  fields,
  filters,
  onFiltersChange,
  className,
}: AdvancedFilterProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const addFilter = () => {
    const newFilter: FilterRule = {
      id: `filter-${Date.now()}`,
      field: fields[0]?.value || '',
      operator: 'eq',
      value: '',
    };
    onFiltersChange([...filters, newFilter]);
    setIsExpanded(true);
  };

  const removeFilter = (id: string) => {
    onFiltersChange(filters.filter(f => f.id !== id));
  };

  const updateFilter = (id: string, updates: Partial<FilterRule>) => {
    onFiltersChange(
      filters.map(f => (f.id === id ? { ...f, ...updates } : f))
    );
  };

  const clearAll = () => {
    onFiltersChange([]);
  };

  const getFieldType = (fieldValue: string): 'string' | 'number' => {
    return fields.find(f => f.value === fieldValue)?.type || 'string';
  };

  return (
    <div className={cn('space-y-2', className)}>
      {/* Filter Summary Bar */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 flex-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="gap-2"
          >
            <Filter className="h-4 w-4" />
            Filters
            {filters.length > 0 && (
              <Badge variant="secondary" className="ml-1">
                {filters.length}
              </Badge>
            )}
            {isExpanded ? (
              <ChevronUp className="h-3 w-3" />
            ) : (
              <ChevronDown className="h-3 w-3" />
            )}
          </Button>

          {/* Active filter badges */}
          <div className="flex items-center gap-1 flex-wrap">
            {filters.slice(0, 3).map(filter => {
              const field = fields.find(f => f.value === filter.field);
              return (
                <Badge
                  key={filter.id}
                  variant="outline"
                  className="gap-1 pr-1"
                >
                  <span className="text-xs">
                    {field?.label}: {filter.operator} {filter.value}
                  </span>
                  <button
                    onClick={() => removeFilter(filter.id)}
                    className="hover:bg-muted rounded-sm p-0.5"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              );
            })}
            {filters.length > 3 && (
              <Badge variant="secondary" className="text-xs">
                +{filters.length - 3} more
              </Badge>
            )}
          </div>
        </div>

        {filters.length > 0 && (
          <Button variant="ghost" size="sm" onClick={clearAll}>
            Clear all
          </Button>
        )}
      </div>

      {/* Filter Panel */}
      {isExpanded && (
        <Card>
          <CardContent className="pt-4 space-y-3">
            {filters.length === 0 ? (
              <div className="text-center text-sm text-muted-foreground py-4">
                No filters applied. Click "Add Filter" to get started.
              </div>
            ) : (
              filters.map(filter => {
                const fieldType = getFieldType(filter.field);
                const availableOperators = operators[fieldType];

                return (
                  <div
                    key={filter.id}
                    className="flex items-center gap-2 p-2 rounded-md border bg-muted/30"
                  >
                    {/* Field Select */}
                    <Select
                      value={filter.field}
                      onValueChange={value =>
                        updateFilter(filter.id, { field: value })
                      }
                    >
                      <SelectTrigger className="w-[180px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {fields.map(field => (
                          <SelectItem key={field.value} value={field.value}>
                            {field.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    {/* Operator Select */}
                    <Select
                      value={filter.operator}
                      onValueChange={value =>
                        updateFilter(filter.id, {
                          operator: value as FilterRule['operator'],
                        })
                      }
                    >
                      <SelectTrigger className="w-[120px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {availableOperators.map(op => (
                          <SelectItem key={op.value} value={op.value}>
                            {op.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    {/* Value Input */}
                    <Input
                      type={fieldType === 'number' ? 'number' : 'text'}
                      value={filter.value}
                      onChange={e =>
                        updateFilter(filter.id, {
                          value:
                            fieldType === 'number'
                              ? parseFloat(e.target.value) || 0
                              : e.target.value,
                        })
                      }
                      placeholder="Value..."
                      className="flex-1"
                    />

                    {/* Remove Button */}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFilter(filter.id)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                );
              })
            )}

            {/* Add Filter Button */}
            <Button
              variant="outline"
              size="sm"
              onClick={addFilter}
              className="w-full gap-2"
            >
              <Plus className="h-4 w-4" />
              Add Filter
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

/**
 * Apply filters to data array
 */
export function applyFilters<T extends Record<string, any>>(
  data: T[],
  filters: FilterRule[]
): T[] {
  if (filters.length === 0) return data;

  return data.filter(item => {
    return filters.every(filter => {
      const value = item[filter.field];

      switch (filter.operator) {
        case 'eq':
          return value == filter.value;
        case 'ne':
          return value != filter.value;
        case 'gt':
          return typeof value === 'number' && value > Number(filter.value);
        case 'lt':
          return typeof value === 'number' && value < Number(filter.value);
        case 'gte':
          return typeof value === 'number' && value >= Number(filter.value);
        case 'lte':
          return typeof value === 'number' && value <= Number(filter.value);
        case 'contains':
          return (
            typeof value === 'string' &&
            value.toLowerCase().includes(String(filter.value).toLowerCase())
          );
        default:
          return true;
      }
    });
  });
}
