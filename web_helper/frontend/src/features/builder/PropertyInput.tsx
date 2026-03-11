/**
 * PropertyInput - Type-aware input component for node properties
 *
 * Supports:
 * - number: Spinner input with increment/decrement
 * - boolean: Toggle switch
 * - string: Text input
 * - select: Dropdown for enum/choices
 * - array: JSON editor or list input
 * - object: JSON editor
 */

import { useState, useCallback, useEffect } from 'react';
import { Check, ChevronDown, Minus, Plus, X } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { Input } from '@/shared/ui/input';
import { Button } from '@/shared/ui/button';
import { Badge } from '@/shared/ui/badge';
import type { ConfigValue } from '@/shared/model/config-types';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/shared/ui/select';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/shared/ui/popover';
import { Textarea } from '@/shared/ui/textarea';

export type PropertyType =
  | 'number'
  | 'integer'
  | 'float'
  | 'boolean'
  | 'string'
  | 'select'
  | 'array'
  | 'object'
  | 'any';

interface PropertyInputProps {
  name: string;
  value: ConfigValue;
  type?: PropertyType;
  choices?: string[] | number[];
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  readOnly?: boolean;
  onChange?: (name: string, value: ConfigValue) => void;
  className?: string;
}

// Infer type from value
function inferType(value: ConfigValue): PropertyType {
  if (value === null || value === undefined) return 'any';
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? 'integer' : 'float';
  }
  if (typeof value === 'string') return 'string';
  if (Array.isArray(value)) return 'array';
  if (typeof value === 'object') return 'object';
  return 'any';
}

// Number Input with spinner
function NumberInput({
  value,
  min,
  max,
  step = 1,
  readOnly,
  onChange,
  className,
}: {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  readOnly?: boolean;
  onChange?: (value: number) => void;
  className?: string;
}) {
  const [localValue, setLocalValue] = useState(String(value));

  useEffect(() => {
    setLocalValue(String(value));
  }, [value]);

  const handleChange = (newValue: number) => {
    let clamped = newValue;
    if (min !== undefined) clamped = Math.max(min, clamped);
    if (max !== undefined) clamped = Math.min(max, clamped);
    setLocalValue(String(clamped));
    onChange?.(clamped);
  };

  const handleBlur = () => {
    const parsed = parseFloat(localValue);
    if (!isNaN(parsed)) {
      handleChange(parsed);
    } else {
      setLocalValue(String(value));
    }
  };

  return (
    <div className={cn('flex items-center gap-0.5', className)}>
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6"
        disabled={readOnly || (min !== undefined && value <= min)}
        onClick={() => handleChange(value - step)}
      >
        <Minus className="h-3 w-3" />
      </Button>
      <Input
        type="text"
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
        onBlur={handleBlur}
        onKeyDown={(e) => {
          if (e.key === 'Enter') handleBlur();
          if (e.key === 'ArrowUp') handleChange(value + step);
          if (e.key === 'ArrowDown') handleChange(value - step);
        }}
        className="h-6 w-16 text-xs text-center px-1"
        readOnly={readOnly}
      />
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6"
        disabled={readOnly || (max !== undefined && value >= max)}
        onClick={() => handleChange(value + step)}
      >
        <Plus className="h-3 w-3" />
      </Button>
    </div>
  );
}

// Boolean Toggle
function BooleanInput({
  value,
  readOnly,
  onChange,
  className,
}: {
  value: boolean;
  readOnly?: boolean;
  onChange?: (value: boolean) => void;
  className?: string;
}) {
  return (
    <Button
      variant={value ? 'default' : 'outline'}
      size="sm"
      className={cn('h-6 px-2 text-xs', className)}
      disabled={readOnly}
      onClick={() => onChange?.(!value)}
    >
      {value ? (
        <>
          <Check className="h-3 w-3 mr-1" />
          True
        </>
      ) : (
        <>
          <X className="h-3 w-3 mr-1" />
          False
        </>
      )}
    </Button>
  );
}

// Select Input
function SelectInput({
  value,
  choices,
  readOnly,
  onChange,
  className,
}: {
  value: ConfigValue;
  choices: (string | number)[];
  readOnly?: boolean;
  onChange?: (value: ConfigValue) => void;
  className?: string;
}) {
  return (
    <Select
      value={String(value)}
      onValueChange={(v) => {
        // Try to preserve type
        const original = choices.find((c) => String(c) === v);
        onChange?.(original ?? v);
      }}
      disabled={readOnly}
    >
      <SelectTrigger className={cn('h-6 text-xs', className)}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {choices.map((choice) => (
          <SelectItem key={String(choice)} value={String(choice)}>
            {String(choice)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

// String Input
function StringInput({
  value,
  placeholder,
  readOnly,
  onChange,
  className,
}: {
  value: string;
  placeholder?: string;
  readOnly?: boolean;
  onChange?: (value: string) => void;
  className?: string;
}) {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  return (
    <Input
      value={localValue}
      onChange={(e) => setLocalValue(e.target.value)}
      onBlur={() => onChange?.(localValue)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') onChange?.(localValue);
      }}
      placeholder={placeholder}
      className={cn('h-6 text-xs', className)}
      readOnly={readOnly}
    />
  );
}

// Array/Object Input (JSON editor in popover)
function JsonInput({
  value,
  readOnly,
  onChange,
  className,
}: {
  value: ConfigValue;
  readOnly?: boolean;
  onChange?: (value: ConfigValue) => void;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [localValue, setLocalValue] = useState(JSON.stringify(value, null, 2));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLocalValue(JSON.stringify(value, null, 2));
  }, [value]);

  const handleSave = () => {
    try {
      const parsed = JSON.parse(localValue);
      setError(null);
      onChange?.(parsed);
      setOpen(false);
    } catch (e) {
      setError('Invalid JSON');
    }
  };

  const displayValue = Array.isArray(value)
    ? `[${value.length} items]`
    : typeof value === 'object'
      ? `{${Object.keys(value).length} keys}`
      : String(value);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn('h-6 px-2 text-xs font-mono justify-between', className)}
          disabled={readOnly}
        >
          <span className="truncate max-w-[80px]">{displayValue}</span>
          <ChevronDown className="h-3 w-3 ml-1 flex-shrink-0" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-2" align="start">
        <div className="space-y-2">
          <Textarea
            value={localValue}
            onChange={(e) => {
              setLocalValue(e.target.value);
              setError(null);
            }}
            className="font-mono text-xs min-h-[120px]"
            readOnly={readOnly}
          />
          {error && (
            <p className="text-xs text-destructive">{error}</p>
          )}
          {!readOnly && (
            <div className="flex justify-end gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setLocalValue(JSON.stringify(value, null, 2));
                  setOpen(false);
                }}
              >
                Cancel
              </Button>
              <Button size="sm" onClick={handleSave}>
                Apply
              </Button>
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}

// Main component
export function PropertyInput({
  name,
  value,
  type: propType,
  choices,
  min,
  max,
  step,
  placeholder,
  readOnly = false,
  onChange,
  className,
}: PropertyInputProps) {
  const type = propType || inferType(value);

  const handleChange = useCallback(
    (newValue: ConfigValue) => {
      onChange?.(name, newValue);
    },
    [name, onChange]
  );

  // Select type if choices provided
  if (choices && choices.length > 0) {
    return (
      <SelectInput
        value={value}
        choices={choices}
        readOnly={readOnly}
        onChange={handleChange}
        className={className}
      />
    );
  }

  // Type-specific inputs
  switch (type) {
    case 'number':
    case 'integer':
    case 'float':
      return (
        <NumberInput
          value={Number(value) || 0}
          min={min}
          max={max}
          step={type === 'integer' ? 1 : step ?? 0.1}
          readOnly={readOnly}
          onChange={handleChange}
          className={className}
        />
      );

    case 'boolean':
      return (
        <BooleanInput
          value={Boolean(value)}
          readOnly={readOnly}
          onChange={handleChange}
          className={className}
        />
      );

    case 'string':
      return (
        <StringInput
          value={String(value ?? '')}
          placeholder={placeholder}
          readOnly={readOnly}
          onChange={handleChange}
          className={className}
        />
      );

    case 'array':
    case 'object':
      return (
        <JsonInput
          value={value}
          readOnly={readOnly}
          onChange={handleChange}
          className={className}
        />
      );

    default:
      // Fallback to string input
      return (
        <StringInput
          value={String(value ?? '')}
          placeholder={placeholder}
          readOnly={readOnly}
          onChange={handleChange}
          className={className}
        />
      );
  }
}

// Type badge helper
export function TypeBadge({ type }: { type: PropertyType }) {
  const colors: Record<PropertyType, string> = {
    number: 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300',
    integer: 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300',
    float: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/50 dark:text-cyan-300',
    boolean: 'bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300',
    string: 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300',
    select: 'bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300',
    array: 'bg-rose-100 text-rose-700 dark:bg-rose-900/50 dark:text-rose-300',
    object: 'bg-slate-100 text-slate-700 dark:bg-slate-900/50 dark:text-slate-300',
    any: 'bg-gray-100 text-gray-700 dark:bg-gray-900/50 dark:text-gray-300',
  };

  return (
    <Badge
      variant="outline"
      className={cn('text-[9px] px-1 py-0 font-medium', colors[type])}
    >
      {type}
    </Badge>
  );
}

// Value source type
export type ValueSource = 'yaml' | 'default' | 'hardcode';

// Value source badge helper
// Shows where parameter value comes from:
// - YAML (blue): Required, must be in YAML config
// - default (yellow): Has code default, can be overridden by YAML
// - hardcode (red): Fixed in code, not configurable
export function ValueSourceBadge({
  source,
  showLabel = false,
}: {
  source: ValueSource;
  showLabel?: boolean;
}) {
  const config: Record<
    ValueSource,
    { color: string; label: string; description: string }
  > = {
    yaml: {
      color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300 border-blue-300',
      label: 'YAML',
      description: 'Required from YAML config',
    },
    default: {
      color: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/50 dark:text-yellow-300 border-yellow-300',
      label: 'default',
      description: 'Has code default, overridable by YAML',
    },
    hardcode: {
      color: 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300 border-red-300',
      label: 'fixed',
      description: 'Fixed in code, not configurable',
    },
  };

  const cfg = config[source];

  return (
    <Badge
      variant="outline"
      className={cn('text-[9px] px-1 py-0 font-medium', cfg.color)}
      title={cfg.description}
    >
      {showLabel ? cfg.label : source === 'yaml' ? '●' : source === 'default' ? '◐' : '○'}
    </Badge>
  );
}

// Value source indicator (small dot for inline use)
export function ValueSourceDot({ source }: { source: ValueSource }) {
  const colors: Record<ValueSource, string> = {
    yaml: 'bg-blue-500',
    default: 'bg-yellow-500',
    hardcode: 'bg-red-500',
  };

  const descriptions: Record<ValueSource, string> = {
    yaml: 'Required from YAML config',
    default: 'Has code default',
    hardcode: 'Fixed in code',
  };

  return (
    <span
      className={cn('inline-block w-2 h-2 rounded-full', colors[source])}
      title={descriptions[source]}
    />
  );
}
