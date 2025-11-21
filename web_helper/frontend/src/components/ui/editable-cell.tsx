import React from "react";

/**
 * Editable cell component for TanStack Table inline editing
 */

import { useState, useEffect, useRef } from 'react';
import { Check, X, Edit2 } from 'lucide-react';
import { cn } from '../../lib/utils';

interface EditableCellProps {
  value: any;
  onSave: (value: any) => void;
  type?: 'text' | 'number' | 'select' | 'boolean';
  options?: string[]; // For select type
  className?: string;
  disabled?: boolean;
  placeholder?: string;
  // Validation
  validator?: (value: any) => boolean | string;
}

export function EditableCell({
  value: initialValue,
  onSave,
  type = 'text',
  options = [],
  className,
  disabled = false,
  placeholder,
  validator,
}: EditableCellProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(initialValue);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | HTMLSelectElement>(null);

  // Reset value when initialValue changes (from external updates)
  useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  // Focus input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      if (inputRef.current instanceof HTMLInputElement) {
        inputRef.current.select();
      }
    }
  }, [isEditing]);

  const handleEdit = () => {
    if (disabled) return;
    setIsEditing(true);
    setError(null);
  };

  const handleSave = () => {
    // Validate value
    if (validator) {
      const validationResult = validator(value);
      if (validationResult !== true) {
        setError(typeof validationResult === 'string' ? validationResult : 'Invalid value');
        return;
      }
    }

    // Save value
    onSave(value);
    setIsEditing(false);
    setError(null);
  };

  const handleCancel = () => {
    setValue(initialValue);
    setIsEditing(false);
    setError(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      handleCancel();
    }
  };

  const formatDisplayValue = (val: any) => {
    if (type === 'boolean') {
      return val ? '✅' : '❌';
    }
    if (val === null || val === undefined) {
      return '-';
    }
    return String(val);
  };

  if (isEditing) {
    return (
      <div className="flex items-center space-x-1 min-w-0">
        {type === 'select' ? (
          <select
            ref={inputRef as React.RefObject<HTMLSelectElement>}
            value={value || ''}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            className={cn(
              'flex h-8 w-full rounded-md border border-input bg-background px-2 py-1 text-sm',
              'focus:outline-none focus:ring-2 focus:ring-ring focus:shadow-md',
              error && 'border-red-500',
              className
            )}
          >
            <option value="">Select...</option>
            {options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        ) : type === 'boolean' ? (
          <select
            ref={inputRef as React.RefObject<HTMLSelectElement>}
            value={value ? 'true' : 'false'}
            onChange={(e) => setValue(e.target.value === 'true')}
            onKeyDown={handleKeyDown}
            className={cn(
              'flex h-8 w-full rounded-md border border-input bg-background px-2 py-1 text-sm',
              'focus:outline-none focus:ring-2 focus:ring-ring focus:shadow-md',
              error && 'border-red-500',
              className
            )}
          >
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
        ) : (
          <input
            ref={inputRef as React.RefObject<HTMLInputElement>}
            type={type}
            value={value || ''}
            onChange={(e) => setValue(type === 'number' ? Number(e.target.value) : e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className={cn(
              'flex h-8 w-full rounded-md border border-input bg-background px-2 py-1 text-sm',
              'focus:outline-none focus:ring-2 focus:ring-ring focus:shadow-md',
              error && 'border-red-500',
              className
            )}
          />
        )}
        <button
          onClick={handleSave}
          className="flex h-6 w-6 items-center justify-center rounded bg-green-100 text-green-600 hover:bg-green-200"
          title="Save"
        >
          <Check className="h-3 w-3" />
        </button>
        <button
          onClick={handleCancel}
          className="flex h-6 w-6 items-center justify-center rounded bg-red-100 text-red-600 hover:bg-red-200"
          title="Cancel"
        >
          <X className="h-3 w-3" />
        </button>
      </div>
    );
  }

  return (
    <div className="group flex items-center space-x-2 min-w-0">
      <span className={cn('flex-1 truncate', disabled && 'text-muted-foreground')}>
        {formatDisplayValue(value)}
      </span>
      {!disabled && (
        <button
          onClick={handleEdit}
          className="flex h-4 w-4 items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          title="Edit"
        >
          <Edit2 className="h-3 w-3 text-muted-foreground hover:text-foreground" />
        </button>
      )}
      {error && (
        <div className="absolute z-10 mt-8 rounded bg-red-100 px-2 py-1 text-xs text-red-600">
          {error}
        </div>
      )}
    </div>
  );
}