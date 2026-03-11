/**
 * MethodSelectorDialog - Select method to call on a component
 *
 * When a component is dropped on the canvas, this dialog shows
 * available methods. Selecting a method generates appropriate
 * input/output ports based on method signature.
 *
 * Similar to Unity Visual Scripting / Unreal Blueprints pattern.
 */

import React, { useState, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/shared/ui/dialog';
import { Button } from '@/shared/ui/button';
import { Input } from '@/shared/ui/input';
import { Badge } from '@/shared/ui/badge';
import { ScrollArea } from '@/shared/ui/scroll-area';
import { Search, ArrowRight, Box, Zap, X } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { ComponentCategory, PortType } from '@/entities/node-system/model/port';

// Port definition for method signature
export interface PortDefinition {
  id: string;
  name: string;
  type: PortType;
  description?: string;
}

// Method definition
export interface MethodDefinition {
  name: string;
  displayName: string;
  description: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  isCommon?: boolean; // Frequently used method
}

// Category-specific methods
const CATEGORY_METHODS: Record<ComponentCategory, MethodDefinition[]> = {
  model: [
    {
      name: 'forward',
      displayName: 'Forward',
      description: 'Forward pass through the model',
      inputs: [
        { id: 'x', name: 'x', type: PortType.TENSOR, description: 'Input tensor' },
      ],
      outputs: [
        { id: 'output', name: 'output', type: PortType.TENSOR, description: 'Output tensor' },
      ],
      isCommon: true,
    },
    {
      name: '__call__',
      displayName: 'Call',
      description: 'Call the model (same as forward)',
      inputs: [
        { id: 'x', name: 'x', type: PortType.TENSOR, description: 'Input tensor' },
      ],
      outputs: [
        { id: 'output', name: 'output', type: PortType.TENSOR, description: 'Output tensor' },
      ],
      isCommon: true,
    },
    {
      name: 'parameters',
      displayName: 'Parameters',
      description: 'Get model parameters (for optimizer)',
      inputs: [],
      outputs: [
        { id: 'params', name: 'parameters', type: PortType.ANY, description: 'Model parameters iterator' },
      ],
    },
    {
      name: 'train',
      displayName: 'Train Mode',
      description: 'Set model to training mode',
      inputs: [],
      outputs: [
        { id: 'self', name: 'self', type: PortType.MODULE, description: 'Model in train mode' },
      ],
    },
    {
      name: 'eval',
      displayName: 'Eval Mode',
      description: 'Set model to evaluation mode',
      inputs: [],
      outputs: [
        { id: 'self', name: 'self', type: PortType.MODULE, description: 'Model in eval mode' },
      ],
    },
  ],
  optimizer: [
    {
      name: 'step',
      displayName: 'Step',
      description: 'Perform optimization step',
      inputs: [],
      outputs: [],
      isCommon: true,
    },
    {
      name: 'zero_grad',
      displayName: 'Zero Grad',
      description: 'Clear gradients',
      inputs: [],
      outputs: [],
      isCommon: true,
    },
  ],
  loss: [
    {
      name: 'forward',
      displayName: 'Forward',
      description: 'Compute loss',
      inputs: [
        { id: 'pred', name: 'prediction', type: PortType.TENSOR, description: 'Model prediction' },
        { id: 'target', name: 'target', type: PortType.TENSOR, description: 'Ground truth' },
      ],
      outputs: [
        { id: 'loss', name: 'loss', type: PortType.SCALAR, description: 'Computed loss value' },
      ],
      isCommon: true,
    },
    {
      name: '__call__',
      displayName: 'Call',
      description: 'Compute loss (same as forward)',
      inputs: [
        { id: 'pred', name: 'prediction', type: PortType.TENSOR, description: 'Model prediction' },
        { id: 'target', name: 'target', type: PortType.TENSOR, description: 'Ground truth' },
      ],
      outputs: [
        { id: 'loss', name: 'loss', type: PortType.SCALAR, description: 'Computed loss value' },
      ],
      isCommon: true,
    },
  ],
  dataset: [
    {
      name: '__getitem__',
      displayName: 'Get Item',
      description: 'Get a single sample by index',
      inputs: [
        { id: 'idx', name: 'index', type: PortType.SCALAR, description: 'Sample index' },
      ],
      outputs: [
        { id: 'sample', name: 'sample', type: PortType.DICT, description: 'Data sample (dict or tuple)' },
      ],
      isCommon: true,
    },
    {
      name: '__len__',
      displayName: 'Length',
      description: 'Get dataset size',
      inputs: [],
      outputs: [
        { id: 'length', name: 'length', type: PortType.SCALAR, description: 'Number of samples' },
      ],
    },
  ],
  dataloader: [
    {
      name: '__iter__',
      displayName: 'Iterate',
      description: 'Iterate over batches',
      inputs: [],
      outputs: [
        { id: 'batch', name: 'batch', type: PortType.DICT, description: 'Batch data' },
      ],
      isCommon: true,
    },
  ],
  transform: [
    {
      name: '__call__',
      displayName: 'Transform',
      description: 'Apply transformation',
      inputs: [
        { id: 'input', name: 'input', type: PortType.TENSOR, description: 'Input data' },
      ],
      outputs: [
        { id: 'output', name: 'output', type: PortType.TENSOR, description: 'Transformed data' },
      ],
      isCommon: true,
    },
  ],
  metric: [
    {
      name: 'update',
      displayName: 'Update',
      description: 'Update metric with new values',
      inputs: [
        { id: 'pred', name: 'prediction', type: PortType.TENSOR, description: 'Model prediction' },
        { id: 'target', name: 'target', type: PortType.TENSOR, description: 'Ground truth' },
      ],
      outputs: [],
      isCommon: true,
    },
    {
      name: 'compute',
      displayName: 'Compute',
      description: 'Compute final metric value',
      inputs: [],
      outputs: [
        { id: 'value', name: 'value', type: PortType.SCALAR, description: 'Metric value' },
      ],
      isCommon: true,
    },
    {
      name: 'reset',
      displayName: 'Reset',
      description: 'Reset metric state',
      inputs: [],
      outputs: [],
    },
  ],
  scheduler: [
    {
      name: 'step',
      displayName: 'Step',
      description: 'Update learning rate',
      inputs: [],
      outputs: [],
      isCommon: true,
    },
    {
      name: 'get_last_lr',
      displayName: 'Get LR',
      description: 'Get current learning rate',
      inputs: [],
      outputs: [
        { id: 'lr', name: 'lr', type: PortType.SCALAR, description: 'Current learning rate' },
      ],
    },
  ],
  sampler: [
    {
      name: '__iter__',
      displayName: 'Iterate',
      description: 'Iterate over indices',
      inputs: [],
      outputs: [
        { id: 'indices', name: 'indices', type: PortType.LIST, description: 'Sample indices' },
      ],
      isCommon: true,
    },
  ],
  checkpoint: [
    {
      name: 'save',
      displayName: 'Save',
      description: 'Save checkpoint',
      inputs: [
        { id: 'state', name: 'state', type: PortType.DICT, description: 'State to save' },
      ],
      outputs: [],
      isCommon: true,
    },
    {
      name: 'load',
      displayName: 'Load',
      description: 'Load checkpoint',
      inputs: [],
      outputs: [
        { id: 'state', name: 'state', type: PortType.DICT, description: 'Loaded state' },
      ],
      isCommon: true,
    },
  ],
  callback: [],
  logger: [],
  agent: [],
  global: [],
  unknown: [
    {
      name: '__call__',
      displayName: 'Call',
      description: 'Call the component',
      inputs: [
        { id: 'input', name: 'input', type: PortType.ANY, description: 'Input' },
      ],
      outputs: [
        { id: 'output', name: 'output', type: PortType.ANY, description: 'Output' },
      ],
      isCommon: true,
    },
  ],
};

interface MethodSelectorDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  componentName: string;
  componentCategory: ComponentCategory;
  onSelectMethod: (method: MethodDefinition) => void;
  onSkip: () => void; // Add without method selection (Setup mode)
}

export function MethodSelectorDialog({
  open,
  onOpenChange,
  componentName,
  componentCategory,
  onSelectMethod,
  onSkip,
}: MethodSelectorDialogProps) {
  const [search, setSearch] = useState('');

  // Get methods for this category
  const methods = useMemo(() => {
    return CATEGORY_METHODS[componentCategory] || CATEGORY_METHODS.unknown;
  }, [componentCategory]);

  // Filter methods by search
  const filteredMethods = useMemo(() => {
    if (!search.trim()) return methods;
    const lower = search.toLowerCase();
    return methods.filter(
      (m) =>
        m.name.toLowerCase().includes(lower) ||
        m.displayName.toLowerCase().includes(lower) ||
        m.description.toLowerCase().includes(lower)
    );
  }, [methods, search]);

  // Sort: common methods first
  const sortedMethods = useMemo(() => {
    return [...filteredMethods].sort((a, b) => {
      if (a.isCommon && !b.isCommon) return -1;
      if (!a.isCommon && b.isCommon) return 1;
      return 0;
    });
  }, [filteredMethods]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <DialogTitle className="flex items-center gap-2">
                <Box className="h-5 w-5 text-muted-foreground" />
                {componentName}
              </DialogTitle>
              <DialogDescription>
                Select a method to use in the flow, or skip to add as Setup
              </DialogDescription>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0"
              onClick={() => onOpenChange(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search methods..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>

        {/* Method List */}
        <ScrollArea className="h-[300px] pr-4">
          <div className="space-y-2">
            {sortedMethods.map((method) => (
              <button
                key={method.name}
                onClick={() => onSelectMethod(method)}
                className={cn(
                  'w-full p-3 rounded-lg border text-left transition-all',
                  'hover:border-primary hover:bg-accent',
                  method.isCommon && 'border-primary/30 bg-primary/5'
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{method.displayName}</span>
                      {method.isCommon && (
                        <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                          Common
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {method.description}
                    </p>

                    {/* Signature preview */}
                    <div className="flex items-center gap-2 mt-2 text-[10px]">
                      {/* Inputs */}
                      <div className="flex items-center gap-1">
                        {method.inputs.length > 0 ? (
                          method.inputs.map((input) => (
                            <Badge
                              key={input.id}
                              variant="outline"
                              className="text-[9px] px-1 py-0 bg-blue-50 dark:bg-blue-950/30"
                            >
                              {input.name}
                            </Badge>
                          ))
                        ) : (
                          <span className="text-muted-foreground">∅</span>
                        )}
                      </div>

                      <ArrowRight className="h-3 w-3 text-muted-foreground" />

                      {/* Outputs */}
                      <div className="flex items-center gap-1">
                        {method.outputs.length > 0 ? (
                          method.outputs.map((output) => (
                            <Badge
                              key={output.id}
                              variant="outline"
                              className="text-[9px] px-1 py-0 bg-green-50 dark:bg-green-950/30"
                            >
                              {output.name}
                            </Badge>
                          ))
                        ) : (
                          <span className="text-muted-foreground">∅</span>
                        )}
                      </div>
                    </div>
                  </div>

                  <Zap className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                </div>
              </button>
            ))}

            {filteredMethods.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <p className="text-sm">No methods found</p>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="flex items-center justify-between pt-2 border-t">
          <p className="text-xs text-muted-foreground">
            {methods.length} methods available
          </p>
          <Button variant="outline" size="sm" onClick={onSkip}>
            Skip (Add as Setup)
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// Export CATEGORY_METHODS for external use
export { CATEGORY_METHODS };
