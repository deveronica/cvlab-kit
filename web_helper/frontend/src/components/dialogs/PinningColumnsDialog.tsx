import React from "react";
/**
 * Pinning Columns Dialog
 * Configure which columns should be pinned left or right in table views
 */

import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { ScrollArea } from '../ui/scroll-area';
import { Pin, PinOff, X } from 'lucide-react';

interface PinningColumnsDialogProps {
  open: boolean;
  onClose: () => void;
  hyperparamColumns: string[];
  metricColumns: string[];
  pinnedLeftHyperparams: Set<string>;
  pinnedRightMetrics: Set<string>;
  onSave: (pinnedLeft: Set<string>, pinnedRight: Set<string>) => void;
}

export function PinningColumnsDialog({
  open,
  onClose,
  hyperparamColumns,
  metricColumns,
  pinnedLeftHyperparams,
  pinnedRightMetrics,
  onSave,
}: PinningColumnsDialogProps) {
  const [localPinnedLeft, setLocalPinnedLeft] = useState<Set<string>>(new Set());
  const [localPinnedRight, setLocalPinnedRight] = useState<Set<string>>(new Set());

  // Sync with props when dialog opens
  useEffect(() => {
    if (open) {
      setLocalPinnedLeft(new Set(pinnedLeftHyperparams));
      setLocalPinnedRight(new Set(pinnedRightMetrics));
    }
  }, [open, pinnedLeftHyperparams, pinnedRightMetrics]);

  const togglePinLeft = (column: string) => {
    setLocalPinnedLeft(prev => {
      const newSet = new Set(prev);
      if (newSet.has(column)) {
        newSet.delete(column);
      } else {
        newSet.add(column);
      }
      return newSet;
    });
  };

  const togglePinRight = (column: string) => {
    setLocalPinnedRight(prev => {
      const newSet = new Set(prev);
      if (newSet.has(column)) {
        newSet.delete(column);
      } else {
        newSet.add(column);
      }
      return newSet;
    });
  };

  const handleSave = () => {
    onSave(localPinnedLeft, localPinnedRight);
    onClose();
  };

  const handleReset = () => {
    setLocalPinnedLeft(new Set());
    setLocalPinnedRight(new Set());
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Configure Pinned Columns</DialogTitle>
          <DialogDescription>
            Pin hyperparameters to the left and metrics to the right for easier comparison
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-4 flex-1 min-h-0">
          {/* Left Pinning - Hyperparameters */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold">Pin Left (Hyperparameters)</h3>
              <Badge variant="outline" className="text-xs">
                {localPinnedLeft.size} selected
              </Badge>
            </div>
            <ScrollArea className="h-[300px] border rounded-md p-2">
              <div className="space-y-1">
                {hyperparamColumns.length === 0 ? (
                  <p className="text-sm text-muted-foreground p-2">No hyperparameters available</p>
                ) : (
                  hyperparamColumns.map(param => {
                    const isPinned = localPinnedLeft.has(param);
                    return (
                      <button
                        key={param}
                        onClick={() => togglePinLeft(param)}
                        className={`w-full flex items-center justify-between p-2 rounded text-sm transition-colors ${
                          isPinned
                            ? 'bg-blue-50 dark:bg-blue-950/30 text-blue-900 dark:text-blue-100'
                            : 'hover:bg-muted'
                        }`}
                      >
                        <span className="truncate">{param}</span>
                        {isPinned ? (
                          <Pin className="h-3 w-3 text-blue-600 dark:text-blue-400 flex-shrink-0 ml-2" />
                        ) : (
                          <PinOff className="h-3 w-3 text-muted-foreground flex-shrink-0 ml-2" />
                        )}
                      </button>
                    );
                  })
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Right Pinning - Metrics */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold">Pin Right (Metrics)</h3>
              <Badge variant="outline" className="text-xs">
                {localPinnedRight.size} selected
              </Badge>
            </div>
            <ScrollArea className="h-[300px] border rounded-md p-2">
              <div className="space-y-1">
                {metricColumns.length === 0 ? (
                  <p className="text-sm text-muted-foreground p-2">No metrics available</p>
                ) : (
                  metricColumns.map(metric => {
                    const isPinned = localPinnedRight.has(metric);
                    return (
                      <button
                        key={metric}
                        onClick={() => togglePinRight(metric)}
                        className={`w-full flex items-center justify-between p-2 rounded text-sm transition-colors ${
                          isPinned
                            ? 'bg-green-50 dark:bg-green-950/30 text-green-900 dark:text-green-100'
                            : 'hover:bg-muted'
                        }`}
                      >
                        <span className="truncate">{metric}</span>
                        {isPinned ? (
                          <Pin className="h-3 w-3 text-green-600 dark:text-green-400 flex-shrink-0 ml-2" />
                        ) : (
                          <PinOff className="h-3 w-3 text-muted-foreground flex-shrink-0 ml-2" />
                        )}
                      </button>
                    );
                  })
                )}
              </div>
            </ScrollArea>
          </div>
        </div>

        <DialogFooter className="flex-col sm:flex-row gap-2 flex-shrink-0">
          <Button variant="outline" onClick={handleReset}>
            <X className="h-4 w-4 mr-2" />
            Reset All
          </Button>
          <div className="flex gap-2 flex-1 justify-end">
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleSave}>
              Save Pinning Configuration
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
