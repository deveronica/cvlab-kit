/**
 * ConflictDialog - Handle code-node synchronization conflicts
 *
 * Shows when both code and nodes have been modified since last sync.
 * Allows user to choose resolution strategy:
 * - Keep Code Changes (regenerate nodes)
 * - Keep Node Changes (regenerate code)
 * - Manual Resolution (show diff, let user decide)
 */

import React from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/shared/ui/alert-dialog';
import { AlertTriangle, Code, Network } from 'lucide-react';
import type { Conflict, ResolutionStrategy } from '@/shared/model/ast-blocks';

interface ConflictDialogProps {
  conflicts: Conflict[];
  open: boolean;
  onResolve: (strategy: ResolutionStrategy) => void;
  onCancel: () => void;
}

export function ConflictDialog({
  conflicts,
  open,
  onResolve,
  onCancel,
}: ConflictDialogProps) {
  if (conflicts.length === 0) return null;

  const mainConflict = conflicts[0];
  const codeChanged =
    typeof mainConflict.details?.['code_changed'] === 'boolean'
      ? (mainConflict.details['code_changed'] as boolean)
      : false;
  const nodesChanged =
    typeof mainConflict.details?.['nodes_changed'] === 'boolean'
      ? (mainConflict.details['nodes_changed'] as boolean)
      : false;

  return (
    <AlertDialog open={open}>
      <AlertDialogContent className="max-w-2xl">
        <AlertDialogHeader>
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-amber-500/10">
              <AlertTriangle className="h-6 w-6 text-amber-600 dark:text-amber-400" />
            </div>
            <div>
              <AlertDialogTitle className="text-lg">
                Synchronization Conflict Detected
              </AlertDialogTitle>
              <AlertDialogDescription className="text-sm">
                Both code and node graph have been modified
              </AlertDialogDescription>
            </div>
          </div>
        </AlertDialogHeader>

        {/* Conflict Details */}
        <div className="space-y-4 py-4">
          <div className="rounded-lg border border-border bg-muted/50 p-4">
            <p className="text-sm font-medium mb-2">Conflict Type:</p>
            <p className="text-sm text-muted-foreground">
              {mainConflict.type === 'both_modified'
                ? 'Both code and nodes were edited since last sync'
                : mainConflict.message}
            </p>
          </div>

          {mainConflict.details && (
            <div className="text-xs text-muted-foreground space-y-1">
              {codeChanged && (
                <div className="flex items-center gap-2">
                  <Code className="h-3 w-3" />
                  <span>Code was modified</span>
                </div>
              )}
              {nodesChanged && (
                <div className="flex items-center gap-2">
                  <Network className="h-3 w-3" />
                  <span>Node graph was modified</span>
                </div>
              )}
            </div>
          )}

          <p className="text-sm text-muted-foreground">
            Choose how to resolve this conflict:
          </p>
        </div>

        <AlertDialogFooter className="flex-col sm:flex-col gap-2">
          <AlertDialogAction
            onClick={() => onResolve('code_wins')}
            className="w-full bg-primary"
          >
            <Code className="h-4 w-4 mr-2" />
            Keep Code Changes
            <span className="ml-2 text-xs opacity-70">(Regenerate Nodes)</span>
          </AlertDialogAction>

          <AlertDialogAction
            onClick={() => onResolve('nodes_win')}
            className="w-full bg-primary"
          >
            <Network className="h-4 w-4 mr-2" />
            Keep Node Changes
            <span className="ml-2 text-xs opacity-70">(Regenerate Code)</span>
          </AlertDialogAction>

          <AlertDialogCancel
            onClick={onCancel}
            className="w-full mt-2"
          >
            Cancel (Review Manually)
          </AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/**
 * Sync Status Indicator - Shows current sync state
 */
interface SyncStatusProps {
  isSyncing: boolean;
  hasConflicts: boolean;
  syncError?: string | null;
}

export function SyncStatus({ isSyncing, hasConflicts, syncError }: SyncStatusProps) {
  if (isSyncing) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <div className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
        <span>Syncing...</span>
      </div>
    );
  }

  if (syncError) {
    return (
      <div className="flex items-center gap-2 text-xs text-destructive">
        <AlertTriangle className="h-3 w-3" />
        <span>Sync Error: {syncError}</span>
      </div>
    );
  }

  if (hasConflicts) {
    return (
      <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400">
        <AlertTriangle className="h-3 w-3" />
        <span>Conflicts detected</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400">
      <div className="h-2 w-2 rounded-full bg-green-500" />
      <span>Synced</span>
    </div>
  );
}
