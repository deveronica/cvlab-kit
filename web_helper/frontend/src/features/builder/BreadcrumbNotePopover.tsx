/**
 * BreadcrumbNotePopover - Add notes to breadcrumb levels
 *
 * Allows users to annotate hierarchy levels with documentation.
 * Notes are stored locally and can be exported with the agent.
 */

import React, { useState, useCallback } from 'react';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/shared/ui/popover';
import { Button } from '@/shared/ui/button';
import { Textarea } from '@/shared/ui/textarea';
import { MessageSquarePlus, MessageSquare, X, Check } from 'lucide-react';
import { cn } from '@/shared/lib/utils';

interface BreadcrumbNotePopoverProps {
  pathKey: string; // Unique key for the path (e.g., "agent/model/layer1")
  note?: string;
  onNoteChange: (pathKey: string, note: string | null) => void;
  children: React.ReactNode;
  className?: string;
}

export function BreadcrumbNotePopover({
  pathKey,
  note,
  onNoteChange,
  children,
  className,
}: BreadcrumbNotePopoverProps) {
  const [open, setOpen] = useState(false);
  const [editingNote, setEditingNote] = useState(note || '');
  const hasNote = !!note && note.trim().length > 0;

  // Handle opening - reset editing state
  const handleOpenChange = useCallback(
    (isOpen: boolean) => {
      if (isOpen) {
        setEditingNote(note || '');
      }
      setOpen(isOpen);
    },
    [note]
  );

  // Handle save
  const handleSave = useCallback(() => {
    const trimmed = editingNote.trim();
    onNoteChange(pathKey, trimmed || null);
    setOpen(false);
  }, [editingNote, pathKey, onNoteChange]);

  // Handle delete
  const handleDelete = useCallback(() => {
    onNoteChange(pathKey, null);
    setEditingNote('');
    setOpen(false);
  }, [pathKey, onNoteChange]);

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <div
          className={cn(
            'relative group cursor-pointer inline-flex items-center',
            className
          )}
        >
          {children}
          {/* Note indicator */}
          {hasNote ? (
            <MessageSquare className="h-3 w-3 ml-1 text-yellow-500 flex-shrink-0" />
          ) : (
            <MessageSquarePlus className="h-3 w-3 ml-1 text-muted-foreground opacity-0 group-hover:opacity-50 transition-opacity flex-shrink-0" />
          )}
        </div>
      </PopoverTrigger>
      <PopoverContent className="w-72 p-3" align="start">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">
              Level Note
            </span>
            {hasNote && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0 text-destructive hover:text-destructive"
                onClick={handleDelete}
              >
                <X className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
          <Textarea
            value={editingNote}
            onChange={(e) => setEditingNote(e.target.value)}
            placeholder="Add a note to this level..."
            className="min-h-[80px] text-xs resize-none"
            autoFocus
          />
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground">
              {editingNote.length}/500
            </span>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                onClick={() => setOpen(false)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                className="h-7"
                onClick={handleSave}
                disabled={editingNote.length > 500}
              >
                <Check className="h-3.5 w-3.5 mr-1" />
                Save
              </Button>
            </div>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

// Hook to manage breadcrumb notes
export function useBreadcrumbNotes() {
  const [notes, setNotes] = useState<Record<string, string>>({});

  const setNote = useCallback((pathKey: string, note: string | null) => {
    setNotes((prev) => {
      if (note === null) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { [pathKey]: _removed, ...rest } = prev;
        return rest;
      }
      return { ...prev, [pathKey]: note };
    });
  }, []);

  const getNote = useCallback(
    (pathKey: string) => notes[pathKey] || undefined,
    [notes]
  );

  const clearNotes = useCallback(() => {
    setNotes({});
  }, []);

  return { notes, setNote, getNote, clearNotes };
}
