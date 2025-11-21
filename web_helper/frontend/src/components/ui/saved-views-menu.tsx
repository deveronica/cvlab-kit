import React from "react";
/**
 * Saved Views Menu Component
 *
 * Dropdown menu for managing saved views
 */

import { useState, useEffect } from 'react';
import { Button } from './button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './dropdown-menu';
import { Save, BookmarkPlus, Trash2, Download} from 'lucide-react';
import { SavedView, getProjectSavedViews, saveView, deleteView, exportViewsAsJSON } from '../../lib/saved-views';

export interface SavedViewsMenuProps {
  projectName: string;
  currentState: SavedView['state'];
  onLoadView: (view: SavedView) => void;
}

export function SavedViewsMenu({
  projectName,
  currentState,
  onLoadView,
}: SavedViewsMenuProps) {
  const [savedViews, setSavedViews] = useState<SavedView[]>([]);
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);
  const [viewName, setViewName] = useState('');

  // Load saved views on mount and when project changes
  useEffect(() => {
    const loadViews = () => {
      setSavedViews(getProjectSavedViews(projectName));
    };

    loadViews();

    // Listen for storage events (from other components saving views)
    window.addEventListener('storage', loadViews);
    return () => {
      window.removeEventListener('storage', loadViews);
    };
  }, [projectName]);

  const handleSaveView = () => {
    if (!viewName.trim()) return;

    const newView = saveView({
      name: viewName,
      projectName,
      state: currentState,
    });

    setSavedViews(prev => [...prev, newView]);
    setViewName('');
    setIsSaveDialogOpen(false);
  };

  const handleDeleteView = (id: string) => {
    if (deleteView(id)) {
      setSavedViews(prev => prev.filter(v => v.id !== id));
    }
  };

  const handleExportViews = () => {
    const json = exportViewsAsJSON(projectName);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${projectName}_saved_views_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm">
            <BookmarkPlus className="h-4 w-4 mr-2" />
            Saved Views ({savedViews.length})
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-64">
          <DropdownMenuLabel>Saved Views</DropdownMenuLabel>
          <DropdownMenuSeparator />

          <DropdownMenuItem onClick={() => setIsSaveDialogOpen(true)}>
            <Save className="h-4 w-4 mr-2" />
            Save Current View
          </DropdownMenuItem>

          {savedViews.length > 0 && (
            <DropdownMenuItem onClick={handleExportViews}>
              <Download className="h-4 w-4 mr-2" />
              Export All Views
            </DropdownMenuItem>
          )}

          {savedViews.length > 0 && <DropdownMenuSeparator />}

          {savedViews.length === 0 ? (
            <div className="px-2 py-6 text-center text-sm text-muted-foreground">
              No saved views yet
            </div>
          ) : (
            <div className="max-h-64 overflow-y-auto">
              {savedViews.map(view => (
                <div
                  key={view.id}
                  className="flex items-center justify-between px-2 py-1 hover:bg-muted/50 rounded group"
                >
                  <button
                    onClick={() => onLoadView(view)}
                    className="flex-1 text-left text-sm px-2 py-1 hover:underline"
                  >
                    {view.name}
                  </button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteView(view.id);
                    }}
                  >
                    <Trash2 className="h-3 w-3 text-destructive" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Save View Dialog */}
      <Dialog open={isSaveDialogOpen} onOpenChange={setIsSaveDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save Current View</DialogTitle>
            <DialogDescription>
              Save your current selection, filters, and column settings to quickly restore later
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div>
              <label className="text-sm font-medium">View Name</label>
              <input
                type="text"
                value={viewName}
                onChange={(e) => setViewName(e.target.value)}
                placeholder="e.g., Best Runs Comparison"
                className="w-full mt-1 px-3 py-2 border rounded-md"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSaveView();
                }}
              />
            </div>

            <div className="text-sm text-muted-foreground space-y-1">
              <p>This view will save:</p>
              <ul className="list-disc list-inside ml-2 space-y-1">
                <li>{currentState.selectedRuns.length} selected runs</li>
                <li>Column visibility settings</li>
                <li>Pinned columns configuration</li>
                <li>Flatten and diff mode preferences</li>
              </ul>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsSaveDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveView} disabled={!viewName.trim()}>
              <Save className="h-4 w-4 mr-2" />
              Save View
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
