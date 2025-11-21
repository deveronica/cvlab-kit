import React from "react";
/**
 * Notes & Tags Editor Component
 *
 * Inline editor for run notes and tags in RunDetailModal.
 *
 * Features:
 * - Edit notes with textarea
 * - Add/remove tags with tag badges
 * - Tag autocomplete from existing project tags
 * - Save/cancel actions
 * - Character count for notes
 */

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';
import { Button } from './button';
import { Input } from './input';
import { Textarea } from './textarea';
import { Badge } from './badge';
import { Save, X, Plus, Tag as TagIcon, AlertCircle, Loader2 } from 'lucide-react';
import { devError } from '@/lib/dev-utils';

interface NotesTagsEditorProps {
  project: string;
  runName: string;
  initialNotes?: string;
  initialTags?: string[];
  onSave?: (notes: string, tags: string[]) => void;
}

export function NotesTagsEditor({
  project,
  runName,
  initialNotes = '',
  initialTags = [],
  onSave,
}: NotesTagsEditorProps) {
  const [notes, setNotes] = useState(initialNotes);
  const [tags, setTags] = useState<string[]>(initialTags);
  const [newTag, setNewTag] = useState('');
  const [projectTags, setProjectTags] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasChanges, setHasChanges] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch project tags for autocomplete
  useEffect(() => {
    const fetchProjectTags = async () => {
      try {
        const response = await fetch(`/api/runs/${project}/tags`);
        if (response.ok) {
          const data = await response.json();
          setProjectTags(data.data.tags || []);
        }
      } catch (err) {
        devError('Failed to fetch project tags:', err);
      }
    };

    fetchProjectTags();
  }, [project]);

  // Track changes
  useEffect(() => {
    const notesChanged = notes !== initialNotes;
    const tagsChanged = JSON.stringify(tags.sort()) !== JSON.stringify([...initialTags].sort());
    setHasChanges(notesChanged || tagsChanged);
  }, [notes, tags, initialNotes, initialTags]);

  // Filter suggestions
  const suggestions = projectTags.filter(
    tag => tag.toLowerCase().includes(newTag.toLowerCase()) && !tags.includes(tag)
  );

  const handleAddTag = (tag: string) => {
    const trimmedTag = tag.trim();
    if (trimmedTag && !tags.includes(trimmedTag)) {
      setTags([...tags, trimmedTag]);
      setNewTag('');
      setShowSuggestions(false);
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setTags(tags.filter(tag => tag !== tagToRemove));
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);

    try {
      // Save notes
      const notesResponse = await fetch(`/api/runs/${project}/${runName}/notes`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes }),
      });

      if (!notesResponse.ok) {
        throw new Error('Failed to save notes');
      }

      // Save tags
      const tagsResponse = await fetch(`/api/runs/${project}/${runName}/tags`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags }),
      });

      if (!tagsResponse.ok) {
        throw new Error('Failed to save tags');
      }

      // Call parent callback
      if (onSave) {
        onSave(notes, tags);
      }

      setHasChanges(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save changes');
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setNotes(initialNotes);
    setTags(initialTags);
    setNewTag('');
    setError(null);
    setHasChanges(false);
  };

  return (
    <Card variant="compact">
      <CardHeader variant="compact">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle size="base">Notes & Tags</CardTitle>
            <CardDescription>
              Add notes and tags to organize and annotate this run
            </CardDescription>
          </div>
          {hasChanges && (
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleCancel}
                disabled={saving}
              >
                <X className="h-4 w-4 mr-2" />
                Cancel
              </Button>
              <Button
                variant="default"
                size="sm"
                onClick={handleSave}
                disabled={saving}
              >
                {saving ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4 mr-2" />
                    Save
                  </>
                )}
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent variant="compact">
        {/* Error Message */}
        {error && (
          <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/20 p-3 rounded-md mb-4">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Notes & Tags - Side by Side Layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Notes Section */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Notes</label>
            <Textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add notes about this run..."
              rows={6}
              maxLength={2000}
              className="resize-none"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Press Save to persist changes</span>
              <span>{notes.length}/2000</span>
            </div>
          </div>

          {/* Tags Section */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Tags</label>

            {/* Tag Input */}
            <div className="relative">
              <div className="flex gap-2">
                <Input
                  ref={inputRef}
                  value={newTag}
                  onChange={(e) => {
                    setNewTag(e.target.value);
                    setShowSuggestions(e.target.value.length > 0);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleAddTag(newTag);
                    }
                  }}
                  placeholder="Add a tag..."
                  className="flex-1"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleAddTag(newTag)}
                  disabled={!newTag.trim()}
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Add
                </Button>
              </div>

              {/* Tag Suggestions */}
              {showSuggestions && suggestions.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-background border rounded-md shadow-lg max-h-48 overflow-auto">
                  {suggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      className="w-full px-3 py-2 text-left text-sm hover:bg-muted transition-colors"
                      onClick={() => handleAddTag(suggestion)}
                    >
                      <TagIcon className="h-3 w-3 inline mr-2 text-muted-foreground" />
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Tag List */}
            {tags.length > 0 ? (
              <div className="flex flex-wrap gap-2 mt-2">
                {tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="pl-2 pr-1 py-1">
                    <TagIcon className="h-3 w-3 mr-1" />
                    {tag}
                    <button
                      onClick={() => handleRemoveTag(tag)}
                      className="ml-1 hover:bg-muted-foreground/20 rounded-full p-0.5 transition-colors"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground italic py-2">
                No tags added yet
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
