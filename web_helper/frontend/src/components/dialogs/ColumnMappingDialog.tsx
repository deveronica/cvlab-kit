import React from "react";
/**
 * Column Mapping Dialog
 *
 * Full-featured editor for reviewing automatic suggestions and creating manual mappings.
 * Supports:
 * - Reviewing all suggestions with confidence scores
 * - Batch accept/reject
 * - Manual mapping creation
 * - Editing existing mappings
 * - Filtering and search
 */

import { useState, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { ScrollArea } from '../ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import {
  Check,
  X,
  Plus,
  Search,
  Filter,
  Trash2,
  Edit2,
  Save,
  Sparkles,
  AlertCircle,
} from 'lucide-react';
import type {
  ColumnMapping,
  ColumnSuggestion,
  ColumnMappingCreate,
} from '../../lib/api/column-mappings';

interface ColumnMappingDialogProps {
  open: boolean;
  onClose: () => void;
  project: string;
  suggestions: ColumnSuggestion[];
  existingMappings: ColumnMapping[];
  onAcceptSuggestion: (suggestion: ColumnSuggestion) => Promise<void>;
  onCreateMapping: (mapping: ColumnMappingCreate) => Promise<void>;
  onUpdateMapping: (id: number, target: string) => Promise<void>;
  onDeleteMapping: (id: number) => Promise<void>;
  isLoading?: boolean;
}

export function ColumnMappingDialog({
  open,
  onClose,
  project,
  suggestions,
  existingMappings,
  onAcceptSuggestion,
  onCreateMapping,
  onUpdateMapping,
  onDeleteMapping,
  isLoading = false,
}: ColumnMappingDialogProps) {
  const [activeTab, setActiveTab] = useState<'suggestions' | 'existing' | 'manual'>('suggestions');
  const [searchQuery, setSearchQuery] = useState('');
  const [confidenceFilter, setConfidenceFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState('');

  // Manual mapping form state
  const [manualSource, setManualSource] = useState('');
  const [manualTarget, setManualTarget] = useState('');
  const [manualType, setManualType] = useState<'hyperparam' | 'metric'>('hyperparam');

  // Filter suggestions
  const filteredSuggestions = useMemo(() => {
    let filtered = suggestions;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        s =>
          s.source_column.toLowerCase().includes(query) ||
          s.target_column.toLowerCase().includes(query)
      );
    }

    // Confidence filter
    if (confidenceFilter !== 'all') {
      filtered = filtered.filter(s => {
        if (confidenceFilter === 'high') return s.confidence_score >= 0.8;
        if (confidenceFilter === 'medium')
          return s.confidence_score >= 0.5 && s.confidence_score < 0.8;
        if (confidenceFilter === 'low') return s.confidence_score < 0.5;
        return true;
      });
    }

    return filtered;
  }, [suggestions, searchQuery, confidenceFilter]);

  // Filter existing mappings
  const filteredMappings = useMemo(() => {
    if (!searchQuery) return existingMappings;

    const query = searchQuery.toLowerCase();
    return existingMappings.filter(
      m =>
        m.source_column.toLowerCase().includes(query) ||
        m.target_column.toLowerCase().includes(query)
    );
  }, [existingMappings, searchQuery]);

  const getConfidenceBadge = (score: number) => {
    if (score >= 0.8) {
      return (
        <Badge variant="default" className="bg-green-500">
          High {(score * 100).toFixed(0)}%
        </Badge>
      );
    } else if (score >= 0.5) {
      return <Badge variant="secondary">Medium {(score * 100).toFixed(0)}%</Badge>;
    } else {
      return <Badge variant="outline">Low {(score * 100).toFixed(0)}%</Badge>;
    }
  };

  const handleEdit = (mapping: ColumnMapping) => {
    setEditingId(mapping.id);
    setEditValue(mapping.target_column);
  };

  const handleSaveEdit = async (id: number) => {
    if (editValue && editValue !== '') {
      await onUpdateMapping(id, editValue);
      setEditingId(null);
      setEditValue('');
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditValue('');
  };

  const handleManualCreate = async () => {
    if (manualSource && manualTarget) {
      await onCreateMapping({
        source_column: manualSource,
        target_column: manualTarget,
        column_type: manualType,
        mapping_method: 'manual',
      });
      setManualSource('');
      setManualTarget('');
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Column Mapping Editor</DialogTitle>
          <DialogDescription>
            Review automatic suggestions and manage column mappings for {project}
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="flex-1 flex flex-col min-h-0">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="suggestions" className="gap-2">
              <Sparkles className="h-4 w-4" />
              Suggestions ({suggestions.length})
            </TabsTrigger>
            <TabsTrigger value="existing" className="gap-2">
              <Check className="h-4 w-4" />
              Existing ({existingMappings.length})
            </TabsTrigger>
            <TabsTrigger value="manual" className="gap-2">
              <Plus className="h-4 w-4" />
              Manual
            </TabsTrigger>
          </TabsList>

          {/* Suggestions Tab */}
          <TabsContent value="suggestions" className="flex-1 flex flex-col space-y-4 min-h-0">
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search columns..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
              <Select value={confidenceFilter} onValueChange={(v) => setConfidenceFilter(v as any)}>
                <SelectTrigger className="w-[180px]">
                  <Filter className="h-4 w-4 mr-2" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Confidence</SelectItem>
                  <SelectItem value="high">High (≥80%)</SelectItem>
                  <SelectItem value="medium">Medium (50-80%)</SelectItem>
                  <SelectItem value="low">Low (&lt;50%)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <ScrollArea className="flex-1 border rounded-md">
              {filteredSuggestions.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  <AlertCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No suggestions found</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Source Column</TableHead>
                      <TableHead>Target Column</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Reason</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredSuggestions.map((suggestion, index) => (
                      <TableRow key={`${suggestion.source_column}-${index}`}>
                        <TableCell>
                          <code className="text-sm bg-muted px-2 py-0.5 rounded">
                            {suggestion.source_column}
                          </code>
                        </TableCell>
                        <TableCell>
                          <code className="text-sm bg-blue-50 dark:bg-blue-950 px-2 py-0.5 rounded font-semibold">
                            {suggestion.target_column}
                          </code>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{suggestion.column_type}</Badge>
                        </TableCell>
                        <TableCell>{getConfidenceBadge(suggestion.confidence_score)}</TableCell>
                        <TableCell className="max-w-xs">
                          <p className="text-xs text-muted-foreground truncate">
                            {suggestion.reason}
                          </p>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-1">
                            <Button
                              size="sm"
                              variant="default"
                              onClick={() => onAcceptSuggestion(suggestion)}
                              disabled={isLoading}
                              className="bg-green-600 hover:bg-green-700"
                            >
                              <Check className="h-3 w-3 mr-1" />
                              Accept
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </ScrollArea>
          </TabsContent>

          {/* Existing Mappings Tab */}
          <TabsContent value="existing" className="flex-1 flex flex-col space-y-4 min-h-0">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search mappings..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>

            <ScrollArea className="flex-1 border rounded-md">
              {filteredMappings.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  <AlertCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No existing mappings</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Source Column</TableHead>
                      <TableHead>Target Column</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Method</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredMappings.map((mapping) => (
                      <TableRow key={mapping.id}>
                        <TableCell>
                          <code className="text-sm bg-muted px-2 py-0.5 rounded">
                            {mapping.source_column}
                          </code>
                        </TableCell>
                        <TableCell>
                          {editingId === mapping.id ? (
                            <Input
                              value={editValue}
                              onChange={(e) => setEditValue(e.target.value)}
                              className="h-8"
                              autoFocus
                            />
                          ) : (
                            <code className="text-sm bg-blue-50 dark:bg-blue-950 px-2 py-0.5 rounded">
                              {mapping.target_column}
                            </code>
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{mapping.column_type}</Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant={mapping.mapping_method === 'manual' ? 'secondary' : 'default'}>
                            {mapping.mapping_method}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          {editingId === mapping.id ? (
                            <div className="flex justify-end gap-1">
                              <Button
                                size="sm"
                                variant="default"
                                onClick={() => handleSaveEdit(mapping.id)}
                                disabled={isLoading}
                              >
                                <Save className="h-3 w-3" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={handleCancelEdit}
                              >
                                <X className="h-3 w-3" />
                              </Button>
                            </div>
                          ) : (
                            <div className="flex justify-end gap-1">
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => handleEdit(mapping)}
                                disabled={isLoading}
                              >
                                <Edit2 className="h-3 w-3" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => onDeleteMapping(mapping.id)}
                                disabled={isLoading}
                                className="text-destructive hover:text-destructive"
                              >
                                <Trash2 className="h-3 w-3" />
                              </Button>
                            </div>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </ScrollArea>
          </TabsContent>

          {/* Manual Creation Tab */}
          <TabsContent value="manual" className="flex-1 flex flex-col space-y-4">
            <div className="space-y-4 p-4 border rounded-md">
              <div className="space-y-2">
                <label className="text-sm font-medium">Source Column Name</label>
                <Input
                  placeholder="e.g., lr, learning_rate"
                  value={manualSource}
                  onChange={(e) => setManualSource(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Target Column Name (Unified)</label>
                <Input
                  placeholder="e.g., learning_rate"
                  value={manualTarget}
                  onChange={(e) => setManualTarget(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Column Type</label>
                <Select value={manualType} onValueChange={(v) => setManualType(v as any)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="hyperparam">Hyperparameter</SelectItem>
                    <SelectItem value="metric">Metric</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button
                onClick={handleManualCreate}
                disabled={!manualSource || !manualTarget || isLoading}
                className="w-full"
              >
                <Plus className="h-4 w-4 mr-2" />
                Create Mapping
              </Button>
            </div>
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
