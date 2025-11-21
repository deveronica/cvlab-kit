import React from "react";
/**
 * Unified Column Manager
 * Modern, comprehensive column management interface
 *
 * Features:
 * - Column Visibility (show/hide)
 * - Column Pinning (left/right)
 * - Column Ordering (drag & drop)
 * - Search/Filter
 * - Bulk Actions
 */

import { useState, useEffect, useMemo, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
  DragStartEvent,
  _DragOverlay,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
  AnimateLayoutChanges,
  _defaultAnimateLayoutChanges,
} from '@dnd-kit/sortable';
import { restrictToVerticalAxis } from '@dnd-kit/modifiers';
import { CSS } from '@dnd-kit/utilities';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from './dropdown-menu';
import { Button } from './button';
import { Badge } from './badge';
import { Input } from './input';
import {
  _Columns,
  Search,
  Eye,
  EyeOff,
  Pin,
  PinOff,
  GripVertical,
  _Lock,
  _RotateCcw,
  _Check,
  X,
  ChevronDown,
} from 'lucide-react';
import { ScrollArea } from './scroll-area';

interface ColumnManagerProps {
  // Column lists
  hyperparamColumns: string[];
  metricColumns: string[];

  // Visibility
  visibleHyperparams: Set<string>;
  visibleMetrics: Set<string>;

  // Pinning
  pinnedLeftHyperparams: Set<string>;
  pinnedRightMetrics: Set<string>;

  // Callbacks
  onVisibilityChange: (hyperparams: Set<string>, metrics: Set<string>) => void;
  onPinnedChange: (pinnedLeft: Set<string>, pinnedRight: Set<string>) => void;
  onPinningEnabledChange?: (enabled: boolean) => void;
  onColumnOrderChange?: (hyperparams: string[], metrics: string[]) => void;
}

type PinState = 'none' | 'left' | 'right';

// Sortable Item Component
interface SortableItemProps {
  id: string;
  isAlwaysVisible?: boolean;
  isVisible: boolean;
  pinState: PinState;
  onVisibilityToggle: () => void;
  onPinToggle: (state: PinState) => void;
  isHyperparam: boolean;
  isHighlighted?: boolean;
  isAnyDragging?: boolean;
}

// Disable layout animations to prevent width flickering on drag/drop
const animateLayoutChanges: AnimateLayoutChanges = () => {
  // Always disable layout animations for instant position changes
  return false;
};

function SortableItem({
  id,
  isAlwaysVisible,
  isVisible,
  pinState,
  onVisibilityToggle,
  onPinToggle,
  isHyperparam,
  isHighlighted,
  isAnyDragging,
}: SortableItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    _transition,
    isDragging,
  } = useSortable({
    id,
    animateLayoutChanges,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    // Disable all transform transitions to prevent drop animation
    _transition: 'none',
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 100 : 'auto',
  };

  const className = `grid grid-cols-[12px_minmax(0,1fr)_auto_auto] gap-1 items-center px-1.5 py-1 rounded relative ${
    isAlwaysVisible
      ? 'bg-muted/50 opacity-60'
      : isHighlighted
      ? 'animate-highlight-flash z-50'
      : 'hover:bg-muted/50 _transition-colors duration-200 z-0'
  }`;

  const content = (
    <>
      <div {...attributes} {...listeners} className="cursor-grab active:cursor-grabbing">
        <GripVertical className="h-3 w-3 flex-shrink-0 text-muted-foreground" />
      </div>

      <span className={`text-[11px] truncate min-w-0 ${isVisible ? 'font-medium' : 'text-muted-foreground line-through'}`}>
        {id}
      </span>

      <button
        className={`h-4 w-4 flex items-center justify-center rounded flex-shrink-0 ${
          isAlwaysVisible
            ? 'bg-primary/40 text-primary-foreground/60 cursor-not-allowed'
            : isVisible
            ? 'bg-primary text-primary-foreground hover:bg-primary/90 _transition-colors'
            : 'bg-muted text-muted-foreground hover:bg-muted/80 _transition-colors'
        }`}
        onClick={(e) => {
          e.stopPropagation();
          if (!isAlwaysVisible) onVisibilityToggle();
        }}
        disabled={isAlwaysVisible}
        aria-label={isVisible ? 'Hide column' : 'Show column'}
      >
        {isVisible ? (
          <Eye className="h-2.5 w-2.5" />
        ) : (
          <EyeOff className="h-2.5 w-2.5" />
        )}
      </button>

      <button
        className={`h-4 w-4 flex items-center justify-center rounded flex-shrink-0 ${
          isAlwaysVisible
            ? 'bg-blue-600/40 text-white/60 cursor-not-allowed'
            : pinState === (isHyperparam ? 'left' : 'right')
            ? isHyperparam
              ? 'bg-blue-600 text-white hover:bg-blue-700 _transition-colors'
              : 'bg-green-600 text-white hover:bg-green-700 _transition-colors'
            : 'bg-muted text-muted-foreground hover:bg-muted/80 _transition-colors'
        }`}
        onClick={(e) => {
          e.stopPropagation();
          if (!isAlwaysVisible) onPinToggle(isHyperparam ? 'left' : 'right');
        }}
        disabled={isAlwaysVisible}
        aria-label={pinState === (isHyperparam ? 'left' : 'right') ? 'Unpin column' : 'Pin column'}
      >
        {pinState === (isHyperparam ? 'left' : 'right') ? (
          <Pin className="h-2.5 w-2.5" />
        ) : (
          <PinOff className="h-2.5 w-2.5" />
        )}
      </button>
    </>
  );

  // Use regular div when any item is dragging for DnD Kit to work properly
  if (isAnyDragging) {
    return (
      <div ref={setNodeRef} style={style} className={className}>
        {content}
      </div>
    );
  }

  // Use simple div (no framer-motion animation to prevent any layout shifts)
  return (
    <div
      ref={setNodeRef}
      style={style}
      className={className}
    >
      {content}
    </div>
  );
}

export function ColumnManager({
  hyperparamColumns,
  metricColumns,
  visibleHyperparams,
  visibleMetrics,
  pinnedLeftHyperparams,
  pinnedRightMetrics,
  onVisibilityChange,
  onPinnedChange,
  onPinningEnabledChange,
  onColumnOrderChange,
}: ColumnManagerProps) {
  const [showMode, setShowMode] = useState<'hyperparams' | 'metrics'>('hyperparams');
  const [searchQuery, setSearchQuery] = useState('');
  const [activeId, setActiveId] = useState<string | null>(null);

  // Local state for column management
  const [localVisibleHyperparams, setLocalVisibleHyperparams] = useState<Set<string>>(new Set(visibleHyperparams));
  const [localVisibleMetrics, setLocalVisibleMetrics] = useState<Set<string>>(new Set(visibleMetrics));
  const [localPinnedLeft, setLocalPinnedLeft] = useState<Set<string>>(new Set(pinnedLeftHyperparams));
  const [localPinnedRight, setLocalPinnedRight] = useState<Set<string>>(new Set(pinnedRightMetrics));

  // Track if pinning change is from internal user action (to avoid sync loops)
  const isInternalChangeRef = useRef(false);

  // Pin feature toggle state (default: true)
  const [isPinningEnabled, setIsPinningEnabled] = useState<boolean>(true);

  // Always visible columns (cannot hide but can reorder)
  const alwaysVisibleColumns = ['Run ID', 'Status', 'Tags'];

  // Column order management (hyperparamColumns already includes Run ID, Status, Tags)
  const [hyperparamOrder, setHyperparamOrder] = useState<string[]>(hyperparamColumns);
  const [metricOrder, setMetricOrder] = useState<string[]>(metricColumns);

  // Track original positions before pinning (for position restoration on unpin)
  const [originalPositions, setOriginalPositions] = useState<Map<string, number>>(new Map());

  // Track highlighted column for animation
  const [highlightedColumn, setHighlightedColumn] = useState<string | null>(null);

  // Sync with props - directly use props order
  useEffect(() => {
    setHyperparamOrder(hyperparamColumns);
  }, [hyperparamColumns]);

  useEffect(() => {
    setMetricOrder(metricColumns);
  }, [metricColumns]);

  useEffect(() => {
    setLocalVisibleHyperparams(new Set(visibleHyperparams));
    setLocalVisibleMetrics(new Set(visibleMetrics));
  }, [visibleHyperparams, visibleMetrics]);

  // Sync pinning state from props only when change originates externally
  useEffect(() => {
    if (!isInternalChangeRef.current) {
      setLocalPinnedLeft(new Set(pinnedLeftHyperparams));
      setLocalPinnedRight(new Set(pinnedRightMetrics));
    }
    // Reset flag after sync
    isInternalChangeRef.current = false;
  }, [pinnedLeftHyperparams, pinnedRightMetrics]);

  // Filter columns by search query and split into pinned/unpinned
  const filteredHyperparams = useMemo(() => {
    const filtered = !searchQuery
      ? hyperparamOrder
      : hyperparamOrder.filter(col => col.toLowerCase().includes(searchQuery.toLowerCase()));
    return filtered;
  }, [hyperparamOrder, searchQuery]);

  const pinnedHyperparams = useMemo(() => {
    // Include both user-pinned columns AND always-visible columns that should be pinned
    // (Run ID, Status, Tags are always pinned in AGGridProjectsTable)
    return filteredHyperparams.filter(col =>
      localPinnedLeft.has(col) || alwaysVisibleColumns.includes(col)
    );
  }, [filteredHyperparams, localPinnedLeft, alwaysVisibleColumns]);

  const unpinnedHyperparams = useMemo(() => {
    // Exclude both pinned columns AND always-visible columns (which are always pinned)
    return filteredHyperparams.filter(col =>
      !localPinnedLeft.has(col) && !alwaysVisibleColumns.includes(col)
    );
  }, [filteredHyperparams, localPinnedLeft, alwaysVisibleColumns]);

  const filteredMetrics = useMemo(() => {
    const filtered = !searchQuery
      ? metricOrder
      : metricOrder.filter(col => col.toLowerCase().includes(searchQuery.toLowerCase()));
    return filtered;
  }, [metricOrder, searchQuery]);

  const pinnedMetrics = useMemo(() => {
    return filteredMetrics.filter(col => localPinnedRight.has(col));
  }, [filteredMetrics, localPinnedRight]);

  const unpinnedMetrics = useMemo(() => {
    return filteredMetrics.filter(col => !localPinnedRight.has(col));
  }, [filteredMetrics, localPinnedRight]);

  // Statistics (Run ID, Status are already included in hyperparamColumns)
  const stats = useMemo(() => {
    const visibleCount = localVisibleHyperparams.size + localVisibleMetrics.size;
    const totalCount = hyperparamColumns.length + metricColumns.length;
    const pinnedCount = localPinnedLeft.size + localPinnedRight.size;

    // For category badges
    const hyperparamsVisibleCount = localVisibleHyperparams.size;
    const hyperparamsTotalCount = hyperparamColumns.length;
    const metricsVisibleCount = localVisibleMetrics.size;
    const metricsTotalCount = metricColumns.length;

    return {
      visibleCount,
      totalCount,
      pinnedCount,
      hyperparamsVisibleCount,
      hyperparamsTotalCount,
      metricsVisibleCount,
      metricsTotalCount
    };
  }, [localVisibleHyperparams, localVisibleMetrics, localPinnedLeft, localPinnedRight, hyperparamColumns, metricColumns]);

  // Sensors for drag and drop
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // Drag handlers
  const handleDragStart = (event: DragStartEvent) => {
    setActiveId(event.active.id as string);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const isHyperparam = showMode === 'hyperparams';
      const currentOrder = isHyperparam ? hyperparamOrder : metricOrder;

      const oldIndex = currentOrder.indexOf(active.id as string);
      const newIndex = currentOrder.indexOf(over.id as string);

      const newOrder = arrayMove(currentOrder, oldIndex, newIndex);

      if (isHyperparam) {
        setHyperparamOrder(newOrder);
      } else {
        setMetricOrder(newOrder);
      }

      if (onColumnOrderChange) {
        onColumnOrderChange(
          isHyperparam ? newOrder : hyperparamOrder,
          isHyperparam ? metricOrder : newOrder
        );
      }

      // Highlight the moved column with CSS animation
      setHighlightedColumn(active.id as string);
      setTimeout(() => setHighlightedColumn(null), 1500);
    }

    setActiveId(null);
  };

  // Visibility handlers
  const toggleVisibility = (column: string) => {
    const isHyperparam = showMode === 'hyperparams';

    if (isHyperparam) {
      const newVisible = new Set(localVisibleHyperparams);
      if (newVisible.has(column)) {
        newVisible.delete(column);
      } else {
        newVisible.add(column);
      }
      setLocalVisibleHyperparams(newVisible);
      onVisibilityChange(newVisible, localVisibleMetrics);
    } else {
      const newVisible = new Set(localVisibleMetrics);
      if (newVisible.has(column)) {
        newVisible.delete(column);
      } else {
        newVisible.add(column);
      }
      setLocalVisibleMetrics(newVisible);
      onVisibilityChange(localVisibleHyperparams, newVisible);
    }
  };

  // Pinning handlers
  const getPinState = (column: string): PinState => {
    // Always-visible columns (Run ID, Status, Tags) are always pinned left
    if (alwaysVisibleColumns.includes(column)) return 'left';
    if (localPinnedLeft.has(column)) return 'left';
    if (localPinnedRight.has(column)) return 'right';
    return 'none';
  };

  const togglePin = (column: string, targetState: PinState) => {
    // Always-visible columns (Run ID, Status, Tags) cannot be unpinned
    if (alwaysVisibleColumns.includes(column)) {
      return;
    }

    const isHyperparam = showMode === 'hyperparams';
    const currentState = getPinState(column);
    const currentOrder = isHyperparam ? hyperparamOrder : metricOrder;
    const pinnedSet = isHyperparam ? localPinnedLeft : localPinnedRight;

    // If clicking the same state, unpin
    const newState = currentState === targetState ? 'none' : targetState;

    // Trigger highlight animation with CSS animation
    setHighlightedColumn(column);
    setTimeout(() => setHighlightedColumn(null), 1500);

    const newPinnedLeft = new Set(localPinnedLeft);
    const newPinnedRight = new Set(localPinnedRight);
    let newOrder = [...currentOrder];

    if (newState === 'none') {
      // UNPINNING: Restore to original position
      const savedPosition = originalPositions.get(column);

      // Get current pinned and unpinned columns
      const pinnedCols = currentOrder.filter(col => pinnedSet.has(col) && col !== column);
      const unpinnedCols = currentOrder.filter(col => !pinnedSet.has(col) && col !== column);

      // Restore to saved position if available, otherwise add after pinned area
      if (savedPosition !== undefined && savedPosition <= unpinnedCols.length) {
        unpinnedCols.splice(savedPosition, 0, column);
      } else {
        unpinnedCols.unshift(column); // Add at beginning of unpinned if no saved position
      }

      newOrder = [...pinnedCols, ...unpinnedCols];

      // Remove from originalPositions
      setOriginalPositions(prev => {
        const next = new Map(prev);
        next.delete(column);
        return next;
      });

      // Step 1: Update pinned state immediately (triggers visual separation)
      newPinnedLeft.delete(column);
      newPinnedRight.delete(column);

      isInternalChangeRef.current = true;
      setLocalPinnedLeft(newPinnedLeft);
      setLocalPinnedRight(newPinnedRight);
      onPinnedChange(newPinnedLeft, newPinnedRight);

      // Step 2: Delay array reordering for smooth animation (400ms)
      setTimeout(() => {
        isInternalChangeRef.current = true;
        if (isHyperparam) {
          setHyperparamOrder(newOrder);
        } else {
          setMetricOrder(newOrder);
        }
        if (onColumnOrderChange) {
          onColumnOrderChange(
            isHyperparam ? newOrder : hyperparamOrder,
            isHyperparam ? metricOrder : newOrder
          );
        }
      }, 400);

    } else {
      // PINNING: Save current position and move to pinned area

      // Get unpinned columns (before pinning this column)
      const unpinnedCols = currentOrder.filter(col => !pinnedSet.has(col));
      const positionInUnpinned = unpinnedCols.indexOf(column);

      // Save original position
      if (positionInUnpinned !== -1) {
        setOriginalPositions(prev => {
          const next = new Map(prev);
          next.set(column, positionInUnpinned);
          return next;
        });
      }

      // Add to target pinned set
      if (newState === 'left' && isHyperparam) {
        newPinnedLeft.add(column);
      } else if (newState === 'right' && !isHyperparam) {
        newPinnedRight.add(column);
      }

      // Calculate new order (but don't apply yet)
      // New pinned column goes to the END of pinned section (후입)
      // Include alwaysVisibleColumns as pinned
      const existingPinnedCols = currentOrder.filter(col =>
        pinnedSet.has(col) || alwaysVisibleColumns.includes(col)
      );
      const unpinnedRemaining = currentOrder.filter(col =>
        !pinnedSet.has(col) && !alwaysVisibleColumns.includes(col) && col !== column
      );
      newOrder = [...existingPinnedCols, column, ...unpinnedRemaining];

      // Step 1: Update pinned state immediately (triggers visual separation)
      isInternalChangeRef.current = true;
      setLocalPinnedLeft(newPinnedLeft);
      setLocalPinnedRight(newPinnedRight);
      onPinnedChange(newPinnedLeft, newPinnedRight);

      // Step 2: Delay array reordering for smooth animation (400ms)
      setTimeout(() => {
        isInternalChangeRef.current = true;
        if (isHyperparam) {
          setHyperparamOrder(newOrder);
        } else {
          setMetricOrder(newOrder);
        }
        if (onColumnOrderChange) {
          onColumnOrderChange(
            isHyperparam ? newOrder : hyperparamOrder,
            isHyperparam ? metricOrder : newOrder
          );
        }
      }, 400);
    }
  };

  // Bulk actions
  const handleShowAll = () => {
    const isHyperparam = showMode === 'hyperparams';

    if (isHyperparam) {
      const newVisible = new Set(hyperparamColumns);
      setLocalVisibleHyperparams(newVisible);
      onVisibilityChange(newVisible, localVisibleMetrics);
    } else {
      const newVisible = new Set(metricColumns);
      setLocalVisibleMetrics(newVisible);
      onVisibilityChange(localVisibleHyperparams, newVisible);
    }
  };

  const handleHideAll = () => {
    const isHyperparam = showMode === 'hyperparams';

    if (isHyperparam) {
      setLocalVisibleHyperparams(new Set());
      onVisibilityChange(new Set(), localVisibleMetrics);
    } else {
      setLocalVisibleMetrics(new Set());
      onVisibilityChange(localVisibleHyperparams, new Set());
    }
  };

  const handleResetPins = () => {
    const isHyperparam = showMode === 'hyperparams';

    // Mark as internal change
    isInternalChangeRef.current = true;

    if (isHyperparam) {
      // Reset pins only (keep visibility and order)
      const newPinnedLeft = new Set<string>();
      setLocalPinnedLeft(newPinnedLeft);
      onPinnedChange(newPinnedLeft, localPinnedRight);
    } else {
      // Reset metric pins
      const newPinnedRight = new Set<string>();
      setLocalPinnedRight(newPinnedRight);
      onPinnedChange(localPinnedLeft, newPinnedRight);
    }
  };

  // Toggle pin feature on/off
  const handleTogglePinning = () => {
    const newEnabled = !isPinningEnabled;
    setIsPinningEnabled(newEnabled);

    // Mark as internal change
    isInternalChangeRef.current = true;

    // Notify parent of pinning enabled state change
    if (onPinningEnabledChange) {
      onPinningEnabledChange(newEnabled);
    }

    // Apply or clear pins immediately
    if (newEnabled) {
      onPinnedChange(localPinnedLeft, localPinnedRight);
    } else {
      onPinnedChange(new Set(), new Set());
    }
  };

  return (
    <DropdownMenu>
      <div className="inline-flex rounded-md shadow-sm" role="group">
        {/* Left part: Toggle pin feature on/off */}
        <Button
          variant={isPinningEnabled ? "default" : "outline"}
          size="sm"
          className="rounded-r-none border-r-0 h-6 text-[10px] px-2"
          onClick={handleTogglePinning}
        >
          {isPinningEnabled ? (
            <Pin className="h-3 w-3 mr-0.5" />
          ) : (
            <PinOff className="h-3 w-3 mr-0.5" />
          )}
          Pin
        </Button>

        {/* Right part: Chevron to open dropdown */}
        <DropdownMenuTrigger asChild>
          <Button
            variant={isPinningEnabled ? "default" : "outline"}
            size="sm"
            className="rounded-l-none px-1 h-6"
          >
            <ChevronDown className="h-3 w-3" />
          </Button>
        </DropdownMenuTrigger>
      </div>

      <DropdownMenuContent className="w-[360px]" align="end">
        <DropdownMenuLabel>Column Management</DropdownMenuLabel>
        <DropdownMenuSeparator />

        {/* Search Bar */}
        <div className="p-2">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search columns..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-7 text-xs pl-7 pr-7"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        </div>

        {/* Statistics */}
        <div className="px-2 pb-2 flex items-center gap-2 text-[10px] text-muted-foreground">
          <div className="flex items-center gap-1">
            <Eye className="h-3 w-3" />
            {stats.visibleCount} visible
          </div>
          <div className="flex items-center gap-1">
            <Pin className="h-3 w-3" />
            {stats.pinnedCount} pinned
          </div>
        </div>

        {/* Category Toggle */}
        <div className="px-2 pb-2">
          <div className="inline-flex rounded-md shadow-sm w-full" role="group">
            <button
              onClick={() => setShowMode('hyperparams')}
              className={`flex-1 px-2 py-1 text-[11px] font-medium border rounded-l-md _transition-colors ${
                showMode === 'hyperparams'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background text-foreground border-input hover:bg-muted'
              }`}
            >
              Hyperparams
              <Badge
                variant={showMode === 'hyperparams' ? 'secondary' : 'outline'}
                className="ml-1 h-3.5 px-1 text-[9px]"
              >
                {stats.hyperparamsVisibleCount}/{stats.hyperparamsTotalCount}
              </Badge>
            </button>
            <button
              onClick={() => setShowMode('metrics')}
              className={`flex-1 px-2 py-1 text-[11px] font-medium border-t border-r border-b rounded-r-md _transition-colors ${
                showMode === 'metrics'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background text-foreground border-input hover:bg-muted'
              }`}
            >
              Metrics
              <Badge
                variant={showMode === 'metrics' ? 'secondary' : 'outline'}
                className="ml-1 h-3.5 px-1 text-[9px]"
              >
                {stats.metricsVisibleCount}/{stats.metricsTotalCount}
              </Badge>
            </button>
          </div>
        </div>

        <DropdownMenuSeparator />

        {/* Column List */}
        <ScrollArea className="h-80 px-2">
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            modifiers={[restrictToVerticalAxis]}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
          >
            {showMode === 'hyperparams' ? (
              <SortableContext
                items={filteredHyperparams}
                strategy={verticalListSortingStrategy}
              >
                <div className="py-2">
                  {filteredHyperparams.length === 0 ? (
                    <div className="text-xs text-muted-foreground px-2 py-8 text-center">
                      {searchQuery ? 'No columns match your search' : 'No hyperparameters available'}
                    </div>
                  ) : (
                    <div className="space-y-0">
                      {pinnedHyperparams.length > 0 && (
                        <div className="px-2 py-1 mb-1 bg-blue-50 dark:bg-blue-950/20 rounded flex items-center gap-1.5">
                          <Pin className="h-3 w-3 text-blue-600 dark:text-blue-400" />
                          <span className="text-[10px] font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wide">
                            Pinned ({pinnedHyperparams.length})
                          </span>
                        </div>
                      )}

                      {filteredHyperparams.map((param, index) => {
                        const isAlwaysVisible = alwaysVisibleColumns.includes(param);
                        const isVisible = isAlwaysVisible || localVisibleHyperparams.has(param);
                        const pinState = getPinState(param);
                        // Always-visible columns (Run ID, Status, Tags) are always pinned
                        const isPinned = localPinnedLeft.has(param) || isAlwaysVisible;

                        // Show unpinned header before first unpinned item
                        const prevParam = index > 0 ? filteredHyperparams[index - 1] : null;
                        const prevIsAlwaysVisible = prevParam ? alwaysVisibleColumns.includes(prevParam) : false;
                        const prevIsPinned = prevParam ? (localPinnedLeft.has(prevParam) || prevIsAlwaysVisible) : null;
                        const showUnpinnedHeader = !isPinned && prevIsPinned === true;

                        return (
                          <React.Fragment key={param}>
                            {showUnpinnedHeader && (
                              <div className="px-2 py-1 mt-3 mb-1 bg-muted/50 rounded flex items-center gap-1.5">
                                <PinOff className="h-3 w-3 text-muted-foreground" />
                                <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">
                                  Unpinned ({unpinnedHyperparams.length})
                                </span>
                              </div>
                            )}
                            <SortableItem
                              id={param}
                              isAlwaysVisible={isAlwaysVisible}
                              isVisible={isVisible}
                              pinState={pinState}
                              onVisibilityToggle={() => toggleVisibility(param)}
                              onPinToggle={(state) => togglePin(param, state)}
                              isHyperparam={true}
                              isHighlighted={highlightedColumn === param}
                              isAnyDragging={!!activeId}
                            />
                          </React.Fragment>
                        );
                      })}
                    </div>
                  )}
                </div>
              </SortableContext>
            ) : (
              <SortableContext
                items={filteredMetrics}
                strategy={verticalListSortingStrategy}
              >
                <div className="py-2">
                  {filteredMetrics.length === 0 ? (
                    <div className="text-xs text-muted-foreground px-2 py-8 text-center">
                      {searchQuery ? 'No columns match your search' : 'No metrics available'}
                    </div>
                  ) : (
                    <div className="space-y-0">
                      {pinnedMetrics.length > 0 && (
                        <div className="px-2 py-1 mb-1 bg-green-50 dark:bg-green-950/20 rounded flex items-center gap-1.5">
                          <Pin className="h-3 w-3 text-green-600 dark:text-green-400" />
                          <span className="text-[10px] font-semibold text-green-600 dark:text-green-400 uppercase tracking-wide">
                            Pinned ({pinnedMetrics.length})
                          </span>
                        </div>
                      )}

                      {filteredMetrics.map((metric, index) => {
                        const isVisible = localVisibleMetrics.has(metric);
                        const pinState = getPinState(metric);
                        const isPinned = localPinnedRight.has(metric);

                        // Show unpinned header before first unpinned item
                        const prevMetric = index > 0 ? filteredMetrics[index - 1] : null;
                        const prevIsPinned = prevMetric ? localPinnedRight.has(prevMetric) : null;
                        const showUnpinnedHeader = !isPinned && prevIsPinned === true;

                        return (
                          <React.Fragment key={metric}>
                            {showUnpinnedHeader && (
                              <div className="px-2 py-1 mt-3 mb-1 bg-muted/50 rounded flex items-center gap-1.5">
                                <PinOff className="h-3 w-3 text-muted-foreground" />
                                <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">
                                  Unpinned ({unpinnedMetrics.length})
                                </span>
                              </div>
                            )}
                            <SortableItem
                              id={metric}
                              isVisible={isVisible}
                              pinState={pinState}
                              onVisibilityToggle={() => toggleVisibility(metric)}
                              onPinToggle={(state) => togglePin(metric, state)}
                              isHyperparam={false}
                              isHighlighted={highlightedColumn === metric}
                              isAnyDragging={!!activeId}
                            />
                          </React.Fragment>
                        );
                      })}
                    </div>
                  )}
                </div>
              </SortableContext>
            )}
          </DndContext>
        </ScrollArea>

        <DropdownMenuSeparator />

        {/* Bulk Actions */}
        <div className="p-2 flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 h-7 text-xs"
            onClick={handleShowAll}
          >
            <Eye className="h-3 w-3 mr-1" />
            Show All
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="flex-1 h-7 text-xs"
            onClick={handleHideAll}
          >
            <EyeOff className="h-3 w-3 mr-1" />
            Hide All
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="flex-1 h-7 text-xs"
            onClick={handleResetPins}
          >
            <PinOff className="h-3 w-3 mr-1" />
            Unpin All
          </Button>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
