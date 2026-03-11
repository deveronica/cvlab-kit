import React from "react";
/**
 * Column Pinning Menu
 * Dropdown menu for managing column pinning with Button Group style
 */

import { useState, useEffect } from 'react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
  DragStartEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
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
import { Pin, ChevronDown, ArrowLeftToLine, ArrowRightToLine, GripVertical, Lock } from 'lucide-react';
import { ScrollArea } from './scroll-area';

interface ColumnPinningMenuProps {
  hyperparamColumns: string[];
  metricColumns: string[];
  pinnedLeftHyperparams: Set<string>;
  pinnedRightMetrics: Set<string>;
  onPinnedChange: (pinnedLeft: Set<string>, pinnedRight: Set<string>) => void;
}

// Sortable Item Component
interface SortablePinItemProps {
  id: string;
  isPinned: boolean;
  onToggle: () => void;
  isHyperparam: boolean;
}

function SortablePinItem({ id, isPinned, onToggle, isHyperparam }: SortablePinItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-muted group"
    >
      <div {...attributes} {...listeners} className="cursor-grab active:cursor-grabbing">
        <GripVertical className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
      </div>
      <span className="text-sm flex-1 truncate">{id}</span>
      <button
        onClick={onToggle}
        className={`px-3 py-1.5 text-xs font-medium border rounded-md transition-colors flex items-center gap-1.5 ${
          isPinned
            ? isHyperparam
              ? 'bg-blue-500 text-white border-blue-500 hover:bg-blue-600'
              : 'bg-green-500 text-white border-green-500 hover:bg-green-600'
            : 'bg-background text-muted-foreground border-input hover:bg-muted hover:text-foreground'
        }`}
        title={isPinned ? 'Click to unpin' : isHyperparam ? 'Click to pin left' : 'Click to pin right'}
      >
        {isPinned ? (
          <>
            <Pin className="h-3 w-3" />
            <span>Pinned</span>
          </>
        ) : (
          <>
            {isHyperparam ? <ArrowLeftToLine className="h-3 w-3" /> : <ArrowRightToLine className="h-3 w-3" />}
            <span>Pin</span>
          </>
        )}
      </button>
    </div>
  );
}

export function ColumnPinningMenu({
  hyperparamColumns,
  metricColumns,
  pinnedLeftHyperparams,
  pinnedRightMetrics,
  onPinnedChange,
}: ColumnPinningMenuProps) {
  const [showMode, setShowMode] = useState<'hyperparams' | 'metrics'>('hyperparams');
  const [_activeId, setActiveId] = useState<string | null>(null);

  // User's selections (what they picked in the dropdown) - source of truth
  const [userPinnedLeft, setUserPinnedLeft] = useState<Set<string>>(new Set(pinnedLeftHyperparams));
  const [userPinnedRight, setUserPinnedRight] = useState<Set<string>>(new Set(pinnedRightMetrics));

  // Pin feature on/off state (whether to apply the selections)
  // Enabled if user has made any selections
  const [isPinningEnabled, setIsPinningEnabled] = useState<boolean>(
    pinnedLeftHyperparams.size > 0 || pinnedRightMetrics.size > 0
  );

  // Fixed columns that cannot be unpinned (always visible on left)
  // Note: Checkbox column is not included here as it's a UI control, not a data column
  const fixedColumns = ['Run ID', 'Status'];

  // Column order management
  const [hyperparamOrder, setHyperparamOrder] = useState<string[]>(hyperparamColumns);
  const [metricOrder, setMetricOrder] = useState<string[]>(metricColumns);

  // Update order when props change
  useEffect(() => {
    setHyperparamOrder(hyperparamColumns);
  }, [hyperparamColumns]);

  useEffect(() => {
    setMetricOrder(metricColumns);
  }, [metricColumns]);

  // Sync userPinned* with props when they change (e.g., SavedView loaded)
  // Only update if props actually changed (prevent infinite loop)
  useEffect(() => {
    const propsLeftSet = new Set(pinnedLeftHyperparams);
    const propsRightSet = new Set(pinnedRightMetrics);

    const leftEqual = userPinnedLeft.size === propsLeftSet.size &&
      [...userPinnedLeft].every(x => propsLeftSet.has(x));
    const rightEqual = userPinnedRight.size === propsRightSet.size &&
      [...userPinnedRight].every(x => propsRightSet.has(x));

    if (!leftEqual || !rightEqual) {
      setUserPinnedLeft(propsLeftSet);
      setUserPinnedRight(propsRightSet);

      // Update isPinningEnabled based on whether there are selections
      const hasSelections = propsLeftSet.size > 0 || propsRightSet.size > 0;
      setIsPinningEnabled(hasSelections);
    }
  }, [pinnedLeftHyperparams, pinnedRightMetrics, userPinnedLeft, userPinnedRight]);

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
    }

    setActiveId(null);
  };

  const togglePinLeft = (column: string) => {
    const newPinned = new Set(userPinnedLeft);
    if (newPinned.has(column)) {
      newPinned.delete(column);
    } else {
      newPinned.add(column);
    }
    setUserPinnedLeft(newPinned);

    // Notify parent immediately if pinning is enabled
    if (isPinningEnabled) {
      onPinnedChange(newPinned, userPinnedRight);
    }
  };

  const togglePinRight = (column: string) => {
    const newPinned = new Set(userPinnedRight);
    if (newPinned.has(column)) {
      newPinned.delete(column);
    } else {
      newPinned.add(column);
    }
    setUserPinnedRight(newPinned);

    // Notify parent immediately if pinning is enabled
    if (isPinningEnabled) {
      onPinnedChange(userPinnedLeft, newPinned);
    }
  };

  // Toggle pin feature on/off
  const handleTogglePinning = () => {
    const newEnabled = !isPinningEnabled;
    setIsPinningEnabled(newEnabled);

    // Apply or clear pins immediately
    if (newEnabled) {
      onPinnedChange(userPinnedLeft, userPinnedRight);
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
          <Pin className="h-3 w-3 mr-0.5" />
          {isPinningEnabled ? 'Unpin' : 'Pin'}
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
        <DropdownMenuContent className="w-80" align="end">
        <DropdownMenuLabel>Column Pinning</DropdownMenuLabel>
        <DropdownMenuSeparator />

        {/* Category Toggle */}
        <div className="p-2">
          <div className="inline-flex rounded-md shadow-sm" role="group">
            <button
              onClick={() => setShowMode('hyperparams')}
              className={`px-4 py-2 text-xs font-medium border rounded-l-md transition-colors ${
                showMode === 'hyperparams'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background text-foreground border-input hover:bg-muted'
              }`}
            >
              Hyperparams
            </button>
            <button
              onClick={() => setShowMode('metrics')}
              className={`px-4 py-2 text-xs font-medium border-t border-r border-b rounded-r-md transition-colors ${
                showMode === 'metrics'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background text-foreground border-input hover:bg-muted'
              }`}
            >
              Metrics
            </button>
          </div>
        </div>

        <DropdownMenuSeparator />

        {/* Column List */}
        <ScrollArea className="h-64 px-2">
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            modifiers={[restrictToVerticalAxis]}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
          >
            {showMode === 'hyperparams' ? (
              <div className="space-y-1 py-2">
                {/* Fixed Columns Section */}
                <div className="mb-3">
                  <div className="text-xs font-medium text-muted-foreground px-2 mb-1 flex items-center gap-1">
                    <Lock className="h-3 w-3" />
                    Always Visible
                  </div>
                  {fixedColumns.map(col => (
                    <div
                      key={col}
                      className="flex items-center justify-between px-2 py-1.5 rounded bg-muted/50 opacity-60"
                    >
                      <span className="text-sm flex-1 truncate">{col}</span>
                      <Badge variant="outline" className="text-xs">
                        Fixed
                      </Badge>
                    </div>
                  ))}
                </div>

                <DropdownMenuSeparator />

                {/* Draggable Hyperparam Columns */}
                <SortableContext
                  items={hyperparamOrder}
                  strategy={verticalListSortingStrategy}
                >
                  {hyperparamOrder.length === 0 ? (
                    <div className="text-xs text-muted-foreground px-2 py-4 text-center">
                      No hyperparameters available
                    </div>
                  ) : (
                    hyperparamOrder.map(param => (
                      <SortablePinItem
                        key={param}
                        id={param}
                        isPinned={userPinnedLeft.has(param)}
                        onToggle={() => togglePinLeft(param)}
                        isHyperparam={true}
                      />
                    ))
                  )}
                </SortableContext>
              </div>
            ) : (
              <div className="space-y-1 py-2">
                {/* Draggable Metric Columns */}
                <SortableContext
                  items={metricOrder}
                  strategy={verticalListSortingStrategy}
                >
                  {metricOrder.length === 0 ? (
                    <div className="text-xs text-muted-foreground px-2 py-4 text-center">
                      No metrics available
                    </div>
                  ) : (
                    metricOrder.map(metric => (
                      <SortablePinItem
                        key={metric}
                        id={metric}
                        isPinned={userPinnedRight.has(metric)}
                        onToggle={() => togglePinRight(metric)}
                        isHyperparam={false}
                      />
                    ))
                  )}
                </SortableContext>
              </div>
            )}
          </DndContext>
        </ScrollArea>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
