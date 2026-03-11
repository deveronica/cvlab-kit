/**
 * SplitPane - Resizable split view component
 *
 * Features:
 * - Horizontal or vertical split
 * - Drag-to-resize with min/max constraints
 * - Double-click to reset to default
 * - Keyboard accessible
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { cn } from '@/shared/lib/utils';

interface SplitPaneProps {
  direction?: 'horizontal' | 'vertical';
  defaultRatio?: number; // 0-1, left/top pane ratio
  minRatio?: number;
  maxRatio?: number;
  left?: React.ReactNode;
  right?: React.ReactNode;
  top?: React.ReactNode;
  bottom?: React.ReactNode;
  className?: string;
  resizerClassName?: string;
  onRatioChange?: (ratio: number) => void;
}

export function SplitPane({
  direction = 'horizontal',
  defaultRatio = 0.4,
  minRatio = 0.2,
  maxRatio = 0.8,
  left,
  right,
  top,
  bottom,
  className,
  resizerClassName,
  onRatioChange,
}: SplitPaneProps) {
  const [ratio, setRatio] = useState(defaultRatio);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const isHorizontal = direction === 'horizontal';
  const first = isHorizontal ? left : top;
  const second = isHorizontal ? right : bottom;

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);
    },
    []
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      let newRatio: number;

      if (isHorizontal) {
        newRatio = (e.clientX - rect.left) / rect.width;
      } else {
        newRatio = (e.clientY - rect.top) / rect.height;
      }

      // Clamp to min/max
      newRatio = Math.max(minRatio, Math.min(maxRatio, newRatio));
      setRatio(newRatio);
      onRatioChange?.(newRatio);
    },
    [isDragging, isHorizontal, minRatio, maxRatio, onRatioChange]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDoubleClick = useCallback(() => {
    setRatio(defaultRatio);
    onRatioChange?.(defaultRatio);
  }, [defaultRatio, onRatioChange]);

  // Global mouse events for dragging
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = isHorizontal ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, handleMouseMove, handleMouseUp, isHorizontal]);

  const firstSize = `${ratio * 100}%`;
  const secondSize = `${(1 - ratio) * 100}%`;

  return (
    <div
      ref={containerRef}
      className={cn(
        'flex h-full w-full overflow-hidden',
        isHorizontal ? 'flex-row' : 'flex-col',
        className
      )}
    >
      {/* First Pane */}
      <div
        className="overflow-hidden"
        style={{
          [isHorizontal ? 'width' : 'height']: firstSize,
          flexShrink: 0,
        }}
      >
        {first}
      </div>

      {/* Resizer */}
      <div
        className={cn(
          'flex-shrink-0 bg-border hover:bg-primary/50 transition-colors duration-150',
          isHorizontal
            ? 'w-1 cursor-col-resize hover:w-1.5'
            : 'h-1 cursor-row-resize hover:h-1.5',
          isDragging && 'bg-primary',
          resizerClassName
        )}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        role="separator"
        aria-orientation={isHorizontal ? 'vertical' : 'horizontal'}
        aria-valuenow={Math.round(ratio * 100)}
        aria-valuemin={Math.round(minRatio * 100)}
        aria-valuemax={Math.round(maxRatio * 100)}
        tabIndex={0}
        onKeyDown={(e) => {
          const step = 0.05;
          if (
            (isHorizontal && e.key === 'ArrowLeft') ||
            (!isHorizontal && e.key === 'ArrowUp')
          ) {
            e.preventDefault();
            const newRatio = Math.max(minRatio, ratio - step);
            setRatio(newRatio);
            onRatioChange?.(newRatio);
          } else if (
            (isHorizontal && e.key === 'ArrowRight') ||
            (!isHorizontal && e.key === 'ArrowDown')
          ) {
            e.preventDefault();
            const newRatio = Math.min(maxRatio, ratio + step);
            setRatio(newRatio);
            onRatioChange?.(newRatio);
          } else if (e.key === 'Home') {
            e.preventDefault();
            setRatio(minRatio);
            onRatioChange?.(minRatio);
          } else if (e.key === 'End') {
            e.preventDefault();
            setRatio(maxRatio);
            onRatioChange?.(maxRatio);
          }
        }}
      />

      {/* Second Pane */}
      <div
        className="overflow-hidden flex-1"
        style={{
          [isHorizontal ? 'width' : 'height']: secondSize,
        }}
      >
        {second}
      </div>
    </div>
  );
}

/**
 * TripleSplitPane - Three-way split (sidebar + main + panel)
 */
interface TripleSplitPaneProps {
  sidebar: React.ReactNode;
  main: React.ReactNode;
  panel?: React.ReactNode;
  sidebarWidth?: number; // Fixed width in px
  panelHeight?: number; // Fixed height in px
  panelOpen?: boolean;
  className?: string;
}

export function TripleSplitPane({
  sidebar,
  main,
  panel,
  sidebarWidth = 220,
  panelHeight = 200,
  panelOpen = true,
  className,
}: TripleSplitPaneProps) {
  return (
    <div className={cn('flex h-full w-full overflow-hidden', className)}>
      {/* Sidebar (fixed width) */}
      <div
        className="flex-shrink-0 border-r border-border overflow-hidden"
        style={{ width: sidebarWidth }}
      >
        {sidebar}
      </div>

      {/* Main + Panel */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Main content */}
        <div className="flex-1 overflow-hidden">{main}</div>

        {/* Bottom Panel (collapsible) */}
        {panel && panelOpen && (
          <>
            <div className="h-px bg-border flex-shrink-0" />
            <div
              className="flex-shrink-0 overflow-hidden"
              style={{ height: panelHeight }}
            >
              {panel}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
