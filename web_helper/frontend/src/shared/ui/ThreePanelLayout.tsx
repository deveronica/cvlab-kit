import { ReactNode, useCallback, useEffect, useRef } from 'react';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/shared/ui/resizable';
import { usePanelPersistence } from './panelPersistence';
import { cn } from '@/shared/lib/utils';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { ImperativePanelHandle } from 'react-resizable-panels';

interface ThreePanelLayoutProps {
  layoutKey: string;
  leftPanel?: ReactNode;
  centerPanel: ReactNode;
  rightPanel?: ReactNode;
  leftMinSize?: number;
  leftMaxSize?: number;
  rightMinSize?: number;
  rightMaxSize?: number;
  leftShouldCollapse?: boolean;
  rightShouldCollapse?: boolean;
  /** Compact sizer width in px — expand opens to this minimum. */
  leftMinExpandPx?: number;
  rightMinExpandPx?: number;
  className?: string;
}

/**
 * ThreePanelLayout - Professional Card-on-Canvas Layout
 */
export function ThreePanelLayout({
  layoutKey,
  leftPanel,
  centerPanel,
  rightPanel,
  leftMinSize = 8,
  leftMaxSize = 40,
  rightMinSize = 12,
  rightMaxSize = 50,
  leftShouldCollapse,
  rightShouldCollapse,
  leftMinExpandPx,
  rightMinExpandPx,
  className,
}: ThreePanelLayoutProps) {
  const {
    leftCollapsed,
    rightCollapsed,
    setLeftCollapsed,
    setRightCollapsed,
  } = usePanelPersistence(layoutKey);

  const leftPanelRef = useRef<ImperativePanelHandle>(null);
  const rightPanelRef = useRef<ImperativePanelHandle>(null);
  const layoutRef = useRef<HTMLDivElement>(null);

  // Guard: skip auto-collapse right after manual expand
  const leftExpandGuard = useRef(false);
  const rightExpandGuard = useRef(false);

  // Auto-collapse when even icon-only can't fit (with guard)
  useEffect(() => {
    if (leftShouldCollapse && !leftCollapsed && !leftExpandGuard.current) {
      leftPanelRef.current?.collapse();
    }
  }, [leftShouldCollapse, leftCollapsed]);

  useEffect(() => {
    if (rightShouldCollapse && !rightCollapsed && !rightExpandGuard.current) {
      rightPanelRef.current?.collapse();
    }
  }, [rightShouldCollapse, rightCollapsed]);

  /** Expand to minimum comfortable width (icon-only + padding). */
  const handleLeftExpand = useCallback(() => {
    leftExpandGuard.current = true;
    if (leftMinExpandPx && layoutRef.current) {
      const totalW = layoutRef.current.getBoundingClientRect().width;
      const pct = Math.ceil((leftMinExpandPx / totalW) * 100) + 1;
      leftPanelRef.current?.resize(Math.max(pct, leftMinSize));
    } else {
      leftPanelRef.current?.expand();
    }
    // Clear guard after resize settles and ResizeObserver fires
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { leftExpandGuard.current = false; });
    });
  }, [leftMinExpandPx, leftMinSize]);

  const handleRightExpand = useCallback(() => {
    rightExpandGuard.current = true;
    if (rightMinExpandPx && layoutRef.current) {
      const totalW = layoutRef.current.getBoundingClientRect().width;
      const pct = Math.ceil((rightMinExpandPx / totalW) * 100) + 1;
      rightPanelRef.current?.resize(Math.max(pct, rightMinSize));
    } else {
      rightPanelRef.current?.expand();
    }
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { rightExpandGuard.current = false; });
    });
  }, [rightMinExpandPx, rightMinSize]);

  return (
    <div ref={layoutRef} className={cn('h-full w-full bg-muted/20 overflow-hidden relative', className)}>
      {/* Floating Expand Triggers (Minimalist Design) */}
      {leftCollapsed && (
        <button
          onClick={handleLeftExpand}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-[100] h-12 w-3 bg-primary/10 hover:bg-primary/20 border border-l-0 border-primary/20 rounded-r-md flex items-center justify-center transition-all shadow-md"
        >
          <ChevronRight className="w-3 h-3 text-primary" />
        </button>
      )}

      {rightCollapsed && (
        <button
          onClick={handleRightExpand}
          className="absolute right-0 top-1/2 -translate-y-1/2 z-[100] h-12 w-3 bg-primary/10 hover:bg-primary/20 border border-r-0 border-primary/20 rounded-l-md flex items-center justify-center transition-all shadow-md"
        >
          <ChevronLeft className="w-3 h-3 text-primary" />
        </button>
      )}

      <ResizablePanelGroup
        direction="horizontal"
        className="h-full w-full"
        autoSaveId={`group-${layoutKey}`}
      >
        {/* 1. Left Panel: Explorer */}
        {leftPanel && (
          <>
            <ResizablePanel
              ref={leftPanelRef}
              id="left"
              order={1}
              defaultSize={20}
              minSize={leftMinSize}
              maxSize={leftMaxSize}
              collapsible={true}
              collapsedSize={0}
              onCollapse={() => setLeftCollapsed(true)}
              onExpand={() => setLeftCollapsed(false)}
              className={cn(
                "p-2",
                leftCollapsed && "opacity-0 pointer-events-none p-0"
              )}
            >
              <div className="h-full w-full bg-background border border-border shadow-sm rounded-lg overflow-hidden">
                {leftPanel}
              </div>
            </ResizablePanel>
            <ResizableHandle className="w-px group flex items-center justify-center bg-transparent transition-all cursor-col-resize hover:bg-primary/20 -mx-1 z-50">
              <div className="h-8 w-1 rounded-full bg-border group-hover:bg-primary/50 transition-colors" />
            </ResizableHandle>
          </>
        )}

        {/* 2. Middle Panel: The Main Stage */}
        <ResizablePanel
          id="center"
          order={2}
          minSize={10}
          className="p-2"
        >
          <div className="h-full w-full flex flex-col bg-background border border-border shadow-sm rounded-lg overflow-hidden">
            {centerPanel}
          </div>
        </ResizablePanel>

        {/* 3. Right Panel: Inspector & Code */}
        {rightPanel && (
          <>
            <ResizableHandle className="w-px group flex items-center justify-center bg-transparent transition-all cursor-col-resize hover:bg-primary/20 -mx-1 z-50">
              <div className="h-8 w-1 rounded-full bg-border group-hover:bg-primary/50 transition-colors" />
            </ResizableHandle>
            <ResizablePanel
              ref={rightPanelRef}
              id="right"
              order={3}
              defaultSize={25}
              minSize={rightMinSize}
              maxSize={rightMaxSize}
              collapsible={true}
              collapsedSize={0}
              onCollapse={() => setRightCollapsed(true)}
              onExpand={() => setRightCollapsed(false)}
              className={cn(
                "p-2",
                rightCollapsed && "opacity-0 pointer-events-none p-0"
              )}
            >
              <div className="h-full w-full bg-background border border-border shadow-sm rounded-lg overflow-hidden">
                {rightPanel}
              </div>
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>
    </div>
  );
}
