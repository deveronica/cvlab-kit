import { ReactNode, useState, useEffect } from 'react';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/shared/ui/resizable';
import { cn } from '@/shared/lib/utils';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/shared/ui/tabs';
import { useSmartCompact } from '@/shared/lib/useSmartCompact';

export interface RightRailTab {
  value: string;
  label: string;
  icon?: React.ElementType;
  content: ReactNode;
}

interface RightRailProps {
  tabs: RightRailTab[];
  defaultTab?: string;
  bottomPanel?: ReactNode;
  bottomPanelHeight?: number; // percentage
  onBottomPanelResize?: (size: number) => void;
  onShouldCollapseChange?: (shouldCollapse: boolean) => void;
  onCompactWidthChange?: (width: number) => void;
  className?: string;
}

/**
 * RightRail - Unified IDE Style Tab Navigation
 */
export function RightRail({
  tabs,
  defaultTab,
  bottomPanel,
  bottomPanelHeight = 30,
  onBottomPanelResize,
  onShouldCollapseChange,
  onCompactWidthChange,
  className,
}: RightRailProps) {
  const { containerRef, fullSizerRef, compactSizerRef, isCompact, shouldCollapse, compactWidth } = useSmartCompact();
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.value);

  useEffect(() => {
    onShouldCollapseChange?.(shouldCollapse);
  }, [shouldCollapse, onShouldCollapseChange]);

  useEffect(() => {
    onCompactWidthChange?.(compactWidth);
  }, [compactWidth, onCompactWidthChange]);

  const content = (
    <div ref={containerRef} className="h-full flex flex-col relative">
      {/* Full sizer: icon + text (global px-2 + local px-3) */}
      <div
        ref={fullSizerRef}
        aria-hidden
        className="absolute top-0 left-0 pointer-events-none flex items-center gap-0 px-2"
        style={{ visibility: 'hidden', whiteSpace: 'nowrap' }}
      >
        {tabs.map((tab) => (
          <span key={tab.value} className="inline-flex items-center gap-1 text-[11px] font-bold px-3">
            {tab.icon && <tab.icon className="h-4 w-4" />}
            <span>{tab.label}</span>
          </span>
        ))}
      </div>
      {/* Compact sizer: icon-only (global px-2 + local px-3) */}
      <div
        ref={compactSizerRef}
        aria-hidden
        className="absolute top-0 left-0 pointer-events-none flex items-center gap-0 px-2"
        style={{ visibility: 'hidden', whiteSpace: 'nowrap' }}
      >
        {tabs.map((tab) => (
          <span key={tab.value} className="inline-flex items-center justify-center px-3">
            {tab.icon && <tab.icon className="h-4 w-4" />}
          </span>
        ))}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
        {/* Header - Unified Underline Style */}
        <div className="flex-shrink-0 px-2 border-b border-border/40 bg-muted/20">
          <TabsList className="h-11 w-full flex justify-center gap-0 bg-transparent p-0">
            {tabs.map((tab) => (
              <TabsTrigger
                key={tab.value}
                value={tab.value}
                className={cn(
                  "h-11 flex-1 rounded-none border-b-2 border-transparent text-[11px] font-bold transition-all px-3",
                  "data-[state=active]:border-primary data-[state=active]:text-primary data-[state=active]:bg-transparent shadow-none flex items-center justify-center gap-1"
                )}
              >
                {tab.icon && <tab.icon className="h-4 w-4 flex-shrink-0" />}
                {!isCompact && <span>{tab.label}</span>}
              </TabsTrigger>
            ))}
          </TabsList>
        </div>
        <div className="flex-1 overflow-hidden">
          {tabs.map((tab) => (
            <TabsContent
              key={tab.value}
              value={tab.value}
              className="h-full mt-0 overflow-auto data-[state=active]:flex data-[state=active]:flex-col"
            >
              {tab.content}
            </TabsContent>
          ))}
        </div>
      </Tabs>
    </div>
  );

  if (!bottomPanel) {
    return <div className={cn('h-full w-full bg-transparent', className)}>{content}</div>;
  }

  return (
    <div className={cn('h-full w-full bg-transparent', className)}>
      <ResizablePanelGroup direction="vertical" className="h-full">
        <ResizablePanel defaultSize={100 - bottomPanelHeight} minSize={20}>
          {content}
        </ResizablePanel>

        <ResizableHandle withHandle />

        <ResizablePanel
          defaultSize={bottomPanelHeight}
          minSize={10}
          onResize={onBottomPanelResize}
        >
          <div className="h-full w-full border-t border-border overflow-hidden">
            {bottomPanel}
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
