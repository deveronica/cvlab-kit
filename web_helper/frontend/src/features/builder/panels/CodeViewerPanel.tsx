import React, { useRef, useEffect } from 'react';
import { Card } from '@/shared/ui/card';
import { Button } from '@/shared/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/shared/ui/tooltip';
import CodeMirror, { ReactCodeMirrorRef } from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import { Code2, FileCode2, PanelLeftClose, PanelRightClose } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { useBuilderStore } from '@/shared/model/builderStore';

export function CodeViewerPanel() {
  const {
    leftPanelOpen,
    toggleLeftPanel,
    rightPanelOpen,
    toggleRightPanel,
    selectedComponent,
    activeComponentVersion,
  } = useBuilderStore();
  
  const editorRef = useRef<ReactCodeMirrorRef>(null);

  // When activeComponentVersion changes, ensure editor content is updated and scrolled to top
  useEffect(() => {
    if (editorRef.current?.view && activeComponentVersion?.content) {
      // CodeMirror automatically updates its content when the 'value' prop changes.
      // We might want to reset scroll or cursor position here if needed.
      editorRef.current.view.dispatch({
        selection: { anchor: 0, head: 0 }, // Set cursor to start
        scrollIntoView: true, // Scroll to cursor
      });
    }
  }, [activeComponentVersion]);

  return (
    <Card className="flex-1 flex flex-col min-w-0 overflow-hidden">
      <div className="px-2 py-1.5 border-b border-border flex items-center gap-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={leftPanelOpen ? "secondary" : "ghost"}
              size="icon"
              className="h-7 w-7 flex-shrink-0"
              onClick={toggleLeftPanel}
            >
              <PanelLeftClose className={cn(
                "h-4 w-4 transition-transform duration-200",
                !leftPanelOpen && "rotate-180"
              )} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Explorer</TooltipContent>
        </Tooltip>

        <div className="flex-1 flex items-center justify-center gap-2 min-w-0">
          {selectedComponent ? (
            <>
              <FileCode2 className="h-4 w-4 text-blue-500 flex-shrink-0" />
              <span className="text-sm font-medium truncate">{selectedComponent.name}.py</span>
              {/* Removed Badge for category for now, can be re-added if space permits or needed */}
              {/* <Badge variant="secondary" className="text-xs">{selectedComponent.category}</Badge> */}
              {activeComponentVersion && (
                <>
                  <span className="text-muted-foreground">·</span>
                  <code className="text-xs text-muted-foreground">{activeComponentVersion.hash.slice(0, 8)}</code>
                </>
              )}
            </>
          ) : (
            <span className="text-sm text-muted-foreground">Select a component</span>
          )}
        </div>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={rightPanelOpen ? "secondary" : "ghost"}
              size="icon"
              className="h-7 w-7 flex-shrink-0"
              onClick={toggleRightPanel}
            >
              <PanelRightClose className={cn(
                "h-4 w-4 transition-transform duration-200",
                !rightPanelOpen && "rotate-180"
              )} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Outline & History</TooltipContent>
        </Tooltip>
      </div>

      <div className="flex-1 overflow-hidden relative">
        {activeComponentVersion?.content ? (
          <div className="absolute inset-0">
            <CodeMirror
              ref={editorRef}
              value={activeComponentVersion.content}
              height="100%"
              theme={vscodeDark}
              extensions={[python()]}
              readOnly
              basicSetup={{
                lineNumbers: true,
                foldGutter: true,
                highlightActiveLine: true,
                highlightActiveLineGutter: true,
              }}
              className="h-full text-[13px]"
            />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-3">
            <Code2 className="h-12 w-12 opacity-20" />
            <div className="text-center">
              <p className="text-sm font-medium">No component selected</p>
              <p className="text-xs mt-1">Select a component from the explorer</p>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
