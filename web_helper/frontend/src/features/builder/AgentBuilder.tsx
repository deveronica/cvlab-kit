/**
 * AgentBuilder - Unified Professional IDE Workbench
 */
import { useState, useCallback, useMemo, useEffect } from 'react';
import { useSmartCompact } from '@/shared/lib/useSmartCompact';
import {
  Settings,
  Play,
  CheckCircle2,
  Code2,
  History,
  Info,
  Files,
  Package,
  ChevronRight,
  Monitor,
  Save,
  Edit3,
  AlertTriangle,
} from 'lucide-react';

import { AgentBuilderProvider, useAgentBuilder } from '@/entities/node-system/model/AgentBuilderContext';
import { BuilderEditPane } from '@/entities/node-system/ui';
import { ThreePanelLayout } from '@/shared/ui/ThreePanelLayout';
import { RightRail } from '@/shared/ui/RightRail';
import { InlineComponentPalette } from './ComponentPalette';
import { CodePane } from './CodePane';
import { PropertiesPane } from './PropertiesPane';
import { BuilderHistoryPanel } from './panels/BuilderHistoryPanel';
import { FileExplorer } from './FileExplorer';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/shared/ui/tabs';
import { Badge } from '@/shared/ui/badge';
import { Button } from '@/shared/ui/button';
import { Separator } from '@/shared/ui/separator';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/shared/ui/alert-dialog';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';

const TAB_METHODS = [
  { value: 'initialize', label: 'setup()', icon: Settings, phase: 'initialize' },
  { value: 'train_step', label: 'train_step()', icon: Play, phase: 'flow' },
  { value: 'val_step', label: 'val_step()', icon: CheckCircle2, phase: 'flow' },
] as const;

function AgentBuilderContent() {
  const {
    isEditingMode,
    setIsEditingMode,
    isDirty,
    selectedAgent,
    saveToCode,
    addNode,
    addControlFlowNode,
  } = useAgentBuilder();

  const [selectedMethod, setSelectedMethod] = useState('initialize');
  const [activeTab, setActiveTab] = useState("files");
  const [showCancelConfirm, setShowCancelConfirm] = useState(false);
  const { containerRef: leftContainerRef, fullSizerRef: leftFullSizerRef, compactSizerRef: leftCompactSizerRef, isCompact: leftCompact, shouldCollapse: leftShouldCollapse, compactWidth: leftCompactWidth } = useSmartCompact();
  const [rightShouldCollapse, setRightShouldCollapse] = useState(false);
  const [rightCompactWidth, setRightCompactWidth] = useState(0);
  const setNodeTab = useNodeStore(s => s.setTab);

  const currentOption = useMemo(() => TAB_METHODS.find((m) => m.value === selectedMethod), [selectedMethod]);
  const phase = currentOption?.phase || 'flow';

  useEffect(() => {
    setNodeTab(phase === 'initialize' ? 'execute' : 'builder');
  }, [phase, setNodeTab]);

  const handleCancelClick = useCallback(() => {
    if (isDirty) {
      setShowCancelConfirm(true);
    } else {
      setIsEditingMode(false);
    }
  }, [isDirty, setIsEditingMode]);

  const handleCancelEdit = useCallback(() => {
    setIsEditingMode(false);
    setShowCancelConfirm(false);
  }, [setIsEditingMode]);

  return (
    <div className="h-full flex flex-col bg-muted/20 select-none overflow-hidden p-0 gap-0">
      {/* 1. Global Header - Floating Card Style */}
      <div className="px-2 pt-2">
        <header className="h-12 bg-card border border-border shadow-sm flex items-center px-4 justify-between shrink-0 z-20 rounded-lg">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center border border-primary/20">
              <Monitor className="w-4 h-4 text-primary" />
            </div>
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-tight">
              <span className="text-muted-foreground opacity-50">cvlab-kit</span>
              <ChevronRight className="w-3 h-3 text-muted-foreground/30" />
              <span className="text-foreground">{selectedAgent?.name || "Ready to Work"}</span>
              <ChevronRight className="w-3 h-3 text-muted-foreground/30" />
              <Badge variant="secondary" className="h-5 px-2 text-[10px] font-mono tracking-tighter">
                {currentOption?.label}
              </Badge>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {!isEditingMode ? (
              <Button variant="outline" size="sm" className="h-8 gap-2 text-[10px] font-black tracking-widest uppercase" onClick={() => setIsEditingMode(true)}>
                <Edit3 className="w-3.5 h-3.5 text-primary" /> Enable Editing
              </Button>
            ) : (
              <>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="h-8 px-3 text-[10px] font-black tracking-widest text-muted-foreground hover:text-destructive hover:bg-destructive/5 transition-all uppercase" 
                  onClick={handleCancelClick}
                >
                  Cancel
                </Button>

                <AlertDialog open={showCancelConfirm} onOpenChange={setShowCancelConfirm}>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 rounded-full bg-destructive/10">
                          <AlertTriangle className="h-5 w-5 text-destructive" />
                        </div>
                        <AlertDialogTitle className="text-sm font-black uppercase tracking-tight">Discard Changes?</AlertDialogTitle>
                      </div>
                      <AlertDialogDescription className="text-xs font-medium text-muted-foreground leading-relaxed">
                        You are about to exit editing mode. All unsaved modifications to the agent architecture will be permanently lost. This action cannot be undone.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter className="gap-2">
                      <AlertDialogCancel className="h-8 text-[10px] font-black uppercase tracking-widest">Keep Editing</AlertDialogCancel>
                      <AlertDialogAction 
                        className="h-8 text-[10px] font-black uppercase tracking-widest bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        onClick={handleCancelEdit}
                      >
                        Discard Changes
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>

                <Button variant="default" size="sm" className="h-8 gap-2 text-[10px] font-black tracking-widest shadow-sm transition-all uppercase" onClick={saveToCode}>
                  <Save className="w-3.5 h-3.5" /> Save Agent
                </Button>
              </>
            )}
            <Separator orientation="vertical" className="h-5 mx-1" />
            <Badge variant="outline" className="h-6 text-[9px] font-mono tracking-[0.2em] uppercase font-black text-muted-foreground">Stable</Badge>
          </div>
        </header>
      </div>

      {/* 2. Resizable Main Area */}
      <main className="flex-1 min-h-0">
        <ThreePanelLayout
          layoutKey="builder-vfinal"
          leftShouldCollapse={leftShouldCollapse}
          rightShouldCollapse={rightShouldCollapse}
          leftMinExpandPx={leftCompactWidth}
          rightMinExpandPx={rightCompactWidth}
          leftPanel={
            <div ref={leftContainerRef} className="h-full flex flex-col bg-transparent overflow-hidden relative">
              {/* Full sizer: icon + text (global px-2 + local px-3) */}
              <div
                ref={leftFullSizerRef}
                aria-hidden
                className="absolute top-0 left-0 pointer-events-none flex items-center gap-0 px-2"
                style={{ visibility: 'hidden', whiteSpace: 'nowrap' }}
              >
                <span className="inline-flex items-center gap-2 text-[11px] font-bold px-3">
                  <Files className="w-4 h-4" /><span>Explorer</span>
                </span>
                <span className="inline-flex items-center gap-2 text-[11px] font-bold px-3">
                  <Package className="w-4 h-4" /><span>Registry</span>
                </span>
              </div>
              {/* Compact sizer: icon-only (global px-2 + local px-3) */}
              <div
                ref={leftCompactSizerRef}
                aria-hidden
                className="absolute top-0 left-0 pointer-events-none flex items-center gap-0 px-2"
                style={{ visibility: 'hidden', whiteSpace: 'nowrap' }}
              >
                <span className="inline-flex items-center justify-center px-3">
                  <Files className="w-4 h-4" />
                </span>
                <span className="inline-flex items-center justify-center px-3">
                  <Package className="w-4 h-4" />
                </span>
              </div>

              <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
                <div className="shrink-0 h-11 flex items-center px-2 border-b border-border/40 bg-muted/20">
                  <TabsList className="bg-transparent h-11 w-full justify-center gap-0 p-0">
                    <TabsTrigger value="files" className="h-11 flex-1 rounded-none border-b-2 border-transparent text-[11px] font-bold data-[state=active]:border-primary data-[state=active]:text-primary transition-all bg-transparent shadow-none flex items-center justify-center gap-2 px-3">
                      <Files className="w-4 h-4 flex-shrink-0" />
                      {!leftCompact && <span>Explorer</span>}
                    </TabsTrigger>
                    <TabsTrigger value="registry" className="h-11 flex-1 rounded-none border-b-2 border-transparent text-[11px] font-bold data-[state=active]:border-primary data-[state=active]:text-primary transition-all bg-transparent shadow-none flex items-center justify-center gap-2 px-3">
                      <Package className="w-4 h-4 flex-shrink-0" />
                      {!leftCompact && <span>Registry</span>}
                    </TabsTrigger>
                  </TabsList>
                </div>
                <TabsContent value="files" className="flex-1 overflow-hidden m-0"><FileExplorer /></TabsContent>
                <TabsContent value="registry" className="flex-1 overflow-y-auto m-0 p-3 bg-muted/5">
                  <InlineComponentPalette onAddComponent={addNode} onAddControlFlow={(i) => addControlFlowNode(i.nodeType, { label: i.label })} />
                </TabsContent>
              </Tabs>
            </div>
          }
          centerPanel={
            <div className="h-full flex flex-col bg-transparent overflow-hidden relative">
              {selectedAgent ? (
                <>
                  <div className="h-11 border-b border-border/40 bg-muted/20 flex items-center px-4">
                    <Tabs value={selectedMethod} onValueChange={setSelectedMethod} className="w-auto">
                      <TabsList className="bg-transparent h-11 gap-6 p-0">
                        {TAB_METHODS.map((m) => (
                          <TabsTrigger 
                            key={m.value} 
                            value={m.value} 
                            className="text-[11px] font-bold gap-2 h-11 px-1 rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:text-primary transition-all bg-transparent shadow-none"
                          >
                            <m.icon className="w-4 h-4" /> {m.label}
                          </TabsTrigger>
                        ))}
                      </TabsList>
                    </Tabs>
                  </div>
                  <div className="flex-1 min-h-0 relative bg-[radial-gradient(circle,hsl(var(--muted-foreground)/0.1)_1px,transparent_1px)] [background-size:20px_20px]">
                    <BuilderEditPane phase={phase} method={selectedMethod} />
                  </div>
                </>
              ) : (
                <div className="h-full flex flex-col items-center justify-center opacity-30">
                  <Monitor className="w-20 h-20 mb-4 text-muted-foreground" />
                  <h3 className="text-xl font-black tracking-tighter uppercase text-muted-foreground">Select Agent to Visualize</h3>
                </div>
              )}
            </div>
          }
          rightPanel={
            <div className="h-full flex flex-col bg-transparent overflow-hidden">
              <RightRail
                tabs={[
                  { value: 'inspector', label: 'Inspector', icon: Info, content: <PropertiesPane className="h-full border-none" /> },
                  { value: 'code', label: 'Code', icon: Code2, content: <CodePane readOnly={!isEditingMode} className="h-full border-none" /> },
                  { value: 'history', label: 'History', icon: History, content: <BuilderHistoryPanel className="h-full border-none" /> },
                ]}
                defaultTab="inspector"
                onShouldCollapseChange={setRightShouldCollapse}
                onCompactWidthChange={setRightCompactWidth}
              />
            </div>
          }
        />
      </main>
    </div>
  );
}

export function AgentBuilder({ className }: { className?: string }) {
  return (
    <div className={className}>
      <AgentBuilderProvider>
        <AgentBuilderContent />
      </AgentBuilderProvider>
    </div>
  );
}
