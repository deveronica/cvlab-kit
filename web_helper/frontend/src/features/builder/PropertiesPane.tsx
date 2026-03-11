/**
 * PropertiesPane - Precision IDE Inspector (vFinal)
 */
import { useMemo, useCallback, useState, useEffect } from 'react';
import { Node } from 'reactflow';
import {
  ChevronDown,
  Settings2,
  Info,
  Plug,
  Database,
  Save,
  Trash2,
  Code2,
} from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { Label } from '@/shared/ui/label';
import { Badge } from '@/shared/ui/badge';
import { Button } from '@/shared/ui/button';
import { ScrollArea } from '@/shared/ui/scroll-area';
import { Separator } from '@/shared/ui/separator';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/shared/ui/collapsible';
import { TooltipProvider } from '@/shared/ui/tooltip';
import { useAgentBuilder } from '@/entities/node-system/model/AgentBuilderContext';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import { useBuilderStore } from '@/features/builder/model/builderStore';
import { getCategoryTheme } from '@/entities/node-system/config/themes';
import { PropertyInput } from './PropertyInput';
import type { ConfigValue } from '@/shared/model/config-types';
import { UnifiedNodeData } from '@/entities/node-system/model/types';

export function PropertiesPane({ className }: { className?: string }) {
  const { removeNode } = useAgentBuilder();
  const selectedNodeId = useNodeStore((s) => s.selectedNodeId);
  const selectedEdgeId = useBuilderStore((s) => s.selectedEdgeId);  // Req 4
  const nodes = useNodeStore((s) => s.nodes);
  const edges = useBuilderStore((s) => s.edges);  // Req 4
  const updateNodes = useNodeStore((s) => s.updateNodes); // useNodeStore에서 가져옴
  const updateEdges = useBuilderStore((s) => s.updateEdges);  // Req 4
  const setIsDirty = useBuilderStore((s) => s.setIsDirty); // useBuilderStore에서 가져옴

  const selectedNodeData = useMemo(() => nodes.find(n => n.id === selectedNodeId) || null, [nodes, selectedNodeId]);
  const selectedEdgeData = useMemo(() => edges.find(e => e.id === selectedEdgeId) || null, [edges, selectedEdgeId]);  // Req 4
  const [localEdits, setLocalEdits] = useState<Record<string, any>>({});
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    setLocalEdits({});
    setHasChanges(false);
  }, [selectedNodeId, selectedEdgeId]);

  const nodeData = selectedNodeData?.data;
  const theme = nodeData?.category ? getCategoryTheme(nodeData.category) : null;

  const handlePropertyChange = useCallback((name: string, value: ConfigValue) => {
    setLocalEdits((prev) => ({ ...prev, [name]: value }));
    setHasChanges(true);
  }, []);

  const handleSave = useCallback(() => {
    if (!hasChanges || (!selectedNodeId && !selectedEdgeId)) return;

    // Req 4: Handle edge updates
    if (selectedEdgeId) {
      updateEdges((prevEdges) =>
        prevEdges.map(e =>
          e.id === selectedEdgeId
            ? {
                ...e,
                data: {
                  ...e.data,
                  sequenceIndex: localEdits.sequence_index !== undefined ? localEdits.sequence_index : (e.data as any)?.sequenceIndex,
                }
              }
            : e
        )
      );
    } else if (selectedNodeId) {
      // 항목 1: 노드 정보 업데이트 (이름, 메서드 등)
      updateNodes((prevNodes: Node<UnifiedNodeData>[]) =>
        prevNodes.map(n =>
          n.id === selectedNodeId
            ? {
                ...n,
                data: {
                  ...n.data,
                  object_name: localEdits.object_name !== undefined ? localEdits.object_name : (n.data as any).object_name,
                  method_name: localEdits.method_name !== undefined ? localEdits.method_name : (n.data as any).method_name,
                  params: (n.data.params || []).map(p =>
                    localEdits[p.name] !== undefined ? { ...p, value: localEdits[p.name] } : p
                  )
                }
              }
            : n
        )
      );
    }

    setIsDirty(true); // 항목 5: 히스토리 관리를 위한 Dirty 설정
    setLocalEdits({});
    setHasChanges(false);
  }, [selectedNodeId, selectedEdgeId, hasChanges, localEdits, updateNodes, updateEdges, setIsDirty]);

  const renderParam = (key: string, value: any) => {
    const isModified = localEdits[key] !== undefined;
    return (
      <div key={key} className="grid grid-cols-[100px_1fr] items-center gap-2 py-1 px-2 hover:bg-muted/30 transition-all group border-b border-border/5 last:border-0">
        <Label className={cn(
          'text-[9px] font-bold truncate opacity-40 group-hover:opacity-100 uppercase tracking-tighter transition-all', 
          isModified && 'text-primary opacity-100 font-black'
        )}>
          {key}
        </Label>
        <PropertyInput
          name={key}
          value={localEdits[key] ?? value}
          type={typeof value === 'number' ? 'number' : (typeof value === 'boolean' ? 'boolean' : 'string')}
          onChange={handlePropertyChange}
        />
      </div>
    );
  };

  return (
    <TooltipProvider>
      <div className={cn('flex flex-col h-full bg-card/30', className)}>
        {/* Header - Item 2 Refactor: Ultra-thin Professional Header */}
        <div className="h-11 flex items-center justify-between px-4 border-b border-border/40 bg-muted/20 shrink-0">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-primary/10 flex items-center justify-center border border-primary/20">
              <Settings2 className="h-3 w-3 text-primary" />
            </div>
            <span className="text-foreground/70 font-black uppercase tracking-[0.15em] text-[9px]">Inspector</span>
          </div>
          {(selectedNodeData || selectedEdgeData) && (
            <div className="flex items-center gap-1">
              <Button
                variant={hasChanges ? "default" : "ghost"}
                size="sm"
                className={cn("h-7 px-2.5 gap-1.5 text-[9px] font-black uppercase tracking-wider shadow-md transition-all", !hasChanges && "opacity-40")}
                onClick={handleSave}
                disabled={!hasChanges}
              >
                <Save className="h-3.5 w-3.5" /> Save Changes
              </Button>
              {selectedNodeData && (
                <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-destructive hover:bg-destructive/10" onClick={() => removeNode(selectedNodeId!)}>
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
          )}
        </div>

        <ScrollArea className="flex-1">
          <div className="p-3 space-y-5">
            {!selectedNodeData && !selectedEdgeData ? (
              <div className="flex flex-col items-center justify-center py-24 opacity-10">
                <Info className="h-16 w-16 mb-4 stroke-[1px]" />
                <p className="text-[10px] font-black tracking-[0.2em] uppercase">No Selection</p>
              </div>
            ) : selectedEdgeData ? (
              // Edge Inspector (Req 4)
              <>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Badge variant="outline" className="text-[8px] h-4 font-black uppercase tracking-[0.1em] border-none px-0 opacity-50 bg-amber-500/10 text-amber-600">
                      EDGE
                    </Badge>
                    <span className="text-[8px] font-mono opacity-30 tracking-tighter">ID: {selectedEdgeId}</span>
                  </div>

                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <Label className="text-[8px] font-bold uppercase opacity-40 tracking-tighter">From Node</Label>
                        <div className="text-[10px] font-mono bg-muted/30 rounded px-2 py-1 mt-1 border border-border/20">{selectedEdgeData.source}</div>
                      </div>
                      <div>
                        <Label className="text-[8px] font-bold uppercase opacity-40 tracking-tighter">To Node</Label>
                        <div className="text-[10px] font-mono bg-muted/30 rounded px-2 py-1 mt-1 border border-border/20">{selectedEdgeData.target}</div>
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <Label className="text-[9px] font-black uppercase tracking-widest text-muted-foreground/60 px-1">Creation Order (sequence_index)</Label>
                      <input
                        type="number"
                        className="w-full bg-muted/30 border border-border/40 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-primary/40 transition-all"
                        value={localEdits.sequence_index ?? (selectedEdgeData.data as any)?.sequenceIndex ?? ''}
                        onChange={(e) => handlePropertyChange('sequence_index', parseInt(e.target.value) || 0)}
                      />
                      <p className="text-[8px] opacity-40 italic">Numeric order for edge execution in setup phase</p>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-[9px]">
                      <div className="bg-muted/20 rounded p-2 border border-border/20">
                        <div className="font-bold opacity-50 uppercase tracking-tight mb-1">Source Port</div>
                        <div className="text-[10px] font-mono">{selectedEdgeData.sourceHandle || 'default'}</div>
                      </div>
                      <div className="bg-muted/20 rounded p-2 border border-border/20">
                        <div className="font-bold opacity-50 uppercase tracking-tight mb-1">Target Port</div>
                        <div className="text-[10px] font-mono">{selectedEdgeData.targetHandle || 'default'}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                {/* 1. Editable Identity Section */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Badge variant="outline" className={cn("text-[8px] h-4 font-black uppercase tracking-[0.1em] border-none px-0 opacity-50", theme?.text || "text-primary")}>
                      {nodeData?.category}
                    </Badge>
                    <span className="text-[8px] font-mono opacity-30 tracking-tighter">ID: {selectedNodeId}</span>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="space-y-1.5">
                      <Label className="text-[9px] font-black uppercase tracking-widest text-muted-foreground/60 px-1">Node Title</Label>
                      <input 
                        className="w-full bg-muted/30 border border-border/40 rounded-lg px-3 py-2 text-sm font-black tracking-tight focus:outline-none focus:ring-1 focus:ring-primary/40 transition-all uppercase"
                        value={localEdits.object_name ?? (nodeData as any)?.object_name ?? ''}
                        onChange={(e) => handlePropertyChange('object_name', e.target.value)}
                      />
                    </div>

                    <div className="space-y-1.5">
                      <Label className="text-[9px] font-black uppercase tracking-widest text-muted-foreground/60 px-1">Method Action</Label>
                      <select 
                        className="w-full bg-muted/30 border border-border/40 rounded-lg px-3 py-2 text-xs font-mono font-bold text-primary focus:outline-none focus:ring-1 focus:ring-primary/40 transition-all appearance-none"
                        value={localEdits.method_name ?? (nodeData as any)?.method_name ?? 'setup'}
                        onChange={(e) => handlePropertyChange('method_name', e.target.value)}
                      >
                        <option value="setup">setup()</option>
                        <option value="forward">forward()</option>
                        <option value="step">step()</option>
                        <option value="parameters">parameters()</option>
                      </select>
                    </div>
                  </div>
                </div>

                <Separator className="opacity-10" />

                {/* 항목 7: 인스펙터로 옮겨진 구현 코드 */}
                <div className="space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/60 flex items-center gap-2 px-1">
                    <Code2 className="w-3.5 h-3.5" />
                    Source Implementation
                  </label>
                  <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 shadow-inner group/code relative overflow-hidden">
                    <pre className="text-[10px] font-mono leading-relaxed text-zinc-300 whitespace-pre-wrap break-all">
                      <code>{(nodeData as any)?.metadata?.implementation || nodeData?.implementation || '# No implementation source available'}</code>
                    </pre>
                    <div className="absolute top-2 right-2 opacity-0 group-hover/code:opacity-100 transition-opacity">
                      <div className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500 text-[8px] font-mono border border-zinc-700">PYTHON</div>
                    </div>
                  </div>
                </div>

                <Separator className="opacity-20" />

                {/* 2. Configuration Grid */}
                <div className="space-y-2.5">
                  <div className="flex items-center gap-2 text-[9px] font-black text-muted-foreground/60 uppercase tracking-widest px-1">
                    <span>Config</span>
                    <Separator className="flex-1 opacity-10" />
                  </div>
                  
                  <div className="bg-muted/5 rounded-xl border border-border/40 overflow-hidden divide-y divide-border/5">
                    {nodeData?.params && nodeData.params.length > 0 ? (
                      nodeData.params.map((p) => renderParam(p.name, p.value))
                    ) : (
                      <div className="py-8 text-center opacity-20 italic text-[9px] uppercase font-bold tracking-widest">Read Only Component</div>
                    )}
                  </div>
                </div>

                {/* 3. Port Interface */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-[9px] font-black text-muted-foreground/60 uppercase tracking-widest px-1">
                    <Plug className="h-3 w-3" />
                    <span>Interface</span>
                    <Separator className="flex-1 opacity-10" />
                  </div>

                  <div className="grid grid-cols-1 gap-2">
                    <div className="bg-card/50 border border-border/30 rounded-xl p-2.5 space-y-2">
                      <span className="text-[8px] font-black opacity-30 uppercase tracking-widest block">Inputs</span>
                      <div className="flex flex-wrap gap-1">
                        {nodeData?.inputs.map(p => <Badge key={p.id} variant="secondary" className="text-[8px] h-4 px-1.5 bg-blue-500/10 text-blue-400 border-blue-500/20 font-bold uppercase">{p.name}</Badge>)}
                        {!nodeData?.inputs.length && <span className="text-[8px] opacity-20 italic">No input ports</span>}
                      </div>
                    </div>
                    <div className="bg-card/50 border border-border/30 rounded-xl p-2.5 space-y-2">
                      <span className="text-[8px] font-black opacity-30 uppercase tracking-widest block">Outputs</span>
                      <div className="flex flex-wrap gap-1">
                        {nodeData?.outputs.map(p => <Badge key={p.id} variant="outline" className="text-[8px] h-4 px-1.5 border-green-500/30 text-green-400 font-bold uppercase">{p.name}</Badge>)}
                        {!nodeData?.outputs.length && <span className="text-[8px] opacity-20 italic">No output ports</span>}
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </ScrollArea>
      </div>
    </TooltipProvider>
  );
}
