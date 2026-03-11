/**
 * ExecuteInspector - Integrated IDE Inspector for Execute Tab
 */
import { memo, useCallback, useState } from 'react';
import { Settings2, Info, Database, Plug, Hash, Plus, Trash2, Code2 } from 'lucide-react';
import { ScrollArea } from '@/shared/ui/scroll-area';
import { Badge } from '@/shared/ui/badge';
import { Separator } from '@/shared/ui/separator';
import { Input } from '@/shared/ui/input';
import { Label } from '@/shared/ui/label';
import { Button } from '@/shared/ui/button';
import { cn } from '@/shared/lib/utils';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import { getCategoryTheme } from '@/entities/node-system/config/themes';

interface ExecuteInspectorProps {
  className?: string;
  onConfigChange?: (nodeId: string, changes: Record<string, unknown>) => void;
}

export const ExecuteInspector = memo(function ExecuteInspector({
  className,
  onConfigChange,
}: ExecuteInspectorProps) {
  const selectedNodeId = useNodeStore((s) => s.selectedNodeId);
  const nodes = useNodeStore((s) => s.nodes);
  const updateNode = useNodeStore((s) => s.updateNode);
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  const [newMethodName, setNewMethodName] = useState('');

  const handleParamChange = useCallback(
    (paramName: string, value: string | number | boolean) => {
      if (selectedNodeId && onConfigChange) {
        onConfigChange(selectedNodeId, { [paramName]: value });
      }
    },
    [selectedNodeId, onConfigChange]
  );

  const handleAddMethod = useCallback(() => {
    if (!selectedNode || !newMethodName.trim()) return;
    
    const currentOutputs = selectedNode.data.outputs || [];
    if (currentOutputs.some(p => p.name === newMethodName)) return;

    const newPort = {
      id: newMethodName,
      name: newMethodName,
      type: 'any',
      kind: 'data' as const
    };

    updateNode(selectedNode.id, {
      outputs: [...currentOutputs, newPort]
    });
    setNewMethodName('');
  }, [selectedNode, newMethodName, updateNode]);

  const handleRemoveMethod = useCallback((portName: string) => {
    if (!selectedNode) return;
    
    const currentOutputs = selectedNode.data.outputs || [];
    updateNode(selectedNode.id, {
      outputs: currentOutputs.filter(p => p.name !== portName)
    });
  }, [selectedNode, updateNode]);

  if (!selectedNode) {
    return (
      <div className={cn('h-full flex flex-col items-center justify-center opacity-20 p-8 text-center', className)}>
        <Settings2 className="h-12 w-12 mb-4" />
        <p className="text-[10px] font-black tracking-widest uppercase">Select Component</p>
      </div>
    );
  }

  const { data } = selectedNode;
  const params = data.params || [];
  const theme = data.category ? getCategoryTheme(data.category) : null;
  const dataOutputs = (data.outputs || []).filter(p => p.kind === 'data' && p.name !== 'self');

  return (
    <div className={cn('flex flex-col h-full bg-card/50', className)}>
      {/* Header - Consistent with PropertiesPane */}
      <div className="h-11 flex items-center px-4 border-b border-border/40 bg-muted/20 shrink-0">
        <div className="flex items-center gap-2 text-primary font-black uppercase tracking-widest text-[10px]">
          <Settings2 className="h-4 w-4" />
          <span>Execution Inspector</span>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-5">
          {/* 1. Identity Card */}
          <div className="bg-card border border-border/60 rounded-2xl p-4 shadow-sm space-y-3 relative overflow-hidden group">
            <div className={cn("absolute top-0 left-0 w-1 h-full opacity-60", theme?.border || "bg-primary")} />
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-background border border-border/40">
                <Database className="h-4.5 w-4.5 text-primary" />
              </div>
              <div className="min-w-0">
                <h4 className="text-sm font-black tracking-tight truncate uppercase">{data.role || selectedNodeId}</h4>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant="outline" className="text-[8px] h-4 font-mono uppercase bg-muted/20">{data.category}</Badge>
                  <span className="text-[10px] font-mono opacity-40 truncate">{data.implementation}</span>
                </div>
              </div>
            </div>
          </div>

          {/* 2. Configuration Grid */}
          <div className="space-y-3">
            <div className="flex items-center justify-between w-full group">
              <div className="flex items-center gap-2 text-[10px] font-black text-muted-foreground uppercase tracking-widest">
                <Info className="h-3 w-3 text-primary" />
                <span>Runtime Configuration</span>
              </div>
              <Separator className="flex-1 ml-4 opacity-10" />
            </div>
            
            <div className="bg-muted/10 rounded-2xl border border-border/40 p-1.5 space-y-0.5">
              {params.length > 0 ? (
                params.map((param) => (
                  <div key={param.name} className="grid grid-cols-[90px_1fr] items-center gap-3 py-2 px-2 hover:bg-muted/40 transition-all rounded-lg group">
                    <Label className="text-[10px] font-black truncate opacity-50 group-hover:opacity-100 uppercase tracking-tighter">
                      {param.name}
                    </Label>
                    <div className="flex items-center gap-2">
                      {param.type === 'boolean' ? (
                        <div className="flex gap-1">
                          <Button 
                            variant={param.value ? 'default' : 'outline'} 
                            size="sm" 
                            className="h-6 text-[10px] px-2 font-bold" 
                            onClick={() => handleParamChange(param.name, true)}
                          >
                            T
                          </Button>
                          <Button 
                            variant={!param.value ? 'default' : 'outline'} 
                            size="sm" 
                            className="h-6 text-[10px] px-2 font-bold" 
                            onClick={() => handleParamChange(param.name, false)}
                          >
                            F
                          </Button>
                        </div>
                      ) : (
                        <Input
                          value={String(param.value)}
                          onChange={(e) => {
                            const val = e.target.value;
                            if (param.type === 'number') {
                              const num = Number(val);
                              if (!isNaN(num)) handleParamChange(param.name, num);
                            } else {
                              handleParamChange(param.name, val);
                            }
                          }}
                          className="h-7 text-[11px] font-mono bg-background/50 border-border/40 focus-visible:ring-primary/20"
                        />
                      )}
                      <Badge variant="outline" className="text-[7px] px-1 h-3 opacity-30">{param.type.toUpperCase()}</Badge>
                    </div>
                  </div>
                ))
              ) : (
                <p className="py-6 text-center text-[10px] text-muted-foreground italic opacity-40">No runtime parameters</p>
              )}
            </div>
          </div>

          {/* 3. Port Interface & Dynamic Methods */}
          <div className="space-y-4">
            <div className="flex items-center justify-between w-full opacity-60">
              <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                <Code2 className="h-3 w-3 text-primary" />
                <span>Interface Methods</span>
              </div>
              <Separator className="flex-1 ml-4 opacity-10" />
            </div>

            {/* Dynamic Output Methods (Req 1, 2) */}
            <div className="bg-muted/10 rounded-2xl border border-border/40 p-3 space-y-3">
              <div className="flex items-center gap-2">
                <Input 
                  placeholder="Method name (e.g. parameters)" 
                  value={newMethodName}
                  onChange={(e) => setNewMethodName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAddMethod()}
                  className="h-8 text-[11px] font-mono bg-background/50 border-border/40"
                />
                <Button size="sm" className="h-8 px-2" onClick={handleAddMethod}>
                  <Plus className="h-4 w-4" />
                </Button>
              </div>

              <div className="space-y-1.5">
                {dataOutputs.length > 0 ? (
                  dataOutputs.map(port => (
                    <div key={port.id} className="flex items-center justify-between p-2 rounded-lg bg-background border border-border/20 group/item hover:border-primary/30 transition-all">
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary/40 animate-pulse" />
                        <span className="text-[10px] font-black font-mono uppercase text-foreground/70">{port.name}()</span>
                      </div>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="h-6 w-6 p-0 opacity-0 group-hover/item:opacity-100 hover:bg-destructive/10 hover:text-destructive text-muted-foreground transition-all"
                        onClick={() => handleRemoveMethod(port.name)}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  ))
                ) : (
                  <p className="text-[9px] text-center text-muted-foreground py-2 opacity-40 italic">No custom output methods defined</p>
                )}
              </div>
            </div>

            {/* Static Inputs (Read-only for setup) */}
            <div className="bg-background border border-border/40 rounded-xl p-3 shadow-sm opacity-60">
              <span className="text-[8px] font-black opacity-30 uppercase tracking-widest block mb-2.5">Input Requirements</span>
              <div className="flex flex-wrap gap-1.5">
                {data.inputs.length > 0 ? (
                  data.inputs.map(p => <Badge key={p.id} variant="secondary" className="text-[9px] h-5 px-2 bg-muted/50 border-border/40 font-bold uppercase">{p.name}</Badge>)
                ) : (
                  <span className="text-[9px] italic opacity-30">No input dependencies</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
});
