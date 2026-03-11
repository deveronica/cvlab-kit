import { memo, useMemo, type CSSProperties, useState, useEffect } from 'react';
import { Handle, Position, useStore } from 'reactflow';
import { Cpu, Zap, Activity, Database, Settings, ChevronRight } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import type { UnifiedNodeData, UnifiedPort } from '../model/types';
import { getCategoryTheme } from '../config/themes';
import { useNodeStore } from '../model/nodeStore';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from '@/shared/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/shared/ui/tooltip';
import { useBuilderStore } from '@/features/builder/model/builderStore';
import { useAgentBuilder } from '../model/AgentBuilderContext';
import { Badge } from '@/shared/ui/badge';
import { Input } from '@/shared/ui/input';
import { Button } from '@/shared/ui/button';
import { Save, Search } from 'lucide-react';
import { ConfigKeyValueList } from './ConfigKeyValueList';


// =============================================================================
// STYLING UTILS
// =============================================================================

function getFlowPinStyle(type: string, colors: Record<string, string>, position: Position, connected: boolean): CSSProperties {
  const isExecution = type === 'execution';
  // 항목 1, 2: DB에서 가져온 색상 적용, 없으면 any 색상
  const color = isExecution ? '#ffffff' : (colors[type] || colors['any'] || '#94a3b8');
  
  const baseStyle: CSSProperties = {
    background: connected ? color : 'transparent',
    border: `2px solid ${color}`,
    width: isExecution ? '12px' : '10px',
    height: isExecution ? '10px' : '10px',
    position: 'absolute',
    top: '50%',
    zIndex: 150,
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    clipPath: isExecution ? 'polygon(0% 0%, 70% 0%, 100% 50%, 70% 100%, 0% 100%)' : undefined,
    borderRadius: isExecution ? '0px' : '50%',
  };

  if (position === Position.Left) {
    return { ...baseStyle, left: '0px', transform: 'translate(-50%, -50%)' };
  }
  return { ...baseStyle, right: '0px', transform: 'translate(50%, -50%)' };
}

const PortRow = memo(({ input, output, isExecution = false }: { 
  input?: UnifiedPort; 
  output?: UnifiedPort;
  isExecution?: boolean;
}) => {
  const typeColors = useBuilderStore(s => s.typeColors);
  
  const connectedHandleKeys = useStore(
    (state) => {
      const connected = new Set<string>();
      state.edges.forEach((edge) => {
        // These already contain prefixes from graph-utils.ts / useNodeGraph.ts
        if (edge.sourceHandle) connected.add(edge.sourceHandle);
        if (edge.targetHandle) connected.add(edge.targetHandle);
      });
      return connected;
    }
  );

  const isConnected = (type: 'source' | 'target', id?: string) => 
    id ? connectedHandleKeys.has(`${type}:${id}`) : false;

  return (
    <div className={cn(
      "flex items-center justify-between h-7 relative group/row px-1",
      isExecution && "h-0"
    )}>
      {/* Input Side */}
      <div className="flex items-center gap-2 flex-1 min-w-0">
        {input && (
          <div className="flex items-center relative h-full">
            <TooltipProvider delayDuration={100}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={`target:${input.name}`}
                    style={getFlowPinStyle(isExecution ? 'execution' : input.type, typeColors, Position.Left, isConnected('target', input.name))}
                    className="!z-[200] hover:scale-150 transition-all shadow-glow"
                  />
                </TooltipTrigger>
                <TooltipContent side="left" className="text-[9px] font-mono font-black bg-zinc-900/90 border-zinc-700 text-white">
                  TYPE: {input.type.toUpperCase()}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            {!isExecution && (
              <span className="text-[10px] font-bold ml-3 truncate text-foreground/50 group-hover/row:text-foreground transition-colors uppercase tracking-tight">
                {input.name}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Output Side */}
      <div className="flex items-center justify-end gap-2 flex-1 min-w-0">
        {output && (
          <div className="flex items-center relative h-full">
            {!isExecution && (
              <span className="text-[10px] font-bold mr-3 truncate text-foreground/50 group-hover/row:text-foreground transition-colors text-right uppercase tracking-tight">
                {output.name}
              </span>
            )}
            <TooltipProvider delayDuration={100}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={`source:${output.name}`}
                    style={getFlowPinStyle(isExecution ? 'execution' : output.type, typeColors, Position.Right, isConnected('source', output.name))}
                    className="!z-[200] hover:scale-150 transition-all shadow-glow"
                  />
                </TooltipTrigger>
                <TooltipContent side="right" className="text-[9px] font-mono font-black bg-zinc-900/90 border-zinc-700 text-white">
                  TYPE: {output.type.toUpperCase()}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        )}
      </div>
    </div>
  );
});

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export const UnifiedNode = memo(function UnifiedNode({
  data,
  selected,
}: { data: UnifiedNodeData; selected: boolean }) {
  const theme = getCategoryTheme(data.category);
  const isYamlConfig = data.nodeType === 'yamlConfig' || data.id === 'yaml_config';
  const Icon = isYamlConfig ? Database : (data.category === 'model' ? Cpu : data.category === 'optimizer' ? Zap : data.category === 'dataset' ? Database : Activity);
  const isConfigMode = data.mode === 'execute';
  
  const isEditingMode = useBuilderStore(s => s.isEditingMode);
  const agentBuilder = useAgentBuilder();
  const nodeMode = data.mode; // 'execute' (Setup) or 'builder'/'flow' (Training)
  
  const hoverNode = useNodeStore(s => s.hoverNode);
  const hoveredNodeId = useNodeStore(s => s.hoveredNodeId);
  const nodes = useNodeStore((s) => s.nodes);

  // --- Config Editor State (ONLY for yamlConfig in Setup mode) ---
  const [configSearch, setConfigSearch] = useState('');
  const [localConfig, setLocalConfig] = useState<Record<string, any>>(data.metadata?.configData || {});
  
  // Sync prop changes to local state
  useEffect(() => {
    if (data.metadata?.configData) {
      setLocalConfig(data.metadata.configData);
    }
  }, [data.metadata?.configData]);
  const [isSaving, setIsSaving] = useState(false);

  const handleConfigChange = (key: string, value: string) => {
    if (nodeMode !== 'execute') return;
    setLocalConfig(prev => ({ ...prev, [key]: value }));
  };

  const saveConfig = async () => {
    if (nodeMode !== 'execute' || !data.metadata?.configPath) return;
    setIsSaving(true);
    try {
      const res = await fetch('/api/configs/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: data.metadata.configPath,
          content: localConfig
        })
      });
      if (res.ok) {
        agentBuilder?.refreshGraph();
      }
    } catch (e) {
      console.error('Failed to save config:', e);
    } finally {
      setIsSaving(false);
    }
  };

  const filteredConfigKeys = useMemo(() => {
    if (nodeMode !== 'execute') return [];
    return Object.keys(localConfig).filter(k => 
      k.toLowerCase().includes(configSearch.toLowerCase())
    ).sort();
  }, [localConfig, configSearch, nodeMode]);

  const hoveredUsedConfigKeys = useMemo(() => {
    if (!hoveredNodeId || hoveredNodeId === 'yaml_config') return new Set<string>();
    const hoveredNode = nodes.find((n) => n.id === hoveredNodeId);
    return new Set<string>(hoveredNode?.data.usedConfigKeys ?? []);
  }, [hoveredNodeId, nodes]);

  // 항목 1: 메서드 드롭다운 상태
  const [method, setMethod] = useState((data as any).method_name || 'forward');

  const handleMethodChange = (newMethod: string) => {
    if (nodeMode !== 'execute') return; // Training flow doesn't allow method switching this way
    setMethod(newMethod);
    agentBuilder?.refreshGraph();
  };

  const execRow = useMemo(() => {
    const pins = data.executionPins || [];
    const inputs = pins.filter(p => p.kind === 'exec');
    const outputs = pins.filter(p => p.kind === 'exec');
    
    // Pick standard names if available, otherwise take the first one
    const input = inputs.find(p => p.name === 'in' || p.name === 'body') || inputs[0];
    const output = outputs.find(p => p.name === 'out' || p.name === 'true' || p.name === 'forward') || outputs[0];
    
    return { input, output };
  }, [data.executionPins]);

  const dataRows = useMemo(() => {
    // Show all ports that are not execution pins
    const inputs = data.inputs || [];
    const outputs = data.outputs || [];
    
    if (isYamlConfig) {
      // For Config Node, we usually only have outputs (config keys)
      // Group them into pairs for a "giant node" layout
      const rows = [];
      for (let i = 0; i < outputs.length; i += 1) {
        rows.push({ output: outputs[i] });
      }
      return rows;
    }

    const maxRows = Math.max(inputs.length, outputs.length);
    return Array.from({ length: maxRows }).map((_, i) => ({
      input: inputs[i],
      output: outputs[i]
    }));
  }, [data.inputs, data.outputs, isYamlConfig]);

  // Determine if this specific node should be highlighted
  const isHighlighted = useMemo(() => {
    if (!hoveredNodeId) return false;
    if (hoveredNodeId === data.id) return true;
    
    // If hovering GLOBAL CONFIG, highlight components that use ANY config key
    if (hoveredNodeId === 'yaml_config' && data.usedConfigKeys && data.usedConfigKeys.length > 0) return true;
    
    // If hovering a component, highlight GLOBAL CONFIG
    if (isYamlConfig && hoveredNodeId !== 'yaml_config') {
      const hoveredNode = nodes.find(n => n.id === hoveredNodeId);
      return hoveredNode?.data.usedConfigKeys && hoveredNode.data.usedConfigKeys.length > 0;
    }
    
    return false;
  }, [hoveredNodeId, data.id, data.usedConfigKeys, isYamlConfig, nodes]);

  return (
    <div
      onMouseEnter={() => hoverNode(data.id)}
      onMouseLeave={() => hoverNode(null)}
      className={cn(
        "group relative flex flex-col select-none transition-all duration-300",
        isYamlConfig ? "w-[280px]" : "w-[240px]",
        "bg-card/95 backdrop-blur-sm text-card-foreground border-2 shadow-lg rounded-xl overflow-visible",
        selected 
          ? "ring-4 ring-primary/20 border-primary shadow-2xl scale-[1.02] z-[100]" 
          : "border-border/40 hover:border-border/80 z-[10]",
        isHighlighted && !selected && "border-primary/60 shadow-primary/10 bg-primary/[0.03] scale-[1.01] z-[50]",
        data.isUncovered && "opacity-40 grayscale blur-[0.5px]",
        isYamlConfig && "border-primary/30 bg-primary/[0.02]"
      )}
    >
      {/* 뱃지 레이어 - Setup 단계에서는 안 보임, 유효한 라인 넘버(>0)만 표시 */}
      {nodeMode !== 'execute' && data.source?.line && data.source.line > 0 && (
        <div className="absolute -top-1 -right-2 flex flex-col items-end gap-1 pointer-events-none z-[110]">
          <div className={cn(
            "w-2.5 h-2.5 rounded-full border border-white/20 z-[111] absolute -top-1 left-1/2 -translate-x-1/2 transition-all shadow-sm", // 점 위치 조정
            selected ? "bg-red-500 shadow-glow-red scale-125 rotate-[4deg]" : "bg-zinc-500 rotate-[4deg]" // 점에도 rotate 적용
          )} />
          <span className={cn(
            "px-1 py-[1px] rounded-[3px] text-[10px] font-mono font-black border flex items-center transition-all shadow-xl",
            selected 
              ? "bg-primary text-primary-foreground border-primary/50 rotate-0 scale-110" 
              : "bg-zinc-900 text-zinc-100 border-zinc-700/50 rotate-[4deg]"
          )}>
            <span className="opacity-50 mr-0.5">L:</span>
            {data.source.line}
          </span>
        </div>
      )}

      {/* Header - 항목 1: 객체명(Title) / 메서드명(Subtitle Dropdown) */}
      <div className={cn(
        "relative flex flex-col px-3.5 py-2.5 border-b bg-muted/30 rounded-t-[10px] transition-colors",
        selected && "bg-muted/50"
      )}>
        <div className="flex items-center gap-3">
          <div className={cn(
            "p-1.5 rounded-lg bg-background border border-border/40 shadow-inner flex items-center justify-center shrink-0",
            theme.icon
          )}>
            <Icon className="w-4 h-4" />
          </div>
          <div className="flex flex-col overflow-hidden min-w-0">
            {/* 항목 7: 제목에 객체명 반영 (대문자 고정, 굵게) */}
            <span className="text-[13px] font-black tracking-tighter truncate text-foreground leading-none mb-1 uppercase">
              {isYamlConfig ? "GLOBAL CONFIG" : ((data as any).object_name || data.label || data.role)}
            </span>
            
            {/* Train_step phase: Method dropdown for editing */}
            {nodeMode !== 'execute' && !isYamlConfig && (
              <div className="flex items-center">
                <Select value={method} onValueChange={handleMethodChange} disabled={!isEditingMode}>
                  <SelectTrigger className={cn(
                    "h-5 py-0 px-1.5 border-none bg-primary/10 text-[9px] font-mono font-bold text-primary transition-all w-fit gap-1 rounded-md",
                    !isEditingMode && "opacity-60 cursor-default bg-muted"
                  )}>
                    <span className="truncate">{method}()</span>
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border/60">
                    <SelectItem value="forward" className="text-[10px] font-mono">forward()</SelectItem>
                    <SelectItem value="setup" className="text-[10px] font-mono">setup()</SelectItem>
                    <SelectItem value="parameters" className="text-[10px] font-mono">parameters()</SelectItem>
                    <SelectItem value="step" className="text-[10px] font-mono">step()</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Execute/Setup phase: NO SUBTITLE for Global Config, SETUP text for components */}
            {nodeMode === 'execute' && !isYamlConfig && (
               <span className="text-[10px] font-mono opacity-40 font-bold tracking-widest uppercase">
                SETUP
              </span>
            )}
          </div>
        </div>
      </div>

      {/* 실행핀 레이어 */}
      {!isConfigMode && (
        <div className="absolute top-[58px] left-0 right-0 h-0 z-[100] pointer-events-none">
          <PortRow input={execRow.input} output={execRow.output} isExecution />
        </div>
      )}

      {/* Body */}
      <div className="flex-1 py-3 px-1.5 space-y-1 bg-gradient-to-b from-background/20 to-background/50 rounded-b-[10px]">
        {/* Giant Config Node UI (ONLY in Execute/Setup mode) */}
        {isYamlConfig ? (
          <div className="px-2 space-y-3">
            {/* Config List */}
            <div className="bg-muted/20 rounded-lg p-2 border border-border/20 shadow-inner max-h-[400px] overflow-y-auto custom-scrollbar">
              <div className="flex items-center justify-between mb-3 border-b border-border/40 pb-2">
                <div className="flex items-center gap-1.5 opacity-60 px-1">
                  <Settings className="w-3.5 h-3.5" />
                  <span className="text-[10px] font-black uppercase tracking-widest">{data.metadata?.configPath || 'Configuration'}</span>
                </div>
              </div>
              <ConfigKeyValueList data={data.metadata?.configData || localConfig || {}} />
            </div>
            
            {isEditingMode && (
              <Button 
                onClick={saveConfig}
                disabled={isSaving}
                className="w-full h-8 gap-2 bg-primary text-primary-foreground hover:bg-primary/90 text-[10px] font-black uppercase tracking-widest shadow-lg shadow-primary/20"
              >
                <Save className={cn("w-3 h-3", isSaving && "animate-spin")} />
                {isSaving ? 'Saving...' : 'Save Changes'}
              </Button>
            )}
          </div>
        ) : (
          <>
            {/* Standard Node Ports - Visible in all modes but distinct content */}
            <div className="space-y-1 mb-2">
              {dataRows.map((row, i) => (
                <PortRow key={i} input={row.input} output={row.output} />
              ))}
            </div>

            {/* Used Config Keys Section - STRICTLY ONLY in Setup/Execute mode */}
            {nodeMode === 'execute' && data.usedConfigKeys && data.usedConfigKeys.length > 0 && (
              <div className="mt-3 pt-3 border-t border-white/5 px-2 space-y-2">
                <div className="flex items-center gap-1.5 mb-1 opacity-40">
                  <Settings className="w-3 h-3" />
                  <span className="text-[8px] font-black uppercase tracking-widest">Active Config</span>
                </div>
                <div className="space-y-1.5">
                  {data.usedConfigKeys.map(key => {
                    // Try to find value from GLOBAL CONFIG data if available in store
                    // For now, use local state if initialized, or fallback to metadata
                    const val = localConfig[key] ?? "";
                    
                    return (
                      <div key={key} className="group/cfg flex flex-col gap-1 p-1.5 rounded bg-primary/5 border border-primary/10 hover:border-primary/30 transition-colors">
                        <div className="flex items-center justify-between">
                          <span className="text-[9px] font-bold text-primary/60 truncate uppercase tracking-tighter">{key}</span>
                          <div className="w-1 h-1 rounded-full bg-primary/30 group-hover/cfg:bg-primary animate-pulse" />
                        </div>
                        <Input 
                          value={val}
                          onChange={(e) => handleConfigChange(key, e.target.value)}
                          disabled={!isEditingMode}
                          placeholder="value..."
                          className="h-6 text-[10px] font-mono bg-background/40 border-none shadow-none focus-visible:ring-1 focus-visible:ring-primary/30 px-1.5"
                        />
                      </div>
                    );
                  })}
                </div>
                
                {/* Individual Save for Component Context */}
                {isEditingMode && (
                  <Button 
                    onClick={saveConfig}
                    disabled={isSaving}
                    variant="ghost"
                    className="w-full h-6 mt-1 bg-primary/10 hover:bg-primary/20 text-primary text-[9px] font-black uppercase tracking-widest"
                  >
                    <Save className={cn("w-2.5 h-2.5 mr-1.5", isSaving && "animate-spin")} />
                    Update {data.label} Config
                  </Button>
                )}
              </div>
            )}

            {/* Properties Section - STRICTLY ONLY in Setup/Execute mode */}


            {nodeMode === 'execute' && data.params && data.params.length > 0 && (
              <div className="mt-2 pt-2 border-t border-white/5 px-2">
                <div className="bg-muted/40 rounded-lg p-2.5 border border-border/30 text-[11px] space-y-2.5 shadow-inner">
                  <div className="flex items-center justify-between bg-background/60 border border-border/40 rounded px-2 py-1.5 shadow-sm">
                    <span className="font-mono text-primary font-bold truncate">PROPERTIES</span>
                    <ChevronRight className="w-3.5 h-3.5 opacity-30" />
                  </div>
                  <div className="h-px bg-border/5 mx-1" />
                  {/* Property list from data.params */}
                  {data.params.slice(0, 3).map((p: any) => (
                    <div key={p.name} className="flex justify-between items-center px-1">
                      <span className="opacity-40 text-[9px] font-bold uppercase tracking-tighter">{p.name}</span>
                      <span className="font-mono text-foreground/90 font-black truncate ml-4 bg-muted/50 px-1 rounded">{String(p.value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
});

UnifiedNode.displayName = 'UnifiedNode';
