/**
 * ExecuteFlowPane - Consistent IDE-style Flow Pane for Execute Tab
 */
import { memo, useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { Loader2, Settings, Play, CheckCircle2 } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger } from '@/shared/ui/tabs';
import { useHierarchyGraph } from '@/shared/model/useHierarchyGraph';
import { useMultipleCategoryImplementations } from '@/shared/model/useCategoryImplementations';
import type { ParsedConfig } from '@/shared/lib/yaml-config-parser';
import { cn } from '@/shared/lib/utils';
import { NodeCanvas } from './NodeCanvas';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import { transformHierarchyNode, transformEdge, applyRowLayout } from '@/entities/node-system/lib/graph-utils';

interface ExecuteFlowPaneProps {
  agentName: string | null;
  parsedConfig?: ParsedConfig | null;
  configPath?: string;
  className?: string;
  onConfigChange?: (nodeId: string, changes: Record<string, unknown>) => void;
  variant?: 'full' | 'embedded';
}

const TAB_METHODS = [
  { value: 'initialize', label: 'setup()', icon: Settings, phase: 'initialize' },
  { value: 'train_step', label: 'train_step()', icon: Play, phase: 'flow' },
  { value: 'val_step', label: 'val_step()', icon: CheckCircle2, phase: 'flow' },
] as const;

function ExecuteFlowPaneInner({
  agentName,
  parsedConfig,
  configPath,
  className,
}: ExecuteFlowPaneProps) {
  const [selectedMethod, setSelectedMethod] = useState<string>('initialize');

  const setNodes = useNodeStore((s) => s.setNodes);
  const setEdges = useNodeStore((s) => s.setEdges);
  const setCodeFlowEdges = useNodeStore((s) => s.setCodeFlowEdges);
  const setTab = useNodeStore((s) => s.setTab);

  const currentOption = TAB_METHODS.find((m) => m.value === selectedMethod);
  const phase = currentOption?.phase || 'flow';

  useEffect(() => {
    setTab(phase === 'initialize' ? 'execute' : 'builder');
  }, [phase, setTab]);

  const { data: graph, isLoading, error } = useHierarchyGraph(
    agentName || '',
    undefined,
    phase as any,
    phase === 'initialize' ? undefined : selectedMethod,
    undefined,
    configPath
  );

  const transformedData = useMemo(() => {
    if (!graph?.nodes) return { nodes: [], edges: [] };
    const nodes = graph.nodes.map(n => transformHierarchyNode(n, phase === 'initialize' ? 'execute' : 'flow'));
    const edges = graph.edges.map(transformEdge).filter((e) => e !== null) as any[];
    return { nodes: applyRowLayout(nodes, edges), edges };
  }, [graph, phase]);

  const lastGraphId = useRef<string | null>(null);
  useEffect(() => {
    // 항목 4: configPath를 포함하여 고유한 그래프 ID 생성 (갱신 보장)
    const graphId = graph?.id || (graph?.nodes?.length ? `${selectedMethod}-${graph.nodes.length}-${configPath || 'no-config'}` : null);
    
    if (transformedData.nodes.length > 0 && lastGraphId.current !== graphId) {
      setNodes(transformedData.nodes);
      setEdges(transformedData.edges);
      lastGraphId.current = graphId;
    }
  }, [transformedData, graph, selectedMethod, configPath, setNodes, setEdges]);

  if (isLoading) return <div className="h-full flex items-center justify-center"><Loader2 className="animate-spin text-muted-foreground" /></div>;
  if (error) return <div className="h-full flex flex-col items-center justify-center opacity-50"><p className="text-xs font-mono">{error.message}</p></div>;

  return (
    <div className={cn('h-full flex flex-col bg-transparent', className)}>
      {/* 항목 4: 탭 네비게이션 UI 단일화 (Underline Style, h-11) */}
      <div className="h-11 border-b border-border/40 bg-muted/20 flex items-center px-4 shrink-0">
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

      <div className="flex-1 min-h-0 relative">
        <NodeCanvas 
          className="h-full" 
          editable={false} 
          elementsSelectable={true} 
          nodesDraggable={false}
        />
      </div>
    </div>
  );
}

export const ExecuteFlowPane = memo(function ExecuteFlowPane(props: ExecuteFlowPaneProps) {
  return (
    <ReactFlowProvider>
      <ExecuteFlowPaneInner {...props} />
    </ReactFlowProvider>
  );
});
