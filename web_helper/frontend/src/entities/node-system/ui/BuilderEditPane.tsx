/**
 * BuilderEditPane - Unified Node System
 */
import { useMemo, useEffect, useRef, memo } from 'react';
import { ReactFlowProvider, useReactFlow } from 'reactflow';
import {
  Loader2,
  Plus,
  RotateCcw,
  Save,
  X,
  AlertCircle,
  Lock,
  Monitor,
} from 'lucide-react';
import { Button } from '@/shared/ui/button';
import { Badge } from '@/shared/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/shared/ui/tooltip';

import {
  useAgentBuilder,
  type ComponentInfo,
} from '@/entities/node-system/model/AgentBuilderContext';
import { useHierarchyNavigation } from '@/shared/model/useHierarchyGraph';
import { isValidConnection as checkPortCompatibility } from '@/shared/config/port-themes';
import { PortType } from '@/entities/node-system/model/port';
import { cn } from '@/shared/lib/utils';

import { NodeCanvas } from './NodeCanvas';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import { PinLegend } from './PinLegend';
import { transformHierarchyNode, transformEdge, applyRowLayout } from '@/entities/node-system/lib/graph-utils';

function BuilderEditPaneInner({
  className,
  phase = 'initialize',
  method,
}: { className?: string; phase?: string; method?: string }) {
  const {
    selectedAgent,
    selectedConfig,
    isEditingMode,
    localNodes,
    localEdges,
    setLocalNodes,
    setLocalEdges,
    addNode,
    removeNode,
    addConnection,
    removeEdge,
    syncNodeToCode,
  } = useAgentBuilder();

  const setNodes = useNodeStore(s => s.setNodes);
  const setEdges = useNodeStore(s => s.setEdges);
  const setTab = useNodeStore(s => s.setTab);
  const setCodeFlowEdges = useNodeStore(s => s.setCodeFlowEdges);
  const currentTab = useNodeStore(s => s.currentTab);
  const draftState = useNodeStore(s => s.draftState);

  const agentName = selectedAgent?.name?.replace(/\.(py|yaml|yml)$/, '') || null;

  // Map currentTab to API phase
  // setup/execute tabs → 'initialize' (component dependencies)
  // builder/flow tabs → 'flow' (data flow, default method='train_step')
  const effectivePhase = useMemo(() => {
    switch (currentTab) {
      case 'execute':
      case 'setup':
        return 'initialize';
      case 'builder':
      case 'flow':
        return 'flow';
      default:
        return phase as any || 'initialize';
    }
  }, [currentTab, phase]);

  // Default method for flow phase
  const effectiveMethod = useMemo(() => {
    if (effectivePhase === 'flow') {
      return method || 'train_step';
    }
    return method;
  }, [effectivePhase, method]);

  const { graph, isLoading, error, reset } = useHierarchyNavigation(
    agentName || '',
    effectivePhase,
    effectiveMethod,
    selectedConfig?.path
  );

  const transformedData = useMemo(() => {
    if (!graph?.nodes) return { nodes: [], edges: [] };
    const nodes = graph.nodes.map(n => transformHierarchyNode(n, isEditingMode ? 'builder' : 'execute'));
    const edges = graph.edges.map(transformEdge).filter((e) => e !== null) as any[];
    return { nodes: applyRowLayout(nodes, edges), edges };
  }, [graph, isEditingMode]);

  const lastGraphId = useRef<string | null>(null);
  useEffect(() => {
    reset();
    lastGraphId.current = null;
  }, [agentName, effectivePhase, effectiveMethod, selectedConfig?.path, reset]);

  useEffect(() => {
    const graphId = graph?.id || (graph?.nodes?.length ? 'has-data' : null);
    if (transformedData.nodes.length > 0 && lastGraphId.current !== graphId) {
      setNodes(transformedData.nodes);
      setEdges(transformedData.edges);
      // setCodeFlowEdges(toCodeFlowEdgesFromReactFlowEdges(transformedData.edges));
      lastGraphId.current = graphId;
    }
  }, [transformedData, graph?.id, graph?.nodes?.length, setNodes, setEdges]);

  // Handle reset layout
  const handleResetLayout = () => {
    const currentNodes = useNodeStore.getState().nodes;
    const currentEdges = useNodeStore.getState().edges;
    // 항목 10: 수치적으로 정밀한 새로운 레이아웃 적용
    const layoutedNodes = applyRowLayout(currentNodes, currentEdges, {});
    setNodes(layoutedNodes);
  };

  if (!agentName) {
    return (
      <div className="h-full flex flex-col items-center justify-center opacity-30">
        <Monitor className="w-16 h-16 mb-4" />
        <h3 className="text-lg font-black tracking-tighter uppercase text-muted-foreground">Select Agent to Visualize</h3>
      </div>
    );
  }

  if (isLoading) return <div className="h-full flex items-center justify-center"><Loader2 className="animate-spin h-8 w-8 text-muted-foreground" /></div>;
  if (error) return (
    <div className="h-full flex flex-col items-center justify-center text-destructive p-4 text-center">
      <AlertCircle className="w-12 h-12 mb-3 opacity-50" />
      <h3 className="text-sm font-bold uppercase tracking-widest mb-1">Error Loading Graph</h3>
      <p className="text-xs text-muted-foreground max-w-xs leading-relaxed">
        {error instanceof Error ? error.message : 'The system encountered an issue while retrieving the agent architecture.'}
      </p>
    </div>
  );

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex-1 relative">
        <NodeCanvas
          className="h-full"
          editable={isEditingMode && currentTab === 'builder'}
          nodesDraggable={isEditingMode}
          elementsSelectable={true}
          onNodeSelect={n => syncNodeToCode(n.id)}
          initialFitTrigger={`${agentName ?? ''}:${effectivePhase}:${effectiveMethod ?? ''}`}
        />
        <div className="absolute top-4 right-4 z-50">
           <Button variant="ghost" size="icon" onClick={handleResetLayout} title="Reset Layout">
             <RotateCcw className="h-4 w-4" />
           </Button>
        </div>
      </div>
    </div>
  );
}

export const BuilderEditPane = memo(({ className, phase, method }: { className?: string; phase?: string; method?: string }) => (
  <ReactFlowProvider>
    <BuilderEditPaneInner className={className} phase={phase} method={method} />
  </ReactFlowProvider>
));
