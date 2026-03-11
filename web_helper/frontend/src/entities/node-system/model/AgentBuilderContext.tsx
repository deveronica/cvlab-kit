/**
 * AgentBuilderContext - Rigorous State Management
 */
import React, { createContext, useContext, useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { Node as ReactFlowNode, Edge as ReactFlowEdge, Connection, addEdge } from 'reactflow';
import type { NodeGraph } from '@/entities/node-system/model/types';
import type { ComponentCategory } from '@/shared/model/hierarchy';
import { useBuilderStore } from '@/features/builder/model/builderStore';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import type { ConfigValue } from '@/shared/model/config-types';

export interface AgentFile {
  type: 'agent' | 'config';
  path: string;
  name: string;
  content?: string;
}

export interface CodeNodeMapping {
  nodeId: string;
  lineStart: number;
  lineEnd: number;
  type: 'component_create' | 'method_call' | 'assignment';
}

export interface ComponentViewerState {
  isOpen: boolean;
  category?: string;
  name?: string;
  implementation?: string;
}

export interface ComponentInfo {
  name: string;
  category: ComponentCategory;
  description?: string;
  parameters?: string[];
}

interface AgentBuilderState {
  selectedAgent: AgentFile | null;
  setSelectedAgent: (file: AgentFile | null) => void;
  // Alias for generic file selection (usually agent)
  selectedFile: AgentFile | null;
  setSelectedFile: (file: AgentFile | null) => void;
  
  selectedConfig: AgentFile | null;
  setSelectedConfig: (file: AgentFile | null) => void;
  isEditingMode: boolean;
  setIsEditingMode: (editing: boolean) => void;
  isDirty: boolean; // 항목 7: 수정 사항 여부 추적
  localNodes: ReactFlowNode[];
  setLocalNodes: React.Dispatch<React.SetStateAction<ReactFlowNode[]>>;
  localEdges: ReactFlowEdge[];
  setLocalEdges: React.Dispatch<React.SetStateAction<ReactFlowEdge[]>>;
  nodeGraph: NodeGraph | null;
  selectedNodeId: string | null;
  code: string;
  setCode: (code: string) => void;
  highlightedLines: number[];
  setHighlightedLines: (lines: number[]) => void;
  syncNodeToCode: (nodeId: string) => void;
  syncCodeToNode: (line: number) => void;
  refreshGraph: () => Promise<void>;
  saveToCode: () => Promise<void>;
  
  // Graph Actions
  addNode: (component: ComponentInfo, position?: { x: number; y: number }) => void;
  addControlFlowNode: (nodeType: string, data: Record<string, unknown>, position?: { x: number; y: number }) => void;
  removeNode: (nodeId: string) => Promise<void> | void;
  addConnection: (connection: Connection) => void;
  removeEdge: (edgeId: string) => void;

  // Component Viewer
  componentViewer: ComponentViewerState;
  openComponentViewer: (category: string, name: string) => void;
  closeComponentViewer: () => void;
}

const AgentBuilderContext = createContext<AgentBuilderState | null>(null);

export function AgentBuilderProvider({ children }: { children: React.ReactNode }) {
  const selectedAgent = useBuilderStore((s) => s.selectedAgent);
  const setSelectedAgent = useBuilderStore((s) => s.setSelectedAgent);
  const selectedConfig = useBuilderStore((s) => s.selectedConfig);
  const setSelectedConfig = useBuilderStore((s) => s.setSelectedConfig);
  const isEditingMode = useBuilderStore((s) => s.isEditingMode);
  const setIsEditingMode = useBuilderStore((s) => s.setIsEditingMode);
  const syncState = useBuilderStore((s) => s.syncState);
  const setIsDirty = useBuilderStore((s) => s.setIsDirty);

  const [localNodes, setLocalNodes] = useState<ReactFlowNode[]>([]);
  const [localEdges, setLocalEdges] = useState<ReactFlowEdge[]>([]);
  const [code, setCode] = useState('');
  const [highlightedLines, setHighlightedLines] = useState<number[]>([]);
  const [nodeGraph, setNodeGraph] = useState<NodeGraph | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  
  // Dirty State Tracking
  const initialCodeRef = useRef('');

  // Component Viewer State
  const [componentViewer, setComponentViewer] = useState<ComponentViewerState>({ isOpen: false });

  const mappings = useBuilderStore((s) => s.codeNodeMappings);
  const selectNodeInStore = useNodeStore((s) => s.selectNode);

  // 항목 7: 수정 사항 감지 로직
  useEffect(() => {
    if (!isEditingMode) {
      setIsDirty(false);
      return;
    }
    const hasCodeChanged = code !== initialCodeRef.current;
    setIsDirty(hasCodeChanged);
  }, [code, isEditingMode, setIsDirty]);

  // Fetch code content when selected file changes
  useEffect(() => {
    const fetchCode = async () => {
      if (selectedAgent) {
        try {
          const agentName = selectedAgent.name.replace(/\.(py|yaml|yml)$/, '');
          const res = await fetch(`/api/nodes/agent-source/${agentName}`);
          const data = await res.json();
          if (data.success) {
            setCode(data.source);
            initialCodeRef.current = data.source;
            setIsDirty(false);
          }
        } catch (e) {
          console.error('Failed to fetch agent source:', e);
        }
      } else if (selectedConfig) {
        try {
          const configPath = selectedConfig.path.replace(/^config\//, '');
          const res = await fetch(`/api/configs/${configPath}`);
          if (!res.ok) {
            throw new Error(`Failed to fetch config content (${res.status})`);
          }
          const data = await res.json();
          if (data?.data?.content) {
            setCode(data.data.content);
            initialCodeRef.current = data.data.content;
            setIsDirty(false);
          }
        } catch (e) {
          console.error('Failed to fetch config content:', e);
        }
      } else {
        setCode('');
        initialCodeRef.current = '';
        setIsDirty(false);
      }
    };

    fetchCode();
  }, [selectedAgent, selectedConfig, setIsDirty]);

  const handleSetSelectedConfig = useCallback((file: AgentFile | null) => {
    if (selectedConfig?.path === file?.path) setSelectedConfig(null);
    else setSelectedConfig(file);
  }, [selectedConfig, setSelectedConfig]);

  const handleSetSelectedAgent = useCallback((file: AgentFile | null) => {
    if (selectedAgent?.path === file?.path) return;
    setSelectedAgent(file);
  }, [selectedAgent, setSelectedAgent]);

  const syncNodeToCode = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId);
    selectNodeInStore(nodeId);
    const mapping = mappings.find(m => m.nodeIds.includes(nodeId));
    if (mapping) {
      const lines = [];
      for (let i = mapping.startLine; i <= mapping.endLine; i++) lines.push(i);
      setHighlightedLines(lines);
    }
  }, [mappings, selectNodeInStore]);

  const syncCodeToNode = useCallback((line: number) => {
    const mapping = mappings.find(m => line >= m.startLine && line <= m.endLine);
    if (mapping && mapping.nodeIds.length > 0) {
      const nodeId = mapping.nodeIds[0];
      setSelectedNodeId(nodeId);
      selectNodeInStore(nodeId);
    }
  }, [mappings, selectNodeInStore]);

  const refreshGraph = useCallback(async () => {
    if (!selectedAgent) return;
    try {
      const name = selectedAgent.name.replace(/\.(py|yaml|yml)$/, '');
      const res = await fetch(`/api/nodes/hierarchy/${name}${selectedConfig ? `?config_path=${selectedConfig.path}` : ''}`);
      const data = await res.json();
      if (data.success) setNodeGraph(data.data);
    } catch (e) { console.error(e); }
  }, [selectedAgent, selectedConfig]);

  const saveToCode = async () => {
    setIsEditingMode(false);
    setIsDirty(false);
    initialCodeRef.current = code;
  };

  const addNode = useCallback((comp: ComponentInfo, pos?: { x: number; y: number }) => {
    console.log('Adding node', comp);
    setIsDirty(true);
  }, [setIsDirty]);

  const addControlFlowNode = useCallback((type: string, data: any, pos?: { x: number; y: number }) => {
    console.log('Adding control flow', type);
    setIsDirty(true);
  }, [setIsDirty]);

  const removeNode = useCallback((id: string) => {
    console.log('Removing node', id);
    setIsDirty(true);
  }, [setIsDirty]);

  const addConnection = useCallback((connection: Connection) => {
    setLocalEdges((eds) => addEdge(connection, eds));
    setIsDirty(true);
  }, [setIsDirty]);

  const removeEdge = useCallback((edgeId: string) => {
    setLocalEdges((eds) => eds.filter(e => e.id !== edgeId));
    setIsDirty(true);
  }, [setIsDirty]);

  const openComponentViewer = useCallback((category: string, name: string) => {
    setComponentViewer({ isOpen: true, category, name });
  }, []);

  const closeComponentViewer = useCallback(() => {
    setComponentViewer(prev => ({ ...prev, isOpen: false }));
  }, []);

  const value: AgentBuilderState = {
    selectedAgent, setSelectedAgent: handleSetSelectedAgent,
    selectedFile: selectedAgent, setSelectedFile: handleSetSelectedAgent, // Alias
    selectedConfig, setSelectedConfig: handleSetSelectedConfig,
    isEditingMode, setIsEditingMode,
    isDirty: syncState.isDirty,
    localNodes, setLocalNodes, localEdges, setLocalEdges,
    nodeGraph, selectedNodeId,
    code, setCode, highlightedLines, setHighlightedLines,
    syncNodeToCode, syncCodeToNode, refreshGraph, saveToCode,
    addNode, addControlFlowNode, removeNode,
    addConnection, removeEdge,
    componentViewer, openComponentViewer, closeComponentViewer
  };

  return <AgentBuilderContext.Provider value={value}>{children}</AgentBuilderContext.Provider>;
}

export function useAgentBuilder() {
  const context = useContext(AgentBuilderContext);
  return context;
}
