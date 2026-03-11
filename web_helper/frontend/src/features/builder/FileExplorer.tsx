import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import yaml from 'js-yaml';
/**
 * FileExplorer - Final Refined Version
 * 
 * Fixes:
 * - Precise icon alignment (Section vs Depth-0)
 * - Robust truncation (...) at any width
 * - Balanced border radii for headers and highlights
 * - Folder-first sorting and intelligent hierarchy lines
 */

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSmartCompact } from '@/shared/lib/useSmartCompact';
import {
  ChevronDown,
  ChevronRight,
  FileCode2,
  FileJson2,
  Plus,
  RefreshCw,
  Bot,
  Settings,
} from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { devWarn } from '@/shared/lib/dev-utils';
import { Button } from '@/shared/ui/button';
import { ScrollArea } from '@/shared/ui/scroll-area';
import { Input } from '@/shared/ui/input';
import { Label } from '@/shared/ui/label';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/shared/ui/dialog';
import { useAgentBuilder, AgentFile } from '@/entities/node-system/model/AgentBuilderContext';

interface FileItem {
  name: string;
  path: string;
  type: 'file' | 'folder';
  children?: FileItem[];
}

// Utility to build a recursive file tree from flat paths with Folder-First sorting
function buildFileTree(files: FileItem[]): FileItem[] {
  const root: FileItem[] = [];
  const map: Record<string, FileItem> = {};

  const sortedFiles = [...files].sort((a, b) => a.path.localeCompare(b.path));

  sortedFiles.forEach(file => {
    const parts = file.path.split('/');
    let currentPath = '';
    let currentLevel = root;

    parts.forEach((part, index) => {
      const isFile = index === parts.length - 1;
      currentPath = currentPath ? `${currentPath}/${part}` : part;

      if (isFile) {
        currentLevel.push({ ...file, name: part });
      } else {
        if (!map[currentPath]) {
          const newFolder: FileItem = {
            name: part,
            path: currentPath,
            type: 'folder',
            children: [],
          };
          map[currentPath] = newFolder;
          currentLevel.push(newFolder);
        }
        currentLevel = map[currentPath].children!;
      }
    });
  });

  const sortTree = (items: FileItem[]) => {
    items.sort((a, b) => {
      if (a.type !== b.type) return a.type === 'folder' ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    items.forEach(item => {
      if (item.children) sortTree(item.children);
    });
  };

  sortTree(root);
  return root;
}

interface FileTreeItemProps {
  item: FileItem;
  fileType: 'agent' | 'config';
  depth?: number;
  selectedPath?: string;
  onSelect: (file: AgentFile) => void;
}

function FileTreeItem({
  item,
  fileType,
  depth = 0,
  selectedPath,
  onSelect,
}: FileTreeItemProps) {
  const [expanded, setExpanded] = useState(true);
  const isSelected = item.type === 'file' && item.path === selectedPath;
  const isFolder = item.type === 'folder';

  const handleClick = () => {
    if (isFolder) {
      setExpanded(!expanded);
    } else {
      onSelect({
        type: fileType,
        path: item.path,
        name: item.name,
      });
    }
  };

  const FileIcon = fileType === 'agent' ? FileCode2 : FileJson2;
  const displayName = item.type === 'file'
    ? item.name.replace(/\.(py|yaml|yml)$/, '')
    : item.name;

  // Exact alignment: Depth-0 items should have chevron exactly under section chevron
  // Line center is at 15px.
  const linePos = (depth - 1) * 16 + 15;
  const itemMarginLeft = depth > 0 ? linePos + 5 : 0;
  const itemPaddingLeft = 12;

  return (
    <div className="w-full min-w-0 relative">
      <button
        onClick={handleClick}
        className={cn(
          'flex items-center gap-2 py-1 pr-2 text-sm text-left rounded-lg transition-all duration-150 min-w-0 group relative',
          'hover:bg-accent/50 hover:text-accent-foreground',
          isSelected && 'bg-primary/10 text-primary font-semibold',
          !isSelected && 'text-muted-foreground'
        )}
        style={{
          marginLeft: `${itemMarginLeft}px`,
          paddingLeft: `${itemPaddingLeft}px`,
          width: `calc(100% - ${itemMarginLeft}px)`
        }}
      >
        {isFolder ? (
          <div className="flex items-center gap-1 min-w-0 flex-1">
            {expanded ? (
              <ChevronDown className="h-4 w-4 flex-shrink-0 text-muted-foreground/70 -ml-1" />
            ) : (
              <ChevronRight className="h-4 w-4 flex-shrink-0 text-muted-foreground/70 -ml-1" />
            )}
            <span className="font-bold text-foreground/90 leading-tight min-w-0 flex-1 block overflow-hidden whitespace-nowrap text-ellipsis">{displayName}</span>
          </div>
        ) : (
          <div className="flex items-center gap-1 min-w-0 flex-1">
            <div className="w-4 h-4 flex-shrink-0 flex items-center justify-center">
              <FileIcon className={cn("h-4 w-4 flex-shrink-0", isSelected ? "text-primary" : "opacity-50")} />
            </div>
            <span className="leading-tight min-w-0 flex-1 block overflow-hidden whitespace-nowrap text-ellipsis">{displayName}</span>
          </div>
        )}
      </button>

      {isFolder && expanded && item.children && item.children.length > 0 && (
        <div className="relative overflow-hidden">
          <div 
            className="absolute left-0 top-0 bottom-0 w-px bg-border/20" 
            style={{ marginLeft: `${depth * 16 + 15}px` }}
          />
          {item.children.map((child) => (
            <FileTreeItem
              key={`${fileType}-${child.path}`}
              item={child}
              fileType={fileType}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface SectionProps {
  title: string;
  icon: React.ElementType;
  items: FileItem[];
  fileType: 'agent' | 'config';
  selectedPath?: string;
  onSelect: (file: AgentFile) => void;
  onRefresh?: () => void;
  onAdd?: () => void;
  isCompact?: boolean;
}

function Section({
  title,
  icon: Icon,
  items,
  fileType,
  selectedPath,
  onSelect,
  onRefresh,
  onAdd,
  isCompact,
}: SectionProps) {
  const [expanded, setExpanded] = useState(true);
  const treeItems = useMemo(() => buildFileTree(items), [items]);

  return (
    <div className="mb-6 w-full min-w-0 overflow-hidden">
      {/* Section Header - Fixed alignment and curvature */}
      <div className="flex items-center group/header w-full pl-4 pr-2 bg-muted/20 border-y border-white/5 mb-2 shadow-sm rounded-md overflow-hidden">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 py-2.5 text-[11px] font-black text-foreground/80 hover:text-foreground transition-colors min-w-0 flex-1 uppercase tracking-[0.15em]"
        >
          {expanded ? (
            <ChevronDown className="h-4 w-4 flex-shrink-0 text-primary/60 -ml-1" />
          ) : (
            <ChevronRight className="h-4 w-4 flex-shrink-0 text-primary/60 -ml-1" />
          )}
          <Icon className={cn("text-primary flex-shrink-0", fileType === 'config' ? "h-5 w-5" : "h-6 w-6")} />
          <span className="min-w-0 flex-1 block overflow-hidden whitespace-nowrap text-ellipsis text-left text-xs">{title}</span>
        </button>

        {!isCompact && (
          <div className="flex items-center gap-1 flex-shrink-0">
            {onRefresh && (
              <Button variant="ghost" size="icon" className="h-7 w-7 hover:bg-muted rounded-md flex-shrink-0" onClick={(e) => { e.stopPropagation(); onRefresh(); }}>
                <RefreshCw className="h-3.5 w-3.5" />
              </Button>
            )}
            {onAdd && (
              <Button variant="ghost" size="icon" className="h-7 w-7 hover:bg-muted rounded-md flex-shrink-0" onClick={(e) => { e.stopPropagation(); onAdd(); }}>
                <Plus className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        )}
      </div>

      {/* File Tree - Fixed padding for alignment */}
      {expanded && (
        <div className="min-w-0 px-1 overflow-hidden">
          {items.length === 0 ? (
            <div className="pl-10 py-3 text-[10px] text-muted-foreground/40 italic uppercase tracking-tighter truncate">Empty</div>
          ) : (
            treeItems.map((item, idx) => (
              <FileTreeItem
                key={`${fileType}-${item.path}-${idx}`}
                item={item}
                fileType={fileType}
                selectedPath={selectedPath}
                onSelect={onSelect}
                depth={0}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

// API fetch functions remain the same...
async function fetchAgentFiles(): Promise<FileItem[]> {
  try {
    const response = await fetch('/api/nodes/agents');
    if (!response.ok) throw new Error('Failed to fetch agents');
    const result = await response.json();
    const agentList = result.agents || [];
    return agentList.map((agentName: string) => ({
      name: `${agentName}.py`,
      path: `${agentName}.py`,
      type: 'file' as const,
    }));
  } catch (error) {
    devWarn('Failed to fetch agent files:', error);
    return [];
  }
}

async function fetchConfigFiles(): Promise<FileItem[]> {
  try {
    const response = await fetch('/api/configs');
    if (!response.ok) throw new Error('Failed to fetch configs');
    const result = await response.json();
    return (result.data || []).map((configPath: string) => ({
      name: configPath,
      path: configPath,
      type: 'file' as const,
    }));
  } catch (error) {
    devWarn('Failed to fetch config files:', error);
    return [];
  }
}

export function FileExplorer() {
  const builderContext = useAgentBuilder();
  const selectedAgent = builderContext?.selectedAgent;
  const selectedConfig = builderContext?.selectedConfig;
  const setSelectedAgent = builderContext?.setSelectedAgent || (() => {});
  const setSelectedConfig = builderContext?.setSelectedConfig || (() => {});
  const setSelectedFile = builderContext?.setSelectedFile || (() => {});
  const refreshGraph = builderContext?.refreshGraph || (() => {});

  const { containerRef: explorerRef, fullSizerRef: explorerFullSizerRef, compactSizerRef: explorerCompactSizerRef, isCompact } = useSmartCompact();

  const [newAgentDialogOpen, setNewAgentDialogOpen] = useState(false);
  const [newConfigDialogOpen, setNewConfigDialogOpen] = useState(false);
  const [newFileName, setNewFileName] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const { data: agentFiles = [], refetch: refetchAgents } = useQuery({
    queryKey: ['builder-agents'],
    queryFn: fetchAgentFiles,
    staleTime: 30000,
  });

  const { data: configFiles = [], refetch: refetchConfigs } = useQuery({
    queryKey: ['builder-configs'],
    queryFn: fetchConfigFiles,
    staleTime: 30000,
  });

  const handleSelectAgent = (file: AgentFile) => {
    setSelectedAgent(file);
    setSelectedFile(file);
    setTimeout(() => refreshGraph(), 100);
  };

  const handleSelectConfig = (file: AgentFile) => {
    setSelectedConfig(file);

    // Req 5: Load config file and update Config Node
    const loadAndUpdateConfigNode = async () => {
      try {
        // Load config file content
        const response = await fetch(`/api/configs/${file.path.replace(/^config\//, '')}`);
        if (!response.ok) {
          console.error('Failed to load config file');
          return;
        }

        const result = await response.json();
        if (result.data) {
          // Update the config node directly in the store instead of refreshing the graph
          const updateNode = useNodeStore.getState().updateNode;
          const nodes = useNodeStore.getState().nodes;
          
          const configNode = nodes.find((n) => String(n.data.category) === 'config' || n.data.origin?.type === 'config' || n.id === 'config_1' || n.id === 'yaml_config');

          
          if (configNode) {
            updateNode(configNode.id, {
              metadata: {
                ...configNode.data.metadata,
                configPath: file.path,
                configData: typeof result.data.content === 'string' ? yaml.load(result.data.content.replace(/!!python\/[^\s]+/g, ''), { schema: yaml.FAILSAFE_SCHEMA }) as Record<string, any> : result.data
              },
              properties: [
                {
                  name: 'file',
                  value: file.name,
                  source: 'required'
                },
                ...(configNode.data.properties?.filter((p: { name: string }) => p.name !== 'file') || [])
              ]
            });
          } else {
            // If there's no config node yet, fetch the whole graph
            refreshGraph();
          }
        }
      } catch (error) {
        console.error('Failed to load config file:', error);
      }
    };

    loadAndUpdateConfigNode();
  };

  const handleNewAgent = () => {
    setNewFileName('');
    setCreateError(null);
    setNewAgentDialogOpen(true);
  };

  const handleNewConfig = () => {
    setNewFileName('');
    setCreateError(null);
    setNewConfigDialogOpen(true);
  };

  const handleCreateAgent = async () => {
    if (!newFileName.trim()) { setCreateError('Name is required'); return; }
    const cleanName = newFileName.trim().replace(/\.py$/, '');
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(cleanName)) { setCreateError('Invalid name'); return; }
    setIsCreating(true);
    try {
      const response = await fetch('/api/nodes/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_name: cleanName, node_graph: { nodes: [], edges: [] }, mode: 'create' }),
      });
      if (!response.ok) throw new Error('Failed to create');
      await refetchAgents();
      handleSelectAgent({ type: 'agent', path: `cvlabkit/agent/${cleanName}.py`, name: `${cleanName}.py` });
      setNewAgentDialogOpen(false);
    } catch (error) { setCreateError('Error creating'); } finally { setIsCreating(false); }
  };

  const handleCreateConfig = async () => {
    if (!newFileName.trim()) { setCreateError('Name is required'); return; }
    const cleanName = newFileName.trim().replace(/\.ya?ml$/, '');
    setIsCreating(true);
    try {
      const response = await fetch('/api/configs/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: cleanName, content: `# ${cleanName}\nagent: classification\n` }),
      });
      if (!response.ok) throw new Error('Failed to create');
      await refetchConfigs();
      handleSelectConfig({ type: 'config', path: `config/${cleanName}.yaml`, name: `${cleanName}.yaml` });
      setNewConfigDialogOpen(false);
    } catch (error) { setCreateError('Error creating'); } finally { setIsCreating(false); }
  };

  return (
    <div ref={explorerRef} className="h-full flex flex-col bg-transparent w-full min-w-0 overflow-hidden relative">
      {/* Full sizer: chevron + icon + text + buttons */}
      <div
        ref={explorerFullSizerRef}
        aria-hidden
        className="absolute top-0 left-0 pointer-events-none flex items-center pl-4 pr-2"
        style={{ visibility: 'hidden', whiteSpace: 'nowrap' }}
      >
        <span className="inline-flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.15em]">
          <ChevronDown className="h-4 w-4" />
          <Bot className="h-6 w-6" />
          <span>Agents</span>
        </span>
        <span className="inline-flex items-center gap-1 ml-2">
          <span className="inline-block h-7 w-7" />
          <span className="inline-block h-7 w-7" />
        </span>
      </div>
      {/* Compact sizer: chevron + icon only (no text, no buttons) */}
      <div
        ref={explorerCompactSizerRef}
        aria-hidden
        className="absolute top-0 left-0 pointer-events-none flex items-center pl-4 pr-2"
        style={{ visibility: 'hidden', whiteSpace: 'nowrap' }}
      >
        <span className="inline-flex items-center gap-2">
          <ChevronDown className="h-4 w-4" />
          <Bot className="h-6 w-6" />
        </span>
      </div>
      <style>{`
        .faint-scrollbar::-webkit-scrollbar { width: 4px; height: 4px; }
        .faint-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .faint-scrollbar::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.05); border-radius: 10px; }
        .faint-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.1); }
        /* Override Radix ScrollArea's inner wrapper that uses display:table,
           which expands to content width and prevents text truncation */
        .faint-scrollbar [data-radix-scroll-area-viewport] > div {
          display: block !important;
          min-width: 0 !important;
        }
      `}</style>

      {(selectedAgent || selectedConfig) && (
        <div className="px-3 py-1.5 border-b border-white/5 bg-accent/5 text-[10px] shrink-0 w-full overflow-hidden rounded-md">
          <div className="space-y-0.5 w-full">
            {selectedAgent && (
              <div className="flex items-center gap-1.5 w-full min-w-0">
                <Bot className="h-3 w-3 text-blue-500 flex-shrink-0" />
                <span className="text-muted-foreground flex-shrink-0">Agent:</span>
                <span className="font-medium min-w-0 flex-1 overflow-hidden whitespace-nowrap text-ellipsis">{selectedAgent.name.replace('.py', '')}</span>
              </div>
            )}
            {selectedConfig && (
              <div className="flex items-center gap-1.5 w-full min-w-0">
                <Settings className="h-3 w-3 text-amber-500 flex-shrink-0" />
                <span className="text-muted-foreground flex-shrink-0">Config:</span>
                <span className="font-medium min-w-0 flex-1 overflow-hidden whitespace-nowrap text-ellipsis">{selectedConfig.name.replace(/\.ya?ml$/, '')}</span>
              </div>
            )}
          </div>
        </div>
      )}

      <ScrollArea className="flex-1 w-full min-w-0 faint-scrollbar overflow-hidden [&>[data-radix-scroll-area-viewport]]:!overflow-x-hidden">
        <div className="py-2 w-full min-w-0 overflow-hidden">
          <Section title="Agents" icon={Bot} items={agentFiles} fileType="agent" selectedPath={selectedAgent?.path} onSelect={handleSelectAgent} onRefresh={() => refetchAgents()} onAdd={handleNewAgent} isCompact={isCompact} />
          <Section title="Configs" icon={Settings} items={configFiles} fileType="config" selectedPath={selectedConfig?.path} onSelect={handleSelectConfig} onRefresh={() => refetchConfigs()} onAdd={handleNewConfig} isCompact={isCompact} />
        </div>
      </ScrollArea>

      <Dialog open={newAgentDialogOpen} onOpenChange={setNewAgentDialogOpen}>
        <DialogContent size="sm">
          <DialogHeader><DialogTitle>New Agent</DialogTitle></DialogHeader>
          <div className="grid gap-2 py-4">
            <Label>Agent Name</Label>
            <Input placeholder="my_agent" value={newFileName} onChange={(e) => setNewFileName(e.target.value)} />
            {createError && <p className="text-xs text-destructive">{createError}</p>}
          </div>
          <DialogFooter><Button onClick={handleCreateAgent} disabled={isCreating}>Create</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={newConfigDialogOpen} onOpenChange={setNewConfigDialogOpen}>
        <DialogContent size="sm">
          <DialogHeader><DialogTitle>New Config</DialogTitle></DialogHeader>
          <div className="grid gap-2 py-4">
            <Label>Config Name</Label>
            <Input placeholder="my-config" value={newFileName} onChange={(e) => setNewFileName(e.target.value)} />
            {createError && <p className="text-xs text-destructive">{createError}</p>}
          </div>
          <DialogFooter><Button onClick={handleCreateConfig} disabled={isCreating}>Create</Button></DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
