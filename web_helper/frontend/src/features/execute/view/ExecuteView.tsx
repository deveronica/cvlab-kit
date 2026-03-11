/**
 * ExecuteView - Professional Card-style IDE for Execution
 */
import React, { useEffect, useCallback, memo, useState, useMemo, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import CodeMirror from '@uiw/react-codemirror';
import { yaml } from '@codemirror/lang-yaml';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import { EditorView } from '@codemirror/view';
import {
  Code2,
  ListPlus,
  Info,
  Monitor,
  ChevronRight,
  Terminal,
  Activity,
  Cpu,
  Loader2,
} from 'lucide-react';
import {
  parseYamlConfig,
  updateYamlComponent,
  updateYamlParam,
} from '@/shared/lib/yaml-config-parser';
import { Button } from '@/shared/ui/button';
import { Badge } from '@/shared/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/shared/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/shared/ui/tooltip';
import { Separator } from '@/shared/ui/separator';
import { useDevices } from '@/shared/model/useDevices';
import { useExecute } from '@/app/ui/ExecuteContext';
import { cn } from '@/shared/lib/utils';
import { ExecuteFlowPane } from '@/entities/node-system/ui';
import { ConfigSelector } from '@/shared/ui/ConfigSelector';
import { ThreePanelLayout } from '@/shared/ui/ThreePanelLayout';
import { RightRail, type RightRailTab } from '@/shared/ui/RightRail';
import { ExecuteOptionsPanel } from '../panels/ExecuteOptionsPanel';
import { ExecuteInspector } from '../panels/ExecuteInspector';

// API fetch functions
const fetchConfigList = async (): Promise<string[]> => {
  const response = await fetch('/api/configs');
  const data = await response.json();
  return data.data || [];
};

const fetchConfigContent = async (configPath: string): Promise<string> => {
  const response = await fetch(`/api/configs/${encodeURIComponent(configPath)}`);
  const data = await response.json();
  return data.data?.content || '';
};

function countGridSearchExperiments(yamlContent: string): number {
  const listPattern = /:\s*\[.+\]/g;
  const matches = yamlContent.match(listPattern) || [];
  let total = 1;
  for (const match of matches) {
    const items = match.match(/\[(.+)\]/)?.[1]?.split(',') || [];
    if (items.length > 1) total *= items.length;
  }
  return total;
}

const ExecuteView = memo(function ExecuteView() {
  const {
    state,
    updateYamlConfig,
    updateSelectedConfigFile,
    updateSelectedDevice,
    updateSelectedPriority,
    updateLogMessages,
  } = useExecute();

  const { data: configFiles = [], isLoading: isLoadingConfigs, error: configsError } = useQuery<string[], Error>({
    queryKey: ['configFiles'],
    queryFn: fetchConfigList,
  });
  const { data: devices = [], isLoading: isLoadingDevices } = useDevices();

  const [isStarting, setIsStarting] = React.useState(false);
  const [isValidating, setIsValidating] = React.useState(false);
  const lastLoadedConfigRef = useRef<string>('');

  const { data: configContent } = useQuery<string | null, Error>({
    queryKey: ['configContent', state.selectedConfigFile],
    queryFn: () => fetchConfigContent(state.selectedConfigFile),
    enabled: !!state.selectedConfigFile,
  });

  const parsedConfig = useMemo(() => state.yamlConfig ? parseYamlConfig(state.yamlConfig) : null, [state.yamlConfig]);
  const agentName = parsedConfig?.agent || null;
  const gridSearchCount = useMemo(() => countGridSearchExperiments(state.yamlConfig), [state.yamlConfig]);

  const handleFlowConfigChange = useCallback((nodeId: string, changes: Record<string, unknown>) => {
    if (!parsedConfig) return;
    let newYaml = state.yamlConfig;
    if ('implementation' in changes) {
      const impl = changes.implementation as string;
      newYaml = updateYamlComponent(newYaml, nodeId, nodeId, impl);
    }
    for (const [paramName, value] of Object.entries(changes)) {
      if (paramName === 'implementation') continue;
      newYaml = updateYamlParam(newYaml, paramName, value);
    }
    updateYamlConfig(newYaml);
  }, [parsedConfig, state.yamlConfig, updateYamlConfig]);

  useEffect(() => {
    if (configContent && state.selectedConfigFile !== lastLoadedConfigRef.current) {
      updateYamlConfig(configContent);
      updateLogMessages([]);
      lastLoadedConfigRef.current = state.selectedConfigFile;
    }
  }, [configContent, state.selectedConfigFile, updateYamlConfig, updateLogMessages]);

  const handleStartExperiment = async () => {
    setIsStarting(true);
    updateLogMessages(['Starting experiment...']);
    try {
      const response = await fetch('/api/queue/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: state.yamlConfig, device: state.selectedDevice, priority: state.selectedPriority }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail?.detail || 'Failed to add experiment');
      updateLogMessages([...state.logMessages, '✅ Added to queue.', `Experiment: ${result.experiment_uid}`]);
    } catch (error) {
      updateLogMessages([...state.logMessages, `❌ Error: ${error instanceof Error ? error.message : 'Unknown'}`]);
    } finally { setIsStarting(false); }
  };

  const rightRailTabs: RightRailTab[] = [
    {
      value: 'inspector',
      label: 'Inspector',
      icon: Info,
      content: <ExecuteInspector onConfigChange={handleFlowConfigChange} className="h-full border-none" />,
    },
    {
      value: 'yaml',
      label: 'YAML',
      icon: Code2,
      content: (
        <div className="h-full flex flex-col bg-[#09090b]">
          <CodeMirror
            value={state.yamlConfig}
            height="100%"
            extensions={[yaml()]}
            onChange={updateYamlConfig}
            theme={vscodeDark}
            className="flex-1"
          />
        </div>
      ),
    },
  ];

  return (
    <TooltipProvider>
      <div className="h-full flex flex-col bg-background select-none overflow-hidden p-0">
        
        {/* 1. Global Header - Floating Card Style (to match Builder) */}
        <div className="px-2 pt-2 pb-2">
          <header className="h-12 bg-card border border-border shadow-sm rounded-lg flex items-center px-4 justify-between shrink-0 z-20">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center border border-primary/20">
              <Activity className="w-4 h-4 text-primary" />
            </div>
            <div className="flex items-center gap-2 text-xs font-bold">
              <span className="text-muted-foreground">cvlab-kit</span>
              <ChevronRight className="w-3 h-3 text-muted-foreground/30" />
              <span className="text-foreground">{state.selectedConfigFile || "Select Config..."}</span>
              {agentName && (
                <>
                  <ChevronRight className="w-3 h-3 text-muted-foreground/30" />
                  <Badge variant="secondary" className="h-5 px-2 text-[10px] bg-muted/50 border-border/40 font-mono">
                    AGENT: {agentName.toUpperCase()}
                  </Badge>
                </>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest opacity-60">Config:</span>
              <ConfigSelector
                value={state.selectedConfigFile}
                onValueChange={updateSelectedConfigFile}
                configFiles={configFiles}
                isLoading={isLoadingConfigs}
                error={configsError}
              />
            </div>
            <Separator orientation="vertical" className="h-5" />
            <Badge className="h-6 text-[10px] font-mono bg-green-500/10 text-green-600 border-green-500/20 uppercase tracking-tighter">Execution Ready</Badge>
          </div>
          </header>
        </div>

        {/* 2. Main Area - Flat Sectors */}
        <main className="flex-1 min-h-0">
          <ThreePanelLayout
            layoutKey="execute-v2"
            leftPanel={
              <div className="h-full flex flex-col bg-card overflow-hidden">
                <div className="h-11 border-b border-border/40 bg-muted/20 flex items-center px-4 shrink-0">
                  <span className="text-[10px] font-black tracking-widest text-foreground/70 uppercase">Presets</span>
                </div>
                <div className="flex-1 overflow-y-auto">
                  <ExecuteOptionsPanel agentName={agentName} onConfigChange={handleFlowConfigChange} />
                </div>
              </div>
            }
            centerPanel={
              <div className="h-full flex flex-col bg-card overflow-hidden relative">
                <ExecuteFlowPane
                  agentName={agentName}
                  parsedConfig={parsedConfig}
                  configPath={state.selectedConfigFile}
                  onConfigChange={handleFlowConfigChange}
                  className="h-full"
                  variant="embedded"
                />
              </div>
            }
            rightPanel={
              <div className="h-full flex flex-col bg-card overflow-hidden">
                <RightRail tabs={rightRailTabs} defaultTab="yaml" />
              </div>
            }
          />
        </main>

        {/* 3. Global Footer - Execution Controls */}
        <footer className="bg-card border-t border-border/50 shadow-sm overflow-hidden shrink-0">
          <div className="flex items-center justify-between p-2.5 gap-4">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5 text-muted-foreground" />
                <Select value={state.selectedDevice} onValueChange={updateSelectedDevice}>
                  <SelectTrigger className="w-[160px] h-8 text-[11px] font-bold bg-background/50 border-border/40">
                    <SelectValue placeholder="Device" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="any">Any Available</SelectItem>
                    {devices.map(d => <SelectItem key={d.host_id} value={d.host_id}>{d.host_id}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-2">
                <Terminal className="w-3.5 h-3.5 text-muted-foreground" />
                <Select value={state.selectedPriority} onValueChange={updateSelectedPriority}>
                  <SelectTrigger className="w-[100px] h-8 text-[11px] font-bold bg-background/50 border-border/40 uppercase">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="normal">Normal</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {gridSearchCount > 1 && (
                <Badge variant="outline" className="h-7 px-3 bg-amber-500/5 text-amber-600 border-amber-500/20 font-black text-[10px]">
                  {gridSearchCount} EXPERIMENTS DETECTED
                </Badge>
              )}
              <Button
                onClick={handleStartExperiment}
                disabled={isStarting || !state.yamlConfig.trim()}
                className="h-8 gap-2 font-black text-xs px-6 shadow-lg shadow-primary/20"
              >
                {isStarting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ListPlus className="w-3.5 h-3.5" />}
                ADD TO QUEUE
              </Button>
            </div>
          </div>

          {state.logMessages.length > 0 && (
            <div className="px-4 py-2 border-t border-border/20 bg-muted/10 max-h-24 overflow-y-auto">
              {state.logMessages.map((msg, i) => (
                <div key={i} className={cn("text-[10px] font-mono leading-relaxed", msg.includes('✅') ? "text-green-600" : "text-muted-foreground")}>
                  {msg}
                </div>
              ))}
            </div>
          )}
        </footer>
      </div>
    </TooltipProvider>
  );
});

export { ExecuteView };
