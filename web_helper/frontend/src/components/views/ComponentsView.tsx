import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { ScrollArea } from '../ui/scroll-area';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';
import CodeMirror, { ReactCodeMirrorRef } from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import {
  Folder,
  FolderOpen,
  FileCode2,
  Check,
  RefreshCw,
  Code2,
  ChevronRight,
  Clock,
  Hash,
  Box,
  Braces,
  RotateCcw,
  ListTree,
  PanelLeftClose,
  PanelRightClose,
} from 'lucide-react';
import { cn } from '../../lib/utils';

// Types
interface ComponentVersion {
  hash: string;
  path: string;
  category: string;
  name: string;
  is_active: boolean;
  created_at: string;
  content?: string;
}

interface ComponentItem {
  path: string;
  category: string;
  name: string;
  active_hash: string | null;
  version_count: number;
  updated_at: string | null;
}

interface CodeSymbol {
  name: string;
  type: 'class' | 'function' | 'method';
  line: number;
  indent: number;
  parent?: string;
}

const CATEGORIES = [
  'agent', 'model', 'dataset', 'dataloader', 'transform',
  'optimizer', 'loss', 'metric', 'scheduler',
];

const STORAGE_KEY = 'components-view-panels';

function parseCodeSymbols(code: string): CodeSymbol[] {
  const symbols: CodeSymbol[] = [];
  const lines = code.split('\n');
  let currentClass: string | null = null;

  lines.forEach((line, index) => {
    const classMatch = line.match(/^(\s*)class\s+(\w+)/);
    const funcMatch = line.match(/^(\s*)def\s+(\w+)/);
    const asyncFuncMatch = line.match(/^(\s*)async\s+def\s+(\w+)/);

    if (classMatch) {
      const [, indent, name] = classMatch;
      currentClass = name;
      symbols.push({ name, type: 'class', line: index + 1, indent: indent.length });
    } else if (funcMatch || asyncFuncMatch) {
      const match = funcMatch || asyncFuncMatch;
      if (match) {
        const [, indent, name] = match;
        const indentLevel = indent.length;
        if (name.startsWith('__') && name !== '__init__') return;
        symbols.push({
          name,
          type: indentLevel > 0 && currentClass ? 'method' : 'function',
          line: index + 1,
          indent: indentLevel,
          parent: indentLevel > 0 ? currentClass || undefined : undefined,
        });
      }
    }
    if (line.trim() && !line.startsWith(' ') && !line.startsWith('\t') && !classMatch) {
      currentClass = null;
    }
  });
  return symbols;
}

function loadPanelState(): { left: boolean; right: boolean } {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return JSON.parse(saved);
  } catch {}
  return { left: true, right: true };
}

function savePanelState(left: boolean, right: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ left, right }));
  } catch {}
}

export function ComponentsView() {
  const [components, setComponents] = useState<ComponentItem[]>([]);
  const [selectedComponent, setSelectedComponent] = useState<ComponentItem | null>(null);
  const [versions, setVersions] = useState<ComponentVersion[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(CATEGORIES));
  const [leftPanelOpen, setLeftPanelOpen] = useState(() => loadPanelState().left);
  const [rightPanelOpen, setRightPanelOpen] = useState(() => loadPanelState().right);
  const editorRef = useRef<ReactCodeMirrorRef>(null);

  useEffect(() => {
    savePanelState(leftPanelOpen, rightPanelOpen);
  }, [leftPanelOpen, rightPanelOpen]);

  useEffect(() => {
    fetchComponents();
  }, []);

  const fetchComponents = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/components/versions');
      const result = await response.json();
      if (result.data) setComponents(result.data);
    } catch (error) {
      console.error('Failed to fetch components:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    if (selectedComponent) {
      fetchVersions(selectedComponent.category, selectedComponent.name);
    }
  }, [selectedComponent]);

  const fetchVersions = async (category: string, name: string) => {
    try {
      const response = await fetch(`/api/components/versions/${category}/${name}?include_content=true`);
      const result = await response.json();
      if (result.data) setVersions(result.data);
    } catch (error) {
      console.error('Failed to fetch versions:', error);
      setVersions([]);
    }
  };

  const handleScanComponents = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/components/versions/scan', { method: 'POST' });
      const result = await response.json();
      if (result.data) fetchComponents();
    } catch (error) {
      console.error('Failed to scan components:', error);
    }
    setLoading(false);
  };

  const handleActivateVersion = async (hash: string) => {
    try {
      const response = await fetch('/api/components/versions/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hash }),
      });
      if (response.ok && selectedComponent) {
        fetchVersions(selectedComponent.category, selectedComponent.name);
        fetchComponents();
      }
    } catch (error) {
      console.error('Failed to activate version:', error);
    }
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) next.delete(category);
      else next.add(category);
      return next;
    });
  };

  const scrollToLine = useCallback((line: number) => {
    const view = editorRef.current?.view;
    if (!view) return;

    try {
      const lineInfo = view.state.doc.line(line);
      view.dispatch({
        selection: { anchor: lineInfo.from, head: lineInfo.from },
      });

      const scrollDOM = view.scrollDOM;
      // Force overflow to enable scrolling (CodeMirror uses overflow:visible by default)
      scrollDOM.setAttribute('style', scrollDOM.getAttribute('style') + '; overflow: auto !important;');

      requestAnimationFrame(() => {
        const coords = view.coordsAtPos(lineInfo.from);
        if (coords) {
          const targetTop = coords.top - scrollDOM.getBoundingClientRect().top + scrollDOM.scrollTop;
          scrollDOM.scrollTop = targetTop - 100;
        }
        view.focus();
      });
    } catch (e) {
      console.error('Failed to scroll to line:', e);
    }
  }, []);

  const groupedComponents = CATEGORIES.reduce((acc, category) => {
    acc[category] = components.filter((c) => c.category === category);
    return acc;
  }, {} as Record<string, ComponentItem[]>);

  const activeVersion = versions.find((v) => v.is_active);

  const codeSymbols = useMemo(() => {
    if (!activeVersion?.content) return [];
    return parseCodeSymbols(activeVersion.content);
  }, [activeVersion?.content]);

  const groupedSymbols = useMemo(() => {
    const classes: { name: string; line: number; methods: CodeSymbol[] }[] = [];
    const functions: CodeSymbol[] = [];
    codeSymbols.forEach((symbol) => {
      if (symbol.type === 'class') {
        classes.push({ name: symbol.name, line: symbol.line, methods: [] });
      } else if (symbol.type === 'method' && symbol.parent) {
        const parentClass = classes.find((c) => c.name === symbol.parent);
        if (parentClass) parentClass.methods.push(symbol);
      } else if (symbol.type === 'function') {
        functions.push(symbol);
      }
    });
    return { classes, functions };
  }, [codeSymbols]);

  const formatTimeAgo = (dateStr: string) => {
    const date = new Date(dateStr);
    const diff = Date.now() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    if (days > 0) return `${days}d ago`;
    const hours = Math.floor(diff / (1000 * 60 * 60));
    if (hours > 0) return `${hours}h ago`;
    return `${Math.floor(diff / (1000 * 60))}m ago`;
  };

  return (
    <TooltipProvider>
      <div className="h-full flex gap-3 p-4">
        {/* Left Panel - Explorer */}
        <div className={cn(
          "flex-shrink-0 transition-all duration-300 ease-in-out overflow-hidden",
          leftPanelOpen ? "w-56 opacity-100" : "w-0 opacity-0"
        )}>
          <Card className="h-full flex flex-col w-56">
            <div className="px-3 py-2 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                <ListTree className="h-4 w-4" />
                <span>Explorer</span>
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleScanComponents} disabled={loading}>
                    <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="right">Scan local</TooltipContent>
              </Tooltip>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-1.5">
                {CATEGORIES.map((category) => {
                  const items = groupedComponents[category] || [];
                  const isExpanded = expandedCategories.has(category);
                  if (items.length === 0) return null;
                  return (
                    <div key={category} className="mb-0.5">
                      <button
                        onClick={() => toggleCategory(category)}
                        className="w-full flex items-center gap-1.5 px-2 py-1 rounded hover:bg-accent/50 text-sm transition-colors"
                      >
                        <ChevronRight className={cn(
                          "h-3.5 w-3.5 text-muted-foreground transition-transform duration-200",
                          isExpanded && "rotate-90"
                        )} />
                        {isExpanded ? <FolderOpen className="h-4 w-4 text-amber-500" /> : <Folder className="h-4 w-4 text-amber-500" />}
                        <span className="font-medium">{category}</span>
                        <span className="ml-auto text-xs text-muted-foreground">{items.length}</span>
                      </button>
                      <div className={cn(
                        "ml-3 pl-2 border-l border-border/50 overflow-hidden transition-all duration-200",
                        isExpanded ? "max-h-[1000px] opacity-100" : "max-h-0 opacity-0"
                      )}>
                        {items.map((item) => (
                          <button
                            key={item.path}
                            onClick={() => setSelectedComponent(item)}
                            className={cn(
                              "w-full flex items-center gap-1.5 px-2 py-1 rounded text-sm transition-colors",
                              selectedComponent?.path === item.path ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
                            )}
                          >
                            <FileCode2 className="h-3.5 w-3.5 text-blue-500 flex-shrink-0" />
                            <span className="truncate">{item.name}.py</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </ScrollArea>
          </Card>
        </div>

        {/* Center Panel - Code Viewer */}
        <Card className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* Header with toggle buttons on corners */}
          <div className="px-2 py-1.5 border-b border-border flex items-center gap-2">
            {/* Left toggle */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={leftPanelOpen ? "secondary" : "ghost"}
                  size="icon"
                  className="h-7 w-7 flex-shrink-0"
                  onClick={() => setLeftPanelOpen(!leftPanelOpen)}
                >
                  <PanelLeftClose className={cn(
                    "h-4 w-4 transition-transform duration-200",
                    !leftPanelOpen && "rotate-180"
                  )} />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">Explorer</TooltipContent>
            </Tooltip>

            {/* File info - center */}
            <div className="flex-1 flex items-center justify-center gap-2 min-w-0">
              {selectedComponent ? (
                <>
                  <FileCode2 className="h-4 w-4 text-blue-500 flex-shrink-0" />
                  <span className="text-sm font-medium truncate">{selectedComponent.name}.py</span>
                  <Badge variant="secondary" className="text-xs">{selectedComponent.category}</Badge>
                  {activeVersion && (
                    <>
                      <span className="text-muted-foreground">·</span>
                      <code className="text-xs text-muted-foreground">{activeVersion.hash.slice(0, 8)}</code>
                    </>
                  )}
                </>
              ) : (
                <span className="text-sm text-muted-foreground">Select a component</span>
              )}
            </div>

            {/* Right toggle */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={rightPanelOpen ? "secondary" : "ghost"}
                  size="icon"
                  className="h-7 w-7 flex-shrink-0"
                  onClick={() => setRightPanelOpen(!rightPanelOpen)}
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

          {/* Code editor */}
          <div className="flex-1 overflow-hidden relative">
            {activeVersion?.content ? (
              <div className="absolute inset-0">
                <CodeMirror
                  ref={editorRef}
                  value={activeVersion.content}
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

        {/* Right Panel - Outline & History */}
        <div className={cn(
          "flex-shrink-0 flex flex-col gap-3 transition-all duration-300 ease-in-out overflow-hidden",
          rightPanelOpen ? "w-56 opacity-100" : "w-0 opacity-0"
        )}>
          {/* Outline */}
          <Card className="flex-1 flex flex-col overflow-hidden min-h-0 w-56">
            <div className="px-3 py-2 border-b border-border">
              <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                <Braces className="h-4 w-4" />
                <span>Outline</span>
                {codeSymbols.length > 0 && (
                  <Badge variant="outline" className="ml-auto text-xs">{codeSymbols.length}</Badge>
                )}
              </div>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-1.5">
                {groupedSymbols.classes.length === 0 && groupedSymbols.functions.length === 0 ? (
                  <p className="text-xs text-muted-foreground text-center py-4">
                    {selectedComponent ? 'No symbols' : 'Select a file'}
                  </p>
                ) : (
                  <>
                    {groupedSymbols.classes.map((cls) => (
                      <div key={cls.name} className="mb-1">
                        <button
                          onClick={() => scrollToLine(cls.line)}
                          className="w-full flex items-center gap-1.5 px-2 py-1 rounded hover:bg-accent/50 text-sm transition-colors group"
                        >
                          <Box className="h-3.5 w-3.5 text-amber-500" />
                          <span className="font-medium truncate group-hover:text-primary">{cls.name}</span>
                          <span className="ml-auto text-xs text-muted-foreground opacity-0 group-hover:opacity-100">L{cls.line}</span>
                        </button>
                        {cls.methods.length > 0 && (
                          <div className="ml-3 pl-2 border-l border-border/50">
                            {cls.methods.map((method) => (
                              <button
                                key={`${cls.name}-${method.name}-${method.line}`}
                                onClick={() => scrollToLine(method.line)}
                                className="w-full flex items-center gap-1.5 px-2 py-0.5 rounded hover:bg-accent/50 text-sm transition-colors group"
                              >
                                <Braces className="h-3 w-3 text-purple-500" />
                                <span className={cn("truncate group-hover:text-primary", method.name === '__init__' && "text-muted-foreground")}>
                                  {method.name}
                                </span>
                                <span className="ml-auto text-xs text-muted-foreground opacity-0 group-hover:opacity-100">L{method.line}</span>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                    {groupedSymbols.functions.length > 0 && (
                      <div className={cn(groupedSymbols.classes.length > 0 && "mt-2 pt-2 border-t border-border/50")}>
                        {groupedSymbols.functions.map((func) => (
                          <button
                            key={`${func.name}-${func.line}`}
                            onClick={() => scrollToLine(func.line)}
                            className="w-full flex items-center gap-1.5 px-2 py-1 rounded hover:bg-accent/50 text-sm transition-colors group"
                          >
                            <Braces className="h-3.5 w-3.5 text-blue-500" />
                            <span className="truncate group-hover:text-primary">{func.name}</span>
                            <span className="ml-auto text-xs text-muted-foreground opacity-0 group-hover:opacity-100">L{func.line}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            </ScrollArea>
          </Card>

          {/* History */}
          <Card className="h-48 flex-shrink-0 flex flex-col overflow-hidden w-56">
            <div className="px-3 py-2 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                <Clock className="h-4 w-4" />
                <span>History</span>
              </div>
              {versions.length > 0 && <Badge variant="outline" className="text-xs">{versions.length}</Badge>}
            </div>
            <ScrollArea className="flex-1">
              <div className="p-1.5 space-y-1">
                {versions.map((version, index) => (
                  <div
                    key={version.hash}
                    className={cn(
                      "px-2 py-1.5 rounded transition-all group",
                      version.is_active ? "bg-primary/10 border border-primary/20" : "hover:bg-accent/50"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        {version.is_active ? <Check className="h-3 w-3 text-primary" /> : <Hash className="h-3 w-3 text-muted-foreground" />}
                        <code className="text-xs">{version.hash.slice(0, 8)}</code>
                      </div>
                      {!version.is_active && (
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-5 w-5 opacity-0 group-hover:opacity-100"
                              onClick={() => handleActivateVersion(version.hash)}
                            >
                              <RotateCcw className="h-3 w-3" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent side="left">Rollback</TooltipContent>
                        </Tooltip>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground mt-0.5 pl-4">
                      {formatTimeAgo(version.created_at)}
                      {index === 0 && !version.is_active && <span className="ml-1 text-amber-500">latest</span>}
                    </div>
                  </div>
                ))}
                {versions.length === 0 && (
                  <p className="text-xs text-muted-foreground text-center py-4">
                    {selectedComponent ? 'No versions' : 'Select a file'}
                  </p>
                )}
              </div>
            </ScrollArea>
          </Card>
        </div>
      </div>
    </TooltipProvider>
  );
}
