/**
 * CodePane - Next-Gen Professional IDE Editor
 * 항목 11: 초고가독성 및 하이엔드 미학적 리팩토링 (Zed/Cursor Style)
 */
import React, { useRef, useMemo, useCallback } from 'react';
import CodeMirror, { ReactCodeMirrorRef } from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { yaml } from '@codemirror/lang-yaml';
import { EditorView, Decoration, DecorationSet } from '@codemirror/view';
import { StateField, StateEffect } from '@codemirror/state';
import { tags as t } from '@lezer/highlight';
import { createTheme } from '@uiw/codemirror-themes';
import { foldGutter } from '@codemirror/language';
import { FileCode2, Terminal, ShieldCheck, ChevronRight, Hash } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { Badge } from '@/shared/ui/badge';
import { useAgentBuilder } from '@/entities/node-system/model/AgentBuilderContext';
import { useCodeSync } from '@/features/builder/model/useCodeSync';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';

// =============================================================================
// 1. ZED-INSPIRED HYPER-CONTRAST THEME
// =============================================================================
const zedTheme = createTheme({
  theme: 'dark',
  settings: {
    background: '#0d1117',      // Deep Deep Black
    foreground: '#e6edf3',      // 항목 2: 밝고 뚜렷한 흰색으로 변경
    caret: '#58a6ff',           // GitHub Blue Caret
    selection: '#1f6feb44',     // Transparent Blue
    selectionMatch: '#1f6feb66',
    lineHighlight: '#161b22',   // Very subtle line highlight
    gutterBackground: '#0d1117',
    gutterForeground: '#6e7681', // 항목 2: 줄번호 색상 개선
    gutterBorder: 'transparent',
  },
  styles: [
    { tag: t.comment, color: '#8b949e', fontStyle: 'italic' },
    { tag: t.variableName, color: '#ffa657' },      // Orange
    { tag: t.propertyName, color: '#79c0ff' },      // Light Blue
    { tag: t.string, color: '#a5d6ff' },            // Very Light Blue
    { tag: t.number, color: '#79c0ff' },
    { tag: t.bool, color: '#79c0ff' },
    { tag: t.keyword, color: '#ff7b72', fontWeight: 'bold' }, // Red-Pink
    { tag: t.className, color: '#ffa657', fontWeight: 'bold' },
    { tag: t.function(t.variableName), color: '#d2a8ff' }, // Purple
    { tag: t.operator, color: '#ff7b72' },
    { tag: t.punctuation, color: '#8b949e' },
    { tag: t.attributeName, color: '#79c0ff' },
  ],
});

// =============================================================================
// 2. LINE HIGHLIGHT DECORATIONS (Sync with Node)
// =============================================================================
const highlightLineEffect = StateEffect.define<number[]>();
const highlightedLinesField = StateField.define<DecorationSet>({
  create() { return Decoration.none; },
  update(underlines, tr) {
    underlines = underlines.map(tr.changes);
    for (const e of tr.effects) {
      if (e.is(highlightLineEffect)) {
        // Note: Simple line class application via CSS is more efficient for large blocks
        // Line highlighting is handled through editor theme and CSS classes
      }
    }
    return underlines;
  },
  provide: f => EditorView.decorations.from(f)
});

const editorBaseStyles = EditorView.baseTheme({
  '&': { height: '100%' },
  '.cm-scroller': { 
    fontFamily: '"JetBrains Mono", "Fira Code", monospace',
    fontSize: '13px',
    lineHeight: '1.8',
  },
  '.cm-content': { padding: '20px 0' },
  '.cm-gutters': { 
    border: 'none !important', 
    backgroundColor: '#0d1117 !important',
    color: '#6e7681', // 항목 3: 줄번호 가독성을 위해 더 밝은 색상으로 변경
    minWidth: '40px',
    display: 'flex',
    justifyContent: 'flex-end',
  },
  '.cm-lineNumbers .cm-gutterElement': { 
    padding: '0 8px 0 8px',
    minWidth: '30px',
    textAlign: 'right',
  },
  '.cm-activeLine': { backgroundColor: 'rgba(255,255,255,0.03) !important' },
  '.cm-activeLineGutter': { 
    backgroundColor: 'transparent !important', 
    color: '#e6edf3 !important',
    fontWeight: 'bold',
  },
  // 노드와 동기화된 라인 스타일
  '.cm-node-sync': {
    backgroundColor: 'rgba(31, 111, 235, 0.1) !important',
    borderLeft: '3px solid #1f6feb !important',
    transition: 'all 0.3s ease',
  },
  // Custom Scrollbar
  '& ::-webkit-scrollbar': { width: '12px', height: '12px' },
  '& ::-webkit-scrollbar-track': { background: '#0d1117' },
  '& ::-webkit-scrollbar-thumb': { 
    background: '#30363d', 
    borderRadius: '10px',
    border: '3px solid #0d1117',
  },
  '& ::-webkit-scrollbar-thumb:hover': { background: '#484f58' },
});

export function CodePane({ className, readOnly = true }: { className?: string; readOnly?: boolean }) {
  const { selectedFile, selectedAgent, isEditingMode, code, setCode, syncCodeToNode, selectedNodeId, highlightedLines } = useAgentBuilder();
  const draftState = useNodeStore((state) => state.draftState);
  const editorRef = useRef<ReactCodeMirrorRef>(null);
  const agentName = selectedAgent?.name?.replace('.py', '') || '';

  const { debouncedSync, selectNodeAtCursor } = useCodeSync({
    agentName,
    enabled: !readOnly && !!agentName,
    draftId: isEditingMode ? draftState.draftId : null,
  });

  const displayFile = selectedAgent || selectedFile;

  // 라인 동기화 로직
  const handleUpdate = useCallback((update: any) => {
    if (update.selectionSet) {
      const line = update.state.doc.lineAt(update.state.selection.main.head).number;
      syncCodeToNode(line);
      if (agentName && !readOnly) selectNodeAtCursor(line);
    }
  }, [syncCodeToNode, selectNodeAtCursor, agentName, readOnly]);

  const extensions = useMemo(() => [
    displayFile?.type === 'config' ? yaml() : python(),
    zedTheme,
    editorBaseStyles,
    foldGutter(),
    EditorView.updateListener.of(handleUpdate),
  ], [displayFile, handleUpdate]);

  return (
    <div className={cn('h-full flex flex-col bg-[#0d1117] overflow-hidden', className)}>
      {/* 1. Breadcrumb Header */}
      <header className="h-10 px-4 bg-[#161b22] border-b border-[#30363d] flex items-center justify-between shrink-0 shadow-sm z-10">
        <div className="flex items-center gap-2 text-[11px] font-medium text-[#8b949e]">
          <div className="flex items-center gap-1.5 hover:text-[#c9d1d9] transition-colors cursor-default">
            <ShieldCheck className="w-3.5 h-3.5 text-green-500/70" />
            <span>cvlab-kit</span>
          </div>
          <ChevronRight className="w-3 h-3 opacity-20" />
          <div className="flex items-center gap-1.5 hover:text-[#c9d1d9] transition-colors cursor-default">
            <span className="opacity-60">agents</span>
          </div>
          <ChevronRight className="w-3 h-3 opacity-20" />
          <div className="flex items-center gap-2 bg-[#0d1117] px-2 py-1 rounded border border-[#30363d] text-[#e6edf3]">
            <FileCode2 className="w-3.5 h-3.5 text-blue-400" />
            <span className="font-bold tracking-tight">{displayFile?.name || 'source.py'}</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {selectedNodeId && (
            <div className="flex items-center gap-1.5 px-2 py-0.5 bg-blue-500/10 border border-blue-500/20 rounded text-[10px] font-mono text-blue-400">
              <Hash className="w-3 h-3" />
              <span>{selectedNodeId}</span>
            </div>
          )}
          <Badge variant="outline" className="h-5 text-[9px] font-mono border-[#30363d] text-[#8b949e] uppercase tracking-tighter">
            {isEditingMode ? 'Read-Write' : 'Reference Only'}
          </Badge>
        </div>
      </header>

      {/* 2. Editor Body */}
      <div className="flex-1 relative group">
        {displayFile ? (
          <div className="absolute inset-0">
            <CodeMirror
              ref={editorRef}
              value={code}
              onChange={(val) => { setCode(val); if (!readOnly) debouncedSync(val); }}
              height="100%"
              extensions={extensions}
              readOnly={readOnly}
              theme="none" // we use our custom theme
              basicSetup={{ 
                lineNumbers: true, 
                foldGutter: true, 
                highlightActiveLine: true, 
                syntaxHighlighting: true,
                bracketMatching: true,
                closeBrackets: true,
                autocompletion: true,
                crosshairCursor: false,
              }}
              className="h-full"
            />
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center space-y-4 opacity-20 select-none">
            <div className="relative">
              <Terminal className="w-20 h-20 text-[#8b949e]" />
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-primary rounded-full animate-pulse" />
            </div>
            <div className="text-center space-y-1">
              <h3 className="text-sm font-black text-[#e6edf3] uppercase tracking-[0.2em]">Ready to Inspect</h3>
              <p className="text-[10px] text-[#8b949e]">Select a node or file to begin code analysis</p>
            </div>
          </div>
        )}
      </div>

      {/* 3. Editor Footer (Status Bar) */}
      <footer className="h-6 px-4 bg-[#0d1117] border-t border-[#30363d] flex items-center justify-between shrink-0 text-[10px] font-mono text-[#484f58]">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-green-500/50" />
            <span>Python 3.10+</span>
          </div>
          <span>UTF-8</span>
        </div>
        <div className="flex items-center gap-4">
          <span>Ln {highlightedLines[0] || 1}, Col 1</span>
          <span className="text-blue-500/80 font-bold">Spaces: 4</span>
        </div>
      </footer>
    </div>
  );
}
