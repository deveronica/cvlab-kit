/**
 * NodeContextMenu - Right-click Context Menu for Node Graph
 *
 * Actions:
 * - Drill Down: Enter node's internal structure
 * - View Source: Open source code location
 * - Edit Config: Open config editor panel
 * - Copy ID: Copy node ID to clipboard
 * - Collapse/Expand: Toggle node details
 *
 * PyTorch-specific features:
 * - Train/Eval mode toggle (.train() / .eval())
 * - Device selection (.to('cuda') / .to('cpu'))
 * - Gradient tracking toggle (requires_grad)
 * - Common method calls (.parameters(), .zero_grad(), etc.)
 */

import React, { memo, useCallback } from 'react';
import {
  ChevronRight,
  FileCode,
  Settings2,
  Copy,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Info,
  Play,
  Pause,
  Cpu,
  MonitorDot,
  Zap,
  ZapOff,
  GitBranch,
  Check,
  Trash2,
} from 'lucide-react';
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuShortcut,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger,
  ContextMenuCheckboxItem,
  ContextMenuRadioGroup,
  ContextMenuRadioItem,
  ContextMenuLabel,
} from '@/shared/ui/context-menu';
import { Node, SourceLocation } from '@/shared/model/node-graph';
import { useToast } from '@/shared/model/use-toast';
import { useBuilderStore } from '@/features/builder/model/builderStore';
import { useCategoryMethods } from '@/shared/model/useCategoryMethods';

// Available devices for PyTorch
const PYTORCH_DEVICES = [
  { id: 'cpu', label: 'CPU' },
  { id: 'cuda:0', label: 'CUDA:0' },
  { id: 'cuda:1', label: 'CUDA:1' },
  { id: 'mps', label: 'MPS (Apple)' },
];

interface NodeContextMenuProps {
  children: React.ReactNode;
  node: Node;
  onDrillDown?: (node: Node) => void;
  onViewSource?: (source: SourceLocation) => void;
  onEditConfig?: (node: Node) => void;
  onToggleCollapse?: (node: Node) => void;
  onMethodCall?: (node: Node, methodName: string) => void;
  onDelete?: (node: Node) => void;
  isCollapsed?: boolean;
  isEditingMode?: boolean;
}

export const NodeContextMenu = memo(
  ({
    children,
    node,
    onDrillDown,
    onViewSource,
    onEditConfig,
    onToggleCollapse,
    onMethodCall,
    onDelete,
    isCollapsed,
    isEditingMode = false,
  }: NodeContextMenuProps) => {
    const { toast } = useToast();

    // Get PyTorch state from builder store
    const { nodeStates, setNodeMode, setNodeDevice, toggleNodeGrad } = useBuilderStore();
    const nodeState = nodeStates.get(node.id);
    const currentMode = nodeState?.mode || 'train';
    const currentDevice = nodeState?.device || 'cpu';
    const requiresGrad = nodeState?.requiresGrad ?? true;

    // Fetch available methods for this category
    const { data: categoryMethods = [] } = useCategoryMethods(node.category || null);

    // Determine which PyTorch features to show based on category
    const showModeToggle = node.category === 'model' || node.category === 'loss';
    const showDeviceSelect = node.category === 'model' || node.category === 'optimizer';
    const showGradToggle = node.category === 'model';
    const showMethods = categoryMethods.length > 0;

    const handleCopyId = useCallback(() => {
      navigator.clipboard.writeText(node.id);
      toast({
        title: 'Copied',
        description: `Node ID "${node.id}" copied to clipboard`,
        duration: 2000,
      });
    }, [node.id, toast]);

    const handleCopyConfig = useCallback(() => {
      const configStr = JSON.stringify(node.config || node.config_params, null, 2);
      navigator.clipboard.writeText(configStr);
      toast({
        title: 'Copied',
        description: 'Node config copied to clipboard',
        duration: 2000,
      });
    }, [node, toast]);

    const canDrillDown = node.has_children;
    const hasSource = !!node.source;
    const hasConfig =
      Object.keys(node.config || {}).length > 0 ||
      (node.config_params &&
        (Object.keys(node.config_params.component || {}).length > 0 ||
          Object.keys(node.config_params.global || {}).length > 0));

    return (
      <ContextMenu>
        <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
        <ContextMenuContent className="w-56">
          {/* Node Info Header */}
          <div className="px-2 py-1.5 border-b border-border/50">
            <div className="font-medium text-sm truncate">
              {node.role || node.label}
            </div>
            {node.implementation && (
              <div className="text-xs text-muted-foreground truncate">
                {node.implementation}
              </div>
            )}
          </div>

          {/* Primary Actions */}
          {canDrillDown && (
            <ContextMenuItem
              onClick={() => onDrillDown?.(node)}
              className="gap-2"
            >
              <ChevronRight className="h-4 w-4" />
              <span>Drill Down</span>
              <ContextMenuShortcut>⏎</ContextMenuShortcut>
            </ContextMenuItem>
          )}

          {hasSource && (
            <ContextMenuItem
              onClick={() => onViewSource?.(node.source!)}
              className="gap-2"
            >
              <FileCode className="h-4 w-4" />
              <span>View Source</span>
              <ContextMenuShortcut>⌘S</ContextMenuShortcut>
            </ContextMenuItem>
          )}

          {hasConfig && (
            <ContextMenuItem
              onClick={() => onEditConfig?.(node)}
              className="gap-2"
            >
              <Settings2 className="h-4 w-4" />
              <span>Edit Config</span>
              <ContextMenuShortcut>⌘E</ContextMenuShortcut>
            </ContextMenuItem>
          )}

          <ContextMenuSeparator />

          {/* PyTorch Mode Toggle (model/loss only) */}
          {showModeToggle && isEditingMode && (
            <>
              <ContextMenuSub>
                <ContextMenuSubTrigger className="gap-2">
                  {currentMode === 'train' ? (
                    <Play className="h-4 w-4 text-green-500" />
                  ) : (
                    <Pause className="h-4 w-4 text-blue-500" />
                  )}
                  <span>Mode: {currentMode === 'train' ? 'Train' : 'Eval'}</span>
                </ContextMenuSubTrigger>
                <ContextMenuSubContent>
                  <ContextMenuRadioGroup value={currentMode}>
                    <ContextMenuRadioItem
                      value="train"
                      onClick={() => setNodeMode(node.id, 'train')}
                    >
                      <Play className="h-3.5 w-3.5 mr-2 text-green-500" />
                      Train Mode
                      <ContextMenuShortcut>.train()</ContextMenuShortcut>
                    </ContextMenuRadioItem>
                    <ContextMenuRadioItem
                      value="eval"
                      onClick={() => setNodeMode(node.id, 'eval')}
                    >
                      <Pause className="h-3.5 w-3.5 mr-2 text-blue-500" />
                      Eval Mode
                      <ContextMenuShortcut>.eval()</ContextMenuShortcut>
                    </ContextMenuRadioItem>
                  </ContextMenuRadioGroup>
                </ContextMenuSubContent>
              </ContextMenuSub>
              <ContextMenuSeparator />
            </>
          )}

          {/* PyTorch Device Selection (model/optimizer only) */}
          {showDeviceSelect && isEditingMode && (
            <>
              <ContextMenuSub>
                <ContextMenuSubTrigger className="gap-2">
                  {currentDevice.startsWith('cuda') ? (
                    <MonitorDot className="h-4 w-4 text-purple-500" />
                  ) : (
                    <Cpu className="h-4 w-4 text-gray-500" />
                  )}
                  <span>Device: {currentDevice}</span>
                </ContextMenuSubTrigger>
                <ContextMenuSubContent>
                  <ContextMenuRadioGroup value={currentDevice}>
                    {PYTORCH_DEVICES.map((device) => (
                      <ContextMenuRadioItem
                        key={device.id}
                        value={device.id}
                        onClick={() => setNodeDevice(node.id, device.id)}
                      >
                        {device.id.startsWith('cuda') || device.id === 'mps' ? (
                          <MonitorDot className="h-3.5 w-3.5 mr-2" />
                        ) : (
                          <Cpu className="h-3.5 w-3.5 mr-2" />
                        )}
                        {device.label}
                        <ContextMenuShortcut>.to('{device.id}')</ContextMenuShortcut>
                      </ContextMenuRadioItem>
                    ))}
                  </ContextMenuRadioGroup>
                </ContextMenuSubContent>
              </ContextMenuSub>
              <ContextMenuSeparator />
            </>
          )}

          {/* PyTorch Gradient Toggle (model only) */}
          {showGradToggle && isEditingMode && (
            <>
              <ContextMenuCheckboxItem
                checked={requiresGrad}
                onCheckedChange={() => toggleNodeGrad(node.id)}
                className="gap-2"
              >
                {requiresGrad ? (
                  <Zap className="h-4 w-4 text-yellow-500" />
                ) : (
                  <ZapOff className="h-4 w-4 text-gray-400" />
                )}
                <span>Track Gradients</span>
                <ContextMenuShortcut>requires_grad</ContextMenuShortcut>
              </ContextMenuCheckboxItem>
              <ContextMenuSeparator />
            </>
          )}

          {/* Available Methods */}
          {showMethods && (
            <>
              <ContextMenuSub>
                <ContextMenuSubTrigger className="gap-2">
                  <GitBranch className="h-4 w-4" />
                  <span>Methods</span>
                </ContextMenuSubTrigger>
                <ContextMenuSubContent className="max-h-[300px] overflow-y-auto">
                  <ContextMenuLabel>Available Methods</ContextMenuLabel>
                  {categoryMethods.slice(0, 15).map((method) => (
                    <ContextMenuItem
                      key={method.name}
                      onClick={() => onMethodCall?.(node, method.name)}
                      className="gap-2"
                    >
                      <span className="font-mono text-xs">.{method.name}()</span>
                      {method.returns && (
                        <ContextMenuShortcut className="text-[10px]">
                          → {method.returns}
                        </ContextMenuShortcut>
                      )}
                    </ContextMenuItem>
                  ))}
                </ContextMenuSubContent>
              </ContextMenuSub>
              <ContextMenuSeparator />
            </>
          )}

          {/* Toggle Collapse */}
          {onToggleCollapse && (
            <ContextMenuItem
              onClick={() => onToggleCollapse(node)}
              className="gap-2"
            >
              {isCollapsed ? (
                <>
                  <ChevronDown className="h-4 w-4" />
                  <span>Expand Details</span>
                </>
              ) : (
                <>
                  <ChevronUp className="h-4 w-4" />
                  <span>Collapse Details</span>
                </>
              )}
            </ContextMenuItem>
          )}

          {/* Copy Submenu */}
          <ContextMenuSub>
            <ContextMenuSubTrigger className="gap-2">
              <Copy className="h-4 w-4" />
              <span>Copy</span>
            </ContextMenuSubTrigger>
            <ContextMenuSubContent className="w-48">
              <ContextMenuItem onClick={handleCopyId}>
                Node ID
              </ContextMenuItem>
              {hasConfig && (
                <ContextMenuItem onClick={handleCopyConfig}>
                  Config JSON
                </ContextMenuItem>
              )}
              {hasSource && (
                <ContextMenuItem
                  onClick={() => {
                    const loc = `${node.source!.file}:${node.source!.line_start}`;
                    navigator.clipboard.writeText(loc);
                    toast({
                      title: 'Copied',
                      description: 'Source location copied',
                      duration: 2000,
                    });
                  }}
                >
                  Source Location
                </ContextMenuItem>
              )}
            </ContextMenuSubContent>
          </ContextMenuSub>

          <ContextMenuSeparator />

          {/* Node Details */}
          <ContextMenuSub>
            <ContextMenuSubTrigger className="gap-2">
              <Info className="h-4 w-4" />
              <span>Details</span>
            </ContextMenuSubTrigger>
            <ContextMenuSubContent className="w-64">
              <div className="px-2 py-1.5 text-xs space-y-1">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Type:</span>
                  <span className="font-mono">{node.type}</span>
                </div>
                {node.category && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Category:</span>
                    <span className="font-mono">{node.category}</span>
                  </div>
                )}
                {node.hierarchy && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Depth:</span>
                    <span className="font-mono">{node.hierarchy.depth}</span>
                  </div>
                )}
                {hasSource && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">File:</span>
                    <span className="font-mono truncate max-w-[150px]">
                      {node.source!.file}
                    </span>
                  </div>
                )}
              </div>
            </ContextMenuSubContent>
          </ContextMenuSub>

          {/* External Link (if metadata contains URL) */}
          {node.metadata?.doc_url && (
            <>
              <ContextMenuSeparator />
              <ContextMenuItem
                onClick={() => {
                  const url = node.metadata?.doc_url;
                  if (typeof url === 'string' && url.length > 0) {
                    window.open(url, '_blank');
                  }
                }}
                className="gap-2"
              >
                <ExternalLink className="h-4 w-4" />
                <span>Documentation</span>
              </ContextMenuItem>
            </>
          )}

          {/* Delete (only in editing mode) */}
          {isEditingMode && onDelete && (
            <>
              <ContextMenuSeparator />
              <ContextMenuItem
                onClick={() => onDelete(node)}
                className="gap-2 text-destructive focus:text-destructive"
              >
                <Trash2 className="h-4 w-4" />
                <span>Delete Node</span>
                <ContextMenuShortcut>⌫</ContextMenuShortcut>
              </ContextMenuItem>
            </>
          )}
        </ContextMenuContent>
      </ContextMenu>
    );
  }
);

NodeContextMenu.displayName = 'NodeContextMenu';

export default NodeContextMenu;
