/**
 * ComponentDetailPanel - Right-side panel for component details and options
 *
 * Shows when a node is selected in the Builder node canvas:
 * - Component category and type
 * - Source location (file:line)
 * - Input/Output ports
 * - Configuration options (editable)
 * - Implementation selector
 */

import { memo, useState, useCallback, useEffect } from 'react';
import { X, Code2, FileCode, Box, ArrowRight, ArrowLeft, Settings2, Edit3, Check, ChevronDown, Plus, Trash2 } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { Button } from '@/shared/ui/button';
import { Badge } from '@/shared/ui/badge';
import { ScrollArea } from '@/shared/ui/scroll-area';
import { Separator } from '@/shared/ui/separator';
import { Input } from '@/shared/ui/input';
import { Label } from '@/shared/ui/label';
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
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/shared/ui/collapsible';
import type { HierarchyNode, Port, ComponentCategory, PropertyInfo, PropertySummary } from '@/shared/model/hierarchy';
import { CATEGORY_THEMES } from '@/shared/model/hierarchy';
import { AlertCircle, FileText, Hash, Link2, Zap } from 'lucide-react';

// ValueSource theme mapping
const VALUE_SOURCE_THEMES: Record<string, { bg: string; text: string; icon: typeof AlertCircle; label: string }> = {
  required: { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-400', icon: AlertCircle, label: 'Required' },
  config: { bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-400', icon: FileText, label: 'Config' },
  default: { bg: 'bg-gray-100 dark:bg-gray-800/30', text: 'text-gray-600 dark:text-gray-400', icon: Hash, label: 'Default' },
  hardcode: { bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-400', icon: Zap, label: 'Hardcode' },
  connected: { bg: 'bg-green-100 dark:bg-green-900/30', text: 'text-green-700 dark:text-green-400', icon: Link2, label: 'Connected' },
};
import { getPortTheme } from '@/shared/config/port-themes';
import { PortType } from '@/shared/model/node-graph';

interface ComponentDetailPanelProps {
  node: HierarchyNode | null;
  onClose: () => void;
  className?: string;
  isEditingMode?: boolean;
  onNodeUpdate?: (nodeId: string, updates: Partial<HierarchyNode>) => void;
  onNodeDelete?: (nodeId: string) => void;
  availableImpls?: string[]; // Available implementations for this category
}

/**
 * Editable parameter row
 */
const EditableParam = memo(({
  name,
  value,
  onChange,
  onDelete,
  isEditing
}: {
  name: string;
  value: string;
  onChange: (name: string, value: string) => void;
  onDelete: (name: string) => void;
  isEditing: boolean;
}) => {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleBlur = useCallback(() => {
    if (localValue !== value) {
      onChange(name, localValue);
    }
  }, [name, localValue, value, onChange]);

  if (!isEditing) {
    return (
      <div className="flex items-center justify-between py-1 text-xs">
        <span className="text-muted-foreground">{name}:</span>
        <span className="font-mono">{value}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 py-1">
      <span className="text-xs text-muted-foreground min-w-[60px]">{name}:</span>
      <Input
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
        onBlur={handleBlur}
        onKeyDown={(e) => e.key === 'Enter' && handleBlur()}
        className="h-6 text-xs font-mono flex-1"
      />
      <Button
        variant="ghost"
        size="sm"
        className="h-6 w-6 p-0 text-destructive hover:text-destructive"
        onClick={() => onDelete(name)}
      >
        <Trash2 className="h-3 w-3" />
      </Button>
    </div>
  );
});
EditableParam.displayName = 'EditableParam';

/**
 * Port list display
 */
const PortList = memo(({ ports, direction }: { ports: Port[]; direction: 'input' | 'output' }) => {
  if (ports.length === 0) {
    return (
      <div className="text-xs text-muted-foreground italic">
        No {direction} ports
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {ports.map((port) => {
        const theme = getPortTheme(port.type as PortType);
        return (
          <div
            key={port.name}
            className="flex items-center gap-2 px-2 py-1.5 rounded bg-muted/30 hover:bg-muted/50 transition-colors"
          >
            <div
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: theme.color }}
            />
            <div className="flex-1 min-w-0">
              <div className="text-xs font-medium truncate">{port.name}</div>
              <div className="text-[10px] text-muted-foreground">{theme.label}</div>
            </div>
            {direction === 'input' && (
              <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
            )}
            {direction === 'output' && (
              <ArrowLeft className="h-3 w-3 text-muted-foreground flex-shrink-0" />
            )}
          </div>
        );
      })}
    </div>
  );
});
PortList.displayName = 'PortList';

/**
 * Section component for consistent styling
 */
const Section = memo(({ title, icon: Icon, children }: {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}) => (
  <div className="space-y-2">
    <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
      <Icon className="h-3.5 w-3.5" />
      {title}
    </div>
    {children}
  </div>
));
Section.displayName = 'Section';

/**
 * Property item with ValueSource indicator
 */
const PropertyItem = memo(({ property }: { property: PropertyInfo }) => {
  const theme = VALUE_SOURCE_THEMES[property.source] || VALUE_SOURCE_THEMES.default;
  const IconComponent = theme.icon;

  return (
    <div className="flex items-center justify-between gap-2 py-1.5 px-2 rounded bg-muted/30 hover:bg-muted/50 transition-colors">
      <div className="flex items-center gap-2 min-w-0">
        <div className={cn("p-1 rounded", theme.bg)}>
          <IconComponent className={cn("h-3 w-3", theme.text)} />
        </div>
        <span className="text-xs font-medium truncate">{property.name}</span>
      </div>
      <div className="flex items-center gap-1.5">
        {property.value !== null && property.value !== undefined ? (
          <span className="text-xs font-mono truncate max-w-[80px]">
            {String(property.value)}
          </span>
        ) : (
          <span className="text-xs text-muted-foreground italic">
            {property.source === 'required' ? 'Not set' : 'null'}
          </span>
        )}
        <Badge
          variant="outline"
          className={cn("text-[9px] px-1.5 py-0 h-4", theme.bg, theme.text)}
        >
          {theme.label}
        </Badge>
      </div>
    </div>
  );
});
PropertyItem.displayName = 'PropertyItem';

export const ComponentDetailPanel = memo(({
  node,
  onClose,
  className,
  isEditingMode = false,
  onNodeUpdate,
  onNodeDelete,
  availableImpls = []
}: ComponentDetailPanelProps) => {
  const [isLabelEditing, setIsLabelEditing] = useState(false);
  const [localLabel, setLocalLabel] = useState(node?.label || '');
  const [params, setParams] = useState<Record<string, string>>({});
  const [newParamName, setNewParamName] = useState('');
  const [isConfigOpen, setIsConfigOpen] = useState(true);

  // Sync local state with node
  useEffect(() => {
    if (node) {
      setLocalLabel(node.label);
      // Extract params from metadata.kwargs if available
      const kwargs = (node as any).metadata?.kwargs || {};
      setParams(kwargs);
    }
  }, [node]);

  const handleLabelSave = useCallback(() => {
    if (node && localLabel !== node.label && onNodeUpdate) {
      onNodeUpdate(node.id, { label: localLabel });
    }
    setIsLabelEditing(false);
  }, [node, localLabel, onNodeUpdate]);

  const handleParamChange = useCallback((name: string, value: string) => {
    setParams(prev => ({ ...prev, [name]: value }));
    if (node && onNodeUpdate) {
      onNodeUpdate(node.id, {
        metadata: { ...((node as any).metadata || {}), kwargs: { ...params, [name]: value } }
      } as any);
    }
  }, [node, params, onNodeUpdate]);

  const handleParamDelete = useCallback((name: string) => {
    setParams(prev => {
      const newParams = { ...prev };
      delete newParams[name];
      return newParams;
    });
    if (node && onNodeUpdate) {
      const newKwargs = { ...params };
      delete newKwargs[name];
      onNodeUpdate(node.id, {
        metadata: { ...((node as any).metadata || {}), kwargs: newKwargs }
      } as any);
    }
  }, [node, params, onNodeUpdate]);

  const handleAddParam = useCallback(() => {
    if (newParamName && !params[newParamName]) {
      handleParamChange(newParamName, '');
      setNewParamName('');
    }
  }, [newParamName, params, handleParamChange]);

  const handleImplChange = useCallback((impl: string) => {
    if (node && onNodeUpdate) {
      onNodeUpdate(node.id, {
        metadata: { ...((node as any).metadata || {}), impl }
      } as any);
    }
  }, [node, onNodeUpdate]);

  if (!node) return null;

  const category = node.category || 'unknown';
  const theme = CATEGORY_THEMES[category] || CATEGORY_THEMES.unknown;
  const inputs = node.inputs || [];
  const outputs = node.outputs || [];
  const source = node.source;
  const origin = node.origin;
  const currentImpl = (node as any).metadata?.impl || '';
  const properties = node.properties || [];
  const propertySummary = node.property_summary;

  return (
    <TooltipProvider>
      <div className={cn(
        "w-72 border-l border-border bg-card flex flex-col h-full",
        className
      )}>
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-border">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <Box className={cn("h-4 w-4 flex-shrink-0", theme.icon)} />
            {isLabelEditing && isEditingMode ? (
              <div className="flex items-center gap-1 flex-1">
                <Input
                  value={localLabel}
                  onChange={(e) => setLocalLabel(e.target.value)}
                  onBlur={handleLabelSave}
                  onKeyDown={(e) => e.key === 'Enter' && handleLabelSave()}
                  className="h-6 text-sm font-semibold"
                  autoFocus
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={handleLabelSave}
                >
                  <Check className="h-3 w-3" />
                </Button>
              </div>
            ) : (
              <div
                className={cn(
                  "text-sm font-semibold truncate",
                  isEditingMode && "cursor-pointer hover:underline"
                )}
                onClick={() => isEditingMode && setIsLabelEditing(true)}
              >
                {localLabel}
                {isEditingMode && <Edit3 className="h-3 w-3 ml-1 inline opacity-50" />}
              </div>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0 flex-shrink-0"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Content */}
        <ScrollArea className="flex-1">
          <div className="p-3 space-y-4">
            {/* Category Badge & Implementation */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className={cn("text-xs", theme.badge)}>
                  {category}
                </Badge>
                {node.can_drill && (
                  <Badge variant="secondary" className="text-[10px]">
                    Drillable
                  </Badge>
                )}
              </div>

              {/* Implementation is set in YAML (Execute tab), not here */}
            </div>

            <Separator />

            {/* Properties Section (ValueSource) */}
            {properties.length > 0 && (
              <>
                <Section title="Properties" icon={AlertCircle}>
                  <div className="space-y-1.5">
                    {propertySummary?.required_count ? (
                      <div className="flex items-center gap-1 text-xs text-red-600 dark:text-red-400 mb-2">
                        <AlertCircle className="h-3 w-3" />
                        <span>{propertySummary.required_count} required value{propertySummary.required_count > 1 ? 's' : ''} not set</span>
                      </div>
                    ) : null}
                    {properties.map((prop) => (
                      <PropertyItem key={prop.name} property={prop} />
                    ))}
                  </div>
                </Section>
                <Separator />
              </>
            )}

            {/* Source Location */}
            {source && (
              <Section title="Source" icon={FileCode}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="text-xs font-mono bg-muted/50 px-2 py-1.5 rounded truncate cursor-pointer hover:bg-muted">
                      {source.file.split('/').pop()}:{source.line}
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <div className="text-xs font-mono">{source.file}:{source.line}</div>
                  </TooltipContent>
                </Tooltip>
              </Section>
            )}

            {/* Origin Info */}
            {origin && (
              <Section title="Origin" icon={Code2}>
                <div className="text-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Type:</span>
                    <Badge variant="outline" className="text-[10px] h-5">
                      {origin.type}
                    </Badge>
                  </div>
                  {origin.create_path && origin.create_path.length > 0 && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Path:</span>
                      <span className="font-mono text-[10px]">
                        {origin.create_path.join('.')}
                      </span>
                    </div>
                  )}
                </div>
                {origin.code_snippet && (
                  <div className="mt-2 text-[10px] font-mono bg-muted/50 px-2 py-1.5 rounded overflow-x-auto whitespace-pre">
                    {origin.code_snippet}
                  </div>
                )}
              </Section>
            )}

            <Separator />

            {/* Input Ports */}
            <Section title={`Inputs (${inputs.length})`} icon={ArrowRight}>
              <PortList ports={inputs} direction="input" />
            </Section>

            {/* Output Ports */}
            <Section title={`Outputs (${outputs.length})`} icon={ArrowLeft}>
              <PortList ports={outputs} direction="output" />
            </Section>

            {/* Configuration */}
            <Separator />

            <Collapsible open={isConfigOpen} onOpenChange={setIsConfigOpen}>
              <CollapsibleTrigger className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide w-full hover:text-foreground transition-colors">
                <Settings2 className="h-3.5 w-3.5" />
                <span className="flex-1 text-left">Parameters</span>
                <ChevronDown className={cn(
                  "h-3.5 w-3.5 transition-transform",
                  isConfigOpen && "rotate-180"
                )} />
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2">
                <div className="space-y-1">
                  {Object.keys(params).length === 0 ? (
                    <div className="text-xs text-muted-foreground italic py-2">
                      No parameters configured
                    </div>
                  ) : (
                    Object.entries(params).map(([key, value]) => (
                      <EditableParam
                        key={key}
                        name={key}
                        value={String(value)}
                        onChange={handleParamChange}
                        onDelete={handleParamDelete}
                        isEditing={isEditingMode}
                      />
                    ))
                  )}

                  {/* Add new parameter */}
                  {isEditingMode && (
                    <div className="flex items-center gap-2 pt-2 border-t border-border/50 mt-2">
                      <Input
                        value={newParamName}
                        onChange={(e) => setNewParamName(e.target.value)}
                        placeholder="param name"
                        className="h-6 text-xs flex-1"
                        onKeyDown={(e) => e.key === 'Enter' && handleAddParam()}
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={handleAddParam}
                        disabled={!newParamName}
                      >
                        <Plus className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="px-3 py-2 border-t border-border space-y-2">
          {/* Delete Button (only in editing mode) */}
          {isEditingMode && onNodeDelete && (
            <Button
              variant="destructive"
              size="sm"
              className="w-full h-8 text-xs"
              onClick={() => {
                if (node) {
                  onNodeDelete(node.id);
                  onClose();
                }
              }}
            >
              <Trash2 className="h-3.5 w-3.5 mr-1.5" />
              Delete Node
            </Button>
          )}
          <div className="text-[10px] text-muted-foreground text-center">
            {isEditingMode ? 'Edit mode - changes auto-save' : 'Click elsewhere to deselect'}
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
});

ComponentDetailPanel.displayName = 'ComponentDetailPanel';
