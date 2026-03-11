/**
 * NodePropertiesPanel - Side panel for selected node details
 *
 * Features:
 * - Shows node metadata (category, label, implementation)
 * - Displays and edits config parameters
 * - Shows port information
 * - Source code location link
 */

import React, { memo, useState, useCallback } from 'react';
import {
  Settings,
  FileCode,
  ArrowRight,
  ArrowLeft,
  Info,
  Edit3,
  Check,
  X,
  ExternalLink,
  Cpu,
  TrendingDown,
  Database,
  Loader,
  Palette,
  Activity,
  Clock,
  Shuffle,
  Save,
  HelpCircle,
} from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { Badge } from '@/shared/ui/badge';
import { Button } from '@/shared/ui/button';
import { Input } from '@/shared/ui/input';
import { Label } from '@/shared/ui/label';
import { Separator } from '@/shared/ui/separator';
import { ScrollArea } from '@/shared/ui/scroll-area';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/shared/ui/collapsible';
import { getCategoryTheme } from '@/shared/config/node-themes';
import { getPortTheme } from '@/shared/config/port-themes';
import { PortType } from '@/shared/model/node-graph';
import type { ComponentCategory, PropertyInfo, PropertySummary, ValueSource } from '@/shared/model/hierarchy';
import type { ConfigValue } from '@/shared/model/types';
import { AlertCircle, FileText, Hash, Link2, Zap } from 'lucide-react';

// ValueSource theme mapping
const VALUE_SOURCE_THEMES: Record<string, { bg: string; text: string; icon: React.ElementType; label: string }> = {
  required: { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-400', icon: AlertCircle, label: 'Required' },
  config: { bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-400', icon: FileText, label: 'Config' },
  default: { bg: 'bg-gray-100 dark:bg-gray-800/30', text: 'text-gray-600 dark:text-gray-400', icon: Hash, label: 'Default' },
  hardcode: { bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-400', icon: Zap, label: 'Hardcode' },
  connected: { bg: 'bg-green-100 dark:bg-green-900/30', text: 'text-green-700 dark:text-green-400', icon: Link2, label: 'Connected' },
};

// Category Icons
const CATEGORY_ICONS: Record<string, React.ElementType> = {
  model: Cpu,
  optimizer: TrendingDown,
  loss: Activity,
  dataset: Database,
  dataloader: Loader,
  transform: Palette,
  metric: Activity,
  scheduler: Clock,
  sampler: Shuffle,
  checkpoint: Save,
  unknown: HelpCircle,
};

interface Port {
  name: string;
  type: string | PortType;
}

interface SelectedNodeData {
  id: string;
  label: string;
  category?: ComponentCategory;
  implementation?: string;
  config?: Record<string, any>;
  inputs?: Port[];
  outputs?: Port[];
  source?: {
    file: string;
    line: number;
  };
  metadata?: Record<string, any>;
  // Properties with ValueSource (Phase 1)
  properties?: PropertyInfo[];
  property_summary?: PropertySummary;
}

interface NodePropertiesPanelProps {
  node: SelectedNodeData | null;
  onConfigChange?: (nodeId: string, key: string, value: ConfigValue) => void;
  onClose?: () => void;
  className?: string;
}

// Port type mapping
function mapStringToPortType(typeStr: string | PortType): PortType {
  if (Object.values(PortType).includes(typeStr as PortType)) {
    return typeStr as PortType;
  }
  const typeMapping: Record<string, PortType> = {
    dataset: PortType.MODULE,
    indices: PortType.LIST,
    batch: PortType.TENSOR,
    forward: PortType.TENSOR,
    data: PortType.TENSOR,
    value: PortType.SCALAR,
    parameters: PortType.PARAMETERS,
    out: PortType.ANY,
  };
  return typeMapping[typeStr.toLowerCase()] || PortType.ANY;
}

// Editable Config Field
const ConfigField = memo(({
  name,
  value,
  onSave,
}: {
  name: string;
  value: ConfigValue;
  onSave: (value: ConfigValue) => void;
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(String(value));

  const handleSave = useCallback(() => {
    // Try to parse as number or boolean
    let parsedValue: ConfigValue = editValue;
    if (editValue === 'true') parsedValue = true;
    else if (editValue === 'false') parsedValue = false;
    else if (!isNaN(Number(editValue)) && editValue !== '') parsedValue = Number(editValue);

    onSave(parsedValue);
    setIsEditing(false);
  }, [editValue, onSave]);

  const handleCancel = useCallback(() => {
    setEditValue(String(value));
    setIsEditing(false);
  }, [value]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSave();
    if (e.key === 'Escape') handleCancel();
  }, [handleSave, handleCancel]);

  return (
    <div className="flex items-center justify-between gap-2 py-1 px-2 rounded hover:bg-muted/50 group">
      <Label className="text-xs text-muted-foreground font-medium min-w-[80px]">
        {name}
      </Label>

      {isEditing ? (
        <div className="flex items-center gap-1 flex-1">
          <Input
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onKeyDown={handleKeyDown}
            className="h-6 text-xs font-mono"
            autoFocus
          />
          <Button size="icon" variant="ghost" className="h-5 w-5" onClick={handleSave}>
            <Check className="h-3 w-3 text-green-600" />
          </Button>
          <Button size="icon" variant="ghost" className="h-5 w-5" onClick={handleCancel}>
            <X className="h-3 w-3 text-red-600" />
          </Button>
        </div>
      ) : (
        <div className="flex items-center gap-1 flex-1 justify-end">
          <span className="text-xs font-mono truncate max-w-[120px]">
            {typeof value === 'object' ? JSON.stringify(value) : String(value)}
          </span>
          <Button
            size="icon"
            variant="ghost"
            className="h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={() => setIsEditing(true)}
          >
            <Edit3 className="h-3 w-3" />
          </Button>
        </div>
      )}
    </div>
  );
});
ConfigField.displayName = 'ConfigField';

// Port List Item
const PortItem = memo(({ port, direction }: { port: Port; direction: 'input' | 'output' }) => {
  const portType = mapStringToPortType(port.type);
  const theme = getPortTheme(portType);

  return (
    <div className="flex items-center gap-2 py-0.5 px-2">
      {direction === 'input' ? (
        <ArrowRight className="h-3 w-3 text-muted-foreground" />
      ) : (
        <ArrowLeft className="h-3 w-3 text-muted-foreground" />
      )}
      <div
        className="w-2 h-2 rounded-full"
        style={{ backgroundColor: theme.color }}
      />
      <span className="text-xs font-medium">{port.name}</span>
      <Badge variant="outline" className="text-[9px] px-1 py-0 h-4 ml-auto">
        {theme.label}
      </Badge>
    </div>
  );
});
PortItem.displayName = 'PortItem';

// Property Item with ValueSource indicator
const PropertyItem = memo(({ property }: { property: PropertyInfo }) => {
  const theme = VALUE_SOURCE_THEMES[property.source] || VALUE_SOURCE_THEMES.default;
  const IconComponent = theme.icon;

  return (
    <div className="flex items-center justify-between gap-2 py-1.5 px-2 rounded hover:bg-muted/50">
      <div className="flex items-center gap-2 min-w-0">
        <div className={cn("p-1 rounded", theme.bg)}>
          <IconComponent className={cn("h-3 w-3", theme.text)} />
        </div>
        <span className="text-xs font-medium truncate">{property.name}</span>
      </div>
      <div className="flex items-center gap-1.5">
        {property.value !== null && property.value !== undefined ? (
          <span className="text-xs font-mono truncate max-w-[100px]">
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

export const NodePropertiesPanel = memo(({
  node,
  onConfigChange,
  onClose,
  className,
}: NodePropertiesPanelProps) => {
  const [propertiesOpen, setPropertiesOpen] = useState(true);
  const [configOpen, setConfigOpen] = useState(true);
  const [portsOpen, setPortsOpen] = useState(true);

  if (!node) {
    return (
      <div className={cn(
        "w-[280px] border-l bg-background flex flex-col items-center justify-center text-muted-foreground",
        className
      )}>
        <Info className="h-8 w-8 mb-2 opacity-50" />
        <p className="text-sm">Select a node to view properties</p>
      </div>
    );
  }

  const category = node.category || 'unknown';
  const theme = getCategoryTheme(category);
  const IconComponent = CATEGORY_ICONS[category] || CATEGORY_ICONS.unknown;
  const config = node.config || {};
  const inputs = node.inputs || [];
  const outputs = node.outputs || [];
  const properties = node.properties || [];
  const propertySummary = node.property_summary;

  const handleConfigSave = useCallback((key: string, value: ConfigValue) => {
    if (onConfigChange) {
      onConfigChange(node.id, key, value);
    }
  }, [node.id, onConfigChange]);

  return (
    <div className={cn("w-[280px] border-l bg-background flex flex-col", className)}>
      {/* Header */}
      <div className="p-3 border-b">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <div className={cn("p-1.5 rounded", theme.badge)}>
              <IconComponent className="h-4 w-4" />
            </div>
            <div>
              <div className={cn(
                "text-[10px] font-semibold uppercase tracking-wide",
                theme.icon
              )}>
                {category}
              </div>
              <div className="text-sm font-medium">{node.label}</div>
            </div>
          </div>
          {onClose && (
            <Button size="icon" variant="ghost" className="h-6 w-6" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>

        {/* Implementation */}
        {node.implementation && (
          <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
            <Settings className="h-3 w-3" />
            <span className="font-mono">{node.implementation}</span>
          </div>
        )}

        {/* Source link */}
        {node.source && (
          <div className="mt-1 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground cursor-pointer">
            <FileCode className="h-3 w-3" />
            <span className="truncate">{node.source.file}:{node.source.line}</span>
            <ExternalLink className="h-3 w-3 ml-auto" />
          </div>
        )}
      </div>

      <ScrollArea className="flex-1">
        {/* Properties Section (ValueSource) */}
        {properties.length > 0 && (
          <>
            <Collapsible open={propertiesOpen} onOpenChange={setPropertiesOpen}>
              <CollapsibleTrigger asChild>
                <div className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-muted/50">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium">Properties</span>
                  </div>
                  <div className="flex items-center gap-1">
                    {propertySummary?.required_count ? (
                      <Badge variant="destructive" className="text-[9px] px-1.5 py-0 h-4">
                        !{propertySummary.required_count}
                      </Badge>
                    ) : null}
                    <Badge variant="secondary" className="text-[10px]">
                      {properties.length}
                    </Badge>
                  </div>
                </div>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="px-1 pb-2">
                  {properties.map((prop) => (
                    <PropertyItem key={prop.name} property={prop} />
                  ))}
                </div>
              </CollapsibleContent>
            </Collapsible>
            <Separator />
          </>
        )}

        {/* Config Section - Only show if enabled */}
        {(configOpen) && Object.keys(config).length > 0 && (
          <>
            <Collapsible open={configOpen} onOpenChange={setConfigOpen}>
              <CollapsibleTrigger asChild>
                <div className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-muted/50">
                  <div className="flex items-center gap-2">
                    <Settings className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium">Configuration</span>
                  </div>
                  <Badge variant="secondary" className="text-[10px]">
                    {Object.keys(config).length}
                  </Badge>
                </div>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="px-1 pb-2">
                  {Object.keys(config).length > 0 ? (
                    Object.entries(config).map(([key, value]) => (
                      <ConfigField
                        key={key}
                        name={key}
                        value={value}
                        onSave={(newValue) => handleConfigSave(key, newValue)}
                      />
                    ))
                  ) : (
                    <div className="px-3 py-2 text-xs text-muted-foreground italic">
                      No configuration parameters
                    </div>
                  )}
                </div>
              </CollapsibleContent>
            </Collapsible>
            <Separator />
          </>
        )}

        <Separator />

        {/* Ports Section */}
        <Collapsible open={portsOpen} onOpenChange={setPortsOpen}>
          <CollapsibleTrigger asChild>
            <div className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-muted/50">
              <div className="flex items-center gap-2">
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Ports</span>
              </div>
              <Badge variant="secondary" className="text-[10px]">
                {inputs.length + outputs.length}
              </Badge>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="px-1 pb-2">
              {/* Inputs */}
              {inputs.length > 0 && (
                <div className="mb-2">
                  <div className="px-2 py-1 text-[10px] font-semibold text-muted-foreground uppercase">
                    Inputs
                  </div>
                  {inputs.map((port) => (
                    <PortItem key={port.name} port={port} direction="input" />
                  ))}
                </div>
              )}

              {/* Outputs */}
              {outputs.length > 0 && (
                <div>
                  <div className="px-2 py-1 text-[10px] font-semibold text-muted-foreground uppercase">
                    Outputs
                  </div>
                  {outputs.map((port) => (
                    <PortItem key={port.name} port={port} direction="output" />
                  ))}
                </div>
              )}

              {inputs.length === 0 && outputs.length === 0 && (
                <div className="px-3 py-2 text-xs text-muted-foreground italic">
                  No ports defined
                </div>
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>

        <Separator />

        {/* Metadata Section */}
        {node.metadata && Object.keys(node.metadata).length > 0 && (
          <div className="px-3 py-2">
            <div className="flex items-center gap-2 mb-2">
              <Info className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Metadata</span>
            </div>
            <div className="space-y-1">
              {Object.entries(node.metadata).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">{key}</span>
                  <span className="font-mono">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </ScrollArea>
    </div>
  );
});

NodePropertiesPanel.displayName = 'NodePropertiesPanel';
