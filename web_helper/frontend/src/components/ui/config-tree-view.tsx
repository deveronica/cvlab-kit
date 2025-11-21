import React from "react";
/**
 * Configuration Tree View
 *
 * Interactive tree view for YAML configuration with:
 * - Collapsible nested structure
 * - Search/filter functionality
 * - Type badges
 * - Copy buttons
 * - Syntax highlighting
 */

import { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Input } from './input';
import { Button } from './button';
import { Badge } from './badge';
import {
  ChevronRight,
  ChevronDown,
  Search,
  Copy,
  Check,
  FileCode,
} from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './tooltip';
import { cn } from '@/lib/utils';

interface ConfigNode {
  key: string;
  value: any;
  type: 'object' | 'array' | 'string' | 'number' | 'boolean' | 'null';
  children?: ConfigNode[];
  path: string;
}

interface ConfigTreeViewProps {
  config: Record<string, any>;
  title?: string;
  className?: string;
  defaultExpanded?: boolean;
  maxHeight?: string | number;
}

/**
 * Parse configuration object into tree nodes
 */
function parseConfigToTree(
  obj: any,
  parentPath: string = ''
): ConfigNode[] {
  if (!obj || typeof obj !== 'object') return [];

  return Object.entries(obj).map(([key, value]) => {
    const path = parentPath ? `${parentPath}.${key}` : key;
    const type = getValueType(value);

    const node: ConfigNode = {
      key,
      value,
      type,
      path,
    };

    if (type === 'object' || type === 'array') {
      node.children = parseConfigToTree(value, path);
    }

    return node;
  });
}

/**
 * Get value type for badge display
 */
function getValueType(value: any): ConfigNode['type'] {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  if (typeof value === 'object') return 'object';
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'number') return 'number';
  return 'string';
}

/**
 * Format value for display
 */
function formatValue(value: any, type: ConfigNode['type']): string {
  if (value === null) return 'null';
  if (type === 'boolean') return value ? 'true' : 'false';
  if (type === 'string') return `"${value}"`;
  if (type === 'number') return String(value);
  if (type === 'array') return `[${value.length} items]`;
  if (type === 'object') return `{${Object.keys(value).length} keys}`;
  return String(value);
}

/**
 * Tree node component
 */
function TreeNode({
  node,
  searchQuery,
  defaultExpanded,
  level = 0,
}: {
  node: ConfigNode;
  searchQuery: string;
  defaultExpanded: boolean;
  level?: number;
}) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [copied, setCopied] = useState(false);

  const hasChildren = node.children && node.children.length > 0;

  // Filter children based on search query
  const filteredChildren = useMemo(() => {
    if (!searchQuery || !hasChildren) return node.children || [];

    return node.children!.filter((child) =>
      child.key.toLowerCase().includes(searchQuery.toLowerCase()) ||
      child.path.toLowerCase().includes(searchQuery.toLowerCase()) ||
      String(child.value).toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [node.children, searchQuery]);

  // Auto-expand if search matches
  const shouldShow = useMemo(() => {
    if (!searchQuery) return true;

    const matches =
      node.key.toLowerCase().includes(searchQuery.toLowerCase()) ||
      node.path.toLowerCase().includes(searchQuery.toLowerCase()) ||
      String(node.value).toLowerCase().includes(searchQuery.toLowerCase());

    if (matches && hasChildren) {
      setIsExpanded(true);
    }

    return matches || filteredChildren.length > 0;
  }, [node, searchQuery, filteredChildren]);

  if (!shouldShow) return null;

  const handleCopy = async () => {
    const copyText =
      node.type === 'object' || node.type === 'array'
        ? JSON.stringify(node.value, null, 2)
        : String(node.value);

    await navigator.clipboard.writeText(copyText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Type badge color
  const typeBadgeVariant = (): 'default' | 'secondary' | 'outline' => {
    if (node.type === 'object' || node.type === 'array') return 'default';
    if (node.type === 'boolean' || node.type === 'number') return 'secondary';
    return 'outline';
  };

  return (
    <div className="select-none">
      <div
        className={cn(
          'flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-muted/50 transition-colors duration-200 group',
          level > 0 && 'ml-4'
        )}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
      >
        {/* Expand/Collapse */}
        {hasChildren ? (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex-shrink-0 p-0.5 hover:bg-accent rounded transition-colors duration-200"
            aria-label={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
            )}
          </button>
        ) : (
          <div className="w-4" />
        )}

        {/* Key */}
        <span className="font-mono text-sm font-medium text-foreground">
          {node.key}
        </span>

        {/* Type Badge */}
        <Badge variant={typeBadgeVariant()} className="text-[10px] px-1.5 py-0">
          {node.type}
        </Badge>

        {/* Value (for primitives) */}
        {!hasChildren && (
          <span className="flex-1 font-mono text-xs text-muted-foreground truncate">
            {formatValue(node.value, node.type)}
          </span>
        )}

        {/* Copy Button */}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                onClick={handleCopy}
                aria-label="Copy value"
              >
                {copied ? (
                  <Check className="h-3 w-3 text-green-600 dark:text-green-400" />
                ) : (
                  <Copy className="h-3 w-3" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>{copied ? 'Copied!' : 'Copy value'}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Children */}
      {hasChildren && isExpanded && (
        <div className="mt-0.5">
          {filteredChildren.map((child, index) => (
            <TreeNode
              key={`${child.path}-${index}`}
              node={child}
              searchQuery={searchQuery}
              defaultExpanded={defaultExpanded}
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Main ConfigTreeView component
 */
export function ConfigTreeView({
  config,
  title = 'Configuration',
  className,
  defaultExpanded = false,
  maxHeight = 600,
}: ConfigTreeViewProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandAll, setExpandAll] = useState(defaultExpanded);

  // Parse config to tree
  const treeNodes = useMemo(() => {
    return parseConfigToTree(config);
  }, [config]);

  // Count total keys
  const totalKeys = useMemo(() => {
    const countKeys = (nodes: ConfigNode[]): number => {
      return nodes.reduce((count, node) => {
        const childCount = node.children ? countKeys(node.children) : 0;
        return count + 1 + childCount;
      }, 0);
    };
    return countKeys(treeNodes);
  }, [treeNodes]);

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileCode className="h-5 w-5 text-primary" />
            <CardTitle size="base">{title}</CardTitle>
            <Badge variant="secondary" className="text-xs">
              {totalKeys} keys
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setExpandAll(!expandAll)}
              className="text-xs h-7"
            >
              {expandAll ? 'Collapse All' : 'Expand All'}
            </Button>
          </div>
        </div>

        {/* Search */}
        <div className="relative mt-3">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search configuration..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-9 text-sm"
          />
        </div>
      </CardHeader>

      <CardContent>
        <div
          className="overflow-y-auto space-y-0.5 pr-2"
          style={{ maxHeight }}
        >
          {treeNodes.length === 0 ? (
            <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
              No configuration data available
            </div>
          ) : (
            treeNodes.map((node, index) => (
              <TreeNode
                key={`${node.path}-${index}`}
                node={node}
                searchQuery={searchQuery}
                defaultExpanded={expandAll}
              />
            ))
          )}

          {searchQuery && treeNodes.every((node) => {
            const matches =
              node.key.toLowerCase().includes(searchQuery.toLowerCase()) ||
              String(node.value).toLowerCase().includes(searchQuery.toLowerCase());
            const hasVisibleChildren = node.children?.some((child) =>
              child.key.toLowerCase().includes(searchQuery.toLowerCase())
            );
            return !matches && !hasVisibleChildren;
          }) && (
            <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
              No results found for "{searchQuery}"
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
