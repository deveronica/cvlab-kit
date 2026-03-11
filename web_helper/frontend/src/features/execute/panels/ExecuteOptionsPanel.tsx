/**
 * ExecuteOptionsPanel - Refined Implementation Selector for Execute Tab
 */
import { memo, useMemo, useState } from 'react';
import { useMultipleCategoryImplementations } from '@/shared/model/useCategoryImplementations';
import { useNodeStore } from '@/entities/node-system/model/nodeStore';
import { ScrollArea } from '@/shared/ui/scroll-area';
import { Input } from '@/shared/ui/input';
import { Badge } from '@/shared/ui/badge';
import { Button } from '@/shared/ui/button';
import { Search, ArrowRightLeft, Box, Library } from 'lucide-react';
import { cn } from '@/shared/lib/utils';

interface ExecuteOptionsPanelProps {
  className?: string;
  agentName: string | null;
  onConfigChange?: (nodeId: string, changes: Record<string, unknown>) => void;
}

export const ExecuteOptionsPanel = memo(function ExecuteOptionsPanel({
  className,
  onConfigChange,
}: ExecuteOptionsPanelProps) {
  const [search, setSearch] = useState('');
  
  const selectedNodeId = useNodeStore((s) => s.selectedNodeId);
  const nodes = useNodeStore((s) => s.nodes);
  const selectedNode = useMemo(() => nodes.find((n) => n.id === selectedNodeId), [nodes, selectedNodeId]);

  const category = selectedNode?.data.category;
  const { data: implementationsMap, isLoading } = useMultipleCategoryImplementations(category ? [category] : []);

  const availableImpls = useMemo(() => {
    if (!category || !implementationsMap) return [];
    return implementationsMap[category] || [];
  }, [category, implementationsMap]);

  const filteredImpls = useMemo(() => {
    const lower = search.toLowerCase();
    return availableImpls.filter(i => i.name.toLowerCase().includes(lower) || i.description?.toLowerCase().includes(lower));
  }, [availableImpls, search]);

  const handleSwap = (implName: string) => {
    if (selectedNodeId && onConfigChange) onConfigChange(selectedNodeId, { implementation: implName });
  };

  if (!selectedNode) {
    return (
      <div className={cn('h-full flex flex-col items-center justify-center opacity-20 p-8 text-center', className)}>
        <Library className="h-12 w-12 mb-4" />
        <p className="text-[10px] font-black tracking-widest uppercase">Select Node to Swap</p>
      </div>
    );
  }

  return (
    <div className={cn('h-full flex flex-col bg-card/50', className)}>
      <div className="h-11 px-4 border-b border-border/40 bg-muted/20 flex items-center shrink-0">
        <span className="text-[10px] font-black tracking-widest text-foreground/70 uppercase">Implementation Library</span>
      </div>

      <div className="p-3 border-b border-border/40 bg-background/40">
        <div className="relative group">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground group-focus-within:text-primary transition-colors" />
          <Input
            placeholder={`Search ${category}s...`}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 pl-8 text-xs bg-background/50 border-border/60 focus-visible:ring-primary/20"
          />
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-3 space-y-2">
          {isLoading ? (
            <div className="py-10 text-center text-[10px] font-bold text-muted-foreground animate-pulse uppercase tracking-widest">Scanning Registry...</div>
          ) : filteredImpls.length === 0 ? (
            <div className="py-10 text-center text-[10px] text-muted-foreground italic">No matches found</div>
          ) : (
            filteredImpls.map((impl) => {
              const isCurrent = impl.name === selectedNode.data.implementation;
              return (
                <div
                  key={impl.name}
                  onClick={() => !isCurrent && handleSwap(impl.name)}
                  className={cn(
                    'group flex flex-col gap-1.5 p-3 rounded-xl border transition-all cursor-pointer shadow-sm',
                    isCurrent 
                      ? 'bg-primary/10 border-primary shadow-primary/10' 
                      : 'bg-background border-border/60 hover:border-primary/40 hover:shadow-md'
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className={cn('text-xs font-black tracking-tight', isCurrent ? 'text-primary' : 'text-foreground/80')}>
                      {impl.name}
                    </span>
                    {isCurrent ? (
                      <Badge className="h-4 text-[8px] font-black bg-primary text-primary-foreground">ACTIVE</Badge>
                    ) : (
                      <ArrowRightLeft className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                    )}
                  </div>
                  {impl.description && (
                    <p className="text-[10px] text-muted-foreground leading-relaxed line-clamp-2 italic opacity-70">
                      {impl.description}
                    </p>
                  )}
                </div>
              );
            })
          )}
        </div>
      </ScrollArea>
    </div>
  );
});
