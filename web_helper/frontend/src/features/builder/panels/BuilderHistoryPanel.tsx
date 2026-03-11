import { memo, useState, useEffect } from 'react';
import { History, GitCommit, Clock, User, Inbox, Loader2 } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { ScrollArea } from '@/shared/ui/scroll-area';
import { Badge } from '@/shared/ui/badge';
import { useAgentBuilder } from '@/entities/node-system/model/AgentBuilderContext';

export const BuilderHistoryPanel = memo(function BuilderHistoryPanel({ className }: { className?: string }) {
  const { selectedAgent } = useAgentBuilder();
  const [history, setHistory] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // 항목 8: 실제 백엔드 버전 데이터 페칭
  useEffect(() => {
    const fetchHistory = async () => {
      if (!selectedAgent) return;
      setIsLoading(true);
      try {
        const agentName = selectedAgent.name.replace('.py', '');
        const res = await fetch(`/api/nodes/hierarchy/${agentName}/versions`);
        const data = await res.json();
        if (data.success) {
          setHistory(data.versions.map((v: any) => ({
            id: v.version_id,
            hash: v.version_id.substring(0, 7),
            date: new Date(v.created_at).toLocaleString(),
            message: v.description || `Version ${v.version_number}`,
            author: v.created_by,
            type: 'commit'
          })));
        }
      } catch (e) {
        console.error('Failed to fetch history:', e);
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, [selectedAgent]);

  return (
    <div className={cn('flex flex-col h-full bg-card/30', className)}>
      <div className="h-11 flex items-center px-4 border-b border-border/40 bg-muted/20 shrink-0">
        <div className="flex items-center gap-2 text-primary font-black uppercase tracking-widest text-[10px]">
          <History className="h-4 w-4" />
          <span>Change History</span>
        </div>
      </div>

      <ScrollArea className="flex-1">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-24 opacity-40">
            <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
            <p className="text-[10px] font-black tracking-widest uppercase">Loading History...</p>
          </div>
        ) : !selectedAgent ? (
          <div className="flex flex-col items-center justify-center py-24 opacity-20 px-8 text-center">
            <Inbox className="h-12 w-12 mb-4" />
            <p className="text-[10px] font-black tracking-widest uppercase leading-relaxed">
              No Agent Selected<br/>to Track History
            </p>
          </div>
        ) : history.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 opacity-20 px-8 text-center">
            <GitCommit className="h-12 w-12 mb-4" />
            <p className="text-[10px] font-black tracking-widest uppercase leading-relaxed">
              No History Available<br/>for this Agent
            </p>
          </div>
        ) : (
          <div className="p-4 space-y-3">
            {history.map((item) => (
              <div key={item.id} className="group bg-card border border-border/60 rounded-xl p-3 shadow-sm hover:border-primary/40 transition-all cursor-pointer">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5">
                    <div className="p-1 rounded bg-muted">
                      <GitCommit className="h-3 w-3 text-muted-foreground" />
                    </div>
                    <code className="text-[10px] font-mono font-bold text-primary/80">{item.hash}</code>
                  </div>
                  <span className="text-[9px] text-muted-foreground font-medium flex items-center gap-1">
                    <Clock className="h-2.5 w-2.5" /> {item.date}
                  </span>
                </div>
                <p className="text-[11px] font-bold leading-snug mb-2 group-hover:text-primary transition-colors text-foreground/90">{item.message}</p>
                <div className="flex items-center justify-between pt-2 border-t border-border/10">
                  <div className="flex items-center gap-1.5 opacity-60">
                    <User className="h-2.5 w-2.5" />
                    <span className="text-[9px] font-bold">{item.author}</span>
                  </div>
                  <Badge variant="outline" className="text-[8px] h-4 uppercase tracking-tighter bg-muted/30">{item.type}</Badge>
                </div>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
});
