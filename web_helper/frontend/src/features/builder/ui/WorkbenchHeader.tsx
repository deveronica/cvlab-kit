import {
  Monitor,
  ChevronRight,
  Edit3,
  Save,
} from 'lucide-react';
import { Badge } from '@/shared/ui/badge';
import { Button } from '@/shared/ui/button';
import { Separator } from '@/shared/ui/separator';
import { useAgentBuilder } from '@/entities/node-system/model/AgentBuilderContext';

interface WorkbenchHeaderProps {
  phaseLabel?: string;
}

export function WorkbenchHeader({ phaseLabel }: WorkbenchHeaderProps) {
  const context = useAgentBuilder() as any;
  const {
    isEditingMode,
    setIsEditingMode,
    selectedAgent,
    saveToCode,
  } = context;

  return (
    <header className="h-11 border-b border-border/60 bg-[#fbfbfb] dark:bg-[#0c0c0e] flex items-center px-4 justify-between shrink-0 z-50">
      <div className="flex items-center gap-3">
        <div className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center border border-primary/20 shadow-sm">
          <Monitor className="w-3.5 h-3.5 text-primary" />
        </div>
        <div className="flex items-center gap-1.5 text-[11px] font-bold">
          <span className="text-muted-foreground font-medium">cvlab-kit</span>
          <ChevronRight className="w-3 h-3 text-muted-foreground/30" />
          <span className="text-foreground">{selectedAgent?.name || "No Agent Selected"}</span>
          <ChevronRight className="w-3 h-3 text-muted-foreground/30" />
          <Badge variant="outline" className="h-5 px-1.5 text-[9px] bg-primary/5 border-primary/20">
            {phaseLabel}
          </Badge>
        </div>
      </div>
      
      <div className="flex items-center gap-2">
        {!isEditingMode ? (
          <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-[10px] font-bold" onClick={() => setIsEditingMode(true)}>
            <Edit3 className="w-3 h-3" /> SWITCH TO EDIT
          </Button>
        ) : (
          <Button variant="default" size="sm" className="h-7 gap-1.5 text-[10px] font-bold shadow-lg shadow-primary/20" onClick={saveToCode}>
            <Save className="w-3.5 h-3.5" /> SAVE TO AGENT
          </Button>
        )}
        <Separator orientation="vertical" className="h-4 mx-1" />
        <Badge variant="secondary" className="h-6 text-[10px] font-mono opacity-70 italic">v0.2.5-final</Badge>
      </div>
    </header>
  );
}
