import React, { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { cn } from '@/shared/lib/utils';

interface ConfigKeyValueListProps {
  data: Record<string, any>;
  depth?: number;
  className?: string;
  initialExpanded?: boolean;
}

export const ConfigKeyValueList: React.FC<ConfigKeyValueListProps> = ({ 
  data, 
  depth = 0,
  className,
  initialExpanded = true
}) => {
  if (!data || typeof data !== 'object' || data === null) return null;

  return (
    <div className={cn("flex flex-col font-mono text-[10px]", className)}>
      {Object.entries(data).map(([key, value]) => (
        <ConfigItem key={key} label={key} value={value} depth={depth} initialExpanded={initialExpanded} />
      ))}
    </div>
  );
};

const ConfigItem = ({ label, value, depth, initialExpanded }: { label: string, value: any, depth: number, initialExpanded: boolean }) => {
  const [isExpanded, setIsExpanded] = useState(initialExpanded);
  const isObject = value !== null && typeof value === 'object';
  const isEmpty = isObject && Object.keys(value).length === 0;
  const isExpandable = isObject && !isEmpty;

  const toggleExpand = (e: React.MouseEvent) => {
    if (isExpandable) {
      e.stopPropagation();
      setIsExpanded(!isExpanded);
    }
  };

  return (
    <div className="flex flex-col">
      <div 
        className={cn(
          "flex items-center gap-1 py-0.5 px-1 hover:bg-accent/20 rounded cursor-default transition-colors group",
          depth > 0 && "ml-2 border-l border-border/20 pl-1"
        )}
        onClick={toggleExpand}
      >
        {isExpandable ? (
          isExpanded ? <ChevronDown className="w-3 h-3 text-muted-foreground/60 shrink-0" /> : <ChevronRight className="w-3 h-3 text-muted-foreground/60 shrink-0" />
        ) : (
          <div className="w-3 shrink-0" />
        )}
        <span className="text-primary/70 font-bold shrink-0">{label}:</span>
        {!isObject && (
          <span className="text-foreground/90 truncate" title={String(value)}>
            {String(value)}
          </span>
        )}
        {isObject && !isExpanded && (
          <span className="text-muted-foreground/40 italic">
            {Array.isArray(value) ? `[...]` : `{...}`}
          </span>
        )}
      </div>
      {isExpandable && isExpanded && (
        <div className="flex flex-col">
          {Array.isArray(value) ? (
            value.map((item, index) => (
              <ConfigItem key={index} label={String(index)} value={item} depth={depth + 1} initialExpanded={initialExpanded} />
            ))
          ) : (
            Object.entries(value).map(([k, v]) => (
              <ConfigItem key={k} label={k} value={v} depth={depth + 1} initialExpanded={initialExpanded} />
            ))
          )}
        </div>
      )}
    </div>
  );
};
