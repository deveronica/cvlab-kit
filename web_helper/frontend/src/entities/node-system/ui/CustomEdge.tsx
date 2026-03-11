import { memo, useMemo, useState } from 'react';
import { getSmoothStepPath, EdgeLabelRenderer, type EdgeProps } from 'reactflow';
import { MessageSquare, Code2 } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import { useBuilderStore } from '@/features/builder/model/builderStore';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/shared/ui/tooltip';

interface CustomEdgeData {
  edgeType?: 'execution' | 'data' | 'ref';
  flowType?: string;
  variableName?: string; // 항목: 전달된 변수명
  label?: string;
  sequenceIndex?: number;
  extracted_from?: 'setup' | 'train_step';
  is_gapped?: boolean;
  comment?: string;
}

const EDGE_CONFIG = {
  execution: { stroke: '#ffffff', strokeWidth: 3, opacity: 1 },
  data: { stroke: '#22d3ee', strokeWidth: 2.5, opacity: 0.8 },
  ref: { stroke: '#4ade80', strokeWidth: 2, opacity: 0.6 },
};

export const CustomEdge = memo(function CustomEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
  animated,
}: EdgeProps<CustomEdgeData>) {
  const type = data?.edgeType || 'data';
  const typeColors = useBuilderStore(s => s.typeColors);
  const [isHovered, setIsHovered] = useState(false);

  // 항목 4: 연결선 색상을 입력 포트 기준으로 동기화
  const edgeColor = useMemo(() => {
    if (type === 'execution') return '#ffffff';
    // flowType이 데이터 타입 정보를 담고 있다고 가정
    const dataType = data?.flowType || 'any';
    return typeColors[dataType] || typeColors['any'] || '#94a3b8';
  }, [type, data?.flowType, typeColors]);

  const config = useMemo(() => ({
    stroke: edgeColor,
    strokeWidth: type === 'execution' ? 3 : 2.5,
    opacity: selected ? 1 : 0.8,
    // 항목 7: 공백 여부에 따른 점선/실선 구분 (실행선만 해당)
    strokeDasharray: (type === 'execution' && data?.is_gapped) ? '8,6' : undefined
  }), [edgeColor, type, selected, data?.is_gapped]);

  // 항목 4: 연결선 시작점/끝점 불일치 해결
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 24,
    offset: 30,
  });

  return (
    <>
      <TooltipProvider delayDuration={0}>
        {/* 항목 4: 카드 형식에 어울리는 글로우 효과 배경 패스 */}
        <path
          d={edgePath}
          fill="none"
          stroke="hsl(var(--background))"
          strokeWidth={config.strokeWidth + 8}
          strokeLinecap="round"
          className="opacity-100"
        />

        <path
          id={id}
          d={edgePath}
          fill="none"
          stroke={selected ? 'hsl(var(--primary))' : (isHovered ? 'hsl(var(--primary) / 0.8)' : config.stroke)}
          strokeWidth={selected ? config.strokeWidth + 1 : config.strokeWidth}
          strokeDasharray={config.strokeDasharray}
          className={cn(
            "transition-all duration-300 ease-in-out",
            animated && "animate-pulse"
          )}
          style={{
            filter: (selected || isHovered) ? 'drop-shadow(0 0 12px hsl(var(--primary) / 0.6))' : 'none',
            opacity: (selected || isHovered) ? 1 : config.opacity,
          }}
          markerEnd="url(#flow-marker)"
        />

        {/* 항목: 변수명 직접 표시 (선 정중앙에 배치, 선을 가리는 마스킹 효과) */}
        {data?.variableName && (
          <EdgeLabelRenderer>
            <div
              style={{
                position: 'absolute',
                transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`, // 선 정중앙
                pointerEvents: 'none',
                zIndex: 1001,
              }}
              className={cn(
                "px-2 py-0.5 rounded-full transition-all duration-200 shadow-xl",
                "bg-background border border-white/20 backdrop-blur-md", 
                "text-[10px] font-mono font-black tracking-tight",
                selected || isHovered ? "text-primary scale-110" : "text-primary/90",
              )}
            >
              {data.variableName}
            </div>
          </EdgeLabelRenderer>
        )}

        {/* Sequence Index Badge (Creation Order) - ONLY for setup phase */}
        {data?.sequenceIndex !== undefined && data?.extracted_from === 'setup' && (
          <EdgeLabelRenderer>
            <div
              style={{
                position: 'absolute',
                // Place at 20% of path for execution flow, 50% for data dependencies
                transform: `translate(-50%, -50%) translate(${sourceX + (targetX - sourceX) * (data.edgeType === 'execution' ? 0.2 : 0.5)}px, ${sourceY + (targetY - sourceY) * (data.edgeType === 'execution' ? 0.2 : 0.5)}px)`,
                pointerEvents: 'none',
                zIndex: 1002,
              }}
              className={cn(
                "w-5 h-5 flex items-center justify-center rounded-full transition-all duration-300",
                "bg-zinc-950 border-2 border-primary/60 text-primary text-[10px] font-black shadow-[0_0_15px_rgba(var(--primary),0.3)]",
                selected && "border-primary scale-125 ring-4 ring-primary/20 bg-primary text-primary-foreground shadow-primary/40",
                "group-hover:border-primary group-hover:scale-110"
              )}
            >
              {data.sequenceIndex}
            </div>
          </EdgeLabelRenderer>
        )}

        {/* 항목 6: 주석 뱃지 안보이는 버그 수정 (zIndex 상향 및 위치 보정) */}
        {data?.comment && (
          <EdgeLabelRenderer>
            <div
              style={{
                position: 'absolute',
                transform: `translate(-50%, -50%) translate(${labelX}px, ${data?.variableName ? labelY - 24 : labelY}px)`,
                pointerEvents: 'all',
                zIndex: 2000, // Z-INDEX 상향
              }}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
              className="group/badge"
            >
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className={cn(
                    "flex items-center gap-1.5 px-2 py-1 bg-amber-500 text-amber-950 rounded-full shadow-2xl border border-amber-400/50 transition-all cursor-help",
                    "hover:scale-125 hover:rotate-3 active:scale-95",
                    "ring-amber-500/30",
                    isHovered && "ring-4"
                  )}>
                    <MessageSquare className="w-3 h-3 fill-current opacity-80" />
                    <span className="text-[9px] font-black uppercase tracking-widest">Note</span>
                  </div>
                </TooltipTrigger>
                <TooltipContent 
                  side="top" 
                  className="bg-zinc-900/95 backdrop-blur-xl border-zinc-800 shadow-2xl p-4 max-w-[350px] animate-in zoom-in-95 duration-200 z-[3000]"
                >
                  <div className="space-y-2.5">
                    <div className="flex items-center gap-2.5 border-b border-zinc-800 pb-2 mb-2">
                      <div className="p-1 rounded bg-amber-500/10 border border-amber-500/20">
                        <Code2 className="w-4 h-4 text-amber-500" />
                      </div>
                      <span className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">Source Logic</span>
                    </div>
                    <p className="text-[11px] font-mono leading-relaxed text-zinc-300 whitespace-pre-wrap">
                      {data.comment}
                    </p>
                  </div>
                </TooltipContent>
              </Tooltip>
            </div>
          </EdgeLabelRenderer>
        )}
      </TooltipProvider>
    </>
  );
});

CustomEdge.displayName = 'CustomEdge';
