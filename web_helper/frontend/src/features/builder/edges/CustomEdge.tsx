import React, { memo } from 'react';
import { EdgeProps, getBezierPath } from 'reactflow';
import { edgeThemes } from '@/shared/config/node-themes';

export const CustomEdge = memo(({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  label,
  selected,
}: EdgeProps) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const theme = selected ? edgeThemes.selected : edgeThemes.default;

  return (
    <>
      <path
        id={id}
        className="react-flow__edge-path"
        d={edgePath}
        stroke={theme.stroke}
        strokeWidth={theme.strokeWidth}
        fill="none"
      />
      {label && (
        <text>
          <textPath
            href={`#${id}`}
            style={{ fontSize: 10 }}
            startOffset="50%"
            textAnchor="middle"
            className="fill-muted-foreground"
          >
            {label}
          </textPath>
        </text>
      )}
    </>
  );
});

CustomEdge.displayName = 'CustomEdge';
