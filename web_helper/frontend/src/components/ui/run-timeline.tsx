import React from "react";
/**
 * Run Timeline Visualization - Modern Gantt Chart
 *
 * Features:
 * - Compact, professional Gantt chart design
 * - Horizontal scrolling for many runs
 * - Time grid with markers
 * - Status-based color coding
 * - Hover tooltips with detailed timing
 */

import { useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Clock, Calendar, TrendingUp } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './tooltip';

interface RunTimelineProps {
  runs: Array<{
    run_name: string;
    status: string;
    started_at?: string;
    finished_at?: string;
    last_updated?: string;
  }>;
  className?: string;
  onRunClick?: (runName: string) => void;
}

// Status color mapping with flat, clean design (semantic colors only)
const STATUS_COLORS = {
  completed: {
    bg: 'bg-emerald-100/80 dark:bg-emerald-950/40',
    border: 'border-emerald-500/30 dark:border-emerald-500/50',
    dot: 'bg-emerald-500',
    text: 'text-emerald-700 dark:text-emerald-400',
    shadow: '',
  },
  running: {
    bg: 'bg-blue-100/80 dark:bg-blue-950/40',
    border: 'border-blue-500/30 dark:border-blue-500/50',
    dot: 'bg-blue-500',
    text: 'text-blue-700 dark:text-blue-400',
    shadow: '',
  },
  failed: {
    bg: 'bg-rose-100/80 dark:bg-rose-950/40',
    border: 'border-rose-500/30 dark:border-rose-500/50',
    dot: 'bg-rose-500',
    text: 'text-rose-700 dark:text-rose-400',
    shadow: '',
  },
  pending: {
    bg: 'bg-amber-100/80 dark:bg-amber-950/40',
    border: 'border-amber-500/30 dark:border-amber-500/50',
    dot: 'bg-amber-500',
    text: 'text-amber-700 dark:text-amber-400',
    shadow: '',
  },
  default: {
    bg: 'bg-muted/50',
    border: 'border-border',
    dot: 'bg-muted-foreground',
    text: 'text-muted-foreground',
    shadow: '',
  },
} as const;

// Helper function: Format duration from milliseconds to human-readable string
const formatDuration = (ms: number): string => {
  const minutes = Math.floor(ms / (1000 * 60));
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m`;
  return `${Math.round(ms / 1000)}s`;
};

// Helper function: Convert TimeRange to hours
const getIntervalHours = (timeRange: TimeRange, totalTimeRangeMs?: number): number => {
  const mapping: Record<TimeRange, number> = {
    '1h': 1,
    '6h': 6,
    '1d': 24,
    '3d': 72,
    '1w': 168,
    'all': 24, // Fallback if totalTimeRangeMs not provided
  };

  // For 'all' mode: calculate optimal interval to fit all runs in one view
  if (timeRange === 'all' && totalTimeRangeMs !== undefined) {
    const totalHours = totalTimeRangeMs / (1000 * 60 * 60);
    const CARD_WIDTH_PX = 1200; // Target card width
    const LEFT_MARGIN = 24;
    const RIGHT_MARGIN = 24;
    const AVAILABLE_WIDTH = CARD_WIDTH_PX - LEFT_MARGIN - RIGHT_MARGIN;
    const PX_PER_INTERVAL = 150; // Each interval should be ~150px

    // Calculate how many intervals fit in available width
    const maxIntervals = AVAILABLE_WIDTH / PX_PER_INTERVAL;

    // Calculate required interval hours to fit all data
    const requiredIntervalHours = totalHours / maxIntervals;

    // Choose the smallest standard interval that's >= required
    const standardIntervals = [1, 6, 24, 72, 168]; // 1h, 6h, 1d, 3d, 1w
    const selectedInterval = standardIntervals.find(interval => interval >= requiredIntervalHours) || 168;

    return selectedInterval;
  }

  return mapping[timeRange];
};

// Data structures for segment-based timeline
interface TimeGap {
  startTime: number;
  endTime: number;
  durationHours: number;
  shouldCompress: boolean;
}

interface TimelineSegment {
  type: 'normal' | 'compressed';
  startTime: number;
  endTime: number;
  offsetPx: number;      // Cumulative x coordinate
  displayWidthPx: number;
}

// Detect _gaps using interval-based logic (not time-based)
/**
 * PHILOSOPHY: Segment 생성 - Interval 경계 기준 양자화
 *
 * Segment의 경계는 interval 경계(n × intervalMs)를 따릅니다.
 * - Interval 경계: 0시, 6시, 12시, 18시, ... (_intervalHours = 6 기준)
 * - Gap detection: Interval 단위로 빈 구간 찾기 (8+ intervals → compress)
 * - Gap boundaries: 항상 interval 경계 (gapStartIndex × intervalMs)
 *
 * 이 함수는 run이 없는 interval을 찾아 gap으로 반환합니다.
 * Gap의 시작/끝 시간은 항상 interval 경계입니다.
 *
 * NOTE: Time marker는 별개로 150px 픽셀 간격으로 생성됩니다 (시간 기준 아님).
 */
const detectTimeGaps = (
  runs: Array<{ started_at?: string; finished_at?: string }>,
  _intervalHours: number
): TimeGap[] => {
  const compressionThreshold = 8; // 8 consecutive empty intervals

  if (runs.length === 0) return [];

  // Get all valid runs with timestamps
  const validRuns = runs.filter(r => r.started_at).map(run => {
    const startTime = new Date(run.started_at!).getTime();
    let endTime = run.finished_at
      ? new Date(run.finished_at).getTime()
      : startTime + 3600000;

    // Fix bad data
    if (endTime < startTime) endTime = startTime + 3600000;
    if (endTime === startTime) endTime = startTime + 3600000;

    return { startTime, endTime };
  });

  if (validRuns.length === 0) return [];

  // Find min/max time
  const allTimes = validRuns.flatMap(r => [r.startTime, r.endTime]);
  const minTime = Math.min(...allTimes);
  const maxTime = Math.max(...allTimes);

  const intervalMs = _intervalHours * 60 * 60 * 1000;

  // Calculate which intervals contain runs (자정 기준 고정)
  // Interval은 항상 UTC 자정(00:00) 기준으로 정렬
  // 예: 6시간 interval → 0시, 6시, 12시, 18시 정시
  const minDate = new Date(minTime);
  const minDayStart = new Date(Date.UTC(minDate.getUTCFullYear(), minDate.getUTCMonth(), minDate.getUTCDate())).getTime();

  const firstIntervalIndex = Math.floor((minTime - minDayStart) / intervalMs);
  const lastIntervalIndex = Math.ceil((maxTime - minDayStart) / intervalMs);

  const intervalsWithRuns = new Set<number>();

  validRuns.forEach(run => {
    const runStartIndex = Math.floor((run.startTime - minDayStart) / intervalMs);
    const runEndIndex = Math.floor((run.endTime - minDayStart) / intervalMs);

    // Mark all intervals that this run touches
    for (let i = runStartIndex; i <= runEndIndex; i++) {
      intervalsWithRuns.add(i);
    }
  });

  // Find consecutive empty intervals (_gaps)
  const _gaps: TimeGap[] = [];
  let gapStartIndex: number | null = null;
  let emptyCount = 0;

  for (let i = firstIntervalIndex; i <= lastIntervalIndex; i++) {
    if (!intervalsWithRuns.has(i)) {
      // Empty interval
      if (gapStartIndex === null) {
        gapStartIndex = i;
        emptyCount = 1;
      } else {
        emptyCount++;
      }
    } else {
      // Interval has run - end gap if exists
      if (gapStartIndex !== null && emptyCount > 0) {
        const gapStartTime = minDayStart + (gapStartIndex * intervalMs);
        const gapEndTime = minDayStart + ((gapStartIndex + emptyCount) * intervalMs);
        const gapHours = emptyCount * _intervalHours;

        _gaps.push({
          startTime: gapStartTime,
          endTime: gapEndTime,
          durationHours: gapHours,
          shouldCompress: emptyCount >= compressionThreshold,
        });
      }
      gapStartIndex = null;
      emptyCount = 0;
    }
  }

  // Handle final gap
  if (gapStartIndex !== null && emptyCount > 0) {
    const gapStartTime = minDayStart + (gapStartIndex * intervalMs);
    const gapEndTime = minDayStart + ((gapStartIndex + emptyCount) * intervalMs);
    const gapHours = emptyCount * _intervalHours;

    _gaps.push({
      startTime: gapStartTime,
      endTime: gapEndTime,
      durationHours: gapHours,
      shouldCompress: emptyCount >= compressionThreshold,
    });
  }

  return _gaps;
};

// Timeline layout constants
const PX_PER_INTERVAL = 150; // Pixel width per time interval

// Create timeline segments with time-proportional layout and optional compression
// Compress: n intervals (e.g., 72h gap) → 150px total (25px fade + 100px compressed + 25px fade)
/**
 * PHILOSOPHY: Segment 생성 - Interval 경계 기준
 *
 * Segment는 gap(interval 경계)을 기준으로 생성됩니다.
 * - Normal segment: minTime ~ gap.startTime (실제 시간 비례)
 * - Compressed segment: gap.startTime ~ gap.endTime (450px 고정, 5등분)
 *
 * Segment는 timeToPixel/_pixelToTime의 기준이 됩니다.
 * - 모든 시간↔픽셀 변환은 segment 기반 선형 보간
 * - Run bar, Time marker 모두 동일한 segments 사용
 */
const createSegments = (
  minTime: number,
  maxTime: number,
  _gaps: TimeGap[],
  _pxPerHour: number,
  _intervalHours: number
): TimelineSegment[] => {
  const segments: TimelineSegment[] = [];
  const intervalMs = _intervalHours * 60 * 60 * 1000;
  const _MARGIN_INTERVALS = 1; // Reserve 1 interval on each side for margins

  // Calculate minDayStart (UTC midnight of minTime)
  const minDate = new Date(minTime);
  const minDayStart = new Date(Date.UTC(
    minDate.getUTCFullYear(),
    minDate.getUTCMonth(),
    minDate.getUTCDate()
  )).getTime();

  // Calculate first and last interval indices
  const firstIntervalIndex = Math.floor((minTime - minDayStart) / intervalMs);
  const lastIntervalIndex = Math.ceil((maxTime - minDayStart) / intervalMs);

  // Sort compressed gaps
  const compressedGaps = _gaps.filter(g => g.shouldCompress).sort((a, b) => a.startTime - b.startTime);

  let currentOffset = 0;
  let i = firstIntervalIndex;

  while (i <= lastIntervalIndex) {
    const intervalStart = minDayStart + (i * intervalMs);
    const intervalEnd = minDayStart + ((i + 1) * intervalMs);

    // Check if this interval is part of a compressed gap
    const compressedGap = compressedGaps.find(gap => {
      const gapStartIdx = Math.floor((gap.startTime - minDayStart) / intervalMs);
      const gapEndIdx = Math.floor((gap.endTime - minDayStart) / intervalMs);
      return i >= gapStartIdx && i < gapEndIdx;
    });

    if (compressedGap) {
      // Pattern: Run → Margin(1) → Compressed(n-2) → Margin(1) → Run
      const gapStartIdx = Math.floor((compressedGap.startTime - minDayStart) / intervalMs);
      const gapEndIdx = Math.floor((compressedGap.endTime - minDayStart) / intervalMs);
      const _totalIntervals = gapEndIdx - gapStartIdx;

      // Leading margin: _MARGIN_INTERVALS normal segments
      for (let m = 0; m < _MARGIN_INTERVALS; m++) {
        const marginIdx = gapStartIdx + m;
        const marginStart = minDayStart + (marginIdx * intervalMs);
        const marginEnd = minDayStart + ((marginIdx + 1) * intervalMs);

        segments.push({
          type: 'normal',
          startTime: marginStart,
          endTime: marginEnd,
          offsetPx: currentOffset,
          displayWidthPx: PX_PER_INTERVAL,
        });
        currentOffset += PX_PER_INTERVAL;
      }

      // Compressed center: (n - 2*_MARGIN_INTERVALS) intervals → 150px
      const compressedStartIdx = gapStartIdx + _MARGIN_INTERVALS;
      const compressedEndIdx = gapEndIdx - _MARGIN_INTERVALS;
      const compressedStart = minDayStart + (compressedStartIdx * intervalMs);
      const compressedEnd = minDayStart + (compressedEndIdx * intervalMs);

      segments.push({
        type: 'compressed',
        startTime: compressedStart,
        endTime: compressedEnd,
        offsetPx: currentOffset,
        displayWidthPx: PX_PER_INTERVAL, // 150px
      });
      currentOffset += PX_PER_INTERVAL;

      // Trailing margin: _MARGIN_INTERVALS normal segments
      for (let m = 0; m < _MARGIN_INTERVALS; m++) {
        const marginIdx = gapEndIdx - _MARGIN_INTERVALS + m;
        const marginStart = minDayStart + (marginIdx * intervalMs);
        const marginEnd = minDayStart + ((marginIdx + 1) * intervalMs);

        segments.push({
          type: 'normal',
          startTime: marginStart,
          endTime: marginEnd,
          offsetPx: currentOffset,
          displayWidthPx: PX_PER_INTERVAL,
        });
        currentOffset += PX_PER_INTERVAL;
      }

      // Skip all intervals in this compressed gap
      i = gapEndIdx;
    } else {
      // Normal interval: 150px
      const segmentStart = Math.max(intervalStart, minTime);
      const segmentEnd = Math.min(intervalEnd, maxTime);

      if (segmentStart < segmentEnd) {
        segments.push({
          type: 'normal',
          startTime: segmentStart,
          endTime: segmentEnd,
          offsetPx: currentOffset,
          displayWidthPx: PX_PER_INTERVAL,
        });
        currentOffset += PX_PER_INTERVAL;
      }

      i++;
    }
  }

  // Calculate total width covered by segments
  const lastSeg = segments[segments.length - 1];
  const _totalSegmentWidth = lastSeg ? lastSeg.offsetPx + lastSeg.displayWidthPx : 0;

  return segments;
};

// Convert time to pixel position using segments
const timeToPixel = (
  time: number,
  segments: TimelineSegment[],
  _gaps: TimeGap[],
  _intervalHours: number,
  debugInfo?: any[]
): number => {
  // Find the segment containing this time
  for (const segment of segments) {
    if (time >= segment.startTime && time <= segment.endTime) {
      // Linear interpolation within segment
      const segmentProgress = (time - segment.startTime) / (segment.endTime - segment.startTime);
      const pixelPos = segment.offsetPx + (segmentProgress * segment.displayWidthPx);

      if (debugInfo) {
        debugInfo.push({
          time: new Date(time).toISOString(),
          segmentType: segment.type,
          segmentIdx: segments.indexOf(segment),
          segmentProgress: segmentProgress.toFixed(4),
          calculatedPixelPos: pixelPos.toFixed(2),
        });
      }

      return pixelPos;
    }
  }

  // Fallback: if time is before first segment
  if (time < segments[0].startTime) {
    if (debugInfo) debugInfo.push({ error: 'Before first segment' });
    return segments[0].offsetPx;
  }

  // Fallback: if time is after last segment
  const lastSeg = segments[segments.length - 1];
  if (debugInfo) debugInfo.push({ error: 'After last segment' });
  return lastSeg.offsetPx + lastSeg.displayWidthPx;
};

type TimeRange = '1h' | '6h' | '1d' | '3d' | '1w' | 'all';

const _TIME_RANGE_MS: Record<TimeRange, number | null> = {
  '1h': 60 * 60 * 1000,
  '6h': 6 * 60 * 60 * 1000,
  '1d': 24 * 60 * 60 * 1000,
  '3d': 3 * 24 * 60 * 60 * 1000,
  '1w': 7 * 24 * 60 * 60 * 1000,
  'all': null,
};

const TIME_RANGE_LABELS: Record<TimeRange, string> = {
  '1h': '1시간 간격',
  '6h': '6시간 간격',
  '1d': '1일 간격',
  '3d': '3일 간격',
  '1w': '1주 간격',
  'all': '자동',
};

export function RunTimeline({ runs, className, onRunClick }: RunTimelineProps) {
  const [timeRange, setTimeRange] = useState<TimeRange>('all');
  // Layout constants
  const CONTAINER_MARGIN_LEFT = 24;
  const CONTAINER_MARGIN_RIGHT = 24;
  const CONTAINER_MARGIN_TOP = 10;
  const CONTAINER_MARGIN_BOTTOM = 10;

  const scrollContainerRef = React.useRef<HTMLDivElement>(null);
  const cardContentRef = React.useRef<HTMLDivElement>(null);
  const [measuredLabelWidth, setMeasuredLabelWidth] = React.useState<number | null>(null);
  const [measuredGreenBounds, setMeasuredGreenBounds] = React.useState<{ top: number; height: number } | null>(null);
  const [measuredScrollWidth, _setMeasuredScrollWidth] = React.useState<number | null>(null);
  const [windowWidth, setWindowWidth] = React.useState<number>(typeof window !== 'undefined' ? window.innerWidth : 1200);
  const timelineContainerRef = React.useRef<HTMLDivElement>(null);

  // Window resize listener
  React.useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const timelineData = useMemo(() => {
    // Filter runs with valid timestamps
    const validRuns = runs.filter(run => run.started_at);

    if (validRuns.length === 0) {
      return { lanes: [], minTime: 0, maxTime: 0, segments: [], _intervalHours: 0, _gaps: [], PX_PER_HOUR: 0, totalDuration: 0 };
    }

    // Sort by started_at
    const sortedRuns = [...validRuns].sort((a, b) => {
      const dateA = new Date(a.started_at!).getTime();
      const dateB = new Date(b.started_at!).getTime();
      return dateA - dateB;
    });

    // Calculate actual time range from all runs
    const timestamps = sortedRuns.map(r => new Date(r.started_at!).getTime());
    const finishTimestamps = sortedRuns
      .filter(r => r.finished_at)
      .map(r => new Date(r.finished_at!).getTime());

    const minTime = Math.min(...timestamps);
    const maxTime = finishTimestamps.length > 0
      ? Math.max(...finishTimestamps, ...timestamps)
      : Math.max(...timestamps);

    // Get interval hours and calculate px per hour based on selected interval
    const totalTimeRangeMs = maxTime - minTime;
    const _intervalHours = getIntervalHours(timeRange, totalTimeRangeMs);
    const PX_PER_INTERVAL = 150; // Selected interval = 150px (visual spacing)
    const PX_PER_HOUR = PX_PER_INTERVAL / _intervalHours; // Density depends on interval

    // Detect _gaps (but don't create segments yet - need marker range first)
    const _gaps = detectTimeGaps(sortedRuns, _intervalHours);

    // Calculate total duration (maxTime - minTime)
    const totalDuration = maxTime - minTime;

    // Return data needed for positioning calculation
    return { minTime, maxTime, _intervalHours, _gaps, PX_PER_HOUR, totalDuration };
  }, [runs, timeRange]);

  // Calculate all positioning data upfront for easier debugging
  const positionedRuns = useMemo(() => {
    const { minTime, maxTime, _intervalHours, _gaps, PX_PER_HOUR } = timelineData;

    if (minTime === 0 || maxTime === 0) {
      return { items: [], timeMarkers: [], layout: { labelAboveHeight: 0, centerBarHeight: 0, labelBelowHeight: 0, totalHeight: 0, timelineWidth: 0, containerWidth: 0, timeMarkerContainerWidth: 0 }, segments: [], minTime: 0, maxTime: 0 };
    }

    // Constants (actual rendered sizes)
    const LABEL_HEIGHT = 24; // Actual: text-[10px] + px-1 padding + border = 24px
    const LABEL_STACK_HEIGHT = 16; // Effective height when stacking (overlap by 8px)
    const BAR_HEIGHT = 28;   // h-7 = 28px
    const LABEL_SPACING = 8; // Gap between label and bar
    const LANE_SPACING = 32; // Spacing between lanes

    // Pixel margin on both sides
    const LEFT_MARGIN_PX = 24; // 24px left margin
    const RIGHT_MARGIN_PX = 24; // 24px right margin
    const CARD_PADDING_PX = 16; // CardContent compact variant: p-4 = 16px each side

    const intervalMs = _intervalHours * 60 * 60 * 1000;

    // ===== Quantize time range to interval boundaries =====
    // Philosophy: All segments MUST start/end at interval boundaries (0h, 6h, 12h, 18h)
    // This ensures perfect alignment between segments, time markers, and run bars
    const minDate = new Date(minTime);
    const minDayStart = new Date(Date.UTC(
      minDate.getUTCFullYear(),
      minDate.getUTCMonth(),
      minDate.getUTCDate()
    )).getTime();

    const firstIntervalIndex = Math.floor((minTime - minDayStart) / intervalMs);
    const lastIntervalIndex = Math.ceil((maxTime - minDayStart) / intervalMs);

    const quantizedMinTime = minDayStart + (firstIntervalIndex * intervalMs);
    const quantizedMaxTime = minDayStart + (lastIntervalIndex * intervalMs);
    // ===== End quantization =====

    // Calculate extended time range for markers
    const actualTimeRange = maxTime - minTime;

    // Calculate actual timeline width in pixels (data only, no extension yet)
    const _actualTimeRangePixels = (actualTimeRange / (60 * 60 * 1000)) * PX_PER_HOUR;

    // Get runs with positions first (needed for createSegments)
    const validRuns = runs.filter(run => run.started_at);
    const sortedByTime = [...validRuns].sort((a, b) => {
      const dateA = new Date(a.started_at!).getTime();
      const dateB = new Date(b.started_at!).getTime();
      return dateA - dateB;
    });

    const runsWithPositions = sortedByTime.map(run => {
      const startTime = new Date(run.started_at!).getTime();
      let endTime = run.finished_at
        ? new Date(run.finished_at).getTime()
        : run.status === 'running'
        ? Date.now()
        : startTime + 3600000;

      // Fix bad data: endTime must be >= startTime
      if (endTime < startTime) {
        endTime = startTime + 3600000;
      }

      // Ensure minimum duration for visibility
      if (endTime === startTime) {
        endTime = startTime + 3600000;
      }

      const duration = endTime - startTime;
      const durationStr = duration > 0 ? formatDuration(duration) : '-';

      return {
        ...run,
        startTime,
        endTime,
        duration,
        durationStr,
      };
    });

    // Recalculate maxTime to include calculated endTimes
    const actualMaxTime = Math.max(
      maxTime,
      ...runsWithPositions.map(r => r.endTime)
    );

    // Quantize actualMaxTime to interval boundary
    const actualMaxTimeQuantized = Math.max(quantizedMaxTime, actualMaxTime);
    const lastIntervalForActual = Math.ceil((actualMaxTimeQuantized - minDayStart) / intervalMs);
    const quantizedActualMaxTime = minDayStart + (lastIntervalForActual * intervalMs);

    // CRITICAL: Use SAME segments for all calculations
    // All systems must use identical segment array
    // Use quantized times to ensure segments align with interval boundaries
    const unifiedSegments = createSegments(quantizedMinTime, quantizedActualMaxTime, _gaps, PX_PER_HOUR, _intervalHours);

    interface RunItem {
      run: any;
      laneIdx: number;
      runIdx: number;
      labelAbove: boolean;
    }

    // Lane assignment: check pixel overlap using segments (2px minimum gap)
    // IMPORTANT: Process runs in chronological order (earliest startTime first)
    type RunWithPosition = typeof runsWithPositions[0];
    const lanes: RunWithPosition[][] = [];
    const MIN_VISUAL_GAP_PX = 2;

    // Sort by startTime to ensure chronological processing
    const sortedRuns = [...runsWithPositions].sort((a, b) => a.startTime - b.startTime);

    for (const run of sortedRuns) {
      const runStartPx = timeToPixel(run.startTime, unifiedSegments, _gaps, _intervalHours);
      const runEndPx = timeToPixel(run.endTime, unifiedSegments, _gaps, _intervalHours);

      let assignedLane = -1;

      // Check all existing lanes to find one where this run doesn't overlap
      for (let i = 0; i < lanes.length; i++) {
        const lane = lanes[i];

        // Check if this run overlaps with ANY run in this lane (not just the last one)
        let overlapsWithLane = false;
        for (const existingRun of lane) {
          const existingStartPx = timeToPixel(existingRun.startTime, unifiedSegments, _gaps, _intervalHours);
          const existingEndPx = timeToPixel(existingRun.endTime, unifiedSegments, _gaps, _intervalHours);

          // Check if runs overlap: run starts before existing ends AND run ends after existing starts
          if (runStartPx < existingEndPx + MIN_VISUAL_GAP_PX && runEndPx > existingStartPx - MIN_VISUAL_GAP_PX) {
            overlapsWithLane = true;
            break;
          }
        }

        if (!overlapsWithLane) {
          assignedLane = i;
          break;
        }
      }

      if (assignedLane === -1) {
        lanes.push([run]);
      } else {
        lanes[assignedLane].push(run);
      }
    }

    // Step 1: Collect all runs with their lane info
    const allRuns: RunItem[] = [];
    lanes.forEach((lane, laneIdx) => {
      lane.forEach((run, runIdx) => {
        // Lane 0 (top) → labels above, Lane 1+ (bottom) → labels below
        const labelAbove = laneIdx === 0;
        allRuns.push({ run, laneIdx, runIdx, labelAbove });
      });
    });

    // Step 2: Sort for row assignment priority (based on pixel position, not percent)
    const sortedForLabels = allRuns.map(item => {
      const runCenterPx = timeToPixel((item.run.startTime + item.run.endTime) / 2, unifiedSegments, _gaps, _intervalHours);
      return {
        ...item,
        centerPx: runCenterPx,
      };
    }).sort((a, b) => {
      // Primary: time (latest first = rightmost first)
      return b.run.startTime - a.run.startTime;
    });

    // Step 3: Assign label rows using pixel-based occupancy detection
    const aboveOccupancy = new Map<number, Array<[number, number]>>();
    const belowOccupancy = new Map<number, Array<[number, number]>>();

    const runWithRows = sortedForLabels.map(item => {
      const occupancyMap = item.labelAbove ? aboveOccupancy : belowOccupancy;
      const estimatedLabelWidth = item.run.run_name.length * 7 + 24; // px
      const labelStartPx = timeToPixel(item.run.startTime, unifiedSegments, _gaps, _intervalHours);
      const labelEndPx = labelStartPx + estimatedLabelWidth;

      // Find first available row
      let labelRow = 0;
      for (let row = 0; row < 10; row++) {
        const ranges = occupancyMap.get(row) || [];
        const overlaps = ranges.some(([start, end]) =>
          labelStartPx < end + 2 && labelEndPx > start - 2 // 2px gap
        );

        if (!overlaps) {
          labelRow = row;
          if (!occupancyMap.has(row)) {
            occupancyMap.set(row, []);
          }
          occupancyMap.get(row)!.push([labelStartPx, labelEndPx]);
          break;
        }
      }

      return { ...item, labelRow };
    });

    // Step 4: Calculate layout dimensions
    const maxAboveRows = aboveOccupancy.size;
    const maxBelowRows = belowOccupancy.size;

    // Calculate heights: stacked labels + one gap to bars + container margin
    // Adjust by -9.5px to make the actual span top align at 10px instead of 19.5px
    const SPAN_TOP_ADJUSTMENT = 9.5;
    const labelAboveHeight = maxAboveRows > 0
      ? CONTAINER_MARGIN_TOP + LABEL_HEIGHT + ((maxAboveRows - 1) * LABEL_STACK_HEIGHT) + LABEL_SPACING - SPAN_TOP_ADJUSTMENT
      : CONTAINER_MARGIN_TOP + LABEL_SPACING - SPAN_TOP_ADJUSTMENT;
    const labelBelowHeight = maxBelowRows > 0
      ? LABEL_SPACING + LABEL_HEIGHT + ((maxBelowRows - 1) * LABEL_STACK_HEIGHT) + CONTAINER_MARGIN_BOTTOM
      : LABEL_SPACING;
    const centerBarHeight = lanes.length > 0
      ? (lanes.length - 1) * LANE_SPACING + BAR_HEIGHT
      : BAR_HEIGHT;
    const totalHeight = labelAboveHeight + centerBarHeight + labelBelowHeight;

    // Step 5: Calculate absolute pixel positions using timeToPixel
    const debugData: any = {
      timeline: {
        minTime: new Date(minTime).toISOString(),
        actualMaxTime: new Date(actualMaxTime).toISOString(),
        PX_PER_HOUR,
        _intervalHours,
      },
      segments: unifiedSegments.map((seg, i) => ({
        idx: i,
        type: seg.type,
        startTime: new Date(seg.startTime).toISOString(),
        endTime: new Date(seg.endTime).toISOString(),
        offsetPx: seg.offsetPx.toFixed(2),
        displayWidthPx: seg.displayWidthPx.toFixed(2),
        timeRangeHours: ((seg.endTime - seg.startTime) / (1000 * 60 * 60)).toFixed(2),
      })),
      runs: [],
    };

    const items = runWithRows.map(item => {
      const { run, laneIdx, runIdx, labelAbove, labelRow } = item;

      // Bar position (absolute from container top)
      const barTop = labelAboveHeight + (laneIdx * LANE_SPACING);

      // Convert time to absolute pixels (segment-aware) - use unifiedSegments for calculation
      const startDebug: any[] = [];
      const endDebug: any[] = [];
      const barStartPx = timeToPixel(run.startTime, unifiedSegments, _gaps, _intervalHours, startDebug);
      const barEndPx = timeToPixel(run.endTime, unifiedSegments, _gaps, _intervalHours, endDebug);
      const barLeft = LEFT_MARGIN_PX + barStartPx; // Align with time axis
      const barWidth = Math.max(barEndPx - barStartPx, 4); // Minimum 4px for visibility

      debugData.runs.push({
        name: run.run_name,
        startTime: new Date(run.startTime).toISOString(),
        endTime: new Date(run.endTime).toISOString(),
        barStartPx: barStartPx.toFixed(2),
        barEndPx: barEndPx.toFixed(2),
        barLeft: barLeft.toFixed(2),
        barWidth: barWidth.toFixed(2),
        startSegment: startDebug[0],
        endSegment: endDebug[0],
      });

      // Label position with stacking overlap
      const labelTop = labelAbove
        ? barTop - LABEL_SPACING - LABEL_HEIGHT - (labelRow * LABEL_STACK_HEIGHT)
        : barTop + BAR_HEIGHT + LABEL_SPACING + (labelRow * LABEL_STACK_HEIGHT);

      // Connector: Always connects label CENTER to bar
      const labelCenter = labelTop + (LABEL_HEIGHT / 2);

      const connectorTop = labelAbove
        ? labelCenter              // Above: Start from label center
        : barTop + BAR_HEIGHT;     // Below: Start from bar bottom

      const connectorBottom = labelAbove
        ? barTop                   // Above: End at bar top
        : labelCenter;             // Below: End at label center

      const connectorHeight = connectorBottom - connectorTop;

      return {
        run,
        laneIdx,
        runIdx,
        labelAbove,
        labelRow,
        // Absolute pixel positions
        barTop,
        barLeft,
        barWidth,
        labelTop,
        labelLeft: barLeft, // Label at same position as bar
        connectorTop,
        connectorHeight,
      };
    });

    // ============================================================
    // COORDINATE NORMALIZATION
    // ============================================================
    // Calculate leftShift to make first run start at LEFT_MARGIN_PX (24px)
    const minBarLeft = items.length > 0 ? Math.min(...items.map(i => i.barLeft)) : LEFT_MARGIN_PX;
    const leftShift = minBarLeft - LEFT_MARGIN_PX;

    // Adjust all coordinates by leftShift
    items.forEach(item => {
      item.barLeft -= leftShift;
      item.labelLeft -= leftShift;
    });

    // DON'T adjust segments - keep them in original coordinate space
    // TimeMarker will handle leftShift during rendering
    // unifiedSegments.forEach(seg => {
    //   seg.offsetPx -= leftShift;
    // });

    // ============================================================
    // WIDTH CALCULATION STRATEGY (8 levels)
    // ============================================================

    // 1. 회색 컨테이너 영역 (최종 스크롤 가능 영역) - 나중에 결정
    // 2. 실제 타임라인 영역 (data only, no labels, no margin)
    const dataTimelineWidth = timeToPixel(maxTime, unifiedSegments, _gaps, _intervalHours); // 실제 데이터 타임라인 폭

    // 3. run 라벨 영역 - Blue + 라벨까지 포함 (렌더링 후 DOM에서 실제 측정)
    // Green = max(Blue 끝, 가장 먼 라벨 끝)
    // Note: leftShift 적용 후 dataTimelineWidth도 조정되었으므로, 이제 24px에서 시작
    const dataTimelineEnd = LEFT_MARGIN_PX + (dataTimelineWidth - leftShift); // Blue의 절대 끝 위치
    const dataWithLabelsWidth = measuredLabelWidth || (dataTimelineEnd + 200); // 초기값: 대략적인 추정

    // Green의 상단/하단 계산 (라벨 포함)
    // 막대 영역 (Blue)
    const blueTop = labelAboveHeight;
    const blueBottom = labelAboveHeight + centerBarHeight;

    // 라벨들을 포함한 Green 영역 계산
    let greenTop = blueTop;
    let greenBottom = blueBottom;

    items.forEach(item => {
      // 라벨의 상단과 하단
      const labelBottom = item.labelTop + LABEL_HEIGHT;
      greenTop = Math.min(greenTop, item.labelTop);
      greenBottom = Math.max(greenBottom, labelBottom);
    });

    const greenHeight = greenBottom - greenTop;

    // 1. Red = Green + 마진
    const LABEL_RIGHT_MARGIN = 10;
    const dataWithLabelsAndMarginWidth = dataWithLabelsWidth + LABEL_RIGHT_MARGIN;

    // 4. 2에 해당하는 시간 마커 (실제 데이터만)
    // → segments 사용 (이미 생성됨)

    // 5. 3을 위한 확장된 시간 마커 (라벨 + 마진)
    // 라벨 + 마진 영역을 채우기 위한 시간 범위 계산
    const timelineWidthFor3 = dataWithLabelsAndMarginWidth - LEFT_MARGIN_PX - RIGHT_MARGIN_PX;
    const timeRangeFor3 = (timelineWidthFor3 / PX_PER_HOUR) * 60 * 60 * 1000;
    const extendedMaxTimeFor3 = minTime + timeRangeFor3;
    const segmentsFor3 = createSegments(minTime, extendedMaxTimeFor3, _gaps, PX_PER_HOUR, _intervalHours);
    const lastSegmentFor3 = segmentsFor3[segmentsFor3.length - 1];
    const actualWidthFor3 = LEFT_MARGIN_PX + (lastSegmentFor3.offsetPx + lastSegmentFor3.displayWidthPx) + RIGHT_MARGIN_PX;

    // 6. 카드 너비 영역 (windowWidth를 사용하여 리사이즈 시 재계산)
    // CardContent p-4 (32px total for left+right) + 외부 마진/패딩 = ~140px
    // But need to account for actual scroll container width
    const cardContentWidth = Math.min(windowWidth - 140, 1400);

    // ALWAYS use measured scroll width if available, otherwise use calculated
    // This ensures consistent behavior regardless of interval changes
    const _actualCardWidth = measuredScrollWidth || cardContentWidth;

    // 1번 = 회색 컨테이너 영역
    // Use only data width for container, not card width
    // This prevents container from growing beyond actual data
    const _finalContainerWidth = actualWidthFor3;

    // 7번 = Red(회색 컨테이너) 기준 타임라인 너비
    // Red = just labels, no extra margin
    const redContainerWidth = (measuredLabelWidth || dataWithLabelsWidth);
    const finalRedTimelineWidth = redContainerWidth - LEFT_MARGIN_PX - RIGHT_MARGIN_PX;

    // Red와 Orange 비교하여 케이스 결정
    const redWidth = (measuredLabelWidth || dataWithLabelsWidth);
    const orangeWidth = cardContentWidth;

    let _finalTimelineWidth: number;
    let markerContainerWidth: number;
    let maxTimelinePixel: number;

    if (redWidth > orangeWidth) {
      // Red case: extend to cover all data (스크롤됨)
      _finalTimelineWidth = finalRedTimelineWidth;
      // Red Container: 실측 기준 56px 더하기
      markerContainerWidth = redWidth - (CARD_PADDING_PX * 2) + 56;
      // Segment는 markerContainer를 채우도록 생성
      // 렌더링 시 leftShift를 빼므로, 생성 시에는 더해줘야 함
      maxTimelinePixel = markerContainerWidth - LEFT_MARGIN_PX + leftShift;
    } else {
      // Orange case: fill card width (스크롤 없음)
      // cardContentWidth에서 좌우 padding 제거
      const scrollableWidth = orangeWidth - (CARD_PADDING_PX * 2);
      _finalTimelineWidth = scrollableWidth - LEFT_MARGIN_PX - RIGHT_MARGIN_PX;
      // Orange Container: 실측 기준 72px 빼기 (24 + 48)
      markerContainerWidth = orangeWidth - (CARD_PADDING_PX * 2) - 72;
      // Segment는 markerContainer를 채우도록 생성
      // 렌더링 시 leftShift를 빼므로, 생성 시에는 더해줘야 함
      maxTimelinePixel = markerContainerWidth - LEFT_MARGIN_PX + leftShift;
    }

    // CRITICAL: Use SAME unifiedSegments, just extend if needed
    const extendedSegments = [...unifiedSegments];

    // 마지막 segment 확인
    let lastSegment = extendedSegments[extendedSegments.length - 1];
    let currentEndPixel = lastSegment.offsetPx + lastSegment.displayWidthPx;

    // maxTimelinePixel까지 segment 추가 (TimeMarker Container를 채우도록)
    if (currentEndPixel < maxTimelinePixel) {
      const remainingPixels = maxTimelinePixel - currentEndPixel;
      const intervalMs = _intervalHours * 60 * 60 * 1000;
      const fullSegmentsToAdd = Math.floor(remainingPixels / PX_PER_INTERVAL);
      const lastSegmentWidth = remainingPixels % PX_PER_INTERVAL;

      // Full 150px segments 추가
      for (let i = 0; i < fullSegmentsToAdd; i++) {
        const segmentStart = lastSegment.endTime;
        const segmentEnd = segmentStart + intervalMs;
        const segmentOffset = currentEndPixel;

        extendedSegments.push({
          type: 'normal',
          startTime: segmentStart,
          endTime: segmentEnd,
          offsetPx: segmentOffset,
          displayWidthPx: PX_PER_INTERVAL,
        });

        currentEndPixel += PX_PER_INTERVAL;
        lastSegment = extendedSegments[extendedSegments.length - 1];
      }

      // 마지막 partial segment 추가 (150px 미만)
      if (lastSegmentWidth > 0) {
        const segmentStart = lastSegment.endTime;
        const timeForLastSegment = (lastSegmentWidth / PX_PER_INTERVAL) * intervalMs;
        const segmentEnd = segmentStart + timeForLastSegment;

        extendedSegments.push({
          type: 'normal',
          startTime: segmentStart,
          endTime: segmentEnd,
          offsetPx: currentEndPixel,
          displayWidthPx: lastSegmentWidth,
        });

        lastSegment = extendedSegments[extendedSegments.length - 1];
      }
    }

    // 1번, 7번의 최종 너비 (동일)
    const _finalScrollableWidth = LEFT_MARGIN_PX + (lastSegment.offsetPx + lastSegment.displayWidthPx) + RIGHT_MARGIN_PX;

    // Find first marker time aligned to interval boundaries
    // Start from minTime and round UP to the nearest interval
    const _firstMarkerTime = Math.ceil(minTime / intervalMs) * intervalMs;

    // Generate markers: 픽셀 기반으로 생성하여 compressed 영역 고려
    const timeMarkers: Array<{ time: number; pixelPosition: number }> = [];

    // Helper: 픽셀 위치 → 시간 역변환
    const _pixelToTime = (pixelPos: number): number => {
      for (const seg of extendedSegments) {
        const segmentStart = seg.offsetPx;
        const segmentEnd = seg.offsetPx + seg.displayWidthPx;

        if (pixelPos >= segmentStart && pixelPos <= segmentEnd) {
          // 이 segment 내에 위치
          const pixelIntoSegment = pixelPos - segmentStart;
          const progressInSegment = seg.displayWidthPx > 0 ? pixelIntoSegment / seg.displayWidthPx : 0;
          return seg.startTime + (seg.endTime - seg.startTime) * progressInSegment;
        }
      }

      // Fallback: 마지막 segment 끝 시간
      const lastSeg = extendedSegments[extendedSegments.length - 1];
      return lastSeg ? lastSeg.endTime : minTime;
    };

    // Compressed zones 정의: [25px 물결] + [100px compressed] + [25px 물결] = 150px (시간 마커 생성 X)
    const WAVE_WIDTH = 25;
    const _compressedZones = extendedSegments
      .map((seg, idx) => {
        if (seg.type === 'compressed') {
          // 물결 25px 앞뒤 포함
          const prevWaveSeg = extendedSegments[idx - 1];
          const nextWaveSeg = extendedSegments[idx + 1];

          const zoneStart = prevWaveSeg && prevWaveSeg.type === 'normal' && prevWaveSeg.displayWidthPx === WAVE_WIDTH
            ? prevWaveSeg.offsetPx
            : seg.offsetPx;

          const zoneEnd = nextWaveSeg && nextWaveSeg.type === 'normal' && nextWaveSeg.displayWidthPx === WAVE_WIDTH
            ? nextWaveSeg.offsetPx + nextWaveSeg.displayWidthPx
            : seg.offsetPx + seg.displayWidthPx;

          return {
            startPx: zoneStart,
            endPx: zoneEnd,
            startTime: prevWaveSeg && prevWaveSeg.displayWidthPx === WAVE_WIDTH ? prevWaveSeg.startTime : seg.startTime,
            endTime: nextWaveSeg && nextWaveSeg.displayWidthPx === WAVE_WIDTH ? nextWaveSeg.endTime : seg.endTime,
          };
        }
        return null;
      })
      .filter(Boolean) as Array<{ startPx: number; endPx: number; startTime: number; endTime: number }>;

    /**
     * PHILOSOPHY: Time Marker 생성 - Segment 경계 기반
     *
     * 각 Segment의 시작 시간에 마커 생성
     * Compressed segment는 시작/끝에만 마커 생성
     */
    // maxTimelinePixel은 위에서 이미 계산됨 (Red/Orange 케이스별로)

    // Generate markers: simple approach - every segment boundary gets a marker
    const _MARGIN_INTERVALS = 1;

    // Add markers for all segment boundaries
    extendedSegments.forEach((seg, segIdx) => {
      // Add start marker
      if (seg.offsetPx <= maxTimelinePixel) {
        timeMarkers.push({
          time: seg.startTime,
          pixelPosition: seg.offsetPx,
        });
      }

      // For normal segments, add internal interval markers
      if (seg.type === 'normal') {
        const segDuration = seg.endTime - seg.startTime;
        const numInternalIntervals = Math.floor(segDuration / intervalMs);

        for (let i = 1; i < numInternalIntervals; i++) {
          const internalTime = seg.startTime + (i * intervalMs);
          const progress = (internalTime - seg.startTime) / segDuration;
          const internalPx = seg.offsetPx + (progress * seg.displayWidthPx);

          if (internalPx <= maxTimelinePixel) {
            timeMarkers.push({
              time: internalTime,
              pixelPosition: internalPx,
            });
          }
        }
      }

      // Add end marker for compressed segments and last segment
      const isCompressed = seg.type === 'compressed';
      const isLast = segIdx === extendedSegments.length - 1;
      const endPx = seg.offsetPx + seg.displayWidthPx;

      if ((isCompressed || isLast) && endPx <= maxTimelinePixel) {
        timeMarkers.push({
          time: seg.endTime,
          pixelPosition: endPx,
        });
      }
    });

    // Sort markers
    timeMarkers.sort((a, b) => a.pixelPosition - b.pixelPosition);

    // Remove duplicates
    const uniqueMarkers = timeMarkers.filter((marker, index, self) =>
      index === self.findIndex(m => Math.abs(m.pixelPosition - marker.pixelPosition) < 1)
    );

    // Filter out compressed zone boundary markers (matching _yellowSegments logic)
    // Remove markers too close to neighbors (< 100px)
    const filteredMarkers = uniqueMarkers.filter((marker, idx) => {
      const prevMarker = uniqueMarkers[idx - 1];
      const nextMarker = uniqueMarkers[idx + 1];

      // Remove markers too close to previous marker (< 100px)
      if (prevMarker) {
        const gapFromPrev = marker.pixelPosition - prevMarker.pixelPosition;
        if (gapFromPrev > 0 && gapFromPrev < 100) {
          return false; // Zone boundary marker - remove
        }
      }

      // Remove markers too close to next marker (< 100px)
      if (nextMarker) {
        const gapToNext = nextMarker.pixelPosition - marker.pixelPosition;
        if (gapToNext > 0 && gapToNext < 100) {
          return false; // Zone boundary marker - remove
        }
      }

      return true;
    });

    // Add time markers to debug data (use filteredMarkers, not uniqueMarkers)
    debugData.timeMarkers = filteredMarkers.map(m => ({
      time: new Date(m.time).toISOString(),
      pixelPosition: m.pixelPosition.toFixed(2),
    }));

    // Removed verbose debug logs - use browser devtools if needed

    return {
      items,
      timeMarkers: filteredMarkers,
      layout: {
        labelAboveHeight,
        centerBarHeight,
        labelBelowHeight,
        totalHeight,
        timelineWidth: dataTimelineWidth, // 2번: Pure timeline width (data only)
        containerWidth: dataWithLabelsWidth, // 3번: Container width (includes label overflow + margin)
        greenTop, // 3번 Green의 상단 위치
        greenHeight, // 3번 Green의 높이
        actualWidthFor3, // 5번: 3번을 채우기 위한 segments의 실제 너비
        cardContentWidth, // 6번: Card content width
        timeMarkerContainerWidth: markerContainerWidth, // 7번: Marker container (final)
        leftShift, // 자정 정렬로 인한 왼쪽 빈 공간
      },
      segments: extendedSegments, // Pass extended segments for compressed region rendering
      minTime,
      maxTime,
    };
  }, [timelineData, timeRange, runs, measuredLabelWidth, windowWidth, measuredScrollWidth]);

  // Calculate Yellow segments
  const _yellowSegments = React.useMemo(() => {
    // TimeMarker Container 전체 영역을 150px씩 쪼개서 채우기
    const segments: Array<{
      idx: number;
      markerLeft: number;
      segmentWidth: number;
      segmentRight: number;
      isLast: boolean;
    }> = [];

    // TimeMarker Container의 전체 너비
    const totalWidth = positionedRuns.layout.timeMarkerContainerWidth;
    const startLeft = 24; // LEFT_MARGIN_PX

    let currentLeft = startLeft;
    let remainingWidth = totalWidth;
    let segmentIdx = 0;

    while (remainingWidth > 0) {
      const segmentWidth = Math.min(150, remainingWidth);
      const segmentRight = currentLeft + segmentWidth;
      const isLast = remainingWidth <= 150;

      segments.push({
        idx: segmentIdx,
        markerLeft: currentLeft,
        segmentWidth,
        segmentRight,
        isLast,
      });

      currentLeft += segmentWidth;
      remainingWidth -= segmentWidth;
      segmentIdx++;
    }

    return segments;
  }, [positionedRuns.layout.timeMarkerContainerWidth]);

  // Measure actual label widths and vertical bounds after rendering
  React.useLayoutEffect(() => {
    if (!timelineContainerRef.current) return;

    // Blue의 끝 위치 (CONTAINER_MARGIN_LEFT + dataTimelineWidth)
    const blueEnd = CONTAINER_MARGIN_LEFT + positionedRuns.layout.timelineWidth;
    const containerRect = timelineContainerRef.current.getBoundingClientRect();

    // 막대 영역 (Blue) - 실제 시각적 막대 요소 측정
    const barVisuals = timelineContainerRef.current.querySelectorAll('[data-bar-visual]');
    let barMinTop = Infinity;
    let barMaxBottom = -Infinity;

    barVisuals.forEach(bar => {
      const rect = bar.getBoundingClientRect();
      const topPosition = rect.top - containerRect.top;
      const bottomPosition = rect.bottom - containerRect.top;
      barMinTop = Math.min(barMinTop, topPosition);
      barMaxBottom = Math.max(barMaxBottom, bottomPosition);
    });

    // 모든 라벨의 span 요소를 찾아서 가장 오른쪽 끝 위치와 상단/하단 위치를 측정
    const labelSpans = timelineContainerRef.current.querySelectorAll('[data-label-span]');

    let maxLabelRight = blueEnd; // 최소값은 Blue 끝
    let labelMinTop = Infinity;
    let labelMaxBottom = -Infinity;

    if (labelSpans.length > 0) {
      labelSpans.forEach(span => {
        const rect = span.getBoundingClientRect();

        // Horizontal measurement
        const rightPosition = rect.right - containerRect.left;
        maxLabelRight = Math.max(maxLabelRight, rightPosition);

        // Vertical measurement
        const topPosition = rect.top - containerRect.top;
        const bottomPosition = rect.bottom - containerRect.top;
        labelMinTop = Math.min(labelMinTop, topPosition);
        labelMaxBottom = Math.max(labelMaxBottom, bottomPosition);
      });
    }

    // Green = min(막대 상단, 라벨 상단) ~ max(막대 하단, 라벨 하단)
    let greenTop = Infinity;
    let greenBottom = -Infinity;

    if (barMinTop !== Infinity) greenTop = Math.min(greenTop, barMinTop);
    if (labelMinTop !== Infinity) greenTop = Math.min(greenTop, labelMinTop);
    if (barMaxBottom !== -Infinity) greenBottom = Math.max(greenBottom, barMaxBottom);
    if (labelMaxBottom !== -Infinity) greenBottom = Math.max(greenBottom, labelMaxBottom);

    // Update horizontal measurement
    // Green = max(Blue 끝, 가장 먼 라벨 끝)
    if (maxLabelRight > 0 && Math.abs(maxLabelRight - (measuredLabelWidth || 0)) > 1) {
      setMeasuredLabelWidth(maxLabelRight);
    }

    // Update vertical measurement
    if (greenTop !== Infinity && greenBottom !== -Infinity) {
      const newBounds = { top: greenTop, height: greenBottom - greenTop };
      if (!measuredGreenBounds ||
          Math.abs(newBounds.top - measuredGreenBounds.top) > 1 ||
          Math.abs(newBounds.height - measuredGreenBounds.height) > 1) {
        setMeasuredGreenBounds(newBounds);
      }
    }

    // Don't measure - rely on cardContentWidth calculation
    // Measuring causes issues because content affects container size
    // cardContentWidth is already accurately calculated from windowWidth
  }, [positionedRuns.items, positionedRuns.layout.timelineWidth, measuredLabelWidth, measuredGreenBounds, measuredScrollWidth]);

  if (positionedRuns.items.length === 0) {
    return (
      <Card variant="compact" className={className}>
        <CardHeader variant="compact">
          <CardTitle size="sm" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Run Timeline
          </CardTitle>
        </CardHeader>
        <CardContent variant="compact">
          <div className="text-center text-muted-foreground py-8 text-sm">
            No timeline data available
          </div>
        </CardContent>
      </Card>
    );
  }

  const totalRuns = positionedRuns.items.length;

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleString('ko-KR', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
      timeZone: 'UTC',
    });
  };

  const getStatusColor = (status: string) => {
    return STATUS_COLORS[status as keyof typeof STATUS_COLORS] || STATUS_COLORS.default;
  };

  return (
    <TooltipProvider>
      <Card variant="compact" className={className}>
      <CardHeader variant="compact">
        <div className="flex items-center justify-between gap-4">
          <CardTitle size="sm" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Run Timeline
          </CardTitle>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <div className="flex items-center gap-1.5">
              <Calendar className="h-3 w-3" />
              <span className="hidden sm:inline">{formatDate(timelineData.minTime)}</span>
              <span className="hidden sm:inline">→</span>
              <span className="hidden sm:inline">{formatDate(timelineData.maxTime)}</span>
            </div>
            <span className="hidden sm:inline">•</span>
            <span className="font-medium">{totalRuns} runs</span>
            <span className="hidden sm:inline">•</span>
            <span className="text-muted-foreground">{positionedRuns.items.filter((item, i, arr) => arr.findIndex(x => x.laneIdx === item.laneIdx) === i).length} lanes</span>
            <span className="hidden sm:inline">•</span>
            {/* Time range selector */}
            <Select value={timeRange} onValueChange={(value) => setTimeRange(value as TimeRange)}>
              <SelectTrigger className="h-8 w-[100px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(Object.keys(TIME_RANGE_LABELS) as TimeRange[]).map((range) => (
                  <SelectItem key={range} value={range} className="text-xs">
                    {TIME_RANGE_LABELS[range]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent ref={cardContentRef} variant="compact" className="pb-3 relative">
        {/* Horizontal scrollable container - scroll for both markers and gray container */}
        <div ref={scrollContainerRef} className="overflow-x-auto overflow-y-hidden scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
          {/* Time markers - at the top, same width as extended segments */}
          {/* 4-7번: 시간 마커들을 구분하여 표시 */}
          <div className="relative pointer-events-none mb-2" style={{
            width: `${positionedRuns.layout.timeMarkerContainerWidth}px`,
            height: '56px',
          }}>
            <div className="relative mb-2">
              <div className="flex items-center h-7 border-b border-border/40">
                {(() => {
                  // Use pre-calculated time markers from positionedRuns
                  const uniqueMarkers = positionedRuns.timeMarkers;

                  // 라벨 간격 필터링
                  return uniqueMarkers
                    .filter((marker, i, arr) => {
                      if (i === 0) return true;
                      const prevMarker = arr[i - 1];

                      // Dynamic minimum spacing based on interval
                      const _intervalHours = getIntervalHours(timeRange);
                      let minLabelSpacing = 80;
                      if (_intervalHours <= 1) {
                        minLabelSpacing = 120;
                      } else if (_intervalHours <= 6) {
                        minLabelSpacing = 100;
                      }

                      const spacing = marker.pixelPosition - prevMarker.pixelPosition;
                      return spacing >= minLabelSpacing;
                    })
                    .map((marker, i, filteredArr) => {
                  // Use pre-calculated pixel position from segment-aware timeToPixel
                  // Subtract leftShift to align with normalized coordinates
                  const markerLeft = 24 + marker.pixelPosition - positionedRuns.layout.leftShift;
                  const isFirst = i === 0;
                  const isLast = i === filteredArr.length - 1;

                  // Format time based on selected interval
                  const _intervalHours = getIntervalHours(timeRange);
                  let timeFormat: Intl.DateTimeFormatOptions;

                  if (_intervalHours <= 6) {
                    // Hourly intervals (1h, 6h): show date + time (24h format)
                    timeFormat = {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                      hour12: false,
                    };
                  } else if (_intervalHours < 24) {
                    // 12h interval: show date + hour (24h format)
                    timeFormat = {
                      month: 'short',
                      day: 'numeric',
                      hour: 'numeric',
                      hour12: false,
                    };
                  } else {
                    // Daily/Weekly+ intervals (24h+): show date only
                    timeFormat = {
                      month: 'short',
                      day: 'numeric',
                    };
                  }

                  // Determine label alignment based on position
                  let labelTransform = '-translate-x-1/2'; // Center (default)
                  if (isFirst) {
                    labelTransform = ''; // Left align
                  } else if (isLast) {
                    labelTransform = '-translate-x-full'; // Right align
                  }

                  const markerDate = new Date(marker.time);
                  const formattedString = markerDate.toLocaleString('ko-KR', {
                    ...timeFormat,
                    timeZone: 'UTC',
                  });

                  return (
                    <div
                      key={`${marker.time}-${i}`}
                      className="absolute top-0 h-full flex flex-col items-center"
                      style={{ left: `${markerLeft}px` }}
                    >
                      <div className="h-full w-px bg-border/30" />
                      <span className={`absolute -bottom-6 text-[11px] whitespace-nowrap transform ${labelTransform} text-muted-foreground`}>
                        {formattedString}
                      </span>
                    </div>
                  );
                })})()}

                {/* Start/End position markers (thick lines) */}
                {positionedRuns.items.length > 0 && (() => {
                  // 모든 run bar의 실제 픽셀 위치에서 최소/최대 찾기
                  const minBarLeft = Math.min(...positionedRuns.items.map(r => r.barLeft));
                  const maxBarRight = Math.max(...positionedRuns.items.map(r => r.barLeft + r.barWidth));

                  return (
                    <>
                      {/* Start marker - 가장 왼쪽 bar 시작 */}
                      <div
                        className="absolute top-0 h-full flex flex-col items-center z-10"
                        style={{ left: `${minBarLeft}px` }}
                      >
                        <div className="h-full w-[2px] bg-border" />
                      </div>
                      {/* End marker - 가장 오른쪽 bar 끝 */}
                      <div
                        className="absolute top-0 h-full flex flex-col items-center z-10"
                        style={{ left: `${maxBarRight}px` }}
                      >
                        <div className="h-full w-[2px] bg-border" />
                      </div>
                    </>
                  );
                })()}

                {/* Now marker (red vertical line) - only show if within rendered segment range */}
                {(() => {
                  const nowTime = Date.now();

                  // 렌더링된 segment 범위 확인
                  const segments = positionedRuns.segments;
                  if (!segments || segments.length === 0) return null;

                  const firstSegment = segments[0];
                  const lastSegment = segments[segments.length - 1];

                  if (!firstSegment || !lastSegment) return null;

                  const segmentMinTime = firstSegment.startTime;
                  const segmentMaxTime = lastSegment.endTime;

                  // 유효한 시간 값인지 확인
                  if (!segmentMinTime || !segmentMaxTime || isNaN(segmentMinTime) || isNaN(segmentMaxTime)) {
                    return null;
                  }

                  // 렌더링된 segment 범위 내에 있는지 확인
                  if (nowTime < segmentMinTime || nowTime > segmentMaxTime) {
                    return null;
                  }

                  const nowPosition = timeToPixel(nowTime, segments, timelineData._gaps, timelineData._intervalHours);
                  const nowLeft = 24 + nowPosition;

                  return (
                    <div
                      className="absolute top-0 h-full flex flex-col items-center pointer-events-none z-10"
                      style={{ left: `${nowLeft}px` }}
                    >
                      <div className="h-full w-[2px] bg-red-500" />
                      <span className="absolute -bottom-6 text-[11px] whitespace-nowrap transform -translate-x-1/2 text-red-500 font-semibold">
                        Now
                      </span>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>

          {/* Gray background container - ONLY bars and labels, extends to marker container width */}
          {/* 1번: 회색 컨테이너 (최종 스크롤 가능 영역) - Red = Green + 좌우 마진 */}
          <div ref={timelineContainerRef} className="relative bg-muted/20 rounded-md" style={{
            height: `${((measuredGreenBounds?.top ?? positionedRuns.layout.greenTop) + (measuredGreenBounds?.height ?? positionedRuns.layout.greenHeight)) + CONTAINER_MARGIN_BOTTOM}px`,
            width: `${(measuredLabelWidth || positionedRuns.layout.containerWidth) + CONTAINER_MARGIN_RIGHT}px`,
          }}>

              {/* Compressed segments (zigzag breaks) */}
              {positionedRuns.segments
                .filter(seg => seg.type === 'compressed')
                .map((seg, idx) => {
                  const segLeft = 24 + seg.offsetPx;
                  const segWidth = seg.displayWidthPx;
                  const segHeight = positionedRuns.layout.totalHeight + CONTAINER_MARGIN_BOTTOM;
                  const gapHours = Math.round((seg.endTime - seg.startTime) / (1000 * 60 * 60));

                  // Format time duration label
                  const formatDuration = (hours: number) => {
                    if (hours < 24) return `${hours}h`;
                    const days = Math.floor(hours / 24);
                    const remainingHours = hours % 24;
                    if (remainingHours === 0) return `${days}d`;
                    return `${days}d ${remainingHours}h`;
                  };

                  // Vertical wave pattern for left and right boundaries using cosine function
                  // 1.5 cycles: starts at cos(0)=1, ends at cos(3π)=-1
                  const createWavePath = (xPosition: number) => {
                    const amplitude = 12; // Wave width (horizontal amplitude)
                    const numCycles = 1.5; // 1.5 complete wave cycles
                    const numPoints = 60; // Number of points to sample for smooth curve

                    if (segHeight < 20) return `M ${xPosition},0 L ${xPosition},${segHeight}`;

                    let path = '';

                    // Generate smooth cosine wave: 1.5 cycles from 0 to 3π
                    for (let i = 0; i <= numPoints; i++) {
                      const t = i / numPoints; // Progress from 0 to 1
                      const y = t * segHeight; // Y position from 0 to segHeight
                      const angle = t * numCycles * 2 * Math.PI; // Angle: 0 to 3π (1.5 cycles)
                      const x = xPosition + amplitude * Math.cos(angle); // X offset using cosine

                      if (i === 0) {
                        path += `M ${x},${y}`; // Start at cos(0) = 1 (xPosition + amplitude)
                      } else {
                        path += ` L ${x},${y}`;
                      }
                    }

                    return path;
                  };

                  const leftWavePath = createWavePath(0);
                  const rightWavePath = createWavePath(segWidth);

                  return (
                    <div
                      key={`compressed-${idx}`}
                      className="absolute pointer-events-none"
                      style={{
                        left: `${segLeft}px`,
                        top: 0,
                        width: `${segWidth}px`,
                        height: `${segHeight}px`,
                      }}
                    >
                      {/* Darker background to indicate omitted region */}
                      <div className="absolute inset-0 bg-muted/40" />

                      {/* Wave patterns on left and right boundaries */}
                      <svg
                        width={segWidth}
                        height={segHeight}
                        className="absolute inset-0"
                        style={{ overflow: 'visible' }}
                      >
                        {/* Clip path to constrain only vertical overflow */}
                        <defs>
                          <clipPath id={`clip-vertical-${idx}`}>
                            <rect x="-20" y="0" width={segWidth + 40} height={segHeight} />
                          </clipPath>
                        </defs>
                        {/* Left boundary wave */}
                        <path
                          d={leftWavePath}
                          stroke="currentColor"
                          strokeWidth="2"
                          fill="none"
                          className="text-border"
                          clipPath={`url(#clip-vertical-${idx})`}
                        />
                        {/* Right boundary wave */}
                        <path
                          d={rightWavePath}
                          stroke="currentColor"
                          strokeWidth="2"
                          fill="none"
                          className="text-border"
                          clipPath={`url(#clip-vertical-${idx})`}
                        />
                      </svg>

                      {/* Gap duration label with omission indicator */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-xs font-medium text-muted-foreground bg-background/95 px-2 py-1 rounded border border-border shadow-sm">
                          ⋯ {formatDuration(gapHours)} ⋯
                        </span>
                      </div>
                    </div>
                  );
                })}


              {/* Timeline bars */}
              {positionedRuns.items.map((item, idx) => {
                const { run, barTop, barLeft, barWidth } = item;
                const colors = getStatusColor(run.status);

                return (
                  <Tooltip key={`bar-${idx}`} delayDuration={0}>
                    <TooltipTrigger asChild>
                      <div
                        className="group absolute cursor-pointer"
                        style={{
                          left: `${barLeft}px`,
                          width: `${barWidth}px`,
                          top: `${barTop}px`,
                        }}
                        onClick={() => onRunClick?.(run.run_name)}
                        data-run-bar
                        data-run-name={run.run_name}
                        data-lane={item.laneIdx}
                      >
                        {/* Run bar */}
                        <div className={`h-7 rounded-md border ${colors.border} ${colors.bg} transition-colors duration-200 hover:brightness-110 shadow-sm relative`} data-bar-visual>
                          {/* Running animation */}
                          {run.status === 'running' && (
                            <div className="absolute inset-0 rounded-md overflow-hidden">
                              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent animate-shimmer" />
                            </div>
                          )}
                          {/* Status dot - only show if bar is wide enough */}
                          {barWidth >= 24 && (
                            <div className={`absolute left-1.5 top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full ${colors.dot} shadow-sm`} />
                          )}
                        </div>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="text-xs">
                      <div className="space-y-1">
                        <div className="font-semibold">{run.run_name}</div>
                        <div className="text-muted-foreground">
                          {formatDate(run.startTime)} → {formatDate(run.endTime)}
                        </div>
                        <div className="text-muted-foreground">Duration: {run.durationStr}</div>
                        <div className="text-muted-foreground">
                          Status: <span className={colors.text}>{run.status}</span>
                        </div>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                );
              })}

              {/* Labels and connectors */}
              {positionedRuns.items.map((item, idx) => {
                const { run, labelTop, labelLeft, labelAbove, labelRow, connectorTop, connectorHeight } = item;

                return (
                  <React.Fragment key={`label-${idx}`}>
                    {/* Connector line - connects label to bar */}
                    {connectorHeight > 0 && (
                      <div
                        className="absolute w-px border-l-2 border-dashed border-muted-foreground/30 pointer-events-none"
                        style={{
                          left: `${labelLeft}px`,
                          top: `${connectorTop}px`,
                          height: `${connectorHeight}px`,
                        }}
                        data-connector
                      />
                    )}

                    {/* Label - positioned at same horizontal location as connector */}
                    <div
                      className="absolute whitespace-nowrap pointer-events-none"
                      style={{
                        left: `${labelLeft + 2}px`, // 2px offset to avoid overlapping connector
                        top: `${labelTop}px`,
                      }}
                      data-run-label
                      data-run-name={run.run_name}
                      data-label-above={labelAbove}
                      data-label-row={labelRow}
                    >
                      <span className="text-[10px] font-medium text-muted-foreground px-1.5 bg-background/95 rounded border border-border/40 shadow-sm" data-label-span>
                        {run.run_name}
                      </span>
                    </div>
                  </React.Fragment>
                );
              })}
          </div>
        </div>

        {/* Enhanced summary footer with better visual hierarchy */}
        <div className="mt-3 pt-4 border-t border-border/40 flex items-center justify-between text-xs">
          <div className="flex items-center gap-5">
            <div className="flex items-center gap-2 px-2 py-1 rounded-md bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200/50 dark:border-emerald-800/30">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/30" />
              <span className="text-emerald-700 dark:text-emerald-400 font-medium">
                {positionedRuns.items.filter(item => item.run.status === 'completed').length} Completed
              </span>
            </div>
            <div className="flex items-center gap-2 px-2 py-1 rounded-md bg-blue-50 dark:bg-blue-950/30 border border-blue-200/50 dark:border-blue-800/30">
              <div className="w-2.5 h-2.5 rounded-full bg-blue-500 shadow-sm shadow-blue-500/30 animate-pulse" />
              <span className="text-blue-700 dark:text-blue-400 font-medium">
                {positionedRuns.items.filter(item => item.run.status === 'running').length} Running
              </span>
            </div>
            <div className="flex items-center gap-2 px-2 py-1 rounded-md bg-rose-50 dark:bg-rose-950/30 border border-rose-200/50 dark:border-rose-800/30">
              <div className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-sm shadow-rose-500/30" />
              <span className="text-rose-700 dark:text-rose-400 font-medium">
                {positionedRuns.items.filter(item => item.run.status === 'failed').length} Failed
              </span>
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-muted-foreground font-medium">
            <Clock className="h-3.5 w-3.5" />
            <span>Total Duration: {formatDuration(timelineData.totalDuration)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
    </TooltipProvider>
  );
}
