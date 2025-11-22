/**
 * Time formatting utilities
 * Standardizes time display to seconds (no microseconds)
 */

export function formatTimestamp(timestamp: string | Date | null | undefined): string {
  if (!timestamp) return 'Never';

  try {
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    
    if (isNaN(date.getTime())) return 'Invalid date';

    // Format: YYYY-MM-DD HH:MM:SS (no microseconds)
    return date.toISOString().split('.')[0].replace('T', ' ');
  } catch {
    return 'Invalid date';
  }
}

export function formatRelativeTime(timestamp: string | Date | null | undefined): string {
  if (!timestamp) return 'Never';

  try {
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    
    if (isNaN(date.getTime())) return 'Invalid date';

    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);

    if (diffSec < 60) return `${diffSec}s ago`;
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
    if (diffSec < 604800) return `${Math.floor(diffSec / 86400)}d ago`;
    
    // Older than a week: show absolute date
    return formatTimestamp(date);
  } catch {
    return 'Invalid date';
  }
}

export function formatDuration(seconds: number): string {
  if (!seconds || seconds < 0) return '0s';

  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
}
