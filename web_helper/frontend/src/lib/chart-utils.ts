/**
 * Chart utility functions
 * Common utilities for chart components
 */

/**
 * Generate a safe filename for chart exports
 * Includes timestamp with seconds to avoid duplicates
 * Removes special characters and replaces spaces with underscores
 *
 * @param baseName - Base name for the file (e.g., metric name, chart type)
 * @param extension - File extension (e.g., 'png', 'svg', 'csv')
 * @returns Safe filename with timestamp
 *
 * @example
 * generateChartFilename('train/loss', 'png')
 * // Returns: 'train_loss_20251006_143052.png'
 */
export function generateChartFilename(baseName: string, extension: string): string {
  // Clean base name: remove special chars, replace spaces and slashes with underscores
  const cleanName = baseName
    .replace(/[^\w\s-]/g, '_')  // Replace special chars with underscore
    .replace(/\s+/g, '_')        // Replace spaces with underscore
    .replace(/_{2,}/g, '_')      // Replace multiple underscores with single
    .replace(/^_+|_+$/g, '');    // Trim underscores from start/end

  // Generate timestamp: YYYYMMDD_HHMMSS
  const now = new Date();
  const timestamp = now
    .toISOString()
    .replace(/[-:]/g, '')         // Remove dashes and colons
    .replace(/\.\d{3}Z$/, '')     // Remove milliseconds and Z
    .replace('T', '_');           // Replace T with underscore

  return `${cleanName}_${timestamp}.${extension}`;
}

/**
 * Sanitize a metric key for use in filenames
 * Converts metric keys like 'train/loss' to 'train_loss'
 *
 * @param metricKey - Raw metric key
 * @returns Sanitized metric key
 */
export function sanitizeMetricKey(metricKey: string): string {
  return metricKey
    .replace(/\//g, '_')
    .replace(/\s+/g, '_')
    .toLowerCase();
}
