import React from "react";
/**
 * Report Export Utilities
 *
 * Export comparison reports in multiple formats:
 * - Markdown: Human-readable text format
 * - PNG: Screenshots of charts
 * - JSON: Machine-readable data format
 */

import type { Run } from './types';

export interface ReportData {
  projectName: string;
  runs: Run[];
  generatedAt: string;
  keyMetric?: string;
  bestRun?: Run;
}

/**
 * Export comparison report as Markdown
 */
export function exportAsMarkdown(data: ReportData): string {
  const { projectName, runs, generatedAt, keyMetric = 'val/acc', bestRun } = data;

  const lines: string[] = [];

  // Header
  lines.push(`# ${projectName} - Experiment Comparison Report`);
  lines.push('');
  lines.push(`**Generated**: ${new Date(generatedAt).toLocaleString()}`);
  lines.push(`**Runs Compared**: ${runs.length}`);
  lines.push('');

  // Best Run Section
  if (bestRun) {
    lines.push('## 🏆 Best Overall Run');
    lines.push('');
    lines.push(`- **Run Name**: \`${bestRun.run_name}\``);
    lines.push(`- **Key Metric** (${keyMetric}): **${bestRun.metrics?.final?.[keyMetric]?.toFixed(4) || 'N/A'}**`);

    if (bestRun.started_at) {
      lines.push(`- **Started**: ${new Date(bestRun.started_at).toLocaleDateString()}`);
    }

    // Top secondary metrics
    const metrics = bestRun.metrics?.final || {};
    const secondaryMetrics = Object.entries(metrics)
      .filter(([key]) => key !== keyMetric && key !== 'step' && key !== 'epoch')
      .slice(0, 3);

    if (secondaryMetrics.length > 0) {
      lines.push('');
      lines.push('**Other Metrics**:');
      secondaryMetrics.forEach(([key, value]) => {
        const display = typeof value === 'number' ? value.toFixed(4) : value;
        lines.push(`- ${key.replace(/_/g, ' ')}: ${display}`);
      });
    }

    lines.push('');
  }

  // Runs Comparison Table
  lines.push('## Runs Comparison');
  lines.push('');

  // Get all unique metric keys
  const allMetrics = new Set<string>();
  runs.forEach(run => {
    const metrics = run.metrics?.final || {};
    Object.keys(metrics).forEach(key => {
      if (key !== 'step' && key !== 'epoch') {
        allMetrics.add(key);
      }
    });
  });

  const metricKeys = Array.from(allMetrics).slice(0, 5); // Limit to 5 metrics for readability

  // Table header
  lines.push(`| Run Name | ${metricKeys.map(k => k.replace(/_/g, ' ')).join(' | ')} |`);
  lines.push(`|----------|${metricKeys.map(() => '----------').join('|')}|`);

  // Table rows
  runs.forEach(run => {
    const metrics = run.metrics?.final || {};
    const values = metricKeys.map(key => {
      const value = metrics[key];
      return typeof value === 'number' ? value.toFixed(4) : 'N/A';
    });

    const runName = run.run_name?.substring(0, 30) || 'Unknown';
    lines.push(`| \`${runName}\` | ${values.join(' | ')} |`);
  });

  lines.push('');

  // Hyperparameters Section
  if (runs.length > 0 && runs[0].hyperparameters) {
    lines.push('## Hyperparameters');
    lines.push('');

    // Get common hyperparameters
    const allHparams = new Set<string>();
    runs.forEach(run => {
      if (run.hyperparameters) {
        Object.keys(run.hyperparameters).forEach(key => allHparams.add(key));
      }
    });

    const hparamKeys = Array.from(allHparams).slice(0, 5);

    lines.push(`| Run Name | ${hparamKeys.join(' | ')} |`);
    lines.push(`|----------|${hparamKeys.map(() => '----------').join('|')}|`);

    runs.forEach(run => {
      const hparams = run.hyperparameters || {};
      const values = hparamKeys.map(key => {
        const value = hparams[key];
        return value !== undefined && value !== null ? String(value) : 'N/A';
      });

      const runName = run.run_name?.substring(0, 30) || 'Unknown';
      lines.push(`| \`${runName}\` | ${values.join(' | ')} |`);
    });

    lines.push('');
  }

  // Footer
  lines.push('---');
  lines.push('');
  lines.push('*Generated with [CVLab-Kit Web Helper](https://github.com/)*');

  return lines.join('\n');
}

/**
 * Export comparison report as JSON
 */
export function exportAsJSON(data: ReportData): string {
  return JSON.stringify(data, null, 2);
}

/**
 * Download report as file
 */
export function downloadReport(content: string, _filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = _filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export chart as PNG (uses existing chart-export utilities)
 */
export function exportChartAsPNG(
  _chartContainerRef: React.RefObject<HTMLDivElement>,
  _filename: string
): Promise<void> {
  // This will be implemented using the existing chart-export.ts utilities
  // For now, return a placeholder
  return Promise.resolve();
}
