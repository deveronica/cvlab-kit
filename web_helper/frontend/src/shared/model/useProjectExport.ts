import { useRef, useCallback } from 'react';
import { getValue } from '@/shared/lib/table-columns';
import { devError } from '@/shared/lib/utils';
import html2canvas from 'html2canvas';
import { toSvg } from 'html-to-image';

export type ExportFormat = 'json' | 'markdown' | 'png' | 'svg' | 'csv';

export interface ExportResult {
  chartsRef: React.RefObject<HTMLDivElement>;
  exportRuns: (format: ExportFormat) => Promise<void>;
  getSelectedExperimentData: () => Record<string, unknown>[];
}

interface UseProjectExportOptions {
  activeProject?: string | null;
  experimentsData: Record<string, unknown>[];
  selectedExperiments: string[];
  hyperparamColumns: string[];
  metricColumns: string[];
  flattenParams: boolean;
}

/**
 * Hook for managing export functionality.
 * Extracted from useProjectsView for better separation of concerns.
 */
export function useProjectExport(options: UseProjectExportOptions): ExportResult {
  const {
    activeProject,
    experimentsData,
    selectedExperiments,
    hyperparamColumns,
    metricColumns,
    flattenParams,
  } = options;

  const chartsRef = useRef<HTMLDivElement>(null);

  const exportRuns = useCallback(async (format: ExportFormat) => {
    const runsToExport = selectedExperiments.length > 0
      ? experimentsData.filter(exp => selectedExperiments.includes(exp.run_name as string))
      : experimentsData;

    if (runsToExport.length === 0) {
      alert('No runs to export');
      return;
    }

    const timestamp = new Date().toISOString().split('T')[0];
    const projectPrefix = activeProject ? `${activeProject}_` : '';

    switch (format) {
      case 'json': {
        const data = JSON.stringify(runsToExport, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${projectPrefix}runs_${timestamp}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        break;
      }

      case 'csv': {
        // Generate CSV with hyperparameters and metrics
        const headers = ['run_name', ...hyperparamColumns.map(h => `hp_${h}`), ...metricColumns.map(m => `metric_${m}`)];
        const rows = runsToExport.map(run => {
          const row: string[] = [run.run_name as string];

          // Add hyperparameters
          hyperparamColumns.forEach(param => {
            const value = getValue((run.hyperparameters || {}) as Record<string, unknown>, param, flattenParams);
            row.push(value !== undefined && value !== null ? String(value) : '');
          });

          // Add metrics
          metricColumns.forEach(metric => {
            const value = getValue((run.final_metrics || {}) as Record<string, unknown>, metric, flattenParams);
            row.push(value !== undefined && value !== null ? String(value) : '');
          });

          return row;
        });

        const csv = [headers.join(','), ...rows.map(r => r.map(cell => {
          const str = String(cell);
          return str.includes(',') ? `"${str.replace(/"/g, '""')}"` : str;
        }).join(','))].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${projectPrefix}runs_${timestamp}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        break;
      }

      case 'markdown': {
        // Generate Markdown table
        let md = `# ${activeProject || 'Project'} Runs Export\n\n`;
        md += `Generated: ${new Date().toISOString()}\n\n`;
        md += `## Summary\n\n`;
        md += `- Total Runs: ${runsToExport.length}\n`;
        md += `- Selected Runs: ${selectedExperiments.length || 'All'}\n\n`;
        md += `## Runs Table\n\n`;

        // Table header
        const headers = ['Run Name', ...hyperparamColumns.slice(0, 5), ...metricColumns.slice(0, 5)];
        md += `| ${headers.join(' | ')} |\n`;
        md += `| ${headers.map(() => '---').join(' | ')} |\n`;

        // Table rows
        runsToExport.forEach(run => {
          const cells: string[] = [run.run_name as string];

          hyperparamColumns.slice(0, 5).forEach(param => {
            const value = getValue((run.hyperparameters || {}) as Record<string, unknown>, param, flattenParams);
            cells.push(value !== undefined && value !== null ? String(value) : 'N/A');
          });

          metricColumns.slice(0, 5).forEach(metric => {
            const value = getValue((run.final_metrics || {}) as Record<string, unknown>, metric, flattenParams);
            cells.push(value !== undefined && value !== null
              ? (typeof value === 'number' ? value.toFixed(4) : String(value))
              : 'N/A');
          });

          md += `| ${cells.join(' | ')} |\n`;
        });

        const blob = new Blob([md], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${projectPrefix}runs_${timestamp}.md`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        break;
      }

      case 'png': {
        if (!chartsRef.current) {
          alert('Charts view not available. Switch to Charts tab first.');
          return;
        }

        try {
          const canvas = await html2canvas(chartsRef.current, {
            backgroundColor: '#ffffff',
            scale: 2,
          });

          canvas.toBlob((blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${projectPrefix}charts_${timestamp}.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
          });
        } catch (error) {
          devError('Failed to export as PNG:', error);
          alert('Failed to export as PNG. Please try again.');
        }
        break;
      }

      case 'svg': {
        if (!chartsRef.current) {
          alert('Charts view not available. Switch to Charts tab first.');
          return;
        }

        try {
          const dataUrl = await toSvg(chartsRef.current, {
            backgroundColor: '#ffffff',
          });

          const link = document.createElement('a');
          link.href = dataUrl;
          link.download = `${projectPrefix}charts_${timestamp}.svg`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        } catch (error) {
          devError('Failed to export as SVG:', error);
          alert('Failed to export as SVG. Please try again.');
        }
        break;
      }
    }
  }, [activeProject, experimentsData, selectedExperiments, hyperparamColumns, metricColumns, flattenParams]);

  const getSelectedExperimentData = useCallback(() => {
    return experimentsData.filter(exp =>
      selectedExperiments.includes(exp.run_name as string)
    );
  }, [experimentsData, selectedExperiments]);

  return {
    chartsRef,
    exportRuns,
    getSelectedExperimentData,
  };
}
