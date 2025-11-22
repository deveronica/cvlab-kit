import React from "react";
/**
 * Centralized Chart Export Utilities
 *
 * Provides reusable functions for exporting Recharts components to PNG, SVG, and CSV.
 * Fixes common issues with SVG dimension calculation and ensures consistent export behavior.
 */

import { generateChartFilename } from './chart-utils';

export interface ExportOptions {
  filename?: string;
  quality?: number; // PNG quality (0-1), default 0.92
  scale?: number; // Resolution multiplier, default 2
  backgroundColor?: string; // Default 'white'
}

/**
 * Find the actual chart SVG element from a container ref
 *
 * This function handles Recharts' DOM structure properly:
 * - Looks for .recharts-responsive-container
 * - Then finds .recharts-wrapper
 * - Finally gets the main svg.recharts-surface
 *
 * @param containerElement - The container element containing the chart
 * @returns The SVG element or null if not found
 */
export function findChartSVG(containerElement: HTMLElement | null): SVGSVGElement | null {
  if (!containerElement) {
    console.error('Container element is null');
    return null;
  }

  // Try finding via recharts-wrapper first (most reliable)
  const wrapper = containerElement.querySelector('.recharts-wrapper');
  if (wrapper) {
    const svg = wrapper.querySelector('svg.recharts-surface') as SVGSVGElement;
    if (svg) return svg;
  }

  // Fallback: Try recharts-responsive-container directly
  const responsiveContainer = containerElement.querySelector('.recharts-responsive-container');
  if (responsiveContainer) {
    const svg = responsiveContainer.querySelector('svg.recharts-surface') as SVGSVGElement;
    if (svg) return svg;
  }

  console.error('Could not find chart SVG element', {
    hasWrapper: !!wrapper,
    hasResponsiveContainer: !!responsiveContainer,
    containerHTML: containerElement.innerHTML.substring(0, 200)
  });

  return null;
}

/**
 * Get SVG dimensions using native attributes (not getBBox which can be unreliable)
 *
 * @param svgElement - The SVG element
 * @returns Width and height in pixels
 */
export function getSVGDimensions(svgElement: SVGSVGElement): { width: number; height: number } {
  // Use SVG's native width/height attributes (set by Recharts ResponsiveContainer)
  const width = svgElement.width.baseVal.value || svgElement.clientWidth || 800;
  const height = svgElement.height.baseVal.value || svgElement.clientHeight || 400;

  return { width, height };
}

/**
 * Export chart as PNG
 *
 * @param containerRef - React ref to the chart container element
 * @param filename - Base filename (timestamp will be added)
 * @param options - Export options
 */
export async function exportChartToPNG(
  containerRef: React.RefObject<HTMLElement>,
  filename: string,
  options: ExportOptions = {}
): Promise<void> {
  const {
    quality = 0.92,
    scale = 2,
    backgroundColor = 'white'
  } = options;

  const svgElement = findChartSVG(containerRef.current);
  if (!svgElement) {
    alert('차트를 찾을 수 없습니다. 페이지를 새로고침 후 다시 시도해주세요.');
    return;
  }

  const { width, height } = getSVGDimensions(svgElement);

  console.log('Exporting chart to PNG:', {
    width,
    height,
    scale,
    filename,
    svgClasses: svgElement.className.baseVal,
    childElements: svgElement.childElementCount
  });

  // Clone SVG to avoid modifying the original
  const clonedSvg = svgElement.cloneNode(true) as SVGSVGElement;

  // Set explicit dimensions
  clonedSvg.setAttribute('width', width.toString());
  clonedSvg.setAttribute('height', height.toString());

  // Serialize SVG
  const svgData = new XMLSerializer().serializeToString(clonedSvg);

  // Create canvas
  const canvas = document.createElement('canvas');
  canvas.width = width * scale;
  canvas.height = height * scale;

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    console.error('Failed to get canvas context');
    return;
  }

  // Create image from SVG
  const img = new Image();

  return new Promise((resolve, reject) => {
    img.onload = () => {
      // Fill background
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Scale for high resolution
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);

      // Convert to blob and download
      canvas.toBlob(
        blob => {
          if (!blob) {
            reject(new Error('Failed to create blob'));
            return;
          }

          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = generateChartFilename(filename, 'png');
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
          resolve();
        },
        'image/png',
        quality
      );
    };

    img.onerror = (err) => {
      console.error('Failed to load SVG as image:', err);
      reject(err);
    };

    // Encode SVG data
    const encodedData = encodeURIComponent(svgData);
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodedData;
  });
}

/**
 * Export chart as SVG
 *
 * @param containerRef - React ref to the chart container element
 * @param filename - Base filename (timestamp will be added)
 */
export function exportChartToSVG(
  containerRef: React.RefObject<HTMLElement>,
  filename: string
): void {
  const svgElement = findChartSVG(containerRef.current);
  if (!svgElement) {
    alert('차트를 찾을 수 없습니다.');
    return;
  }

  const svgData = new XMLSerializer().serializeToString(svgElement);
  const blob = new Blob([svgData], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = generateChartFilename(filename, 'svg');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export chart data as CSV
 *
 * @param data - Array of data objects
 * @param filename - Base filename (timestamp will be added)
 * @param columns - Optional column names (if not provided, uses all object keys)
 */
export function exportChartDataToCSV<T extends Record<string, any>>(
  data: T[],
  filename: string,
  columns?: string[]
): void {
  if (data.length === 0) {
    console.warn('No data to export');
    return;
  }

  // Determine columns
  const cols = columns || Object.keys(data[0]);

  // Create CSV content
  const headers = cols.join(',');
  const rows = data.map(row =>
    cols.map(col => {
      const value = row[col];
      // Handle values that might contain commas
      if (typeof value === 'string' && value.includes(',')) {
        return `"${value}"`;
      }
      return value !== undefined && value !== null ? value : '';
    }).join(',')
  );

  const csvContent = [headers, ...rows].join('\n');

  // Download
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = generateChartFilename(filename, 'csv');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
