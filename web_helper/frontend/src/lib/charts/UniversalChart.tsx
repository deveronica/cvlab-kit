import React from "react";
/**
 * Universal Chart Component
 *
 * Renders charts using pluggable adapters (Recharts, Victory, Nivo, etc.)
 * Allows users to switch between different chart libraries on the fly.
 */

import { useMemo } from 'react';
import type { ChartRenderer, UniversalChartConfig, ChartAdapter } from './types';
import { rechartsAdapter } from './adapters/RechartsAdapter';
import { chartjsAdapter } from './adapters/ChartJsAdapter';
import { plotlyAdapter } from './adapters/PlotlyAdapter';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertCircle } from 'lucide-react';
import { ChartErrorBoundary } from './ChartErrorBoundary';

/**
 * Adapter registry
 * Maps renderer names to adapter instances
 * Only includes top 3 most popular chart libraries for React
 */
const ADAPTER_REGISTRY = new Map<ChartRenderer, ChartAdapter>([
  ['recharts', rechartsAdapter],
  ['chartjs', chartjsAdapter],
  ['plotly', plotlyAdapter],
]);

/**
 * Get adapter by renderer name
 */
function getAdapter(renderer: ChartRenderer): ChartAdapter | null {
  return ADAPTER_REGISTRY.get(renderer) || null;
}

/**
 * Get list of available renderers
 */
export function getAvailableRenderers(): ChartRenderer[] {
  return Array.from(ADAPTER_REGISTRY.keys());
}

/**
 * Check if a renderer is available
 */
export function isRendererAvailable(renderer: ChartRenderer): boolean {
  return ADAPTER_REGISTRY.has(renderer);
}

export interface UniversalChartProps extends UniversalChartConfig {
  /** Chart renderer to use */
  renderer?: ChartRenderer;
  /** Show renderer badge */
  showRendererBadge?: boolean;
  /** Wrap in card */
  wrapInCard?: boolean;
  /** Card className */
  cardClassName?: string;
}

/**
 * Universal Chart Component
 *
 * @example
 * <UniversalChart
 *   type="line"
 *   renderer="recharts"
 *   data={data}
 *   series={[
 *     { dataKey: 'train_loss', name: 'Train Loss', color: '#3b82f6' },
 *     { dataKey: 'val_loss', name: 'Val Loss', color: '#10b981' },
 *   ]}
 *   xAxis={{ dataKey: 'epoch', label: 'Epoch' }}
 *   yAxis={{ label: 'Loss' }}
 * />
 */
export function UniversalChart({
  renderer = 'recharts',
  showRendererBadge = false,
  wrapInCard = false,
  cardClassName,
  ...config
}: UniversalChartProps) {
  // Get adapter
  const adapter = useMemo(() => getAdapter(renderer), [renderer]);

  // Validate adapter
  if (!adapter) {
    return (
      <ErrorDisplay
        title="Renderer Not Available"
        message={`Chart renderer "${renderer}" is not available. Available renderers: ${getAvailableRenderers().join(', ')}`}
        wrapInCard={wrapInCard}
        cardClassName={cardClassName}
      />
    );
  }

  // Check if adapter supports this chart type
  if (!adapter.supportsType(config.type)) {
    return (
      <ErrorDisplay
        title="Chart Type Not Supported"
        message={`${adapter.name} does not support "${config.type}" charts. Supported types: ${adapter.supportedTypes.join(', ')}`}
        wrapInCard={wrapInCard}
        cardClassName={cardClassName}
      />
    );
  }

  // Render chart with error boundary
  const chartElement = (
    <ChartErrorBoundary chartType={config.type} renderer={adapter.name}>
      {adapter.render(config)}
    </ChartErrorBoundary>
  );

  // Wrap in card if requested
  if (wrapInCard) {
    return (
      <Card className={cardClassName}>
        {(config.title || config.description) && (
          <CardHeader>
            <div className="flex items-start justify-between">
              <div>
                {config.title && <CardTitle size="lg">{config.title}</CardTitle>}
                {config.description && <CardDescription>{config.description}</CardDescription>}
              </div>
              {showRendererBadge && (
                <Badge variant="outline" className="ml-2 flex-shrink-0">
                  {adapter.name}
                </Badge>
              )}
            </div>
          </CardHeader>
        )}
        <CardContent>{chartElement}</CardContent>
      </Card>
    );
  }

  // Render without card
  return (
    <div className="relative">
      {showRendererBadge && (
        <Badge variant="outline" className="absolute top-2 right-2 z-10">
          {adapter.name}
        </Badge>
      )}
      {chartElement}
    </div>
  );
}

/**
 * Error display component
 */
function ErrorDisplay({
  title,
  message,
  wrapInCard,
  cardClassName,
}: {
  title: string;
  message: string;
  wrapInCard: boolean;
  cardClassName?: string;
}) {
  const content = (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      <AlertCircle className="h-12 w-12 text-destructive mb-4" />
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground max-w-md">{message}</p>
    </div>
  );

  if (wrapInCard) {
    return (
      <Card className={cardClassName}>
        <CardContent className="pt-6">{content}</CardContent>
      </Card>
    );
  }

  return content;
}

/**
 * Register a custom adapter
 * Useful for adding third-party or custom adapters
 */
export function registerAdapter(renderer: ChartRenderer, adapter: ChartAdapter): void {
  ADAPTER_REGISTRY.set(renderer, adapter);
}

/**
 * Unregister an adapter
 */
export function unregisterAdapter(renderer: ChartRenderer): boolean {
  return ADAPTER_REGISTRY.delete(renderer);
}
