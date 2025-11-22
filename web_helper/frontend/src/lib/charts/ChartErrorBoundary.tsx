/**
 * Chart Error Boundary
 *
 * Catches and displays errors from chart rendering.
 * Provides detailed debugging information for developers.
 */

import { Component, ErrorInfo, ReactNode } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertCircle, Bug, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface Props {
  children: ReactNode;
  chartType?: string;
  renderer?: string;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorCount: number;
}

export class ChartErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to console for debugging
    console.error('[ChartErrorBoundary] Caught error:', {
      error,
      errorInfo,
      chartType: this.props.chartType,
      renderer: this.props.renderer,
      componentStack: errorInfo.componentStack,
    });

    this.setState(prevState => ({
      error,
      errorInfo,
      errorCount: prevState.errorCount + 1,
    }));
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorInfo, errorCount } = this.state;
      const { chartType, renderer } = this.props;

      return (
        <Card className="border-red-500 bg-red-50/50 dark:bg-red-950/20">
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
              <CardTitle className="text-red-900 dark:text-red-100">Chart Rendering Error</CardTitle>
              {errorCount > 1 && (
                <Badge variant="destructive" className="ml-auto">
                  {errorCount} errors
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Error Summary */}
            <div className="p-3 bg-white dark:bg-gray-900 border border-red-200 dark:border-red-800 rounded-lg">
              <div className="flex items-start gap-2 mb-2">
                <Bug className="h-4 w-4 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-semibold text-red-900 dark:text-red-100 mb-1">
                    {error?.name || 'Error'}
                  </p>
                  <p className="text-sm text-red-700 dark:text-red-300 font-mono">
                    {error?.message || 'An unknown error occurred'}
                  </p>
                </div>
              </div>

              {/* Context Information */}
              {(chartType || renderer) && (
                <div className="mt-3 pt-3 border-t border-red-200 dark:border-red-800">
                  <p className="text-xs text-red-600 dark:text-red-400 mb-1 font-medium">Context:</p>
                  <div className="space-y-1">
                    {chartType && (
                      <p className="text-xs text-red-700 dark:text-red-300">
                        <strong>Chart Type:</strong> {chartType}
                      </p>
                    )}
                    {renderer && (
                      <p className="text-xs text-red-700 dark:text-red-300">
                        <strong>Renderer:</strong> {renderer}
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Component Stack (collapsed by default) */}
            {errorInfo?.componentStack && (
              <details className="p-3 bg-white dark:bg-gray-900 border border-red-200 dark:border-red-800 rounded-lg">
                <summary className="text-xs font-medium text-red-900 dark:text-red-100 cursor-pointer">
                  Component Stack (click to expand)
                </summary>
                <pre className="mt-2 text-[10px] text-red-700 dark:text-red-300 overflow-x-auto font-mono whitespace-pre-wrap">
                  {errorInfo.componentStack}
                </pre>
              </details>
            )}

            {/* Error Stack (collapsed by default) */}
            {error?.stack && (
              <details className="p-3 bg-white dark:bg-gray-900 border border-red-200 dark:border-red-800 rounded-lg">
                <summary className="text-xs font-medium text-red-900 dark:text-red-100 cursor-pointer">
                  Error Stack (click to expand)
                </summary>
                <pre className="mt-2 text-[10px] text-red-700 dark:text-red-300 overflow-x-auto font-mono whitespace-pre-wrap">
                  {error.stack}
                </pre>
              </details>
            )}

            {/* Troubleshooting Tips */}
            <div className="p-3 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <p className="text-xs font-medium text-blue-900 dark:text-blue-100 mb-2">
                Troubleshooting Tips:
              </p>
              <ul className="text-xs text-blue-700 dark:text-blue-300 space-y-1 list-disc list-inside">
                <li>Check that your data format matches the chart configuration</li>
                <li>Ensure all required fields (dataKey, name, etc.) are present</li>
                <li>Try switching to a different chart renderer</li>
                <li>Verify that numeric fields contain valid numbers</li>
                <li>Check the browser console for additional error details</li>
              </ul>
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={this.handleReset}
                className="flex-1"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Try Again
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => window.location.reload()}
                className="flex-1"
              >
                Reload Page
              </Button>
            </div>
          </CardContent>
        </Card>
      );
    }

    return this.props.children;
  }
}
