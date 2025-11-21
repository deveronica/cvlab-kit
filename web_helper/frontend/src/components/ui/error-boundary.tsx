/**
 * Error Boundary Component
 *
 * Catches React errors and prevents them from crashing the entire app.
 * Provides graceful fallback UI with error details and recovery options.
 */

import { Component, ErrorInfo, ReactNode } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';
import { Button } from './button';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import { devError } from '../../lib/dev-utils';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, _errorInfo: ErrorInfo) => void;
  resetKeys?: Array<string | number>;
  level?: 'app' | 'view' | 'section' | 'component';
}

interface State {
  hasError: boolean;
  error: Error | null;
  _errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      _errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      _errorInfo: null,
    };
  }

  componentDidCatch(error: Error, _errorInfo: ErrorInfo) {
    // Log error to console in development
    devError('Error Boundary caught error:', error, _errorInfo);

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, _errorInfo);
    }

    this.setState({
      error,
      _errorInfo,
    });
  }

  componentDidUpdate(prevProps: Props) {
    // Reset error boundary when resetKeys change
    if (this.state.hasError && this.props.resetKeys) {
      const prevKeys = prevProps.resetKeys || [];
      const currentKeys = this.props.resetKeys || [];

      if (prevKeys.length !== currentKeys.length ||
          prevKeys.some((key, index) => key !== currentKeys[index])) {
        this.resetErrorBoundary();
      }
    }
  }

  resetErrorBoundary = () => {
    this.setState({
      hasError: false,
      error: null,
      _errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { level = 'component' } = this.props;
      const { error, _errorInfo } = this.state;

      // Different fallback UIs based on error level
      if (level === 'app') {
        return (
          <div className="min-h-screen flex items-center justify-center bg-background p-4">
            <Card className="max-w-2xl w-full">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <AlertTriangle className="h-8 w-8 text-destructive" />
                  <div>
                    <CardTitle className="text-xl">Application Error</CardTitle>
                    <CardDescription>
                      Something went wrong. Please try reloading the page.
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 bg-muted rounded-lg">
                  <p className="font-mono text-sm text-destructive">
                    {error?.toString()}
                  </p>
                </div>
                <div className="flex gap-2">
                  <Button onClick={() => window.location.reload()} className="gap-2">
                    <RefreshCw className="h-4 w-4" />
                    Reload Page
                  </Button>
                  <Button variant="outline" onClick={() => window.location.href = '/'} className="gap-2">
                    <Home className="h-4 w-4" />
                    Go Home
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        );
      }

      if (level === 'view') {
        return (
          <div className="flex items-center justify-center min-h-[400px] p-4">
            <Card className="max-w-lg w-full" variant="compact">
              <CardHeader variant="compact">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-destructive" />
                  <CardTitle size="base">View Error</CardTitle>
                </div>
                <CardDescription>
                  This view encountered an error. Try refreshing or going back.
                </CardDescription>
              </CardHeader>
              <CardContent variant="compact" className="space-y-3">
                <details className="text-sm">
                  <summary className="cursor-pointer text-muted-foreground hover:text-foreground transition-colors duration-200">
                    Error details
                  </summary>
                  <div className="mt-2 p-3 bg-muted rounded-md">
                    <p className="font-mono text-xs text-destructive break-all">
                      {error?.toString()}
                    </p>
                  </div>
                </details>
                <div className="flex gap-2">
                  <Button size="sm" onClick={this.resetErrorBoundary} className="gap-1.5">
                    <RefreshCw className="h-3.5 w-3.5" />
                    Try Again
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => window.history.back()}>
                    Go Back
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        );
      }

      if (level === 'section') {
        return (
          <Card variant="compact" className="border-destructive/50">
            <CardContent variant="compact" className="py-6">
              <div className="flex items-start gap-3">
                <AlertTriangle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />
                <div className="flex-1 space-y-2">
                  <p className="text-sm font-medium">Section Error</p>
                  <p className="text-xs text-muted-foreground">
                    This section failed to load. {error?.message}
                  </p>
                  <Button size="sm" variant="outline" onClick={this.resetErrorBoundary} className="gap-1.5 h-7">
                    <RefreshCw className="h-3 w-3" />
                    Retry
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        );
      }

      // Component level (default)
      return (
        <div className="p-3 border border-destructive/50 rounded-lg bg-destructive/5">
          <div className="flex items-start gap-2">
            <AlertTriangle className="h-3.5 w-3.5 text-destructive flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-xs font-medium text-destructive">Component Error</p>
              <p className="text-xs text-muted-foreground mt-1">
                {error?.message || 'This component failed to render'}
              </p>
              <button
                onClick={this.resetErrorBoundary}
                className="text-xs text-primary hover:underline mt-1.5"
              >
                Try again
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Hook-based wrapper for functional components
 */
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<Props, 'children'>
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;

  return WrappedComponent;
}
