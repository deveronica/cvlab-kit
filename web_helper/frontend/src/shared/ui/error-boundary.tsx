import { Component, type ErrorInfo, type ReactNode } from 'react';

type ErrorBoundaryLevel = 'app' | 'view' | 'section' | 'component';

interface ErrorBoundaryProps {
  children: ReactNode;
  level?: ErrorBoundaryLevel;
  resetKeys?: readonly unknown[];
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

function areResetKeysEqual(
  prevResetKeys: readonly unknown[] = [],
  nextResetKeys: readonly unknown[] = []
): boolean {
  if (prevResetKeys.length !== nextResetKeys.length) {
    return false;
  }
  return prevResetKeys.every((key, index) => Object.is(key, nextResetKeys[index]));
}

const LEVEL_TITLE: Record<ErrorBoundaryLevel, string> = {
  app: 'Application Error',
  view: 'View Error',
  section: 'Section Error',
  component: 'Component Error',
};

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    error: null,
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.props.onError?.(error, errorInfo);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    const prevResetKeys = prevProps.resetKeys ?? [];
    const nextResetKeys = this.props.resetKeys ?? [];
    if (!areResetKeysEqual(prevResetKeys, nextResetKeys) && this.state.hasError) {
      this.reset();
    }
  }

  reset(): void {
    this.setState({ hasError: false, error: null });
  }

  render(): ReactNode {
    const { children, fallback, level = 'component' } = this.props;
    const { hasError, error } = this.state;

    if (!hasError || !error) {
      return children;
    }

    if (typeof fallback === 'function') {
      return fallback(error, this.reset.bind(this));
    }

    if (fallback) {
      return fallback;
    }

    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-red-900 dark:border-red-900 dark:bg-red-950/40 dark:text-red-100">
        <div className="mb-2 text-sm font-semibold">{LEVEL_TITLE[level]}</div>
        <div className="mb-3 text-xs opacity-90">{error.message}</div>
        <button
          type="button"
          onClick={this.reset.bind(this)}
          className="rounded border border-red-300 px-2 py-1 text-xs hover:bg-red-100 dark:border-red-800 dark:hover:bg-red-900/40"
        >
          Retry
        </button>
      </div>
    );
  }
}
