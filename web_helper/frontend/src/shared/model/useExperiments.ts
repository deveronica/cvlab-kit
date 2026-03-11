/**
 * Unified Experiments Hook
 *
 * Single hook for fetching and syncing experiment data.
 * Replaces separate useQueue and Results fetching logic.
 *
 * Features:
 * - Smart retry with exponential backoff
 * - Infinite loading prevention
 * - SSE integration for real-time updates
 * - Optimistic updates for actions
 */

import { useEffect, useCallback, useRef, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  useExperimentsStore,
  useExperimentsByStatus,
  type Experiment,
  type ExperimentStatus,
} from '@/entities/experiment/model/experimentsStore';
import { devInfo, devError, devWarn } from '@/shared/lib/utils';

// =============================================================================
// API Functions
// =============================================================================

const API_BASE = '/api/queue';

async function fetchAllExperiments(): Promise<Experiment[]> {
  const response = await fetch(`${API_BASE}/experiments?limit=1000`);
  if (!response.ok) {
    throw new Error(`Failed to fetch experiments: ${response.status}`);
  }
  const data = await response.json();
  return data.data?.experiments || [];
}

async function updateExperimentStatus(
  uid: string,
  action: 'cancel' | 'pause' | 'resume'
): Promise<void> {
  const response = await fetch(`${API_BASE}/experiments/${uid}/${action}`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`Failed to ${action} experiment: ${response.status}`);
  }
}

// =============================================================================
// Main Hook
// =============================================================================

interface UseExperimentsOptions {
  /** Enable automatic polling (default: false, use SSE instead) */
  enablePolling?: boolean;
  /** Polling interval in ms (default: 30000) */
  pollingInterval?: number;
  /** Enable SSE real-time updates (default: true) */
  enableSSE?: boolean;
}

export function useExperiments(options: UseExperimentsOptions = {}) {
  const {
    enablePolling = false,
    pollingInterval = 30000,
    enableSSE = true,
  } = options;

  const queryClient = useQueryClient();

  // Select each store action individually - these are stable references in Zustand
  // This prevents the infinite loop caused by object selectors
  const setExperiments = useExperimentsStore((s) => s.setExperiments);
  const updateExperiment = useExperimentsStore((s) => s.updateExperiment);
  const setConnectionState = useExperimentsStore((s) => s.setConnectionState);
  const setFetchError = useExperimentsStore((s) => s.setFetchError);
  const incrementRetry = useExperimentsStore((s) => s.incrementRetry);
  const resetRetry = useExperimentsStore((s) => s.resetRetry);
  const setLoading = useExperimentsStore((s) => s.setLoading);

  // Select state values individually
  const connectionState = useExperimentsStore((s) => s.connectionState);

  // Refs for cleanup
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const sseRef = useRef<EventSource | null>(null);
  const mountedRef = useRef(true);

  // ==========================================================================
  // Fetch Logic with Smart Retry
  // ==========================================================================

  // Get retry values from store directly in callbacks (no re-render dependency)
  const getRetryState = useCallback(() => {
    const state = useExperimentsStore.getState();
    return { retryCount: state.retryCount, maxRetries: state.maxRetries };
  }, []);

  const fetchData = useCallback(async () => {
    if (!mountedRef.current) return;

    setLoading('list', true);
    setConnectionState('connecting');

    try {
      const experiments = await fetchAllExperiments();

      if (!mountedRef.current) return;

      setExperiments(experiments);
      resetRetry();
      devInfo(`✅ Loaded ${experiments.length} experiments`);
    } catch (error) {
      if (!mountedRef.current) return;

      const message = error instanceof Error ? error.message : 'Unknown error';
      devError('❌ Failed to fetch experiments:', message);

      setFetchError(message);
      incrementRetry();

      // Exponential backoff retry - get current values from store
      const { retryCount, maxRetries } = getRetryState();
      if (retryCount < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, retryCount), 30000);
        devWarn(`🔄 Retrying in ${delay / 1000}s (attempt ${retryCount + 1}/${maxRetries})`);

        setTimeout(() => {
          if (mountedRef.current) {
            fetchData();
          }
        }, delay);
      }
    } finally {
      if (mountedRef.current) {
        setLoading('list', false);
      }
    }
  }, [
    setExperiments,
    setConnectionState,
    setFetchError,
    incrementRetry,
    resetRetry,
    setLoading,
    getRetryState,
  ]);

  // ==========================================================================
  // SSE Integration
  // ==========================================================================

  const setupSSE = useCallback(() => {
    if (!enableSSE || sseRef.current) return;

    try {
      const eventSource = new EventSource('/api/events/stream');
      sseRef.current = eventSource;

      eventSource.onopen = () => {
        devInfo('📡 SSE connected for experiments');
        setConnectionState('connected');
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Handle experiment updates
          if (data.type === 'experiment_update' && data.experiment) {
            updateExperiment(data.experiment.experiment_uid, data.experiment);
          }

          // Handle queue updates (refresh all)
          if (data.type === 'queue_update') {
            fetchData();
          }
        } catch (e) {
          // Ignore parse errors for keepalive messages
        }
      };

      eventSource.onerror = () => {
        devWarn('📡 SSE disconnected, will reconnect...');
        setConnectionState('retrying');

        // Close and cleanup
        eventSource.close();
        sseRef.current = null;

        // Reconnect after delay
        setTimeout(() => {
          if (mountedRef.current && enableSSE) {
            setupSSE();
          }
        }, 5000);
      };
    } catch (error) {
      devError('Failed to setup SSE:', error);
    }
  }, [enableSSE, updateExperiment, fetchData, setConnectionState]);

  // ==========================================================================
  // Polling (fallback if SSE disabled)
  // ==========================================================================

  useEffect(() => {
    if (!enablePolling) return;

    pollingRef.current = setInterval(() => {
      if (mountedRef.current) {
        fetchData();
      }
    }, pollingInterval);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [enablePolling, pollingInterval, fetchData]);

  // ==========================================================================
  // Initial Load and Cleanup
  // ==========================================================================

  useEffect(() => {
    mountedRef.current = true;

    // Initial fetch
    fetchData();

    // Setup SSE
    if (enableSSE) {
      setupSSE();
    }

    return () => {
      mountedRef.current = false;

      // Cleanup SSE
      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }

      // Cleanup polling
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [fetchData, setupSSE, enableSSE]);

  // ==========================================================================
  // Actions with Optimistic Updates
  // ==========================================================================

  const cancelExperiment = useCallback(
    async (uid: string) => {
      // Optimistic update
      updateExperiment(uid, { status: 'cancelled' });
      setLoading('action', true);

      try {
        await updateExperimentStatus(uid, 'cancel');
        devInfo(`✅ Cancelled experiment ${uid}`);
      } catch (error) {
        // Revert on error
        devError('Failed to cancel:', error);
        fetchData(); // Refresh to get actual state
      } finally {
        setLoading('action', false);
      }
    },
    [updateExperiment, setLoading, fetchData]
  );

  const pauseExperiment = useCallback(
    async (uid: string) => {
      updateExperiment(uid, { status: 'paused' });
      setLoading('action', true);

      try {
        await updateExperimentStatus(uid, 'pause');
        devInfo(`✅ Paused experiment ${uid}`);
      } catch (error) {
        devError('Failed to pause:', error);
        fetchData();
      } finally {
        setLoading('action', false);
      }
    },
    [updateExperiment, setLoading, fetchData]
  );

  const resumeExperiment = useCallback(
    async (uid: string) => {
      updateExperiment(uid, { status: 'queued' });
      setLoading('action', true);

      try {
        await updateExperimentStatus(uid, 'resume');
        devInfo(`✅ Resumed experiment ${uid}`);
      } catch (error) {
        devError('Failed to resume:', error);
        fetchData();
      } finally {
        setLoading('action', false);
      }
    },
    [updateExperiment, setLoading, fetchData]
  );

  const refresh = useCallback(() => {
    resetRetry();
    fetchData();
  }, [resetRetry, fetchData]);

  return {
    // Actions
    cancelExperiment,
    pauseExperiment,
    resumeExperiment,
    refresh,

    // State
    connectionState,
    isRetrying: connectionState === 'retrying',
    hasError: connectionState === 'error',
  };
}

// =============================================================================
// Convenience Hooks
// =============================================================================

// Status arrays as constants (stable references)
const QUEUE_STATUSES = ['pending', 'queued', 'running', 'paused'] as const;

/**
 * Hook specifically for Queue view (pending, running, paused jobs)
 */
export function useQueueExperiments() {
  return useExperimentsByStatus(QUEUE_STATUSES);
}

/**
 * Hook specifically for Results view (all experiments)
 * Selects the experiments object and converts to array in useMemo
 */
export function useResultsExperiments() {
  const experiments = useExperimentsStore((s) => s.experiments);
  return useMemo(() => Object.values(experiments), [experiments]);
}

/**
 * Hook for a single experiment by UID
 * Note: Parameterized selectors need special handling
 */
export function useExperiment(uid: string | null) {
  // Memoize selector to prevent recreation on every render
  const selector = useMemo(
    () => (state: ReturnType<typeof useExperimentsStore.getState>) =>
      uid ? state.experiments[uid] : undefined,
    [uid]
  );
  return useExperimentsStore(selector);
}
