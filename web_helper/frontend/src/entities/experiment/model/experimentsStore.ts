/**
 * Unified Experiments Store
 *
 * Manages all experiment state in one place to prevent Queue/Results sync issues.
 * Uses Zustand with middleware for:
 * - subscribeWithSelector: Fine-grained subscriptions
 * - immer: Immutable state updates
 *
 * Single source of truth for:
 * - All experiments (pending, running, completed, failed)
 * - Selected experiment for detail view
 * - Polling/SSE connection state
 * - Error handling and retry logic
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { shallow } from 'zustand/shallow';
import { useMemo } from 'react';

// =============================================================================
// Types
// =============================================================================

export type ExperimentStatus =
  | 'pending'
  | 'queued'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface Experiment {
  experiment_uid: string;
  project: string | null;
  config_path: string;
  log_path?: string;
  error_log_path?: string;
  status: ExperimentStatus;
  priority: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  assigned_device?: string | null;
  exit_code?: number | null;
  error_message?: string | null;
  last_indexed?: string;
  metadata?: {
    has_config?: boolean;
    [key: string]: unknown;
  };
}

export interface ExperimentStats {
  total: number;
  pending: number;
  queued: number;
  running: number;
  paused: number;
  completed: number;
  failed: number;
  cancelled: number;
}

// Connection state for infinite loading prevention
type ConnectionState = 'idle' | 'connecting' | 'connected' | 'error' | 'retrying';

interface ExperimentsState {
  // Data (use Record instead of Map for Zustand compatibility)
  experiments: Record<string, Experiment>;
  selectedExperimentId: string | null;

  // Derived (computed on access)
  getExperimentsByStatus: (status: ExperimentStatus | readonly ExperimentStatus[]) => Experiment[];
  getStats: () => ExperimentStats;

  // Connection state (prevents infinite loading)
  connectionState: ConnectionState;
  lastFetchTime: number | null;
  fetchError: string | null;
  retryCount: number;
  maxRetries: number;

  // Loading states per-section (not global)
  loadingStates: {
    list: boolean;
    detail: boolean;
    action: boolean;
  };

  // Actions
  setExperiments: (experiments: Experiment[]) => void;
  updateExperiment: (uid: string, updates: Partial<Experiment>) => void;
  removeExperiment: (uid: string) => void;
  setSelectedExperiment: (uid: string | null) => void;

  // Connection management
  setConnectionState: (state: ConnectionState) => void;
  setFetchError: (error: string | null) => void;
  incrementRetry: () => void;
  resetRetry: () => void;

  // Loading management
  setLoading: (section: keyof ExperimentsState['loadingStates'], loading: boolean) => void;

  // Batch operations
  batchUpdateStatus: (uids: string[], status: ExperimentStatus) => void;
}

// =============================================================================
// Store
// =============================================================================

export const useExperimentsStore = create<ExperimentsState>()(
  subscribeWithSelector(
    immer((set, get) => ({
      // Initial state
      experiments: {} as Record<string, Experiment>,
      selectedExperimentId: null,

      connectionState: 'idle' as ConnectionState,
      lastFetchTime: null,
      fetchError: null,
      retryCount: 0,
      maxRetries: 3,

      loadingStates: {
        list: false,
        detail: false,
        action: false,
      },

      // Derived getters
      getExperimentsByStatus: (status) => {
        const experiments = Object.values(get().experiments);
        const statuses = Array.isArray(status) ? status : [status];
        return experiments.filter((exp) => statuses.includes(exp.status));
      },

      getStats: () => {
        const experiments = Object.values(get().experiments);
        return {
          total: experiments.length,
          pending: experiments.filter((e) => e.status === 'pending').length,
          queued: experiments.filter((e) => e.status === 'queued').length,
          running: experiments.filter((e) => e.status === 'running').length,
          paused: experiments.filter((e) => e.status === 'paused').length,
          completed: experiments.filter((e) => e.status === 'completed').length,
          failed: experiments.filter((e) => e.status === 'failed').length,
          cancelled: experiments.filter((e) => e.status === 'cancelled').length,
        };
      },

      // Actions
      setExperiments: (experiments) =>
        set((state) => {
          const newExperiments: Record<string, Experiment> = {};
          experiments.forEach((exp) => {
            newExperiments[exp.experiment_uid] = exp;
          });
          state.experiments = newExperiments;
          state.lastFetchTime = Date.now();
          state.fetchError = null;
          state.connectionState = 'connected';
        }),

      updateExperiment: (uid, updates) =>
        set((state) => {
          const existing = state.experiments[uid];
          if (existing) {
            state.experiments[uid] = { ...existing, ...updates };
          }
        }),

      removeExperiment: (uid) =>
        set((state) => {
          delete state.experiments[uid];
          if (state.selectedExperimentId === uid) {
            state.selectedExperimentId = null;
          }
        }),

      setSelectedExperiment: (uid) =>
        set((state) => {
          state.selectedExperimentId = uid;
        }),

      // Connection management
      setConnectionState: (connectionState) =>
        set((state) => {
          state.connectionState = connectionState;
        }),

      setFetchError: (error) =>
        set((state) => {
          state.fetchError = error;
          if (error) {
            state.connectionState = 'error';
          }
        }),

      incrementRetry: () =>
        set((state) => {
          state.retryCount += 1;
          if (state.retryCount >= state.maxRetries) {
            state.connectionState = 'error';
          } else {
            state.connectionState = 'retrying';
          }
        }),

      resetRetry: () =>
        set((state) => {
          state.retryCount = 0;
          state.connectionState = 'idle';
        }),

      // Loading management
      setLoading: (section, loading) =>
        set((state) => {
          state.loadingStates[section] = loading;
        }),

      // Batch operations
      batchUpdateStatus: (uids, status) =>
        set((state) => {
          uids.forEach((uid) => {
            const existing = state.experiments[uid];
            if (existing) {
              state.experiments[uid] = { ...existing, status };
            }
          });
        }),
    }))
  )
);

// =============================================================================
// Selectors (for performance - only re-render when specific data changes)
// =============================================================================

export const selectExperimentsList = (state: ExperimentsState) =>
  Object.values(state.experiments);

export const selectSelectedExperiment = (state: ExperimentsState) =>
  state.selectedExperimentId
    ? state.experiments[state.selectedExperimentId]
    : null;

export const selectConnectionStatus = (state: ExperimentsState) => ({
  state: state.connectionState,
  error: state.fetchError,
  retryCount: state.retryCount,
  maxRetries: state.maxRetries,
});

export const selectIsLoading = (state: ExperimentsState) =>
  state.loadingStates.list || state.loadingStates.detail;

// =============================================================================
// Hooks for common patterns
// =============================================================================

/**
 * Hook to get experiments filtered by status with stable reference
 * Selects experiments and computes filtered list in useMemo
 */
export function useExperimentsByStatus(status: ExperimentStatus | readonly ExperimentStatus[]) {
  const experiments = useExperimentsStore(experimentsSelector);

  return useMemo(() => {
    const statuses = Array.isArray(status) ? status : [status];
    return Object.values(experiments).filter((exp) =>
      statuses.includes(exp.status)
    );
  }, [experiments, status]);
}

// Stable selectors for primitive values (defined outside hooks for stability)
const experimentsSelector = (state: ExperimentsState) => state.experiments;
const connectionStateSelector = (state: ExperimentsState) => state.connectionState;
const fetchErrorSelector = (state: ExperimentsState) => state.fetchError;
const retryCountSelector = (state: ExperimentsState) => state.retryCount;
const maxRetriesSelector = (state: ExperimentsState) => state.maxRetries;

/**
 * Hook to get experiment stats
 * Computes stats from experiments and memoizes the result
 */
export function useExperimentStats() {
  const experiments = useExperimentsStore(experimentsSelector);

  return useMemo(() => {
    const experimentsList = Object.values(experiments);
    return {
      total: experimentsList.length,
      pending: experimentsList.filter((e) => e.status === 'pending').length,
      queued: experimentsList.filter((e) => e.status === 'queued').length,
      running: experimentsList.filter((e) => e.status === 'running').length,
      paused: experimentsList.filter((e) => e.status === 'paused').length,
      completed: experimentsList.filter((e) => e.status === 'completed').length,
      failed: experimentsList.filter((e) => e.status === 'failed').length,
      cancelled: experimentsList.filter((e) => e.status === 'cancelled').length,
    };
  }, [experiments]);
}

/**
 * Hook to check if we should show error state (prevents infinite loading)
 * Selects primitive values individually and memoizes the computed result
 */
export function useConnectionHealth() {
  // Select primitive values individually (no shallow needed for primitives)
  const connectionState = useExperimentsStore(connectionStateSelector);
  const fetchError = useExperimentsStore(fetchErrorSelector);
  const retryCount = useExperimentsStore(retryCountSelector);
  const maxRetries = useExperimentsStore(maxRetriesSelector);

  // Memoize the computed result
  return useMemo(() => ({
    isHealthy: connectionState === 'connected' || connectionState === 'idle',
    isRetrying: connectionState === 'retrying',
    hasError: connectionState === 'error',
    error: fetchError,
    canRetry: retryCount < maxRetries,
    retryCount,
  }), [connectionState, fetchError, retryCount, maxRetries]);
}
