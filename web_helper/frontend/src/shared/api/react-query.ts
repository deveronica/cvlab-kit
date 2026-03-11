/**
 * React Query configuration and setup
 */

import { QueryClient } from '@tanstack/react-query';

// Create QueryClient with optimized defaults
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Cache data for 30 seconds before considering it stale
      staleTime: 30 * 1000,
      // Keep data in cache for 5 minutes
      gcTime: 5 * 60 * 1000,
      // Don't refetch on window focus (we have SSE for real-time updates)
      refetchOnWindowFocus: false,
      // Retry failed requests 2 times
      retry: 2,
      // Retry delay: 1s, 2s, 4s
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
    mutations: {
      // Retry mutations once
      retry: 1,
    },
  },
});

// Query keys for consistent cache management
export const queryKeys = {
  // Projects and runs
  projects: ['projects'] as const,
  project: (name: string) => ['projects', name] as const,
  projectExperiments: (name: string | null) => ['projects', name, 'experiments'] as const,
  runs: (project?: string) => ['runs', project] as const,
  run: (project: string, runId: string) => ['runs', project, runId] as const,

  // Devices
  devices: ['devices'] as const,
  device: (hostId: string) => ['devices', hostId] as const,

  // Queue
  queue: ['queue'] as const,
  queueJobs: ['queue', 'jobs'] as const,

  // Components (for YAML autocomplete)
  components: ['components'] as const,
  componentsByCategory: (category: string) => ['components', category] as const,

  // Metrics
  metrics: (project: string, runId: string) => ['metrics', project, runId] as const,
} as const;