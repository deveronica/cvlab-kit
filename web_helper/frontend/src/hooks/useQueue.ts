/**
 * Queue management API hooks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../lib/api-client';
import { queryKeys } from '../lib/react-query';
import type { QueueResponse, QueueJob } from '../lib/types';
import { devInfo, devError } from '../lib/dev-utils';

export function useQueue() {
  return useQuery<QueueResponse>({
    queryKey: queryKeys.queue,
    queryFn: () => apiClient.listJobs(),
    // Refetch every 10 seconds as fallback (SSE should update real-time)
    refetchInterval: 10 * 1000,
    refetchIntervalInBackground: true, // Keep refetching even when tab is not focused
    select: (data): QueueResponse => {
      // Handle both possible response formats
      if (Array.isArray(data)) {
        return { jobs: data as QueueJob[], total: data.length };
      }
      return data as QueueResponse;
    },
  });
}

export function useReindexMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => apiClient.reindex(),
    onSuccess: (data) => {
      // Invalidate projects to show newly indexed runs
      queryClient.invalidateQueries({ queryKey: queryKeys.projects });

      devInfo(`✅ Reindexed ${data.indexed} files`);
    },
    onError: (error) => {
      devError('Failed to reindex:', error);
    },
  });
}

// Derived hooks for queue statistics
export function useQueueStats() {
  const { data: queueData } = useQueue();

  if (!queueData || !queueData.jobs) {
    return {
      total: 0,
      pending: 0,
      running: 0,
      completed: 0,
    };
  }

  const jobs = queueData.jobs;
  const pending = jobs.filter((job: any) => job.status === 'pending' || job.status === 'queued').length;
  const running = jobs.filter((job: any) => job.status === 'running').length;
  const completed = jobs.filter((job: any) => job.status === 'completed' || job.status === 'finished').length;

  return {
    total: jobs.length,
    pending,
    running,
    completed,
  };
}