/**
 * Queue management API hooks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/shared/api/api-client';
import { queryKeys } from '@/shared/api/react-query';
import type { QueueResponse, QueueJob } from '@/entities/node-system/model/types';
import { devInfo, devError } from '@/shared/lib/utils';

export function useQueue() {
  return useQuery<QueueResponse>({
    queryKey: queryKeys.queue,
    queryFn: () => apiClient.listJobs(),
    // SSE handles real-time updates, polling disabled to reduce CPU/memory usage
    // refetchInterval: 10 * 1000,
    // refetchIntervalInBackground: true,
    staleTime: 60 * 1000, // Consider data fresh for 60 seconds
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
  const pending = jobs.filter((job: QueueJob) => job.status === 'pending' || job.status === 'queued').length;
  const running = jobs.filter((job: QueueJob) => job.status === 'running').length;
  const completed = jobs.filter((job: QueueJob) => job.status === 'completed').length;

  return {
    total: jobs.length,
    pending,
    running,
    completed,
  };
}