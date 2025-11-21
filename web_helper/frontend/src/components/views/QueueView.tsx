import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { useQueue } from '../../hooks/useQueue';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../../lib/api-client';

import { useNavigationStore } from '@/store/navigationStore';

export function QueueView() {
  const { navigateToExperiment } = useNavigationStore();
  const queryClient = useQueryClient();
  const { data: queue = {jobs: [], total: 0}, isLoading } = useQueue();

  const cancelMutation = useMutation({
      mutationFn: (jobId: string) => apiClient.cancelJob(jobId),
      onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ['queue'] });
      }
  });

  const pauseMutation = useMutation({
      mutationFn: (jobId: string) => apiClient.pauseJob(jobId),
      onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ['queue'] });
      }
  });

  const resumeMutation = useMutation({
      mutationFn: (jobId: string) => apiClient.resumeJob(jobId),
      onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ['queue'] });
      }
  });

  const queueJobs = Array.isArray(queue) ? queue : queue?.jobs || [];
  const pendingJobs = queueJobs.filter(job => job.status === 'pending' || job.status === 'queued');
  const runningJobs = queueJobs.filter(job => job.status === 'running');
  const pausedJobs = queueJobs.filter(job => job.status === 'paused');
  const failedJobs = queueJobs.filter(job => job.status === 'failed');
  const completedJobs = queueJobs.filter(job => job.status === 'completed');

  // Handle job click to navigate to Results tab with specific experiment
  const handleJobClick = (job: any) => {
    if (job.experiment_uid) {
      navigateToExperiment(job.experiment_uid);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Queue</h1>
        <p className="text-muted-foreground">
          Manage your experiment execution queue
        </p>
      </div>

      {/* Queue Stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Pending
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{pendingJobs.length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Running
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{runningJobs.length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Paused
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">{pausedJobs.length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Failed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{failedJobs.length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Completed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{completedJobs.length}</div>
          </CardContent>
        </Card>
      </div>

      {/* Queue Items */}
      <div className="grid grid-cols-1 gap-4">
        {/* Running Jobs */}
        {runningJobs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Running Jobs</CardTitle>
              <CardDescription>Currently executing jobs</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {runningJobs.map((job, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800 cursor-pointer hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
                    onClick={() => handleJobClick(job)}
                  >
                    <div>
                      <p className="font-medium">{job.config_path || 'Experiment'}</p>
                      <p className="text-sm text-muted-foreground">
                        Device: {job.assigned_device || 'Unknown'} • Started: {job.started_at || 'Unknown'}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            pauseMutation.mutate(job.id!);
                          }}
                          variant="warning"
                          size="sm"
                        >
                          Pause
                        </Button>
                      <div className="h-2 w-2 bg-blue-600 rounded-full animate-pulse" />
                      <span className="text-sm text-blue-600">Running</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Paused Jobs */}
        {pausedJobs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Paused Jobs</CardTitle>
              <CardDescription>Temporarily paused jobs</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {pausedJobs.map((job, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-950/20 rounded-lg border border-purple-200 dark:border-purple-800 cursor-pointer hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
                    onClick={() => handleJobClick(job)}
                  >
                    <div>
                      <p className="font-medium">{job.config_path || 'Experiment'}</p>
                      <p className="text-sm text-muted-foreground">
                        Paused at: {job.queued_at || 'Unknown'}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            resumeMutation.mutate(job.id!);
                          }}
                          variant="success"
                          size="sm"
                        >
                          Resume
                        </Button>
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            cancelMutation.mutate(job.id!);
                          }}
                          variant="destructive"
                          size="sm"
                        >
                          Cancel
                        </Button>
                      <div className="h-2 w-2 bg-purple-600 rounded-full" />
                      <span className="text-sm text-purple-600">Paused</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Failed Jobs */}
        {failedJobs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Failed Jobs</CardTitle>
              <CardDescription>Failed jobs</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {failedJobs.map((job, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-red-50 dark:bg-red-950/20 rounded-lg border border-red-200 dark:border-red-800 cursor-pointer hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
                    onClick={() => handleJobClick(job)}
                  >
                    <div>
                      <p className="font-medium">{job.config_path || 'Experiment'}</p>
                      <p className="text-sm text-muted-foreground">
                        Failed at: {job.queued_at || 'Unknown'}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            cancelMutation.mutate(job.id!);
                          }}
                          variant="destructive"
                          size="sm"
                        >
                          Remove
                        </Button>
                      <div className="h-2 w-2 bg-red-600 rounded-full" />
                      <span className="text-sm text-red-600">Failed</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Pending Jobs */}
        {pendingJobs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Pending Jobs</CardTitle>
              <CardDescription>Waiting for available resources</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {pendingJobs.map((job, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border cursor-pointer hover:bg-muted transition-colors"
                    onClick={() => handleJobClick(job)}
                  >
                    <div>
                      <p className="font-medium">{job.config_path || 'Experiment'}</p>
                      <p className="text-sm text-muted-foreground">
                        Queued: {job.queued_at || 'Unknown'}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            cancelMutation.mutate(job.id!);
                          }}
                          variant="destructive"
                          size="sm"
                        >
                          Cancel
                        </Button>
                      <div className="h-2 w-2 bg-gray-600 rounded-full" />
                      <span className="text-sm text-gray-600">Pending</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recently Completed */}
        {completedJobs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Recently Completed</CardTitle>
              <CardDescription>Latest finished jobs</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {completedJobs.slice(0, 5).map((job, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800 cursor-pointer hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors"
                    onClick={() => handleJobClick(job)}
                  >
                    <div>
                      <p className="font-medium">{job.config_path || 'Experiment'}</p>
                      <p className="text-sm text-muted-foreground">
                        Completed: {job.completed_at || 'Unknown'}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 bg-green-600 rounded-full" />
                      <span className="text-sm text-green-600">Completed</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Empty State */}
        {queueJobs.length === 0 && !isLoading && (
          <Card>
            <CardContent className="p-12 text-center">
              <p className="text-muted-foreground">No jobs in queue</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}