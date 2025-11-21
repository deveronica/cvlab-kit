/**
 * Projects and runs API hooks
 */

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api-client';
import { queryKeys } from '../lib/react-query';
import type { Project } from '../lib/types';

export function useProjects() {
  return useQuery({
    queryKey: queryKeys.projects,
    queryFn: () => apiClient.getProjects(),
    // Refetch every 2 minutes as fallback (SSE should update real-time)
    refetchInterval: 2 * 60 * 1000,
  });
}

export function useProject(projectName: string) {
  return useQuery({
    queryKey: queryKeys.project(projectName),
    queryFn: () => apiClient.getProjects().then(projects =>
      projects.find(project => project.name === projectName)
    ),
    enabled: !!projectName,
  });
}

export function useRuns(projectName?: string) {
  return useQuery({
    queryKey: queryKeys.runs(projectName),
    queryFn: () => {
      if (!projectName) {
        // Return all runs from all projects
        return apiClient.getProjects().then(projects =>
          projects.flatMap(project =>
            project.runs.map(run => ({ ...run, project: project.name }))
          )
        );
      }

      // Return runs for specific project
      return apiClient.getProjects().then(projects => {
        const project = projects.find(p => p.name === projectName);
        return project?.runs.map(run => ({ ...run, project: projectName })) || [];
      });
    },
  });
}

export function useRun(projectName: string, runId: string) {
  return useQuery({
    queryKey: queryKeys.run(projectName, runId),
    queryFn: () => apiClient.getProjects().then(projects => {
      const project = projects.find(p => p.name === projectName);
      const run = project?.runs.find(r => r.run_name === runId);
      return run ? { ...run, project: projectName } : null;
    }),
    enabled: !!(projectName && runId),
  });
}

// Derived hooks for project statistics
export function useProjectStats() {
  const { data: projects = [] } = useProjects();

  const stats = projects.reduce((acc, project) => {
    const runs = project.runs;

    acc.totalProjects += 1;
    acc.totalRuns += runs.length;

    // Count by status
    runs.forEach(run => {
      switch (run.status) {
        case 'completed':
          acc.completed += 1;
          break;
        case 'running':
          acc.running += 1;
          break;
        case 'failed':
          acc.failed += 1;
          break;
        default:
          acc.pending += 1;
      }
    });

    return acc;
  }, {
    totalProjects: 0,
    totalRuns: 0,
    completed: 0,
    running: 0,
    failed: 0,
    pending: 0,
  });

  return stats;
}

export function useRecentRuns(limit: number = 10) {
  const { data: projects = [] } = useProjects();

  const allRuns = projects.flatMap(project =>
    project.runs.map(run => ({
      ...run,
      project: project.name,
    }))
  );

  // Sort by started_at descending and take most recent
  const recentRuns = allRuns
    .filter(run => run.started_at)
    .sort((a, b) => new Date(b.started_at!).getTime() - new Date(a.started_at!).getTime())
    .slice(0, limit);

  return { data: recentRuns };
}

// Hook for getting detailed experiment data with hyperparameters
export function useProjectExperiments(projectName: string | null) {
  return useQuery({
    queryKey: queryKeys.projectExperiments(projectName),
    queryFn: () => {
      if (!projectName) return null;
      return apiClient.getProjectExperiments(projectName);
    },
    enabled: !!projectName,
    // Refetch every 30 seconds
    refetchInterval: 30 * 1000,
  });
}