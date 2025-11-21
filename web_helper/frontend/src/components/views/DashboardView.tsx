import React, { memo } from 'react';
import { useProjects, useRuns } from '../../hooks/useProjects';
import { useDevices } from '../../hooks/useDevices';
import { useQueueStats } from '../../hooks/useQueue';
import { Card, CardContent, CardHeader} from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import {
  LayoutDashboard,
  Monitor,
  List,
  BarChart3,
  Activity,
  TrendingUp,
  Cpu
} from 'lucide-react';
import { formatRelativeTime } from '../../lib/time-format';

import { useNavigationStore } from '@/store/navigationStore';
import { useNavigate } from 'react-router-dom';

const DashboardView = memo(function DashboardView() {
  const navigate = useNavigate();
  const { data: projects = [], isLoading: projectsLoading } = useProjects();
  const { data: allRuns = [], isLoading: _runsLoading } = useRuns();
  const { data: devices = [], isLoading: devicesLoading } = useDevices();
  const queueStats = useQueueStats();

  // Calculate statistics with useMemo for performance
  const { totalProjects, totalRuns, _completedRuns, runningRuns, _failedRuns, pendingRuns, activeNodes, totalGPUs } = React.useMemo(() => {
    // Use backend-computed status instead of recalculating on frontend
    const healthyDevices = devices.filter(d => d.status === 'healthy');

    // Total GPUs should only count from healthy nodes
    // Disconnected nodes' GPUs are not available for use
    const totalGpuCount = healthyDevices.reduce((sum, device) => {
      return sum + (device.gpu_count || 0);
    }, 0);

    return {
      totalProjects: projects.length,
      totalRuns: allRuns.length,
      _completedRuns: allRuns.filter(run => run.status === 'completed').length,
      runningRuns: queueStats.running, // Use queue data for accurate job counts
      _failedRuns: allRuns.filter(run => run.status === 'failed').length,
      pendingRuns: queueStats.pending, // Use queue data for accurate job counts
      activeNodes: healthyDevices.length,
      totalGPUs: totalGpuCount, // Sum of GPU counts from healthy nodes only
    };
  }, [projects.length, allRuns, devices, queueStats]);

  // Recent projects (sorted by most recent run)
  const recentProjects = React.useMemo(() => {
    const projectMap = new Map();

    // Group runs by project and find most recent
    allRuns.forEach(run => {
      const existing = projectMap.get(run.project);
      if (!existing || (run.started_at && run.started_at > existing.lastRun)) {
        // Calculate run count dynamically from allRuns
        const runCount = allRuns.filter(r => r.project === run.project).length;
        projectMap.set(run.project, {
          name: run.project,
          lastRun: run.started_at || '',
          runCount: runCount,
        });
      }
    });

    return Array.from(projectMap.values())
      .sort((a, b) => b.lastRun.localeCompare(a.lastRun))
      .slice(0, 5);
  }, [allRuns, projects]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
          <LayoutDashboard className="h-8 w-8" />
          Dashboard
        </h1>
        <p className="text-muted-foreground">
          System overview and navigation hub for ML experimentation platform
        </p>
      </div>

      {/* Stats Grid (4 cards) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Projects Card */}
        <Card
          className="cursor-pointer hover:shadow-md transition-shadow"
          onClick={() => navigate('/projects')}
        >
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-blue-600" />
              <span className="text-sm font-medium">Projects</span>
            </div>
            <div className="text-2xl font-bold mt-1">
              {totalProjects}
            </div>
            <div className="text-xs text-muted-foreground">
              {totalRuns} runs total
            </div>
          </CardContent>
        </Card>

        {/* Monitoring Card */}
        <Card
          className="cursor-pointer hover:shadow-md transition-shadow"
          onClick={() => navigate('/monitoring')}
        >
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Monitor className={`h-4 w-4 ${activeNodes === 0 && totalGPUs > 0 ? 'text-yellow-600' : 'text-green-600'}`} />
              <span className="text-sm font-medium">Monitoring</span>
            </div>
            <div className="text-2xl font-bold mt-1">
              {activeNodes} / {totalGPUs}
            </div>
            <div className="text-xs text-muted-foreground">
              {activeNodes === 0 && totalGPUs > 0 ? '⚠️ No active heartbeats' : 'Active nodes / Total GPUs'}
            </div>
          </CardContent>
        </Card>

        {/* Queue Card */}
        <Card
          className="cursor-pointer hover:shadow-md transition-shadow"
          onClick={() => navigate('/queue')}
        >
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <List className="h-4 w-4 text-orange-600" />
              <span className="text-sm font-medium">Queue</span>
            </div>
            <div className="text-2xl font-bold mt-1">
              {pendingRuns} / {runningRuns}
            </div>
            <div className="text-xs text-muted-foreground">
              Pending jobs / Running jobs
            </div>
          </CardContent>
        </Card>

        {/* Status Card */}
        <Card
          className="cursor-pointer hover:shadow-md transition-shadow"
          onClick={() => {
            // Refresh connection status
            window.location.reload();
          }}
        >
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-emerald-600" />
              <span className="text-sm font-medium">Status</span>
            </div>
            <div className="text-2xl font-bold mt-1 text-emerald-600">Online</div>
            <div className="text-xs text-muted-foreground">
              Backend connection status
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity Grid (2 cards) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Projects */}
        <Card>
          <CardHeader>
            <h2 className="text-2xl font-semibold leading-none tracking-tight flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Recent Projects
            </h2>
          </CardHeader>
          <CardContent>
            {projectsLoading ? (
              <div className="space-y-3">
                {[1, 2, 3].map(i => (
                  <div key={i} className="animate-pulse">
                    <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                    <div className="h-3 bg-muted rounded w-1/2" />
                  </div>
                ))}
              </div>
            ) : recentProjects.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                No projects found
              </div>
            ) : (
              <div className="space-y-3">
                {recentProjects.map(project => (
                  <div
                    key={project.name}
                    className="flex items-center justify-between p-3 border rounded-lg cursor-pointer hover:bg-muted/50 transition-colors"
                    onClick={() => navigate(`/projects/${project.name}`)}
                  >
                    <div>
                      <div className="font-medium">{project.name}</div>
                      <p className="text-sm text-muted-foreground">
                        Last activity: {formatRelativeTime(project.lastRun)}
                      </p>
                    </div>
                    <Badge variant="secondary">{project.runCount} runs</Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* System Resources */}
        <Card>
          <CardHeader>
            <h2 className="text-2xl font-semibold leading-none tracking-tight flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              System Resources
            </h2>
          </CardHeader>
          <CardContent>
            {devicesLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {[1, 2, 3].map(i => (
                  <div key={i} className="animate-pulse border rounded-lg p-3">
                    <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                    <div className="h-3 bg-muted rounded w-1/2" />
                  </div>
                ))}
              </div>
            ) : devices.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                No devices found
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {devices.map(device => (
                  <div
                    key={device.host_id || device.id}
                    className="border rounded-lg p-4 space-y-3 hover:border-primary/50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-sm truncate pr-2">{device.host_id || device.name || device.id}</span>
                      <Badge
                        variant={device.status === 'healthy' ? 'default' : 'secondary'}
                        className={
                          device.status === 'healthy' ? 'bg-green-100 text-green-800 flex-shrink-0' :
                          device.status === 'stale' ? 'bg-yellow-100 text-yellow-800 flex-shrink-0' :
                          'bg-red-100 text-red-800 flex-shrink-0'
                        }
                      >
                        {device.status}
                      </Badge>
                    </div>

                    {device.status === 'disconnected' ? (
                      <div className="text-xs text-muted-foreground text-center py-2">
                        Device disconnected
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {/* CPU & RAM (우선순위) */}
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-xs text-muted-foreground">CPU</span>
                            <span className="text-xs font-medium">
                              {device.cpu_util !== null && device.cpu_util !== undefined ? `${device.cpu_util.toFixed(1)}%` : 'N/A'}
                            </span>
                          </div>
                          <div className="w-full bg-muted rounded-full h-1.5 mb-1">
                            <div
                              className="bg-green-600 h-1.5 rounded-full transition-all duration-300"
                              style={{ width: `${device.cpu_util || 0}%` }}
                            />
                          </div>
                          <div className="text-right text-[10px] text-muted-foreground">
                            {device.memory_used !== null && device.memory_total !== null
                              ? `${device.memory_used.toFixed(1)}GB / ${device.memory_total.toFixed(1)}GB RAM`
                              : 'RAM N/A'}
                          </div>
                        </div>

                        {/* GPU & VRAM (있을 때만) */}
                        {(device.gpu_util !== null || device.vram_used !== null) && (
                          <div>
                            <div className="flex justify-between items-center mb-1">
                              <span className="text-xs text-muted-foreground">
                                GPU
                                {device.gpu_count && device.gpu_count > 1 && device.gpus && device.gpus.length > 0 && (
                                  <span className="ml-1 text-[10px]">
                                    ({device.gpu_count}x {device.gpus[0].name.split(' ').slice(-1)[0]})
                                  </span>
                                )}
                              </span>
                              <span className="text-xs font-medium">
                                {device.gpu_util !== null && device.gpu_util !== undefined ? `${device.gpu_util.toFixed(1)}%` : 'N/A'}
                              </span>
                            </div>
                            <div className="w-full bg-muted rounded-full h-1.5 mb-1">
                              <div
                                className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                                style={{ width: `${device.gpu_util || 0}%` }}
                              />
                            </div>
                            <div className="text-right text-[10px] text-muted-foreground">
                              {device.vram_used !== null && device.vram_total !== null
                                ? `${device.vram_used.toFixed(1)}GB / ${device.vram_total.toFixed(1)}GB VRAM${device.gpu_count && device.gpu_count > 1 ? ' (total)' : ''}`
                                : 'VRAM N/A'}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

    </div>
  );
});

export { DashboardView };