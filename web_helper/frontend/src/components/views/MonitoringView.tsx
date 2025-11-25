import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { useDevices } from '../../hooks/useDevices';
import {
  AlertCircle,
  Terminal,
  Cpu,
  MemoryStick,
  Gauge,
  Zap,
  Thermometer,
  HardDrive,
  Activity,
  Trash2,
  Loader2
} from 'lucide-react';
import { formatRelativeTime } from '../../lib/time-format';
import { useQueryClient } from '@tanstack/react-query';

// Enhanced progress bar component with gradient and animation
const MetricBar = ({
  label,
  value,
  total,
  icon: Icon,
  color = 'blue',
  showPercentage = true
}: {
  label: string;
  value: number;
  total?: number;
  icon: React.ComponentType<any>;
  color?: string;
  showPercentage?: boolean;
}) => {
  const safeValue = value ?? 0;
  const safeTotal = total ?? 0;
  const percentage = safeTotal ? (safeValue / safeTotal) * 100 : safeValue;
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    purple: 'from-purple-500 to-purple-600',
    green: 'from-green-500 to-green-600',
    orange: 'from-orange-500 to-orange-600',
    red: 'from-red-500 to-red-600',
  }[color] || 'from-blue-500 to-blue-600';

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{label}</span>
        </div>
        <span className="text-sm font-semibold text-foreground">
          {showPercentage ? `${percentage.toFixed(1)}%` : `${safeValue.toFixed(1)}GB`}
          {safeTotal > 0 && <span className="text-muted-foreground font-normal"> / {safeTotal.toFixed(1)}GB</span>}
        </span>
      </div>
      <div className="relative w-full bg-secondary/50 rounded-full h-2.5 overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 bg-gradient-to-r ${colorClasses} rounded-full transition-all duration-500 ease-out shadow-sm`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 animate-pulse" />
        </div>
      </div>
    </div>
  );
};

// Status badge component
const StatusBadge = ({ status }: { status: string }) => {
  const variants = {
    healthy: { variant: 'default' as const, className: 'bg-green-500/10 text-green-700 dark:text-green-400 border-green-500/20', label: 'Healthy' },
    stale: { variant: 'secondary' as const, className: 'bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border-yellow-500/20', label: 'Stale' },
    disconnected: { variant: 'destructive' as const, className: 'bg-red-500/10 text-red-700 dark:text-red-400 border-red-500/20', label: 'Disconnected' },
  }[status] || { variant: 'secondary' as const, className: '', label: status };

  return (
    <Badge variant={variants.variant} className={`${variants.className} font-medium`}>
      <div className={`mr-1.5 h-2 w-2 rounded-full ${
        status === 'healthy' ? 'bg-green-500 animate-pulse' :
        status === 'stale' ? 'bg-yellow-500' :
        'bg-red-500'
      }`} />
      {variants.label}
    </Badge>
  );
};

export function MonitoringView() {
  const { data: devices = [], isLoading } = useDevices();
  const queryClient = useQueryClient();
  const [deletingDevice, setDeletingDevice] = useState<string | null>(null);

  // Check if there are devices but none are sending heartbeats (all stale/disconnected)
  // Use backend-computed status instead of recalculating on frontend
  const hasActiveDevices = devices.some(d => d.status === 'healthy');

  const handleDeleteDevice = async (hostId: string) => {
    if (!confirm(`Delete device "${hostId}"? This action cannot be undone.`)) {
      return;
    }

    setDeletingDevice(hostId);
    try {
      const response = await fetch(`/api/devices/${encodeURIComponent(hostId)}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        alert(error.detail?.detail || 'Failed to delete device');
        return;
      }

      // Invalidate devices query to refresh the list
      queryClient.invalidateQueries({ queryKey: ['devices'] });
    } catch (error) {
      console.error('Failed to delete device:', error);
      alert('Failed to delete device');
    } finally {
      setDeletingDevice(null);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Devices</h1>
        <p className="text-muted-foreground">
          Monitor and manage your compute resources
        </p>
      </div>

      {/* Warning banner when devices exist but no active heartbeats */}
      {!isLoading && devices.length > 0 && !hasActiveDevices && (
        <Card className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
              <div className="flex-1 space-y-2">
                <p className="text-sm font-medium text-yellow-900 dark:text-yellow-200">
                  No active heartbeats detected
                </p>
                <p className="text-xs text-yellow-800 dark:text-yellow-300">
                  Devices are registered but not sending heartbeats. Make sure the server is running in dev mode with the integrated client agent:
                </p>
                <div className="bg-yellow-100 dark:bg-yellow-900/30 rounded p-2 font-mono text-xs flex items-start gap-2">
                  <Terminal className="h-3 w-3 mt-0.5 flex-shrink-0" />
                  <code>uv run app.py --dev</code>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {isLoading ? (
          [...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="p-6">
                <div className="animate-pulse space-y-2">
                  <div className="h-4 bg-muted rounded w-3/4" />
                  <div className="h-6 bg-muted rounded w-1/2" />
                  <div className="h-3 bg-muted rounded w-full" />
                </div>
              </CardContent>
            </Card>
          ))
        ) : devices.length > 0 ? (
          devices.map((device) => {
            // Multi-GPU device: render single card with all GPUs inside
            if (device.gpu_count && device.gpu_count > 1 && device.gpus && device.gpus.length > 0) {
              return (
                <Card
                  key={device.host_id}
                  className="hover:shadow-lg transition-all duration-300 border-l-4 border-l-purple-500"
                >
                  <CardHeader className="pb-4">
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <CardTitle className="text-xl font-bold flex items-center gap-2">
                          <Activity className="h-5 w-5 text-purple-500" />
                          {device.host_id}
                        </CardTitle>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="font-mono text-xs">
                            {device.gpu_count} GPUs
                          </Badge>
                          <StatusBadge status={device.status} />
                        </div>
                      </div>
                      {device.status === 'disconnected' && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-muted-foreground hover:text-destructive"
                          onClick={() => handleDeleteDevice(device.host_id)}
                          disabled={deletingDevice === device.host_id}
                        >
                          {deletingDevice === device.host_id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      )}
                    </div>
                    <CardDescription className="text-sm pt-2">
                      <span className="block text-xs text-muted-foreground">
                        Last seen: {formatRelativeTime(device.last_heartbeat)}
                      </span>
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* CPU & RAM */}
                    {device.cpu_util != null && (
                      <MetricBar
                        label="CPU Utilization"
                        value={device.cpu_util}
                        icon={Cpu}
                        color="green"
                      />
                    )}
                    {device.memory_used != null && device.memory_total != null && (
                      <MetricBar
                        label="RAM"
                        value={device.memory_used}
                        total={device.memory_total}
                        icon={MemoryStick}
                        color="blue"
                        showPercentage={false}
                      />
                    )}

                    {/* GPU Sections */}
                    <div className="border-t pt-4 space-y-6">
                      {device.gpus.map((gpu) => (
                        <div key={gpu.id} className="space-y-4">
                          <div className="flex items-center gap-2 mb-3">
                            <Badge variant="outline" className="font-mono text-xs bg-purple-50 dark:bg-purple-950/30">
                              GPU {gpu.id}
                            </Badge>
                            <span className="text-sm text-muted-foreground">{gpu.name}</span>
                          </div>

                          <MetricBar
                            label="GPU Utilization"
                            value={gpu.util}
                            icon={Gauge}
                            color="purple"
                          />

                          <MetricBar
                            label="VRAM"
                            value={gpu.vram_used}
                            total={gpu.vram_total}
                            icon={MemoryStick}
                            color="orange"
                            showPercentage={false}
                          />

                          <div className="grid grid-cols-2 gap-3">
                            {gpu.temperature != null && (
                              <div className="bg-secondary/30 rounded-lg p-3 space-y-1">
                                <div className="flex items-center gap-1.5 text-muted-foreground">
                                  <Thermometer className="h-3.5 w-3.5" />
                                  <span className="text-xs font-medium">Temperature</span>
                                </div>
                                <div className="text-2xl font-bold">
                                  {gpu.temperature}
                                  <span className="text-sm text-muted-foreground">°C</span>
                                </div>
                              </div>
                            )}
                            {gpu.power_usage != null && (
                              <div className="bg-secondary/30 rounded-lg p-3 space-y-1">
                                <div className="flex items-center gap-1.5 text-muted-foreground">
                                  <Zap className="h-3.5 w-3.5" />
                                  <span className="text-xs font-medium">Power</span>
                                </div>
                                <div className="text-2xl font-bold">
                                  {gpu.power_usage.toFixed(0)}
                                  <span className="text-sm text-muted-foreground">W</span>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              );
            }

            // Single GPU or legacy device: render single card
            return (
              <Card
                key={device.id || device.host_id}
                className="hover:shadow-lg transition-all duration-300 border-l-4 border-l-blue-500"
              >
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <CardTitle className="text-xl font-bold flex items-center gap-2">
                        <HardDrive className="h-5 w-5 text-blue-500" />
                        {device.name || device.id || device.host_id}
                      </CardTitle>
                      <StatusBadge status={device.status} />
                    </div>
                    {device.status === 'disconnected' && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-muted-foreground hover:text-destructive"
                        onClick={() => handleDeleteDevice(device.host_id)}
                        disabled={deletingDevice === device.host_id}
                      >
                        {deletingDevice === device.host_id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                      </Button>
                    )}
                  </div>
                  <CardDescription className="text-sm pt-2">
                    Last seen: {formatRelativeTime(device.last_heartbeat)}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* CPU Utilization */}
                  {device.cpu_util != null && (
                    <MetricBar
                      label="CPU Utilization"
                      value={device.cpu_util}
                      icon={Cpu}
                      color="green"
                    />
                  )}

                  {/* RAM Usage */}
                  {device.memory_used != null && device.memory_total != null && (
                    <MetricBar
                      label="RAM"
                      value={device.memory_used}
                      total={device.memory_total}
                      icon={MemoryStick}
                      color="blue"
                      showPercentage={false}
                    />
                  )}

                  {/* GPU Section - Show if GPU exists */}
                  {(device.gpu_count && device.gpu_count > 0 && device.gpus && device.gpus.length > 0) && (
                    <>
                      <div className="border-t pt-4">
                        <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                          <Activity className="h-4 w-4" />
                          GPU Metrics
                        </h4>
                        <div className="space-y-4">
                          {/* Use first GPU from gpus array if available, otherwise legacy fields */}
                          {device.gpus && device.gpus.length > 0 ? (
                            <>
                              <MetricBar
                                label="GPU Utilization"
                                value={device.gpus[0].util}
                                icon={Gauge}
                                color="purple"
                              />
                              <MetricBar
                                label="VRAM"
                                value={device.gpus[0].vram_used}
                                total={device.gpus[0].vram_total}
                                icon={MemoryStick}
                                color="orange"
                                showPercentage={false}
                              />
                            </>
                          ) : (
                            <>
                              <MetricBar
                                label="GPU Utilization"
                                value={device.gpu_util || 0}
                                icon={Gauge}
                                color="purple"
                              />
                              {device.vram_used != null && device.vram_total != null && (
                                <MetricBar
                                  label="VRAM"
                                  value={device.vram_used}
                                  total={device.vram_total}
                                  icon={MemoryStick}
                                  color="orange"
                                  showPercentage={false}
                                />
                              )}
                            </>
                          )}
                        </div>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            );
          })
        ) : (
          <div className="col-span-full">
            <Card>
              <CardContent className="p-12 text-center">
                <p className="text-muted-foreground">No devices found</p>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}