/**
 * Devices API hooks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../lib/api-client';
import { queryKeys } from '../lib/react-query';
import type { Device } from '../lib/types';
import { devError } from '../lib/dev-utils';

export function useDevices() {
  return useQuery({
    queryKey: queryKeys.devices,
    queryFn: () => apiClient.getDevices(),
    // Refetch every 5 seconds for real-time status updates
    refetchInterval: 5 * 1000,
  });
}

export function useDevice(hostId: string) {
  return useQuery({
    queryKey: queryKeys.device(hostId),
    queryFn: () => apiClient.getDevices().then(devices =>
      devices.find(device => device.host_id === hostId)
    ),
    enabled: !!hostId,
  });
}

export function useHeartbeatMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (heartbeatData: any) => apiClient.sendHeartbeat(heartbeatData),
    onSuccess: (data, variables) => {
      // Optimistically update the device in cache
      queryClient.setQueryData(queryKeys.devices, (oldDevices: Device[] = []) => {
        return oldDevices.map(device =>
          device.host_id === data.host_id
            ? { ...device, ...variables, status: 'online', last_heartbeat: new Date().toISOString() }
            : device
        );
      });
    },
    onError: (error) => {
      devError('Failed to send heartbeat:', error);
    },
  });
}

// Derived hooks for device status
export function useDevicesByStatus() {
  const { data: devices = [] } = useDevices();

  const now = Date.now();
  const categorized = devices.reduce((acc, device) => {
    const lastHeartbeat = device.last_heartbeat ? new Date(device.last_heartbeat).getTime() : 0;
    const timeDiff = now - lastHeartbeat;

    if (timeDiff <= 3000) {
      acc.healthy.push(device);
    } else if (timeDiff <= 60000) {
      acc.stale.push(device);
    } else {
      acc.disconnected.push(device);
    }

    return acc;
  }, {
    healthy: [] as Device[],
    stale: [] as Device[],
    disconnected: [] as Device[],
  });

  return categorized;
}

export function useDeviceStats() {
  const { data: devices = [] } = useDevices();

  const stats = devices.reduce((acc, device) => {
    // GPU utilization
    if (device.gpu_util !== null && device.gpu_util !== undefined) {
      acc.gpu.total += device.gpu_util;
      acc.gpu.count += 1;
    }

    // CPU utilization
    if (device.cpu_util !== null) {
      acc.cpu.total += device.cpu_util;
      acc.cpu.count += 1;
    }

    // Memory utilization
    if (device.memory_used !== null && device.memory_total !== null) {
      acc.memory.used += device.memory_used;
      acc.memory.total += device.memory_total;
    }

    // VRAM utilization
    if (device.vram_used !== null && device.vram_total !== null) {
      acc.vram.used += device.vram_used;
      acc.vram.total += device.vram_total;
    }

    return acc;
  }, {
    gpu: { total: 0, count: 0 },
    cpu: { total: 0, count: 0 },
    memory: { used: 0, total: 0 },
    vram: { used: 0, total: 0 },
  });

  return {
    gpuAverage: stats.gpu.count > 0 ? stats.gpu.total / stats.gpu.count : 0,
    cpuAverage: stats.cpu.count > 0 ? stats.cpu.total / stats.cpu.count : 0,
    memoryUsage: stats.memory.total > 0 ? (stats.memory.used / stats.memory.total) * 100 : 0,
    vramUsage: stats.vram.total > 0 ? (stats.vram.used / stats.vram.total) * 100 : 0,
  };
}