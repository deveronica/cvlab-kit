/**
 * Server-Sent Events hook for real-time updates
 */

import { useEffect, useRef, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '../lib/react-query';
import { devInfo, devDebug, devError, devWarn } from '../lib/dev-utils';

export interface SSEEvent {
  type: string;
  data?: any;
  timestamp: string;
}

export interface SSEOptions {
  // Automatically reconnect on connection loss
  autoReconnect?: boolean;
  // Reconnect delay in milliseconds
  reconnectDelay?: number;
  // Maximum reconnection attempts
  maxReconnectAttempts?: number;
  // Custom event handlers
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

export function useSSE(options: SSEOptions = {}) {
  const queryClient = useQueryClient();
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);

  const {
    autoReconnect = true,
    reconnectDelay = 3000,
    maxReconnectAttempts = 5,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const handleDeviceUpdate = useCallback((data: any) => {
    // Update device cache with real-time data
    queryClient.setQueryData(queryKeys.devices, (oldDevices: any[] = []) => {
      const updatedDevices = oldDevices.map(device =>
        device.host_id === data.host_id ? { ...device, ...data } : device
      );

      // Add new device if not exists
      if (!updatedDevices.find(d => d.host_id === data.host_id)) {
        updatedDevices.push(data);
      }

      return updatedDevices;
    });
  }, [queryClient]);

  const handleRunUpdate = useCallback((data: any) => {
    // Invalidate projects and runs queries to trigger refetch
    queryClient.invalidateQueries({ queryKey: queryKeys.projects });

    if (data.project) {
      queryClient.invalidateQueries({
        queryKey: queryKeys.runs(data.project)
      });
    }
  }, [queryClient]);

  const handleQueueUpdate = useCallback((data: any) => {
    // Update queue cache
    queryClient.setQueryData(queryKeys.queue, data);
  }, [queryClient]);

  const handleSSEMessage = useCallback((event: MessageEvent) => {
    try {
      const eventData: SSEEvent = JSON.parse(event.data);

      // Handle different event types
      switch (eventData.type) {
        case 'device_update':
          handleDeviceUpdate(eventData.data);
          break;

        case 'run_update':
        case 'file_monitor_update':
          handleRunUpdate(eventData.data);
          break;

        case 'queue_update':
          handleQueueUpdate(eventData.data);
          break;

        case 'heartbeat':
        case 'connection':
          // Connection/heartbeat events - no action needed
          break;

        case 'file_monitor_started':
          devInfo('📁 File monitor started:', eventData.data?.message);
          break;

        default:
          devDebug('Unknown SSE event:', eventData.type, eventData);
      }
    } catch (error) {
      devError('Error parsing SSE message:', error, event.data);
    }
  }, [handleDeviceUpdate, handleRunUpdate, handleQueueUpdate]);

  const connect = useCallback(() => {
    if (eventSourceRef.current?.readyState === EventSource.OPEN) {
      return; // Already connected
    }

    devInfo('🔌 Connecting to SSE...');

    const eventSource = new EventSource('/api/events/stream');
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      devInfo('✅ SSE connected');
      reconnectAttemptRef.current = 0;
      onConnect?.();
    };

    eventSource.onmessage = handleSSEMessage;

    eventSource.onerror = (error) => {
      devError('❌ SSE error:', error);
      onError?.(error);

      if (autoReconnect && reconnectAttemptRef.current < maxReconnectAttempts) {
        reconnectAttemptRef.current += 1;
        devInfo(`🔄 Reconnecting SSE (attempt ${reconnectAttemptRef.current}/${maxReconnectAttempts})...`);

        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, reconnectDelay);
      } else {
        devWarn('🚫 Max reconnection attempts reached');
        onDisconnect?.();
      }
    };
  }, [
    autoReconnect,
    maxReconnectAttempts,
    reconnectDelay,
    handleSSEMessage,
    onConnect,
    onDisconnect,
    onError,
  ]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      devInfo('🔌 SSE disconnected');
      onDisconnect?.();
    }
  }, [onDisconnect]);

  // Connect on mount, disconnect on unmount
  // Dependencies array is intentionally empty to prevent infinite reconnection loop
  // that would occur if connect/disconnect functions are recreated
  useEffect(() => {
    connect();
    return () => disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Return connection control functions
  return {
    connect,
    disconnect,
    isConnected: eventSourceRef.current?.readyState === EventSource.OPEN,
  };
}