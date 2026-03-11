/**
 * Server-Sent Events hook for real-time updates
 */

import { useEffect, useRef, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/shared/api/react-query';
import { devInfo, devDebug, devError, devWarn } from '@/shared/lib/utils';

// SSE event data types
interface DeviceUpdateData {
  host_id: string;
  [key: string]: unknown;
}

interface RunUpdateData {
  project?: string;
  run_name?: string;
  [key: string]: unknown;
}

interface QueueUpdateData {
  action?: string;
  experiment_uid?: string;
  [key: string]: unknown;
}

export interface SSEEvent {
  type: string;
  data?: DeviceUpdateData | RunUpdateData | QueueUpdateData | unknown;
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

  const handleDeviceUpdate = useCallback((_data: DeviceUpdateData) => {
    // Device updates may be partial; refetch the canonical shape.
    queryClient.invalidateQueries({ queryKey: queryKeys.devices });
  }, [queryClient]);

  const handleRunUpdate = useCallback((data: RunUpdateData) => {
    // Invalidate projects and runs queries to trigger refetch
    queryClient.invalidateQueries({ queryKey: queryKeys.projects });

    if (data.project) {
      queryClient.invalidateQueries({
        queryKey: queryKeys.runs(data.project)
      });
    }
  }, [queryClient]);

  const handleQueueUpdate = useCallback((data: QueueUpdateData) => {
    // Invalidate queue queries to trigger refetch for real-time updates
    devDebug('Queue update received:', data?.action, data?.experiment_uid);
    queryClient.invalidateQueries({ queryKey: queryKeys.queue });
  }, [queryClient]);

  function isDeviceUpdateData(value: unknown): value is DeviceUpdateData {
    return !!value && typeof value === 'object' && 'host_id' in value && typeof (value as { host_id?: unknown }).host_id === 'string';
  }

  function isRunUpdateData(value: unknown): value is RunUpdateData {
    return !!value && typeof value === 'object';
  }

  function isQueueUpdateData(value: unknown): value is QueueUpdateData {
    return !!value && typeof value === 'object';
  }

  const handleSSEMessage = useCallback((event: MessageEvent) => {
    try {
      const eventData: SSEEvent = JSON.parse(event.data);

      // Handle different event types
      switch (eventData.type) {
        case 'device_update':
          if (isDeviceUpdateData(eventData.data)) {
            handleDeviceUpdate(eventData.data);
          }
          break;

        case 'run_update':
        case 'file_monitor_update':
          if (isRunUpdateData(eventData.data)) {
            handleRunUpdate(eventData.data);
          }
          break;

        case 'queue_update':
          if (isQueueUpdateData(eventData.data)) {
            handleQueueUpdate(eventData.data);
          }
          break;

        case 'heartbeat':
        case 'connection':
          // Connection/heartbeat events - no action needed
          break;

        case 'file_monitor_started':
          if (
            eventData.data &&
            typeof eventData.data === 'object' &&
            'message' in eventData.data
          ) {
            devInfo(
              '📁 File monitor started:',
              (eventData.data as { message?: unknown }).message
            );
          } else {
            devInfo('📁 File monitor started');
          }
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
  useEffect(() => {
    connect();

    // Cleanup: disconnect when component unmounts
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Return connection control functions
  return {
    connect,
    disconnect,
    isConnected: eventSourceRef.current?.readyState === EventSource.OPEN,
  };
}
