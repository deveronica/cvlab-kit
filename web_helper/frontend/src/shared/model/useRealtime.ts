/**
 * Dual-stack real-time communication hook.
 *
 * Tries WebSocket first, falls back to SSE if unavailable.
 * Provides unified interface for real-time updates across the application.
 */

import { useEffect, useRef, useCallback, useState, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/shared/api/react-query';
import { devInfo, devDebug, devError, devWarn } from '@/shared/lib/utils';

// Simple debounce utility
function debounce<T extends (...args: unknown[]) => void>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      func(...args);
      timeoutId = null;
    }, wait);
  };
}

// Event types from backend
export interface RealtimeEvent {
  type: 'connection' | 'device_update' | 'queue_update' | 'run_update' | 'node_update' | 'heartbeat' | 'pong' | 'subscribed' | 'unsubscribed' | 'status' | 'error';
  data?: unknown;
  message?: string;
  channel?: string;
  timestamp: string;
}

// Available channels for subscription
export type RealtimeChannel = 'all' | 'queue' | 'devices' | 'runs' | 'nodes';

// Transport type
export type TransportType = 'websocket' | 'sse' | 'none';

export interface RealtimeOptions {
  // Channels to subscribe to (WebSocket only)
  channels?: RealtimeChannel[];
  // Prefer SSE over WebSocket (useful for debugging)
  preferSSE?: boolean;
  // Auto reconnect on connection loss
  autoReconnect?: boolean;
  // Reconnect delay in milliseconds
  reconnectDelay?: number;
  // Maximum reconnection attempts before fallback
  maxReconnectAttempts?: number;
  // WebSocket endpoint
  wsUrl?: string;
  // SSE endpoint
  sseUrl?: string;
  // Debounce delay for node updates (default: 500ms)
  nodeUpdateDebounce?: number;
  // Custom event handlers
  onConnect?: (transport: TransportType) => void;
  onDisconnect?: (transport: TransportType) => void;
  onMessage?: (event: RealtimeEvent) => void;
  onError?: (error: Event | Error) => void;
  onTransportChange?: (from: TransportType, to: TransportType) => void;
}

export interface RealtimeState {
  isConnected: boolean;
  transport: TransportType;
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error' | 'fallback';
  reconnectAttempt: number;
}

function isBrowserEvent(error: unknown): error is Event {
  return typeof Event !== 'undefined' && error instanceof Event;
}

export function useRealtime(options: RealtimeOptions = {}) {
  const queryClient = useQueryClient();

  const {
    channels = ['all'],
    preferSSE = false,
    autoReconnect = true,
    reconnectDelay = 3000,
    maxReconnectAttempts = 3,
    wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/ws`,
    sseUrl = '/api/events/stream',
    nodeUpdateDebounce = 500,
    onConnect,
    onDisconnect,
    onMessage,
    onError,
    onTransportChange,
  } = options;

  // State
  const [state, setState] = useState<RealtimeState>({
    isConnected: false,
    transport: 'none',
    connectionState: 'disconnected',
    reconnectAttempt: 0,
  });

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const sseRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const connectStartTimeoutRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  const currentTransportRef = useRef<TransportType>('none');

  // Store callbacks in refs to avoid reconnection on option changes
  const callbacksRef = useRef({ onConnect, onDisconnect, onMessage, onError, onTransportChange });
  useEffect(() => {
    callbacksRef.current = { onConnect, onDisconnect, onMessage, onError, onTransportChange };
  }, [onConnect, onDisconnect, onMessage, onError, onTransportChange]);

  // Query cache update handlers
  const handleDeviceUpdate = useCallback((data: { host_id: string; [key: string]: unknown }) => {
    // 항목 1: CPU 데이터 즉시 갱신을 위해 setQueryData 사용
    queryClient.setQueryData(queryKeys.devices, (old: any[] | undefined) => {
      if (!old) return old;
      return old.map(d => d.host_id === data.host_id ? { ...d, ...data, status: 'healthy', last_heartbeat: new Date().toISOString() } : d);
    });
    // canonical shape를 보장하기 위해 여전히 무효화는 수행 (백그라운드 리페치)
    queryClient.invalidateQueries({ queryKey: queryKeys.devices, refetchType: 'none' });
  }, [queryClient]);

  const handleRunUpdate = useCallback((data: { project?: string; [key: string]: unknown }) => {
    queryClient.invalidateQueries({ queryKey: queryKeys.projects });
    if (data.project) {
      queryClient.invalidateQueries({ queryKey: queryKeys.runs(data.project) });
    }
  }, [queryClient]);

  const handleQueueUpdate = useCallback((data: { action?: string; experiment_uid?: string; [key: string]: unknown }) => {
    devDebug('Queue update received:', data?.action, data?.experiment_uid);
    queryClient.invalidateQueries({ queryKey: queryKeys.queue });
  }, [queryClient]);

  // Debounced node update handler to prevent rapid-fire invalidations
  const debouncedNodeInvalidate = useMemo(
    () => debounce(() => {
      queryClient.invalidateQueries({ queryKey: ['node_graph'] });
    }, nodeUpdateDebounce),
    [queryClient, nodeUpdateDebounce]
  );

  const handleNodeUpdate = useCallback((data: unknown) => {
    devDebug('Node update received:', data);
    // Debounce node graph query invalidation to prevent UI flicker
    debouncedNodeInvalidate();
  }, [debouncedNodeInvalidate]);

  // Process incoming event
  const processEvent = useCallback((event: RealtimeEvent) => {
    callbacksRef.current.onMessage?.(event);

    switch (event.type) {
      case 'device_update':
        handleDeviceUpdate(event.data as { host_id: string });
        break;
      case 'run_update':
        handleRunUpdate(event.data as { project?: string });
        break;
      case 'queue_update':
        handleQueueUpdate(event.data as { action?: string; experiment_uid?: string });
        break;
      case 'node_update':
        handleNodeUpdate(event.data);
        break;
      case 'connection':
        devInfo('Connection confirmed:', event.message);
        break;
      case 'heartbeat':
      case 'pong':
        devDebug('Heartbeat/pong received');
        break;
      case 'subscribed':
        devDebug('Subscribed to channel:', event.channel);
        break;
      case 'unsubscribed':
        devDebug('Unsubscribed from channel:', event.channel);
        break;
      case 'error':
        devError('Server error:', event.message);
        break;
      default:
        devDebug('Unknown event type:', event.type, event);
    }
  }, [handleDeviceUpdate, handleRunUpdate, handleQueueUpdate, handleNodeUpdate]);

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (!mountedRef.current) return;
    if (connectStartTimeoutRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    setState(prev => ({ ...prev, connectionState: 'connecting' }));
    devInfo('Connecting WebSocket...');

    try {
      // Defer actual socket creation by one tick.
      // This avoids noisy browser warnings in React 18 StrictMode where effects
      // mount/unmount back-to-back in dev (close before the connection establishes).
      connectStartTimeoutRef.current = window.setTimeout(() => {
        connectStartTimeoutRef.current = null;
        if (!mountedRef.current) return;
        if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
          return;
        }

        wsRef.current = new WebSocket(wsUrl);

        wsRef.current.onopen = () => {
          if (!mountedRef.current) return;

        currentTransportRef.current = 'websocket';
        setState({
          isConnected: true,
          transport: 'websocket',
          connectionState: 'connected',
          reconnectAttempt: 0,
        });

        devInfo('WebSocket connected');
        callbacksRef.current.onConnect?.('websocket');

        // Subscribe to channels
        channels.forEach(channel => {
          if (channel !== 'all') {
            wsRef.current?.send(JSON.stringify({ type: 'subscribe', channel }));
          }
        });
        };

        wsRef.current.onmessage = (event) => {
          if (!mountedRef.current) return;
        try {
          const data: RealtimeEvent = JSON.parse(event.data);
          processEvent(data);
        } catch (error) {
          devError('Failed to parse WebSocket message:', error);
        }
        };

        wsRef.current.onclose = () => {
          if (!mountedRef.current) return;

        devInfo('WebSocket disconnected');
        callbacksRef.current.onDisconnect?.('websocket');

        // Attempt reconnect or fallback to SSE
        if (autoReconnect) {
          setState(prev => {
            const newAttempt = prev.reconnectAttempt + 1;
            if (newAttempt < maxReconnectAttempts) {
              devInfo(`Reconnecting WebSocket (${newAttempt}/${maxReconnectAttempts})...`);
              reconnectTimeoutRef.current = window.setTimeout(connectWebSocket, reconnectDelay);
              return { ...prev, isConnected: false, connectionState: 'connecting', reconnectAttempt: newAttempt };
            } else {
              // Fall back to SSE
              devWarn('WebSocket failed, falling back to SSE');
              callbacksRef.current.onTransportChange?.('websocket', 'sse');
              setTimeout(connectSSE, 100);
              return { ...prev, isConnected: false, connectionState: 'fallback', reconnectAttempt: 0 };
            }
          });
        } else {
          setState(prev => ({ ...prev, isConnected: false, transport: 'none', connectionState: 'disconnected' }));
        }
        };

        wsRef.current.onerror = (error) => {
          if (!mountedRef.current) return;
          if (isBrowserEvent(error)) {
            devWarn('WebSocket transport event:', error.type);
            return;
          }
          devError('WebSocket error:', error);
          callbacksRef.current.onError?.(error);
        };
      }, 0);

    } catch (error) {
      devError('Failed to create WebSocket:', error);
      setState(prev => ({ ...prev, connectionState: 'error' }));
      // Try SSE fallback
      if (autoReconnect) {
        setTimeout(connectSSE, 100);
      }
    }
  }, [wsUrl, channels, autoReconnect, maxReconnectAttempts, reconnectDelay, processEvent]);

  // SSE connection
  const connectSSE = useCallback(() => {
    if (!mountedRef.current) return;
    if (sseRef.current?.readyState === EventSource.OPEN || sseRef.current?.readyState === EventSource.CONNECTING) {
      return;
    }

    setState(prev => ({ ...prev, connectionState: 'connecting' }));
    devInfo('Connecting SSE...');

    try {
      sseRef.current = new EventSource(sseUrl);

      sseRef.current.onopen = () => {
        if (!mountedRef.current) return;

        currentTransportRef.current = 'sse';
        setState({
          isConnected: true,
          transport: 'sse',
          connectionState: 'connected',
          reconnectAttempt: 0,
        });

        devInfo('SSE connected');
        callbacksRef.current.onConnect?.('sse');
      };

      sseRef.current.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const data: RealtimeEvent = JSON.parse(event.data);
          processEvent(data);
        } catch (error) {
          devError('Failed to parse SSE message:', error);
        }
      };

      sseRef.current.onerror = (error) => {
        if (!mountedRef.current) return;
        if (isBrowserEvent(error)) {
          devWarn('SSE transport event:', error.type);
        } else {
          devError('SSE error:', error);
          callbacksRef.current.onError?.(error);
        }

        if (autoReconnect) {
          setState(prev => {
            const newAttempt = prev.reconnectAttempt + 1;
            if (newAttempt < maxReconnectAttempts) {
              devInfo(`Reconnecting SSE (${newAttempt}/${maxReconnectAttempts})...`);
              reconnectTimeoutRef.current = window.setTimeout(connectSSE, reconnectDelay);
              return { ...prev, isConnected: false, connectionState: 'connecting', reconnectAttempt: newAttempt };
            } else {
              devWarn('SSE reconnection failed');
              callbacksRef.current.onDisconnect?.('sse');
              return { ...prev, isConnected: false, transport: 'none', connectionState: 'disconnected' };
            }
          });
        } else {
          setState(prev => ({ ...prev, isConnected: false, transport: 'none', connectionState: 'disconnected' }));
        }
      };

    } catch (error) {
      devError('Failed to create SSE:', error);
      setState(prev => ({ ...prev, connectionState: 'error' }));
    }
  }, [sseUrl, autoReconnect, maxReconnectAttempts, reconnectDelay, processEvent]);

  // Main connect function
  const connect = useCallback(() => {
    if (preferSSE) {
      connectSSE();
    } else {
      connectWebSocket();
    }
  }, [preferSSE, connectSSE, connectWebSocket]);

  // Disconnect function
  const disconnect = useCallback(() => {
    if (connectStartTimeoutRef.current) {
      clearTimeout(connectStartTimeoutRef.current);
      connectStartTimeoutRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }

    currentTransportRef.current = 'none';
    setState({
      isConnected: false,
      transport: 'none',
      connectionState: 'disconnected',
      reconnectAttempt: 0,
    });
  }, []);

  // Subscribe to a channel (WebSocket only)
  const subscribe = useCallback((channel: RealtimeChannel) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'subscribe', channel }));
    }
  }, []);

  // Unsubscribe from a channel (WebSocket only)
  const unsubscribe = useCallback((channel: RealtimeChannel) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'unsubscribe', channel }));
    }
  }, []);

  // Send a ping (WebSocket only)
  const ping = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
    }
  }, []);

  // Connect on mount
  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Memoize return value
  return useMemo(() => ({
    ...state,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    ping,
  }), [state, connect, disconnect, subscribe, unsubscribe, ping]);
}
