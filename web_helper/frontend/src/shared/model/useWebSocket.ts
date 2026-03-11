import { useState, useEffect, useRef, useCallback } from 'react';
import { devInfo, devError, devWarn } from '@/shared/lib/utils';

interface WebSocketMessage {
  type: 'metrics' | 'experiment_update' | 'system_alert' | 'device_status' | 'queue_update';
  data: unknown;
  timestamp: string;
}

interface WebSocketOptions {
  url?: string;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

export function useWebSocket(options: WebSocketOptions = {}) {
  const {
    url = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws`,
    autoReconnect = true,
    maxReconnectAttempts = 5,
    reconnectInterval = 3000,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [messageHistory, setMessageHistory] = useState<WebSocketMessage[]>([]);

  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const mountedRef = useRef(true);

  // Store options in refs to avoid reconnection on option changes
  const optionsRef = useRef({
    url,
    autoReconnect,
    maxReconnectAttempts,
    reconnectInterval,
  });
  const callbacksRef = useRef({ onMessage, onConnect, onDisconnect, onError });

  // Update refs when options change (without triggering reconnect)
  useEffect(() => {
    optionsRef.current = { url, autoReconnect, maxReconnectAttempts, reconnectInterval };
  }, [url, autoReconnect, maxReconnectAttempts, reconnectInterval]);

  useEffect(() => {
    callbacksRef.current = { onMessage, onConnect, onDisconnect, onError };
  }, [onMessage, onConnect, onDisconnect, onError]);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN || ws.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    setConnectionState('connecting');

    try {
      ws.current = new WebSocket(optionsRef.current.url);

      ws.current.onopen = () => {
        if (!mountedRef.current) return;

        setIsConnected(true);
        setConnectionState('connected');
        reconnectAttemptRef.current = 0;
        callbacksRef.current.onConnect?.();

        devInfo('🔗 WebSocket connected');
      };

      ws.current.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          setMessageHistory(prev => [...prev.slice(-99), message]); // Keep last 100 messages
          callbacksRef.current.onMessage?.(message);
        } catch (error) {
          devError('Failed to parse WebSocket message:', error);
        }
      };

      ws.current.onclose = () => {
        if (!mountedRef.current) return;

        setIsConnected(false);
        setConnectionState('disconnected');
        callbacksRef.current.onDisconnect?.();

        devInfo('🔌 WebSocket disconnected');

        // Auto-reconnect logic using current ref values
        const opts = optionsRef.current;
        if (opts.autoReconnect && reconnectAttemptRef.current < opts.maxReconnectAttempts) {
          reconnectAttemptRef.current += 1;
          devInfo(`🔄 Reconnecting WebSocket (attempt ${reconnectAttemptRef.current}/${opts.maxReconnectAttempts})...`);

          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, opts.reconnectInterval);
        }
      };

      ws.current.onerror = (error) => {
        if (!mountedRef.current) return;

        setConnectionState('error');
        callbacksRef.current.onError?.(error);
        devError('❌ WebSocket error:', error);
      };

    } catch (error) {
      devError('Failed to create WebSocket connection:', error);
      setConnectionState('error');
    }
  }, []); // No dependencies - uses refs for latest values

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }

    setIsConnected(false);
    setConnectionState('disconnected');
  }, []);

  const sendMessage = useCallback((message: Partial<WebSocketMessage>) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      const fullMessage: WebSocketMessage = {
        type: 'metrics',
        data: null,
        timestamp: new Date().toISOString(),
        ...message,
      };

      ws.current.send(JSON.stringify(fullMessage));
      return true;
    }

    devWarn('WebSocket is not connected. Cannot send message.');
    return false;
  }, []);

  const clearHistory = useCallback(() => {
    setMessageHistory([]);
    setLastMessage(null);
  }, []);

  // Connection management - only runs on mount/unmount
  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Intentionally empty - connect/disconnect are stable

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    isConnected,
    connectionState,
    lastMessage,
    messageHistory,
    connect,
    disconnect,
    sendMessage,
    clearHistory,
    reconnectAttempt: reconnectAttemptRef.current,
    maxReconnectAttempts,
  };
}