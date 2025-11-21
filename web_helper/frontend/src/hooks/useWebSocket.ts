import { useState, useEffect, useRef, useCallback } from 'react';
import { devInfo, devError, devWarn } from '../lib/dev-utils';

interface WebSocketMessage {
  type: 'metrics' | 'experiment_update' | 'system_alert' | 'device_status' | 'queue_update';
  data: any;
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
    url = `ws://${window.location.host}/ws`,
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

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN || ws.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    setConnectionState('connecting');

    try {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        if (!mountedRef.current) return;

        setIsConnected(true);
        setConnectionState('connected');
        reconnectAttemptRef.current = 0;
        onConnect?.();

        devInfo('🔗 WebSocket connected');
      };

      ws.current.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          setMessageHistory(prev => [...prev.slice(-99), message]); // Keep last 100 messages
          onMessage?.(message);
        } catch (error) {
          devError('Failed to parse WebSocket message:', error);
        }
      };

      ws.current.onclose = () => {
        if (!mountedRef.current) return;

        setIsConnected(false);
        setConnectionState('disconnected');
        onDisconnect?.();

        devInfo('🔌 WebSocket disconnected');

        // Auto-reconnect logic
        if (autoReconnect && reconnectAttemptRef.current < maxReconnectAttempts) {
          reconnectAttemptRef.current += 1;
          devInfo(`🔄 Reconnecting WebSocket (attempt ${reconnectAttemptRef.current}/${maxReconnectAttempts})...`);

          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.current.onerror = (error) => {
        if (!mountedRef.current) return;

        setConnectionState('error');
        onError?.(error);
        devError('❌ WebSocket error:', error);
      };

    } catch (error) {
      devError('Failed to create WebSocket connection:', error);
      setConnectionState('error');
    }
  }, [url, autoReconnect, maxReconnectAttempts, reconnectInterval, onConnect, onDisconnect, onError, onMessage]);

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

  // Connection management
  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [connect, disconnect]);

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