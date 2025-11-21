import { useState, useEffect, useRef } from 'react';
import { devLog, devError } from '../lib/dev-utils';

/**
 * Hook for streaming job logs via WebSocket
 * @param jobId - The job ID to stream logs for (from queue system)
 */
export const useLogStream = (jobId: string | null) => {
  const [logs, setLogs] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!jobId) {
      setLogs([]);
      setIsConnected(false);
      return;
    }

    const connect = () => {
      const protocol = window.location.protocol === 'https' ? 'wss' : 'ws';
      const host = window.location.host;
      // Updated to use queue-based log streaming endpoint
      ws.current = new WebSocket(`${protocol}://${host}/api/queue/job/${jobId}/ws/logs`);

      ws.current.onopen = () => {
        devLog(`WebSocket connected for job ${jobId}`);
        setIsConnected(true);
      };

      ws.current.onmessage = (event) => {
        setLogs(prevLogs => [...prevLogs, event.data]);
      };

      ws.current.onerror = (error) => {
        devError('WebSocket error:', error);
        setIsConnected(false);
      };

      ws.current.onclose = () => {
        devLog(`WebSocket disconnected for job ${jobId}`);
        setIsConnected(false);
      };
    };

    connect();

    return () => {
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
    };
  }, [jobId]);

  return { logs, isConnected };
};