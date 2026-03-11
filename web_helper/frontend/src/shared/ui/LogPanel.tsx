import { useEffect, useRef } from 'react';
import { useLogStream } from '@/shared/model/useLogStream';
import { Card, CardContent, CardHeader, CardTitle } from '@shared/ui/card';

interface LogPanelProps {
  runName: string;
}

export function LogPanel({ runName }: LogPanelProps) {
  const { logs, isConnected } = useLogStream(runName);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Live Logs</span>
          <span className={`text-sm font-medium ${isConnected ? 'text-green-500' : 'text-red-500'}`}>
            {isConnected ? '● Connected' : '○ Disconnected'}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={scrollRef} className="h-96 bg-gray-900 text-white font-mono text-sm p-4 rounded-lg overflow-y-auto">
          {logs.map((log, index) => (
            <div key={index}>{log}</div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
