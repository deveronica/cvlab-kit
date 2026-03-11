import { AppProviders } from '@/app/ui/AppProviders';
import { ScrollToTop } from '@/shared/ui/ScrollToTop';
import { ErrorBoundary } from '@/shared/ui/error-boundary';
import { useRealtime } from '@/shared/model/useRealtime';
import { AppRoutes } from '@/app/router/AppRoutes';
import { devInfo, devWarn, devError } from '@/shared/lib/utils';
import { useEffect } from 'react';
import { useBuilderStore } from '@/features/builder/model/builderStore';

function App() {
  const setTypeColors = useBuilderStore(s => s.setTypeColors);

  // 항목 2: 동적 타입 색상 로드
  useEffect(() => {
    fetch('/api/nodes/types')
      .then(res => res.json())
      .then(data => {
        if (data.success) setTypeColors(data.types);
      })
      .catch(err => devWarn('Failed to load type colors:', err));
  }, [setTypeColors]);

  // Initialize real-time connection (WebSocket with SSE fallback)
  useRealtime({
    channels: ['queue', 'devices', 'runs', 'nodes'],
    autoReconnect: true,
    reconnectDelay: 3000,
    maxReconnectAttempts: 3,
    onConnect: (transport) => devInfo(`📡 Real-time updates connected via ${transport}`),
    onDisconnect: (transport) => devWarn(`📡 Real-time updates disconnected (${transport})`),
    onError: (error) => devError('📡 Real-time update error:', error),
    onTransportChange: (from, to) => devInfo(`📡 Transport fallback: ${from} → ${to}`),
  });

  return (
    <ErrorBoundary level="app">
      <AppProviders>
        <ScrollToTop />
        <AppRoutes />
      </AppProviders>
    </ErrorBoundary>
  );
}

export default App;
