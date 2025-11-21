import React from "react";
import { MainLayout } from '../components/layout/MainLayout';
import { useSSE } from '../hooks/useSSE';

interface DashboardProps {
  _activeTab?: string;
}

export default function Dashboard({ _activeTab = 'dashboard' }: DashboardProps) {
  // Connect to Server-Sent Events for real-time updates
  useSSE({
    onConnect: () => console.info('🔌 Real-time updates connected'),
    onDisconnect: () => console.warn('🔌 Real-time updates disconnected'),
    onError: (error) => console.error('🔌 Real-time update error:', error),
  });

  return <MainLayout />;
}
