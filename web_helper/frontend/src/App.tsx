import React from "react";
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { ComparisonProvider } from './contexts/ComparisonContext';
import { ChartInteractionProvider } from './contexts/ChartInteractionContext';
import { ExecuteProvider } from './contexts/ExecuteContext';
import { MainLayout } from './components/layout/MainLayout';
import { ScrollToTop } from './components/ScrollToTop';
import { ErrorBoundary } from './components/ui/error-boundary';
import { useSSE } from './hooks/useSSE';
import { DashboardView } from './components/views/DashboardView';
import { MonitoringView } from './components/views/MonitoringView';
import { QueueView } from './components/views/QueueView';
import { ExecuteView } from './components/views/ExecuteView';
import { ProjectsView } from './components/views/ProjectsView';
import { ResultsView } from './components/views/ResultsView';
import { ComponentsView } from './components/views/ComponentsView';
import { devInfo, devWarn, devError } from './lib/dev-utils';

function App() {
  // Initialize SSE connection for real-time updates
  useSSE({
    autoReconnect: true,
    reconnectDelay: 3000,
    maxReconnectAttempts: 5,
    onConnect: () => devInfo('📡 Real-time updates connected'),
    onDisconnect: () => devWarn('📡 Real-time updates disconnected'),
    onError: (error) => devError('📡 Real-time update error:', error),
  });

  return (
    <ErrorBoundary level="app">
      <ThemeProvider>
        <ChartInteractionProvider>
          <ComparisonProvider>
            <ExecuteProvider>
              <ScrollToTop />
              <Routes>
                <Route path="/" element={<MainLayout />}>
                  <Route index element={
                    <ErrorBoundary level="view">
                      <DashboardView />
                    </ErrorBoundary>
                  } />
                  <Route path="monitoring" element={
                    <ErrorBoundary level="view">
                      <MonitoringView />
                    </ErrorBoundary>
                  } />
                  <Route path="execute" element={
                    <ErrorBoundary level="view">
                      <ExecuteView />
                    </ErrorBoundary>
                  } />
                  <Route path="queue" element={
                    <ErrorBoundary level="view">
                      <QueueView />
                    </ErrorBoundary>
                  } />
                  <Route path="results" element={
                    <ErrorBoundary level="view">
                      <ResultsView />
                    </ErrorBoundary>
                  } />
                  <Route path="projects" element={
                    <ErrorBoundary level="view" resetKeys={['projects']}>
                      <ProjectsView />
                    </ErrorBoundary>
                  } />
                  <Route path="projects/:projectName" element={
                    <ErrorBoundary level="view" resetKeys={['projects', ':projectName']}>
                      <ProjectsView />
                    </ErrorBoundary>
                  } />
                  <Route path="components" element={
                    <ErrorBoundary level="view">
                      <ComponentsView />
                    </ErrorBoundary>
                  } />
                </Route>
              </Routes>
            </ExecuteProvider>
          </ComparisonProvider>
        </ChartInteractionProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;