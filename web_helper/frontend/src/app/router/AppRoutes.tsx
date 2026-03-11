import { Route, Routes, Navigate } from 'react-router-dom';
import { MainLayout } from '@app/layouts/MainLayout';
import { ErrorBoundary } from '@/shared/ui/error-boundary';
import { DashboardPage } from '@/pages/DashboardPage';
import { MonitoringPage } from '@/pages/MonitoringPage';
import { ExecutePage } from '@/pages/ExecutePage';
import { ProjectsPage } from '@/pages/ProjectsPage';
import { BuilderPage } from '@/pages/BuilderPage';
import { ExperimentsPage } from '@/pages/ExperimentsPage';

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route
          index
          element={
            <ErrorBoundary level="view">
              <DashboardPage />
            </ErrorBoundary>
          }
        />
        <Route
          path="monitoring"
          element={
            <ErrorBoundary level="view">
              <MonitoringPage />
            </ErrorBoundary>
          }
        />
        <Route
          path="execute"
          element={
            <ErrorBoundary level="view">
              <ExecutePage />
            </ErrorBoundary>
          }
        />
        <Route
          path="experiments"
          element={
            <ErrorBoundary level="view">
              <ExperimentsPage />
            </ErrorBoundary>
          }
        />
        <Route path="queue" element={<Navigate to="/experiments" replace />} />
        <Route path="results" element={<Navigate to="/experiments" replace />} />
        <Route
          path="projects"
          element={
            <ErrorBoundary level="view" resetKeys={['projects']}>
              <ProjectsPage />
            </ErrorBoundary>
          }
        />
        <Route
          path="projects/:projectName"
          element={
            <ErrorBoundary level="view" resetKeys={['projects', ':projectName']}>
              <ProjectsPage />
            </ErrorBoundary>
          }
        />
        <Route
          path="builder"
          element={
            <ErrorBoundary level="view">
              <BuilderPage />
            </ErrorBoundary>
          }
        />
        <Route path="components" element={<Navigate to="/builder" replace />} />
      </Route>
    </Routes>
  );
}
