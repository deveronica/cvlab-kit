import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Outlet } from 'react-router-dom';
import type { ReactNode } from 'react';
import { AppRoutes } from './AppRoutes';

vi.mock('@app/layouts/MainLayout', () => ({
  MainLayout: () => (
    <div>
      <Outlet />
    </div>
  ),
}));

vi.mock('@/shared/ui/error-boundary', () => ({
  ErrorBoundary: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/pages/DashboardPage', () => ({ DashboardPage: () => <div>Dashboard Page</div> }));
vi.mock('@/pages/MonitoringPage', () => ({ MonitoringPage: () => <div>Monitoring Page</div> }));
vi.mock('@/pages/ExecutePage', () => ({ ExecutePage: () => <div>Execute Page</div> }));
vi.mock('@/pages/ProjectsPage', () => ({ ProjectsPage: () => <div>Projects Page</div> }));
vi.mock('@/pages/BuilderPage', () => ({ BuilderPage: () => <div>Builder Page</div> }));
vi.mock('@/pages/ExperimentsPage', () => ({ ExperimentsPage: () => <div>Experiments Page</div> }));

describe('AppRoutes', () => {
  it('routes builder path', () => {
    render(
      <MemoryRouter initialEntries={['/builder']}>
        <AppRoutes />
      </MemoryRouter>
    );

    expect(screen.getByText('Builder Page')).toBeInTheDocument();
  });

  it('redirects old queue route to experiments', () => {
    render(
      <MemoryRouter initialEntries={['/queue']}>
        <AppRoutes />
      </MemoryRouter>
    );

    expect(screen.getByText('Experiments Page')).toBeInTheDocument();
  });
});
