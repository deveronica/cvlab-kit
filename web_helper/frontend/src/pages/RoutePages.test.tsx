import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DashboardPage } from './DashboardPage';
import { MonitoringPage } from './MonitoringPage';
import { ExecutePage } from './ExecutePage';
import { ProjectsPage } from './ProjectsPage';
import { BuilderPage } from './BuilderPage';
import { ExperimentsPage } from './ExperimentsPage';

vi.mock('@/features/dashboard/view/DashboardView', () => ({
  DashboardView: () => <div>DashboardView Mock</div>,
}));

vi.mock('@/features/monitoring/view/MonitoringView', () => ({
  MonitoringView: () => <div>MonitoringView Mock</div>,
}));

vi.mock('@/features/execute/view/ExecuteView', () => ({
  ExecuteView: () => <div>ExecuteView Mock</div>,
}));

vi.mock('@/features/projects/view/ProjectsView', () => ({
  ProjectsView: () => <div>ProjectsView Mock</div>,
}));

vi.mock('@/features/builder/view/BuilderView', () => ({
  BuilderView: () => <div>BuilderView Mock</div>,
}));

vi.mock('@/features/experiments/view/ExperimentsView', () => ({
  ExperimentsView: () => <div>ExperimentsView Mock</div>,
}));

describe('Route pages wrappers', () => {
  it('renders DashboardPage wrapper', () => {
    render(<DashboardPage />);
    expect(screen.getByText('DashboardView Mock')).toBeInTheDocument();
  });

  it('renders MonitoringPage wrapper', () => {
    render(<MonitoringPage />);
    expect(screen.getByText('MonitoringView Mock')).toBeInTheDocument();
  });

  it('renders ExecutePage wrapper', () => {
    render(<ExecutePage />);
    expect(screen.getByText('ExecuteView Mock')).toBeInTheDocument();
  });

  it('renders ProjectsPage wrapper', () => {
    render(<ProjectsPage />);
    expect(screen.getByText('ProjectsView Mock')).toBeInTheDocument();
  });

  it('renders BuilderPage wrapper', () => {
    render(<BuilderPage />);
    expect(screen.getByText('BuilderView Mock')).toBeInTheDocument();
  });

  it('renders ExperimentsPage wrapper', () => {
    render(<ExperimentsPage />);
    expect(screen.getByText('ExperimentsView Mock')).toBeInTheDocument();
  });
});
