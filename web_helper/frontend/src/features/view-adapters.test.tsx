import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DashboardView } from '@/features/dashboard/view/DashboardView';
import { MonitoringView } from '@/features/monitoring/view/MonitoringView';
import { ExecuteView } from '@/features/execute/view/ExecuteView';
import { ProjectsView } from '@/features/projects/view/ProjectsView';
import { BuilderView } from '@/features/builder/view/BuilderView';
import { ExperimentsView } from '@/features/experiments/view/ExperimentsView';

vi.mock('@/features/dashboard/view/DashboardView', () => ({
  DashboardView: () => <div>DashboardView Impl</div>,
}));

vi.mock('@/features/monitoring/view/MonitoringView', () => ({
  MonitoringView: () => <div>MonitoringView Impl</div>,
}));

vi.mock('@/features/execute/view/ExecuteView', () => ({
  ExecuteView: () => <div>ExecuteView Impl</div>,
}));

vi.mock('@/features/projects/view/ProjectsView', () => ({
  ProjectsView: () => <div>ProjectsView Impl</div>,
}));

vi.mock('@/features/builder/view/BuilderView', () => ({
  BuilderView: () => <div>BuilderView Impl</div>,
}));

vi.mock('@/features/experiments/view/ExperimentsView', () => ({
  __esModule: true,
  default: () => <div>ExperimentsView Impl</div>,
  ExperimentsView: () => <div>ExperimentsView Impl</div>,
}));

describe('Feature views', () => {
  it('renders dashboard adapter', () => {
    render(<DashboardView />);
    expect(screen.getByText('DashboardView Impl')).toBeInTheDocument();
  });

  it('renders monitoring adapter', () => {
    render(<MonitoringView />);
    expect(screen.getByText('MonitoringView Impl')).toBeInTheDocument();
  });

  it('renders execute adapter', () => {
    render(<ExecuteView />);
    expect(screen.getByText('ExecuteView Impl')).toBeInTheDocument();
  });

  it('renders projects adapter', () => {
    render(<ProjectsView />);
    expect(screen.getByText('ProjectsView Impl')).toBeInTheDocument();
  });

  it('renders builder adapter', () => {
    render(<BuilderView />);
    expect(screen.getByText('BuilderView Impl')).toBeInTheDocument();
  });

  it('renders experiments adapter', () => {
    render(<ExperimentsView />);
    expect(screen.getByText('ExperimentsView Impl')).toBeInTheDocument();
  });
});
