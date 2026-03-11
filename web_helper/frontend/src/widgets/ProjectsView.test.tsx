import React from "react";
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { ProjectsView } from './ProjectsView';

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useParams: () => ({}),
    useNavigate: () => vi.fn(),
  };
});

vi.mock('@/shared/model/useProjectsView', () => ({
  useProjectsView: () => ({
    _projects: [],
    projectSummaries: [],
    experimentsData: [],
    hyperparamColumns: [],
    metricColumns: [],
    getBestRun: vi.fn(),
    projectsLoading: false,
    flattenParams: false,
    setFlattenParams: vi.fn(),
    diffMode: false,
    setDiffMode: vi.fn(),
    selectedExperiments: [],
    setSelectedExperiments: vi.fn(),
    activeView: 'overview',
    setActiveView: vi.fn(),
    isComparisonModalOpen: false,
    setIsComparisonModalOpen: vi.fn(),
    visibleHyperparams: [],
    setVisibleHyperparams: vi.fn(),
    visibleMetrics: [],
    setVisibleMetrics: vi.fn(),
    pinnedLeftHyperparams: [],
    setPinnedLeftHyperparams: vi.fn(),
    pinnedRightMetrics: [],
    setPinnedRightMetrics: vi.fn(),
    hyperparamOrder: [],
    setHyperparamOrder: vi.fn(),
    metricOrder: [],
    setMetricOrder: vi.fn(),
    clearSelection: vi.fn(),
    exportRuns: vi.fn(),
    getSelectedExperimentData: vi.fn(),
    fetchRunDetails: vi.fn(),
    chartsRef: { current: null },
    _filters: {},
    _setFilters: vi.fn(),
    selectedMetric: null,
    setSelectedMetric: vi.fn(),
    metricDirection: 'max',
    setMetricDirection: vi.fn(),
    selectedRunDetail: null,
    setSelectedRunDetail: vi.fn(),
    columnMappings: [],
    columnSuggestions: [],
    isColumnMappingDialogOpen: false,
    setIsColumnMappingDialogOpen: vi.fn(),
    isLoadingColumnMappings: false,
    _isGeneratingSuggestions: false,
    handleAcceptSuggestion: vi.fn(),
    handleRejectSuggestion: vi.fn(),
    handleCreateMapping: vi.fn(),
    handleUpdateMapping: vi.fn(),
    handleDeleteMapping: vi.fn(),
    cellOutliers: {},
    highlightedRun: null,
    setHighlightedRun: vi.fn(),
  }),
}));

describe('ProjectsView', () => {
  it('renders without crashing', () => {
    render(
      <MemoryRouter>
        <ProjectsView />
      </MemoryRouter>
    );
    expect(screen.getByText('Projects')).toBeInTheDocument();
  });
});
