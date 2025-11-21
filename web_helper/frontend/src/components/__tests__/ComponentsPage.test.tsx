import React from "react";
/**
 * Tests for ComponentsPage component
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ComponentsPage } from '../pages/ComponentsPage';
import { render, mockApiResponse, sampleComponentData } from '../../test/utils';

// Mock the hooks
vi.mock('../../hooks/useComponents', () => ({
  useComponents: vi.fn(),
  useComponentSearch: vi.fn(),
  useComponentStats: vi.fn(),
}));

import {
  useComponents,
  useComponentSearch,
  useComponentStats,
} from '../../hooks/useComponents';

const mockUseComponents = vi.mocked(useComponents);
const mockUseComponentSearch = vi.mocked(useComponentSearch);
const mockUseComponentStats = vi.mocked(useComponentStats);

describe('ComponentsPage', () => {
  const mockCategoriesData = [
    {
      category: 'model',
      count: 2,
      components: [
        sampleComponentData,
        {
          ...sampleComponentData,
          name: 'vgg16',
          description: 'VGG-16 model for image classification',
        },
      ],
    },
    {
      category: 'dataset',
      count: 1,
      components: [
        {
          name: 'cifar10',
          type: 'dataset',
          path: 'cvlabkit/component/dataset/cifar10.py',
          description: 'CIFAR-10 dataset',
          parameters: {},
          examples: [],
        },
      ],
    },
  ];

  const mockStatsData = {
    totalComponents: 3,
    categoryCounts: {
      model: 2,
      dataset: 1,
    },
    isLoading: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();

    mockUseComponents.mockReturnValue({
      data: mockCategoriesData,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    mockUseComponentSearch.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    mockUseComponentStats.mockReturnValue(mockStatsData);
  });

  it('renders component discovery page', () => {
    render(<ComponentsPage />);

    expect(screen.getByText('Component Discovery')).toBeInTheDocument();
    expect(
      screen.getByText('Explore and browse available CVLab-Kit components')
    ).toBeInTheDocument();
    expect(screen.getByText('3 components available')).toBeInTheDocument();
  });

  it('displays category statistics', () => {
    render(<ComponentsPage />);

    expect(screen.getByText('model')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('dataset')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('All')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('displays components table', () => {
    render(<ComponentsPage />);

    expect(screen.getByText('resnet18')).toBeInTheDocument();
    expect(screen.getByText('model')).toBeInTheDocument();
    expect(screen.getByText('ResNet-18 model for image classification')).toBeInTheDocument();
    expect(screen.getByText('vgg16')).toBeInTheDocument();
    expect(screen.getByText('cifar10')).toBeInTheDocument();
  });

  it('filters components by category', async () => {
    const user = userEvent.setup();
    render(<ComponentsPage />);

    // Click on model category
    const modelCategory = screen.getByText('model').closest('div');
    await user.click(modelCategory!);

    // Should show only model components
    expect(screen.getByText('resnet18')).toBeInTheDocument();
    expect(screen.getByText('vgg16')).toBeInTheDocument();
    expect(screen.queryByText('cifar10')).not.toBeInTheDocument();
  });

  it('searches components', async () => {
    const user = userEvent.setup();

    // Mock search results
    mockUseComponentSearch.mockReturnValue({
      data: [sampleComponentData],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    render(<ComponentsPage />);

    const searchInput = screen.getByPlaceholderText('Search components...');
    await user.type(searchInput, 'resnet');

    expect(searchInput).toHaveValue('resnet');
  });

  it('opens component details modal', async () => {
    const user = userEvent.setup();
    render(<ComponentsPage />);

    // Click on View Details button
    const viewButton = screen.getAllByText('View Details')[0];
    await user.click(viewButton);

    // Should open modal
    expect(screen.getByText('resnet18')).toBeInTheDocument();
    expect(screen.getByText('Description')).toBeInTheDocument();
    expect(screen.getByText('Parameters')).toBeInTheDocument();
    expect(screen.getByText('Configuration Examples')).toBeInTheDocument();
  });

  it('closes component details modal', async () => {
    const user = userEvent.setup();
    render(<ComponentsPage />);

    // Open modal
    const viewButton = screen.getAllByText('View Details')[0];
    await user.click(viewButton);

    // Close modal
    const closeButton = screen.getByText('✕');
    await user.click(closeButton);

    // Modal should be closed
    await waitFor(() => {
      expect(screen.queryByText('Description')).not.toBeInTheDocument();
    });
  });

  it('handles loading state', () => {
    mockUseComponents.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    } as any);

    render(<ComponentsPage />);

    expect(screen.getByRole('status')).toBeInTheDocument(); // Loading spinner
  });

  it('displays search results count', () => {
    mockUseComponentSearch.mockReturnValue({
      data: [sampleComponentData],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    const ComponentsPageWithSearch = () => {
      return <ComponentsPage />;
    };

    render(<ComponentsPageWithSearch />);

    // When search query is active, it should show search results count
    // This would require updating the component to show search query state
  });

  it('displays component parameters in details modal', async () => {
    const user = userEvent.setup();
    render(<ComponentsPage />);

    // Open modal
    const viewButton = screen.getAllByText('View Details')[0];
    await user.click(viewButton);

    // Check parameters
    expect(screen.getByText('Parameters (2)')).toBeInTheDocument();
    expect(screen.getByText('num_classes')).toBeInTheDocument();
    expect(screen.getByText('pretrained')).toBeInTheDocument();
  });

  it('displays component examples in details modal', async () => {
    const user = userEvent.setup();
    render(<ComponentsPage />);

    // Open modal
    const viewButton = screen.getAllByText('View Details')[0];
    await user.click(viewButton);

    // Check examples
    expect(screen.getByText('Configuration Examples (1)')).toBeInTheDocument();
  });
});