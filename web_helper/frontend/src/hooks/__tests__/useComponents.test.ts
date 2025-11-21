/**
 * Tests for useComponents hooks
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import {
  useComponents,
  useComponentsByCategory,
  useComponentSearch,
  useComponentDetails,
  useValidateComponentConfig,
  useComponentStats,
} from '../useComponents';
import { apiClient } from '../../lib/api-client';

// Mock the API client
vi.mock('../../lib/api-client', () => ({
  apiClient: {
    getAllComponents: vi.fn(),
    getComponentsByCategory: vi.fn(),
    searchComponents: vi.fn(),
    getComponentDetails: vi.fn(),
    validateComponentConfig: vi.fn(),
  },
}));

const mockApiClient = vi.mocked(apiClient);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

const mockCategoriesData = [
  {
    category: 'model',
    count: 2,
    components: [
      {
        name: 'resnet18',
        type: 'model',
        path: 'cvlabkit/component/model/resnet18.py',
        description: 'ResNet-18 model',
        parameters: {},
        examples: [],
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

describe('useComponents', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches all components', async () => {
    mockApiClient.getAllComponents.mockResolvedValue(mockCategoriesData);

    const { result } = renderHook(() => useComponents(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockCategoriesData);
    expect(mockApiClient.getAllComponents).toHaveBeenCalledOnce();
  });

  it('handles fetch error', async () => {
    mockApiClient.getAllComponents.mockRejectedValue(new Error('API Error'));

    const { result } = renderHook(() => useComponents(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeInstanceOf(Error);
  });
});

describe('useComponentsByCategory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches components by category', async () => {
    const mockData = [
      {
        name: 'resnet18',
        type: 'model',
        path: 'cvlabkit/component/model/resnet18.py',
        description: 'ResNet-18 model',
        parameters: {},
        examples: [],
      },
    ];

    mockApiClient.getComponentsByCategory.mockResolvedValue(mockData);

    const { result } = renderHook(
      () => useComponentsByCategory('model'),
      {
        wrapper: createWrapper(),
      }
    );

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockData);
    expect(mockApiClient.getComponentsByCategory).toHaveBeenCalledWith('model');
  });

  it('does not fetch when category is empty', () => {
    const { result } = renderHook(
      () => useComponentsByCategory(''),
      {
        wrapper: createWrapper(),
      }
    );

    expect(result.current.fetchStatus).toBe('idle');
    expect(mockApiClient.getComponentsByCategory).not.toHaveBeenCalled();
  });
});

describe('useComponentSearch', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('searches components', async () => {
    const mockResults = [
      {
        name: 'resnet18',
        type: 'model',
        path: 'cvlabkit/component/model/resnet18.py',
        description: 'ResNet-18 model',
        parameters: {},
        examples: [],
      },
    ];

    mockApiClient.searchComponents.mockResolvedValue(mockResults);

    const { result } = renderHook(
      () => useComponentSearch('resnet'),
      {
        wrapper: createWrapper(),
      }
    );

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResults);
    expect(mockApiClient.searchComponents).toHaveBeenCalledWith('resnet');
  });

  it('does not search when query is empty', () => {
    const { result } = renderHook(
      () => useComponentSearch(''),
      {
        wrapper: createWrapper(),
      }
    );

    expect(result.current.fetchStatus).toBe('idle');
    expect(mockApiClient.searchComponents).not.toHaveBeenCalled();
  });
});

describe('useComponentDetails', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches component details', async () => {
    const mockDetails = {
      name: 'resnet18',
      type: 'model',
      path: 'cvlabkit/component/model/resnet18.py',
      description: 'ResNet-18 model for image classification',
      parameters: {
        num_classes: { type: 'int', default: '1000', required: false },
      },
      examples: [
        {
          name: 'cifar10_resnet18',
          type: 'model.resnet18',
          params: { num_classes: 10 },
        },
      ],
    };

    mockApiClient.getComponentDetails.mockResolvedValue(mockDetails);

    const { result } = renderHook(
      () => useComponentDetails('model', 'resnet18'),
      {
        wrapper: createWrapper(),
      }
    );

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockDetails);
    expect(mockApiClient.getComponentDetails).toHaveBeenCalledWith('model', 'resnet18');
  });

  it('does not fetch when category or name is empty', () => {
    const { result: result1 } = renderHook(
      () => useComponentDetails('', 'resnet18'),
      {
        wrapper: createWrapper(),
      }
    );

    const { result: result2 } = renderHook(
      () => useComponentDetails('model', ''),
      {
        wrapper: createWrapper(),
      }
    );

    expect(result1.current.fetchStatus).toBe('idle');
    expect(result2.current.fetchStatus).toBe('idle');
    expect(mockApiClient.getComponentDetails).not.toHaveBeenCalled();
  });
});

describe('useValidateComponentConfig', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('validates component configuration', async () => {
    const mockResponse = { valid: true };
    mockApiClient.validateComponentConfig.mockResolvedValue(mockResponse);

    const { result } = renderHook(
      () => useValidateComponentConfig(),
      {
        wrapper: createWrapper(),
      }
    );

    const config = {
      type: 'model.resnet18',
      params: { num_classes: 10 },
    };

    result.current.mutate(config);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockResponse);
    expect(mockApiClient.validateComponentConfig).toHaveBeenCalledWith(config);
  });

  it('handles validation errors', async () => {
    const mockError = new Error('Validation failed');
    mockApiClient.validateComponentConfig.mockRejectedValue(mockError);

    const { result } = renderHook(
      () => useValidateComponentConfig(),
      {
        wrapper: createWrapper(),
      }
    );

    const config = {
      type: 'invalid.component',
      params: {},
    };

    result.current.mutate(config);

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(mockError);
  });
});

describe('useComponentStats', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('calculates component statistics', () => {
    mockApiClient.getAllComponents.mockResolvedValue(mockCategoriesData);

    const { result } = renderHook(() => useComponents(), {
      wrapper: createWrapper(),
    });

    // Wait for data to load
    waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const { result: statsResult } = renderHook(() => useComponentStats(), {
      wrapper: createWrapper(),
    });

    expect(statsResult.current.totalComponents).toBe(3);
    expect(statsResult.current.categoryCounts).toEqual({
      model: 2,
      dataset: 1,
    });
    expect(statsResult.current.isLoading).toBe(false);
  });

  it('handles loading state', () => {
    const { result } = renderHook(() => useComponentStats(), {
      wrapper: createWrapper(),
    });

    expect(statsResult.current.totalComponents).toBe(0);
    expect(statsResult.current.categoryCounts).toEqual({});
    expect(statsResult.current.isLoading).toBe(true);
  });
});