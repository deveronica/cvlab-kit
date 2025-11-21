/**
 * React hooks for Component Discovery API
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../lib/api-client';
import type { ComponentCategory, ComponentInfo } from '../lib/types';

// Query keys
export const componentKeys = {
  all: ['components'] as const,
  categories: () => [...componentKeys.all, 'categories'] as const,
  category: (category: string) => [...componentKeys.all, 'category', category] as const,
  search: (query: string) => [...componentKeys.all, 'search', query] as const,
  details: (category: string, name: string) => [...componentKeys.all, 'details', category, name] as const,
};

/**
 * Get all components organized by category
 */
export function useComponents() {
  return useQuery({
    queryKey: componentKeys.categories(),
    queryFn: () => apiClient.getAllComponents(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Get components in a specific category
 */
export function useComponentsByCategory(category: string) {
  return useQuery({
    queryKey: componentKeys.category(category),
    queryFn: () => apiClient.getComponentsByCategory(category),
    enabled: !!category,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });
}

/**
 * Search components by name or description
 */
export function useComponentSearch(query: string) {
  return useQuery({
    queryKey: componentKeys.search(query),
    queryFn: () => apiClient.searchComponents(query),
    enabled: query.length > 0,
    staleTime: 2 * 60 * 1000, // 2 minutes for search results
    gcTime: 5 * 60 * 1000,
  });
}

/**
 * Get detailed information about a specific component
 */
export function useComponentDetails(category: string, name: string) {
  return useQuery({
    queryKey: componentKeys.details(category, name),
    queryFn: () => apiClient.getComponentDetails(category, name),
    enabled: !!(category && name),
    staleTime: 10 * 60 * 1000, // 10 minutes for details
    gcTime: 15 * 60 * 1000,
  });
}

/**
 * Validate component configuration
 */
export function useValidateComponentConfig() {
  const _queryClient = useQueryClient();

  return useMutation({
    mutationFn: (config: any) => apiClient.validateComponentConfig(config),
    onSuccess: () => {
      // Could invalidate related queries if needed
    },
  });
}

/**
 * Component statistics hook for dashboard
 */
export function useComponentStats() {
  const { data: categories } = useComponents();

  if (!categories) {
    return {
      totalComponents: 0,
      categoryCounts: {},
      isLoading: true,
    };
  }

  const categoryCounts: Record<string, number> = {};
  let totalComponents = 0;

  categories.forEach((category: ComponentCategory) => {
    categoryCounts[category.category] = category.count;
    totalComponents += category.count;
  });

  return {
    totalComponents,
    categoryCounts,
    isLoading: false,
  };
}