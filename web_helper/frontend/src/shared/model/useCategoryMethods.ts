/**
 * Hook for fetching and caching category methods from the API.
 *
 * Methods are callable operations on component outputs:
 * - model: parameters(), eval(), train(), to(), cuda(), cpu()
 * - optimizer: step(), zero_grad(), state_dict()
 * - loss: eval(), train(), to()
 * - metric: update(), compute(), reset()
 * - scheduler: step(), get_last_lr()
 */

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/shared/api/api-client';
import type { MethodDefinition, CategoryMethodsAPIResponse } from '@/shared/model/node-graph';

/**
 * Fetch methods for a specific component category.
 *
 * @param category - Component category (e.g., 'model', 'optimizer')
 * @returns Query result with methods array
 *
 * @example
 * const { data: methods, isLoading } = useCategoryMethods('model');
 * // methods = [{ name: 'parameters', returns: 'params', args: [], description: '...' }, ...]
 */
export function useCategoryMethods(category: string | null) {
  return useQuery({
    queryKey: ['categoryMethods', category],
    queryFn: async (): Promise<MethodDefinition[]> => {
      if (!category) return [];

      const response = await apiClient.getRaw<CategoryMethodsAPIResponse>(
        `/nodes/methods/${category}`
      );

      if (!response.success || !response.data) {
        console.warn(`Failed to fetch methods for category: ${category}`);
        return [];
      }

      return response.data.methods;
    },
    enabled: !!category,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes (methods rarely change)
    gcTime: 30 * 60 * 1000,   // Keep in cache for 30 minutes
  });
}

/**
 * Get a specific method by name from a category's methods.
 *
 * @param category - Component category
 * @param methodName - Method name to find
 * @returns The method definition or undefined
 */
export function useMethodByName(category: string | null, methodName: string | null) {
  const { data: methods } = useCategoryMethods(category);

  if (!methods || !methodName) return undefined;

  return methods.find((m) => m.name === methodName);
}

/**
 * Prefetch methods for multiple categories.
 * Useful when rendering many nodes to avoid waterfall requests.
 *
 * @param categories - Array of category names to prefetch
 */
export function usePrefetchCategoryMethods(categories: string[]) {
  // This hook doesn't return anything, it just triggers prefetch queries
  // Each unique category will be cached separately
  categories.forEach((category) => {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    useCategoryMethods(category);
  });
}
