/**
 * Hook for fetching available implementations for a component category.
 *
 * Used in Execute tab to populate the implementation selector dropdown.
 */

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/shared/api/api-client';

export interface Implementation {
  name: string;
  type: string;
  path: string;
  hash: string;
  description: string | null;
  parameters: Record<string, unknown>;
}

interface CategoryImplementationsResponse {
  data: Implementation[];
}

/**
 * Fetch all available implementations for a component category.
 *
 * @param category - Component category (e.g., 'model', 'optimizer', 'loss')
 * @returns Query result with implementations array
 *
 * @example
 * const { data: implementations } = useCategoryImplementations('model');
 * // implementations = [{ name: 'resnet18', ... }, { name: 'unet', ... }]
 */
export function useCategoryImplementations(category: string | null) {
  return useQuery({
    queryKey: ['categoryImplementations', category],
    queryFn: async (): Promise<Implementation[]> => {
      if (!category) return [];

      const implementations = await apiClient.getComponentsByCategory(category);
      return implementations || [];
    },
    enabled: !!category,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    gcTime: 30 * 60 * 1000,   // Keep in cache for 30 minutes
  });
}

/**
 * Fetch implementations for multiple categories at once.
 * Returns a map of category -> implementations.
 */
export function useMultipleCategoryImplementations(categories: string[]) {
  // Filter out empty/null categories and deduplicate
  const uniqueCategories = [...new Set(categories.filter(Boolean))];

  return useQuery({
    queryKey: ['categoryImplementations', 'multiple', uniqueCategories.sort().join(',')],
    queryFn: async () => {
      const results = await Promise.all(
        uniqueCategories.map(async (category) => {
          try {
            const implementations = await apiClient.getComponentsByCategory(category);
            return {
              category,
              implementations: implementations || [],
            };
          } catch (e) {
            console.error(`[useCategoryImplementations] Error fetching ${category}:`, e);
            return { category, implementations: [] };
          }
        })
      );

      // Convert to map
      const map: Record<string, Implementation[]> = {};
      for (const { category, implementations } of results) {
        map[category] = implementations;
      }
      return map;
    },
    enabled: uniqueCategories.length > 0,
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
  });
}
