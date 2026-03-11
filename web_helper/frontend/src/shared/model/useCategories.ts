/**
 * React hook for fetching and syncing category data from API
 *
 * This hook:
 * 1. Fetches categories from /api/nodes/categories
 * 2. Updates the CategoryRegistry singleton
 * 3. Provides loading/error state
 */

import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import {
  categoryRegistry,
  type CategoryInfo,
  type CategoryTheme,
} from "@/shared/lib/category-registry";

const API_BASE = "/api/nodes";

interface ApiCategoryResponse {
  categories: Array<{
    name: string;
    label: string;
    theme: CategoryTheme;
    drillable: boolean;
    description: string;
    has_analyzer: boolean;
  }>;
}

// Query key for categories
export const categoriesKey = ["categories"] as const;

/**
 * Fetch categories from API
 */
async function fetchCategories(): Promise<ApiCategoryResponse> {
  const response = await fetch(`${API_BASE}/categories`);

  if (!response.ok) {
    throw new Error(`Failed to fetch categories: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Hook to fetch and sync categories
 *
 * Call this once at app initialization to populate the registry
 */
export function useCategories() {
  const query = useQuery({
    queryKey: categoriesKey,
    queryFn: fetchCategories,
    staleTime: 30 * 60 * 1000, // 30 minutes - categories rarely change
    gcTime: 60 * 60 * 1000, // 1 hour
    retry: 2,
  });

  // Update registry when data changes
  useEffect(() => {
    if (query.data?.categories) {
      categoryRegistry.updateFromApi(query.data.categories);
    }
  }, [query.data]);

  return {
    categories: query.data?.categories || [],
    isLoading: query.isLoading,
    error: query.error,
    isInitialized: categoryRegistry.isInitialized(),
    refetch: query.refetch,
  };
}

/**
 * Hook to get a specific category's info
 * Uses the registry (no API call)
 */
export function useCategoryInfo(name: string): CategoryInfo {
  return categoryRegistry.getCategory(name);
}

/**
 * Hook to get a specific category's theme
 * Uses the registry (no API call)
 */
export function useCategoryTheme(name: string): CategoryTheme {
  return categoryRegistry.getTheme(name);
}

/**
 * Hook to check if a category is drillable
 * Uses the registry (no API call)
 */
export function useIsDrillable(name: string): boolean {
  return categoryRegistry.isDrillable(name);
}
