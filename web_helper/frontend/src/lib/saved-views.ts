/**
 * Saved Views Management
 *
 * Allows users to save and load custom project views including:
 * - Selected runs
 * - Visible columns
 * - Pinned columns
 * - Sort state
 * - Filter settings
 */

export interface SavedView {
  id: string;
  name: string;
  projectName: string;
  createdAt: string;
  updatedAt: string;
  state: {
    selectedRuns: string[];
    flatten: boolean;
    diffOnly: boolean;
    visibleHyperparams: string[];
    visibleMetrics: string[];
    pinnedLeftHyperparams: string[];
    pinnedRightMetrics: string[];
  };
}

const STORAGE_KEY = 'cvlabkit_saved_views';

/**
 * Get all saved views from localStorage
 */
export function getAllSavedViews(): SavedView[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.error('Failed to load saved views:', error);
    return [];
  }
}

/**
 * Get saved views for a specific project
 */
export function getProjectSavedViews(projectName: string): SavedView[] {
  return getAllSavedViews().filter(view => view.projectName === projectName);
}

/**
 * Save a new view
 */
export function saveView(view: Omit<SavedView, 'id' | 'createdAt' | 'updatedAt'>): SavedView {
  const allViews = getAllSavedViews();

  const newView: SavedView = {
    ...view,
    id: `view_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };

  allViews.push(newView);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(allViews));

  return newView;
}

/**
 * Update an existing view
 */
export function updateView(id: string, updates: Partial<Omit<SavedView, 'id' | 'createdAt'>>): SavedView | null {
  const allViews = getAllSavedViews();
  const index = allViews.findIndex(v => v.id === id);

  if (index === -1) return null;

  const updatedView: SavedView = {
    ...allViews[index],
    ...updates,
    updatedAt: new Date().toISOString(),
  };

  allViews[index] = updatedView;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(allViews));

  return updatedView;
}

/**
 * Delete a saved view
 */
export function deleteView(id: string): boolean {
  const allViews = getAllSavedViews();
  const filtered = allViews.filter(v => v.id !== id);

  if (filtered.length === allViews.length) return false;

  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  return true;
}

/**
 * Export views as JSON
 */
export function exportViewsAsJSON(projectName?: string): string {
  const views = projectName
    ? getProjectSavedViews(projectName)
    : getAllSavedViews();

  return JSON.stringify(views, null, 2);
}

/**
 * Import views from JSON
 */
export function importViewsFromJSON(jsonString: string): number {
  try {
    const imported = JSON.parse(jsonString) as SavedView[];
    const allViews = getAllSavedViews();

    // Merge imported views (avoiding duplicates by ID)
    const existingIds = new Set(allViews.map(v => v.id));
    const newViews = imported.filter(v => !existingIds.has(v.id));

    const merged = [...allViews, ...newViews];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));

    return newViews.length;
  } catch (error) {
    console.error('Failed to import views:', error);
    return 0;
  }
}
