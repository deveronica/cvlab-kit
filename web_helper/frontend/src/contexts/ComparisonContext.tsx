import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { Comparison, ComparisonSettings, Run } from '../lib/types';
import { generateComparisonName } from '../lib/comparison-utils';

interface ComparisonContextType {
  // Current active comparison
  activeComparison: Comparison | null;
  setActiveComparison: (comparison: Comparison | null) => void;

  // Saved comparisons
  savedComparisons: Comparison[];
  setSavedComparisons: (comparisons: Comparison[]) => void;

  // Comparison creation
  createComparison: (runs: Run[], settings?: Partial<ComparisonSettings>) => Comparison;
  updateComparison: (id: string, updates: Partial<Comparison>) => void;
  deleteComparison: (id: string) => void;

  // Quick comparison (temporary, not saved)
  quickCompare: (runs: Run[], settings?: Partial<ComparisonSettings>) => void;

  // UI state
  showCompareModal: boolean;
  setShowCompareModal: (show: boolean) => void;
  selectedRuns: Run[];
  setSelectedRuns: (runs: Run[]) => void;
}

const ComparisonContext = createContext<ComparisonContextType | undefined>(undefined);

export function useComparison() {
  const context = useContext(ComparisonContext);
  if (context === undefined) {
    throw new Error('useComparison must be used within a ComparisonProvider');
  }
  return context;
}

interface ComparisonProviderProps {
  children: ReactNode;
}

export function ComparisonProvider({ children }: ComparisonProviderProps) {
  const [activeComparison, setActiveComparison] = useState<Comparison | null>(null);
  const [savedComparisons, setSavedComparisons] = useState<Comparison[]>([]);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [selectedRuns, setSelectedRuns] = useState<Run[]>([]);

  const createComparison = useCallback((runs: Run[], settings?: Partial<ComparisonSettings>): Comparison => {
    const defaultSettings: ComparisonSettings = {
      viewType: 'summary',
      flattenParams: false,
      diffMode: false,
      selectedMetrics: [],
      selectedParams: [],
    };

    const comparison: Comparison = {
      id: `comp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: generateComparisonName(runs),
      project: runs[0]?.project || '',
      runs: runs.map(r => r.run_name),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      settings: { ...defaultSettings, ...settings },
    };

    return comparison;
  }, []);

  const updateComparison = useCallback((id: string, updates: Partial<Comparison>) => {
    setSavedComparisons(prev =>
      prev.map(comp =>
        comp.id === id
          ? { ...comp, ...updates, updated_at: new Date().toISOString() }
          : comp
      )
    );

    // Update active comparison if it's the one being updated
    if (activeComparison?.id === id) {
      setActiveComparison(prev =>
        prev ? { ...prev, ...updates, updated_at: new Date().toISOString() } : null
      );
    }
  }, [activeComparison]);

  const deleteComparison = useCallback((id: string) => {
    setSavedComparisons(prev => prev.filter(comp => comp.id !== id));

    // Clear active comparison if it's the one being deleted
    if (activeComparison?.id === id) {
      setActiveComparison(null);
    }
  }, [activeComparison]);

  const quickCompare = useCallback((runs: Run[], settings?: Partial<ComparisonSettings>) => {
    const comparison = createComparison(runs, settings);
    setActiveComparison(comparison);
    setSelectedRuns(runs);
    setShowCompareModal(true);
  }, [createComparison]);

  const value: ComparisonContextType = {
    activeComparison,
    setActiveComparison,
    savedComparisons,
    setSavedComparisons,
    createComparison,
    updateComparison,
    deleteComparison,
    quickCompare,
    showCompareModal,
    setShowCompareModal,
    selectedRuns,
    setSelectedRuns,
  };

  return (
    <ComparisonContext.Provider value={value}>
      {children}
    </ComparisonContext.Provider>
  );
}