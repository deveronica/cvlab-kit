import { useState, useEffect, useCallback } from 'react';

export interface PanelState {
  leftSize: number;
  rightSize: number;
  leftCollapsed: boolean;
  rightCollapsed: boolean;
}

const DEFAULT_PANEL_STATE: PanelState = {
  leftSize: 20,
  rightSize: 25,
  leftCollapsed: false,
  rightCollapsed: false,
};

export function usePanelPersistence(layoutKey: string) {
  const storageKey = `workbench-layout-${layoutKey}`;

  // Initialize from storage or default
  const [state, setState] = useState<PanelState>(() => {
    if (typeof window === 'undefined') return DEFAULT_PANEL_STATE;
    try {
      const stored = localStorage.getItem(storageKey);
      return stored ? { ...DEFAULT_PANEL_STATE, ...JSON.parse(stored) } : DEFAULT_PANEL_STATE;
    } catch (e) {
      console.warn('Failed to load panel state', e);
      return DEFAULT_PANEL_STATE;
    }
  });

  // Save to storage whenever state changes
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
    } catch (e) {
      console.warn('Failed to save panel state', e);
    }
  }, [state, storageKey]);

  const setLeftSize = useCallback((size: number) => {
    setState(prev => ({ ...prev, leftSize: size }));
  }, []);

  const setRightSize = useCallback((size: number) => {
    setState(prev => ({ ...prev, rightSize: size }));
  }, []);

  const setLeftCollapsed = useCallback((collapsed: boolean) => {
    setState(prev => ({ ...prev, leftCollapsed: collapsed }));
  }, []);

  const setRightCollapsed = useCallback((collapsed: boolean) => {
    setState(prev => ({ ...prev, rightCollapsed: collapsed }));
  }, []);

  return {
    ...state,
    setLeftSize,
    setRightSize,
    setLeftCollapsed,
    setRightCollapsed,
  };
}
