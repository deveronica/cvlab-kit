import React from "react";

/**
 * Chart Theme Context
 *
 * Provides global chart theming functionality.
 * Allows users to select and persist chart color schemes across the application.
 */

import { createContext, useContext, useState, useEffect } from 'react';
import { ChartTheme, getTheme, CHART_THEMES } from '../lib/chart-themes';
import { devWarn } from '../lib/dev-utils';

interface ChartThemeContextType {
  currentTheme: ChartTheme;
  themeName: string;
  setTheme: (themeName: string) => void;
  availableThemes: typeof CHART_THEMES;
}

const ChartThemeContext = createContext<ChartThemeContextType | undefined>(undefined);

const STORAGE_KEY = 'cvlabkit-chart-theme';

export function ChartThemeProvider({ children }: { children: React.ReactNode }) {
  // Load theme from localStorage or use default
  const [themeName, setThemeName] = useState<string>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved || 'default';
    } catch {
      return 'default';
    }
  });

  const [currentTheme, setCurrentTheme] = useState<ChartTheme>(() => getTheme(themeName));

  // Update theme when themeName changes
  useEffect(() => {
    const theme = getTheme(themeName);
    setCurrentTheme(theme);

    // Persist to localStorage
    try {
      localStorage.setItem(STORAGE_KEY, themeName);
    } catch (error) {
      devWarn('Failed to save theme preference:', error);
    }
  }, [themeName]);

  const value: ChartThemeContextType = {
    currentTheme,
    themeName,
    setTheme: setThemeName,
    availableThemes: CHART_THEMES,
  };

  return (
    <ChartThemeContext.Provider value={value}>
      {children}
    </ChartThemeContext.Provider>
  );
}

/**
 * Hook to access chart theme context
 *
 * @example
 * ```tsx
 * function MyChart() {
 *   const { currentTheme, setTheme } = useChartTheme();
 *
 *   return (
 *     <div>
 *       <select onChange={(e) => setTheme(e.target.value)}>
 *         <option value="default">Default</option>
 *         <option value="publication">Publication</option>
 *       </select>
 *       <LineChart colors={currentTheme.colors} />
 *     </div>
 *   );
 * }
 * ```
 */
export function useChartTheme() {
  const context = useContext(ChartThemeContext);
  if (!context) {
    throw new Error('useChartTheme must be used within ChartThemeProvider');
  }
  return context;
}
