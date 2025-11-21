import { useState, useEffect, useCallback } from 'react';
import { devWarn, devError } from '../lib/dev-utils';

interface UserSettings {
  theme: 'light' | 'dark' | 'system';
  dashboardLayout: 'compact' | 'expanded';
  defaultChartType: 'line' | 'bar' | 'scatter';
  autoRefresh: boolean;
  refreshInterval: number; // in seconds
  defaultTimeRange: '1h' | '6h' | '24h' | '7d' | '30d';
  notifications: {
    experimentComplete: boolean;
    experimentFailed: boolean;
    systemAlerts: boolean;
  };
  comparison: {
    maxRuns: number;
    defaultMetrics: string[];
    showDifferencesOnly: boolean;
  };
  tableSettings: {
    pageSize: number;
    sortBy: string;
    sortOrder: 'asc' | 'desc';
  };
}

const DEFAULT_SETTINGS: UserSettings = {
  theme: 'system',
  dashboardLayout: 'expanded',
  defaultChartType: 'line',
  autoRefresh: true,
  refreshInterval: 30,
  defaultTimeRange: '24h',
  notifications: {
    experimentComplete: true,
    experimentFailed: true,
    systemAlerts: false,
  },
  comparison: {
    maxRuns: 5,
    defaultMetrics: ['loss', 'accuracy'],
    showDifferencesOnly: false,
  },
  tableSettings: {
    pageSize: 20,
    sortBy: 'started_at',
    sortOrder: 'desc',
  },
};

const STORAGE_KEY = 'cvlab-user-settings';

export function useUserSettings() {
  const [settings, setSettings] = useState<UserSettings>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        // Merge with defaults to handle new settings
        return { ...DEFAULT_SETTINGS, ...parsed };
      }
    } catch (error) {
      devWarn('Failed to load user settings from localStorage:', error);
    }
    return DEFAULT_SETTINGS;
  });

  const [isLoading, setIsLoading] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);

  // Save settings to localStorage
  const saveSettings = useCallback(async (newSettings: Partial<UserSettings>) => {
    setIsLoading(true);
    try {
      const updatedSettings = { ...settings, ...newSettings };
      setSettings(updatedSettings);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedSettings));
      setLastSaved(new Date());

      // Optionally sync with server
      try {
        await fetch('/api/user/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updatedSettings),
        });
      } catch (error) {
        devWarn('Failed to sync settings with server:', error);
      }
    } catch (error) {
      devError('Failed to save user settings:', error);
    } finally {
      setIsLoading(false);
    }
  }, [settings]);

  // Reset to defaults
  const resetSettings = useCallback(async () => {
    setIsLoading(true);
    try {
      setSettings(DEFAULT_SETTINGS);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_SETTINGS));
      setLastSaved(new Date());
    } catch (error) {
      devError('Failed to reset user settings:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Import/Export settings
  const exportSettings = useCallback(() => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `cvlab-settings-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [settings]);

  const importSettings = useCallback(async (file: File) => {
    try {
      const text = await file.text();
      const importedSettings = JSON.parse(text);
      await saveSettings(importedSettings);
      return true;
    } catch (error) {
      devError('Failed to import settings:', error);
      return false;
    }
  }, [saveSettings]);

  // Individual setting updaters
  const updateTheme = useCallback((theme: UserSettings['theme']) => {
    saveSettings({ theme });
  }, [saveSettings]);

  const updateDashboardLayout = useCallback((dashboardLayout: UserSettings['dashboardLayout']) => {
    saveSettings({ dashboardLayout });
  }, [saveSettings]);

  const updateNotifications = useCallback((notifications: Partial<UserSettings['notifications']>) => {
    saveSettings({ notifications: { ...settings.notifications, ...notifications } });
  }, [saveSettings, settings.notifications]);

  const updateComparison = useCallback((comparison: Partial<UserSettings['comparison']>) => {
    saveSettings({ comparison: { ...settings.comparison, ...comparison } });
  }, [saveSettings, settings.comparison]);

  const updateTableSettings = useCallback((tableSettings: Partial<UserSettings['tableSettings']>) => {
    saveSettings({ tableSettings: { ...settings.tableSettings, ...tableSettings } });
  }, [saveSettings, settings.tableSettings]);

  // Load settings from server on mount
  useEffect(() => {
    const loadServerSettings = async () => {
      try {
        const response = await fetch('/api/user/settings');
        if (response.ok) {
          const serverSettings = await response.json();
          if (serverSettings.data) {
            const mergedSettings = { ...DEFAULT_SETTINGS, ...serverSettings.data };
            setSettings(mergedSettings);
            localStorage.setItem(STORAGE_KEY, JSON.stringify(mergedSettings));
          }
        }
      } catch (error) {
        devWarn('Failed to load settings from server:', error);
      }
    };

    loadServerSettings();
  }, []);

  return {
    settings,
    isLoading,
    lastSaved,
    saveSettings,
    resetSettings,
    exportSettings,
    importSettings,
    updateTheme,
    updateDashboardLayout,
    updateNotifications,
    updateComparison,
    updateTableSettings,
  };
}