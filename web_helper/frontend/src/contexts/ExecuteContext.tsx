/**
 * Execute Context for persisting Execute tab state across navigation
 * State is preserved during tab switches but reset on page refresh
 */

import { createContext, useContext, useState, ReactNode } from 'react';

interface ExecuteState {
  yamlConfig: string;
  selectedConfigFile: string;
  selectedDevice: string;
  selectedPriority: string;
  logMessages: string[];
}

interface ExecuteContextType {
  state: ExecuteState;
  updateYamlConfig: (config: string) => void;
  updateSelectedConfigFile: (file: string) => void;
  updateSelectedDevice: (device: string) => void;
  updateSelectedPriority: (priority: string) => void;
  updateLogMessages: (messages: string[]) => void;
  addLogMessage: (message: string) => void;
  clearLogs: () => void;
  resetState: () => void;
}

const defaultState: ExecuteState = {
  yamlConfig: '# Select a configuration file or start typing...',
  selectedConfigFile: '',
  selectedDevice: 'any',
  selectedPriority: 'normal',
  logMessages: [],
};

const ExecuteContext = createContext<ExecuteContextType | undefined>(undefined);

export function ExecuteProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ExecuteState>(defaultState);

  const updateYamlConfig = (config: string) => {
    setState(prev => ({ ...prev, yamlConfig: config }));
  };

  const updateSelectedConfigFile = (file: string) => {
    setState(prev => ({ ...prev, selectedConfigFile: file }));
  };

  const updateSelectedDevice = (device: string) => {
    setState(prev => ({ ...prev, selectedDevice: device }));
  };

  const updateSelectedPriority = (priority: string) => {
    setState(prev => ({ ...prev, selectedPriority: priority }));
  };

  const updateLogMessages = (messages: string[]) => {
    setState(prev => ({ ...prev, logMessages: messages }));
  };

  const addLogMessage = (message: string) => {
    setState(prev => ({ ...prev, logMessages: [...prev.logMessages, message] }));
  };

  const clearLogs = () => {
    setState(prev => ({ ...prev, logMessages: [] }));
  };

  const resetState = () => {
    setState(defaultState);
  };

  return (
    <ExecuteContext.Provider
      value={{
        state,
        updateYamlConfig,
        updateSelectedConfigFile,
        updateSelectedDevice,
        updateSelectedPriority,
        updateLogMessages,
        addLogMessage,
        clearLogs,
        resetState,
      }}
    >
      {children}
    </ExecuteContext.Provider>
  );
}

export function useExecute() {
  const context = useContext(ExecuteContext);
  if (context === undefined) {
    throw new Error('useExecute must be used within an ExecuteProvider');
  }
  return context;
}
