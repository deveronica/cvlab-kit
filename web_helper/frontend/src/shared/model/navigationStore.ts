import { create } from 'zustand';

interface NavigationState {
  activeTab: string;
  selectedExperimentUid: string | null;
  selectedProject: string | null;
  setActiveTab: (tabId: string) => void;
  navigateToExperiment: (experimentUid: string) => void;
  navigateToProject: (projectName: string) => void;
}

export const useNavigationStore = create<NavigationState>((set) => ({
  activeTab: 'dashboard',
  selectedExperimentUid: null,
  selectedProject: null,

  setActiveTab: (tabId) => set({ activeTab: tabId }),

  navigateToExperiment: (experimentUid) => set({
    selectedExperimentUid: experimentUid,
    activeTab: 'results', // Automatically switch to results tab for a specific experiment
  }),

  navigateToProject: (projectName) => set({
    selectedProject: projectName,
    activeTab: 'projects', // Automatically switch to projects tab
  }),
}));
