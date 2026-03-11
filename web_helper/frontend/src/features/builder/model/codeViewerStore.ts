/**
 * Code Viewer Store - Zustand store for side panel code viewer
 *
 * Manages:
 * - Panel open/close state
 * - Current file path and content
 * - Line highlighting (covered/uncovered)
 * - Selected range (when clicking a node)
 */

import { create } from 'zustand';
import type { SourceLocation } from '@/entities/node-system/model/types';

interface HighlightedLines {
  covered: number[];
  uncovered: number[];
}

interface CodeViewerState {
  // Panel state
  isOpen: boolean;

  // Current file
  filePath: string | null;
  content: string | null;
  isLoading: boolean;
  error: string | null;

  // Highlighting
  highlightedLines: HighlightedLines;
  selectedRange: { start: number; end: number } | null;

  // Actions
  openViewer: (source: SourceLocation) => void;
  closeViewer: () => void;
  setContent: (content: string) => void;
  setError: (error: string) => void;
  setHighlightedLines: (covered: number[], uncovered: number[]) => void;
  selectRange: (start: number, end: number) => void;
  clearSelection: () => void;
  reset: () => void;
}

const initialState = {
  isOpen: false,
  filePath: null,
  content: null,
  isLoading: false,
  error: null,
  highlightedLines: { covered: [], uncovered: [] },
  selectedRange: null,
};

export const useCodeViewerStore = create<CodeViewerState>((set, get) => ({
  ...initialState,

  openViewer: (source: SourceLocation) => {
    const { filePath } = get();

    // If same file, just update selection
    if (filePath === source.file) {
      set({
        isOpen: true,
        selectedRange: {
          start: source.line,
          end: source.endLine || source.line,
        },
      });
      return;
    }

    // New file - reset and load
    set({
      isOpen: true,
      filePath: source.file,
      content: null,
      isLoading: true,
      error: null,
      selectedRange: {
        start: source.line,
        end: source.endLine || source.line,
      },
    });
  },

  closeViewer: () => {
    set({ isOpen: false });
  },

  setContent: (content: string) => {
    set({ content, isLoading: false, error: null });
  },

  setError: (error: string) => {
    set({ error, isLoading: false, content: null });
  },

  setHighlightedLines: (covered: number[], uncovered: number[]) => {
    set({ highlightedLines: { covered, uncovered } });
  },

  selectRange: (start: number, end: number) => {
    set({ selectedRange: { start, end } });
  },

  clearSelection: () => {
    set({ selectedRange: null });
  },

  reset: () => {
    set(initialState);
  },
}));

/**
 * Hook to fetch file content when viewer opens
 */
export function useCodeViewerContent() {
  const { filePath, isLoading, setContent, setError } = useCodeViewerStore();

  const fetchContent = async (path: string) => {
    try {
      // API call to fetch file content
      const response = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`);

      if (!response.ok) {
        throw new Error(`Failed to load file: ${response.statusText}`);
      }

      const data = await response.json();
      setContent(data.content);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load file');
    }
  };

  return { fetchContent, filePath, isLoading };
}
