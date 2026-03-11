import type { ReactNode } from 'react';

import {
  ThemeProvider,
  ChartInteractionProvider,
  ComparisonProvider,
  ExecuteProvider,
} from '@/app/ui';

interface AppProvidersProps {
  children: ReactNode;
}

export function AppProviders({ children }: AppProvidersProps) {
  return (
    <ThemeProvider>
      <ChartInteractionProvider>
        <ComparisonProvider>
          <ExecuteProvider>{children}</ExecuteProvider>
        </ComparisonProvider>
      </ChartInteractionProvider>
    </ThemeProvider>
  );
}
