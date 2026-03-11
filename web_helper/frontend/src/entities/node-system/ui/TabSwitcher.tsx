/**
 * TabSwitcher - Toggle between Execute Tab and Builder Tab
 *
 * ADR-008: 2-Tab System
 * - Execute Tab: Config Graph (setup dependencies)
 * - Builder Tab: Data Flow Graph (Input → Loss)
 */

import { memo } from 'react';
import { Settings, Workflow } from 'lucide-react';
import { cn } from '@/shared/lib/utils';
import type { TabMode } from '@/entities/node-system/model/types';

interface TabSwitcherProps {
  currentTab: TabMode;
  onTabChange: (tab: TabMode) => void;
  className?: string;
}

export const TabSwitcher = memo(function TabSwitcher({
  currentTab,
  onTabChange,
  className,
}: TabSwitcherProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-1 p-1 bg-background border rounded-lg shadow-sm',
        className
      )}
    >
      <button
        onClick={() => onTabChange('execute')}
        className={cn(
          'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
          currentTab === 'execute'
            ? 'bg-primary text-primary-foreground'
            : 'text-muted-foreground hover:text-foreground hover:bg-muted'
        )}
        title="Config graph: setup dependencies (primarily shown in Execute page)"
      >
        <Settings className="h-4 w-4" />
        <span>Config</span>
      </button>

      <button
        onClick={() => onTabChange('builder')}
        className={cn(
          'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
          currentTab === 'builder'
            ? 'bg-primary text-primary-foreground'
            : 'text-muted-foreground hover:text-foreground hover:bg-muted'
        )}
        title="Flow graph: data/execution flow (primarily edited in Builder page)"
      >
        <Workflow className="h-4 w-4" />
        <span>Flow</span>
      </button>
    </div>
  );
});

TabSwitcher.displayName = 'TabSwitcher';
