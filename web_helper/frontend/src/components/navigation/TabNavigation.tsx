import React from "react";

import { NavLink } from 'react-router-dom';
import { Settings } from 'lucide-react';
import { ThemeToggle } from '../ui/theme-toggle';
import { Button } from '../ui/button';
import { NotificationButton } from '../notifications/NotificationPanel';

interface Tab {
  id: string;
  label: string;
  icon: React.ReactNode;
  description?: string;
}

interface TabNavigationProps {
  tabs: Tab[];
  onOpenSettings: () => void;
}

export function TabNavigation({ tabs, onOpenSettings }: TabNavigationProps) {
  return (
    <nav className="bg-card border-b border-border shadow-sm">
      <div className="research-container px-2 sm:px-4 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-2 md:gap-6">
          {/* Logo and Title */}
          <div className="flex items-center gap-3 min-w-0 flex-shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center shadow-md">
                <svg
                  className="w-5 h-5 text-primary-foreground"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 4 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-bold text-foreground">CVLab</h1>
                <p className="text-xs text-muted-foreground leading-tight">Experiment Platform</p>
              </div>
            </div>
          </div>

          {/* Responsive Tab Navigation */}
          <div className="flex-1 flex items-center justify-center max-w-4xl mx-1 md:mx-4 min-w-0">
            <div className="flex bg-muted rounded-lg p-1 space-x-1 overflow-x-auto scrollbar-hide max-w-full">
              {tabs.map((tab) => (
                <NavLink
                  key={tab.id}
                  to={tab.id === 'dashboard' ? '/' : `/${tab.id}`}
                  end={tab.id === 'dashboard'} // Dashboard: exact match only. Others: match nested routes (e.g., /projects/:projectName)
                  className={({ isActive }) => `
                    flex items-center justify-center px-2 md:px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200 flex-shrink-0
                    ${isActive
                      ? 'bg-background text-foreground shadow-sm'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10'
                    }
                  `}
                  title={tab.description}
                >
                  <span className="w-4 h-4 flex items-center justify-center">{tab.icon}</span>
                  <span className="ml-1 md:ml-2 hidden sm:inline leading-none text-xs md:text-sm whitespace-nowrap">{tab.label}</span>
                </NavLink>
              ))}
            </div>
          </div>

          {/* Theme Toggle, Notifications and Actions */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <ThemeToggle />
            <NotificationButton />
            <Button
              variant="ghost"
              size="icon"
              onClick={onOpenSettings}
              title="Settings"
              className="h-9 w-9"
            >
              <Settings size={18} />
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
}