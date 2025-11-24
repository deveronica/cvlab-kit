import React from "react";

import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import {
  LayoutDashboard,
  Monitor,
  Play,
  List,
  ClipboardList,
  BarChart3,
  Settings,
  Package,
} from 'lucide-react';
import { useNavigationStore } from '@/store/navigationStore';
import { TabNavigation } from '../navigation/TabNavigation';
import { SettingsModal } from '../modals/settings-modal';
import { NotificationPanel } from '../notifications/NotificationPanel';

interface Tab {
  id: string;
  label: string;
  icon: React.ReactNode;
  description?: string;
}

interface MainLayoutProps {
  _children?: React.ReactNode;
}

const tabs: Tab[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    description: 'System overview and navigation',
    icon: <LayoutDashboard size={20} />,
  },
  {
    id: 'monitoring',
    label: 'Monitoring',
    description: 'Monitor compute resources',
    icon: <Monitor size={20} />,
  },
  {
    id: 'execute',
    label: 'Execute',
    description: 'Run new experiments',
    icon: <Play size={20} />,
  },
  {
    id: 'queue',
    label: 'Queue',
    description: 'Manage experiment queue',
    icon: <List size={20} />,
  },
  {
    id: 'results',
    label: 'Results',
    description: 'View experiment execution timeline',
    icon: <ClipboardList size={20} />,
  },
  {
    id: 'projects',
    label: 'Projects',
    description: 'Compare experiments within projects',
    icon: <BarChart3 size={20} />,
  },
  {
    id: 'components',
    label: 'Components',
    description: 'Manage versioned components',
    icon: <Package size={20} />,
  },
];

export function MainLayout({ _children }: MainLayoutProps) {
  const { activeTab } = useNavigationStore();
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  return (
    <div className="flex flex-col h-screen w-full bg-background">
      <TabNavigation
        tabs={tabs}
        onOpenSettings={() => setShowSettingsModal(true)}
      />

      {/* Main content area */}
      <main className="flex-1 overflow-y-auto bg-background">
        <div className="research-container h-full px-2 sm:px-4 md:px-6 lg:px-8 py-4 md:py-6">
          <div className="animate-fade-in h-full" key={activeTab}>
            <Outlet />
          </div>
        </div>
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
      />

      {/* Notification Panel (slides from right) */}
      <NotificationPanel />
    </div>
  );
}
