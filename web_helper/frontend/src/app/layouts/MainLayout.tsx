import React from 'react';
import { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Monitor,
  Play,
  ClipboardList,
  BarChart3,
  Boxes,
} from 'lucide-react';

import { useNavigationStore } from '@/shared/model/navigationStore';
import { TabNavigation } from '@/widgets/TabNavigation';
import { SettingsModal } from '@/widgets/settings-modal';
import { NotificationPanel } from '@/shared/ui/NotificationPanel';

interface Tab {
  id: string;
  label: string;
  icon: React.ReactNode;
  description?: string;
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
    id: 'builder',
    label: 'Builder',
    description: 'Visual agent and config editor',
    icon: <Boxes size={20} />,
  },
  {
    id: 'execute',
    label: 'Execute',
    description: 'Run new experiments',
    icon: <Play size={20} />,
  },
  {
    id: 'experiments',
    label: 'Experiments',
    description: 'Queue management and results',
    icon: <ClipboardList size={20} />,
  },
  {
    id: 'projects',
    label: 'Projects',
    description: 'Compare experiments within projects',
    icon: <BarChart3 size={20} />,
  },
];

export function MainLayout() {
  const { activeTab } = useNavigationStore();
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const location = useLocation();
  const isBuilder = location.pathname.startsWith('/builder');

  return (
    <div className="flex flex-col h-screen w-full bg-background">
      <TabNavigation tabs={tabs} onOpenSettings={() => setShowSettingsModal(true)} />

      <main className="flex-1 overflow-hidden bg-background">
        <div className={isBuilder ? "h-full w-full" : "research-container h-full px-2 sm:px-4 md:px-6 lg:px-8 py-4 md:py-6 overflow-y-auto"}>
          <div className="animate-fade-in h-full" key={activeTab}>
            <Outlet />
          </div>
        </div>
      </main>

      <SettingsModal isOpen={showSettingsModal} onClose={() => setShowSettingsModal(false)} />
      <NotificationPanel />
    </div>
  );
}
