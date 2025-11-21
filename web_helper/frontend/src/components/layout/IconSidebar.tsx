import React from "react";
import {
  BarChart3,
  Folder,
  Computer,
  Clock,
  Package,
  Settings,
} from 'lucide-react'
import { cn } from '../../lib/utils.ts';

const navigationItems = [
  { id: 'overview', label: 'Overview', icon: BarChart3 },
  { id: 'projects', label: 'Projects', icon: Folder },
  { id: 'devices', label: 'Devices', icon: Computer },
  { id: 'components', label: 'Components', icon: Package },
  { id: 'queue', label: 'Queue', icon: Clock },
  { id: 'settings', label: 'Settings', icon: Settings },
]

interface IconSidebarProps {
  activeView: string;
  setActiveView: (view: string) => void;
}

export function IconSidebar({ activeView, setActiveView }: IconSidebarProps) {
  return (
    <div className="flex flex-col items-center w-16 py-4 space-y-4 bg-gray-800 text-white h-full border-r border-gray-700">
      {/* Logo */}
      <div className="p-2 rounded-lg bg-gray-700">
        <svg fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" className="h-6 w-6 text-white" viewBox="0 0 24 24">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
        </svg>
      </div>

      <nav className="flex flex-col items-center flex-1 space-y-2">
        {navigationItems.map((item) => {
          const Icon = item.icon
          return (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={cn(
                "p-3 rounded-lg hover:bg-gray-700 transition-colors duration-200",
                activeView === item.id && "bg-violet-600 hover:bg-violet-700"
              )}
              aria-label={item.label}
            >
              <Icon className="h-6 w-6" />
            </button>
          )
        })}
      </nav>
    </div>
  )
}
