import React from "react";
import type { Project } from '../../lib/types.ts';
import { cn } from '../../lib/utils.ts';
import { Folder, Search } from 'lucide-react';

interface MainSidebarProps {
  activeView: string;
  projects: Project[];
  activeProject: string | null;
  setActiveProject: (projectName: string) => void;
  _isLoading?: boolean;
}

export function MainSidebar({ activeView, projects, activeProject, setActiveProject, _isLoading }: MainSidebarProps) {
  const renderProjectsView = () => (
    <div>
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xl font-bold">Projects</h2>
        <div className="relative mt-4">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search..."
            className="w-full pl-10 pr-4 py-2 rounded-lg bg-gray-100 border-transparent focus:border-violet-500 focus:bg-white focus:ring-0"
          />
        </div>
      </div>
      <nav className="flex-1 p-2 space-y-1">
        {projects.map((project) => (
          <button
            key={project.name}
            onClick={() => setActiveProject(project.name)}
            className={cn(
              "w-full flex items-center px-4 py-2 text-left rounded-lg transition-colors duration-200",
              activeProject === project.name
                ? "bg-violet-100 text-violet-700"
                : "hover:bg-gray-100"
            )}
          >
            <Folder className="h-5 w-5 mr-3 flex-shrink-0" />
            <span className="flex-1 truncate font-medium">{project.name}</span>
            <span className="text-xs font-medium px-2 py-1 rounded-full bg-gray-200 text-gray-600">
              {project.runs.length}
            </span>
          </button>
        ))}
      </nav>
    </div>
  );

  const renderDefaultView = (title: string) => (
      <div className="p-4">
          <h2 className="text-xl font-bold">{title}</h2>
          <p className="mt-4 text-gray-500">This section will show details for {title}.</p>
      </div>
  );

  const renderContent = () => {
    switch (activeView) {
      case 'projects':
        return renderProjectsView();
      case 'overview':
        return renderDefaultView('Overview');
      case 'devices':
        return renderDefaultView('Devices');
      case 'queue':
        return renderDefaultView('Queue');
      case 'settings':
        return renderDefaultView('Settings');
      default:
        return <div className="p-4"><h2 className="text-xl font-bold">Select a view</h2></div>;
    }
  }

  return (
    <div className="w-80 bg-white border-r border-gray-200 h-full overflow-y-auto">
      {renderContent()}
    </div>
  );
}
