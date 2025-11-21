import React from "react";
import { useState, useMemo } from 'react';
import { useProjects } from '../../hooks/useProjects';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Input } from '../ui/input';
import { Search } from 'lucide-react'; // Assuming lucide-react is available for icons

interface ProjectSelectionProps {
  setActiveProject: (project: string | null) => void;
}

export function ProjectSelection({ setActiveProject }: ProjectSelectionProps) {
  const { data: projects = [], isLoading: projectsLoading } = useProjects();
  const [searchTerm, setSearchTerm] = useState('');

  const filteredProjects = useMemo(() => {
    return projects.filter(project =>
      project.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [projects, searchTerm]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Projects</h1>
        <p className="text-muted-foreground">
          Manage your experiment projects and view run details
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Select Project</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search projects..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>

          {projectsLoading ? (
            <div className="text-center p-8">
              <div className="animate-spin h-8 w-8 border-b-2 border-primary mx-auto" />
              <p className="text-muted-foreground mt-2">Loading projects...</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredProjects.length > 0 ? (
                filteredProjects.map((project) => (
                  <Card
                    key={project.name}
                    className="cursor-pointer hover:border-primary transition-colors"
                    onClick={() => setActiveProject(project.name)}
                  >
                    <CardContent className="p-4">
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <h3 className="font-semibold text-lg">{project.name}</h3>
                          <Badge variant="secondary">{project.run_count} runs</Badge>
                        </div>
                        <Button
                          onClick={(e) => { // Prevent card click from firing twice
                            e.stopPropagation();
                            setActiveProject(project.name);
                          }}
                          className="w-full"
                        >
                          View Details
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))
              ) : (
                <div className="col-span-full text-center p-8">
                  <p className="text-muted-foreground">No projects found matching "{searchTerm}".</p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
