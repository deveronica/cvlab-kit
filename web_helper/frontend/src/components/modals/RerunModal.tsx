import React from "react";
import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import CodeMirror from '@uiw/react-codemirror';
import { yaml } from '@codemirror/lang-yaml';
import { Dialog, DialogContent, DialogTitle, DialogDescription, DialogFooter, DialogTrigger } from '../ui/dialog';
import { Button } from '../ui/button';
import { X } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

type Run = {
  run_name: string;
  project: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  config_path: string;
};

const fetchConfigFile = async (configPath: string) => {
    const response = await fetch(`/api/configs/${configPath}`);
    if (!response.ok) {
        throw new Error('Failed to fetch config file');
    }
    const data = await response.json();
    return data.data.content;
};

const startExperiment = async (config: string) => {
    const response = await fetch('/api/queue/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config, device: 'any', priority: 'normal' }),
    });
    if (!response.ok) {
        throw new Error('Failed to start experiment');
    }
    const result = await response.json();
    // Handle RFC 7807 compliant response format
    return result.data || result;
};

export function RerunModal({ run }: { run: Run }) {
  const [isOpen, setIsOpen] = useState(false);
  const [config, setConfig] = useState('');
  const queryClient = useQueryClient();
  const { theme } = useTheme();

  const { data: configContent, isLoading: isLoadingConfig } = useQuery({
      queryKey: ['config', run.config_path],
      queryFn: () => fetchConfigFile(run.config_path),
      enabled: isOpen,
  });

  useEffect(() => {
      if (configContent) {
          setConfig(configContent);
      }
  }, [configContent]);

  const mutation = useMutation({ 
      mutationFn: startExperiment, 
      onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ['runs'] });
          setIsOpen(false);
          alert('Rerun started successfully!');
      },
      onError: () => {
          alert('Failed to rerun experiment.');
      }
  });

  const onConfigChange = useCallback((value: string) => {
    setConfig(value);
  }, []);

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="default" size="sm">
          Edit & Rerun
        </Button>
      </DialogTrigger>
      <DialogContent className="w-full max-w-4xl h-[90vh] overflow-y-auto flex flex-col gap-4" hideCloseButton>
        <div className="flex flex-col space-y-1.5 text-left">
          <div className="flex items-start justify-between gap-4">
            <DialogTitle>Edit & Rerun: {run.run_name}</DialogTitle>
            <Button variant="ghost" size="sm" onClick={() => {}} aria-label="Close">
              <X className="h-4 w-4" />
            </Button>
          </div>
          <DialogDescription>
            Modify the configuration and rerun this experiment
          </DialogDescription>
        </div>

        {isLoadingConfig ? (
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            Loading configuration...
          </div>
        ) : (
          <div className="space-y-4">
            <CodeMirror
              value={config}
              height="400px"
              extensions={[yaml()]}
              onChange={onConfigChange}
              theme={theme}
            />
          </div>
        )}

        <DialogFooter>
          <Button
            onClick={() => mutation.mutate(config)}
            className="w-full"
            disabled={mutation.isPending || isLoadingConfig}
          >
            {mutation.isPending ? 'Starting...' : 'Rerun Experiment'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}