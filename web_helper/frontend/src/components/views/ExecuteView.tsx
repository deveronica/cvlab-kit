import React, { useEffect, useCallback, memo } from 'react';
import { useQuery } from '@tanstack/react-query';
import CodeMirror from '@uiw/react-codemirror';
import { yaml } from '@codemirror/lang-yaml';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { useDevices } from '../../hooks/useDevices';
import { useTheme } from '../../contexts/ThemeContext';
import { useExecute } from '../../contexts/ExecuteContext';

// API fetch functions
const fetchConfigList = async (): Promise<string[]> => {
  const response = await fetch('/api/configs');
  if (!response.ok) {
    throw new Error('Failed to fetch config list');
  }
  const data = await response.json();
  return data.data || [];
};

const fetchConfigContent = async (configPath: string): Promise<string> => {
  const response = await fetch(`/api/configs/${encodeURIComponent(configPath)}`);
  if (!response.ok) {
    throw new Error('Failed to fetch config content');
  }
  const data = await response.json();
  return data.data?.content || '';
};

const ExecuteView = memo(function ExecuteView() {
  const { theme } = useTheme();
  const {
    state,
    updateYamlConfig,
    updateSelectedConfigFile,
    updateSelectedDevice,
    updateSelectedPriority,
    updateLogMessages,
  } = useExecute();

  const { data: configFiles = [], isLoading: isLoadingConfigs, error: configsError } = useQuery<string[], Error>({ queryKey: ['configFiles'], queryFn: fetchConfigList });
  const { data: devices = [], isLoading: isLoadingDevices } = useDevices();

  const [isStarting, setIsStarting] = React.useState(false);

  const { data: configContent, isLoading: isLoadingContent } = useQuery<string | null, Error>({
    queryKey: ['configContent', state.selectedConfigFile],
    queryFn: () => fetchConfigContent(state.selectedConfigFile),
    enabled: !!state.selectedConfigFile,
  });

  useEffect(() => {
    if (configContent) {
      updateYamlConfig(configContent);
      updateLogMessages([]); // Clear logs on new content
    }
  }, [configContent, updateYamlConfig, updateLogMessages]);

  const onYamlChange = useCallback((value: string) => {
    updateYamlConfig(value);
  }, [updateYamlConfig]);

  const handleStartExperiment = async () => {
    setIsStarting(true);
    updateLogMessages(['Starting experiment...']);

    try {
      const response = await fetch('/api/queue/add', { // Corrected endpoint
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          config: state.yamlConfig,
          device: state.selectedDevice,
          priority: state.selectedPriority,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail?.detail || result.detail || 'Failed to add experiment to queue');
      }

      // Handle the JobResponse format: {"job": {...}}
      const job = result.job || result;
      updateLogMessages([
        ...state.logMessages,
        '✅ Successfully added to the queue.',
        `Job ID: ${job.job_id}`,
        `Status: ${job.status}`,
        `Project: ${job.project}`
      ]);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      updateLogMessages([...state.logMessages, `❌ Error: ${errorMessage}`]);
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-6">
      <div className="flex-shrink-0">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Execute</h1>
        <p className="text-muted-foreground">Create and run new experiments</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 flex-1 min-h-0">
        {/* Configuration Editor (takes 2/3 width) */}
        <div className="xl:col-span-2 flex flex-col xl:min-h-0">
            <Card className="flex flex-col xl:h-full">
              <CardHeader className="flex-shrink-0">
                <CardTitle>Configuration</CardTitle>
                <CardDescription>Define your experiment parameters using YAML</CardDescription>
              </CardHeader>
              <CardContent className="xl:flex-1 flex flex-col xl:min-h-0">
                <div className="flex flex-col xl:h-full space-y-4">
                  <div className="flex-shrink-0">
                    <label htmlFor="config-select" className="text-sm font-medium">Load from file</label>
                    <select
                      id="config-select"
                      value={state.selectedConfigFile}
                      onChange={(e) => updateSelectedConfigFile(e.target.value)}
                      disabled={isLoadingConfigs}
                      className="w-full mt-2 p-2 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-primary focus:border-transparent"
                    >
                      <option value="" disabled>{isLoadingConfigs ? 'Loading configs...' : 'Select a config file'}</option>
                      {configsError && <option value="" disabled>Error loading configs</option>}
                      {configFiles.map(file => (<option key={file} value={file}>{file}</option>))}
                    </select>
                  </div>

                  <div className="flex-1 flex flex-col min-h-0">
                    <label className="text-sm font-medium flex-shrink-0">YAML Configuration</label>
                    <div className="mt-2 flex-1 min-h-0">
                      <CodeMirror
                        value={isLoadingContent ? 'Loading...' : state.yamlConfig}
                        height="100%"
                        extensions={[yaml()]}
                        onChange={onYamlChange}
                        className="w-full h-full border border-border rounded-lg overflow-hidden"
                        theme={theme}
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
        </div>

        {/* Execution Options (takes 1/3 width) */}
        <div className="space-y-4 md:space-y-6 flex flex-col xl:h-full xl:min-h-0">
            <Card className="flex-shrink-0">
              <CardHeader>
                <CardTitle>Execution Options</CardTitle>
                <CardDescription>Configure how to run it</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Target Device</label>
                    <select
                      value={state.selectedDevice}
                      onChange={(e) => updateSelectedDevice(e.target.value)}
                      disabled={isLoadingDevices}
                      className="w-full mt-2 p-2 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-primary focus:border-transparent"
                    >
                      <option value="any">Any Available</option>
                      {devices.flatMap(device => {
                        const statusEmoji = device.status === 'healthy' ? '🟢' : device.status === 'stale' ? '🟡' : '🔴';

                        // Multi-GPU: create option for each GPU
                        if (device.gpu_count && device.gpu_count > 1 && device.gpus && device.gpus.length > 0) {
                          return device.gpus.map(gpu => (
                            <option key={`${device.host_id}:${gpu.id}`} value={`${device.host_id}:${gpu.id}`}>
                              {statusEmoji} {device.host_id} - GPU {gpu.id} ({gpu.name})
                            </option>
                          ));
                        }

                        // Single GPU or no GPU info
                        const gpuInfo = device.gpus && device.gpus.length > 0
                          ? device.gpus[0].name
                          : 'No GPU';
                        return (
                          <option key={device.host_id} value={device.host_id}>
                            {statusEmoji} {device.host_id} - {gpuInfo}
                          </option>
                        );
                      })}
                    </select>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Priority</label>
                    <select
                      value={state.selectedPriority}
                      onChange={(e) => updateSelectedPriority(e.target.value)}
                      className="w-full mt-2 p-2 text-sm border border-border rounded-lg bg-background focus:ring-2 focus:ring-primary focus:border-transparent"
                    >
                      <option value="normal">🟢 Normal</option>
                      <option value="high">🔴 High</option>
                      <option value="low">🔵 Low</option>
                    </select>
                  </div>

                  <Button
                    onClick={handleStartExperiment}
                    disabled={isStarting || !state.yamlConfig.trim()}
                    variant="default"
                    size="lg"
                    className="w-full"
                  >
                    {isStarting ? '⏳ Adding to Queue...' : '🚀 Add to Queue'}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Log Panel */}
            <Card className="flex flex-col flex-1 min-h-0">
                <CardHeader className="flex-shrink-0">
                    <CardTitle>Log</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col min-h-0 pb-4">
                    <div className="bg-muted/80 p-3 rounded-lg flex-1 min-h-0 overflow-y-auto text-xs md:text-sm font-mono">
                        {state.logMessages.length > 0 ? (
                            state.logMessages.map((msg, i) => <div key={i}>{`> ${msg}`}</div>)
                        ) : (
                            <div className="text-muted-foreground">Awaiting experiment submission...</div>
                        )}
                    </div>
                </CardContent>
            </Card>
        </div>
      </div>
    </div>
  );
});

export { ExecuteView };