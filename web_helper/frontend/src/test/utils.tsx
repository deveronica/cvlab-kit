/**
 * Test utilities and helpers
 */

import { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

// Create a test query client with default options
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

// Test wrapper with providers
interface TestProvidersProps {
  children: React.ReactNode;
}

function TestProviders({ children }: TestProvidersProps) {
  const queryClient = createTestQueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  );
}

// Custom render function with providers
const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) =>
  render(ui, {
    wrapper: TestProviders,
    ...options,
  });

// Mock API responses
export const mockApiResponse = <T>(data: T) => ({
  data,
  meta: {
    timestamp: new Date().toISOString(),
    version: '1.0.0',
  },
});

// Mock error response
export const mockErrorResponse = (
  title: string,
  detail: string,
  status: number = 400
) => ({
  type: 'about:blank',
  title,
  status,
  detail,
  instance: '/test/error',
});

// Sample test data
export const sampleDeviceData = {
  host_id: 'test-device-1',
  gpu_util: 75.5,
  vram_used: 6000000000,
  vram_total: 8000000000,
  cpu_util: 45.2,
  memory_used: 12000000000,
  memory_total: 32000000000,
  status: 'online',
  last_heartbeat: '2023-01-01T12:00:00Z',
  torch_version: '2.0.0',
  cuda_version: '11.8',
};

export const sampleProjectData = {
  name: 'test-project',
  runs: [
    {
      run_name: 'run-123',
      status: 'completed',
      started_at: '2023-01-01T10:00:00Z',
      finished_at: '2023-01-01T11:00:00Z',
      project: 'test-project',
    },
    {
      run_name: 'run-124',
      status: 'running',
      started_at: '2023-01-01T11:30:00Z',
      finished_at: null,
      project: 'test-project',
    },
  ],
};

export const sampleComponentData = {
  name: 'resnet18',
  type: 'model',
  path: 'cvlabkit/component/model/resnet18.py',
  description: 'ResNet-18 model for image classification',
  parameters: {
    num_classes: {
      type: 'int',
      default: '1000',
      required: false,
    },
    pretrained: {
      type: 'bool',
      default: 'False',
      required: false,
    },
  },
  examples: [
    {
      name: 'cifar10_resnet18',
      type: 'model.resnet18',
      params: {
        num_classes: 10,
        pretrained: false,
      },
    },
  ],
};

export const sampleQueueJobData = {
  job_id: 'job-123',
  name: 'Test Job',
  project: 'test-project',
  config_path: '/path/to/config.yaml',
  status: 'queued',
  priority: 'normal',
  created_at: '2023-01-01T10:00:00Z',
  queued_at: '2023-01-01T10:00:01Z',
  started_at: null,
  completed_at: null,
  requirements: {
    cpu_cores: 4,
    memory_gb: 8.0,
    gpu_count: 1,
  },
  assigned_device: null,
  progress: 0.0,
  current_epoch: null,
  total_epochs: null,
  current_metrics: {},
  error_message: null,
  retry_count: 0,
  max_retries: 3,
  user: null,
  tags: ['test'],
  metadata: {},
};

// Export customRender as default render
export { customRender as render };

// Re-export everything from testing-library
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';