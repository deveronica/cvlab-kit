/**
 * API client with typed responses and error handling
 */

import type { Project, Device } from './types';

// API response types based on RFC 7807 standard
export interface ApiResponse<T> {
  data: T;
  meta: {
    timestamp: string;
    version: string;
    [key: string]: any;
  };
}

export interface ApiError {
  type: string;
  title: string;
  status: number;
  detail: string;
  instance?: string;
  errors?: any[];
}

class ApiClient {
  private baseUrl = '/api';

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok) {
        try {
          const error: ApiError = await response.json();
          throw new Error(`${error.title}: ${error.detail}`);
        } catch {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
      }

      const result = await response.json();

      // Handle both RFC 7807 and legacy response formats
      if (result && typeof result === 'object' && 'data' in result) {
        // RFC 7807 compliant response
        return result.data;
      } else {
        // Legacy response format
        return result;
      }
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(`Network error: ${error}`);
    }
  }

  // Projects API
  async getProjects(): Promise<Project[]> {
    return this.request<Project[]>('/projects/');
  }

  async getProjectExperiments(projectName: string): Promise<{
    project: string;
    experiment_count: number;
    experiments: any[];
  }> {
    return this.request(`/projects/${encodeURIComponent(projectName)}/experiments`);
  }

  // Devices API
  async getDevices(): Promise<Device[]> {
    return this.request<Device[]>('/devices/');
  }

  async sendHeartbeat(heartbeatData: any): Promise<{ message: string; host_id: string }> {
    return this.request('/devices/heartbeat', {
      method: 'POST',
      body: JSON.stringify(heartbeatData),
    });
  }

  // Advanced Queue API
  async listJobs(params?: {
    status?: string;
    project?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ jobs: any[]; total: number }> {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.set('status', params.status);
    if (params?.project) searchParams.set('project', params.project);
    if (params?.limit) searchParams.set('limit', params.limit.toString());
    if (params?.offset) searchParams.set('offset', params.offset.toString());

    const query = searchParams.toString();
    return this.request(`/queue/list${query ? `?${query}` : ''}`);
  }

  async getQueueStats(): Promise<any> {
    return this.request('/queue/stats');
  }

  async submitJob(submission: any): Promise<{ job: any }> {
    return this.request('/queue/submit', {
      method: 'POST',
      body: JSON.stringify(submission),
    });
  }

  async getJob(jobId: string): Promise<{ job: any }> {
    return this.request(`/queue/job/${jobId}`);
  }

  async cancelJob(jobId: string): Promise<{ message: string }> {
    return this.request(`/queue/job/${jobId}/cancel`, {
      method: 'POST',
    });
  }

  async pauseJob(jobId: string): Promise<{ message: string }> {
    return this.request(`/queue/job/${jobId}/pause`, {
      method: 'POST',
    });
  }

  async resumeJob(jobId: string): Promise<{ message: string }> {
    return this.request(`/queue/job/${jobId}/resume`, {
      method: 'POST',
    });
  }

  async setJobPriority(jobId: string, priority: string): Promise<{ message: string }> {
    return this.request(`/queue/job/${jobId}/priority`, {
      method: 'POST',
      body: JSON.stringify(priority),
    });
  }

  async cleanupOldJobs(hours: number = 168): Promise<{ message: string; cleaned_count: number }> {
    return this.request(`/queue/cleanup?hours=${hours}`, {
      method: 'POST',
    });
  }

  async reindex(): Promise<{ message: string; indexed: number }> {
    return this.request('/queue/reindex', {
      method: 'POST',
    });
  }

  // Components Discovery API
  async getAllComponents(): Promise<any[]> {
    return this.request('/components/');
  }

  async getComponentsByCategory(category: string): Promise<any[]> {
    return this.request(`/components/category/${category}`);
  }

  async searchComponents(query: string): Promise<any[]> {
    return this.request(`/components/search?q=${encodeURIComponent(query)}`);
  }

  async getComponentDetails(category: string, name: string): Promise<any> {
    return this.request(`/components/component/${category}/${name}`);
  }

  async validateComponentConfig(config: any): Promise<{ valid: boolean; errors?: string[] }> {
    return this.request('/components/validate-config', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // Metrics API (for future implementation)
  async getMetrics(project: string, runId: string, options?: {
    downsample?: number;
    startTime?: string;
    endTime?: string;
  }): Promise<any[]> {
    const params = new URLSearchParams();
    if (options?.downsample) params.set('downsample', options.downsample.toString());
    if (options?.startTime) params.set('start', options.startTime);
    if (options?.endTime) params.set('end', options.endTime);

    const query = params.toString();
    const endpoint = `/metrics/${project}/${runId}${query ? `?${query}` : ''}`;

    return this.request(endpoint);
  }

  // Run Details API - Timeseries Metrics
  async getRunMetrics(project: string, runName: string, downsample?: number): Promise<{
    data: any[];
    total_steps: number;
    file_path: string;
    columns: string[];
  }> {
    const params = new URLSearchParams();
    if (downsample) params.set('downsample', downsample.toString());
    const query = params.toString();
    return this.request(`/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/metrics${query ? `?${query}` : ''}`);
  }

  async getRunConfig(project: string, runName: string): Promise<{
    content: string;
    file_path: string;
    file_size: number;
    last_modified: number;
  }> {
    return this.request(`/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/config`);
  }

  async getRunLogs(project: string, runName: string): Promise<{
    content: string;
    files: any[];
    total_files: number;
  }> {
    return this.request(`/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/logs`);
  }

  async getRunArtifacts(project: string, runName: string): Promise<{
    run_name: string;
    project: string;
    artifacts: Array<{
      name: string;
      type: string;
      size: number;
      path: string;
      last_modified: number;
    }>;
    total_count: number;
  }> {
    return this.request(`/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/artifacts`);
  }

  async downloadArtifact(project: string, runName: string, filePath: string): Promise<void> {
    const url = `${this.baseUrl}/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/artifacts/download?file_path=${encodeURIComponent(filePath)}`;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }

      // Get filename from Content-Disposition header or use default
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = filePath.split('/').pop() || 'download';
      if (contentDisposition) {
        const matches = /filename="(.+)"/.exec(contentDisposition);
        if (matches && matches[1]) {
          filename = matches[1];
        }
      }

      // Create blob and download
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(`Download error: ${error}`);
    }
  }
}

export const apiClient = new ApiClient();