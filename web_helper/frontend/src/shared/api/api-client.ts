/**
 * API client with typed responses and error handling
 */

import type {
  Project,
  Device,
  HeartbeatData,
  ApiResponse as ApiResponseType,
  Config,
} from './types';
import type {
  ApiComponentInfo,
  ExperimentResponse,
  TimeSeriesMetricsPoint,
} from './api-types';
import type { TimeSeriesPoint } from './charts/data-types';

// API response types based on RFC 7807 standard
export interface ApiResponse<T> {
  data: T;
  meta: {
    timestamp: string;
    version: string;
    [key: string]: unknown;
  };
}

export interface ApiError {
  type: string;
  title: string;
  status: number;
  detail: string;
  instance?: string;
  errors?: Array<{ message: string; field?: string; code?: string }>;
}

class ApiClient {
  private baseUrl = '/api';

  private async requestRaw<T>(endpoint: string, options?: RequestInit): Promise<T> {
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

      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(`Network error: ${error}`);
    }
  }

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

  async getRaw<T>(endpoint: string, options?: { signal?: AbortSignal }): Promise<T> {
    return this.requestRaw<T>(endpoint, options?.signal ? { signal: options.signal } : undefined);
  }

  // Projects API
  async getProjects(): Promise<Project[]> {
    return this.request<Project[]>('/projects/');
  }

  async getProjectExperiments(projectName: string): Promise<{
    project: string;
    experiment_count: number;
    experiments: ExperimentResponse[];
  }> {
    return this.request(`/projects/${encodeURIComponent(projectName)}/experiments`);
  }

  // Devices API
  async getDevices(): Promise<Device[]> {
    return this.request<Device[]>('/devices/');
  }

  async sendHeartbeat(heartbeatData: HeartbeatData): Promise<{ message: string; host_id: string }> {
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
  }): Promise<{ jobs: unknown[]; total: number }> {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.set('status', params.status);
    if (params?.project) searchParams.set('project', params.project);
    if (params?.limit) searchParams.set('limit', params.limit.toString());
    if (params?.offset) searchParams.set('offset', params.offset.toString());

    const query = searchParams.toString();
    return this.request(`/queue/list${query ? `?${query}` : ''}`);
  }

  async getQueueStats(): Promise<unknown> {
    return this.request('/queue/stats');
  }

  async submitJob(submission: unknown): Promise<{ job: unknown }> {
    return this.request('/queue/submit', {
      method: 'POST',
      body: JSON.stringify(submission),
    });
  }

  async getJob(jobId: string): Promise<{ job: unknown }> {
    return this.request(`/queue/job/${jobId}`);
  }

  async cancelJob(jobId: string): Promise<{ message: string }> {
    return this.request(`/queue/experiment/${jobId}/cancel`, {
      method: 'POST',
    });
  }

  async pauseJob(jobId: string): Promise<{ message: string }> {
    return this.request(`/queue/experiment/${jobId}/pause`, {
      method: 'POST',
    });
  }

  async resumeJob(jobId: string): Promise<{ message: string }> {
    return this.request(`/queue/experiment/${jobId}/resume`, {
      method: 'POST',
    });
  }

  async setJobPriority(jobId: string, priority: string): Promise<{ message: string }> {
    return this.request(`/queue/experiment/${jobId}/priority`, {
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
  async getAllComponents(): Promise<ApiComponentInfo[]> {
    return this.request<ApiComponentInfo[]>('/ui/');
  }

  async getComponentsByCategory(category: string): Promise<ApiComponentInfo[]> {
    return this.request<ApiComponentInfo[]>(`/ui/${category}`);
  }

  async searchComponents(query: string): Promise<ApiComponentInfo[]> {
    return this.request<ApiComponentInfo[]>(`/ui/search?q=${encodeURIComponent(query)}`);
  }

  async getComponentDetails(category: string, name: string): Promise<ApiComponentInfo> {
    return this.request<ApiComponentInfo>(`/ui/${name}`);
  }

  async validateComponentConfig(config: Config): Promise<{ valid: boolean; errors?: string[] }> {
    return this.request('/ui/validate-config', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async validateConfig(yamlConfig: string): Promise<{ valid: boolean; errors?: string[] }> {
    return this.request('/configs/validate', {
      method: 'POST',
      body: JSON.stringify({ config: yamlConfig }),
    });
  }

  // Metrics API (for future implementation)
  async getMetrics(project: string, runId: string, options?: {
    downsample?: number;
    startTime?: string;
    endTime?: string;
  }): Promise<TimeSeriesPoint[]> {
    const params = new URLSearchParams();
    if (options?.downsample) params.set('downsample', options.downsample.toString());
    if (options?.startTime) params.set('start', options.startTime);
    if (options?.endTime) params.set('end', options.endTime);

    const query = params.toString();
    const endpoint = `/metrics/${project}/${runId}${query ? `?${query}` : ''}`;

    return this.request<TimeSeriesPoint[]>(endpoint);
  }

  // Run Details API - Timeseries Metrics
  async getRunMetrics(
    project: string,
    runName: string,
    options?: { downsample?: number; signal?: AbortSignal }
  ): Promise<{
    data: TimeSeriesPoint[];
    total_steps: number;
    file_path: string;
    columns: string[];
  }> {
    const params = new URLSearchParams();
    if (options?.downsample) params.set('downsample', options.downsample.toString());
    const query = params.toString();
    return this.request(
      `/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/metrics${query ? `?${query}` : ''}`,
      options?.signal ? { signal: options.signal } : undefined
    );
  }

  async getRunConfig(
    project: string,
    runName: string,
    options?: { signal?: AbortSignal }
  ): Promise<{
    content: string;
    file_path: string;
    file_size: number;
    last_modified: number;
  }> {
    return this.request(
      `/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/config`,
      options?.signal ? { signal: options.signal } : undefined
    );
  }

  async getRunLogs(
    project: string,
    runName: string,
    options?: { signal?: AbortSignal }
  ): Promise<{
    content: string;
    files: Array<{ name: string; path: string; size?: number }>;
    total_files: number;
  }> {
    return this.request(
      `/runs/${encodeURIComponent(project)}/${encodeURIComponent(runName)}/logs`,
      options?.signal ? { signal: options.signal } : undefined
    );
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

  // ===========================================================================
  // Node Graph API - Draft Management
  // ===========================================================================

  /**
   * Create a new draft for editing an agent's node graph
   */
  async createDraft(agentName: string, basePath: string = ''): Promise<{
    success: boolean;
    draft_id: string;
    agent_name: string;
    status: string;
    created_at: string;
  }> {
    const params = new URLSearchParams();
    if (basePath) params.set('base_path', basePath);
    const query = params.toString();
    return this.request(`/nodes/hierarchy/${encodeURIComponent(agentName)}/draft${query ? `?${query}` : ''}`, {
      method: 'POST',
    });
  }

  /**
   * Get the current state of a draft
   */
  async getDraft(agentName: string, draftId: string): Promise<{
    success: boolean;
    draft: unknown;
  }> {
    return this.request(`/nodes/hierarchy/${encodeURIComponent(agentName)}/draft/${draftId}`);
  }

  /**
   * Delete a node from the graph
   * @param agentName - Agent file name
   * @param nodeId - Node ID to delete
   * @param draftId - Optional draft ID for temporary changes. If omitted, modifies code directly.
   */
  async deleteNode(agentName: string, nodeId: string, draftId?: string): Promise<{
    success: boolean;
    node_id: string;
    mode: 'draft' | 'direct';
    edit?: unknown;
    error?: string;
  }> {
    const params = new URLSearchParams();
    if (draftId) params.set('draft_id', draftId);
    const query = params.toString();
    return this.request(
      `/nodes/hierarchy/${encodeURIComponent(agentName)}/nodes/${encodeURIComponent(nodeId)}${query ? `?${query}` : ''}`,
      { method: 'DELETE' }
    );
  }

  /**
   * Delete an edge from the graph
   * @param agentName - Agent file name
   * @param edgeId - Edge ID to delete
   * @param draftId - Optional draft ID for temporary changes
   */
  async deleteEdge(agentName: string, edgeId: string, draftId?: string): Promise<{
    success: boolean;
    edge_id: string;
    mode: 'draft' | 'direct';
    edit?: unknown;
    error?: string;
  }> {
    const params = new URLSearchParams();
    if (draftId) params.set('draft_id', draftId);
    const query = params.toString();
    return this.request(
      `/nodes/hierarchy/${encodeURIComponent(agentName)}/${encodeURIComponent(edgeId)}${query ? `?${query}` : ''}`,
      { method: 'DELETE' }
    );
  }

  /**
   * Add a new node to the graph
   * @param agentName - Agent file name
   * @param request - Node creation request with category, implementation, role, config, position
   * @param draftId - Optional draft ID for temporary changes. If omitted, modifies code directly.
   */
  async addNode(
    agentName: string,
    request: {
      category: string;
      implementation: string;
      role?: string;
      config?: Config;
      position?: { x: number; y: number };
    },
    draftId?: string
  ): Promise<{
    success: boolean;
    node_id?: string;
    role?: string;
    mode: 'draft' | 'direct';
    edit?: unknown;
    error?: string;
  }> {
    const params = new URLSearchParams();
    if (draftId) params.set('draft_id', draftId);
    const query = params.toString();
    return this.request(
      `/nodes/hierarchy/${encodeURIComponent(agentName)}/nodes${query ? `?${query}` : ''}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      }
    );
  }

  /**
   * Update an existing node
   * @param agentName - Agent file name
   * @param nodeId - Node ID to update
   * @param request - Update request with implementation, config, position
   * @param draftId - Optional draft ID for temporary changes
   */
  async updateNode(
    agentName: string,
    nodeId: string,
    request: {
      implementation?: string;
      config?: Record<string, any>;
      position?: { x: number; y: number };
    },
    draftId?: string
  ): Promise<{
    success: boolean;
    node_id?: string;
    mode: 'draft' | 'direct';
    edit?: unknown;
    error?: string;
  }> {
    const params = new URLSearchParams();
    if (draftId) params.set('draft_id', draftId);
    const query = params.toString();
    return this.request(
      `/nodes/hierarchy/${encodeURIComponent(agentName)}/nodes/${encodeURIComponent(nodeId)}${query ? `?${query}` : ''}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      }
    );
  }

  /**
   * Add a new edge between nodes
   * @param agentName - Agent file name
   * @param request - Edge creation request with source, target, ports, flow_type
   * @param draftId - Optional draft ID for temporary changes
   */
  async addEdge(
    agentName: string,
    request: {
      source: string;
      target: string;
      source_port?: string;
      target_port?: string;
      flow_type?: string;
    },
    draftId?: string
  ): Promise<{
    success: boolean;
    edge_id?: string;
    mode: 'draft' | 'direct';
    code_change?: string;
    edit?: unknown;
    error?: string;
  }> {
    const params = new URLSearchParams();
    if (draftId) params.set('draft_id', draftId);
    const query = params.toString();
    return this.request(
      `/nodes/hierarchy/${encodeURIComponent(agentName)}/edges${query ? `?${query}` : ''}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      }
    );
  }

  /**
   * Commit a draft, applying all changes to the actual code
   */
  async commitDraft(agentName: string, draftId: string, description: string = ''): Promise<{
    success: boolean;
    version_id?: string;
    version_number?: number;
    code_changes?: any[];
    error?: string;
  }> {
    const params = new URLSearchParams();
    if (description) params.set('description', description);
    const query = params.toString();
    return this.request(
      `/nodes/hierarchy/${encodeURIComponent(agentName)}/draft/${draftId}/commit${query ? `?${query}` : ''}`,
      { method: 'POST' }
    );
  }

  /**
   * Discard a draft without committing
   */
  async discardDraft(agentName: string, draftId: string): Promise<{ success: boolean }> {
    return this.request(
      `/nodes/hierarchy/${encodeURIComponent(agentName)}/draft/${draftId}`,
      { method: 'DELETE' }
    );
  }

  /**
   * Undo the last edit in a draft
   */
  async undoDraftEdit(agentName: string, draftId: string): Promise<{
    success: boolean;
    edit?: unknown;
  }> {
    return this.request(
      `/nodes/hierarchy/${encodeURIComponent(agentName)}/draft/${draftId}/undo`,
      { method: 'POST' }
    );
  }

  /**
   * Redo the last undone edit in a draft
   */
  async redoDraftEdit(agentName: string, draftId: string): Promise<{
    success: boolean;
    edit?: unknown;
  }> {
    return this.request(
      `/nodes/hierarchy/${encodeURIComponent(agentName)}/draft/${draftId}/redo`,
      { method: 'POST' }
    );
  }

  /**
   * Get version history for an agent
   */
  async getVersions(agentName: string): Promise<{
    success: boolean;
    versions: Array<{
      version_id: string;
      version_number: number;
      created_at: string;
      description: string;
      created_by: string;
    }>;
  }> {
    return this.request(`/nodes/hierarchy/${encodeURIComponent(agentName)}/versions`);
  }

  // ===========================================================================
  // Model Layers API (Drill-down Visualization)
  // ===========================================================================

  /**
   * Get Level 1 layer graph for a model.
   * Extracts nn.Module assignments from model __init__ for drill-down visualization.
   * @param agentName - Agent name (e.g., "classification")
   * @param impl - Model implementation name (e.g., "resnet18")
   */
  async getModelLayers(agentName: string, impl: string): Promise<{
    success: boolean;
    data: {
      id: string;
      label: string;
      level: string;
      nodes: Array<{
        id: string;
        label: string;
        level: string;
        category: string;
        can_drill: boolean;
        metadata: Record<string, any>;
      }>;
      edges: any[];
      hierarchy: {
        depth: number;
        path: Array<{
          level: string;
          label: string;
          node_id: string;
          graph_id: string;
          category?: string;
          implementation?: string;
        }>;
      };
    };
  }> {
    return this.request(`/nodes/model-layers/${encodeURIComponent(agentName)}?impl=${encodeURIComponent(impl)}`);
  }

  /**
   * Get Level 2 nested modules for a layer.
   * Extracts nested nn.* calls within a layer (e.g., Sequential children).
   * @param agentName - Agent name
   * @param layerName - Layer name (e.g., "encoder", "block")
   * @param impl - Model implementation name
   */
  async getLayerModules(agentName: string, layerName: string, impl: string): Promise<{
    success: boolean;
    data: {
      id: string;
      label: string;
      level: string;
      nodes: Array<{
        id: string;
        label: string;
        level: string;
        category: string;
        can_drill: boolean;
        metadata: Record<string, any>;
      }>;
      edges: any[];
      hierarchy: {
        depth: number;
        path: Array<{
          level: string;
          label: string;
          node_id: string;
          graph_id: string;
          category?: string;
          implementation?: string;
        }>;
      };
    };
  }> {
    return this.request(`/nodes/model-layers/${encodeURIComponent(agentName)}/${encodeURIComponent(layerName)}/modules?impl=${encodeURIComponent(impl)}`);
  }

  // ===========================================================================
  // Git History API
  // ===========================================================================

  /**
   * Get git commit history for an agent file
   */
  async getGitHistory(agentName: string, limit: number = 20): Promise<{
    success: boolean;
    file_path: string;
    commits: Array<{
      hash: string;
      full_hash: string;
      message: string;
      author: string;
      date: string;
      relative_date: string;
    }>;
    current_status: 'clean' | 'modified' | 'untracked' | 'unknown';
    error?: string;
  }> {
    return this.request(`/git/history/${encodeURIComponent(agentName)}?limit=${limit}`);
  }

  /**
   * Get file content at a specific git commit
   * @param commit - Commit hash or "working" for current working directory
   * @param agentName - Agent name
   */
  async getGitFile(commit: string, agentName: string): Promise<{
    success: boolean;
    commit: string;
    file_path: string;
    content: string;
    error?: string;
  }> {
    return this.request(`/git/file/${encodeURIComponent(commit)}/${encodeURIComponent(agentName)}`);
  }
}

export const apiClient = new ApiClient();
