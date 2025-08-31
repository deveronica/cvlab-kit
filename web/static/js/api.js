export async function fetchFile(path) {
    if (!path || path === 'null' || path === 'undefined') {
        throw new Error(`File path is invalid: ${path}`);
    }
    const response = await fetch(`/files/${path}`);
    if (!response.ok) {
        throw new Error(`File not found: ${path}`);
    }
    return await response.json();
}

export async function fetchConfigs() {
    const response = await fetch('/configs');
    if (!response.ok) {
        throw new Error('Failed to fetch configs');
    }
    return await response.json();
}

export async function fetchExperimentGroups() {
    const response = await fetch('/api/v1/experiment_groups');
    if (!response.ok) {
        throw new Error('Failed to fetch experiment groups');
    }
    return await response.json();
}

export async function fetchVisualizerExperiments() {
    const response = await fetch('/api/v1/visualizer/experiments');
    if (!response.ok) {
        throw new Error('Failed to fetch visualizer experiments');
    }
    return await response.json();
}

export async function stopExperimentGroup(groupId) {
    const response = await fetch(`/api/v1/experiment_groups/${groupId}/stop`, {
        method: 'POST',
    });
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to stop experiment group');
    }
    return await response.json();
}
