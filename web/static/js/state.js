// --- Globals & State ---
export let globalRunsForComparison = new Map();
export let isMetricsFlattened = false;
export let isRunsFlattened = false;
export let runsVisibleColumns = new Set();
export let currentProjectDetail = null;
export const chartColors = ['#0D6EFD', '#198754', '#DC3545', '#FFC107', '#0DCAF0', '#6F42C1', '#FD7E14'];

// --- State Manipulation Functions ---
export function updateComparisonState(checkbox) {
    const path = checkbox.dataset.csvPath;
    if (checkbox.checked) {
        if (!globalRunsForComparison.has(path)) {
            const color = chartColors[globalRunsForComparison.size % chartColors.length];
            globalRunsForComparison.set(path, { 
                name: checkbox.dataset.runName, 
                configPath: checkbox.dataset.configPath, 
                csvPath: path,
                color: color
            });
        }
    } else {
        globalRunsForComparison.delete(path);
    }
}

export function removeRunFromComparison(path) {
    globalRunsForComparison.delete(path);
}

export function updateRunColor(runId, newColor) {
    // Assuming runId is the csvPath, which is the key in globalRunsForComparison
    if (globalRunsForComparison.has(runId)) {
        const run = globalRunsForComparison.get(runId);
        run.color = newColor;
        globalRunsForComparison.set(runId, run); // Update the map with the modified run object
    }
}


export function getComparisonList() {
    return Array.from(globalRunsForComparison.values());
}

export function getComparisonCount() {
    return globalRunsForComparison.size;
}

export function toggleMetricsFlatten() {
    isMetricsFlattened = !isMetricsFlattened;
    return isMetricsFlattened;
}

export function getIsMetricsFlattened() {
    return isMetricsFlattened;
}

export function toggleRunsFlatten() {
    isRunsFlattened = !isRunsFlattened;
    return isRunsFlattened;
}

export function getIsRunsFlattened() {
    return isRunsFlattened;
}

export function initializeVisibleColumns(columns) {
    runsVisibleColumns = new Set(columns);
}

export function getVisibleColumns() {
    return runsVisibleColumns;
}

export function setCurrentProject(project) {
    currentProjectDetail = project;
}

export function getCurrentProject() {
    return currentProjectDetail;
}

export function clearCurrentProject() {
    currentProjectDetail = null;
}
