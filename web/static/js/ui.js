import * as State from './state.js';
import { flattenObject, formatValue, formatDuration, formatTimestamp, getTextWidth } from './utils.js';

export function adjustColumnWidths(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const headers = Array.from(table.querySelectorAll('thead th'));
    const rows = Array.from(table.querySelectorAll('tbody tr'));
    const buffer = 10; // Small buffer for extra space

    headers.forEach((header, i) => {
        if (header.classList.contains('actions-col') || header.classList.contains('checkbox-col')) return;

        let maxWidth = 0;

        // Measure header content
        const headerContentSpan = header.querySelector('.th-content span');
        if (headerContentSpan) {
            // Temporarily set white-space to nowrap to get true content width
            const originalWhiteSpace = headerContentSpan.style.whiteSpace;
            headerContentSpan.style.whiteSpace = 'nowrap';
            maxWidth = headerContentSpan.scrollWidth;
            headerContentSpan.style.whiteSpace = originalWhiteSpace; // Restore original
        }

        // Measure cell content
        rows.forEach(row => {
            const cell = row.cells[i];
            if (cell) {
                const pre = cell.querySelector('pre');
                let contentElement = pre || cell; // Use pre if exists, otherwise the cell itself

                // Temporarily set white-space to nowrap for accurate measurement
                const originalWhiteSpace = contentElement.style.whiteSpace;
                contentElement.style.whiteSpace = 'nowrap';
                const cellWidth = contentElement.scrollWidth;
                contentElement.style.whiteSpace = originalWhiteSpace; // Restore original

                if (cellWidth > maxWidth) {
                    maxWidth = cellWidth;
                }
            }
        });
        
        header.style.width = `${maxWidth + buffer}px`;
    });
}

export function updateComparisonBadge() {
    const badge = document.getElementById('comparison-count-badge');
    const count = State.getComparisonCount();
    badge.textContent = count;
    badge.style.display = count > 0 ? 'inline-block' : 'none';
}

export function renderDiffTable(containerId, runs, paramValues) {
    const container = document.getElementById(containerId);
    const runNames = runs.map(run => run.name);
    let tableHtml = `<table class="table table-dark table-bordered diff-table"><thead><tr><th class="wrap-text">Parameter</th>`;
    runNames.forEach(name => tableHtml += `<th class="wrap-text">${name}</th>`);
    tableHtml += `</tr></thead><tbody>`;

    paramValues.forEach((values, key) => {
        const isDifferent = new Set(values).size > 1;
        const rowClass = isDifferent ? 'diff-row' : '';
        tableHtml += `<tr class="${rowClass}"><td class="wrap-text"><strong>${key}</strong></td>`;
        // Add param-value-cell class to the td for values
        values.forEach(value => tableHtml += `<td class="param-value-cell"><pre>${value}</pre></td>`);
        tableHtml += `</tr>`;
    });
    container.innerHTML = tableHtml + `</tbody></table>`;
}

export function renderHistoryTable(groups) {
    const historyTableBody = document.getElementById('history-table-body');
    historyTableBody.innerHTML = '';
    groups.forEach(group => {
        const isRunning = group.status === 'running';
        const statusClass = `status-${group.status.toLowerCase()}`;
        
        let statusColor = 'primary';
        if (group.status === 'completed') statusColor = 'success';
        else if (group.status === 'failed') statusColor = 'danger';
        else if (group.status === 'terminated') statusColor = 'secondary';
        else if (group.status === 'unknown') statusColor = 'warning';
        
        const statusBadge = `<span class="badge rounded-pill bg-${statusColor} ${isRunning ? 'status-running' : ''}">${group.status}</span>`;

        const stopButtonHtml = (isRunning && group.pid)
            ? `<button class="btn btn-sm btn-danger stop-group-btn" data-group-id="${group.id}" data-pid="${group.pid}"><i class="bi bi-stop-circle"></i> Stop</button>`
            : '';
            
        const logButtonHtml = group.terminal_log_path ? `<button class="btn btn-sm btn-outline-warning view-log-btn" data-path="${group.terminal_log_path}" data-status="${group.status}"><i class="bi bi-terminal"></i> Log</button>` : '';
        const configButtonHtml = `<button class="btn btn-sm btn-outline-info view-config-btn" data-path="${group.launch_config_path}"><i class="bi bi-file-earmark-code"></i> Config</button>`;

        const finishedAt = formatTimestamp(group.finished_at);
        const duration = (group.finished_at && group.created_at) ? formatDuration(new Date(group.finished_at) - new Date(group.created_at)) : 'N/A';

        const row = `
            <tr class="align-middle history-group-row">
                <td class="wrap-text name-col">
                    <a href="#" class="view-log-btn" data-path="${group.terminal_log_path}" data-status="${group.status}">
                        <strong>${group.name}</strong>
                    </a>
                </td>
                <td>${statusBadge}</td>
                <td>${formatTimestamp(group.created_at)}</td>
                <td>${finishedAt}</td>
                <td>${duration}</td>
                <td class="text-end">
                    <div class="btn-group btn-group-sm">
                        ${stopButtonHtml}
                        ${configButtonHtml}
                        ${logButtonHtml}
                    </div>
                </td>
            </tr>`;
        historyTableBody.innerHTML += row;
    });
}


export function renderProjectsList(experiments) {
    const projectList = document.getElementById('runs-project-list');
    projectList.innerHTML = '';
    if (experiments.length === 0) {
        projectList.innerHTML = '<p class="text-muted">No completed runs found.</p>';
        return;
    }
    experiments.forEach((exp) => {
        const projectItem = document.createElement('a');
        projectItem.href = '#';
        projectItem.className = 'list-group-item list-group-item-action bg-dark text-white project-list-item';
        projectItem.dataset.experiment = JSON.stringify(exp);
        projectItem.innerHTML = `<div class="d-flex w-100 justify-content-between align-items-center">
                                    <h5 class="mb-1 wrap-text">${exp.name}</h5>
                                    <small class="text-muted d-flex align-items-center"><i class="bi bi-bar-chart-fill me-2"></i>${exp.runs.length} Runs</small>
                                 </div>`;
        projectList.appendChild(projectItem);
    });
}

export function renderRunDetailsTable(table, fullRunData, isFlattened) {
    const tableHead = table.querySelector('thead');
    const tableBody = table.querySelector('tbody');

    const allParams = new Set();
    const parsedData = fullRunData.map(run => {
        try {
            const original = jsyaml.load(run.configContent) || {};
            return { ...run, original, flattened: flattenObject(original) };
        } catch(e){ 
            console.error("YAML Parse Error:", e); 
            return { ...run, original: {}, flattened: {} }; 
        } 
    });

    parsedData.forEach(run => {
        const configToIterate = isFlattened ? run.flattened : run.original;
        Object.keys(configToIterate).forEach(key => allParams.add(key));
    });

    const sortedParams = Array.from(allParams).sort();

    const differingParams = new Set();
    sortedParams.forEach(param => {
        const values = new Set(parsedData.map(run => {
            const config = isFlattened ? run.flattened : run.original;
            return JSON.stringify(config[param]);
        }));
        if (values.size > 1) differingParams.add(param);
    });
    table.dataset.diffParams = JSON.stringify(Array.from(differingParams));

    let headHtml = `<tr class="title-row"><th class="checkbox-col"><i class="bi bi-check2-square"></i></th><th class="name-col sortable" data-column-index="1"><div class="th-content"><i class="bi bi-arrow-down-up sort-icon"></i><span>Run Name</span></div><div class="resizer"></div></th>`;
    sortedParams.forEach((param, i) => {
        headHtml += `<th class="param-col sortable" data-column-index="${i+2}" data-param="${param}"><div class="th-content"><i class="bi bi-arrow-down-up sort-icon"></i><span>${param}</span></div><div class="resizer"></div></th>`
    });
    headHtml += '<th class="actions-col">Actions</th></tr>';
    tableHead.innerHTML = headHtml;

    let bodyHtml = '';
    parsedData.forEach((run) => {
        const isChecked = State.globalRunsForComparison.has(run.path);
        
        bodyHtml += `<tr>
                        <td class="text-center checkbox-col">
                            <input class="form-check-input run-comparison-checkbox" type="checkbox" ${isChecked ? 'checked' : ''} 
                                   data-run-name="${run.name}" data-config-path="${run.configPath}" data-csv-path="${run.path}">
                        </td>
                        <td class="wrap-text"><strong>${run.name}</strong></td>`;
        sortedParams.forEach(param => {
            const config = isFlattened ? run.flattened : run.original;
            const value = formatValue(config[param]);
            bodyHtml += `<td data-param="${param}" class="wrap-text"><pre>${value}</pre></td>`;
        });
        bodyHtml += `<td class="btn-group-sm actions-col">
                        <button class="btn btn-outline-info py-0 view-config-btn" data-path="${run.configPath}">Config</button>
                        <button class="btn btn-outline-primary py-0 plot-csv-btn" data-path="${run.path}">Plot</button>
                    </td>`;
        bodyHtml += '</tr>';
    });
    tableBody.innerHTML = bodyHtml;
    document.getElementById('runs-diff-only-toggle').checked = false;
    adjustColumnWidths(table.id);
}

export function renderColumnSelector(params) {
    const menu = document.getElementById('runs-column-selector-menu');
    menu.innerHTML = '';
    State.initializeVisibleColumns(params);
    params.forEach(param => {
        const li = document.createElement('li');
        li.innerHTML = `<a class="dropdown-item" href="#">
                            <input class="form-check-input me-2" type="checkbox" checked value="${param}">
                            <span>${param}</span>
                        </a>`;
        menu.appendChild(li);
    });
}

export function renderSelectedRunsList() {
    const listContainer = document.getElementById('selected-runs-list');
    listContainer.innerHTML = '';
    const runs = State.getComparisonList();
    if (runs.length === 0) {
        listContainer.innerHTML = '<p class="text-muted small">No runs selected.</p>';
        return;
    }
    runs.forEach(run => {
        const item = document.createElement('div');
        item.className = 'list-group-item d-flex justify-content-between align-items-center';
        item.innerHTML = `
            <span class="run-name-text small">${run.name}</span>
            <div class="d-flex align-items-center">
                <div class="color-circle me-2" style="background-color: ${run.color};" data-csv-path="${run.csvPath}" title="Click to change color"></div>
                <input type="color" class="color-picker-hidden" value="${run.color}" data-csv-path="${run.csvPath}" style="visibility: hidden; width: 0; height: 0; border: none; padding: 0;">
                <button class="btn-close ms-2 remove-run-btn" data-csv-path="${run.csvPath}"></button>
            </div>
        `;
        listContainer.appendChild(item);

        const colorCircle = item.querySelector('.color-circle');
        const colorInput = item.querySelector('.color-picker-hidden');

        // When the visible color circle is clicked, trigger the hidden input's click
        colorCircle.addEventListener('click', () => {
            colorInput.click();
        });

        // When the hidden input's value changes, update the State and the visible circle
        colorInput.addEventListener('change', (e) => {
            const newColor = e.target.value;
            State.updateRunColor(run.csvPath, newColor); // Assuming State.updateRunColor exists and handles persistence
            colorCircle.style.backgroundColor = newColor; // Update the visible circle
        });
    });
}


export function renderMetricsChart(runData, metricsChartInstance, options) {
    const datasets = runData.map((run) => {
        const lines = run.csvContent.trim().split('\n');
        if (lines.length < 2) return null;
        const rows = lines.slice(1).map(line => line.split(','));
        const lastColumn = rows.map(r => r.length > 0 ? parseFloat(r[r.length - 1]) : null).filter(v => v !== null && !isNaN(v));
        return { 
            label: run.name, 
            data: lastColumn, 
            borderColor: options.colors[run.csvPath], 
            tension: options.smoothedLines ? 0.4 : 0, 
            pointRadius: options.showPoints ? 3 : 0,
            fill: false 
        };
    }).filter(d => d);

    const maxLen = Math.max(0, ...datasets.map(d => d.data.length));
    const labels = Array.from({length: maxLen}, (_, i) => i + 1);

    const ctx = document.getElementById('metrics-chart').getContext('2d');
    if (metricsChartInstance) metricsChartInstance.destroy();
    
    return new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: { 
            responsive: true, 
            maintainAspectRatio: false, 
            scales: { x: { ticks: { color: '#ADB5BD'}}, y: { ticks: { color: '#ADB5BD'}}}, 
            plugins: { legend: { labels: { color: '#E9ECEF'}}}}});
}
