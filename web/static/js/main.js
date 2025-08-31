import * as State from './state.js';
import * as API from './api.js';
import * as UI from './ui.js';
import * as Utils from './utils.js';

document.addEventListener('DOMContentLoaded', function() {
    // --- Globals & State ---
    const yamlCompareModal = new bootstrap.Modal(document.getElementById('yaml-compare-modal'));
    const textViewerModal = new bootstrap.Modal(document.getElementById('text-viewer-modal'));
    const plotViewerModal = new bootstrap.Modal(document.getElementById('plot-viewer-modal'));
    const editor = CodeMirror.fromTextArea(document.getElementById('config-content'), { mode: 'text/x-yaml', theme: 'dracula', lineNumbers: true, tabSize: 2 });
    let historyPoller = null;
    let singlePlotChart = null;
    let metricsChartInstance = null;
    let logPoller = null; // [BUG FIX] For real-time log viewing

    // --- Event Listeners ---
    document.querySelector('#v-pills-history-tab').addEventListener('shown.bs.tab', () => {
        loadHistory();
        if (!historyPoller) historyPoller = setInterval(loadHistory, 5000);
    });
    document.querySelector('#v-pills-history-tab').addEventListener('hidden.bs.tab', () => {
        if (historyPoller) { clearInterval(historyPoller); historyPoller = null; }
    });
    document.querySelector('#v-pills-runs-tab').addEventListener('shown.bs.tab', loadRuns);
    document.querySelector('#v-pills-metrics-tab').addEventListener('shown.bs.tab', renderMetricsComparison);
    document.getElementById('run-btn').addEventListener('click', launchExperiment);
    document.getElementById('config-select').addEventListener('change', (e) => loadConfigContent(e.target.value));
    document.getElementById('back-to-projects-btn').addEventListener('click', () => {
        State.clearCurrentProject();
        document.getElementById('runs-detail-view').style.display = 'none';
        document.getElementById('runs-list-view').style.display = 'block';
    });
    
    // [BUG FIX] Stop log polling when the modal is closed
    document.getElementById('text-viewer-modal').addEventListener('hidden.bs.modal', () => {
        if (logPoller) {
            clearInterval(logPoller);
            logPoller = null;
        }
    });


    document.body.addEventListener('change', event => {
        if (event.target.classList.contains('run-comparison-checkbox')) {
            State.updateComparisonState(event.target);
            UI.updateComparisonBadge();
        } else if (event.target.classList.contains('color-picker-hidden')) {
            // State update is handled in ui.js, just trigger re-render
            renderMetricsComparison();
        } else if (event.target.id === 'chart-show-points-toggle' || event.target.id === 'chart-smooth-lines-toggle') {
            renderMetricsComparison();
        }
    });
    
    document.body.addEventListener('click', event => {
        if (event.target.classList.contains('remove-run-btn')) {
            const path = event.target.dataset.csvPath;
            State.removeRunFromComparison(path);
            const checkbox = document.querySelector(`.run-comparison-checkbox[data-csv-path="${path}"]`);
            if (checkbox) checkbox.checked = false;
            UI.updateComparisonBadge();
            renderMetricsComparison();
        }
    });

    const setupDiffToggle = (toggleId, tableId) => {
        document.getElementById(toggleId).addEventListener('change', (e) => {
            const table = document.getElementById(tableId);
            if (!table) return;
            const isChecked = e.target.checked;
            if (table.dataset.diffParams) {
                const differingParams = new Set(JSON.parse(table.dataset.diffParams));
                table.querySelectorAll('thead th, tbody td').forEach(cell => {
                    const param = cell.dataset.param;
                    if (param && !differingParams.has(param)) {
                         cell.style.display = isChecked ? 'none' : 'table-cell';
                    }
                });
            } else { // Fallback for simple row-based diff
                 table.querySelectorAll('tbody tr:not(.diff-row)').forEach(row => {
                    row.style.display = isChecked ? 'none' : 'table-row';
                });
            }
        });
    };
    setupDiffToggle('show-diff-only-toggle', 'yaml-compare-table-container');
    setupDiffToggle('metrics-diff-only-toggle', 'metrics-diff-table-container');
    setupDiffToggle('runs-diff-only-toggle', 'runs-detail-table');

    document.getElementById('flatten-params-btn').addEventListener('click', (e) => {
        const isFlattened = State.toggleMetricsFlatten();
        e.target.textContent = isFlattened ? 'Unflatten' : 'Flatten';
        e.target.classList.toggle('btn-primary', isFlattened);
        e.target.classList.toggle('btn-secondary', !isFlattened);
        renderMetricsComparison();
    });
    
    document.getElementById('runs-flatten-params-btn').addEventListener('click', (e) => {
        const isFlattened = State.toggleRunsFlatten();
        e.target.textContent = isFlattened ? 'Unflatten' : 'Flatten';
        e.target.classList.toggle('btn-primary', isFlattened);
        e.target.classList.toggle('btn-secondary', !isFlattened);
        const currentProject = State.getCurrentProject();
        if(currentProject) {
            showRunDetails(currentProject.name, currentProject.runs);
        }
    });
    
    document.getElementById('runs-column-selector-menu').addEventListener('change', (e) => {
        if (e.target.type === 'checkbox') {
            const param = e.target.value;
            const visibleColumns = State.getVisibleColumns();
            if (e.target.checked) visibleColumns.add(param);
            else visibleColumns.delete(param);
            
            document.getElementById('runs-detail-table').querySelectorAll(`[data-param="${param}"]`).forEach(cell => {
                cell.style.display = e.target.checked ? 'table-cell' : 'none';
            });
        }
    });


    // Mobile sidebar toggle
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.content-overlay');
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('active');
        overlay.classList.toggle('active');
    });
    overlay.addEventListener('click', () => {
        sidebar.classList.remove('active');
        overlay.classList.remove('active');
    });
    
    // --- Event Delegation ---
    document.body.addEventListener('click', async (event) => {
        const target = event.target.closest('button, a');
        if (!target) return;

        const projectItem = target.closest('.project-list-item');

        if (target.classList.contains('view-config-btn')) viewTextFile(target.dataset.path, 'Launch Config', event);
        else if (target.classList.contains('view-log-btn')) viewTextFile(target.dataset.path, 'Terminal Log', event, target.dataset.status === 'running');
        else if (target.classList.contains('view-csv-btn')) viewTextFile(target.dataset.path, 'CSV Results', event);
        else if (target.classList.contains('plot-csv-btn')) plotCsv(target.dataset.path, event);
        else if (target.classList.contains('compare-group-btn')) compareGroupYAMLs(target, event);
        else if (target.classList.contains('stop-group-btn')) {
            event.stopPropagation();
            const groupId = target.dataset.groupId;
            const pid = target.dataset.pid;
            if (confirm(`Are you sure you want to stop this experiment group (PID: ${pid})?`)) {
                try {
                    target.disabled = true;
                    target.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
                    const result = await API.stopExperimentGroup(groupId);
                    alert(result.message);
                    loadHistory();
                } catch (error) {
                    alert(`Error: ${error.message}`);
                    target.disabled = false;
                    target.innerHTML = '<i class="bi bi-stop-circle"></i> Stop';
                }
            }
        }
        else if (projectItem) {
            event.preventDefault();
            const exp = JSON.parse(projectItem.dataset.experiment);
            State.setCurrentProject(exp);
            showRunDetails(exp.name, exp.runs);
        }
    });


    // --- Executor Tab Logic ---
    async function loadConfigContent(path) {
        if (!path) { editor.setValue(""); return; }
        try {
            const data = await API.fetchFile(path);
            editor.setValue(data.content || '');
        } catch (error) {
            console.error(`Failed to load config file ${path}:`, error);
        }
    }

    async function loadAvailableConfigs() {
        try {
            const data = await API.fetchConfigs();
            const configSelect = document.getElementById('config-select');
            configSelect.innerHTML = '';
            if (data.configs && data.configs.length > 0) {
                data.configs.forEach(c => configSelect.add(new Option(c, c)));
                loadConfigContent(data.configs[0]);
            } else {
                configSelect.add(new Option("No config files found", ""));
                editor.setValue("# No config files in 'config/' directory.\n# Create one to start an experiment.");
            }
        } catch (error) {
            console.error("Failed to load configs:", error);
        }
    }

    function launchExperiment() {
        const executorLogOutput = document.getElementById('executor-log-output');
        executorLogOutput.innerHTML = '<span class="text-info">🚀 Launching experiment group...</span>\n';
        const socket = new WebSocket(`ws://${window.location.host}/run`);
        socket.onopen = () => socket.send(JSON.stringify({ content: editor.getValue() }));
        socket.onmessage = (event) => {
            executorLogOutput.innerHTML += event.data + '\n';
            executorLogOutput.scrollTop = executorLogOutput.scrollHeight; // Scroll to bottom
        };
        socket.onerror = () => executorLogOutput.innerHTML += '<span class="text-danger">❌ WebSocket connection error.</span>\n';
        socket.onclose = () => {
            executorLogOutput.innerHTML += '\n<span class="text-muted">--- Connection Closed ---</span>\n';
            loadHistory();
        };
    }

    // --- History Tab Logic ---
    async function loadHistory() {
        // [BUG FIX] 갱신 전, 현재 확장된 그룹들의 ID를 저장합니다.
        const historyTableBody = document.getElementById('history-table-body');
        const expandedIds = new Set(
            Array.from(historyTableBody.querySelectorAll('.collapse.show')).map(el => el.id)
        );

        try {
            const groups = await API.fetchExperimentGroups();
            UI.renderHistoryTable(groups);

            // [BUG FIX] 렌더링 후, 저장했던 ID를 가진 그룹들을 다시 확장시킵니다.
            expandedIds.forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    new bootstrap.Collapse(element, { toggle: false }).show();
                }
            });

        } catch (error) {
            console.error("Failed to load history:", error);
            if(historyPoller) { clearInterval(historyPoller); historyPoller = null; }
            historyTableBody.innerHTML = `<tr><td colspan="6" class="text-center text-danger">Failed to load history. Check console.</td></tr>`;
        }
    }

    // --- Runs Tab Logic ---
    async function loadRuns() {
        const savedProject = State.getCurrentProject();
        if (savedProject) {
            showRunDetails(savedProject.name, savedProject.runs);
        } else {
            try {
                const experiments = await API.fetchVisualizerExperiments();
                UI.renderProjectsList(experiments);
                document.getElementById('runs-detail-view').style.display = 'none';
                document.getElementById('runs-list-view').style.display = 'block';
            } catch (error) {
                console.error("Failed to load runs data:", error);
            }
        }
    }

    async function showRunDetails(projectName, runs) {
        document.getElementById('runs-list-view').style.display = 'none';
        document.getElementById('runs-detail-view').style.display = 'block';
        document.getElementById('runs-detail-title').innerText = `Project: ${projectName}`;
        const table = document.getElementById('runs-detail-table');
        table.querySelector('thead').innerHTML = '<tr><th>Loading...</th></tr>';
        table.querySelector('tbody').innerHTML = '<tr><td><div class="spinner-border" role="status"></div></td></tr>';

        try {
            const runDataPromises = runs.map(async (run) => {
                const configPath = run.path.replace('.csv', '_config.yaml');
                const config = await API.fetchFile(configPath).catch(() => ({content: '{}'}));
                return {
                    name: run.path.split('/').pop().replace('.csv', ''),
                    path: run.path,
                    configPath: configPath,
                    configContent: config.content,
                };
            });
            const fullRunData = await Promise.all(runDataPromises);
            
            const allParams = new Set();
            fullRunData.forEach(run => {
                try {
                    const configData = jsyaml.load(run.configContent);
                    const originalConfig = configData || {};
                    
                    if (State.getIsRunsFlattened()) {
                        const flattenedConfig = Utils.flattenObject(originalConfig);
                        Object.keys(flattenedConfig).forEach(key => allParams.add(key));
                    } else {
                         Object.keys(originalConfig).forEach(key => allParams.add(key));
                    }
                } catch(e){ console.error("YAML Parse Error:", e); }
            });

            const sortedParams = Array.from(allParams).sort();
            
            UI.renderColumnSelector(sortedParams);
            UI.renderRunDetailsTable(table, fullRunData, State.getIsRunsFlattened());
            initializeTableInteractivity('runs-detail-table');
        } catch (error) {
            console.error(`Failed to show details for ${projectName}:`, error);
        }
    }
    
    function initializeTableInteractivity(tableId) {
        const table = document.getElementById(tableId);
        if (!table) return;

        let sortState = {}; // Stores current sort direction for each column
        
        // Helper to get original rows (before sorting)
        const getOriginalRows = () => Array.from(table.querySelector('tbody').querySelectorAll('tr'));
        let originalRows = getOriginalRows(); // Store initial order

        // Helper function to parse date strings (YYYY-MM-DD)
        const parseDate = (dateString) => {
            const parts = dateString.split('-');
            // Month is 0-indexed in JavaScript Date
            return new Date(parts[0], parts[1] - 1, parts[2]);
        };

        // --- Sorting Logic (attached to .sort-icon) ---
        table.querySelector('thead').addEventListener('click', (e) => {
            const sortIcon = e.target.closest('.sort-icon');
            if (!sortIcon) return; // Only react to clicks on the sort icon

            const header = sortIcon.closest('th');
            if (!header) return;

            const columnIndex = Array.from(header.parentNode.children).indexOf(header); // Get column index
            
            const currentDir = sortState[columnIndex];
            let newDir;
            if (currentDir === 'desc') newDir = 'asc';
            else if (currentDir === 'asc') newDir = 'none'; // Cycle: desc -> asc -> none (original)
            else newDir = 'desc';

            sortState = { [columnIndex]: newDir }; // Update sort state

            const tBody = table.querySelector('tbody');
            let rows = Array.from(tBody.querySelectorAll('tr')); // Get current rows

            // Reset all sort icons
            table.querySelectorAll('.sort-icon').forEach(icon => {
                icon.classList.remove('active', 'bi-sort-up', 'bi-sort-down');
                icon.classList.add('bi-arrow-down-up');
            });

            if (newDir === 'none') {
                // Restore original order
                tBody.innerHTML = ''; // Clear current rows
                originalRows.forEach(row => tBody.appendChild(row)); // Append original rows
                return;
            }
            
            // Sort rows
            rows.sort((a, b) => {
                const aText = a.cells[columnIndex]?.innerText || '';
                const bText = b.cells[columnIndex]?.innerText || '';
                
                let comparison = 0;

                // Attempt to parse as date
                const aDate = parseDate(aText);
                const bDate = parseDate(bText);

                if (!isNaN(aDate) && !isNaN(bDate)) { // Both are valid dates
                    comparison = aDate.getTime() - bDate.getTime();
                } else { // Fallback to numeric or string comparison
                    const aNum = parseFloat(aText);
                    const bNum = parseFloat(bText);

                    if (!isNaN(aNum) && !isNaN(bNum)) { // Both are valid numbers
                        comparison = aNum - bNum;
                    } else { // Compare as strings
                        comparison = aText.localeCompare(bText);
                    }
                }
                return newDir === 'asc' ? comparison : -comparison;
            });
            
            // Append sorted rows
            tBody.innerHTML = ''; // Clear current rows
            rows.forEach(row => tBody.appendChild(row)); // Append sorted rows

            // Update clicked sort icon
            sortIcon.classList.add('active');
            sortIcon.classList.remove('bi-arrow-down-up');
            sortIcon.classList.add(newDir === 'asc' ? 'bi-sort-up' : 'bi-sort-down');
        });

        // --- Resizing Logic (attached to .resizer) ---
        table.querySelectorAll('th').forEach(header => {
            const resizer = header.querySelector('.resizer');
            if (!resizer) return; // Skip if no resizer found

            let startX, startWidth;

            const doMouseMove = (e) => {
                const diff = e.pageX - startX;
                header.style.width = `${startWidth + diff}px`;
                header.style.minWidth = `${startWidth + diff}px`; // Ensure min-width also updates
            };

            const doMouseUp = () => {
                document.removeEventListener('mousemove', doMouseMove);
                document.removeEventListener('mouseup', doMouseUp);
                // Optional: Save column width to local storage for persistence
            };

            resizer.addEventListener('mousedown', (e) => {
                startX = e.pageX;
                startWidth = header.offsetWidth;
                document.addEventListener('mousemove', doMouseMove);
                document.addEventListener('mouseup', doMouseUp);
            });

            // Double-click to auto-fit
            resizer.addEventListener('dblclick', () => {
                header.style.width = 'auto';
                header.style.minWidth = 'auto';
            });
        });
    }


    // --- Metrics Tab Logic ---
    async function renderMetricsComparison() {
        const content = document.getElementById('metrics-content');
        const results = document.getElementById('metrics-results');
        const runList = State.getComparisonList();
        
        UI.renderSelectedRunsList();

        if (runList.length < 1) {
            content.style.display = 'block';
            results.style.display = 'none';
            return;
        }
        content.style.display = 'none';
        results.style.display = 'block';

        const runData = await Promise.all(
            runList.map(async (run) => {
                const [config, csv] = await Promise.all([
                    API.fetchFile(run.configPath).catch(() => ({content: 'Error: Not Found'})),
                    API.fetchFile(run.csvPath).catch(() => ({content: ''}))
                ]);
                return { ...run, configContent: config.content, csvContent: csv.content };
            })
        );

        const paramValues = Utils.createDiffObject(runData, State.getIsMetricsFlattened());
        UI.renderDiffTable('metrics-diff-table-container', runData, paramValues);
        
        const chartOptions = {
            showPoints: document.getElementById('chart-show-points-toggle').checked,
            smoothedLines: document.getElementById('chart-smooth-lines-toggle').checked,
            colors: runList.reduce((acc, run) => {
                acc[run.csvPath] = run.color;
                return acc;
            }, {})
        };
        metricsChartInstance = UI.renderMetricsChart(runData, metricsChartInstance, chartOptions);
    }
    
    // --- Modal & Global Functions ---
    async function viewTextFile(path, title, event, isRunning = false) {
        if (event) event.stopPropagation();
        
        const textViewerTitle = document.getElementById('text-viewer-title');
        const textViewerContent = document.getElementById('text-viewer-content');
        textViewerTitle.textContent = title;
        textViewerContent.textContent = 'Loading...';
        textViewerModal.show();
        
        if (logPoller) clearInterval(logPoller);

        const fetchAndUpdateLog = async () => {
            try {
                const data = await API.fetchFile(path);
                textViewerContent.textContent = data.content;
                textViewerContent.scrollTop = textViewerContent.scrollHeight; // Scroll to bottom
            } catch (error) {
                textViewerContent.textContent = `Error loading file:\n\n${error.message}`;
                if (logPoller) clearInterval(logPoller);
            }
        };

        await fetchAndUpdateLog(); // Initial load

        if (isRunning) {
            logPoller = setInterval(fetchAndUpdateLog, 1000); // Poll every second
        }
    }

    async function plotCsv(path, event) {
        if (event) event.stopPropagation();
        let currentCsvData = null;
        const drawChart = () => {
            if (!currentCsvData) return;
            const { headers, rows } = currentCsvData;
            const xIndex = parseInt(document.getElementById('x-axis-select').value);
            const yIndex = parseInt(document.getElementById('y-axis-select').value);
            const labels = rows.map(row => row[xIndex]);
            const chartData = rows.map(row => parseFloat(row[yIndex]));
            
            if (singlePlotChart) singlePlotChart.destroy();
            singlePlotChart = new Chart(document.getElementById('csv-chart').getContext('2d'), { 
                type: 'line', data: { labels, datasets: [{ label: headers[yIndex], data: chartData, borderColor: '#0D6EFD', tension: 0.1 }] }
            });
        };

        try {
            const data = await API.fetchFile(path);
            const lines = data.content.trim().split('\n');
            if (lines.length < 2) { alert("CSV data is empty or invalid."); return; }
            const headers = lines.slice(0,1)[0].split(',');
            const rows = lines.slice(1).map(line => line.split(','));
            currentCsvData = { headers, rows };

            const xAxisSelect = document.getElementById('x-axis-select');
            const yAxisSelect = document.getElementById('y-axis-select');
            xAxisSelect.innerHTML = '';
            yAxisSelect.innerHTML = '';
            headers.forEach((h, i) => {
                xAxisSelect.add(new Option(h, i));
                yAxisSelect.add(new Option(h, i));
            });
            xAxisSelect.value = 0;
            yAxisSelect.value = headers.length > 1 ? headers.length - 1 : 0;
            
            xAxisSelect.onchange = drawChart;
            yAxisSelect.onchange = drawChart;
            
            drawChart();
            document.getElementById('plot-viewer-title').textContent = `Plot for ${path.split('/').pop()}`;
            plotViewerModal.show();
        } catch (error) { console.error("Failed to plot CSV:", error); }
    }

    async function compareGroupYAMLs(button, event) {
        event.stopPropagation();
        const container = button.closest('.collapse');
        const selectedCheckboxes = container.querySelectorAll('.run-comparison-checkbox:checked');
        if (selectedCheckboxes.length < 2) { alert("Please select at least two runs to compare."); return; }
        
        const runs = await Promise.all(Array.from(selectedCheckboxes).map(async (cb) => {
            const config = await API.fetchFile(cb.dataset.configPath);
            return { name: cb.dataset.runName, configContent: config.content };
        }));

        const paramValues = Utils.createDiffObject(runs);
        UI.renderDiffTable('yaml-compare-table-container', runs, paramValues);
        document.getElementById('show-diff-only-toggle').checked = false;
        yamlCompareModal.show();
    }

    // --- Initial Load ---
    loadAvailableConfigs();
});
