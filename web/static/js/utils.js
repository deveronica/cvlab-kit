// --- YAML & Object Utilities ---
export function flattenObject(obj, prefix = '') {
    if (obj === null || typeof obj !== 'object') {
        // If prefix is empty, it means the root is not an object, which we don't flatten.
        return prefix ? { [prefix]: obj } : {};
    }
    return Object.keys(obj).reduce((acc, k) => {
        const pre = prefix.length ? prefix + '.' : '';
        const key = pre + k;
        if (typeof obj[k] === 'object' && obj[k] !== null && !Array.isArray(obj[k])) {
            Object.assign(acc, flattenObject(obj[k], key));
        } else {
            acc[key] = obj[k];
        }
        return acc;
    }, {});
}

export function formatValue(value) {
    if (value === undefined || value === null) return '-';
    if (typeof value === 'object') {
        // Pretty print object with indentation
        return JSON.stringify(value, null, 2);
    }
    return String(value);
}

export function createDiffObject(runs, flatten = false) {
    const allKeys = new Set();
    const parsedConfigs = runs.map(run => {
        try {
            const config = jsyaml.load(run.configContent);
            return flatten ? flattenObject(config || {}) : (config || {});
        } catch (e) {
            console.error(`YAML parsing error for ${run.name}:`, e);
            return {};
        }
    });

    parsedConfigs.forEach(config => {
        if (config) Object.keys(config).forEach(key => allKeys.add(key));
    });

    const paramValues = new Map();
    Array.from(allKeys).sort().forEach(key => {
        const values = parsedConfigs.map(config => formatValue(config[key]));
        paramValues.set(key, values);
    });
    return paramValues;
}

export function formatDuration(ms) {
    if (ms < 0) ms = 0;
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return [
        hours > 0 ? `${hours}h` : '',
        minutes > 0 ? `${minutes}m` : '',
        `${seconds}s`
    ].filter(Boolean).join(' ');
}

export function formatTimestamp(isoString) {
    if (!isoString) return 'N/A';
    // Append 'Z' to the timestamp string if it doesn't have timezone information.
    // This forces JavaScript to interpret the date as UTC.
    const utcIsoString = (typeof isoString === 'string' && !isoString.endsWith('Z')) 
        ? isoString + 'Z' 
        : isoString;
    const d = new Date(utcIsoString);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

const canvas = document.createElement('canvas');
const context = canvas.getContext('2d');

export function getTextWidth(text, font) {
    context.font = font;
    const metrics = context.measureText(text);
    return metrics.width;
}