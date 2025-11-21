/**
 * YAML processing utilities for Projects View
 */

export type NestedObject = {
  [key: string]: any;
};

export type FlatObject = {
  [key: string]: any;
};

/**
 * Flatten nested object using dot notation
 * Example: { config: { model: { lr: 0.001 } } } → { "config.model.lr": 0.001 }
 */
export const flattenObject = (obj: NestedObject, prefix = '', result: FlatObject = {}): FlatObject => {
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const newKey = prefix ? `${prefix}.${key}` : key;
      const value = obj[key];

      if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
        // Recursively flatten nested objects
        flattenObject(value, newKey, result);
      } else {
        // Store primitive values and arrays as-is
        result[newKey] = value;
      }
    }
  }
  return result;
};

/**
 * Unflatten dot-notation keys back to nested object
 * Example: { "config.model.lr": 0.001 } → { config: { model: { lr: 0.001 } } }
 */
export const unflattenObject = (flatObj: FlatObject): NestedObject => {
  const result: NestedObject = {};

  for (const key in flatObj) {
    if (flatObj.hasOwnProperty(key)) {
      const keys = key.split('.');
      let current = result;

      for (let i = 0; i < keys.length - 1; i++) {
        const k = keys[i];
        if (!(k in current) || typeof current[k] !== 'object' || Array.isArray(current[k])) {
          current[k] = {};
        }
        current = current[k];
      }

      current[keys[keys.length - 1]] = flatObj[key];
    }
  }

  return result;
};

/**
 * Get all unique keys from a list of objects (either flat or nested)
 */
export const getAllKeys = (objects: (NestedObject | FlatObject)[], flatten: boolean = false): string[] => {
  const keySet = new Set<string>();

  objects.forEach(obj => {
    const processedObj = flatten ? flattenObject(obj) : obj;
    Object.keys(processedObj).forEach(key => keySet.add(key));
  });

  return Array.from(keySet).sort();
};

/**
 * Convert run configurations to either flat or nested format
 */
export const processRunConfigs = (runs: any[], flatten: boolean): any[] => {
  return runs.map(run => {
    if (!run.config) return run;

    return {
      ...run,
      config: flatten ? flattenObject(run.config) : run.config
    };
  });
};

/**
 * Detect which columns have different values across runs (for Diff toggle)
 */
export const getDifferentColumns = (runs: any[], configKey = 'config'): string[] => {
  if (runs.length === 0) return [];

  // First, get all possible keys
  const allKeys = new Set<string>();
  runs.forEach(run => {
    if (run[configKey]) {
      Object.keys(run[configKey]).forEach(key => allKeys.add(key));
    }
  });

  // Filter keys that have different values
  const differentKeys: string[] = [];

  allKeys.forEach(key => {
    const values = runs.map(run => run[configKey]?.[key]);
    const uniqueValues = new Set(values.map(v => JSON.stringify(v)));

    if (uniqueValues.size > 1) {
      differentKeys.push(key);
    }
  });

  return differentKeys.sort();
};

/**
 * Apply column mapping to transform keys
 */
export const applyColumnMapping = (
  obj: FlatObject | NestedObject,
  mapping: Record<string, string>
): FlatObject | NestedObject => {
  const result: any = {};

  for (const [key, value] of Object.entries(obj)) {
    const newKey = mapping[key] || key;
    result[newKey] = value;
  }

  return result;
};

/**
 * Format value for display in table cells
 */
export const formatConfigValue = (value: any): string => {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return value.toString();
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return `[${value.length} items]`;
  if (typeof value === 'object') return '[Object]';
  return String(value);
};

/**
 * Validate column mapping
 */
export const validateColumnMapping = (
  mapping: Record<string, string>,
  availableKeys: string[]
): { valid: boolean; errors: string[] } => {
  const errors: string[] = [];

  Object.entries(mapping).forEach(([from, to]) => {
    if (!availableKeys.includes(from)) {
      errors.push(`Source column "${from}" does not exist`);
    }
    if (!to.trim()) {
      errors.push(`Target column for "${from}" cannot be empty`);
    }
  });

  return {
    valid: errors.length === 0,
    errors
  };
};