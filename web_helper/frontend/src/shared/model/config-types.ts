/**
 * Type-safe config value system
 *
 * Replaces ~43 instances of Record<string, any>
 * Used for configurations, hyperparameters, and component parameters
 */

export type ConfigPrimitive = string | number | boolean | null;
export type ConfigArray = ConfigPrimitive[];

export interface ConfigObject {
  [key: string]: ConfigValue;
}

export type ConfigValue = ConfigPrimitive | ConfigArray | ConfigObject;

export interface Config {
  [key: string]: ConfigValue;
}

export interface Hyperparameters extends Config {}
export interface ComponentParams extends Config {}

/** Type guards */
export function isConfigPrimitive(value: unknown): value is ConfigPrimitive {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value);
}

export function isConfigArray(value: unknown): value is ConfigArray {
  return Array.isArray(value) && value.every(item => isConfigPrimitive(item));
}

export function isConfigObject(value: unknown): value is ConfigObject {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }
  return Object.values(value as Record<string, unknown>).every(v =>
    isConfigValue(v)
  );
}

export function isConfigValue(value: unknown): value is ConfigValue {
  return isConfigPrimitive(value) || isConfigArray(value) || isConfigObject(value);
}

/**
 * Deep clone configuration object
 */
export function cloneConfig(config: Config): Config {
  return JSON.parse(JSON.stringify(config));
}

/**
 * Merge two config objects (deep)
 */
export function mergeConfigs(base: Config, override: Config): Config {
  const result = { ...base };

  for (const [key, value] of Object.entries(override)) {
    if (isConfigObject(value) && isConfigObject(result[key])) {
      result[key] = mergeConfigs(
        result[key] as Config,
        value as Config
      );
    } else {
      result[key] = value;
    }
  }

  return result;
}
