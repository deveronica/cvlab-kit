/**
 * Development utilities for conditional logging
 */

export const isDevelopment = import.meta.env.DEV;

/**
 * Conditional console.log that only runs in development
 */
export const devLog = (...args: any[]) => {
  if (isDevelopment) {
    console.log(...args);
  }
};

/**
 * Conditional console.info that only runs in development
 */
export const devInfo = (...args: any[]) => {
  if (isDevelopment) {
    console.info(...args);
  }
};

/**
 * Conditional console.warn that only runs in development
 */
export const devWarn = (...args: any[]) => {
  if (isDevelopment) {
    console.warn(...args);
  }
};

/**
 * Conditional console.debug that only runs in development
 */
export const devDebug = (...args: any[]) => {
  if (isDevelopment) {
    console.debug(...args);
  }
};

/**
 * console.error wrapper (always runs, but can be extended for error tracking)
 */
export const devError = (...args: any[]) => {
  console.error(...args);
};
