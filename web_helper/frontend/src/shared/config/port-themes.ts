/**
 * Port Theme System - ComfyUI-style type-based port colors
 *
 * Each PortType has a distinct color for easy visual identification.
 * Colors are designed to work well in both light and dark themes.
 */

import { PortType, InputKind, ValueSource } from '@/shared/model/node-graph';

// ============================================================================
// Port Type Colors (ComfyUI-inspired)
// ============================================================================

export interface PortTypeTheme {
  // Main color (for port dot/handle)
  color: string;
  // Background when hovered/selected
  bgHover: string;
  // Light background for property badges
  bgLight: string;
  // Border color
  border: string;
  // Text color on dark background
  textOnColor: string;
  // Label for tooltip
  label: string;
}

export const portTypeThemes: Record<PortType, PortTypeTheme> = {
  [PortType.TENSOR]: {
    color: 'rgb(59, 130, 246)',      // Blue-500
    bgHover: 'rgb(59, 130, 246, 0.2)',
    bgLight: 'rgb(239, 246, 255)',   // Blue-50
    border: 'rgb(147, 197, 253)',    // Blue-300
    textOnColor: 'white',
    label: 'Tensor',
  },
  [PortType.MODULE]: {
    color: 'rgb(168, 85, 247)',      // Purple-500
    bgHover: 'rgb(168, 85, 247, 0.2)',
    bgLight: 'rgb(250, 245, 255)',   // Purple-50
    border: 'rgb(216, 180, 254)',    // Purple-300
    textOnColor: 'white',
    label: 'Module',
  },
  execution: { color: '#2563eb', bgHover: 'bg-blue-100', bgLight: 'bg-blue-50', border: 'border-blue-200', textOnColor: 'text-white', label: 'Exec' },
  [PortType.OPTIMIZER]: {
    color: 'rgb(34, 197, 94)',       // Green-500
    bgHover: 'rgb(34, 197, 94, 0.2)',
    bgLight: 'rgb(240, 253, 244)',   // Green-50
    border: 'rgb(134, 239, 172)',    // Green-300
    textOnColor: 'white',
    label: 'Optimizer',
  },
  [PortType.SCALAR]: {
    color: 'rgb(249, 115, 22)',      // Orange-500
    bgHover: 'rgb(249, 115, 22, 0.2)',
    bgLight: 'rgb(255, 247, 237)',   // Orange-50
    border: 'rgb(253, 186, 116)',    // Orange-300
    textOnColor: 'white',
    label: 'Scalar',
  },
  [PortType.BOOL]: {
    color: 'rgb(16, 185, 129)',      // Emerald-500
    bgHover: 'rgb(16, 185, 129, 0.2)',
    bgLight: 'rgb(236, 253, 245)',   // Emerald-50
    border: 'rgb(110, 231, 183)',    // Emerald-300
    textOnColor: 'white',
    label: 'Bool',
  },
  [PortType.STRING]: {
    color: 'rgb(99, 102, 241)',      // Indigo-500
    bgHover: 'rgb(99, 102, 241, 0.2)',
    bgLight: 'rgb(238, 242, 255)',   // Indigo-50
    border: 'rgb(165, 180, 252)',    // Indigo-300
    textOnColor: 'white',
    label: 'String',
  },
  [PortType.DEVICE]: {
    color: 'rgb(148, 163, 184)',     // Slate-400
    bgHover: 'rgb(148, 163, 184, 0.2)',
    bgLight: 'rgb(248, 250, 252)',   // Slate-50
    border: 'rgb(203, 213, 225)',    // Slate-300
    textOnColor: 'white',
    label: 'Device',
  },
  [PortType.DICT]: {
    color: 'rgb(236, 72, 153)',      // Pink-500
    bgHover: 'rgb(236, 72, 153, 0.2)',
    bgLight: 'rgb(253, 242, 248)',   // Pink-50
    border: 'rgb(249, 168, 212)',    // Pink-300
    textOnColor: 'white',
    label: 'Dict',
  },
  [PortType.LIST]: {
    color: 'rgb(14, 165, 233)',      // Sky-500
    bgHover: 'rgb(14, 165, 233, 0.2)',
    bgLight: 'rgb(240, 249, 255)',   // Sky-50
    border: 'rgb(125, 211, 252)',    // Sky-300
    textOnColor: 'white',
    label: 'List',
  },
  [PortType.CONFIG]: {
    color: 'rgb(234, 179, 8)',       // Yellow-500
    bgHover: 'rgb(234, 179, 8, 0.2)',
    bgLight: 'rgb(254, 252, 232)',   // Yellow-50
    border: 'rgb(253, 224, 71)',     // Yellow-300
    textOnColor: 'black',
    label: 'Config',
  },
  [PortType.PARAMETERS]: {
    color: 'rgb(6, 182, 212)',       // Cyan-500
    bgHover: 'rgb(6, 182, 212, 0.2)',
    bgLight: 'rgb(236, 254, 255)',   // Cyan-50
    border: 'rgb(103, 232, 249)',    // Cyan-300
    textOnColor: 'white',
    label: 'Parameters',
  },
  [PortType.ANY]: {
    color: 'rgb(107, 114, 128)',     // Gray-500
    bgHover: 'rgb(107, 114, 128, 0.2)',
    bgLight: 'rgb(249, 250, 251)',   // Gray-50
    border: 'rgb(209, 213, 219)',    // Gray-300
    textOnColor: 'white',
    label: 'Any',
  },

  // Control Flow ports - execution order, not data
  [PortType.CONTROL_IN]: {
    color: 'rgb(75, 85, 99)',        // Gray-600 (darker for control)
    bgHover: 'rgb(75, 85, 99, 0.2)',
    bgLight: 'rgb(243, 244, 246)',   // Gray-100
    border: 'rgb(156, 163, 175)',    // Gray-400
    textOnColor: 'white',
    label: 'Control In',
  },
  [PortType.CONTROL_OUT]: {
    color: 'rgb(75, 85, 99)',        // Gray-600 (darker for control)
    bgHover: 'rgb(75, 85, 99, 0.2)',
    bgLight: 'rgb(243, 244, 246)',   // Gray-100
    border: 'rgb(156, 163, 175)',    // Gray-400
    textOnColor: 'white',
    label: 'Control Out',
  },
};

// ============================================================================
// Value Source Icons & Styles
// ============================================================================

export interface ValueSourceTheme {
  icon: string;      // Icon character or name
  color: string;     // Icon color
  bgColor: string;   // Background color
  tooltip: string;   // Tooltip text
  editable?: boolean; // Whether the value can be edited
}

/**
 * 5-state Value Source Theme System
 *
 * Visual distinction for property sources:
 * - REQUIRED: Red warning, must provide value
 * - CONFIG: Blue, editable from YAML
 * - DEFAULT: Gray, has default value
 * - HARDCODE: Disabled gray, fixed in code
 * - CONNECTED: Green, from connected node
 */
export const valueSourceThemes: Record<ValueSource, ValueSourceTheme> = {
  [ValueSource.REQUIRED]: {
    icon: '!',        // Warning
    color: 'rgb(239, 68, 68)',       // Red-500
    bgColor: 'rgb(254, 242, 242)',   // Red-50
    tooltip: 'Required - no default, must provide value',
    editable: true,
  },
  [ValueSource.CONFIG]: {
    icon: 'C',        // Config
    color: 'rgb(59, 130, 246)',      // Blue-500 (changed from green)
    bgColor: 'rgb(239, 246, 255)',   // Blue-50
    tooltip: 'From YAML config file',
    editable: true,
  },
  [ValueSource.DEFAULT]: {
    icon: 'D',        // Default
    color: 'rgb(107, 114, 128)',     // Gray-500
    bgColor: 'rgb(249, 250, 251)',   // Gray-50
    tooltip: 'Using default value (optional)',
    editable: true,
  },
  [ValueSource.HARDCODE]: {
    icon: '=',        // Fixed
    color: 'rgb(156, 163, 175)',     // Gray-400 (disabled look)
    bgColor: 'rgb(243, 244, 246)',   // Gray-100
    tooltip: 'Fixed in code, not editable',
    editable: false,
  },
  [ValueSource.CONNECTED]: {
    icon: '→',        // Connected
    color: 'rgb(34, 197, 94)',       // Green-500
    bgColor: 'rgb(240, 253, 244)',   // Green-50
    tooltip: 'Value from connected node',
    editable: false,
  },
};

// ============================================================================
// Input Kind Positioning (3-way layout)
// ============================================================================

export interface InputKindLayout {
  position: 'left' | 'right' | 'top' | 'bottom';
  handlePosition: 'Left' | 'Right' | 'Top' | 'Bottom';
}

export const inputKindLayouts: Record<InputKind, InputKindLayout> = {
  [InputKind.COMPONENT]: {
    position: 'left',
    handlePosition: 'Left',
  },
  [InputKind.PROPERTY]: {
    position: 'top',
    handlePosition: 'Top',
  },
  [InputKind.DEPENDENCY]: {
    position: 'bottom',
    handlePosition: 'Bottom',
  },
};

// ============================================================================
// Control Port Layout (separate from InputKind)
// ============================================================================

export interface ControlPortLayout {
  position: 'top' | 'bottom';
  handlePosition: 'Top' | 'Bottom';
  style: {
    strokeDasharray: string;
    strokeWidth: number;
  };
}

export const controlPortLayouts: Record<'in' | 'out', ControlPortLayout> = {
  in: {
    position: 'top',
    handlePosition: 'Top',
    style: {
      strokeDasharray: '4 2',  // Dashed line
      strokeWidth: 1.5,
    },
  },
  out: {
    position: 'bottom',
    handlePosition: 'Bottom',
    style: {
      strokeDasharray: '4 2',  // Dashed line
      strokeWidth: 1.5,
    },
  },
};

// ============================================================================
// Connection Validation (ComfyUI-style)
// ============================================================================

/**
 * Type compatibility matrix for port connections.
 * Key = source type, Value = array of compatible target types.
 */
const typeCompatibility: Record<PortType, PortType[]> = {
  // Data types
  [PortType.TENSOR]: [PortType.TENSOR, PortType.ANY],
  [PortType.MODULE]: [PortType.MODULE, PortType.OPTIMIZER, PortType.ANY],
  [PortType.OPTIMIZER]: [PortType.OPTIMIZER, PortType.MODULE, PortType.ANY],
  [PortType.SCALAR]: [PortType.SCALAR, PortType.CONFIG, PortType.ANY],
  [PortType.BOOL]: [PortType.BOOL, PortType.SCALAR, PortType.ANY],
  [PortType.STRING]: [PortType.STRING, PortType.ANY],
  [PortType.DICT]: [PortType.DICT, PortType.CONFIG, PortType.ANY],
  [PortType.LIST]: [PortType.LIST, PortType.ANY],
  [PortType.CONFIG]: [PortType.CONFIG, PortType.SCALAR, PortType.DICT, PortType.ANY],
  [PortType.DEVICE]: [PortType.DEVICE, PortType.STRING, PortType.ANY],
  [PortType.PARAMETERS]: [PortType.PARAMETERS, PortType.ANY],
  [PortType.ANY]: Object.values(PortType).filter(t => !t.startsWith('control')), // ANY connects to data types only

  // Control flow types - only connect to each other
  [PortType.CONTROL_IN]: [PortType.CONTROL_OUT],   // CONTROL_IN receives from CONTROL_OUT
  [PortType.CONTROL_OUT]: [PortType.CONTROL_IN],   // CONTROL_OUT sends to CONTROL_IN
};

/**
 * Check if a connection between two port types is valid.
 *
 * @param sourceType - Type of the source port (output)
 * @param targetType - Type of the target port (input)
 * @param targetCompatibleTypes - Optional explicit list of compatible types for target
 * @returns true if connection is valid
 */
export function isValidConnection(
  sourceType: PortType,
  targetType: PortType,
  targetCompatibleTypes?: PortType[]
): boolean {
  // If target has explicit compatible types, use those
  if (targetCompatibleTypes && targetCompatibleTypes.length > 0) {
    return targetCompatibleTypes.includes(sourceType);
  }

  // Otherwise use default compatibility matrix
  const compatible = typeCompatibility[sourceType] || [];
  return compatible.includes(targetType) || targetType === PortType.ANY;
}

/**
 * Get visual feedback color for a potential connection.
 *
 * @param sourceType - Type of the source port
 * @param targetType - Type of the target port
 * @param targetCompatibleTypes - Optional explicit compatible types
 * @returns Color string for visual feedback
 */
export function getConnectionFeedbackColor(
  sourceType: PortType,
  targetType: PortType,
  targetCompatibleTypes?: PortType[]
): string {
  const valid = isValidConnection(sourceType, targetType, targetCompatibleTypes);
  if (valid) {
    // Use the source port color for valid connections
    return portTypeThemes[sourceType].color;
  }
  // Red for invalid connections
  return 'rgb(239, 68, 68)'; // Red-500
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get the theme for a port type.
 */
export function getPortTheme(type: PortType): PortTypeTheme {
  return portTypeThemes[type] || portTypeThemes[PortType.ANY];
}

/**
 * Get the theme for a value source.
 */
export function getValueSourceTheme(source: ValueSource): ValueSourceTheme {
  return valueSourceThemes[source] || valueSourceThemes[ValueSource.DEFAULT];
}

/**
 * Get CSS class string for a port handle based on type.
 */
export function getPortHandleClassName(type: PortType): string {
  const theme = getPortTheme(type);
  return `w-2.5 h-2.5 border-2 transition-all duration-200`;
}

/**
 * Get inline style for a port handle.
 */
export function getPortHandleStyle(type: PortType): React.CSSProperties {
  const theme = getPortTheme(type);
  return {
    backgroundColor: theme.color,
    borderColor: theme.color,
  };
}

/**
 * Format a value for display (handles various types).
 */
export function formatPortValue(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return value.toString();
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return `[${value.length} items]`;
  if (typeof value === 'object') return `{${Object.keys(value).length} keys}`;
  return String(value);
}

// ============================================================================
// Control Port Helpers
// ============================================================================

/**
 * Check if a port type is a control flow port.
 */
export function isControlPortType(type: PortType): boolean {
  return type === PortType.CONTROL_IN || type === PortType.CONTROL_OUT;
}

/**
 * Get the control port layout (position and style).
 */
export function getControlPortLayout(type: PortType): ControlPortLayout | null {
  if (type === PortType.CONTROL_IN) return controlPortLayouts.in;
  if (type === PortType.CONTROL_OUT) return controlPortLayouts.out;
  return null;
}

/**
 * Get CSS styles for control port handle (diamond shape, dashed border).
 */
export function getControlPortHandleStyle(type: PortType): React.CSSProperties {
  const theme = getPortTheme(type);
  return {
    backgroundColor: 'transparent',
    borderColor: theme.color,
    borderWidth: 2,
    borderStyle: 'dashed',
    transform: 'rotate(45deg)',  // Diamond shape
    width: 8,
    height: 8,
  };
}

/**
 * Get CSS styles for control flow edge (dashed, gray).
 */
export function getControlEdgeStyle(): React.CSSProperties {
  return {
    stroke: 'rgb(75, 85, 99)',  // Gray-600
    strokeDasharray: '4 2',
    strokeWidth: 1.5,
  };
}
