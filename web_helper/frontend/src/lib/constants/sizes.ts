/**
 * Design system size constants for modals, views, and components
 * Following shadcn/ui and Tailwind CSS standards
 */

export const MODAL_SIZES = {
  sm: 'max-w-md',    // 448px - Small dialogs, confirmations
  md: 'max-w-2xl',   // 672px - Medium dialogs, settings
  lg: 'max-w-4xl',   // 896px - Large dialogs, forms
  xl: 'max-w-6xl',   // 1152px - Extra large, detail views
  full: 'max-w-7xl', // 1280px - Full-width, compare views
} as const;

export const VIEW_HEIGHTS = {
  compact: 'h-[400px]',  // Compact views, small charts
  standard: 'h-[600px]', // Standard views, most content
  tall: 'h-[800px]',     // Tall views, complex visualizations
} as const;

export const BUTTON_SIZES = {
  sm: 'h-8 px-3 text-sm',      // Small buttons, inline controls
  default: 'h-10 px-4 text-sm', // Default buttons, most contexts
  lg: 'h-11 px-8 text-base',   // Large buttons, emphasis
  icon: 'h-10 w-10',            // Icon-only buttons
} as const;

export const HEADER_HEIGHTS = {
  compact: 'h-12',  // Compact headers
  standard: 'h-14', // Standard headers (default)
  tall: 'h-16',     // Tall headers with extra content
} as const;

export const CARD_PADDING = {
  none: 'p-0',
  sm: 'p-4',      // Compact cards
  default: 'p-6',  // Standard cards (default)
  lg: 'p-8',      // Large cards, emphasis
} as const;

export type ModalSize = keyof typeof MODAL_SIZES;
export type ViewHeight = keyof typeof VIEW_HEIGHTS;
export type ButtonSize = keyof typeof BUTTON_SIZES;
export type HeaderHeight = keyof typeof HEADER_HEIGHTS;
export type CardPadding = keyof typeof CARD_PADDING;
