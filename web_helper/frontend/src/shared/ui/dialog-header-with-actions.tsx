import React from "react";

/**
 * Dialog Header with Actions
 *
 * Reusable dialog header component that properly handles action buttons
 * alongside the dialog's close (X) button to prevent overlap.
 *
 * Architecture:
 * - The dialog's X button is positioned at `absolute right-4 top-4`
 * - This component reserves sufficient space (pr-12) to prevent overlap
 * - Ensures consistent header layout across all dialogs
 *
 * Usage:
 * ```tsx
 * <DialogHeaderWithActions
 *   title="Run Details"
 *   description="Detailed analysis for this run"
 *   actions={
 *     <>
 *       <Button>Compare</Button>
 *       <Button>Export</Button>
 *     </>
 *   }
 * />
 * ```
 */

import { DialogTitle, DialogDescription } from './dialog';

interface DialogHeaderWithActionsProps {
  /** Dialog title */
  title: React.ReactNode;
  /** Optional description below title */
  description?: React.ReactNode;
  /** Action buttons (rendered on the right) */
  actions?: React.ReactNode;
  /** Additional className for title */
  titleClassName?: string;
  /** Additional className for description */
  descriptionClassName?: string;
  /** Additional className for container */
  className?: string;
}

export function DialogHeaderWithActions({
  title,
  description,
  actions,
  titleClassName = 'text-xl font-bold',
  descriptionClassName = 'mt-1',
  className = '',
}: DialogHeaderWithActionsProps) {
  return (
    <div className={`flex items-start justify-between gap-4 ${className}`}>
      {/* Title Section */}
      <div className="flex-1 min-w-0 pr-2">
        <DialogTitle className={titleClassName}>
          {title}
        </DialogTitle>
        {description && (
          <DialogDescription className={descriptionClassName}>
            {description}
          </DialogDescription>
        )}
      </div>

      {/* Actions Section - Reserve space for X button */}
      {actions && (
        <div className="flex items-start gap-2 pr-14 flex-shrink-0">
          {actions}
        </div>
      )}
    </div>
  );
}
