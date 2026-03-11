/**
 * CodeMode - Code editor view for Builder tab
 * Simplified version of ComponentsMode focused on code viewing/editing
 */

import React from 'react';
import { Code2 } from 'lucide-react';
import { BreadcrumbItem } from './BreadCrumb';
import { ComponentsMode } from './ComponentsMode';

interface CodeModeProps {
  breadcrumb: BreadcrumbItem[];
  onBreadcrumbUpdate: (breadcrumb: BreadcrumbItem[]) => void;
}

export function CodeMode({ breadcrumb, onBreadcrumbUpdate }: CodeModeProps) {
  // For now, CodeMode reuses ComponentsMode functionality
  // In the future, this could be extended with:
  // - Monaco Editor for advanced editing
  // - Auto-save on Ctrl+S
  // - Syntax highlighting improvements
  // - LSP integration

  return (
    <ComponentsMode
      breadcrumb={breadcrumb}
      onBreadcrumbUpdate={onBreadcrumbUpdate}
    />
  );
}
