/**
 * BuilderView - Agent Builder view
 *
 * New unified layout with Code + Node split view.
 * Replaces the old 3-tab (Components/Code/Nodes) approach.
 */

import React from 'react';
import { AgentBuilder } from '@/features/builder/AgentBuilder';

export function BuilderView() {
  return <AgentBuilder className="h-full" />;
}
