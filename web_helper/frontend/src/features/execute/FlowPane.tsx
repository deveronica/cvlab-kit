import type { ComponentProps } from 'react';
import { ExecuteFlowPane } from '@/entities/node-system/ui';

export type FlowPaneProps = ComponentProps<typeof ExecuteFlowPane>;

/**
 * @deprecated Use ExecuteFlowPane from '@/entities/node-system/model/types' instead.
 */
export function FlowPane(props: FlowPaneProps) {
  return <ExecuteFlowPane {...props} />;
}
