/**
 * ConnectionContext - Track connection drag state for port highlighting
 *
 * Provides:
 * - Current dragging port info (type, position)
 * - Compatibility check for target ports
 * - Visual feedback state for handles
 */

import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import { PortType } from '@/shared/model/node-graph';
import { isValidConnection as checkValidConnection } from '@/shared/config/port-themes';

interface DraggingPort {
  nodeId: string;
  portId: string;
  portType: PortType;
  handleType: 'source' | 'target';
}

interface ConnectionContextValue {
  // Current drag state
  draggingPort: DraggingPort | null;
  isDragging: boolean;

  // Actions
  startDrag: (port: DraggingPort) => void;
  endDrag: () => void;

  // Check if a port is compatible with the currently dragging port
  isCompatibleWithDragging: (portType: PortType, handleType: 'source' | 'target') => boolean;

  // Get highlight state for a port
  getPortHighlight: (portType: PortType, handleType: 'source' | 'target') => 'compatible' | 'incompatible' | 'none';
}

const ConnectionContext = createContext<ConnectionContextValue | null>(null);

export function ConnectionProvider({ children }: { children: React.ReactNode }) {
  const [draggingPort, setDraggingPort] = useState<DraggingPort | null>(null);

  const startDrag = useCallback((port: DraggingPort) => {
    setDraggingPort(port);
  }, []);

  const endDrag = useCallback(() => {
    setDraggingPort(null);
  }, []);

  const isDragging = draggingPort !== null;

  // Check if a port is compatible with the dragging port
  const isCompatibleWithDragging = useCallback((portType: PortType, handleType: 'source' | 'target'): boolean => {
    if (!draggingPort) return false;

    // Can't connect same handle types (source to source, target to target)
    if (draggingPort.handleType === handleType) return false;

    // Check type compatibility
    if (draggingPort.handleType === 'source') {
      // Dragging from source (output), checking target (input)
      return checkValidConnection(draggingPort.portType, portType);
    } else {
      // Dragging from target (input), checking source (output)
      return checkValidConnection(portType, draggingPort.portType);
    }
  }, [draggingPort]);

  // Get highlight state for visual feedback
  const getPortHighlight = useCallback((portType: PortType, handleType: 'source' | 'target'): 'compatible' | 'incompatible' | 'none' => {
    if (!draggingPort) return 'none';

    // Same handle type = can't connect
    if (draggingPort.handleType === handleType) return 'none';

    // Check compatibility
    const compatible = isCompatibleWithDragging(portType, handleType);
    return compatible ? 'compatible' : 'incompatible';
  }, [draggingPort, isCompatibleWithDragging]);

  const value = useMemo(() => ({
    draggingPort,
    isDragging,
    startDrag,
    endDrag,
    isCompatibleWithDragging,
    getPortHighlight,
  }), [draggingPort, isDragging, startDrag, endDrag, isCompatibleWithDragging, getPortHighlight]);

  return (
    <ConnectionContext.Provider value={value}>
      {children}
    </ConnectionContext.Provider>
  );
}

export function useConnection() {
  const context = useContext(ConnectionContext);
  if (!context) {
    throw new Error('useConnection must be used within ConnectionProvider');
  }
  return context;
}

// Safe hook that returns null if outside provider (for components that may be used outside)
export function useConnectionSafe() {
  return useContext(ConnectionContext);
}
