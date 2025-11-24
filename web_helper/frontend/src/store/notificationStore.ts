import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type NotificationType = 'info' | 'warning' | 'error' | 'success' | 'action';

export type NotificationAction = {
  label: string;
  value: string;
  variant?: 'default' | 'destructive' | 'outline';
};

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: number;
  read: boolean;
  // For action-required notifications
  requiresAction?: boolean;
  actions?: NotificationAction[];
  metadata?: Record<string, unknown>;
  // For sync conflicts
  conflictPath?: string;
  conflictType?: string;
}

interface NotificationState {
  notifications: Notification[];
  isOpen: boolean;

  // Actions
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => string;
  removeNotification: (id: string) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  clearAll: () => void;
  clearResolved: () => void;
  setOpen: (open: boolean) => void;
  toggle: () => void;

  // Selectors
  getUnreadCount: () => number;
  getPendingActions: () => Notification[];

  // Action response
  resolveAction: (id: string, action: string) => void;
}

export const useNotificationStore = create<NotificationState>()(
  persist(
    (set, get) => ({
      notifications: [],
      isOpen: false,

      addNotification: (notification) => {
        const id = `notif-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
        const newNotification: Notification = {
          ...notification,
          id,
          timestamp: Date.now(),
          read: false,
        };

        set((state) => ({
          notifications: [newNotification, ...state.notifications].slice(0, 100), // Keep max 100
        }));

        return id;
      },

      removeNotification: (id) => {
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        }));
      },

      markAsRead: (id) => {
        set((state) => ({
          notifications: state.notifications.map((n) =>
            n.id === id ? { ...n, read: true } : n
          ),
        }));
      },

      markAllAsRead: () => {
        set((state) => ({
          notifications: state.notifications.map((n) => ({ ...n, read: true })),
        }));
      },

      clearAll: () => {
        set({ notifications: [] });
      },

      clearResolved: () => {
        set((state) => ({
          notifications: state.notifications.filter(
            (n) => n.requiresAction && !n.read
          ),
        }));
      },

      setOpen: (open) => {
        set({ isOpen: open });
      },

      toggle: () => {
        set((state) => ({ isOpen: !state.isOpen }));
      },

      getUnreadCount: () => {
        return get().notifications.filter((n) => !n.read).length;
      },

      getPendingActions: () => {
        return get().notifications.filter((n) => n.requiresAction && !n.read);
      },

      resolveAction: (id, action) => {
        const notification = get().notifications.find((n) => n.id === id);
        if (!notification) return;

        // Dispatch custom event for listeners
        window.dispatchEvent(
          new CustomEvent('notification-action', {
            detail: { notificationId: id, action, metadata: notification.metadata },
          })
        );

        // Mark as read after action
        set((state) => ({
          notifications: state.notifications.map((n) =>
            n.id === id ? { ...n, read: true } : n
          ),
        }));
      },
    }),
    {
      name: 'cvlabkit-notifications',
      partialize: (state) => ({
        // Only persist unresolved action notifications
        notifications: state.notifications.filter((n) => n.requiresAction && !n.read),
      }),
    }
  )
);

// Helper to create sync conflict notification
export function createSyncConflictNotification(conflict: {
  path: string;
  conflictType: string;
  localHash?: string;
  serverHash?: string;
}): Omit<Notification, 'id' | 'timestamp' | 'read'> {
  const typeLabels: Record<string, string> = {
    local_newer: 'Local changes detected',
    server_newer: 'Server has newer version',
    local_only: 'New local component',
    server_only: 'Component missing locally',
  };

  const title = typeLabels[conflict.conflictType] || 'Sync conflict';

  let message = `Component: ${conflict.path}`;
  if (conflict.conflictType === 'local_newer') {
    message += '\nYour local changes are not on the server.';
  } else if (conflict.conflictType === 'server_newer') {
    message += '\nServer has a newer version than your local copy.';
  }

  const actions: NotificationAction[] = [];

  if (conflict.conflictType === 'local_newer' || conflict.conflictType === 'local_only') {
    actions.push(
      { label: 'Upload to Server', value: 'use_local', variant: 'default' },
      { label: 'Discard Local', value: 'use_server', variant: 'destructive' },
      { label: 'Skip', value: 'skip', variant: 'outline' }
    );
  } else {
    actions.push(
      { label: 'Download from Server', value: 'use_server', variant: 'default' },
      { label: 'Keep Local', value: 'skip', variant: 'outline' }
    );
  }

  return {
    type: 'action',
    title,
    message,
    requiresAction: true,
    actions,
    conflictPath: conflict.path,
    conflictType: conflict.conflictType,
    metadata: {
      path: conflict.path,
      conflictType: conflict.conflictType,
      localHash: conflict.localHash,
      serverHash: conflict.serverHash,
    },
  };
}
