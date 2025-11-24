import React from 'react';
import { X, Bell, BellOff, Check, Trash2, AlertCircle, Info, AlertTriangle, CheckCircle } from 'lucide-react';
import { Button } from '../ui/button';
import { ScrollArea } from '../ui/scroll-area';
import { Badge } from '../ui/badge';
import { cn } from '@/lib/utils';
import {
  useNotificationStore,
  Notification,
  NotificationType,
} from '@/store/notificationStore';

const typeIcons: Record<NotificationType, React.ReactNode> = {
  info: <Info className="h-4 w-4 text-blue-500" />,
  warning: <AlertTriangle className="h-4 w-4 text-yellow-500" />,
  error: <AlertCircle className="h-4 w-4 text-red-500" />,
  success: <CheckCircle className="h-4 w-4 text-green-500" />,
  action: <Bell className="h-4 w-4 text-primary" />,
};

const typeColors: Record<NotificationType, string> = {
  info: 'border-l-blue-500',
  warning: 'border-l-yellow-500',
  error: 'border-l-red-500',
  success: 'border-l-green-500',
  action: 'border-l-primary',
};

function formatTimeAgo(timestamp: number): string {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'Just now';
}

interface NotificationItemProps {
  notification: Notification;
  onResolve: (id: string, action: string) => void;
  onDismiss: (id: string) => void;
}

function NotificationItem({ notification, onResolve, onDismiss }: NotificationItemProps) {
  return (
    <div
      className={cn(
        'p-3 border-l-4 rounded-r-md bg-card transition-all duration-200',
        typeColors[notification.type],
        notification.read ? 'opacity-60' : 'opacity-100',
        notification.requiresAction && !notification.read && 'ring-1 ring-primary/20'
      )}
    >
      <div className="flex items-start gap-2">
        <div className="mt-0.5">{typeIcons[notification.type]}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-sm font-medium truncate">{notification.title}</h4>
            <span className="text-xs text-muted-foreground whitespace-nowrap">
              {formatTimeAgo(notification.timestamp)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1 whitespace-pre-line">
            {notification.message}
          </p>

          {/* Actions for action-required notifications */}
          {notification.requiresAction && notification.actions && !notification.read && (
            <div className="flex flex-wrap gap-2 mt-3">
              {notification.actions.map((action) => (
                <Button
                  key={action.value}
                  size="sm"
                  variant={action.variant || 'default'}
                  className="h-7 text-xs"
                  onClick={() => onResolve(notification.id, action.value)}
                >
                  {action.label}
                </Button>
              ))}
            </div>
          )}
        </div>

        {/* Dismiss button for non-action notifications */}
        {!notification.requiresAction && (
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 opacity-0 group-hover:opacity-100"
            onClick={() => onDismiss(notification.id)}
          >
            <X className="h-3 w-3" />
          </Button>
        )}
      </div>
    </div>
  );
}

export function NotificationPanel() {
  const {
    notifications,
    isOpen,
    setOpen,
    removeNotification,
    markAllAsRead,
    clearAll,
    resolveAction,
    getUnreadCount,
    getPendingActions,
  } = useNotificationStore();

  const unreadCount = getUnreadCount();
  const pendingActions = getPendingActions();

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-background/50 backdrop-blur-sm z-40"
        onClick={() => setOpen(false)}
      />

      {/* Panel */}
      <div
        className={cn(
          'fixed top-0 right-0 h-full w-80 bg-background border-l border-border shadow-xl z-50',
          'transform transition-transform duration-300 ease-in-out',
          isOpen ? 'translate-x-0' : 'translate-x-full'
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            <h2 className="font-semibold">Notifications</h2>
            {unreadCount > 0 && (
              <Badge variant="default" className="h-5 px-1.5 text-xs">
                {unreadCount}
              </Badge>
            )}
          </div>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setOpen(false)}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Action required banner */}
        {pendingActions.length > 0 && (
          <div className="px-4 py-2 bg-primary/10 border-b border-primary/20">
            <p className="text-xs text-primary font-medium">
              {pendingActions.length} action{pendingActions.length > 1 ? 's' : ''} required
            </p>
          </div>
        )}

        {/* Notification list */}
        <ScrollArea className="flex-1 h-[calc(100vh-120px)]">
          <div className="p-2 space-y-2">
            {notifications.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <BellOff className="h-10 w-10 mb-3 opacity-30" />
                <p className="text-sm">No notifications</p>
              </div>
            ) : (
              notifications.map((notification) => (
                <div key={notification.id} className="group">
                  <NotificationItem
                    notification={notification}
                    onResolve={resolveAction}
                    onDismiss={removeNotification}
                  />
                </div>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Footer actions */}
        {notifications.length > 0 && (
          <div className="absolute bottom-0 left-0 right-0 px-4 py-3 border-t border-border bg-background">
            <div className="flex items-center justify-between gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="text-xs"
                onClick={markAllAsRead}
                disabled={unreadCount === 0}
              >
                <Check className="h-3 w-3 mr-1" />
                Mark all read
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-destructive hover:text-destructive"
                onClick={clearAll}
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Clear all
              </Button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

// Notification trigger button (to be placed in header)
export function NotificationButton() {
  const { toggle, getUnreadCount, getPendingActions } = useNotificationStore();
  const unreadCount = getUnreadCount();
  const pendingActions = getPendingActions();

  return (
    <Button
      variant="ghost"
      size="icon"
      className="relative h-8 w-8"
      onClick={toggle}
    >
      <Bell className={cn('h-4 w-4', pendingActions.length > 0 && 'text-primary')} />
      {unreadCount > 0 && (
        <span
          className={cn(
            'absolute -top-0.5 -right-0.5 flex items-center justify-center',
            'min-w-[16px] h-4 px-1 rounded-full text-[10px] font-medium',
            pendingActions.length > 0
              ? 'bg-primary text-primary-foreground animate-pulse'
              : 'bg-muted-foreground text-background'
          )}
        >
          {unreadCount > 99 ? '99+' : unreadCount}
        </span>
      )}
    </Button>
  );
}
