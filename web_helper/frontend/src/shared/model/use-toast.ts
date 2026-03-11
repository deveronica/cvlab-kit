import { message } from 'antd';

type ToastVariant = 'default' | 'success' | 'error' | 'warning' | 'info';

export interface ToastOptions {
  title?: string;
  description?: string;
  variant?: ToastVariant;
  duration?: number;
}

export function useToast() {
  return {
    toast: ({ title, description, variant = 'default', duration }: ToastOptions) => {
      const content = [title, description].filter(Boolean).join('\n');
      const durationSeconds = typeof duration === 'number' ? duration / 1000 : undefined;

      switch (variant) {
        case 'success':
          message.success({ content, duration: durationSeconds });
          return;
        case 'error':
          message.error({ content, duration: durationSeconds });
          return;
        case 'warning':
          message.warning({ content, duration: durationSeconds });
          return;
        case 'info':
          message.info({ content, duration: durationSeconds });
          return;
        default:
          message.open({ type: 'info', content, duration: durationSeconds });
      }
    },
  };
}
