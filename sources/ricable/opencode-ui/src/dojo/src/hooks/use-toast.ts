/**
 * Simple toast hook for displaying notifications
 */

import { useCallback } from 'react';

export interface Toast {
  id: string;
  title?: string;
  description?: string;
  variant?: 'default' | 'destructive' | 'success';
  duration?: number;
}

export interface UseToastReturn {
  toast: (props: Omit<Toast, 'id'>) => void;
  dismiss: (toastId?: string) => void;
}

// Simple implementation without state management
// In a real app, you'd use a toast provider/context
export function useToast(): UseToastReturn {
  const toast = useCallback((props: Omit<Toast, 'id'>) => {
    // For now, just log to console
    // In a real implementation, you'd add to a toast state/context
    console.log('Toast:', props);
    
    // Show a simple browser notification as fallback
    if (props.title || props.description) {
      const message = [props.title, props.description].filter(Boolean).join(': ');
      alert(message);
    }
  }, []);

  const dismiss = useCallback((toastId?: string) => {
    console.log('Dismiss toast:', toastId);
  }, []);

  return { toast, dismiss };
}