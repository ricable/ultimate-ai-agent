import { useRef, useCallback } from 'react';

export interface PendingUpdate<T> {
  id: string;
  timestamp: number;
  data: T;
  operation: 'create' | 'update' | 'delete' | 'reorder';
}

/**
 * Hook for tracking optimistic updates to prevent re-applying server echoes
 * 
 * @example
 * const { addPendingUpdate, isPendingUpdate } = useOptimisticUpdates<Task>();
 * 
 * // When making an optimistic update
 * addPendingUpdate({
 *   id: task.id,
 *   timestamp: Date.now(),
 *   data: updatedTask,
 *   operation: 'update'
 * });
 * 
 * // When receiving server update
 * if (!isPendingUpdate(task.id, serverTask)) {
 *   // Apply the update
 * }
 */
export function useOptimisticUpdates<T extends { id: string }>() {
  const pendingUpdatesRef = useRef<Map<string, PendingUpdate<T>>>(new Map());
  
  const addPendingUpdate = useCallback((update: PendingUpdate<T>) => {
    pendingUpdatesRef.current.set(update.id, update);
    // Auto-cleanup after 5 seconds
    setTimeout(() => {
      pendingUpdatesRef.current.delete(update.id);
    }, 5000);
  }, []);
  
  const isPendingUpdate = useCallback((id: string, data: T): boolean => {
    const pending = pendingUpdatesRef.current.get(id);
    if (!pending) return false;
    
    // Compare relevant fields based on operation type
    return JSON.stringify(pending.data) === JSON.stringify(data);
  }, []);
  
  const removePendingUpdate = useCallback((id: string) => {
    pendingUpdatesRef.current.delete(id);
  }, []);
  
  return { addPendingUpdate, isPendingUpdate, removePendingUpdate };
}