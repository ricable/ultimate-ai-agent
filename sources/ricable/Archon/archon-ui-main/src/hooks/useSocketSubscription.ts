import { useEffect, useCallback, DependencyList } from 'react';
import { WebSocketService, WebSocketMessage } from '../services/socketIOService';

/**
 * Hook for managing Socket.IO subscriptions with proper cleanup and memoization
 * 
 * @example
 * useSocketSubscription(
 *   taskUpdateSocketIO,
 *   'task_updated',
 *   (data) => {
 *     console.log('Task updated:', data);
 *   },
 *   [dependency1, dependency2]
 * );
 */
export function useSocketSubscription<T = any>(
  socket: WebSocketService,
  eventName: string,
  handler: (data: T) => void,
  deps: DependencyList = []
) {
  // Memoize the handler
  const stableHandler = useCallback(handler, deps);
  
  useEffect(() => {
    const messageHandler = (message: WebSocketMessage) => {
      stableHandler(message.data || message);
    };
    
    socket.addMessageHandler(eventName, messageHandler);
    
    return () => {
      socket.removeMessageHandler(eventName, messageHandler);
    };
  }, [socket, eventName, stableHandler]);
}