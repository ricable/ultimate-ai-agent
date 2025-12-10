/**
 * Task Socket Hook - Simplified real-time task synchronization
 * 
 * This hook provides a clean interface to the task socket service,
 * replacing the complex useOptimisticUpdates pattern with a simpler
 * approach that avoids conflicts and connection issues.
 */

import { useEffect, useRef, useCallback } from 'react';
import { taskSocketService, TaskSocketEvents } from '../services/taskSocketService';
import { WebSocketState } from '../services/socketIOService';

export interface UseTaskSocketOptions {
  projectId: string;
  onTaskCreated?: (task: any) => void;
  onTaskUpdated?: (task: any) => void;
  onTaskDeleted?: (task: any) => void;
  onTaskArchived?: (task: any) => void;
  onTasksReordered?: (data: any) => void;
  onInitialTasks?: (tasks: any[]) => void;
  onConnectionStateChange?: (state: WebSocketState) => void;
}

export function useTaskSocket(options: UseTaskSocketOptions) {
  const {
    projectId,
    onTaskCreated,
    onTaskUpdated,
    onTaskDeleted,
    onTaskArchived,
    onTasksReordered,
    onInitialTasks,
    onConnectionStateChange
  } = options;

  const componentIdRef = useRef<string>(`task-socket-${Math.random().toString(36).substring(7)}`);
  const currentProjectIdRef = useRef<string | null>(null);
  const isInitializedRef = useRef<boolean>(false);

  // Memoized handlers to prevent unnecessary re-registrations
  const memoizedHandlers = useCallback((): Partial<TaskSocketEvents> => {
    return {
      onTaskCreated,
      onTaskUpdated,
      onTaskDeleted,
      onTaskArchived,
      onTasksReordered,
      onInitialTasks,
      onConnectionStateChange
    };
  }, [
    onTaskCreated,
    onTaskUpdated,
    onTaskDeleted,
    onTaskArchived,
    onTasksReordered,
    onInitialTasks,
    onConnectionStateChange
  ]);

  // Initialize connection once and register handlers
  useEffect(() => {
    if (!projectId || isInitializedRef.current) return;

    const initializeConnection = async () => {
      try {
        console.log(`[USE_TASK_SOCKET] Initializing connection for project: ${projectId}`);
        
        // Register handlers first
        taskSocketService.registerHandlers(componentIdRef.current, memoizedHandlers());
        
        // Connect to project (singleton service will handle deduplication)
        await taskSocketService.connectToProject(projectId);
        
        currentProjectIdRef.current = projectId;
        isInitializedRef.current = true;
        console.log(`[USE_TASK_SOCKET] Successfully initialized for project: ${projectId}`);
        
      } catch (error) {
        console.error(`[USE_TASK_SOCKET] Failed to initialize for project ${projectId}:`, error);
      }
    };

    initializeConnection();

  }, [projectId, memoizedHandlers]);

  // Update handlers when they change (without reconnecting)
  useEffect(() => {
    if (isInitializedRef.current && currentProjectIdRef.current === projectId) {
      console.log(`[USE_TASK_SOCKET] Updating handlers for component: ${componentIdRef.current}`);
      taskSocketService.registerHandlers(componentIdRef.current, memoizedHandlers());
    }
  }, [memoizedHandlers, projectId]);

  // Handle project change (different project)
  useEffect(() => {
    if (!projectId) return;

    // If project changed, reconnect
    if (isInitializedRef.current && currentProjectIdRef.current !== projectId) {
      console.log(`[USE_TASK_SOCKET] Project changed from ${currentProjectIdRef.current} to ${projectId}`);
      
      const switchProject = async () => {
        try {
          // Update handlers for new project
          taskSocketService.registerHandlers(componentIdRef.current, memoizedHandlers());
          
          // Connect to new project (service handles disconnecting from old)
          await taskSocketService.connectToProject(projectId);
          
          currentProjectIdRef.current = projectId;
          console.log(`[USE_TASK_SOCKET] Successfully switched to project: ${projectId}`);
          
        } catch (error) {
          console.error(`[USE_TASK_SOCKET] Failed to switch to project ${projectId}:`, error);
        }
      };

      switchProject();
    }
  }, [projectId, memoizedHandlers]);

  // Cleanup on unmount
  useEffect(() => {
    const componentId = componentIdRef.current;
    
    return () => {
      console.log(`[USE_TASK_SOCKET] Cleaning up component: ${componentId}`);
      taskSocketService.unregisterHandlers(componentId);
      isInitializedRef.current = false;
    };
  }, []);

  // Return utility functions
  return {
    isConnected: taskSocketService.isConnected(),
    connectionState: taskSocketService.getConnectionState(),
    reconnect: taskSocketService.reconnect.bind(taskSocketService),
    getCurrentProjectId: taskSocketService.getCurrentProjectId.bind(taskSocketService)
  };
}