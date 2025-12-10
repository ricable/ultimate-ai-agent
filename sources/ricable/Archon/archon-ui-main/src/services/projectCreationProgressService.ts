import type { Project } from '../types/project';
import { createWebSocketService, WebSocketService, WebSocketState } from './socketIOService';

export interface ProjectCreationProgressData {
  progressId: string;
  status: 'starting' | 'initializing_agents' | 'generating_docs' | 'processing_requirements' | 'ai_generation' | 'finalizing_docs' | 'saving_to_database' | 'completed' | 'error';
  percentage: number;
  step?: string;
  currentStep?: string;
  eta?: string;
  error?: string;
  logs: string[];
  project?: Project; // The created project when completed
  duration?: string;
}

interface StreamProgressOptions {
  autoReconnect?: boolean;
  reconnectDelay?: number;
}

type ProgressCallback = (data: ProjectCreationProgressData) => void;

class ProjectCreationProgressService {
  private wsService: WebSocketService | null = null;
  private currentProgressId: string | null = null;
  private progressCallback: ProgressCallback | null = null;
  public isReconnecting = false;

  /**
   * Stream project creation progress using Socket.IO
   */
  async streamProgress(
    progressId: string,
    onMessage: ProgressCallback,
    options: StreamProgressOptions = {}
  ): Promise<void> {
    const { autoReconnect = true, reconnectDelay = 5000 } = options;

    // Close existing connection if any
    this.disconnect();

    this.currentProgressId = progressId;
    this.progressCallback = onMessage;

    // Create new WebSocket service with Socket.IO
    this.wsService = createWebSocketService({
      maxReconnectAttempts: autoReconnect ? 10 : 0,
      reconnectInterval: reconnectDelay,
      enableAutoReconnect: autoReconnect
    });

    // Set up state change handler
    this.wsService.addStateChangeHandler((state) => {
      if (state === WebSocketState.CONNECTED) {
        console.log(`🚀 Connected to project creation progress stream: ${progressId}`);
        this.isReconnecting = false;
        // Note: subscribe_progress is now automatically emitted by webSocketService on connect
      } else if (state === WebSocketState.RECONNECTING) {
        this.isReconnecting = true;
      } else if (state === WebSocketState.DISCONNECTED || state === WebSocketState.FAILED) {
        this.isReconnecting = false;
      }
    });

    // Set up message handlers
    this.wsService.addMessageHandler('project_progress', (message) => {
      console.log(`📨 [PROGRESS] Received project_progress event:`, message);
      if (message.data) {
        console.log(`📨 [PROGRESS] Calling onMessage with data:`, message.data);
        onMessage(message.data);
      } else {
        console.warn(`📨 [PROGRESS] project_progress event had no data:`, message);
      }
    });

    this.wsService.addMessageHandler('project_completed', (message) => {
      console.log(`✅ [PROGRESS] Received project_completed event:`, message);
      if (message.data) {
        console.log(`✅ [PROGRESS] Calling onMessage with completion data:`, message.data);
        onMessage(message.data);
      } else {
        console.warn(`✅ [PROGRESS] project_completed event had no data:`, message);
      }
    });

    this.wsService.addMessageHandler('project_error', (message) => {
      console.log(`❌ [PROGRESS] Received project_error event:`, message);
      if (message.data) {
        console.log(`❌ [PROGRESS] Calling onMessage with error data:`, message.data);
        onMessage(message.data);
      } else {
        console.warn(`❌ [PROGRESS] project_error event had no data:`, message);
      }
    });

    // Set up error handler
    this.wsService.addErrorHandler((error) => {
      console.error('Project creation progress Socket.IO error:', error);
    });

    // Connect to the default namespace and join progress room
    try {
      console.log(`📡 [PROGRESS] Connecting to Socket.IO for progress: ${progressId}`);
      await this.wsService.connect('/');
      console.log(`📡 [PROGRESS] Connected! Sending subscribe_progress event for: ${progressId}`);
      
      // Subscribe to progress updates for this specific progress ID
      this.wsService.send({
        type: 'subscribe_progress',
        data: { progress_id: progressId }
      });
      console.log(`📡 [PROGRESS] Sent subscribe_progress event, now listening for project_progress/project_completed/project_error events`);
    } catch (error) {
      console.error('Failed to connect to project creation progress:', error);
      throw error;
    }
  }

  /**
   * Disconnect from progress stream
   */
  disconnect(): void {
    if (this.wsService) {
      // Unsubscribe if connected
      if (this.currentProgressId && this.wsService.isConnected()) {
        this.wsService.send({
          type: 'unsubscribe_progress',
          data: { progress_id: this.currentProgressId }
        });
      }
      
      this.wsService.disconnect();
      this.wsService = null;
    }
    
    this.currentProgressId = null;
    this.progressCallback = null;
    this.isReconnecting = false;
  }

  /**
   * Check if currently connected to a progress stream
   */
  isConnected(): boolean {
    return this.wsService?.isConnected() ?? false;
  }

  // Backward compatibility methods - now just wrappers around streamProgress
  connect(progressId: string): void {
    // This method is kept for backward compatibility but does nothing
    // Use streamProgress instead
    console.warn('projectCreationProgressService.connect() is deprecated. Use streamProgress() instead.');
  }

  onProgress(callback: ProgressCallback): void {
    console.warn('projectCreationProgressService.onProgress() is deprecated. Pass callback to streamProgress() instead.');
  }

  onCompleted(callback: ProgressCallback): void {
    console.warn('projectCreationProgressService.onCompleted() is deprecated. Pass callback to streamProgress() instead.');
  }

  onError(callback: (error: Error) => void): void {
    console.warn('projectCreationProgressService.onError() is deprecated. Pass callback to streamProgress() instead.');
  }
}

// Export singleton instance
export const projectCreationProgressService = new ProjectCreationProgressService(); 