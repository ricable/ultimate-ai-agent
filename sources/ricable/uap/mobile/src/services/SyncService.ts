import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import BackgroundJob from 'react-native-background-job';
import { store } from '../store';
import { setSyncStatus, addPendingOperation, removePendingOperation } from '../store/slices/syncSlice';
import { setOnlineStatus } from '../store/slices/offlineSlice';
import { APIClient } from './APIClient';
import { Logger } from '../utils/Logger';

export interface PendingOperation {
  id: string;
  type: 'chat_message' | 'document_upload' | 'agent_interaction';
  data: any;
  timestamp: number;
  retryCount: number;
  maxRetries: number;
}

class SyncServiceClass {
  private syncInterval: NodeJS.Timeout | null = null;
  private backgroundJobKey = 'uap-sync';
  private isInitialized = false;
  private isSyncing = false;
  private maxRetries = 3;
  private syncIntervalMs = 30000; // 30 seconds

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    Logger.info('SyncService', 'Initializing sync service');

    // Monitor network status
    NetInfo.addEventListener(state => {
      const isOnline = state.isConnected ?? false;
      store.dispatch(setOnlineStatus(isOnline));
      
      if (isOnline && !this.isSyncing) {
        this.syncPendingOperations();
      }
    });

    // Load pending operations from storage
    await this.loadPendingOperations();

    // Start periodic sync
    this.startPeriodicSync();

    this.isInitialized = true;
    Logger.info('SyncService', 'Sync service initialized');
  }

  async addPendingOperation(operation: Omit<PendingOperation, 'id' | 'timestamp' | 'retryCount'>): Promise<void> {
    const pendingOp: PendingOperation = {
      ...operation,
      id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      retryCount: 0,
    };

    store.dispatch(addPendingOperation(pendingOp));
    await this.savePendingOperations();

    Logger.info('SyncService', `Added pending operation: ${pendingOp.type}`, pendingOp);

    // Try to sync immediately if online
    const networkState = await NetInfo.fetch();
    if (networkState.isConnected) {
      this.syncPendingOperations();
    }
  }

  async syncPendingOperations(): Promise<void> {
    if (this.isSyncing) return;

    const networkState = await NetInfo.fetch();
    if (!networkState.isConnected) {
      Logger.warn('SyncService', 'Cannot sync - device is offline');
      return;
    }

    this.isSyncing = true;
    store.dispatch(setSyncStatus('syncing'));

    try {
      const state = store.getState();
      const pendingOps = state.sync.pendingOperations;

      Logger.info('SyncService', `Starting sync of ${pendingOps.length} operations`);

      for (const operation of pendingOps) {
        try {
          await this.syncOperation(operation);
          store.dispatch(removePendingOperation(operation.id));
          Logger.info('SyncService', `Synced operation: ${operation.type}`);
        } catch (error) {
          Logger.error('SyncService', `Failed to sync operation: ${operation.type}`, error);
          
          // Increment retry count
          const updatedOp = {
            ...operation,
            retryCount: operation.retryCount + 1,
          };

          if (updatedOp.retryCount >= updatedOp.maxRetries) {
            Logger.warn('SyncService', `Max retries reached for operation: ${operation.type}`);
            store.dispatch(removePendingOperation(operation.id));
          } else {
            // Update with incremented retry count
            store.dispatch(removePendingOperation(operation.id));
            store.dispatch(addPendingOperation(updatedOp));
          }
        }
      }

      await this.savePendingOperations();
      store.dispatch(setSyncStatus('completed'));
      Logger.info('SyncService', 'Sync completed successfully');
    } catch (error) {
      Logger.error('SyncService', 'Sync failed', error);
      store.dispatch(setSyncStatus('failed'));
    } finally {
      this.isSyncing = false;
    }
  }

  async syncNow(): Promise<void> {
    Logger.info('SyncService', 'Manual sync requested');
    await this.syncPendingOperations();
  }

  private async syncOperation(operation: PendingOperation): Promise<void> {
    switch (operation.type) {
      case 'chat_message':
        await APIClient.sendChatMessage(operation.data);
        break;
      case 'document_upload':
        await APIClient.uploadDocument(operation.data);
        break;
      case 'agent_interaction':
        await APIClient.sendAgentInteraction(operation.data);
        break;
      default:
        throw new Error(`Unknown operation type: ${operation.type}`);
    }
  }

  private async loadPendingOperations(): Promise<void> {
    try {
      const stored = await AsyncStorage.getItem('pending_operations');
      if (stored) {
        const operations: PendingOperation[] = JSON.parse(stored);
        operations.forEach(op => {
          store.dispatch(addPendingOperation(op));
        });
        Logger.info('SyncService', `Loaded ${operations.length} pending operations`);
      }
    } catch (error) {
      Logger.error('SyncService', 'Failed to load pending operations', error);
    }
  }

  private async savePendingOperations(): Promise<void> {
    try {
      const state = store.getState();
      const operations = state.sync.pendingOperations;
      await AsyncStorage.setItem('pending_operations', JSON.stringify(operations));
    } catch (error) {
      Logger.error('SyncService', 'Failed to save pending operations', error);
    }
  }

  private startPeriodicSync(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
    }

    this.syncInterval = setInterval(() => {
      this.syncPendingOperations();
    }, this.syncIntervalMs);

    Logger.info('SyncService', `Started periodic sync every ${this.syncIntervalMs}ms`);
  }

  scheduleBackgroundSync(): void {
    BackgroundJob.register({
      jobKey: this.backgroundJobKey,
      period: 15000, // 15 seconds
    });

    BackgroundJob.on('background-job', () => {
      Logger.info('SyncService', 'Background sync triggered');
      this.syncPendingOperations();
    });

    BackgroundJob.start(this.backgroundJobKey);
    Logger.info('SyncService', 'Background sync scheduled');
  }

  resumeSync(): void {
    BackgroundJob.stop(this.backgroundJobKey);
    this.syncPendingOperations();
    Logger.info('SyncService', 'Resumed foreground sync');
  }

  async getPendingOperationsCount(): Promise<number> {
    const state = store.getState();
    return state.sync.pendingOperations.length;
  }

  async clearPendingOperations(): Promise<void> {
    const state = store.getState();
    const operations = state.sync.pendingOperations;
    
    operations.forEach(op => {
      store.dispatch(removePendingOperation(op.id));
    });
    
    await AsyncStorage.removeItem('pending_operations');
    Logger.info('SyncService', 'Cleared all pending operations');
  }

  destroy(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
    }
    
    BackgroundJob.stop(this.backgroundJobKey);
    this.isInitialized = false;
    Logger.info('SyncService', 'Sync service destroyed');
  }
}

export const SyncService = new SyncServiceClass();