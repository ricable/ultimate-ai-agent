import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import { Logger } from '../utils/Logger';
import { store } from '../store';
import { setOnlineStatus } from '../store/slices/offlineSlice';

export interface OfflineData {
  id: string;
  type: 'conversation' | 'message' | 'document' | 'agent_state' | 'user_preference';
  data: any;
  timestamp: number;
  version: number;
  isDeleted: boolean;
  needsSync: boolean;
}

export interface OfflineOperation {
  id: string;
  type: 'create' | 'update' | 'delete' | 'sync';
  entityType: string;
  entityId: string;
  data: any;
  timestamp: number;
  retryCount: number;
  maxRetries: number;
}

class OfflineServiceClass {
  private isInitialized = false;
  private storagePrefix = 'uap_offline_';
  private operationsKey = 'uap_offline_operations';
  private maxStorageSize = 50 * 1024 * 1024; // 50MB
  private cleanupInterval: NodeJS.Timeout | null = null;

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    Logger.info('OfflineService', 'Initializing offline service');

    try {
      // Monitor network status
      NetInfo.addEventListener(state => {
        const isOnline = state.isConnected ?? false;
        store.dispatch(setOnlineStatus(isOnline));
        
        if (isOnline) {
          Logger.info('OfflineService', 'Device came online');
        } else {
          Logger.info('OfflineService', 'Device went offline');
        }
      });

      // Start periodic cleanup
      this.startCleanupTimer();

      // Load offline operations on startup
      await this.loadOfflineOperations();

      this.isInitialized = true;
      Logger.info('OfflineService', 'Offline service initialized');

    } catch (error) {
      Logger.error('OfflineService', 'Failed to initialize offline service', error);
      throw error;
    }
  }

  // Data Storage Methods

  async storeData(type: string, id: string, data: any): Promise<void> {
    try {
      const offlineData: OfflineData = {
        id,
        type: type as any,
        data,
        timestamp: Date.now(),
        version: 1,
        isDeleted: false,
        needsSync: true,
      };

      // Check if data already exists and increment version
      const existing = await this.getData(type, id);
      if (existing) {
        offlineData.version = existing.version + 1;
      }

      const key = this.getStorageKey(type, id);
      await AsyncStorage.setItem(key, JSON.stringify(offlineData));

      // Add to sync queue
      await this.addOfflineOperation('create', type, id, data);

      Logger.info('OfflineService', `Stored ${type} data offline`, { id, version: offlineData.version });
    } catch (error) {
      Logger.error('OfflineService', `Failed to store ${type} data`, error);
      throw error;
    }
  }

  async getData(type: string, id: string): Promise<OfflineData | null> {
    try {
      const key = this.getStorageKey(type, id);
      const stored = await AsyncStorage.getItem(key);
      
      if (!stored) return null;
      
      const data = JSON.parse(stored) as OfflineData;
      return data.isDeleted ? null : data;
    } catch (error) {
      Logger.error('OfflineService', `Failed to get ${type} data`, error);
      return null;
    }
  }

  async updateData(type: string, id: string, data: any): Promise<void> {
    try {
      const existing = await this.getData(type, id);
      if (!existing) {
        await this.storeData(type, id, data);
        return;
      }

      const updatedData: OfflineData = {
        ...existing,
        data,
        timestamp: Date.now(),
        version: existing.version + 1,
        needsSync: true,
      };

      const key = this.getStorageKey(type, id);
      await AsyncStorage.setItem(key, JSON.stringify(updatedData));

      // Add to sync queue
      await this.addOfflineOperation('update', type, id, data);

      Logger.info('OfflineService', `Updated ${type} data offline`, { id, version: updatedData.version });
    } catch (error) {
      Logger.error('OfflineService', `Failed to update ${type} data`, error);
      throw error;
    }
  }

  async deleteData(type: string, id: string): Promise<void> {
    try {
      const existing = await this.getData(type, id);
      if (!existing) return;

      const deletedData: OfflineData = {
        ...existing,
        timestamp: Date.now(),
        version: existing.version + 1,
        isDeleted: true,
        needsSync: true,
      };

      const key = this.getStorageKey(type, id);
      await AsyncStorage.setItem(key, JSON.stringify(deletedData));

      // Add to sync queue
      await this.addOfflineOperation('delete', type, id, null);

      Logger.info('OfflineService', `Deleted ${type} data offline`, { id });
    } catch (error) {
      Logger.error('OfflineService', `Failed to delete ${type} data`, error);
      throw error;
    }
  }

  async getAllData(type: string): Promise<OfflineData[]> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const prefix = this.getStorageKey(type, '');
      const matchingKeys = keys.filter(key => key.startsWith(prefix));

      const items = await AsyncStorage.multiGet(matchingKeys);
      const data: OfflineData[] = [];

      for (const [key, value] of items) {
        if (value) {
          try {
            const parsed = JSON.parse(value) as OfflineData;
            if (!parsed.isDeleted) {
              data.push(parsed);
            }
          } catch (parseError) {
            Logger.error('OfflineService', `Failed to parse stored data for key ${key}`, parseError);
          }
        }
      }

      return data.sort((a, b) => b.timestamp - a.timestamp);
    } catch (error) {
      Logger.error('OfflineService', `Failed to get all ${type} data`, error);
      return [];
    }
  }

  // Offline Operations Management

  async addOfflineOperation(
    operationType: 'create' | 'update' | 'delete' | 'sync',
    entityType: string,
    entityId: string,
    data: any
  ): Promise<void> {
    try {
      const operation: OfflineOperation = {
        id: `${operationType}_${entityType}_${entityId}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: operationType,
        entityType,
        entityId,
        data,
        timestamp: Date.now(),
        retryCount: 0,
        maxRetries: 3,
      };

      const operations = await this.getOfflineOperations();
      operations.push(operation);
      
      await AsyncStorage.setItem(this.operationsKey, JSON.stringify(operations));
      
      Logger.info('OfflineService', 'Added offline operation', {
        type: operationType,
        entityType,
        entityId
      });
    } catch (error) {
      Logger.error('OfflineService', 'Failed to add offline operation', error);
    }
  }

  async getOfflineOperations(): Promise<OfflineOperation[]> {
    try {
      const stored = await AsyncStorage.getItem(this.operationsKey);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      Logger.error('OfflineService', 'Failed to get offline operations', error);
      return [];
    }
  }

  async removeOfflineOperation(operationId: string): Promise<void> {
    try {
      const operations = await this.getOfflineOperations();
      const filtered = operations.filter(op => op.id !== operationId);
      
      await AsyncStorage.setItem(this.operationsKey, JSON.stringify(filtered));
      
      Logger.info('OfflineService', 'Removed offline operation', { operationId });
    } catch (error) {
      Logger.error('OfflineService', 'Failed to remove offline operation', error);
    }
  }

  async clearOfflineOperations(): Promise<void> {
    try {
      await AsyncStorage.removeItem(this.operationsKey);
      Logger.info('OfflineService', 'Cleared all offline operations');
    } catch (error) {
      Logger.error('OfflineService', 'Failed to clear offline operations', error);
    }
  }

  // Network Status

  async isOnline(): Promise<boolean> {
    try {
      const state = await NetInfo.fetch();
      return state.isConnected ?? false;
    } catch (error) {
      Logger.error('OfflineService', 'Failed to check network status', error);
      return false;
    }
  }

  // Storage Management

  async getStorageInfo(): Promise<{
    totalSize: number;
    itemCount: number;
    operationCount: number;
  }> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const offlineKeys = keys.filter(key => key.startsWith(this.storagePrefix));
      const items = await AsyncStorage.multiGet(offlineKeys);
      
      let totalSize = 0;
      for (const [key, value] of items) {
        if (value) {
          totalSize += value.length * 2; // Rough estimate (2 bytes per character)
        }
      }

      const operations = await this.getOfflineOperations();
      
      return {
        totalSize,
        itemCount: offlineKeys.length,
        operationCount: operations.length,
      };
    } catch (error) {
      Logger.error('OfflineService', 'Failed to get storage info', error);
      return { totalSize: 0, itemCount: 0, operationCount: 0 };
    }
  }

  async cleanupOldData(maxAge: number = 7 * 24 * 60 * 60 * 1000): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const offlineKeys = keys.filter(key => key.startsWith(this.storagePrefix));
      const items = await AsyncStorage.multiGet(offlineKeys);
      
      const now = Date.now();
      const keysToRemove: string[] = [];
      
      for (const [key, value] of items) {
        if (value) {
          try {
            const data = JSON.parse(value) as OfflineData;
            
            // Remove old deleted items or old items that have been synced
            if ((data.isDeleted && (now - data.timestamp) > maxAge) ||
                (!data.needsSync && (now - data.timestamp) > maxAge)) {
              keysToRemove.push(key);
            }
          } catch (parseError) {
            // Remove corrupted data
            keysToRemove.push(key);
          }
        }
      }
      
      if (keysToRemove.length > 0) {
        await AsyncStorage.multiRemove(keysToRemove);
        Logger.info('OfflineService', `Cleaned up ${keysToRemove.length} old offline items`);
      }
      
      // Cleanup old operations
      const operations = await this.getOfflineOperations();
      const validOperations = operations.filter(op => (now - op.timestamp) < maxAge);
      
      if (validOperations.length !== operations.length) {
        await AsyncStorage.setItem(this.operationsKey, JSON.stringify(validOperations));
        Logger.info('OfflineService', `Cleaned up ${operations.length - validOperations.length} old operations`);
      }
      
    } catch (error) {
      Logger.error('OfflineService', 'Failed to cleanup old data', error);
    }
  }

  async checkStorageSize(): Promise<void> {
    try {
      const info = await this.getStorageInfo();
      
      if (info.totalSize > this.maxStorageSize) {
        Logger.warn('OfflineService', `Storage size limit exceeded: ${info.totalSize} bytes`);
        
        // Perform aggressive cleanup
        await this.cleanupOldData(3 * 24 * 60 * 60 * 1000); // 3 days instead of 7
        
        const newInfo = await this.getStorageInfo();
        Logger.info('OfflineService', `Storage after cleanup: ${newInfo.totalSize} bytes`);
      }
    } catch (error) {
      Logger.error('OfflineService', 'Failed to check storage size', error);
    }
  }

  // Data Synchronization Helpers

  async markDataSynced(type: string, id: string): Promise<void> {
    try {
      const data = await this.getData(type, id);
      if (!data) return;

      const syncedData: OfflineData = {
        ...data,
        needsSync: false,
      };

      const key = this.getStorageKey(type, id);
      await AsyncStorage.setItem(key, JSON.stringify(syncedData));
      
      Logger.info('OfflineService', `Marked ${type} data as synced`, { id });
    } catch (error) {
      Logger.error('OfflineService', `Failed to mark ${type} data as synced`, error);
    }
  }

  async getUnsyncedData(type?: string): Promise<OfflineData[]> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const prefix = type ? this.getStorageKey(type, '') : this.storagePrefix;
      const matchingKeys = keys.filter(key => key.startsWith(prefix));

      const items = await AsyncStorage.multiGet(matchingKeys);
      const unsyncedData: OfflineData[] = [];

      for (const [key, value] of items) {
        if (value) {
          try {
            const parsed = JSON.parse(value) as OfflineData;
            if (parsed.needsSync && !parsed.isDeleted) {
              unsyncedData.push(parsed);
            }
          } catch (parseError) {
            Logger.error('OfflineService', `Failed to parse stored data for key ${key}`, parseError);
          }
        }
      }

      return unsyncedData.sort((a, b) => a.timestamp - b.timestamp);
    } catch (error) {
      Logger.error('OfflineService', 'Failed to get unsynced data', error);
      return [];
    }
  }

  // Utility Methods

  private getStorageKey(type: string, id: string): string {
    return `${this.storagePrefix}${type}_${id}`;
  }

  private async loadOfflineOperations(): Promise<void> {
    try {
      const operations = await this.getOfflineOperations();
      Logger.info('OfflineService', `Loaded ${operations.length} offline operations`);
    } catch (error) {
      Logger.error('OfflineService', 'Failed to load offline operations', error);
    }
  }

  private startCleanupTimer(): void {
    // Run cleanup every hour
    this.cleanupInterval = setInterval(() => {
      this.cleanupOldData();
      this.checkStorageSize();
    }, 60 * 60 * 1000);
  }

  async clearAllOfflineData(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const offlineKeys = keys.filter(key => key.startsWith(this.storagePrefix));
      
      await AsyncStorage.multiRemove([...offlineKeys, this.operationsKey]);
      
      Logger.info('OfflineService', 'Cleared all offline data');
    } catch (error) {
      Logger.error('OfflineService', 'Failed to clear all offline data', error);
    }
  }

  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    
    this.isInitialized = false;
    Logger.info('OfflineService', 'Offline service destroyed');
  }
}

export const OfflineService = new OfflineServiceClass();