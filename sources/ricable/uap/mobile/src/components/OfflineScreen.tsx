import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
  Alert,
  Switch,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useNetwork } from '../hooks/useNetwork';
import { SyncService, PendingOperation } from '../services/SyncService';
import { OfflineService } from '../services/OfflineService';
import { Logger } from '../utils/Logger';

interface OfflineData {
  conversations: number;
  documents: number;
  settings: number;
  totalSize: string;
}

const OperationCard: React.FC<{ operation: PendingOperation; onRetry: () => void; onRemove: () => void }> = ({ 
  operation, 
  onRetry, 
  onRemove 
}) => {
  const getOperationIcon = (type: string) => {
    switch (type) {
      case 'chat_message': return 'chat';
      case 'document_upload': return 'cloud-upload';
      case 'agent_interaction': return 'smart-toy';
      default: return 'sync';
    }
  };

  const getOperationColor = (type: string) => {
    switch (type) {
      case 'chat_message': return '#2563EB';
      case 'document_upload': return '#059669';
      case 'agent_interaction': return '#7C3AED';
      default: return '#6B7280';
    }
  };

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = Date.now();
    const diff = now - timestamp;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;
    return date.toLocaleDateString();
  };

  return (
    <View style={styles.operationCard}>
      <View style={styles.operationHeader}>
        <View style={styles.operationInfo}>
          <Icon 
            name={getOperationIcon(operation.type)} 
            size={20} 
            color={getOperationColor(operation.type)} 
          />
          <View style={styles.operationDetails}>
            <Text style={styles.operationType}>
              {operation.type.replace('_', ' ').toUpperCase()}
            </Text>
            <Text style={styles.operationTime}>
              {formatTimestamp(operation.timestamp)}
            </Text>
          </View>
        </View>
        <View style={styles.operationActions}>
          <TouchableOpacity style={styles.actionButton} onPress={onRetry}>
            <Icon name="refresh" size={16} color="#2563EB" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton} onPress={onRemove}>
            <Icon name="close" size={16} color="#EF4444" />
          </TouchableOpacity>
        </View>
      </View>
      
      <View style={styles.operationMeta}>
        <Text style={styles.retryCount}>
          Retry {operation.retryCount}/{operation.maxRetries}
        </Text>
        {operation.retryCount > 0 && (
          <View style={styles.warningBadge}>
            <Icon name="warning" size={12} color="#F59E0B" />
            <Text style={styles.warningText}>Failed</Text>
          </View>
        )}
      </View>
    </View>
  );
};

const OfflineScreen: React.FC = () => {
  const { isOnline } = useNetwork();
  const [pendingOperations, setPendingOperations] = useState<PendingOperation[]>([]);
  const [offlineData, setOfflineData] = useState<OfflineData>({
    conversations: 0,
    documents: 0,
    settings: 0,
    totalSize: '0 MB'
  });
  const [refreshing, setRefreshing] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [autoSync, setAutoSync] = useState(true);
  const [backgroundSync, setBackgroundSync] = useState(true);

  useEffect(() => {
    loadOfflineData();
    loadPendingOperations();
  }, []);

  const loadOfflineData = async () => {
    try {
      const data = await OfflineService.getStorageStats();
      setOfflineData(data);
    } catch (error) {
      Logger.error('OfflineScreen', 'Failed to load offline data', error);
    }
  };

  const loadPendingOperations = async () => {
    try {
      // Get pending operations from store
      const operations = await SyncService.getPendingOperations();
      setPendingOperations(operations);
    } catch (error) {
      Logger.error('OfflineScreen', 'Failed to load pending operations', error);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await Promise.all([
        loadOfflineData(),
        loadPendingOperations()
      ]);
    } catch (error) {
      Logger.error('OfflineScreen', 'Failed to refresh', error);
    } finally {
      setRefreshing(false);
    }
  };

  const handleSyncNow = async () => {
    if (!isOnline) {
      Alert.alert('Offline', 'Cannot sync while offline. Please check your connection.');
      return;
    }

    setSyncing(true);
    try {
      await SyncService.syncNow();
      await loadPendingOperations();
      Alert.alert('Success', 'Sync completed successfully');
    } catch (error) {
      Logger.error('OfflineScreen', 'Sync failed', error);
      Alert.alert('Error', 'Failed to sync data. Please try again.');
    } finally {
      setSyncing(false);
    }
  };

  const handleClearOfflineData = () => {
    Alert.alert(
      'Clear Offline Data',
      'This will remove all offline data including conversations, documents, and settings. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            try {
              await OfflineService.clearAll();
              await loadOfflineData();
              Alert.alert('Success', 'Offline data cleared successfully');
            } catch (error) {
              Logger.error('OfflineScreen', 'Failed to clear offline data', error);
              Alert.alert('Error', 'Failed to clear offline data');
            }
          }
        }
      ]
    );
  };

  const handleClearPendingOperations = () => {
    Alert.alert(
      'Clear Pending Operations',
      'This will remove all pending sync operations. Unsaved changes may be lost.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            try {
              await SyncService.clearPendingOperations();
              await loadPendingOperations();
              Alert.alert('Success', 'Pending operations cleared');
            } catch (error) {
              Logger.error('OfflineScreen', 'Failed to clear pending operations', error);
              Alert.alert('Error', 'Failed to clear pending operations');
            }
          }
        }
      ]
    );
  };

  const handleRetryOperation = async (operationId: string) => {
    if (!isOnline) {
      Alert.alert('Offline', 'Cannot retry while offline');
      return;
    }

    try {
      // Trigger sync for specific operation
      await SyncService.syncNow();
      await loadPendingOperations();
    } catch (error) {
      Logger.error('OfflineScreen', 'Failed to retry operation', error);
      Alert.alert('Error', 'Failed to retry operation');
    }
  };

  const handleRemoveOperation = async (operationId: string) => {
    try {
      await SyncService.removePendingOperation(operationId);
      await loadPendingOperations();
    } catch (error) {
      Logger.error('OfflineScreen', 'Failed to remove operation', error);
    }
  };

  const handleAutoSyncToggle = (value: boolean) => {
    setAutoSync(value);
    // Save setting
    OfflineService.setAutoSync(value);
  };

  const handleBackgroundSyncToggle = (value: boolean) => {
    setBackgroundSync(value);
    if (value) {
      SyncService.scheduleBackgroundSync();
    } else {
      SyncService.resumeSync();
    }
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
      }
    >
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Icon name="cloud-off" size={32} color={isOnline ? '#10B981' : '#EF4444'} />
          <View style={styles.headerText}>
            <Text style={styles.headerTitle}>Offline Mode</Text>
            <Text style={[styles.headerStatus, { color: isOnline ? '#10B981' : '#EF4444' }]}>
              {isOnline ? 'Connected' : 'Offline'}
            </Text>
          </View>
        </View>
      </View>

      {/* Offline Data Stats */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Offline Data</Text>
        <View style={styles.statsGrid}>
          <View style={styles.statCard}>
            <Icon name="chat" size={24} color="#2563EB" />
            <Text style={styles.statNumber}>{offlineData.conversations}</Text>
            <Text style={styles.statLabel}>Conversations</Text>
          </View>
          <View style={styles.statCard}>
            <Icon name="description" size={24} color="#059669" />
            <Text style={styles.statNumber}>{offlineData.documents}</Text>
            <Text style={styles.statLabel}>Documents</Text>
          </View>
          <View style={styles.statCard}>
            <Icon name="settings" size={24} color="#7C3AED" />
            <Text style={styles.statNumber}>{offlineData.settings}</Text>
            <Text style={styles.statLabel}>Settings</Text>
          </View>
          <View style={styles.statCard}>
            <Icon name="storage" size={24} color="#F59E0B" />
            <Text style={styles.statNumber}>{offlineData.totalSize}</Text>
            <Text style={styles.statLabel}>Total Size</Text>
          </View>
        </View>
      </View>

      {/* Sync Settings */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Sync Settings</Text>
        <View style={styles.settingCard}>
          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text style={styles.settingTitle}>Auto Sync</Text>
              <Text style={styles.settingDescription}>
                Automatically sync when connection is restored
              </Text>
            </View>
            <Switch
              value={autoSync}
              onValueChange={handleAutoSyncToggle}
              trackColor={{ false: '#D1D5DB', true: '#BFDBFE' }}
              thumbColor={autoSync ? '#2563EB' : '#9CA3AF'}
            />
          </View>
        </View>
        
        <View style={styles.settingCard}>
          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text style={styles.settingTitle}>Background Sync</Text>
              <Text style={styles.settingDescription}>
                Sync data when app is in background
              </Text>
            </View>
            <Switch
              value={backgroundSync}
              onValueChange={handleBackgroundSyncToggle}
              trackColor={{ false: '#D1D5DB', true: '#BFDBFE' }}
              thumbColor={backgroundSync ? '#2563EB' : '#9CA3AF'}
            />
          </View>
        </View>
      </View>

      {/* Pending Operations */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>
            Pending Operations ({pendingOperations.length})
          </Text>
          {pendingOperations.length > 0 && (
            <TouchableOpacity
              style={styles.syncButton}
              onPress={handleSyncNow}
              disabled={!isOnline || syncing}
            >
              {syncing ? (
                <ActivityIndicator size="small" color="white" />
              ) : (
                <>
                  <Icon name="sync" size={16} color="white" />
                  <Text style={styles.syncButtonText}>Sync Now</Text>
                </>
              )}
            </TouchableOpacity>
          )}
        </View>
        
        {pendingOperations.length === 0 ? (
          <View style={styles.emptyState}>
            <Icon name="done-all" size={48} color="#10B981" />
            <Text style={styles.emptyStateText}>All synced!</Text>
            <Text style={styles.emptyStateSubtext}>
              No pending operations to sync
            </Text>
          </View>
        ) : (
          <View style={styles.operationsList}>
            {pendingOperations.map(operation => (
              <OperationCard
                key={operation.id}
                operation={operation}
                onRetry={() => handleRetryOperation(operation.id)}
                onRemove={() => handleRemoveOperation(operation.id)}
              />
            ))}
          </View>
        )}
      </View>

      {/* Actions */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Actions</Text>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={handleClearPendingOperations}
          disabled={pendingOperations.length === 0}
        >
          <Icon name="clear-all" size={20} color="#F59E0B" />
          <Text style={styles.actionButtonText}>Clear Pending Operations</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[styles.actionButton, styles.dangerButton]}
          onPress={handleClearOfflineData}
        >
          <Icon name="delete-forever" size={20} color="#EF4444" />
          <Text style={[styles.actionButtonText, styles.dangerButtonText]}>
            Clear All Offline Data
          </Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  header: {
    backgroundColor: 'white',
    padding: 20,
    marginBottom: 16,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  headerText: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#111827',
  },
  headerStatus: {
    fontSize: 16,
    fontWeight: '500',
    marginTop: 4,
  },
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
    paddingHorizontal: 16,
    marginBottom: 12,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 16,
    gap: 12,
  },
  statCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#111827',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 4,
  },
  settingCard: {
    backgroundColor: 'white',
    marginHorizontal: 16,
    borderRadius: 12,
    marginBottom: 8,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  settingInfo: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
  },
  settingDescription: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 4,
  },
  syncButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2563EB',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    gap: 6,
  },
  syncButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  operationsList: {
    paddingHorizontal: 16,
  },
  operationCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  operationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  operationInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  operationDetails: {
    flex: 1,
  },
  operationType: {
    fontSize: 14,
    fontWeight: '600',
    color: '#111827',
  },
  operationTime: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 2,
  },
  operationActions: {
    flexDirection: 'row',
    gap: 8,
  },
  actionButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#F3F4F6',
    justifyContent: 'center',
    alignItems: 'center',
  },
  operationMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 12,
  },
  retryCount: {
    fontSize: 12,
    color: '#6B7280',
  },
  warningBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FEF3C7',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 12,
    gap: 4,
  },
  warningText: {
    fontSize: 10,
    color: '#F59E0B',
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 32,
    paddingHorizontal: 16,
  },
  emptyStateText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#10B981',
    marginTop: 12,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 4,
    textAlign: 'center',
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    marginHorizontal: 16,
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderRadius: 12,
    marginBottom: 8,
    gap: 12,
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#F59E0B',
  },
  dangerButton: {
    // Additional styling for danger actions
  },
  dangerButtonText: {
    color: '#EF4444',
  },
});

export default OfflineScreen;