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
  Dimensions,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useNavigation } from '@react-navigation/native';
import { useAgents } from '../hooks/useAgents';
import { useNetwork } from '../hooks/useNetwork';
import { SyncService } from '../services/SyncService';
import { Logger } from '../utils/Logger';

const { width } = Dimensions.get('window');

interface Agent {
  id: string;
  name: string;
  type: 'copilot' | 'agno' | 'mastra';
  status: 'online' | 'offline' | 'busy';
  description: string;
  lastUsed?: Date;
  capabilities: string[];
  responseTime?: number;
}

const AgentCard: React.FC<{ agent: Agent; onPress: () => void }> = ({ agent, onPress }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return '#10B981';
      case 'busy': return '#F59E0B';
      case 'offline': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'copilot': return 'code';
      case 'agno': return 'description';
      case 'mastra': return 'work';
      default: return 'smart-toy';
    }
  };

  return (
    <TouchableOpacity style={styles.agentCard} onPress={onPress}>
      <View style={styles.cardHeader}>
        <View style={styles.agentInfo}>
          <Icon name={getTypeIcon(agent.type)} size={24} color="#2563EB" />
          <View style={styles.agentDetails}>
            <Text style={styles.agentName}>{agent.name}</Text>
            <Text style={styles.agentType}>{agent.type.toUpperCase()}</Text>
          </View>
        </View>
        <View style={[styles.statusIndicator, { backgroundColor: getStatusColor(agent.status) }]} />
      </View>
      
      <Text style={styles.agentDescription} numberOfLines={2}>
        {agent.description}
      </Text>
      
      <View style={styles.capabilities}>
        {agent.capabilities.slice(0, 3).map((capability, index) => (
          <View key={index} style={styles.capabilityTag}>
            <Text style={styles.capabilityText}>{capability}</Text>
          </View>
        ))}
        {agent.capabilities.length > 3 && (
          <Text style={styles.moreCapabilities}>+{agent.capabilities.length - 3} more</Text>
        )}
      </View>
      
      {agent.responseTime && (
        <View style={styles.performanceInfo}>
          <Icon name="speed" size={14} color="#6B7280" />
          <Text style={styles.responseTime}>{agent.responseTime}ms avg</Text>
        </View>
      )}
    </TouchableOpacity>
  );
};

const AgentDashboard: React.FC = () => {
  const navigation = useNavigation();
  const { agents, loading, error, fetchAgents } = useAgents();
  const { isOnline } = useNetwork();
  const [refreshing, setRefreshing] = useState(false);
  const [pendingOpsCount, setPendingOpsCount] = useState(0);

  useEffect(() => {
    loadPendingOperationsCount();
  }, []);

  const loadPendingOperationsCount = async () => {
    try {
      const count = await SyncService.getPendingOperationsCount();
      setPendingOpsCount(count);
    } catch (error) {
      Logger.error('AgentDashboard', 'Failed to load pending operations count', error);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetchAgents();
      await loadPendingOperationsCount();
      if (isOnline) {
        await SyncService.syncNow();
      }
    } catch (error) {
      Logger.error('AgentDashboard', 'Failed to refresh', error);
    } finally {
      setRefreshing(false);
    }
  };

  const handleAgentPress = (agent: Agent) => {
    navigation.navigate('Chat', { agentId: agent.id, agentName: agent.name });
  };

  const handleSyncPress = async () => {
    if (!isOnline) {
      Alert.alert('Offline', 'Cannot sync while offline. Please check your connection.');
      return;
    }

    try {
      await SyncService.syncNow();
      await loadPendingOperationsCount();
      Alert.alert('Success', 'Sync completed successfully');
    } catch (error) {
      Logger.error('AgentDashboard', 'Sync failed', error);
      Alert.alert('Error', 'Failed to sync data. Please try again.');
    }
  };

  if (loading && !refreshing) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator large color="#2563EB" />
        <Text style={styles.loadingText}>Loading agents...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.errorContainer}>
        <Icon name="error" size={48} color="#EF4444" />
        <Text style={styles.errorText}>Failed to load agents</Text>
        <TouchableOpacity style={styles.retryButton} onPress={fetchAgents}>
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Text style={styles.headerTitle}>UAP Agents</Text>
          <View style={styles.headerActions}>
            {pendingOpsCount > 0 && (
              <TouchableOpacity style={styles.syncButton} onPress={handleSyncPress}>
                <Icon name="sync" size={20} color="#2563EB" />
                <Text style={styles.syncButtonText}>{pendingOpsCount}</Text>
              </TouchableOpacity>
            )}
            <View style={[styles.connectionStatus, { backgroundColor: isOnline ? '#10B981' : '#EF4444' }]}>
              <Icon name={isOnline ? 'wifi' : 'wifi-off'} size={16} color="white" />
            </View>
          </View>
        </View>
      </View>

      {/* Agent List */}
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
        }
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.agentGrid}>
          {agents.map((agent) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              onPress={() => handleAgentPress(agent)}
            />
          ))}
        </View>

        {agents.length === 0 && (
          <View style={styles.emptyState}>
            <Icon name="smart-toy" size={64} color="#D1D5DB" />
            <Text style={styles.emptyStateText}>No agents available</Text>
            <Text style={styles.emptyStateSubtext}>
              Check your connection and try refreshing
            </Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#6B7280',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
    padding: 20,
  },
  errorText: {
    marginTop: 16,
    fontSize: 18,
    fontWeight: '600',
    color: '#EF4444',
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 16,
    paddingHorizontal: 24,
    paddingVertical: 12,
    backgroundColor: '#2563EB',
    borderRadius: 8,
  },
  retryButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  header: {
    backgroundColor: 'white',
    paddingTop: 60,
    paddingBottom: 16,
    paddingHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#111827',
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  syncButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#EBF4FF',
    borderRadius: 20,
    gap: 4,
  },
  syncButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#2563EB',
  },
  connectionStatus: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scrollView: {
    flex: 1,
  },
  agentGrid: {
    padding: 16,
    gap: 16,
  },
  agentCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  agentInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  agentDetails: {
    flex: 1,
  },
  agentName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
  },
  agentType: {
    fontSize: 12,
    fontWeight: '500',
    color: '#6B7280',
    marginTop: 2,
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  agentDescription: {
    fontSize: 14,
    color: '#6B7280',
    lineHeight: 20,
    marginBottom: 12,
  },
  capabilities: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 8,
  },
  capabilityTag: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    backgroundColor: '#F3F4F6',
    borderRadius: 12,
  },
  capabilityText: {
    fontSize: 12,
    color: '#374151',
  },
  moreCapabilities: {
    fontSize: 12,
    color: '#6B7280',
    fontStyle: 'italic',
  },
  performanceInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  responseTime: {
    fontSize: 12,
    color: '#6B7280',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 64,
  },
  emptyStateText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#9CA3AF',
    marginTop: 16,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#D1D5DB',
    marginTop: 8,
    textAlign: 'center',
  },
});

export default AgentDashboard;