import React, { useEffect, useState } from 'react';
import {
  View,
  StyleSheet,
  FlatList,
  RefreshControl,
  Alert,
} from 'react-native';
import {
  Surface,
  Text,
  Card,
  Button,
  Chip,
  ActivityIndicator,
  Snackbar,
} from 'react-native-paper';
import { useSelector, useDispatch } from 'react-redux';
import NetInfo from '@react-native-community/netinfo';
import Icon from 'react-native-vector-icons/MaterialIcons';

import { RootState, AppDispatch } from '../store';
import { fetchAgents, selectAgent } from '../store/slices/agentsSlice';
import { AgentCard } from '../components/AgentCard';
import { OfflineIndicator } from '../components/OfflineIndicator';
import { Agent } from '../types/agent';
import { SyncService } from '../services/SyncService';

export const AgentScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { agents, loading, error, selectedAgent } = useSelector(
    (state: RootState) => state.agents
  );
  const { isOnline } = useSelector((state: RootState) => state.offline);
  
  const [refreshing, setRefreshing] = useState(false);
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  useEffect(() => {
    loadAgents();
  }, []);

  const loadAgents = async () => {
    try {
      await dispatch(fetchAgents()).unwrap();
    } catch (err) {
      showSnackbar('Failed to load agents. Using cached data.');
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadAgents();
    setRefreshing(false);
  };

  const handleAgentSelect = (agent: Agent) => {
    dispatch(selectAgent(agent.id));
    // Navigate to chat screen would go here
  };

  const handleSyncNow = async () => {
    if (!isOnline) {
      showSnackbar('Cannot sync while offline');
      return;
    }
    
    try {
      await SyncService.syncNow();
      showSnackbar('Sync completed successfully');
    } catch (error) {
      showSnackbar('Sync failed. Will retry automatically.');
    }
  };

  const showSnackbar = (message: string) => {
    setSnackbarMessage(message);
    setSnackbarVisible(true);
  };

  const renderAgent = ({ item }: { item: Agent }) => (
    <AgentCard
      agent={item}
      onSelect={handleAgentSelect}
      isSelected={item.id === selectedAgent}
      offline={!isOnline}
    />
  );

  const renderHeader = () => (
    <View style={styles.header}>
      <Text variant="headlineMedium" style={styles.title}>
        AI Agents
      </Text>
      <View style={styles.headerActions}>
        <OfflineIndicator />
        {!isOnline && (
          <Button
            mode="outlined"
            icon="sync"
            onPress={handleSyncNow}
            disabled={!isOnline}
            style={styles.syncButton}
          >
            Sync
          </Button>
        )}
      </View>
    </View>
  );

  const renderEmptyState = () => (
    <Surface style={styles.emptyState}>
      <Icon name="smart-toy" size={64} color="#666" />
      <Text variant="headlineSmall" style={styles.emptyTitle}>
        No Agents Available
      </Text>
      <Text variant="bodyMedium" style={styles.emptySubtitle}>
        {isOnline
          ? 'Pull down to refresh and load agents'
          : 'Connect to internet to load agents'}
      </Text>
    </Surface>
  );

  if (loading && agents.length === 0) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" />
        <Text style={styles.loadingText}>Loading agents...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {renderHeader()}
      
      <FlatList
        data={agents}
        renderItem={renderAgent}
        keyExtractor={(item) => item.id}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={renderEmptyState}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.listContent}
      />

      <Snackbar
        visible={snackbarVisible}
        onDismiss={() => setSnackbarVisible(false)}
        duration={3000}
      >
        {snackbarMessage}
      </Snackbar>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 20,
    backgroundColor: 'white',
    elevation: 2,
  },
  title: {
    fontWeight: 'bold',
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  syncButton: {
    marginLeft: 8,
  },
  listContent: {
    padding: 16,
    paddingBottom: 100,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    marginTop: 60,
  },
  emptyTitle: {
    marginTop: 16,
    fontWeight: 'bold',
  },
  emptySubtitle: {
    marginTop: 8,
    textAlign: 'center',
    color: '#666',
  },
});