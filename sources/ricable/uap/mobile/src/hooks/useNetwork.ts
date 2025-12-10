import { useState, useEffect, useContext, createContext, ReactNode } from 'react';
import NetInfo, { NetInfoState } from '@react-native-community/netinfo';
import { Logger } from '../utils/Logger';

interface NetworkContextType {
  isOnline: boolean;
  connectionType: string | null;
  isInternetReachable: boolean | null;
  networkStrength: number | null;
  connectionQuality: 'poor' | 'fair' | 'good' | 'excellent' | 'unknown';
  lastConnectedAt: Date | null;
  reconnectAttempts: number;
}

const NetworkContext = createContext<NetworkContextType | undefined>(undefined);

export const NetworkProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [networkState, setNetworkState] = useState<NetworkContextType>({
    isOnline: false,
    connectionType: null,
    isInternetReachable: null,
    networkStrength: null,
    connectionQuality: 'unknown',
    lastConnectedAt: null,
    reconnectAttempts: 0,
  });

  useEffect(() => {
    // Subscribe to network state changes
    const unsubscribe = NetInfo.addEventListener((state: NetInfoState) => {
      const isConnected = state.isConnected ?? false;
      const wasOnline = networkState.isOnline;
      
      // Calculate connection quality based on connection type and details
      const quality = calculateConnectionQuality(state);
      
      // Update network state
      setNetworkState(prev => ({
        ...prev,
        isOnline: isConnected,
        connectionType: state.type,
        isInternetReachable: state.isInternetReachable,
        networkStrength: getNetworkStrength(state),
        connectionQuality: quality,
        lastConnectedAt: isConnected ? new Date() : prev.lastConnectedAt,
        reconnectAttempts: isConnected ? 0 : prev.reconnectAttempts,
      }));
      
      // Log network state changes
      if (isConnected && !wasOnline) {
        Logger.info('Network', 'Connection restored', {
          type: state.type,
          quality,
          isInternetReachable: state.isInternetReachable,
        });
      } else if (!isConnected && wasOnline) {
        Logger.warn('Network', 'Connection lost', {
          type: state.type,
          previousQuality: prev.connectionQuality,
        });
      }
    });

    // Initial network state check
    NetInfo.fetch().then((state: NetInfoState) => {
      const isConnected = state.isConnected ?? false;
      const quality = calculateConnectionQuality(state);
      
      setNetworkState({
        isOnline: isConnected,
        connectionType: state.type,
        isInternetReachable: state.isInternetReachable,
        networkStrength: getNetworkStrength(state),
        connectionQuality: quality,
        lastConnectedAt: isConnected ? new Date() : null,
        reconnectAttempts: 0,
      });
      
      Logger.info('Network', 'Initial network state', {
        isConnected,
        type: state.type,
        quality,
        isInternetReachable: state.isInternetReachable,
      });
    });

    return unsubscribe;
  }, []);

  // Periodic connection quality monitoring
  useEffect(() => {
    if (!networkState.isOnline) return;

    const interval = setInterval(async () => {
      try {
        const state = await NetInfo.fetch();
        const quality = calculateConnectionQuality(state);
        
        if (quality !== networkState.connectionQuality) {
          setNetworkState(prev => ({
            ...prev,
            connectionQuality: quality,
            networkStrength: getNetworkStrength(state),
          }));
          
          Logger.info('Network', 'Connection quality changed', {
            previousQuality: networkState.connectionQuality,
            newQuality: quality,
          });
        }
      } catch (error) {
        Logger.error('Network', 'Failed to check connection quality', error);
      }
    }, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, [networkState.isOnline, networkState.connectionQuality]);

  return (
    <NetworkContext.Provider value={networkState}>
      {children}
    </NetworkContext.Provider>
  );
};

export const useNetwork = (): NetworkContextType => {
  const context = useContext(NetworkContext);
  if (context === undefined) {
    throw new Error('useNetwork must be used within a NetworkProvider');
  }
  return context;
};

// Helper functions

function calculateConnectionQuality(state: NetInfoState): 'poor' | 'fair' | 'good' | 'excellent' | 'unknown' {
  if (!state.isConnected) {
    return 'unknown';
  }

  switch (state.type) {
    case 'wifi':
      if (state.details && 'strength' in state.details) {
        const strength = state.details.strength as number;
        if (strength >= 80) return 'excellent';
        if (strength >= 60) return 'good';
        if (strength >= 40) return 'fair';
        return 'poor';
      }
      return 'good'; // Default for WiFi without strength info
    
    case 'cellular':
      if (state.details && 'cellularGeneration' in state.details) {
        const generation = state.details.cellularGeneration;
        switch (generation) {
          case '5g':
            return 'excellent';
          case '4g':
            return 'good';
          case '3g':
            return 'fair';
          case '2g':
            return 'poor';
          default:
            return 'unknown';
        }
      }
      return 'fair'; // Default for cellular without generation info
    
    case 'ethernet':
      return 'excellent';
    
    case 'bluetooth':
      return 'poor';
    
    default:
      return 'unknown';
  }
}

function getNetworkStrength(state: NetInfoState): number | null {
  if (!state.isConnected || !state.details) {
    return null;
  }

  if (state.type === 'wifi' && 'strength' in state.details) {
    return state.details.strength as number;
  }

  if (state.type === 'cellular' && 'strength' in state.details) {
    return state.details.strength as number;
  }

  return null;
}

// Hook for checking specific network conditions
export const useNetworkConditions = () => {
  const network = useNetwork();
  
  const isGoodConnection = network.isOnline && 
    ['good', 'excellent'].includes(network.connectionQuality);
  
  const canSync = network.isOnline && network.isInternetReachable !== false;
  
  const shouldReduceQuality = network.connectionQuality === 'poor' || 
    network.connectionType === 'cellular';
  
  const isSlowConnection = network.connectionQuality === 'poor' || 
    (network.connectionType === 'cellular' && network.connectionQuality !== 'excellent');
  
  return {
    isGoodConnection,
    canSync,
    shouldReduceQuality,
    isSlowConnection,
    ...network,
  };
};