// Custom hook for real-time analytics data via WebSocket
import { useEffect, useState, useCallback, useRef } from 'react';

interface MetricUpdate {
  metric_name: string;
  value: number;
  timestamp: number;
}

interface DashboardUpdate {
  system_health: any;
  performance_metrics: any;
  business_metrics: any;
}

interface RealtimeData {
  metrics: Record<string, MetricUpdate>;
  dashboardData: DashboardUpdate | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastUpdate: Date | null;
}

export const useRealtimeAnalytics = (autoConnect: boolean = true) => {
  const [data, setData] = useState<RealtimeData>({
    metrics: {},
    dashboardData: null,
    connectionStatus: 'disconnected',
    lastUpdate: null
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    setData(prev => ({ ...prev, connectionStatus: 'connecting' }));

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/analytics`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Analytics WebSocket connected');
        setData(prev => ({ 
          ...prev, 
          connectionStatus: 'connected',
          lastUpdate: new Date()
        }));
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          const now = new Date();

          switch (message.type) {
            case 'metric_update':
              setData(prev => ({
                ...prev,
                metrics: {
                  ...prev.metrics,
                  [message.payload.metric_name]: {
                    metric_name: message.payload.metric_name,
                    value: message.payload.value,
                    timestamp: message.payload.timestamp
                  }
                },
                lastUpdate: now
              }));
              break;

            case 'dashboard_update':
              setData(prev => ({
                ...prev,
                dashboardData: message.payload,
                lastUpdate: now
              }));
              break;

            case 'connection_established':
              console.log('Analytics WebSocket established:', message.message);
              break;

            default:
              console.log('Unknown analytics message type:', message.type);
          }
        } catch (error) {
          console.error('Failed to parse analytics WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('Analytics WebSocket closed:', event.code, event.reason);
        setData(prev => ({ ...prev, connectionStatus: 'disconnected' }));

        // Attempt to reconnect if not a clean close and we haven't exceeded max attempts
        if (!event.wasClean && reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000); // Exponential backoff, max 30s
          reconnectAttempts.current++;
          
          console.log(`Attempting to reconnect analytics WebSocket in ${delay}ms (attempt ${reconnectAttempts.current})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };

      ws.onerror = (error) => {
        console.error('Analytics WebSocket error:', error);
        setData(prev => ({ ...prev, connectionStatus: 'error' }));
      };

    } catch (error) {
      console.error('Failed to create analytics WebSocket:', error);
      setData(prev => ({ ...prev, connectionStatus: 'error' }));
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'User initiated disconnect');
      wsRef.current = null;
    }

    setData(prev => ({ ...prev, connectionStatus: 'disconnected' }));
  }, []);

  const getMetricValue = useCallback((metricName: string): number | null => {
    const metric = data.metrics[metricName];
    return metric ? metric.value : null;
  }, [data.metrics]);

  const getMetricTimestamp = useCallback((metricName: string): Date | null => {
    const metric = data.metrics[metricName];
    return metric ? new Date(metric.timestamp * 1000) : null;
  }, [data.metrics]);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    data,
    connect,
    disconnect,
    getMetricValue,
    getMetricTimestamp,
    isConnected: data.connectionStatus === 'connected',
    isConnecting: data.connectionStatus === 'connecting',
    hasError: data.connectionStatus === 'error'
  };
};