// Real-time Metrics Display Component
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Activity, Zap, TrendingUp, TrendingDown, AlertCircle, Wifi, WifiOff } from 'lucide-react';
import { Card } from '../ui/Card';
import { SystemMetricChart, ResponseTimeChart, ThroughputChart } from './MetricsChart';

interface RealTimeMetric {
  name: string;
  value: number;
  timestamp: number;
  unit: string;
  category: string;
  labels: Record<string, string>;
}

interface MetricUpdate {
  type: 'metric_update';
  payload: {
    metric_name: string;
    value: number;
    timestamp: number;
    unit?: string;
    category?: string;
    labels?: Record<string, string>;
  };
}

interface RealTimeMetricsProps {
  className?: string;
  autoConnect?: boolean;
  reconnectInterval?: number;
  maxDataPoints?: number;
  updateInterval?: number;
}

interface ConnectionState {
  status: 'connected' | 'disconnected' | 'connecting' | 'error';
  lastUpdate: Date | null;
  reconnectAttempts: number;
  error: string | null;
}

interface MetricStats {
  current: number;
  previous: number;
  min: number;
  max: number;
  average: number;
  trend: 'up' | 'down' | 'stable';
  change: number;
}

const METRIC_CATEGORIES = {
  system: { icon: Activity, color: 'blue' },
  application: { icon: Zap, color: 'purple' },
  business: { icon: TrendingUp, color: 'green' },
  users: { icon: Activity, color: 'cyan' },
  custom: { icon: Activity, color: 'gray' }
};

const WEBSOCKET_URL = 'ws://localhost:8000/ws/analytics/metrics';

export const RealTimeMetrics: React.FC<RealTimeMetricsProps> = ({
  className = '',
  autoConnect = true,
  reconnectInterval = 5000,
  maxDataPoints = 100,
  updateInterval = 1000
}) => {
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    status: 'disconnected',
    lastUpdate: null,
    reconnectAttempts: 0,
    error: null
  });
  
  const [metrics, setMetrics] = useState<Record<string, RealTimeMetric[]>>({});
  const [metricStats, setMetricStats] = useState<Record<string, MetricStats>>({});
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['system_cpu_percent', 'system_memory_percent', 'agent_response_time']);
  const [isExpanded, setIsExpanded] = useState(false);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const statsUpdateRef = useRef<NodeJS.Timeout | null>(null);

  // Calculate statistics for a metric
  const calculateStats = useCallback((data: RealTimeMetric[]): MetricStats => {
    if (data.length === 0) {
      return {
        current: 0,
        previous: 0,
        min: 0,
        max: 0,
        average: 0,
        trend: 'stable',
        change: 0
      };
    }

    const values = data.map(point => point.value);
    const current = values[values.length - 1];
    const previous = values.length > 1 ? values[values.length - 2] : current;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const average = values.reduce((sum, val) => sum + val, 0) / values.length;
    
    const change = ((current - previous) / Math.max(previous, 0.001)) * 100;
    const trend = Math.abs(change) < 1 ? 'stable' : change > 0 ? 'up' : 'down';

    return {
      current,
      previous,
      min,
      max,
      average,
      trend,
      change
    };
  }, []);

  // Update metric statistics
  const updateMetricStats = useCallback(() => {
    const newStats: Record<string, MetricStats> = {};
    
    Object.entries(metrics).forEach(([metricName, data]) => {
      newStats[metricName] = calculateStats(data);
    });
    
    setMetricStats(newStats);
  }, [metrics, calculateStats]);

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    setConnectionState(prev => ({ 
      ...prev, 
      status: 'connecting', 
      error: null 
    }));

    try {
      const token = localStorage.getItem('token');
      const ws = new WebSocket(`${WEBSOCKET_URL}?token=${token}`);
      
      ws.onopen = () => {
        console.log('Real-time metrics WebSocket connected');
        setConnectionState(prev => ({
          ...prev,
          status: 'connected',
          reconnectAttempts: 0,
          error: null
        }));
        
        // Clear any pending reconnect
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const update: MetricUpdate = JSON.parse(event.data);
          
          if (update.type === 'metric_update') {
            const { metric_name, value, timestamp, unit = '', category = 'custom', labels = {} } = update.payload;
            
            const newMetric: RealTimeMetric = {
              name: metric_name,
              value,
              timestamp: timestamp * 1000, // Convert to milliseconds
              unit,
              category,
              labels
            };

            setMetrics(prev => {
              const updated = { ...prev };
              if (!updated[metric_name]) {
                updated[metric_name] = [];
              }
              
              // Add new metric and keep only the latest points
              updated[metric_name] = [...updated[metric_name], newMetric].slice(-maxDataPoints);
              
              return updated;
            });

            setConnectionState(prev => ({
              ...prev,
              lastUpdate: new Date()
            }));
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('Real-time metrics WebSocket disconnected:', event.code, event.reason);
        setConnectionState(prev => ({
          ...prev,
          status: 'disconnected'
        }));
        
        // Attempt to reconnect if it wasn't a manual close
        if (event.code !== 1000 && autoConnect) {
          scheduleReconnect();
        }
      };

      ws.onerror = (error) => {
        console.error('Real-time metrics WebSocket error:', error);
        setConnectionState(prev => ({
          ...prev,
          status: 'error',
          error: 'WebSocket connection failed'
        }));
      };

      wsRef.current = ws;
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionState(prev => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Connection failed'
      }));
    }
  }, [autoConnect, maxDataPoints]);

  // Schedule reconnection
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    setConnectionState(prev => ({
      ...prev,
      reconnectAttempts: prev.reconnectAttempts + 1
    }));
    
    const delay = Math.min(reconnectInterval * Math.pow(1.5, connectionState.reconnectAttempts), 30000);
    
    reconnectTimeoutRef.current = setTimeout(() => {
      connectWebSocket();
    }, delay);
  }, [connectWebSocket, reconnectInterval, connectionState.reconnectAttempts]);

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setConnectionState(prev => ({
      ...prev,
      status: 'disconnected',
      reconnectAttempts: 0
    }));
  }, []);

  // Initialize connection
  useEffect(() => {
    if (autoConnect) {
      connectWebSocket();
    }
    
    return () => {
      disconnectWebSocket();
    };
  }, [autoConnect, connectWebSocket, disconnectWebSocket]);

  // Update statistics periodically
  useEffect(() => {
    const updateStats = () => {
      updateMetricStats();
      statsUpdateRef.current = setTimeout(updateStats, updateInterval);
    };
    
    updateStats();
    
    return () => {
      if (statsUpdateRef.current) {
        clearTimeout(statsUpdateRef.current);
      }
    };
  }, [updateMetricStats, updateInterval]);

  // Format metric value
  const formatValue = (value: number, unit: string): string => {
    if (unit === '%') {
      return `${value.toFixed(1)}%`;
    }
    if (unit === 'ms') {
      return `${value.toFixed(0)}ms`;
    }
    if (unit === 'bytes') {
      if (value >= 1024 * 1024 * 1024) {
        return `${(value / (1024 * 1024 * 1024)).toFixed(1)}GB`;
      }
      if (value >= 1024 * 1024) {
        return `${(value / (1024 * 1024)).toFixed(1)}MB`;
      }
      if (value >= 1024) {
        return `${(value / 1024).toFixed(1)}KB`;
      }
      return `${value.toFixed(0)}B`;
    }
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    }
    if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toFixed(2);
  };

  // Get connection status indicator
  const getConnectionIndicator = () => {
    const { status, lastUpdate, error } = connectionState;
    
    const statusConfig = {
      connected: {
        icon: Wifi,
        color: 'text-green-500',
        bgColor: 'bg-green-100',
        text: 'Connected'
      },
      connecting: {
        icon: Wifi,
        color: 'text-yellow-500',
        bgColor: 'bg-yellow-100',
        text: 'Connecting...'
      },
      disconnected: {
        icon: WifiOff,
        color: 'text-gray-500',
        bgColor: 'bg-gray-100',
        text: 'Disconnected'
      },
      error: {
        icon: AlertCircle,
        color: 'text-red-500',
        bgColor: 'bg-red-100',
        text: 'Error'
      }
    };
    
    const config = statusConfig[status];
    const Icon = config.icon;
    
    return (
      <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${config.bgColor}`}>
        <Icon className={`h-4 w-4 ${config.color}`} />
        <span className={`text-sm font-medium ${config.color}`}>
          {config.text}
        </span>
        {lastUpdate && status === 'connected' && (
          <span className="text-xs text-gray-500">
            {lastUpdate.toLocaleTimeString()}
          </span>
        )}
      </div>
    );
  };

  // Render metric card
  const renderMetricCard = (metricName: string) => {
    const data = metrics[metricName] || [];
    const stats = metricStats[metricName];
    const latestMetric = data[data.length - 1];
    
    if (!latestMetric || !stats) {
      return (
        <Card key={metricName} className="p-4">
          <div className="text-center text-gray-500">
            <div className="w-8 h-8 mx-auto mb-2 bg-gray-200 rounded-full flex items-center justify-center">
              <Activity className="h-4 w-4" />
            </div>
            <p className="text-sm">No data for {metricName}</p>
          </div>
        </Card>
      );
    }
    
    const categoryConfig = METRIC_CATEGORIES[latestMetric.category as keyof typeof METRIC_CATEGORIES] || METRIC_CATEGORIES.custom;
    const Icon = categoryConfig.icon;
    
    return (
      <Card key={metricName} className="p-4 hover:shadow-md transition-shadow">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Icon className={`h-5 w-5 text-${categoryConfig.color}-500`} />
            <h3 className="text-sm font-medium text-gray-700 truncate">
              {metricName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </h3>
          </div>
          
          {/* Trend indicator */}
          <div className="flex items-center space-x-1">
            {stats.trend === 'up' ? (
              <TrendingUp className="h-4 w-4 text-green-500" />
            ) : stats.trend === 'down' ? (
              <TrendingDown className="h-4 w-4 text-red-500" />
            ) : (
              <div className="h-4 w-4 bg-gray-400 rounded-full" />
            )}
            {stats.trend !== 'stable' && (
              <span className={`text-xs font-medium ${
                stats.trend === 'up' ? 'text-green-600' : 'text-red-600'
              }`}>
                {Math.abs(stats.change).toFixed(1)}%
              </span>
            )}
          </div>
        </div>
        
        {/* Current value */}
        <div className="mb-2">
          <span className="text-2xl font-bold text-gray-900">
            {formatValue(stats.current, latestMetric.unit)}
          </span>
        </div>
        
        {/* Statistics */}
        <div className="grid grid-cols-2 gap-2 text-xs text-gray-500">
          <div>
            <span className="block">Min: {formatValue(stats.min, latestMetric.unit)}</span>
            <span className="block">Max: {formatValue(stats.max, latestMetric.unit)}</span>
          </div>
          <div>
            <span className="block">Avg: {formatValue(stats.average, latestMetric.unit)}</span>
            <span className="block">Prev: {formatValue(stats.previous, latestMetric.unit)}</span>
          </div>
        </div>
        
        {/* Mini chart */}
        <div className="mt-3 h-12">
          {data.length > 1 && (
            <div className="flex items-end space-x-px h-full">
              {data.slice(-20).map((point, index) => {
                const height = stats.max > stats.min ? 
                  ((point.value - stats.min) / (stats.max - stats.min)) * 100 : 
                  50;
                
                return (
                  <div
                    key={index}
                    className={`flex-1 bg-${categoryConfig.color}-200 rounded-sm transition-all duration-200`}
                    style={{ height: `${Math.max(height, 2)}%` }}
                    title={`${formatValue(point.value, latestMetric.unit)} at ${new Date(point.timestamp).toLocaleTimeString()}`}
                  />
                );
              })}
            </div>
          )}
        </div>
      </Card>
    );
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <h2 className="text-xl font-semibold text-gray-900">Real-time Metrics</h2>
          {getConnectionIndicator()}
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={connectionState.status === 'connected' ? disconnectWebSocket : connectWebSocket}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              connectionState.status === 'connected'
                ? 'bg-red-100 text-red-700 hover:bg-red-200'
                : 'bg-green-100 text-green-700 hover:bg-green-200'
            }`}
          >
            {connectionState.status === 'connected' ? 'Disconnect' : 'Connect'}
          </button>
          
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="px-3 py-1 bg-gray-100 text-gray-700 rounded text-sm font-medium hover:bg-gray-200 transition-colors"
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
      </div>

      {/* Connection error */}
      {connectionState.error && (
        <Card className="p-4 border-red-200 bg-red-50">
          <div className="flex items-center space-x-2 text-red-700">
            <AlertCircle className="h-5 w-5" />
            <span className="font-medium">Connection Error</span>
          </div>
          <p className="text-sm text-red-600 mt-1">{connectionState.error}</p>
          {connectionState.reconnectAttempts > 0 && (
            <p className="text-xs text-red-500 mt-1">
              Reconnect attempts: {connectionState.reconnectAttempts}
            </p>
          )}
        </Card>
      )}

      {/* Metrics grid */}
      {Object.keys(metrics).length > 0 ? (
        <div className={`grid gap-4 ${
          isExpanded 
            ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'
            : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
        }`}>
          {selectedMetrics
            .filter(metricName => metrics[metricName])
            .map(renderMetricCard)
          }
          
          {/* Show other metrics if expanded */}
          {isExpanded && Object.keys(metrics)
            .filter(metricName => !selectedMetrics.includes(metricName))
            .map(renderMetricCard)
          }
        </div>
      ) : (
        <Card className="p-8">
          <div className="text-center text-gray-500">
            <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-medium mb-2">No Metrics Available</h3>
            <p className="text-sm">
              {connectionState.status === 'connected'
                ? 'Waiting for real-time metric updates...'
                : 'Connect to start receiving real-time metrics'
              }
            </p>
          </div>
        </Card>
      )}

      {/* Detailed charts for selected metrics */}
      {isExpanded && selectedMetrics.some(name => metrics[name]) && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-900">Detailed Charts</h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {selectedMetrics
              .filter(metricName => metrics[metricName] && metrics[metricName].length > 1)
              .map(metricName => {
                const data = metrics[metricName].map(point => ({
                  timestamp: point.timestamp,
                  value: point.value
                }));
                
                const latestMetric = metrics[metricName][metrics[metricName].length - 1];
                
                if (metricName.includes('cpu') || metricName.includes('memory') || metricName.includes('disk')) {
                  return (
                    <SystemMetricChart
                      key={metricName}
                      title={metricName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      data={data}
                      isRealTime
                      enableExport
                    />
                  );
                }
                
                if (metricName.includes('response_time')) {
                  return (
                    <ResponseTimeChart
                      key={metricName}
                      title="Response Time"
                      data={data}
                      isRealTime
                      enableExport
                    />
                  );
                }
                
                if (metricName.includes('request') || metricName.includes('throughput')) {
                  return (
                    <ThroughputChart
                      key={metricName}
                      title={metricName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      data={data}
                      isRealTime
                      enableExport
                    />
                  );
                }
                
                // Generic chart for other metrics
                return (
                  <div key={metricName}>
                    {/* Generic chart would go here */}
                  </div>
                );
              })
            }
          </div>
        </div>
      )}
    </div>
  );
};
