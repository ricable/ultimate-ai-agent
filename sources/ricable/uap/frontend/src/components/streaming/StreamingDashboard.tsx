// frontend/src/components/streaming/StreamingDashboard.tsx
/**
 * Streaming Dashboard Component
 * Real-time dashboard for monitoring streaming, edge AI, and ultra-low latency processing.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/Card';
import RealTimeMetrics from './RealTimeMetrics';
import EdgeDeploymentPanel from './EdgeDeploymentPanel';
import AnomalyDetectionPanel from './AnomalyDetectionPanel';
import OptimizationPanel from './OptimizationPanel';
import { AlertTriangle, Activity, Zap, Cloud, Cpu, BarChart3 } from 'lucide-react';

interface StreamingMetrics {
  events_processed: number;
  anomalies_detected: number;
  edge_optimizations: number;
  avg_processing_latency_ms: number;
  throughput_events_per_sec: number;
  buffer_utilization: number;
  memory_usage_mb: number;
  cpu_usage_percent: number;
}

interface ComponentStatus {
  stream_processor: boolean;
  low_latency_processor: boolean;
  edge_optimizer: boolean;
  anomaly_detection: boolean;
  edge_inference: boolean;
  cloud_sync: boolean;
}

interface StreamingAgentStatus {
  agent_id: string;
  is_initialized: boolean;
  is_running: boolean;
  metrics: StreamingMetrics;
  components: ComponentStatus;
}

interface StreamingDashboardProps {
  agentId?: string;
  refreshInterval?: number;
  enableRealTimeUpdates?: boolean;
}

const StreamingDashboard: React.FC<StreamingDashboardProps> = ({
  agentId = 'agent-37-streaming',
  refreshInterval = 1000,
  enableRealTimeUpdates = true
}) => {
  const [agentStatus, setAgentStatus] = useState<StreamingAgentStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [selectedTab, setSelectedTab] = useState<string>('overview');

  // Fetch streaming agent status
  const fetchAgentStatus = useCallback(async () => {
    try {
      const response = await fetch(`/api/agents/streaming/status`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setAgentStatus(data);
      setError(null);
      setIsConnected(true);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch agent status');
      setIsConnected(false);
    } finally {
      setLoading(false);
    }
  }, []);

  // Real-time updates
  useEffect(() => {
    if (!enableRealTimeUpdates) return;

    fetchAgentStatus();
    const interval = setInterval(fetchAgentStatus, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchAgentStatus, refreshInterval, enableRealTimeUpdates]);

  // Process stream event
  const processStreamEvent = async (eventType: string, data: any, priority: string = 'normal') => {
    try {
      const response = await fetch('/api/agents/streaming/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          event_type: eventType,
          data: data,
          priority: priority
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to process stream event');
      }
      
      const result = await response.json();
      console.log('Stream event processed:', result);
      
      // Refresh status after processing
      await fetchAgentStatus();
      
      return result;
    } catch (err) {
      console.error('Error processing stream event:', err);
      throw err;
    }
  };

  // Load edge model
  const loadEdgeModel = async (modelId: string, modelPath: string, options: any = {}) => {
    try {
      const response = await fetch('/api/agents/streaming/models', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: modelId,
          model_path: modelPath,
          ...options
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to load edge model');
      }
      
      const result = await response.json();
      console.log('Edge model loaded:', result);
      
      // Refresh status after loading model
      await fetchAgentStatus();
      
      return result;
    } catch (err) {
      console.error('Error loading edge model:', err);
      throw err;
    }
  };

  // Get status indicator color
  const getStatusColor = (isActive: boolean, hasError: boolean = false) => {
    if (hasError) return 'text-red-500';
    return isActive ? 'text-green-500' : 'text-gray-400';
  };

  // Format latency with appropriate units
  const formatLatency = (latencyMs: number) => {
    if (latencyMs < 1) {
      return `${(latencyMs * 1000).toFixed(1)}μs`;
    } else if (latencyMs < 1000) {
      return `${latencyMs.toFixed(2)}ms`;
    } else {
      return `${(latencyMs / 1000).toFixed(2)}s`;
    }
  };

  // Format throughput
  const formatThroughput = (eventsPerSec: number) => {
    if (eventsPerSec < 1000) {
      return `${eventsPerSec.toFixed(1)} events/s`;
    } else if (eventsPerSec < 1000000) {
      return `${(eventsPerSec / 1000).toFixed(1)}K events/s`;
    } else {
      return `${(eventsPerSec / 1000000).toFixed(1)}M events/s`;
    }
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (error && !agentStatus) {
    return (
      <div className="p-6">
        <Card className="border-red-200">
          <CardHeader>
            <CardTitle className="text-red-600 flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              Streaming Agent Error
            </CardTitle>
            <CardDescription className="text-red-500">
              {error}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <button
              onClick={fetchAgentStatus}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Retry Connection
            </button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Real-Time Stream Processing & Edge AI
          </h1>
          <p className="text-gray-600 mt-1">
            Agent 37 - Ultra-low latency streaming with edge optimization
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className={`flex items-center space-x-2 ${
            isConnected ? 'text-green-600' : 'text-red-600'
          }`}>
            <div className={`w-3 h-3 rounded-full ${
              isConnected ? 'bg-green-500' : 'bg-red-500'
            } animate-pulse`}></div>
            <span className="text-sm font-medium">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          {lastUpdate && (
            <span className="text-sm text-gray-500">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Status Overview */}
      {agentStatus && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Processing Latency */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                Avg Processing Latency
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-blue-600">
                  {formatLatency(agentStatus.metrics.avg_processing_latency_ms)}
                </span>
                <Zap className="h-8 w-8 text-blue-500" />
              </div>
              <p className="text-sm text-gray-500 mt-1">
                Target: &lt;1ms for critical events
              </p>
            </CardContent>
          </Card>

          {/* Throughput */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                Event Throughput
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-green-600">
                  {formatThroughput(agentStatus.metrics.throughput_events_per_sec)}
                </span>
                <Activity className="h-8 w-8 text-green-500" />
              </div>
              <p className="text-sm text-gray-500 mt-1">
                {agentStatus.metrics.events_processed.toLocaleString()} total processed
              </p>
            </CardContent>
          </Card>

          {/* Anomalies Detected */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                Anomalies Detected
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className={`text-2xl font-bold ${
                  agentStatus.metrics.anomalies_detected > 0 ? 'text-red-600' : 'text-gray-400'
                }`}>
                  {agentStatus.metrics.anomalies_detected}
                </span>
                <AlertTriangle className={`h-8 w-8 ${
                  agentStatus.metrics.anomalies_detected > 0 ? 'text-red-500' : 'text-gray-400'
                }`} />
              </div>
              <p className="text-sm text-gray-500 mt-1">
                Real-time detection active
              </p>
            </CardContent>
          </Card>

          {/* Edge Optimizations */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">
                Edge Optimizations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-purple-600">
                  {agentStatus.metrics.edge_optimizations}
                </span>
                <Cloud className="h-8 w-8 text-purple-500" />
              </div>
              <p className="text-sm text-gray-500 mt-1">
                Edge-cloud placement decisions
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Component Status */}
      {agentStatus && (
        <Card>
          <CardHeader>
            <CardTitle>Component Status</CardTitle>
            <CardDescription>
              Status of streaming agent components
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  agentStatus.components.stream_processor ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm">Stream Processor</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  agentStatus.components.low_latency_processor ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm">Ultra-Low Latency</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  agentStatus.components.edge_optimizer ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm">Edge Optimizer</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  agentStatus.components.anomaly_detection ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm">Anomaly Detection</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  agentStatus.components.edge_inference ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm">Edge Inference</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  agentStatus.components.cloud_sync ? 'bg-green-500' : 'bg-gray-400'
                }`}></div>
                <span className="text-sm">Cloud Sync</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'realtime', label: 'Real-Time Metrics', icon: Activity },
            { id: 'edge', label: 'Edge Deployment', icon: Cloud },
            { id: 'anomaly', label: 'Anomaly Detection', icon: AlertTriangle },
            { id: 'optimization', label: 'Optimization', icon: Zap }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id)}
              className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                selectedTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {selectedTab === 'overview' && agentStatus && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* System Resource Usage */}
            <Card>
              <CardHeader>
                <CardTitle>System Resources</CardTitle>
                <CardDescription>Current resource utilization</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>Buffer Utilization</span>
                      <span>{agentStatus.metrics.buffer_utilization.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${agentStatus.metrics.buffer_utilization}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>Memory Usage</span>
                      <span>{agentStatus.metrics.memory_usage_mb.toFixed(1)} MB</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-600 h-2 rounded-full" 
                        style={{ width: `${Math.min(agentStatus.metrics.memory_usage_mb / 1024 * 100, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>CPU Usage</span>
                      <span>{agentStatus.metrics.cpu_usage_percent.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-orange-600 h-2 rounded-full" 
                        style={{ width: `${agentStatus.metrics.cpu_usage_percent}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
                <CardDescription>Common streaming operations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <button
                    onClick={() => processStreamEvent('data', { test: 'value' }, 'normal')}
                    className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-left"
                  >
                    Test Stream Event
                  </button>
                  
                  <button
                    onClick={() => processStreamEvent('data', { test: 'critical' }, 'critical')}
                    className="w-full px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 text-left"
                  >
                    Test Critical Event
                  </button>
                  
                  <button
                    onClick={() => loadEdgeModel('test-model', '/models/test.onnx')}
                    className="w-full px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 text-left"
                  >
                    Load Test Model
                  </button>
                  
                  <button
                    onClick={fetchAgentStatus}
                    className="w-full px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 text-left"
                  >
                    Refresh Status
                  </button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {selectedTab === 'realtime' && (
          <RealTimeMetrics 
            agentStatus={agentStatus} 
            refreshInterval={refreshInterval}
          />
        )}

        {selectedTab === 'edge' && (
          <EdgeDeploymentPanel 
            agentStatus={agentStatus}
            onLoadModel={loadEdgeModel}
          />
        )}

        {selectedTab === 'anomaly' && (
          <AnomalyDetectionPanel 
            agentStatus={agentStatus}
            onProcessEvent={processStreamEvent}
          />
        )}

        {selectedTab === 'optimization' && (
          <OptimizationPanel 
            agentStatus={agentStatus}
            onOptimize={fetchAgentStatus}
          />
        )}
      </div>
    </div>
  );
};

export default StreamingDashboard;
