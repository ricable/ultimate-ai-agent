// frontend/src/components/streaming/RealTimeMetrics.tsx
/**
 * Real-Time Metrics Component
 * Displays live streaming metrics with charts and performance indicators.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/Card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { Activity, Zap, AlertTriangle, TrendingUp, Clock, Cpu } from 'lucide-react';

interface MetricsDataPoint {
  timestamp: string;
  latency: number;
  throughput: number;
  anomalies: number;
  cpuUsage: number;
  memoryUsage: number;
  bufferUtilization: number;
}

interface RealTimeMetricsProps {
  agentStatus: any;
  refreshInterval: number;
}

const RealTimeMetrics: React.FC<RealTimeMetricsProps> = ({ 
  agentStatus, 
  refreshInterval 
}) => {
  const [metricsHistory, setMetricsHistory] = useState<MetricsDataPoint[]>([]);
  const [isRecording, setIsRecording] = useState(true);
  const [timeRange, setTimeRange] = useState<number>(60); // seconds

  // Update metrics history when agent status changes
  useEffect(() => {
    if (agentStatus && isRecording) {
      const now = new Date();
      const newDataPoint: MetricsDataPoint = {
        timestamp: now.toLocaleTimeString(),
        latency: agentStatus.metrics.avg_processing_latency_ms || 0,
        throughput: agentStatus.metrics.throughput_events_per_sec || 0,
        anomalies: agentStatus.metrics.anomalies_detected || 0,
        cpuUsage: agentStatus.metrics.cpu_usage_percent || 0,
        memoryUsage: agentStatus.metrics.memory_usage_mb || 0,
        bufferUtilization: agentStatus.metrics.buffer_utilization || 0
      };

      setMetricsHistory(prev => {
        const updated = [...prev, newDataPoint];
        // Keep only data points within time range
        const maxPoints = Math.ceil(timeRange * 1000 / refreshInterval);
        return updated.slice(-maxPoints);
      });
    }
  }, [agentStatus, isRecording, timeRange, refreshInterval]);

  // Calculate statistics
  const statistics = useMemo(() => {
    if (metricsHistory.length === 0) {
      return {
        avgLatency: 0,
        maxLatency: 0,
        minLatency: 0,
        avgThroughput: 0,
        totalAnomalies: 0,
        avgCpuUsage: 0,
        p95Latency: 0
      };
    }

    const latencies = metricsHistory.map(d => d.latency).filter(l => l > 0);
    const throughputs = metricsHistory.map(d => d.throughput);
    const cpuUsages = metricsHistory.map(d => d.cpuUsage);
    
    // Calculate percentiles
    const sortedLatencies = [...latencies].sort((a, b) => a - b);
    const p95Index = Math.ceil(sortedLatencies.length * 0.95) - 1;

    return {
      avgLatency: latencies.reduce((a, b) => a + b, 0) / latencies.length,
      maxLatency: Math.max(...latencies),
      minLatency: Math.min(...latencies),
      avgThroughput: throughputs.reduce((a, b) => a + b, 0) / throughputs.length,
      totalAnomalies: Math.max(...metricsHistory.map(d => d.anomalies)),
      avgCpuUsage: cpuUsages.reduce((a, b) => a + b, 0) / cpuUsages.length,
      p95Latency: sortedLatencies[p95Index] || 0
    };
  }, [metricsHistory]);

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
      return `${eventsPerSec.toFixed(1)}`;
    } else if (eventsPerSec < 1000000) {
      return `${(eventsPerSec / 1000).toFixed(1)}K`;
    } else {
      return `${(eventsPerSec / 1000000).toFixed(1)}M`;
    }
  };

  // Get performance status color
  const getPerformanceStatus = (latency: number) => {
    if (latency < 1) return 'text-green-600';
    if (latency < 10) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Real-Time Metrics</h2>
        
        <div className="flex items-center space-x-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(Number(e.target.value))}
            className="px-3 py-1 border border-gray-300 rounded-md text-sm"
          >
            <option value={30}>Last 30s</option>
            <option value={60}>Last 1m</option>
            <option value={300}>Last 5m</option>
            <option value={900}>Last 15m</option>
          </select>
          
          <button
            onClick={() => setIsRecording(!isRecording)}
            className={`px-3 py-1 rounded-md text-sm font-medium ${
              isRecording 
                ? 'bg-red-100 text-red-700 hover:bg-red-200'
                : 'bg-green-100 text-green-700 hover:bg-green-200'
            }`}
          >
            {isRecording ? 'Pause' : 'Resume'}
          </button>
          
          <button
            onClick={() => setMetricsHistory([])}
            className="px-3 py-1 bg-gray-100 text-gray-700 rounded-md text-sm font-medium hover:bg-gray-200"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600 flex items-center">
              <Zap className="h-4 w-4 mr-1" />
              Avg Latency
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getPerformanceStatus(statistics.avgLatency)}`}>
              {formatLatency(statistics.avgLatency)}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              P95: {formatLatency(statistics.p95Latency)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600 flex items-center">
              <Activity className="h-4 w-4 mr-1" />
              Throughput
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">
              {formatThroughput(statistics.avgThroughput)}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              events/sec
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600 flex items-center">
              <AlertTriangle className="h-4 w-4 mr-1" />
              Anomalies
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${
              statistics.totalAnomalies > 0 ? 'text-red-600' : 'text-green-600'
            }`}>
              {statistics.totalAnomalies}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              detected
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600 flex items-center">
              <Cpu className="h-4 w-4 mr-1" />
              CPU Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${
              statistics.avgCpuUsage > 80 ? 'text-red-600' : 
              statistics.avgCpuUsage > 60 ? 'text-yellow-600' : 'text-green-600'
            }`}>
              {statistics.avgCpuUsage.toFixed(1)}%
            </div>
            <p className="text-xs text-gray-500 mt-1">
              average
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Latency Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Processing Latency</CardTitle>
          <CardDescription>
            Real-time latency monitoring (target: &lt;1ms for critical events)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metricsHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tick={{ fontSize: 12 }}
                  interval="preserveStartEnd"
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value: any) => [formatLatency(value), 'Latency']}
                  labelFormatter={(label) => `Time: ${label}`}
                />
                <Line 
                  type="monotone" 
                  dataKey="latency" 
                  stroke="#2563eb" 
                  strokeWidth={2}
                  dot={false}
                  connectNulls={false}
                />
                {/* Target line at 1ms */}
                <Line 
                  type="monotone" 
                  dataKey={() => 1} 
                  stroke="#ef4444" 
                  strokeDasharray="5 5"
                  strokeWidth={1}
                  dot={false}
                  connectNulls={true}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Throughput and Resource Usage */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Throughput Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Event Throughput</CardTitle>
            <CardDescription>
              Events processed per second
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={metricsHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tick={{ fontSize: 12 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Events/sec', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    formatter={(value: any) => [formatThroughput(value), 'Throughput']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="throughput" 
                    stroke="#059669" 
                    fill="#10b981" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* System Resources */}
        <Card>
          <CardHeader>
            <CardTitle>System Resources</CardTitle>
            <CardDescription>
              CPU and memory utilization
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metricsHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tick={{ fontSize: 12 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Usage (%)', angle: -90, position: 'insideLeft' }}
                    domain={[0, 100]}
                  />
                  <Tooltip 
                    formatter={(value: any, name: any) => [
                      `${Number(value).toFixed(1)}%`, 
                      name === 'cpuUsage' ? 'CPU' : name === 'bufferUtilization' ? 'Buffer' : name
                    ]}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="cpuUsage" 
                    stroke="#dc2626" 
                    strokeWidth={2}
                    dot={false}
                    name="CPU Usage"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="bufferUtilization" 
                    stroke="#7c3aed" 
                    strokeWidth={2}
                    dot={false}
                    name="Buffer Utilization"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Summary</CardTitle>
          <CardDescription>
            Current session statistics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-900">
                {formatLatency(statistics.minLatency)}
              </div>
              <div className="text-sm text-gray-500">Min Latency</div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-900">
                {formatLatency(statistics.maxLatency)}
              </div>
              <div className="text-sm text-gray-500">Max Latency</div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-900">
                {formatLatency(statistics.p95Latency)}
              </div>
              <div className="text-sm text-gray-500">P95 Latency</div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-900">
                {formatThroughput(statistics.avgThroughput)}
              </div>
              <div className="text-sm text-gray-500">Avg Throughput</div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-900">
                {statistics.totalAnomalies}
              </div>
              <div className="text-sm text-gray-500">Total Anomalies</div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-900">
                {metricsHistory.length}
              </div>
              <div className="text-sm text-gray-500">Data Points</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default RealTimeMetrics;
