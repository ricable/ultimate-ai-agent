// Performance monitoring dashboard with detailed metrics and analytics
import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Activity, Clock, Zap, AlertTriangle, TrendingUp, TrendingDown, Server, Database, Wifi, RefreshCw } from 'lucide-react';

interface PerformanceMetric {
  timestamp: string;
  value: number;
  tags?: Record<string, string>;
}

interface AgentPerformance {
  agent_id: string;
  framework: string;
  total_requests: number;
  avg_response_time_ms: number;
  p95_response_time_ms: number;
  p99_response_time_ms: number;
  success_rate: number;
  last_request_time: string | null;
}

interface PerformanceData {
  time_window_minutes: number;
  metrics_summary: Record<string, any>;
  agent_stats: Record<string, AgentPerformance>;
  websocket_summary: {
    total_active_connections: number;
    connections_by_agent: Record<string, number>;
    connection_details: Array<{
      connection_id: string;
      agent_id: string;
      connected_at: string;
      duration_seconds: number;
      messages_sent: number;
      messages_received: number;
      bytes_sent: number;
      bytes_received: number;
    }>;
  };
}

const CHART_COLORS = {
  primary: '#3B82F6',
  secondary: '#10B981',
  warning: '#F59E0B',
  error: '#EF4444',
  purple: '#8B5CF6',
};

export const PerformanceMonitoring: React.FC = () => {
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null);
  const [responseTimeData, setResponseTimeData] = useState<PerformanceMetric[]>([]);
  const [requestVolumeData, setRequestVolumeData] = useState<PerformanceMetric[]>([]);
  const [errorRateData, setErrorRateData] = useState<PerformanceMetric[]>([]);
  const [timeWindow, setTimeWindow] = useState<number>(60); // minutes
  const [selectedAgent, setSelectedAgent] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchPerformanceData = async () => {
    try {
      const response = await fetch(`/api/monitoring/overview`);
      if (!response.ok) throw new Error('Failed to fetch performance data');
      
      const data = await response.json();
      setPerformanceData({
        time_window_minutes: timeWindow,
        metrics_summary: data,
        agent_stats: data.agent_stats || {},
        websocket_summary: {
          total_active_connections: data.active_connections || 0,
          connections_by_agent: {},
          connection_details: []
        }
      });
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch performance data:', error);
    }
  };

  const fetchMetricTimeSeries = async (metricName: string, setter: (data: PerformanceMetric[]) => void) => {
    try {
      const tagFilter = selectedAgent !== 'all' ? `{"agent_id":"${selectedAgent}"}` : '';
      const response = await fetch(`/api/monitoring/metrics/${metricName}?time_window=${timeWindow}&tags=${encodeURIComponent(tagFilter)}`);
      
      if (!response.ok) throw new Error(`Failed to fetch ${metricName} data`);
      
      const data = await response.json();
      const formattedData = data.data_points?.map((point: any) => ({
        timestamp: new Date(point.timestamp).toLocaleTimeString(),
        value: point.value,
        tags: point.tags,
      })) || [];
      
      setter(formattedData);
    } catch (error) {
      console.error(`Failed to fetch ${metricName} data:`, error);
      // Set empty data if API fails - no mock fallback
      setter([]);
    }
  };

  // Removed mock data generation - using real API data only

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      await Promise.all([
        fetchPerformanceData(),
        fetchMetricTimeSeries('agent_response_time', setResponseTimeData),
        fetchMetricTimeSeries('request_volume', setRequestVolumeData),
        fetchMetricTimeSeries('error_rate', setErrorRateData),
      ]);
      
      setIsLoading(false);
    };

    loadData();

    // Set up real-time updates
    const interval = setInterval(loadData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [timeWindow, selectedAgent]);

  const getPerformanceSummary = () => {
    if (!performanceData) return null;

    const agents = Object.values(performanceData.agent_stats);
    const totalRequests = agents.reduce((sum, agent) => sum + agent.total_requests, 0);
    const avgResponseTime = agents.length > 0 
      ? agents.reduce((sum, agent) => sum + agent.avg_response_time_ms, 0) / agents.length 
      : 0;
    const avgSuccessRate = agents.length > 0 
      ? agents.reduce((sum, agent) => sum + agent.success_rate, 0) / agents.length 
      : 0;

    return {
      totalRequests,
      avgResponseTime,
      avgSuccessRate,
      activeAgents: agents.length,
      activeConnections: performanceData.websocket_summary.total_active_connections,
    };
  };

  const summary = getPerformanceSummary();
  const agents = performanceData ? Object.values(performanceData.agent_stats) : [];

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-32"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-80"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header with Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h1 className="text-3xl font-bold text-gray-900">Performance Monitoring</h1>
        
        <div className="flex items-center space-x-4">
          {/* Time Window Selector */}
          <select
            value={timeWindow}
            onChange={(e) => setTimeWindow(Number(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value={15}>Last 15 minutes</option>
            <option value={60}>Last hour</option>
            <option value={240}>Last 4 hours</option>
            <option value={1440}>Last 24 hours</option>
          </select>

          {/* Agent Filter */}
          <select
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Agents</option>
            {agents.map(agent => (
              <option key={agent.agent_id} value={agent.agent_id}>
                {agent.agent_id} ({agent.framework})
              </option>
            ))}
          </select>

          {/* Last Update */}
          <div className="flex items-center text-sm text-gray-500">
            <RefreshCw className="h-4 w-4 mr-2" />
            {lastUpdate.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Performance Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Requests</p>
                <p className="text-2xl font-bold text-gray-900">{summary.totalRequests.toLocaleString()}</p>
              </div>
              <Database className="h-8 w-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Response Time</p>
                <p className="text-2xl font-bold text-gray-900">{Math.round(summary.avgResponseTime)}ms</p>
              </div>
              <Clock className="h-8 w-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Success Rate</p>
                <p className="text-2xl font-bold text-gray-900">{(summary.avgSuccessRate * 100).toFixed(1)}%</p>
              </div>
              <Zap className="h-8 w-8 text-yellow-500" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Agents</p>
                <p className="text-2xl font-bold text-gray-900">{summary.activeAgents}</p>
              </div>
              <Server className="h-8 w-8 text-purple-500" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Connections</p>
                <p className="text-2xl font-bold text-gray-900">{summary.activeConnections}</p>
              </div>
              <Wifi className="h-8 w-8 text-green-500" />
            </div>
          </div>
        </div>
      )}

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Response Time Chart */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Activity className="h-5 w-5 mr-2 text-blue-500" />
            Response Time Trends
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={responseTimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip formatter={(value: any) => [`${Math.round(value)}ms`, 'Response Time']} />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke={CHART_COLORS.primary} 
                strokeWidth={2}
                dot={{ fill: CHART_COLORS.primary, strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Request Volume Chart */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 mr-2 text-green-500" />
            Request Volume
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={requestVolumeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip formatter={(value: any) => [`${value}`, 'Requests']} />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke={CHART_COLORS.secondary} 
                fill={CHART_COLORS.secondary} 
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Error Rate Chart */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2 text-red-500" />
            Error Rate
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={errorRateData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip formatter={(value: any) => [`${value.toFixed(2)}%`, 'Error Rate']} />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke={CHART_COLORS.error} 
                fill={CHART_COLORS.error} 
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Agent Performance Comparison */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4">Agent Performance Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={agents}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="agent_id" />
              <YAxis />
              <Tooltip 
                formatter={(value: any, name: string) => [
                  name === 'avg_response_time_ms' ? `${Math.round(value)}ms` : `${(value * 100).toFixed(1)}%`,
                  name === 'avg_response_time_ms' ? 'Avg Response Time' : 'Success Rate'
                ]}
              />
              <Legend />
              <Bar dataKey="avg_response_time_ms" fill={CHART_COLORS.primary} name="Response Time (ms)" />
              <Bar dataKey="success_rate" fill={CHART_COLORS.secondary} name="Success Rate" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Agent Statistics */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold">Detailed Agent Statistics</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Framework</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Requests</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Response</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">P95 Response</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">P99 Response</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Success Rate</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Request</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {agents.map((agent) => (
                <tr key={agent.agent_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {agent.agent_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                      {agent.framework}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {agent.total_requests.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {Math.round(agent.avg_response_time_ms)}ms
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {Math.round(agent.p95_response_time_ms)}ms
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {Math.round(agent.p99_response_time_ms)}ms
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      agent.success_rate > 0.95 
                        ? 'bg-green-100 text-green-800' 
                        : agent.success_rate > 0.9 
                        ? 'bg-yellow-100 text-yellow-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {(agent.success_rate * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {agent.last_request_time 
                      ? new Date(agent.last_request_time).toLocaleString()
                      : 'Never'
                    }
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};