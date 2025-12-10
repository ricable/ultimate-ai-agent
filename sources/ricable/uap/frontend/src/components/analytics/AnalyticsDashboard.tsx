// Advanced Analytics Dashboard for UAP Platform
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, AreaChart, Area } from 'recharts';
import { 
  Activity, AlertCircle, TrendingUp, TrendingDown, Users, Clock, 
  Server, Cpu, HardDrive, Wifi, DollarSign, Zap, Settings,
  RefreshCw, Download, Filter, Calendar, ChevronDown, ChevronUp,
  Bell, CheckCircle, XCircle, AlertTriangle, Info
} from 'lucide-react';
import { Card } from '../ui/Card';

// Types
interface DashboardData {
  timestamp: number;
  system_health: {
    overall_status: string;
    system_metrics: {
      cpu_percent: number;
      memory_percent: number;
      disk_usage_percent: number;
      process_count: number;
      load_average: [number, number, number];
    };
    services: Record<string, boolean>;
    anomalies: {
      recent_count: number;
      critical_count: number;
      warning_count: number;
    };
  };
  performance_metrics: {
    response_time: {
      current: number;
      target: number;
      status: string;
    };
    throughput: {
      current: number;
      requests_per_second: number;
      messages_per_second: number;
    };
    error_rate: {
      current: number;
      target: number;
      status: string;
    };
  };
  business_metrics: {
    user_engagement: {
      total_users: number;
      active_users: number;
      total_sessions: number;
      avg_session_duration: number;
      growth: number;
    };
    platform_usage: {
      total_requests: number;
      requests_per_user: number;
      framework_distribution: Record<string, number>;
      growth: number;
    };
    cost_metrics: {
      daily_cost: number;
      cost_per_request: number;
      cost_trend: string;
    };
  };
  real_time_activity: Array<{
    type: string;
    timestamp: number;
    description: string;
    metadata: Record<string, any>;
  }>;
  alerts: Array<{
    id: string;
    type: string;
    severity: string;
    title: string;
    message: string;
    timestamp: number;
  }>;
  predictions: {
    recent_predictions: Record<string, any>;
    model_performance: Record<string, any>;
  };
  experiments: {
    active_experiments: Array<any>;
    completed_experiments: Array<any>;
    total_participants: number;
  };
}

interface MetricHistory {
  name: string;
  data: Array<{
    timestamp: number;
    value: number;
  }>;
}

interface TimeRange {
  value: string;
  label: string;
  hours: number;
}

const TIME_RANGES: TimeRange[] = [
  { value: '1h', label: 'Last Hour', hours: 1 },
  { value: '6h', label: 'Last 6 Hours', hours: 6 },
  { value: '24h', label: 'Last 24 Hours', hours: 24 },
  { value: '7d', label: 'Last 7 Days', hours: 168 },
  { value: '30d', label: 'Last 30 Days', hours: 720 },
];

const CHART_COLORS = {
  primary: '#3B82F6',
  success: '#10B981', 
  warning: '#F59E0B',
  danger: '#EF4444',
  purple: '#8B5CF6',
  cyan: '#06B6D4',
  gray: '#6B7280'
};

const SEVERITY_COLORS = {
  critical: 'bg-red-100 text-red-800 border-red-200',
  warning: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  info: 'bg-blue-100 text-blue-800 border-blue-200',
  debug: 'bg-gray-100 text-gray-800 border-gray-200'
};

export const AnalyticsDashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState('24h');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['system_cpu_percent', 'system_memory_percent']);
  const [metricHistory, setMetricHistory] = useState<Record<string, MetricHistory>>({});
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    overview: true,
    performance: true,
    business: true,
    alerts: true
  });
  
  // Real-time updates
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');

  // Fetch dashboard data
  const fetchDashboardData = useCallback(async () => {
    try {
      setError(null);
      
      const response = await fetch('/api/analytics/dashboard', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch dashboard data: ${response.statusText}`);
      }
      
      const data = await response.json();
      setDashboardData(data);
      setLastUpdate(new Date());
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      console.error('Error fetching dashboard data:', err);
    }
  }, []);

  // Fetch metric history
  const fetchMetricHistory = useCallback(async (metricName: string) => {
    try {
      const timeRangeHours = TIME_RANGES.find(tr => tr.value === timeRange)?.hours || 24;
      
      const response = await fetch(`/api/analytics/metrics/${metricName}/history?hours=${timeRangeHours}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch metric history: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      setMetricHistory(prev => ({
        ...prev,
        [metricName]: {
          name: metricName,
          data: data.map((point: any) => ({
            timestamp: point.timestamp * 1000, // Convert to milliseconds
            value: point.value
          }))
        }
      }));
      
    } catch (err) {
      console.error(`Error fetching metric history for ${metricName}:`, err);
    }
  }, [timeRange]);

  // Initialize dashboard
  useEffect(() => {
    const initializeDashboard = async () => {
      setIsLoading(true);
      await fetchDashboardData();
      
      // Fetch history for selected metrics
      for (const metric of selectedMetrics) {
        await fetchMetricHistory(metric);
      }
      
      setIsLoading(false);
    };
    
    initializeDashboard();
  }, [fetchDashboardData, fetchMetricHistory, selectedMetrics]);

  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      fetchDashboardData();
    }, 30000); // Refresh every 30 seconds
    
    return () => clearInterval(interval);
  }, [autoRefresh, fetchDashboardData]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!autoRefresh) return;
    
    const connectWebSocket = () => {
      setConnectionStatus('connecting');
      
      const ws = new WebSocket(`ws://localhost:8000/ws/analytics`);
      
      ws.onopen = () => {
        setConnectionStatus('connected');
        console.log('Analytics WebSocket connected');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'dashboard_update') {
            setDashboardData(data.payload);
            setLastUpdate(new Date());
          } else if (data.type === 'metric_update') {
            // Update specific metric history
            const { metric_name, value, timestamp } = data.payload;
            
            setMetricHistory(prev => {
              if (!prev[metric_name]) return prev;
              
              const updated = { ...prev };
              updated[metric_name] = {
                ...updated[metric_name],
                data: [...updated[metric_name].data, { timestamp: timestamp * 1000, value }].slice(-100)
              };
              
              return updated;
            });
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };
      
      ws.onclose = () => {
        setConnectionStatus('disconnected');
        console.log('Analytics WebSocket disconnected');
        
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      ws.onerror = (error) => {
        console.error('Analytics WebSocket error:', error);
        setConnectionStatus('disconnected');
      };
      
      return ws;
    };
    
    const ws = connectWebSocket();
    
    return () => {
      ws.close();
    };
  }, [autoRefresh]);

  // Utility functions
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  const formatTimestamp = (timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'excellent':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'warning':
      case 'degraded':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case 'error':
      case 'critical':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Info className="h-5 w-5 text-blue-500" />;
    }
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const exportData = async () => {
    try {
      const response = await fetch('/api/analytics/export', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          format: 'csv',
          time_period: timeRange,
          include_details: true
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to export data');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `uap-analytics-${timeRange}-${new Date().toISOString().split('T')[0]}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
      
    } catch (err) {
      console.error('Error exporting data:', err);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-32"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-80"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error && !dashboardData) {
    return (
      <div className="p-6">
        <Card className="p-6">
          <div className="flex items-center space-x-3 text-red-600">
            <AlertCircle className="h-6 w-6" />
            <div>
              <h3 className="text-lg font-semibold">Error Loading Dashboard</h3>
              <p className="text-sm text-red-500">{error}</p>
              <button
                onClick={fetchDashboardData}
                className="mt-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Retry
              </button>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="p-6">
        <Card className="p-6">
          <div className="text-center text-gray-500">
            <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No dashboard data available</p>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
          <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
            <div className="flex items-center space-x-1">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-500' : 
                connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
              }`}></div>
              <span className="capitalize">{connectionStatus}</span>
            </div>
            <span>•</span>
            <span>Last updated: {lastUpdate.toLocaleTimeString()}</span>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
          >
            {TIME_RANGES.map(range => (
              <option key={range.value} value={range.value}>
                {range.label}
              </option>
            ))}
          </select>
          
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`p-2 rounded-lg transition-colors ${
              autoRefresh 
                ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            title={autoRefresh ? 'Disable auto-refresh' : 'Enable auto-refresh'}
          >
            <RefreshCw className={`h-4 w-4 ${autoRefresh ? 'animate-spin' : ''}`} />
          </button>
          
          <button
            onClick={exportData}
            className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </button>
        </div>
      </div>

      {/* System Overview */}
      <Card className="p-6">
        <div 
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('overview')}
        >
          <div className="flex items-center space-x-3">
            <Server className="h-6 w-6 text-blue-600" />
            <h2 className="text-xl font-semibold">System Overview</h2>
            {getStatusIcon(dashboardData.system_health.overall_status)}
          </div>
          {expandedSections.overview ? 
            <ChevronUp className="h-5 w-5 text-gray-400" /> : 
            <ChevronDown className="h-5 w-5 text-gray-400" />
          }
        </div>
        
        {expandedSections.overview && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* CPU Usage */}
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-700">CPU Usage</p>
                  <p className="text-2xl font-bold text-blue-900">
                    {dashboardData.system_health.system_metrics.cpu_percent.toFixed(1)}%
                  </p>
                </div>
                <Cpu className="h-8 w-8 text-blue-600" />
              </div>
              <div className="mt-2 bg-blue-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 rounded-full h-2 transition-all duration-300"
                  style={{ width: `${Math.min(dashboardData.system_health.system_metrics.cpu_percent, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Memory Usage */}
            <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-700">Memory Usage</p>
                  <p className="text-2xl font-bold text-green-900">
                    {dashboardData.system_health.system_metrics.memory_percent.toFixed(1)}%
                  </p>
                </div>
                <Activity className="h-8 w-8 text-green-600" />
              </div>
              <div className="mt-2 bg-green-200 rounded-full h-2">
                <div 
                  className="bg-green-600 rounded-full h-2 transition-all duration-300"
                  style={{ width: `${Math.min(dashboardData.system_health.system_metrics.memory_percent, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Disk Usage */}
            <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 p-4 rounded-lg border border-yellow-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-yellow-700">Disk Usage</p>
                  <p className="text-2xl font-bold text-yellow-900">
                    {dashboardData.system_health.system_metrics.disk_usage_percent.toFixed(1)}%
                  </p>
                </div>
                <HardDrive className="h-8 w-8 text-yellow-600" />
              </div>
              <div className="mt-2 bg-yellow-200 rounded-full h-2">
                <div 
                  className="bg-yellow-600 rounded-full h-2 transition-all duration-300"
                  style={{ width: `${Math.min(dashboardData.system_health.system_metrics.disk_usage_percent, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Network Activity */}
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-700">Active Connections</p>
                  <p className="text-2xl font-bold text-purple-900">
                    {dashboardData.business_metrics.user_engagement.active_users}
                  </p>
                  <p className="text-xs text-purple-600 mt-1">
                    {formatNumber(dashboardData.performance_metrics.throughput.requests_per_second)} req/s
                  </p>
                </div>
                <Wifi className="h-8 w-8 text-purple-600" />
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Performance Metrics */}
      <Card className="p-6">
        <div 
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('performance')}
        >
          <div className="flex items-center space-x-3">
            <Zap className="h-6 w-6 text-yellow-600" />
            <h2 className="text-xl font-semibold">Performance Metrics</h2>
            {getStatusIcon(dashboardData.performance_metrics.response_time.status)}
          </div>
          {expandedSections.performance ? 
            <ChevronUp className="h-5 w-5 text-gray-400" /> : 
            <ChevronDown className="h-5 w-5 text-gray-400" />
          }
        </div>
        
        {expandedSections.performance && (
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Response Time */}
            <div className="bg-white p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">Response Time</h3>
                <Clock className="h-5 w-5 text-gray-400" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Current</span>
                  <span className="text-lg font-bold">
                    {dashboardData.performance_metrics.response_time.current.toFixed(0)}ms
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Target</span>
                  <span className="text-sm text-gray-500">
                    {dashboardData.performance_metrics.response_time.target}ms
                  </span>
                </div>
                <div className="bg-gray-200 rounded-full h-2">
                  <div 
                    className={`rounded-full h-2 transition-all duration-300 ${
                      dashboardData.performance_metrics.response_time.status === 'excellent' ? 'bg-green-500' :
                      dashboardData.performance_metrics.response_time.status === 'good' ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ 
                      width: `${Math.min(
                        (dashboardData.performance_metrics.response_time.current / 
                         dashboardData.performance_metrics.response_time.target) * 100, 
                        100
                      )}%` 
                    }}
                  ></div>
                </div>
                <div className="text-xs text-green-600">
                  {(dashboardData.performance_metrics.response_time.target / 
                    Math.max(dashboardData.performance_metrics.response_time.current, 1)
                  ).toFixed(0)}x better than target
                </div>
              </div>
            </div>

            {/* Throughput */}
            <div className="bg-white p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">Throughput</h3>
                <TrendingUp className="h-5 w-5 text-gray-400" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Requests/sec</span>
                  <span className="text-lg font-bold">
                    {dashboardData.performance_metrics.throughput.requests_per_second.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Messages/sec</span>
                  <span className="text-sm text-gray-700">
                    {dashboardData.performance_metrics.throughput.messages_per_second.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>

            {/* Error Rate */}
            <div className="bg-white p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">Error Rate</h3>
                <AlertCircle className="h-5 w-5 text-gray-400" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Current</span>
                  <span className="text-lg font-bold">
                    {(dashboardData.performance_metrics.error_rate.current * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Target</span>
                  <span className="text-sm text-gray-500">
                    {(dashboardData.performance_metrics.error_rate.target * 100).toFixed(1)}%
                  </span>
                </div>
                <div className={`text-xs ${
                  dashboardData.performance_metrics.error_rate.status === 'excellent' ? 'text-green-600' :
                  dashboardData.performance_metrics.error_rate.status === 'good' ? 'text-yellow-600' :
                  'text-red-600'
                }`}>
                  {dashboardData.performance_metrics.error_rate.status === 'excellent' ? '✓ Excellent' :
                   dashboardData.performance_metrics.error_rate.status === 'good' ? '⚠ Good' :
                   '✗ Needs attention'}
                </div>
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Business Metrics */}
      <Card className="p-6">
        <div 
          className="flex items-center justify-between cursor-pointer"
          onClick={() => toggleSection('business')}
        >
          <div className="flex items-center space-x-3">
            <DollarSign className="h-6 w-6 text-green-600" />
            <h2 className="text-xl font-semibold">Business Intelligence</h2>
          </div>
          {expandedSections.business ? 
            <ChevronUp className="h-5 w-5 text-gray-400" /> : 
            <ChevronDown className="h-5 w-5 text-gray-400" />
          }
        </div>
        
        {expandedSections.business && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* User Engagement */}
            <div className="bg-white p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">User Engagement</h3>
                <Users className="h-5 w-5 text-blue-600" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Total Users</span>
                  <span className="font-medium">
                    {formatNumber(dashboardData.business_metrics.user_engagement.total_users)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Active Users</span>
                  <span className="font-medium text-green-600">
                    {formatNumber(dashboardData.business_metrics.user_engagement.active_users)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Avg Session</span>
                  <span className="font-medium">
                    {formatDuration(dashboardData.business_metrics.user_engagement.avg_session_duration)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Growth</span>
                  <div className="flex items-center space-x-1">
                    {dashboardData.business_metrics.user_engagement.growth >= 0 ? 
                      <TrendingUp className="h-3 w-3 text-green-500" /> : 
                      <TrendingDown className="h-3 w-3 text-red-500" />
                    }
                    <span className={`text-sm font-medium ${
                      dashboardData.business_metrics.user_engagement.growth >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {dashboardData.business_metrics.user_engagement.growth.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Platform Usage */}
            <div className="bg-white p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">Platform Usage</h3>
                <Activity className="h-5 w-5 text-purple-600" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Total Requests</span>
                  <span className="font-medium">
                    {formatNumber(dashboardData.business_metrics.platform_usage.total_requests)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Req/User</span>
                  <span className="font-medium">
                    {dashboardData.business_metrics.platform_usage.requests_per_user.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Growth</span>
                  <div className="flex items-center space-x-1">
                    {dashboardData.business_metrics.platform_usage.growth >= 0 ? 
                      <TrendingUp className="h-3 w-3 text-green-500" /> : 
                      <TrendingDown className="h-3 w-3 text-red-500" />
                    }
                    <span className={`text-sm font-medium ${
                      dashboardData.business_metrics.platform_usage.growth >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {dashboardData.business_metrics.platform_usage.growth.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Cost Metrics */}
            <div className="bg-white p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">Cost Analysis</h3>
                <DollarSign className="h-5 w-5 text-green-600" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Daily Cost</span>
                  <span className="font-medium">
                    ${dashboardData.business_metrics.cost_metrics.daily_cost.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Cost/Request</span>
                  <span className="font-medium">
                    ${dashboardData.business_metrics.cost_metrics.cost_per_request.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Trend</span>
                  <span className={`text-sm font-medium capitalize ${
                    dashboardData.business_metrics.cost_metrics.cost_trend === 'increasing' ? 'text-red-600' :
                    dashboardData.business_metrics.cost_metrics.cost_trend === 'decreasing' ? 'text-green-600' :
                    'text-gray-600'
                  }`}>
                    {dashboardData.business_metrics.cost_metrics.cost_trend}
                  </span>
                </div>
              </div>
            </div>

            {/* Framework Distribution */}
            <div className="bg-white p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">Framework Usage</h3>
                <Settings className="h-5 w-5 text-gray-600" />
              </div>
              <div className="space-y-2">
                {Object.entries(dashboardData.business_metrics.platform_usage.framework_distribution).map(
                  ([framework, count], index) => {
                    const total = Object.values(dashboardData.business_metrics.platform_usage.framework_distribution)
                      .reduce((sum, val) => sum + val, 0);
                    const percentage = total > 0 ? (count / total) * 100 : 0;
                    
                    return (
                      <div key={framework} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="capitalize">{framework}</span>
                          <span>{percentage.toFixed(1)}%</span>
                        </div>
                        <div className="bg-gray-200 rounded-full h-2">
                          <div 
                            className={`rounded-full h-2 transition-all duration-300 ${
                              index === 0 ? 'bg-blue-500' :
                              index === 1 ? 'bg-green-500' :
                              'bg-purple-500'
                            }`}
                            style={{ width: `${percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    );
                  }
                )}
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Alerts and Notifications */}
      {dashboardData.alerts.length > 0 && (
        <Card className="p-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection('alerts')}
          >
            <div className="flex items-center space-x-3">
              <Bell className="h-6 w-6 text-red-600" />
              <h2 className="text-xl font-semibold">Active Alerts</h2>
              <span className="bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full">
                {dashboardData.alerts.length}
              </span>
            </div>
            {expandedSections.alerts ? 
              <ChevronUp className="h-5 w-5 text-gray-400" /> : 
              <ChevronDown className="h-5 w-5 text-gray-400" />
            }
          </div>
          
          {expandedSections.alerts && (
            <div className="mt-6 space-y-3">
              {dashboardData.alerts.slice(0, 10).map((alert) => (
                <div 
                  key={alert.id} 
                  className={`p-4 rounded-lg border-l-4 ${
                    SEVERITY_COLORS[alert.severity as keyof typeof SEVERITY_COLORS] || SEVERITY_COLORS.info
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <h4 className="font-medium">{alert.title}</h4>
                        <span className={`text-xs px-2 py-1 rounded-full uppercase font-medium ${
                          alert.severity === 'critical' ? 'bg-red-100 text-red-800' :
                          alert.severity === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-blue-100 text-blue-800'
                        }`}>
                          {alert.severity}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{alert.message}</p>
                    </div>
                    <span className="text-xs text-gray-500 whitespace-nowrap ml-4">
                      {formatTimestamp(alert.timestamp)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Recent Activity Feed */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <Activity className="h-6 w-6 text-blue-600 mr-2" />
          Real-time Activity
        </h2>
        
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {dashboardData.real_time_activity.slice(0, 20).map((activity, index) => (
            <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
              <div className={`w-2 h-2 rounded-full ${
                activity.type === 'anomaly' ? 'bg-red-500' :
                activity.type === 'agent_request' ? 'bg-blue-500' :
                'bg-green-500'
              }`}></div>
              <div className="flex-1">
                <p className="text-sm">{activity.description}</p>
                <p className="text-xs text-gray-500">{formatTimestamp(activity.timestamp)}</p>
              </div>
              {activity.metadata.response_time && (
                <span className="text-xs text-gray-500">
                  {activity.metadata.response_time.toFixed(0)}ms
                </span>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
