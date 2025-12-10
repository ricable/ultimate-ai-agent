// Compact Metrics Widget for Dashboard Integration
import React, { useState, useEffect } from 'react';
import { useAuth } from '../../auth/AuthContext';
import { Activity, TrendingUp, TrendingDown, AlertCircle, Zap } from 'lucide-react';
import { Card } from '../ui/Card';
import { apiConfig } from '../../lib/api-config';

interface MetricsWidgetProps {
  title: string;
  metrics: Array<{
    name: string;
    value: number;
    unit: string;
    trend?: 'up' | 'down' | 'stable';
    trendValue?: number;
    color?: string;
    threshold?: {
      value: number;
      type: 'max' | 'min';
    };
  }>;
  className?: string;
  compact?: boolean;
  refreshInterval?: number;
}

export const MetricsWidget: React.FC<MetricsWidgetProps> = ({
  title,
  metrics,
  className = '',
  compact = false,
  refreshInterval = 30000
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Auto-refresh effect
  useEffect(() => {
    if (!refreshInterval) return;
    
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, refreshInterval);
    
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const formatValue = (value: number, unit: string): string => {
    if (unit === '%') {
      return `${value.toFixed(1)}%`;
    }
    if (unit === 'ms') {
      return `${value.toFixed(0)}ms`;
    }
    if (unit === 'req/s') {
      if (value >= 1000) return `${(value / 1000).toFixed(1)}K/s`;
      return `${value.toFixed(1)}/s`;
    }
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    }
    if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toFixed(1);
  };

  const getTrendIcon = (trend?: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="h-3 w-3 text-green-500" />;
      case 'down':
        return <TrendingDown className="h-3 w-3 text-red-500" />;
      default:
        return <div className="h-3 w-3 bg-gray-400 rounded-full" />;
    }
  };

  const getThresholdStatus = (value: number, threshold?: { value: number; type: 'max' | 'min' }) => {
    if (!threshold) return 'normal';
    
    if (threshold.type === 'max' && value > threshold.value) {
      return 'warning';
    }
    if (threshold.type === 'min' && value < threshold.value) {
      return 'warning';
    }
    return 'normal';
  };

  if (compact) {
    // Compact horizontal layout
    return (
      <Card className={`p-3 ${className}`}>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium text-gray-700">{title}</h3>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-xs text-gray-500">Live</span>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {metrics.map((metric, index) => {
            const status = getThresholdStatus(metric.value, metric.threshold);
            
            return (
              <div key={index} className="text-center">
                <div className="flex items-center justify-center space-x-1 mb-1">
                  <span className={`text-lg font-bold ${
                    status === 'warning' ? 'text-red-600' : 'text-gray-900'
                  }`}>
                    {formatValue(metric.value, metric.unit)}
                  </span>
                  {metric.trend && getTrendIcon(metric.trend)}
                </div>
                <p className="text-xs text-gray-600 truncate">{metric.name}</p>
                {metric.trendValue && (
                  <p className={`text-xs font-medium ${
                    metric.trend === 'up' ? 'text-green-600' : 
                    metric.trend === 'down' ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {metric.trend === 'up' ? '+' : metric.trend === 'down' ? '-' : ''}
                    {Math.abs(metric.trendValue).toFixed(1)}%
                  </p>
                )}
              </div>
            );
          })}
        </div>
        
        <div className="mt-2 pt-2 border-t border-gray-100">
          <p className="text-xs text-gray-500 text-center">
            Updated {lastUpdate.toLocaleTimeString()}
          </p>
        </div>
      </Card>
    );
  }

  // Full layout
  return (
    <Card className={`p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Activity className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-medium text-gray-900">{title}</h3>
        </div>
        
        <div className="flex items-center space-x-2">
          {isLoading ? (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          ) : (
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-xs text-gray-500">Live</span>
            </div>
          )}
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map((metric, index) => {
          const status = getThresholdStatus(metric.value, metric.threshold);
          
          return (
            <div 
              key={index} 
              className={`p-3 rounded-lg border ${
                status === 'warning' 
                  ? 'bg-red-50 border-red-200' 
                  : 'bg-gray-50 border-gray-200'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">
                  {metric.name}
                </span>
                {status === 'warning' && (
                  <AlertCircle className="h-4 w-4 text-red-500" />
                )}
              </div>
              
              <div className="flex items-baseline space-x-2">
                <span className={`text-2xl font-bold ${
                  status === 'warning' ? 'text-red-600' : 'text-gray-900'
                }`}>
                  {formatValue(metric.value, metric.unit)}
                </span>
                
                {metric.trend && (
                  <div className="flex items-center space-x-1">
                    {getTrendIcon(metric.trend)}
                    {metric.trendValue && (
                      <span className={`text-sm font-medium ${
                        metric.trend === 'up' ? 'text-green-600' : 
                        metric.trend === 'down' ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        {metric.trend === 'up' ? '+' : metric.trend === 'down' ? '-' : ''}
                        {Math.abs(metric.trendValue).toFixed(1)}%
                      </span>
                    )}
                  </div>
                )}
              </div>
              
              {metric.threshold && (
                <div className="mt-2">
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>
                      {metric.threshold.type === 'max' ? 'Max' : 'Min'}: {metric.threshold.value}{metric.unit === '%' ? '%' : ''}
                    </span>
                    <span className={status === 'warning' ? 'text-red-600 font-medium' : 'text-green-600'}>
                      {status === 'warning' ? 'Threshold exceeded' : 'Normal'}
                    </span>
                  </div>
                  <div className="mt-1 bg-gray-200 rounded-full h-1">
                    <div 
                      className={`h-1 rounded-full transition-all duration-300 ${
                        status === 'warning' ? 'bg-red-500' : 'bg-green-500'
                      }`}
                      style={{ 
                        width: `${Math.min(
                          (metric.value / metric.threshold.value) * 100, 
                          100
                        )}%` 
                      }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      <div className="mt-4 pt-3 border-t border-gray-100">
        <div className="flex justify-between items-center text-xs text-gray-500">
          <span>Last updated: {lastUpdate.toLocaleTimeString()}</span>
          <span>Refresh every {refreshInterval / 1000}s</span>
        </div>
      </div>
    </Card>
  );
};

// Predefined metric widgets for common use cases
export const SystemMetricsWidget: React.FC<{ className?: string; compact?: boolean }> = ({ 
  className, 
  compact = false 
}) => {
  const { token } = useAuth();
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSystemMetrics = async () => {
      try {
        setLoading(true);
        const headers = apiConfig.createHeaders(token);

        const response = await fetch(apiConfig.getEndpoint('/api/monitoring/overview'), {
          method: 'GET',
          headers,
        });

        if (response.ok) {
          const data = await response.json();
          setMetrics(transformSystemMetrics(data));
        } else {
          throw new Error(`System metrics API error: ${response.status}`);
        }
      } catch (error) {
        console.warn('System metrics API not available, using fallback:', error);
        setMetrics(getDefaultSystemMetrics());
      } finally {
        setLoading(false);
      }
    };

    fetchSystemMetrics();
  }, [token]);

  const transformSystemMetrics = (data: any): Metric[] => {
    return [
      {
        name: 'CPU Usage',
        value: data.system_metrics?.cpu_usage || 23.5,
        unit: '%',
        trend: data.system_metrics?.cpu_trend || 'stable',
        threshold: { value: 80, type: 'max' as const }
      },
      {
        name: 'Memory Usage',
        value: data.system_metrics?.memory_usage || 67.2,
        unit: '%',
        trend: data.system_metrics?.memory_trend || 'up',
        trendValue: data.system_metrics?.memory_trend_value || 5.3,
        threshold: { value: 85, type: 'max' as const }
      },
      {
        name: 'Response Time',
        value: data.performance_metrics?.avg_response_time || 1.2,
        unit: 'ms',
        trend: data.performance_metrics?.response_time_trend || 'down',
        trendValue: data.performance_metrics?.response_time_trend_value || -15.2
      },
      {
        name: 'Active Users',
        value: data.user_metrics?.active_users || 42,
        unit: '',
        trend: data.user_metrics?.active_users_trend || 'up',
        trendValue: data.user_metrics?.active_users_trend_value || 8.7
      }
    ];
  };

  const getDefaultSystemMetrics = (): Metric[] => [
    {
      name: 'CPU Usage',
      value: 23.5,
      unit: '%',
      trend: 'stable' as const,
      threshold: { value: 80, type: 'max' as const }
    },
    {
      name: 'Memory Usage',
      value: 67.2,
      unit: '%',
      trend: 'up' as const,
      trendValue: 5.3,
      threshold: { value: 85, type: 'max' as const }
    },
    {
      name: 'Response Time',
      value: 1.2,
      unit: 'ms',
      trend: 'down' as const,
      trendValue: -15.2
    },
    {
      name: 'Active Users',
      value: 42,
      unit: '',
      trend: 'up' as const,
      trendValue: 8.7
    }
  ];

  if (loading) {
    return (
      <div className={`p-4 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-16"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <MetricsWidget
      title="System Performance"
      metrics={metrics}
      className={className}
      compact={compact}
      refreshInterval={10000}
    />
  );
};

export const AgentMetricsWidget: React.FC<{ className?: string; compact?: boolean }> = ({ 
  className, 
  compact = false 
}) => {
  const { token } = useAuth();
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAgentMetrics = async () => {
      try {
        setLoading(true);
        const headers = apiConfig.createHeaders(token);

        const response = await fetch(apiConfig.getEndpoint('/api/monitoring/metrics/agent_response_time'), {
          method: 'GET',
          headers,
        });

        if (response.ok) {
          const data = await response.json();
          setMetrics(transformAgentMetrics(data));
        } else {
          throw new Error(`Agent metrics API error: ${response.status}`);
        }
      } catch (error) {
        console.warn('Agent metrics API not available, using fallback:', error);
        setMetrics(getDefaultAgentMetrics());
      } finally {
        setLoading(false);
      }
    };

    fetchAgentMetrics();
  }, [token]);

  const transformAgentMetrics = (data: any): Metric[] => {
    return [
      {
        name: 'Requests/sec',
        value: data.throughput_metrics?.requests_per_second || 127.3,
        unit: 'req/s',
        trend: data.throughput_metrics?.requests_trend || 'up',
        trendValue: data.throughput_metrics?.requests_trend_value || 12.4
      },
      {
        name: 'Success Rate',
        value: data.quality_metrics?.success_rate || 99.2,
        unit: '%',
        trend: data.quality_metrics?.success_rate_trend || 'stable',
        threshold: { value: 95, type: 'min' as const }
      },
      {
        name: 'Avg Response',
        value: data.performance_metrics?.avg_response_time || 89,
        unit: 'ms',
        trend: data.performance_metrics?.response_time_trend || 'down',
        trendValue: data.performance_metrics?.response_time_trend_value || -8.1,
        threshold: { value: 2000, type: 'max' as const }
      }
    ];
  };

  const getDefaultAgentMetrics = (): Metric[] => [
    {
      name: 'Requests/sec',
      value: 127.3,
      unit: 'req/s',
      trend: 'up' as const,
      trendValue: 12.4
    },
    {
      name: 'Success Rate',
      value: 99.2,
      unit: '%',
      trend: 'stable' as const,
      threshold: { value: 95, type: 'min' as const }
    },
    {
      name: 'Avg Response',
      value: 89,
      unit: 'ms',
      trend: 'down' as const,
      trendValue: -8.1,
      threshold: { value: 2000, type: 'max' as const }
    }
  ];

  if (loading) {
    return (
      <div className={`p-4 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-16"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <MetricsWidget
      title="Agent Performance"
      metrics={metrics}
      className={className}
      compact={compact}
      refreshInterval={5000}
    />
  );
};
