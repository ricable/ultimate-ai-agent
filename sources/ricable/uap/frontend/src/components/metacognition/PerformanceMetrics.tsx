// File: frontend/src/components/metacognition/PerformanceMetrics.tsx
import { useState, useEffect } from 'react';

// Dummy UI components
const Card = ({ children, className }: any) => <div className={`border rounded-lg p-4 shadow-md bg-white ${className}`}>{children}</div>;
const CardHeader = ({ children }: any) => <div className="font-bold text-lg mb-2">{children}</div>;
const CardContent = ({ children }: any) => <div className="text-sm text-gray-700">{children}</div>;
const Badge = ({ children, className }: any) => <span className={`text-xs font-semibold mr-2 px-2.5 py-0.5 rounded-full ${className}`}>{children}</span>;
const Button = (props: any) => <button className="bg-blue-500 text-white rounded px-4 py-2 disabled:bg-gray-400" {...props} />;
const Select = (props: any) => <select className="border rounded px-3 py-2" {...props} />;

interface PerformanceMetricsProps {
  agentId: string;
}

interface Metric {
  name: string;
  current_value: number;
  target_value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  change_percentage: number;
}

interface OptimizationResult {
  timestamp: string;
  optimization_type: string;
  target_metric: string;
  improvement: number;
  success: boolean;
}

interface PerformanceData {
  current_metrics: Metric[];
  recent_optimizations: OptimizationResult[];
  optimization_status: {
    active_targets: number;
    queued_actions: number;
    success_rate: number;
    current_strategy: string;
  };
  performance_history: {
    timestamps: string[];
    response_time: number[];
    accuracy: number[];
    efficiency: number[];
  };
}

export function PerformanceMetrics({ agentId }: PerformanceMetricsProps) {
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('24h');
  const [triggeringOptimization, setTriggeringOptimization] = useState(false);

  useEffect(() => {
    fetchPerformanceData();
    const interval = setInterval(fetchPerformanceData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [agentId, timeRange]);

  const fetchPerformanceData = async () => {
    try {
      setLoading(true);
      // In a real implementation, this would fetch actual performance data
      // For now, we'll simulate the data
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const mockData: PerformanceData = {
        current_metrics: [
          {
            name: 'Response Time',
            current_value: 1.2,
            target_value: 1.0,
            unit: 'seconds',
            trend: 'down',
            change_percentage: -8.5
          },
          {
            name: 'Accuracy',
            current_value: 0.87,
            target_value: 0.90,
            unit: '%',
            trend: 'up',
            change_percentage: 2.3
          },
          {
            name: 'Efficiency',
            current_value: 0.78,
            target_value: 0.85,
            unit: '%',
            trend: 'up',
            change_percentage: 5.1
          },
          {
            name: 'CPU Utilization',
            current_value: 0.65,
            target_value: 0.70,
            unit: '%',
            trend: 'stable',
            change_percentage: 0.8
          },
          {
            name: 'Memory Usage',
            current_value: 0.52,
            target_value: 0.60,
            unit: '%',
            trend: 'down',
            change_percentage: -3.2
          },
          {
            name: 'User Satisfaction',
            current_value: 0.82,
            target_value: 0.85,
            unit: '%',
            trend: 'up',
            change_percentage: 1.8
          }
        ],
        recent_optimizations: [
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
            optimization_type: 'cache_optimization',
            target_metric: 'response_time',
            improvement: -0.15,
            success: true
          },
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 120).toISOString(),
            optimization_type: 'algorithm_tuning',
            target_metric: 'accuracy',
            improvement: 0.03,
            success: true
          },
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 180).toISOString(),
            optimization_type: 'resource_allocation',
            target_metric: 'efficiency',
            improvement: 0.08,
            success: true
          }
        ],
        optimization_status: {
          active_targets: 3,
          queued_actions: 1,
          success_rate: 0.89,
          current_strategy: 'adaptive_hybrid'
        },
        performance_history: {
          timestamps: Array.from({ length: 24 }, (_, i) => 
            new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString()
          ),
          response_time: Array.from({ length: 24 }, () => 
            Math.random() * 0.5 + 1.0
          ),
          accuracy: Array.from({ length: 24 }, () => 
            Math.random() * 0.1 + 0.8
          ),
          efficiency: Array.from({ length: 24 }, () => 
            Math.random() * 0.15 + 0.7
          )
        }
      };
      
      setPerformanceData(mockData);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch performance data');
      setLoading(false);
    }
  };

  const triggerOptimization = async () => {
    setTriggeringOptimization(true);
    try {
      // In a real implementation, this would trigger actual optimization
      await new Promise(resolve => setTimeout(resolve, 2000));
      await fetchPerformanceData();
    } catch (err) {
      setError('Failed to trigger optimization');
    } finally {
      setTriggeringOptimization(false);
    }
  };

  const formatValue = (value: number, unit: string) => {
    if (unit === '%') {
      return `${(value * 100).toFixed(1)}%`;
    } else if (unit === 'seconds') {
      return `${value.toFixed(2)}s`;
    }
    return value.toFixed(2);
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return '↗️';
      case 'down': return '↘️';
      case 'stable': return '➡️';
      default: return '—';
    }
  };

  const getTrendColor = (trend: string, metricName: string) => {
    // For response time and resource usage, down is good
    const lowerIsBetter = metricName.toLowerCase().includes('time') || 
                         metricName.toLowerCase().includes('utilization') ||
                         metricName.toLowerCase().includes('usage');
    
    if (trend === 'up') {
      return lowerIsBetter ? 'text-red-600' : 'text-green-600';
    } else if (trend === 'down') {
      return lowerIsBetter ? 'text-green-600' : 'text-red-600';
    }
    return 'text-gray-600';
  };

  const getMetricStatus = (current: number, target: number, metricName: string) => {
    const lowerIsBetter = metricName.toLowerCase().includes('time') || 
                         metricName.toLowerCase().includes('utilization') ||
                         metricName.toLowerCase().includes('usage');
    
    const isGood = lowerIsBetter ? current <= target : current >= target;
    return isGood;
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3, 4, 5, 6].map(i => (
              <Card key={i}>
                <div className="h-24 bg-gray-200 rounded"></div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="border-red-200 bg-red-50">
        <CardContent>
          <div className="text-red-600">Error: {error}</div>
          <Button onClick={fetchPerformanceData} className="mt-2 bg-red-500 hover:bg-red-600">
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!performanceData) {
    return (
      <Card>
        <CardContent>
          <div className="text-gray-600">No performance data available</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Controls */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-900">⚡ Performance Metrics</h2>
        <div className="flex items-center space-x-4">
          <Select 
            value={timeRange} 
            onChange={(e: any) => setTimeRange(e.target.value)}
            className="text-sm"
          >
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
          </Select>
          <Button 
            onClick={triggerOptimization}
            disabled={triggeringOptimization}
            className="bg-green-500 hover:bg-green-600"
          >
            {triggeringOptimization ? 'Optimizing...' : '🚀 Trigger Optimization'}
          </Button>
        </div>
      </div>

      {/* Current Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {performanceData.current_metrics.map((metric, index) => {
          const isGood = getMetricStatus(metric.current_value, metric.target_value, metric.name);
          
          return (
            <Card key={index} className={isGood ? 'border-green-200' : 'border-yellow-200'}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{metric.name}</span>
                  <div className={`text-lg ${getTrendColor(metric.trend, metric.name)}`}>
                    {getTrendIcon(metric.trend)}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex items-end justify-between">
                    <div className="text-2xl font-bold">
                      {formatValue(metric.current_value, metric.unit)}
                    </div>
                    <Badge className={metric.change_percentage >= 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                      {metric.change_percentage >= 0 ? '+' : ''}{metric.change_percentage.toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="text-xs text-gray-500">
                    Target: {formatValue(metric.target_value, metric.unit)}
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        isGood ? 'bg-green-500' : 'bg-yellow-500'
                      }`}
                      style={{ 
                        width: `${Math.min(100, (metric.current_value / metric.target_value) * 100)}%` 
                      }}
                    ></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Optimization Status */}
      <Card>
        <CardHeader>🎯 Optimization Status</CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {performanceData.optimization_status.active_targets}
              </div>
              <div className="text-sm text-gray-500">Active Targets</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {performanceData.optimization_status.queued_actions}
              </div>
              <div className="text-sm text-gray-500">Queued Actions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {(performanceData.optimization_status.success_rate * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">Success Rate</div>
            </div>
            <div className="text-center">
              <Badge className="bg-purple-100 text-purple-800 text-sm">
                {performanceData.optimization_status.current_strategy.replace(/_/g, ' ')}
              </Badge>
              <div className="text-sm text-gray-500 mt-1">Strategy</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recent Optimizations */}
      <Card>
        <CardHeader>📈 Recent Optimizations</CardHeader>
        <CardContent>
          <div className="space-y-3">
            {performanceData.recent_optimizations.map((optimization, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    optimization.success ? 'bg-green-500' : 'bg-red-500'
                  }`}></div>
                  <div>
                    <div className="font-medium capitalize">
                      {optimization.optimization_type.replace(/_/g, ' ')}
                    </div>
                    <div className="text-sm text-gray-500">
                      Target: {optimization.target_metric.replace(/_/g, ' ')}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`font-medium ${
                    optimization.improvement > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {optimization.improvement > 0 ? '+' : ''}{(optimization.improvement * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">
                    {new Date(optimization.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Performance History Chart Placeholder */}
      <Card>
        <CardHeader>📊 Performance History ({timeRange})</CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-2">📈</div>
            <div>Performance chart visualization would be displayed here</div>
            <div className="text-sm mt-2">
              Showing trends for response time, accuracy, and efficiency over {timeRange}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
