// Reusable Metrics Chart Component for Analytics Dashboard
import React, { useState, useEffect, useMemo } from 'react';
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  Legend, PieChart, Pie, Cell, ScatterChart, Scatter
} from 'recharts';
import { TrendingUp, TrendingDown, MoreHorizontal, Download, Maximize2, Minimize2 } from 'lucide-react';
import { Card } from '../ui/Card';

interface MetricDataPoint {
  timestamp: number;
  value: number;
  label?: string;
  category?: string;
  metadata?: Record<string, any>;
}

interface MetricsChartProps {
  title: string;
  data: MetricDataPoint[];
  chartType?: 'line' | 'area' | 'bar' | 'pie' | 'scatter';
  unit?: string;
  color?: string;
  height?: number;
  showLegend?: boolean;
  showGrid?: boolean;
  showTrend?: boolean;
  isRealTime?: boolean;
  refreshInterval?: number;
  onDataRefresh?: () => Promise<MetricDataPoint[]>;
  formatValue?: (value: number) => string;
  formatTimestamp?: (timestamp: number) => string;
  className?: string;
  enableZoom?: boolean;
  enableExport?: boolean;
  threshold?: {
    value: number;
    label: string;
    color: string;
  };
  multiSeries?: {
    key: string;
    name: string;
    color: string;
  }[];
}

interface ChartState {
  isExpanded: boolean;
  selectedTimeRange: string;
  isLoading: boolean;
  error: string | null;
}

const DEFAULT_COLORS = {
  primary: '#3B82F6',
  success: '#10B981',
  warning: '#F59E0B',
  danger: '#EF4444',
  purple: '#8B5CF6',
  cyan: '#06B6D4'
};

const TIME_RANGES = [
  { value: '1h', label: '1 Hour', hours: 1 },
  { value: '6h', label: '6 Hours', hours: 6 },
  { value: '24h', label: '24 Hours', hours: 24 },
  { value: '7d', label: '7 Days', hours: 168 },
  { value: '30d', label: '30 Days', hours: 720 }
];

export const MetricsChart: React.FC<MetricsChartProps> = ({
  title,
  data,
  chartType = 'line',
  unit = '',
  color = DEFAULT_COLORS.primary,
  height = 300,
  showLegend = false,
  showGrid = true,
  showTrend = true,
  isRealTime = false,
  refreshInterval = 30000,
  onDataRefresh,
  formatValue,
  formatTimestamp,
  className = '',
  enableZoom = false,
  enableExport = false,
  threshold,
  multiSeries
}) => {
  const [chartState, setChartState] = useState<ChartState>({
    isExpanded: false,
    selectedTimeRange: '24h',
    isLoading: false,
    error: null
  });

  // Real-time data refresh
  useEffect(() => {
    if (!isRealTime || !onDataRefresh) return;

    const interval = setInterval(async () => {
      try {
        setChartState(prev => ({ ...prev, isLoading: true, error: null }));
        await onDataRefresh();
        setChartState(prev => ({ ...prev, isLoading: false }));
      } catch (error) {
        setChartState(prev => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : 'Failed to refresh data'
        }));
      }
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [isRealTime, onDataRefresh, refreshInterval]);

  // Calculate trend
  const trendData = useMemo(() => {
    if (!showTrend || data.length < 2) {
      return { direction: 'stable', percentage: 0, isPositive: true };
    }

    const recent = data.slice(-10); // Last 10 points
    const older = data.slice(-20, -10); // Previous 10 points

    if (recent.length === 0 || older.length === 0) {
      return { direction: 'stable', percentage: 0, isPositive: true };
    }

    const recentAvg = recent.reduce((sum, point) => sum + point.value, 0) / recent.length;
    const olderAvg = older.reduce((sum, point) => sum + point.value, 0) / older.length;

    const percentage = olderAvg !== 0 ? ((recentAvg - olderAvg) / olderAvg) * 100 : 0;
    const direction = Math.abs(percentage) < 1 ? 'stable' : percentage > 0 ? 'up' : 'down';
    const isPositive = percentage >= 0;

    return { direction, percentage: Math.abs(percentage), isPositive };
  }, [data, showTrend]);

  // Format data for different chart types
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    return data.map(point => ({
      ...point,
      timestamp: point.timestamp,
      value: point.value,
      formattedTime: formatTimestamp ? 
        formatTimestamp(point.timestamp) : 
        new Date(point.timestamp).toLocaleTimeString(),
      formattedValue: formatValue ? 
        formatValue(point.value) : 
        `${point.value.toFixed(2)}${unit ? ` ${unit}` : ''}`
    }));
  }, [data, formatValue, formatTimestamp, unit]);

  // Multi-series data preparation
  const multiSeriesData = useMemo(() => {
    if (!multiSeries || !data.length) return chartData;

    // Group data by timestamp and series
    const grouped = data.reduce((acc, point) => {
      const key = point.timestamp;
      if (!acc[key]) {
        acc[key] = {
          timestamp: key,
          formattedTime: formatTimestamp ? 
            formatTimestamp(key) : 
            new Date(key).toLocaleTimeString()
        };
      }
      
      const seriesKey = point.category || 'default';
      acc[key][seriesKey] = point.value;
      
      return acc;
    }, {} as Record<number, any>);

    return Object.values(grouped);
  }, [data, multiSeries, formatTimestamp]);

  // Export functionality
  const exportData = () => {
    const csvData = [
      ['Timestamp', 'Value', 'Formatted Time', 'Formatted Value'],
      ...chartData.map(point => [
        point.timestamp,
        point.value,
        point.formattedTime,
        point.formattedValue
      ])
    ];
    
    const csvContent = csvData.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(/\s+/g, '-')}-metrics.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    return (
      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
        <p className="text-sm font-medium text-gray-900">
          {formatTimestamp ? formatTimestamp(label) : new Date(label).toLocaleString()}
        </p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: {formatValue ? formatValue(entry.value) : `${entry.value.toFixed(2)}${unit ? ` ${unit}` : ''}`}
          </p>
        ))}
      </div>
    );
  };

  // Render chart based on type
  const renderChart = () => {
    const commonProps = {
      data: multiSeries ? multiSeriesData : chartData,
      width: '100%',
      height: chartState.isExpanded ? height * 1.5 : height
    };

    switch (chartType) {
      case 'area':
        return (
          <ResponsiveContainer {...commonProps}>
            <AreaChart data={commonProps.data}>
              {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />}
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => formatTimestamp ? formatTimestamp(value) : new Date(value).toLocaleTimeString()}
                stroke="#6b7280"
              />
              <YAxis stroke="#6b7280" />
              <Tooltip content={<CustomTooltip />} />
              {showLegend && <Legend />}
              {multiSeries ? (
                multiSeries.map((series, index) => (
                  <Area
                    key={series.key}
                    type="monotone"
                    dataKey={series.key}
                    stroke={series.color}
                    fill={series.color}
                    fillOpacity={0.3}
                    name={series.name}
                  />
                ))
              ) : (
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={color}
                  fill={color}
                  fillOpacity={0.3}
                  name={title}
                />
              )}
              {threshold && (
                <Line
                  type="monotone"
                  dataKey={() => threshold.value}
                  stroke={threshold.color}
                  strokeDasharray="5 5"
                  name={threshold.label}
                  dot={false}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        );

      case 'bar':
        return (
          <ResponsiveContainer {...commonProps}>
            <BarChart data={commonProps.data}>
              {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />}
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => formatTimestamp ? formatTimestamp(value) : new Date(value).toLocaleTimeString()}
                stroke="#6b7280"
              />
              <YAxis stroke="#6b7280" />
              <Tooltip content={<CustomTooltip />} />
              {showLegend && <Legend />}
              {multiSeries ? (
                multiSeries.map((series, index) => (
                  <Bar
                    key={series.key}
                    dataKey={series.key}
                    fill={series.color}
                    name={series.name}
                  />
                ))
              ) : (
                <Bar dataKey="value" fill={color} name={title} />
              )}
            </BarChart>
          </ResponsiveContainer>
        );

      case 'pie':
        const pieData = chartData.map((point, index) => ({
          name: point.label || `Point ${index + 1}`,
          value: point.value,
          fill: multiSeries ? multiSeries[index % multiSeries.length]?.color || color : color
        }));

        return (
          <ResponsiveContainer {...commonProps}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
              {showLegend && <Legend />}
            </PieChart>
          </ResponsiveContainer>
        );

      case 'scatter':
        return (
          <ResponsiveContainer {...commonProps}>
            <ScatterChart data={commonProps.data}>
              {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />}
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => formatTimestamp ? formatTimestamp(value) : new Date(value).toLocaleTimeString()}
                stroke="#6b7280"
              />
              <YAxis stroke="#6b7280" />
              <Tooltip content={<CustomTooltip />} />
              {showLegend && <Legend />}
              <Scatter name={title} data={commonProps.data} fill={color} />
            </ScatterChart>
          </ResponsiveContainer>
        );

      default: // 'line'
        return (
          <ResponsiveContainer {...commonProps}>
            <LineChart data={commonProps.data}>
              {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />}
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => formatTimestamp ? formatTimestamp(value) : new Date(value).toLocaleTimeString()}
                stroke="#6b7280"
              />
              <YAxis stroke="#6b7280" />
              <Tooltip content={<CustomTooltip />} />
              {showLegend && <Legend />}
              {multiSeries ? (
                multiSeries.map((series, index) => (
                  <Line
                    key={series.key}
                    type="monotone"
                    dataKey={series.key}
                    stroke={series.color}
                    strokeWidth={2}
                    dot={{ fill: series.color, strokeWidth: 2, r: 4 }}
                    name={series.name}
                  />
                ))
              ) : (
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={color}
                  strokeWidth={2}
                  dot={{ fill: color, strokeWidth: 2, r: 4 }}
                  name={title}
                />
              )}
              {threshold && (
                <Line
                  type="monotone"
                  dataKey={() => threshold.value}
                  stroke={threshold.color}
                  strokeDasharray="5 5"
                  name={threshold.label}
                  dot={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        );
    }
  };

  if (!data || data.length === 0) {
    return (
      <Card className={`p-6 ${className}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">{title}</h3>
        </div>
        <div className="flex items-center justify-center h-48 text-gray-500">
          <div className="text-center">
            <div className="w-12 h-12 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
              <TrendingUp className="h-6 w-6" />
            </div>
            <p>No data available</p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className={`p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <h3 className="text-lg font-semibold">{title}</h3>
          {isRealTime && (
            <div className="flex items-center space-x-1">
              <div className={`w-2 h-2 rounded-full ${
                chartState.isLoading ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'
              }`}></div>
              <span className="text-xs text-gray-500">Live</span>
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          {/* Trend indicator */}
          {showTrend && (
            <div className="flex items-center space-x-1 text-sm">
              {trendData.direction === 'up' ? (
                <TrendingUp className="h-4 w-4 text-green-500" />
              ) : trendData.direction === 'down' ? (
                <TrendingDown className="h-4 w-4 text-red-500" />
              ) : (
                <div className="h-4 w-4 bg-gray-400 rounded-full" />
              )}
              <span className={`font-medium ${
                trendData.direction === 'up' ? 'text-green-600' :
                trendData.direction === 'down' ? 'text-red-600' :
                'text-gray-600'
              }`}>
                {trendData.direction === 'stable' ? 'Stable' : `${trendData.percentage.toFixed(1)}%`}
              </span>
            </div>
          )}
          
          {/* Action buttons */}
          <div className="flex items-center space-x-1">
            {enableExport && (
              <button
                onClick={exportData}
                className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
                title="Export data"
              >
                <Download className="h-4 w-4" />
              </button>
            )}
            
            <button
              onClick={() => setChartState(prev => ({ ...prev, isExpanded: !prev.isExpanded }))}
              className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
              title={chartState.isExpanded ? "Minimize" : "Maximize"}
            >
              {chartState.isExpanded ? (
                <Minimize2 className="h-4 w-4" />
              ) : (
                <Maximize2 className="h-4 w-4" />
              )}
            </button>
            
            <button className="p-1 text-gray-400 hover:text-gray-600 transition-colors">
              <MoreHorizontal className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Error state */}
      {chartState.error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">{chartState.error}</p>
        </div>
      )}

      {/* Current value display */}
      {chartData.length > 0 && (
        <div className="mb-4">
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold">
              {formatValue ? 
                formatValue(chartData[chartData.length - 1].value) : 
                chartData[chartData.length - 1].value.toFixed(2)
              }
            </span>
            {unit && <span className="text-sm text-gray-500">{unit}</span>}
          </div>
          <p className="text-xs text-gray-500">
            Last updated: {chartData[chartData.length - 1].formattedTime}
          </p>
        </div>
      )}

      {/* Chart */}
      <div className="relative">
        {chartState.isLoading && (
          <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center z-10">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        )}
        {renderChart()}
      </div>

      {/* Threshold info */}
      {threshold && (
        <div className="mt-4 flex items-center space-x-2 text-sm text-gray-600">
          <div className="w-3 h-px" style={{ backgroundColor: threshold.color }}></div>
          <span>{threshold.label}: {formatValue ? formatValue(threshold.value) : threshold.value}</span>
        </div>
      )}
    </Card>
  );
};

// Specialized chart components
export const SystemMetricChart: React.FC<Omit<MetricsChartProps, 'threshold'>> = (props) => {
  const threshold = useMemo(() => {
    if (props.title.toLowerCase().includes('cpu')) {
      return { value: 80, label: 'High Usage Threshold', color: '#EF4444' };
    }
    if (props.title.toLowerCase().includes('memory')) {
      return { value: 85, label: 'High Usage Threshold', color: '#EF4444' };
    }
    if (props.title.toLowerCase().includes('disk')) {
      return { value: 90, label: 'High Usage Threshold', color: '#EF4444' };
    }
    return undefined;
  }, [props.title]);

  return <MetricsChart {...props} threshold={threshold} unit="%" />;
};

export const ResponseTimeChart: React.FC<Omit<MetricsChartProps, 'threshold' | 'formatValue'>> = (props) => {
  const formatValue = (value: number) => `${value.toFixed(0)}ms`;
  const threshold = { value: 2000, label: 'Target Response Time', color: '#F59E0B' };

  return (
    <MetricsChart 
      {...props} 
      threshold={threshold} 
      formatValue={formatValue}
      unit="ms"
      color={DEFAULT_COLORS.success}
    />
  );
};

export const ThroughputChart: React.FC<Omit<MetricsChartProps, 'formatValue'>> = (props) => {
  const formatValue = (value: number) => {
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    return value.toFixed(0);
  };

  return (
    <MetricsChart 
      {...props} 
      formatValue={formatValue}
      unit="req/s"
      color={DEFAULT_COLORS.purple}
      chartType="area"
    />
  );
};

export const CostChart: React.FC<Omit<MetricsChartProps, 'formatValue'>> = (props) => {
  const formatValue = (value: number) => `$${value.toFixed(2)}`;

  return (
    <MetricsChart 
      {...props} 
      formatValue={formatValue}
      unit="USD"
      color={DEFAULT_COLORS.success}
      chartType="line"
    />
  );
};
