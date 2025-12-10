/**
 * Provider Health Monitor - Real-time health monitoring and failover management
 * Tracks uptime, response times, error rates, and regional availability
 */

"use client";

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, 
  Wifi, 
  WifiOff, 
  AlertTriangle, 
  CheckCircle,
  Clock,
  BarChart3,
  Globe,
  Zap,
  Target,
  RefreshCw,
  Settings,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  Map
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/lib/session-store';

interface HealthMetric {
  timestamp: number;
  value: number;
  status: 'good' | 'warning' | 'error';
}

interface RegionStatus {
  region: string;
  status: 'online' | 'offline' | 'error';
  response_time: number;
  uptime: number;
}

const PROVIDER_REGIONS = {
  'anthropic': ['us-east-1', 'us-west-2', 'eu-west-1'],
  'openai': ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
  'google': ['us-central1', 'us-east1', 'europe-west1', 'asia-southeast1'],
  'groq': ['us-east-1', 'us-west-2']
};

const PROVIDER_LOGOS = {
  'anthropic': '🧠',
  'openai': '🤖',
  'google': '🔍',
  'groq': '⚡'
};

export const ProviderHealth: React.FC = () => {
  const { providers, providerHealth, actions } = useSessionStore();
  const [healthHistory, setHealthHistory] = useState<Record<string, HealthMetric[]>>({});
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'regional'>('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    // Simulate real-time health updates
    const generateHealthHistory = () => {
      const now = Date.now();
      const history: Record<string, HealthMetric[]> = {};
      
      providers.forEach(provider => {
        const metrics: HealthMetric[] = [];
        for (let i = 23; i >= 0; i--) {
          const timestamp = now - (i * 60 * 60 * 1000); // Hourly data points
          const baseResponseTime = provider.avg_response_time;
          const variation = (Math.random() - 0.5) * 0.3; // ±30% variation
          const responseTime = Math.max(100, baseResponseTime + (baseResponseTime * variation));
          
          let status: 'good' | 'warning' | 'error' = 'good';
          if (responseTime > baseResponseTime * 1.5) status = 'warning';
          if (responseTime > baseResponseTime * 2) status = 'error';
          if (Math.random() < 0.1) status = 'error'; // Random outages
          
          metrics.push({
            timestamp,
            value: responseTime,
            status
          });
        }
        history[provider.id] = metrics;
      });
      
      setHealthHistory(history);
    };

    generateHealthHistory();
    
    if (autoRefresh) {
      const interval = setInterval(() => {
        actions.loadProviderHealth();
        generateHealthHistory();
      }, 30000); // Update every 30 seconds
      
      return () => clearInterval(interval);
    }
  }, [providers, autoRefresh]);

  const getStatusIcon = (status: string, size: 'sm' | 'md' | 'lg' = 'md') => {
    const sizeClass = size === 'sm' ? 'h-3 w-3' : size === 'lg' ? 'h-6 w-6' : 'h-4 w-4';
    
    switch (status) {
      case 'online':
        return <CheckCircle className={cn(sizeClass, 'text-success')} />;
      case 'error':
        return <AlertTriangle className={cn(sizeClass, 'text-warning')} />;
      case 'offline':
      case 'error':
        return <AlertCircle className={cn(sizeClass, 'text-destructive')} />;
      default:
        return <Minus className={cn(sizeClass, 'text-muted-foreground')} />;
    }
  };

  const getUptimeColor = (uptime: number) => {
    if (uptime >= 99.9) return 'text-success';
    if (uptime >= 99.0) return 'text-warning';
    return 'text-destructive';
  };

  const getResponseTimeColor = (current: number, baseline: number) => {
    const ratio = current / baseline;
    if (ratio <= 1.2) return 'text-success';
    if (ratio <= 1.5) return 'text-warning';
    return 'text-destructive';
  };

  const formatUptime = (uptime: number) => `${uptime.toFixed(2)}%`;
  const formatResponseTime = (time: number) => `${Math.round(time)}ms`;

  const renderHealthChart = (providerId: string) => {
    const metrics = healthHistory[providerId] || [];
    if (metrics.length === 0) return null;

    const maxValue = Math.max(...metrics.map(m => m.value));
    
    return (
      <div className="flex items-end space-x-1 h-16">
        {metrics.map((metric, index) => {
          const height = (metric.value / maxValue) * 100;
          const color = metric.status === 'good' ? 'bg-success' : 
                       metric.status === 'warning' ? 'bg-warning' : 'bg-destructive';
          
          return (
            <motion.div
              key={index}
              initial={{ height: 0 }}
              animate={{ height: `${height}%` }}
              transition={{ delay: index * 0.05 }}
              className={cn(
                'w-2 rounded-t-sm opacity-80 hover:opacity-100 transition-opacity',
                color
              )}
              title={`${new Date(metric.timestamp).toLocaleTimeString()}: ${formatResponseTime(metric.value)}`}
            />
          );
        })}
      </div>
    );
  };

  const renderRegionalStatus = (providerId: string) => {
    const regions = PROVIDER_REGIONS[providerId as keyof typeof PROVIDER_REGIONS] || [];
    
    return (
      <div className="space-y-3">
        {regions.map(region => {
          // Mock regional data
          const baseHealth = providerHealth.find(h => h.provider_id === providerId);
          const regionHealth: RegionStatus = {
            region,
            status: Math.random() > 0.1 ? 'online' : Math.random() > 0.5 ? 'error' : 'offline',
            response_time: (baseHealth?.response_time || 1000) + (Math.random() - 0.5) * 200,
            uptime: 99.0 + Math.random() * 1.0
          };
          
          return (
            <div key={region} className="flex items-center justify-between p-3 rounded-lg border bg-card">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(regionHealth.status, 'sm')}
                  <span className="font-medium">{region}</span>
                </div>
                <Badge variant="outline" className="text-xs">
                  {regionHealth.region}
                </Badge>
              </div>
              
              <div className="flex items-center space-x-4 text-sm">
                <div className="text-center">
                  <div className={getUptimeColor(regionHealth.uptime)}>
                    {formatUptime(regionHealth.uptime)}
                  </div>
                  <div className="text-xs text-muted-foreground">Uptime</div>
                </div>
                <div className="text-center">
                  <div className={getResponseTimeColor(regionHealth.response_time, baseHealth?.response_time || 1000)}>
                    {formatResponseTime(regionHealth.response_time)}
                  </div>
                  <div className="text-xs text-muted-foreground">Response</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const overallStatus = {
    online: providerHealth.filter(h => h.status === 'online').length,
    error: providerHealth.filter(h => h.status === 'error').length,
    offline: providerHealth.filter(h => h.status === 'offline').length,
    total: providerHealth.length
  };

  const averageUptime = providerHealth.length > 0 
    ? providerHealth.reduce((sum, h) => sum + (h.uptime || 0), 0) / providerHealth.length 
    : 0;

  const averageResponseTime = providerHealth.length > 0
    ? providerHealth.reduce((sum, h) => sum + h.response_time, 0) / providerHealth.length
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <Activity className="h-6 w-6" />
            <span>Provider Health Monitor</span>
          </h2>
          <p className="text-muted-foreground">Real-time monitoring and failover management</p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={autoRefresh ? 'bg-success/10 text-success border-success/20' : ''}
          >
            <RefreshCw className={cn('h-4 w-4 mr-2', autoRefresh && 'animate-spin')} />
            {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Configure Alerts
          </Button>
        </div>
      </div>

      {/* Overall Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900 border-green-200 dark:border-green-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-green-700 dark:text-green-300">Online</p>
                <p className="text-2xl font-bold text-green-900 dark:text-green-100">{overallStatus.online}</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gradient-to-br from-yellow-50 to-yellow-100 dark:from-yellow-950 dark:to-yellow-900 border-yellow-200 dark:border-yellow-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-yellow-700 dark:text-yellow-300">Degraded</p>
                <p className="text-2xl font-bold text-yellow-900 dark:text-yellow-100">{overallStatus.error}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900 border-blue-200 dark:border-blue-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-blue-700 dark:text-blue-300">Avg Uptime</p>
                <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">{formatUptime(averageUptime)}</p>
              </div>
              <BarChart3 className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950 dark:to-purple-900 border-purple-200 dark:border-purple-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-purple-700 dark:text-purple-300">Avg Response</p>
                <p className="text-2xl font-bold text-purple-900 dark:text-purple-100">{formatResponseTime(averageResponseTime)}</p>
              </div>
              <Clock className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={viewMode} onValueChange={(value: any) => setViewMode(value)}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="detailed">Detailed</TabsTrigger>
          <TabsTrigger value="regional">Regional</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {providers.map(provider => {
              const health = providerHealth.find(h => h.provider_id === provider.id);
              if (!health) return null;

              return (
                <motion.div
                  key={provider.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <Card className={cn(
                    "transition-all duration-200 hover:shadow-lg cursor-pointer",
                    selectedProvider === provider.id && "ring-2 ring-primary"
                  )}
                  onClick={() => setSelectedProvider(selectedProvider === provider.id ? null : provider.id)}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center space-x-3">
                          <span className="text-2xl">
                            {PROVIDER_LOGOS[provider.id as keyof typeof PROVIDER_LOGOS] || '🤖'}
                          </span>
                          <span>{provider.name}</span>
                        </CardTitle>
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(health.status)}
                          <Badge variant={health.status === 'online' ? 'default' : 'secondary'}>
                            {health.status}
                          </Badge>
                        </div>
                      </div>
                    </CardHeader>
                    
                    <CardContent className="space-y-4">
                      {/* Health Metrics */}
                      <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                          <div className={cn("text-lg font-semibold", getUptimeColor(health.uptime || 0))}>
                            {formatUptime(health.uptime || 0)}
                          </div>
                          <div className="text-xs text-muted-foreground">Uptime</div>
                        </div>
                        <div>
                          <div className={cn("text-lg font-semibold", getResponseTimeColor(health.response_time, provider.avg_response_time))}>
                            {formatResponseTime(health.response_time)}
                          </div>
                          <div className="text-xs text-muted-foreground">Response</div>
                        </div>
                        <div>
                          <div className="text-lg font-semibold">
                            {health.region}
                          </div>
                          <div className="text-xs text-muted-foreground">Region</div>
                        </div>
                      </div>
                      
                      {/* Health History Chart */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>24h Response Times</span>
                          <span className="text-xs text-muted-foreground">
                            Last check: {new Date(health.last_check).toLocaleTimeString()}
                          </span>
                        </div>
                        {renderHealthChart(provider.id)}
                      </div>
                      
                      {/* Status Progress */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>Health Score</span>
                          <span>{Math.round((health.uptime || 0))}%</span>
                        </div>
                        <Progress 
                          value={health.uptime || 0} 
                          className="h-2"
                        />
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="detailed" className="space-y-4">
          {selectedProvider ? (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">
                    {PROVIDER_LOGOS[selectedProvider as keyof typeof PROVIDER_LOGOS] || '🤖'}
                  </span>
                  <span>Detailed Health Metrics</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="font-medium">Performance Metrics</h4>
                    {renderHealthChart(selectedProvider)}
                  </div>
                  <div className="space-y-4">
                    <h4 className="font-medium">Regional Status</h4>
                    {renderRegionalStatus(selectedProvider)}
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="text-center py-12">
              <Target className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">Select a Provider</h3>
              <p className="text-muted-foreground">Choose a provider from the overview to see detailed metrics</p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="regional" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {providers.map(provider => (
              <Card key={provider.id}>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <span className="text-xl">
                      {PROVIDER_LOGOS[provider.id as keyof typeof PROVIDER_LOGOS] || '🤖'}
                    </span>
                    <span>{provider.name} Regions</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {renderRegionalStatus(provider.id)}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};