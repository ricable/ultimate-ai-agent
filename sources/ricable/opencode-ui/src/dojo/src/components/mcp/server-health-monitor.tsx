/**
 * MCP Server Health Monitor - Real-time monitoring and alerting for MCP server health
 * Provides comprehensive health analytics, predictive alerts, and performance optimization
 */

"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  Zap,
  Cpu,
  MemoryStick,
  Network,
  AlertCircle,
  RefreshCw,
  Settings,
  Download,
  Filter,
  Calendar,
  BarChart3,
  LineChart,
  PieChart,
  Target,
  Shield,
  Bell,
  Thermometer,
  Gauge,
  RadioIcon,
  Wifi,
  WifiOff,
  Timer,
  Server as ServerIcon,
  Database as DatabaseIcon,
  HardDrive,
  Terminal,
  Globe,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  Info,
  CheckCircle2,
  Plus,
  Minus
} from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from '@/components/ui/dialog';
import { 
  Tooltip, 
  TooltipContent, 
  TooltipTrigger, 
  TooltipProvider 
} from '@/components/ui/tooltip';
import { Alert, AlertDescription } from '@/components/ui/alert';

import { 
  MCPServerWithStatus,
  MCPServerHealth,
  MCPServerMetrics,
  MCPHealthAlert,
  MCPHealthThreshold
} from '@/lib/types/mcp';
import { cn } from '@/lib/utils';

interface HealthMetrics {
  timestamp: number;
  responseTime: number;
  cpuUsage: number;
  memoryUsage: number;
  requestsPerMinute: number;
  errorRate: number;
  healthScore: number;
}

interface HealthAlert {
  id: string;
  serverId: string;
  serverName: string;
  type: 'critical' | 'warning' | 'info';
  metric: 'response_time' | 'cpu_usage' | 'memory_usage' | 'error_rate' | 'uptime' | 'connectivity';
  message: string;
  threshold: number;
  currentValue: number;
  timestamp: number;
  acknowledged: boolean;
  resolvedAt?: number;
}

interface HealthThreshold {
  metric: string;
  warning: number;
  critical: number;
  enabled: boolean;
}

interface ServerHealthMonitorProps {
  servers: MCPServerWithStatus[];
  onRefresh: () => void;
  onConfigureThresholds: (thresholds: HealthThreshold[]) => void;
  refreshing?: boolean;
}

export function ServerHealthMonitor({
  servers,
  onRefresh,
  onConfigureThresholds,
  refreshing = false
}: ServerHealthMonitorProps) {
  const [alerts, setAlerts] = useState<HealthAlert[]>([]);
  const [thresholds, setThresholds] = useState<HealthThreshold[]>([
    { metric: 'response_time', warning: 2000, critical: 5000, enabled: true },
    { metric: 'cpu_usage', warning: 70, critical: 90, enabled: true },
    { metric: 'memory_usage', warning: 80, critical: 95, enabled: true },
    { metric: 'error_rate', warning: 5, critical: 10, enabled: true },
    { metric: 'uptime', warning: 95, critical: 90, enabled: true }
  ]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('6h');
  const [showAcknowledgedAlerts, setShowAcknowledgedAlerts] = useState(false);
  const [selectedServer, setSelectedServer] = useState<string | null>(null);
  const [compactView, setCompactView] = useState(false);

  // Auto-refresh logic
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      onRefresh();
      checkThresholds();
    }, refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, onRefresh]);

  // Initialize threshold checking
  useEffect(() => {
    checkThresholds();
  }, [servers, thresholds]);

  // Generate mock historical data for demonstration
  const generateMockHistoricalData = (serverId: string, hours: number): HealthMetrics[] => {
    const data: HealthMetrics[] = [];
    const now = Date.now();
    const intervalMs = (hours * 60 * 60 * 1000) / 100; // 100 points

    for (let i = 0; i < 100; i++) {
      const timestamp = now - (99 - i) * intervalMs;
      data.push({
        timestamp,
        responseTime: 500 + Math.random() * 2000 + Math.sin(i / 10) * 300,
        cpuUsage: 20 + Math.random() * 60 + Math.sin(i / 15) * 15,
        memoryUsage: 40 + Math.random() * 40 + Math.sin(i / 8) * 10,
        requestsPerMinute: 10 + Math.random() * 50 + Math.sin(i / 12) * 20,
        errorRate: Math.random() * 10 + Math.sin(i / 20) * 3,
        healthScore: 75 + Math.random() * 20 + Math.sin(i / 18) * 15
      });
    }

    return data;
  };

  // Check thresholds and generate alerts
  const checkThresholds = () => {
    const newAlerts: HealthAlert[] = [];

    servers.forEach(server => {
      thresholds.forEach(threshold => {
        if (!threshold.enabled) return;

        let currentValue: number | undefined;
        let metric = threshold.metric;

        switch (metric) {
          case 'response_time':
            currentValue = server.health.response_time;
            break;
          case 'cpu_usage':
            currentValue = server.status.cpu_usage;
            break;
          case 'memory_usage':
            currentValue = server.status.memory_usage ? 
              (server.status.memory_usage / (1024 * 1024 * 1024)) * 100 : undefined; // Convert to percentage
            break;
          case 'error_rate':
            currentValue = server.metrics.requests_total > 0 ? 
              ((server.metrics.requests_total - server.metrics.requests_successful) / server.metrics.requests_total) * 100 : 0;
            break;
          case 'uptime':
            currentValue = server.health.availability_percentage;
            break;
        }

        if (currentValue === undefined) return;

        const existingAlert = alerts.find(a => 
          a.serverId === server.id && 
          a.metric === metric && 
          !a.resolvedAt
        );

        if (currentValue >= threshold.critical) {
          if (!existingAlert || existingAlert.type !== 'critical') {
            newAlerts.push({
              id: `${server.id}-${metric}-${Date.now()}`,
              serverId: server.id,
              serverName: server.name,
              type: 'critical',
              metric: metric as any,
              message: `${metric.replace('_', ' ').toUpperCase()} exceeded critical threshold`,
              threshold: threshold.critical,
              currentValue,
              timestamp: Date.now(),
              acknowledged: false
            });
          }
        } else if (currentValue >= threshold.warning) {
          if (!existingAlert || existingAlert.type === 'info') {
            newAlerts.push({
              id: `${server.id}-${metric}-${Date.now()}`,
              serverId: server.id,
              serverName: server.name,
              type: 'warning',
              metric: metric as any,
              message: `${metric.replace('_', ' ').toUpperCase()} exceeded warning threshold`,
              threshold: threshold.warning,
              currentValue,
              timestamp: Date.now(),
              acknowledged: false
            });
          }
        } else if (existingAlert) {
          // Resolve existing alert if value is back to normal
          setAlerts(prev => prev.map(a => 
            a.id === existingAlert.id ? { ...a, resolvedAt: Date.now() } : a
          ));
        }
      });
    });

    if (newAlerts.length > 0) {
      setAlerts(prev => [...prev.slice(-100), ...newAlerts]); // Keep last 100 alerts
    }
  };

  // Calculate overall system health
  const systemHealth = useMemo(() => {
    if (servers.length === 0) return { score: 0, status: 'unknown' as const };

    const totalScore = servers.reduce((acc, server) => {
      let score = 100;
      
      // Response time impact
      if (server.health.response_time) {
        if (server.health.response_time > 5000) score -= 30;
        else if (server.health.response_time > 2000) score -= 15;
      }

      // Availability impact
      if (server.health.availability_percentage !== undefined) {
        if (server.health.availability_percentage < 90) score -= 40;
        else if (server.health.availability_percentage < 95) score -= 20;
      }

      // Error rate impact
      const errorRate = server.metrics.requests_total > 0 ? 
        ((server.metrics.requests_total - server.metrics.requests_successful) / server.metrics.requests_total) * 100 : 0;
      if (errorRate > 10) score -= 25;
      else if (errorRate > 5) score -= 10;

      // Connection status impact
      if (server.status.status !== 'connected') score -= 50;

      return acc + Math.max(0, score);
    }, 0);

    const averageScore = totalScore / servers.length;
    let status: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';

    if (averageScore >= 90) status = 'excellent';
    else if (averageScore >= 75) status = 'good';
    else if (averageScore >= 60) status = 'fair';
    else if (averageScore >= 40) status = 'poor';
    else status = 'critical';

    return { score: Math.round(averageScore), status };
  }, [servers]);

  // Get active alerts (unresolved and optionally unacknowledged)
  const activeAlerts = useMemo(() => {
    return alerts.filter(alert => 
      !alert.resolvedAt && 
      (showAcknowledgedAlerts || !alert.acknowledged)
    ).sort((a, b) => {
      // Sort by severity, then by timestamp
      const severityOrder = { critical: 0, warning: 1, info: 2 };
      const severityDiff = severityOrder[a.type] - severityOrder[b.type];
      if (severityDiff !== 0) return severityDiff;
      return b.timestamp - a.timestamp;
    });
  }, [alerts, showAcknowledgedAlerts]);

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  };

  const getHealthIcon = (score: number) => {
    if (score >= 90) return <CheckCircle className="h-5 w-5 text-green-500" />;
    if (score >= 75) return <CheckCircle2 className="h-5 w-5 text-blue-500" />;
    if (score >= 60) return <AlertCircle className="h-5 w-5 text-yellow-500" />;
    if (score >= 40) return <AlertTriangle className="h-5 w-5 text-orange-500" />;
    return <XCircle className="h-5 w-5 text-red-500" />;
  };

  const getHealthColor = (score: number) => {
    if (score >= 90) return 'text-green-600 bg-green-100';
    if (score >= 75) return 'text-blue-600 bg-blue-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    if (score >= 40) return 'text-orange-600 bg-orange-100';
    return 'text-red-600 bg-red-100';
  };

  const getAlertIcon = (type: HealthAlert['type']) => {
    switch (type) {
      case 'critical': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'info': return <Info className="h-4 w-4 text-blue-500" />;
    }
  };

  const formatBytes = (bytes?: number) => {
    if (!bytes) return "0 B";
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  const formatUptime = (uptime?: number) => {
    if (!uptime) return "N/A";
    const hours = Math.floor(uptime / 3600000);
    const minutes = Math.floor((uptime % 3600000) / 60000);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  return (
    <TooltipProvider>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">Server Health Monitor</h2>
            <p className="text-muted-foreground">
              Real-time monitoring and alerting for MCP server health
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCompactView(!compactView)}
            >
              {compactView ? <Maximize2 className="h-4 w-4" /> : <Minimize2 className="h-4 w-4" />}
            </Button>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <Settings className="h-4 w-4 mr-2" />
                  Configure
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Health Monitor Configuration</DialogTitle>
                </DialogHeader>
                <ThresholdConfiguration 
                  thresholds={thresholds}
                  onUpdate={setThresholds}
                  autoRefresh={autoRefresh}
                  refreshInterval={refreshInterval}
                  onAutoRefreshChange={setAutoRefresh}
                  onRefreshIntervalChange={setRefreshInterval}
                />
              </DialogContent>
            </Dialog>
            <Button
              variant="outline"
              onClick={onRefresh}
              disabled={refreshing}
            >
              <RefreshCw className={cn("h-4 w-4 mr-2", refreshing && "animate-spin")} />
              Refresh
            </Button>
          </div>
        </div>

        {/* System Health Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Activity className="h-5 w-5" />
                <span>System Health Overview</span>
              </div>
              <Badge className={cn("px-3 py-1", getHealthColor(systemHealth.score))}>
                {systemHealth.status.toUpperCase()} - {systemHealth.score}%
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center space-y-2">
                <div className="mx-auto w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                  <ServerIcon className="h-6 w-6 text-blue-600" />
                </div>
                <div className="text-2xl font-bold">{servers.length}</div>
                <div className="text-sm text-muted-foreground">Total Servers</div>
              </div>
              
              <div className="text-center space-y-2">
                <div className="mx-auto w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                </div>
                <div className="text-2xl font-bold text-green-600">
                  {servers.filter(s => s.status.status === 'connected').length}
                </div>
                <div className="text-sm text-muted-foreground">Connected</div>
              </div>
              
              <div className="text-center space-y-2">
                <div className="mx-auto w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                  <Bell className="h-6 w-6 text-red-600" />
                </div>
                <div className="text-2xl font-bold text-red-600">
                  {activeAlerts.filter(a => a.type === 'critical').length}
                </div>
                <div className="text-sm text-muted-foreground">Critical Alerts</div>
              </div>
              
              <div className="text-center space-y-2">
                <div className="mx-auto w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
                  <AlertTriangle className="h-6 w-6 text-yellow-600" />
                </div>
                <div className="text-2xl font-bold text-yellow-600">
                  {activeAlerts.filter(a => a.type === 'warning').length}
                </div>
                <div className="text-sm text-muted-foreground">Warnings</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className={cn("grid gap-6", compactView ? "grid-cols-1" : "grid-cols-1 lg:grid-cols-3")}>
          {/* Active Alerts */}
          <Card className={cn(compactView ? "" : "lg:col-span-2")}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Bell className="h-5 w-5" />
                  <span>Active Alerts</span>
                  {activeAlerts.length > 0 && (
                    <Badge variant="destructive">{activeAlerts.length}</Badge>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  <Label htmlFor="show-acknowledged" className="text-sm">
                    Show acknowledged
                  </Label>
                  <Switch
                    id="show-acknowledged"
                    checked={showAcknowledgedAlerts}
                    onCheckedChange={setShowAcknowledgedAlerts}
                  />
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-64">
                {activeAlerts.length === 0 ? (
                  <div className="text-center py-8">
                    <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
                    <h3 className="text-lg font-semibold mb-2">All systems healthy</h3>
                    <p className="text-muted-foreground">No active alerts at this time</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {activeAlerts.map((alert) => (
                      <motion.div
                        key={alert.id}
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={cn(
                          "p-3 rounded-lg border",
                          alert.type === 'critical' && "bg-red-50 border-red-200",
                          alert.type === 'warning' && "bg-yellow-50 border-yellow-200",
                          alert.type === 'info' && "bg-blue-50 border-blue-200",
                          alert.acknowledged && "opacity-60"
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            {getAlertIcon(alert.type)}
                            <div>
                              <div className="font-medium text-sm">
                                {alert.serverName}: {alert.message}
                              </div>
                              <div className="text-xs text-muted-foreground">
                                {alert.currentValue.toFixed(1)} / {alert.threshold} threshold
                                {' • '}
                                {new Date(alert.timestamp).toLocaleTimeString()}
                              </div>
                            </div>
                          </div>
                          {!alert.acknowledged && (
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => acknowledgeAlert(alert.id)}
                            >
                              <CheckCircle className="h-4 w-4" />
                            </Button>
                          )}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Quick Stats */}
          {!compactView && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5" />
                  <span>Quick Stats</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Avg Response Time</span>
                    <span className="font-medium">
                      {Math.round(servers.reduce((acc, s) => acc + (s.health.response_time || 0), 0) / servers.length || 0)}ms
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">System Uptime</span>
                    <span className="font-medium">
                      {Math.round(servers.reduce((acc, s) => acc + (s.health.availability_percentage || 0), 0) / servers.length || 0)}%
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Total Requests</span>
                    <span className="font-medium">
                      {servers.reduce((acc, s) => acc + s.metrics.requests_total, 0).toLocaleString()}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Success Rate</span>
                    <span className="font-medium">
                      {Math.round(servers.reduce((acc, s) => {
                        return acc + (s.metrics.requests_total > 0 ? (s.metrics.requests_successful / s.metrics.requests_total) * 100 : 100);
                      }, 0) / servers.length || 0)}%
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Server Health Grid */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Server Health Status</h3>
            <div className="flex items-center space-x-2">
              <Select value={selectedTimeRange} onValueChange={(value: any) => setSelectedTimeRange(value)}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1h">1 Hour</SelectItem>
                  <SelectItem value="6h">6 Hours</SelectItem>
                  <SelectItem value="24h">24 Hours</SelectItem>
                  <SelectItem value="7d">7 Days</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className={cn("grid gap-4", compactView ? "grid-cols-1" : "grid-cols-1 md:grid-cols-2 xl:grid-cols-3")}>
            {servers.map((server) => {
              const serverHealth = Math.max(0, Math.min(100, 100 - 
                (server.health.response_time ? Math.min(40, (server.health.response_time - 1000) / 100) : 0) -
                (server.status.cpu_usage ? Math.min(30, Math.max(0, server.status.cpu_usage - 70)) : 0) -
                (server.status.status !== 'connected' ? 50 : 0)
              ));

              return (
                <Card key={server.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      {/* Header */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {server.type === 'local' ? (
                            <Terminal className="h-4 w-4 text-blue-500" />
                          ) : (
                            <Globe className="h-4 w-4 text-green-500" />
                          )}
                          <div className="font-medium text-sm">{server.name}</div>
                        </div>
                        <div className="flex items-center space-x-1">
                          {getHealthIcon(serverHealth)}
                          <span className="text-sm font-medium">{Math.round(serverHealth)}%</span>
                        </div>
                      </div>

                      {/* Status and Metrics */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Status</span>
                          <Badge 
                            variant={server.status.status === 'connected' ? 'default' : 'destructive'}
                            className="text-xs px-2 py-0.5"
                          >
                            {server.status.status}
                          </Badge>
                        </div>

                        {server.health.response_time && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Response Time</span>
                            <span className={cn(
                              "font-medium",
                              server.health.response_time > 5000 ? "text-red-600" :
                              server.health.response_time > 2000 ? "text-yellow-600" : "text-green-600"
                            )}>
                              {server.health.response_time}ms
                            </span>
                          </div>
                        )}

                        {server.status.cpu_usage !== undefined && (
                          <div className="space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground">CPU Usage</span>
                              <span className="font-medium">{server.status.cpu_usage.toFixed(1)}%</span>
                            </div>
                            <Progress value={server.status.cpu_usage} className="h-1" />
                          </div>
                        )}

                        {server.status.memory_usage && (
                          <div className="space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground">Memory</span>
                              <span className="font-medium">{formatBytes(server.status.memory_usage)}</span>
                            </div>
                            <Progress value={(server.status.memory_usage / (1024 * 1024 * 1024)) * 10} className="h-1" />
                          </div>
                        )}

                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Uptime</span>
                          <span className="font-medium">{formatUptime(server.status.uptime)}</span>
                        </div>

                        {server.health.availability_percentage !== undefined && (
                          <div className="space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground">Availability</span>
                              <span className="font-medium">{server.health.availability_percentage.toFixed(1)}%</span>
                            </div>
                            <Progress value={server.health.availability_percentage} className="h-1" />
                          </div>
                        )}
                      </div>

                      {/* Server-specific alerts */}
                      {activeAlerts.filter(a => a.serverId === server.id).length > 0 && (
                        <div className="pt-2 border-t">
                          <div className="text-xs text-muted-foreground mb-1">Active Alerts</div>
                          <div className="space-y-1">
                            {activeAlerts.filter(a => a.serverId === server.id).slice(0, 2).map(alert => (
                              <div key={alert.id} className="flex items-center space-x-1 text-xs">
                                {getAlertIcon(alert.type)}
                                <span className="truncate">{alert.message}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}

// Threshold Configuration Component
function ThresholdConfiguration({
  thresholds,
  onUpdate,
  autoRefresh,
  refreshInterval,
  onAutoRefreshChange,
  onRefreshIntervalChange
}: {
  thresholds: HealthThreshold[];
  onUpdate: (thresholds: HealthThreshold[]) => void;
  autoRefresh: boolean;
  refreshInterval: number;
  onAutoRefreshChange: (enabled: boolean) => void;
  onRefreshIntervalChange: (interval: number) => void;
}) {
  const updateThreshold = (index: number, field: keyof HealthThreshold, value: any) => {
    const newThresholds = [...thresholds];
    newThresholds[index] = { ...newThresholds[index], [field]: value };
    onUpdate(newThresholds);
  };

  return (
    <div className="space-y-6">
      {/* Refresh Settings */}
      <div className="space-y-4">
        <h4 className="font-medium">Refresh Settings</h4>
        <div className="flex items-center space-x-2">
          <Switch
            checked={autoRefresh}
            onCheckedChange={onAutoRefreshChange}
          />
          <Label>Auto-refresh</Label>
        </div>
        {autoRefresh && (
          <div className="space-y-2">
            <Label>Refresh interval (seconds)</Label>
            <Select 
              value={refreshInterval.toString()} 
              onValueChange={(value) => onRefreshIntervalChange(parseInt(value))}
            >
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="10">10s</SelectItem>
                <SelectItem value="30">30s</SelectItem>
                <SelectItem value="60">1m</SelectItem>
                <SelectItem value="300">5m</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      </div>

      <Separator />

      {/* Threshold Settings */}
      <div className="space-y-4">
        <h4 className="font-medium">Alert Thresholds</h4>
        <div className="space-y-4">
          {thresholds.map((threshold, index) => (
            <Card key={threshold.metric}>
              <CardContent className="p-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label className="font-medium">
                      {threshold.metric.replace('_', ' ').toUpperCase()}
                    </Label>
                    <Switch
                      checked={threshold.enabled}
                      onCheckedChange={(enabled) => updateThreshold(index, 'enabled', enabled)}
                    />
                  </div>
                  
                  {threshold.enabled && (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">Warning Threshold</Label>
                        <Input
                          type="number"
                          value={threshold.warning}
                          onChange={(e) => updateThreshold(index, 'warning', parseFloat(e.target.value))}
                          className="h-8"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">Critical Threshold</Label>
                        <Input
                          type="number"
                          value={threshold.critical}
                          onChange={(e) => updateThreshold(index, 'critical', parseFloat(e.target.value))}
                          className="h-8"
                        />
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}