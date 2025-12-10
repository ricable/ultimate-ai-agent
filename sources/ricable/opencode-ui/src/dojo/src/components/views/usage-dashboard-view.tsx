"use client";

import React, { useEffect, useState, useMemo } from "react";
import { motion } from "framer-motion";
import { 
  BarChart3, 
  DollarSign, 
  Activity, 
  TrendingUp, 
  Clock,
  Zap,
  Users,
  Database,
  ArrowLeft,
  Filter,
  Calendar,
  Download,
  PieChart,
  LineChart,
  BarChart,
  Target
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useSessionStore } from "@/lib/session-store";
import {
  LineChart as RechartsLineChart,
  AreaChart,
  BarChart as RechartsBarChart,
  PieChart as RechartsPieChart,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  Line,
  Bar,
  Pie
} from 'recharts';
import {
  processTimeSeriesData,
  processProviderBreakdown,
  processProjectAnalytics,
  processCacheAnalytics,
  exportToCSV,
  exportToJSON,
  exportToPDF,
  formatCurrency,
  formatNumber,
  formatTokens,
  formatPercentage,
  formatDuration,
  TIME_FRAMES,
  PROVIDER_COLORS,
  type TimeSeriesDataPoint,
  type ProviderBreakdown,
  type ProjectAnalytics,
  type CacheAnalytics,
  type UsageTimeframe
} from "@/lib/analytics-utils";
import { OpenCodeView } from "@/types/opencode";

interface UsageDashboardViewProps {
  onViewChange: (view: OpenCodeView) => void;
}

export function UsageDashboardView({ onViewChange }: UsageDashboardViewProps) {
  const { 
    sessions, 
    providers, 
    providerMetrics,
    actions 
  } = useSessionStore();

  const [selectedTimeframe, setSelectedTimeframe] = useState<UsageTimeframe>(TIME_FRAMES[1]); // Last 30 Days
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Process analytics data
  const timeSeriesData = useMemo(() => 
    processTimeSeriesData(sessions, selectedTimeframe), 
    [sessions, selectedTimeframe]
  );
  
  const providerBreakdown = useMemo(() => 
    processProviderBreakdown(sessions, providerMetrics), 
    [sessions, providerMetrics]
  );
  
  const projectAnalytics = useMemo(() => 
    processProjectAnalytics(sessions), 
    [sessions]
  );
  
  const cacheAnalytics = useMemo(() => 
    processCacheAnalytics(sessions), 
    [sessions]
  );

  useEffect(() => {
    loadUsageData();
  }, [selectedTimeframe]);

  const loadUsageData = async () => {
    try {
      setLoading(true);
      setError(null);
      await Promise.all([
        actions.loadSessions(),
        actions.loadProviders(),
        actions.loadProviderMetrics()
      ]);
    } catch (err) {
      console.error("Failed to load usage data:", err);
      setError("Failed to load usage statistics. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Export functionality
  const handleExport = (format: 'csv' | 'json' | 'pdf') => {
    const timestamp = new Date().toISOString().split('T')[0];
    const baseFilename = `opencode-usage-${timestamp}`;
    
    const summary = {
      totalCost: providerBreakdown.reduce((sum, p) => sum + p.cost, 0),
      totalSessions: sessions.length,
      totalTokens: sessions.reduce((sum, s) => sum + (s.token_usage?.input_tokens || 0) + (s.token_usage?.output_tokens || 0), 0),
      timeframe: selectedTimeframe.label,
      generatedAt: new Date().toISOString()
    };

    switch (format) {
      case 'csv':
        exportToCSV(timeSeriesData, `${baseFilename}-timeseries`);
        exportToCSV(providerBreakdown, `${baseFilename}-providers`);
        exportToCSV(projectAnalytics, `${baseFilename}-projects`);
        break;
      case 'json':
        exportToJSON({
          summary,
          timeSeriesData,
          providerBreakdown,
          projectAnalytics,
          cacheAnalytics
        }, baseFilename);
        break;
      case 'pdf':
        exportToPDF(summary, [
          { title: 'Usage Over Time', data: timeSeriesData },
          { title: 'Provider Breakdown', data: providerBreakdown },
          { title: 'Project Analytics', data: projectAnalytics }
        ], baseFilename);
        break;
    }
  };

  // Calculate aggregated metrics from processed data
  const totalCost = providerBreakdown.reduce((sum, p) => sum + p.cost, 0);
  const totalSessions = sessions.length;
  const totalTokens = sessions.reduce((sum, session) => {
    if (session.token_usage) {
      return sum + session.token_usage.input_tokens + session.token_usage.output_tokens;
    }
    return sum;
  }, 0);
  const avgCostPerSession = totalSessions > 0 ? totalCost / totalSessions : 0;

  // Custom tooltip formatters for charts
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center space-x-2 text-xs">
              <div 
                className="w-3 h-3 rounded-full" 
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-muted-foreground">{entry.dataKey}:</span>
              <span className="font-medium">
                {entry.dataKey.includes('cost') ? formatCurrency(entry.value) :
                 entry.dataKey.includes('tokens') ? formatTokens(entry.value) :
                 formatNumber(entry.value)}
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  // Calculate token breakdown from actual data
  const inputTokens = sessions.reduce((sum, s) => sum + (s.token_usage?.input_tokens || 0), 0);
  const outputTokens = sessions.reduce((sum, s) => sum + (s.token_usage?.output_tokens || 0), 0);
  const cacheTokens = sessions.reduce((sum, s) => sum + ((s.token_usage as any)?.cache_tokens || 0), 0);

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      {/* Header matching Claudia's design */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button 
              variant="ghost" 
              size="icon"
              onClick={() => onViewChange("welcome")}
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-2xl font-bold">Usage Dashboard</h1>
              <p className="text-sm text-muted-foreground">Track your OpenCode usage and costs</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Select value={selectedTimeframe.label} onValueChange={(value) => {
              const timeframe = TIME_FRAMES.find(tf => tf.label === value);
              if (timeframe) setSelectedTimeframe(timeframe);
            }}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Select timeframe" />
              </SelectTrigger>
              <SelectContent>
                {TIME_FRAMES.map((timeframe) => (
                  <SelectItem key={timeframe.label} value={timeframe.label}>
                    {timeframe.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <div className="flex items-center space-x-1">
              <Button variant="outline" size="sm" onClick={() => handleExport('csv')}>
                <Download className="h-3 w-3 mr-1" />
                CSV
              </Button>
              <Button variant="outline" size="sm" onClick={() => handleExport('json')}>
                <Download className="h-3 w-3 mr-1" />
                JSON
              </Button>
              <Button variant="outline" size="sm" onClick={() => handleExport('pdf')}>
                <Download className="h-3 w-3 mr-1" />
                PDF
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Loading and Error States */}
        {loading ? (
          <div className="flex items-center justify-center h-full min-h-96">
            <div className="text-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="h-8 w-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-4"
              />
              <p className="text-sm text-muted-foreground">Loading usage statistics...</p>
            </div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full min-h-96">
            <div className="text-center max-w-md">
              <p className="text-sm text-destructive mb-4">{error}</p>
              <Button onClick={loadUsageData} size="sm">
                Try Again
              </Button>
            </div>
          </div>
        ) : (
          <>
            {/* Key Metrics - Matches Claudia's layout exactly */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="grid grid-cols-4 gap-6 mb-8"
            >
              <Card className="shimmer-hover">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Total Cost</div>
                      <div className="text-3xl font-bold">{formatCurrency(totalCost)}</div>
                    </div>
                    <DollarSign className="h-8 w-8 text-muted-foreground/20" />
                  </div>
                </CardContent>
              </Card>

              <Card className="shimmer-hover">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Total Sessions</div>
                      <div className="text-3xl font-bold">{formatNumber(totalSessions)}</div>
                    </div>
                    <Activity className="h-8 w-8 text-muted-foreground/20" />
                  </div>
                </CardContent>
              </Card>

              <Card className="shimmer-hover">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Total Tokens</div>
                      <div className="text-3xl font-bold">{formatTokens(totalTokens)}</div>
                    </div>
                    <Database className="h-8 w-8 text-muted-foreground/20" />
                  </div>
                </CardContent>
              </Card>

              <Card className="shimmer-hover">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Avg Cost/Session</div>
                      <div className="text-3xl font-bold">{formatCurrency(avgCostPerSession)}</div>
                    </div>
                    <TrendingUp className="h-8 w-8 text-muted-foreground/20" />
                  </div>
                </CardContent>
              </Card>
            </motion.div>

        {/* Tabs for different views */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 max-w-md">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
            <TabsTrigger value="providers">Providers</TabsTrigger>
            <TabsTrigger value="projects">Projects</TabsTrigger>
            <TabsTrigger value="cache">Cache</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            {/* Token Breakdown */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Database className="h-5 w-5" />
                  <span>Token Breakdown</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-6">
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Input Tokens</div>
                    <div className="text-2xl font-bold">{formatTokens(inputTokens)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Output Tokens</div>
                    <div className="text-2xl font-bold">{formatTokens(outputTokens)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Cache Write</div>
                    <div className="text-2xl font-bold">{formatTokens(cacheTokens / 2)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Cache Read</div>
                    <div className="text-2xl font-bold">{formatTokens(cacheTokens / 2)}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-2 gap-6">
              {/* Provider Distribution Chart */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <PieChart className="h-5 w-5" />
                    <span>Cost Distribution by Provider</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsPieChart>
                        <Pie
                          data={providerBreakdown.slice(0, 6)}
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          dataKey="cost"
                          nameKey="provider"
                        >
                          {providerBreakdown.slice(0, 6).map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                      </RechartsPieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Top Projects */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <BarChart3 className="h-5 w-5" />
                    <span>Top Projects by Cost</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsBarChart data={projectAnalytics.slice(0, 5)}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis 
                          dataKey="name" 
                          className="text-xs"
                          tick={{ fontSize: 10 }}
                          angle={-45}
                          textAnchor="end"
                          height={60}
                        />
                        <YAxis className="text-xs" tick={{ fontSize: 10 }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Bar dataKey="totalCost" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                      </RechartsBarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="trends" className="space-y-6">
            {/* Usage Over Time */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <LineChart className="h-5 w-5" />
                  <span>Token Usage Over Time ({selectedTimeframe.label})</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timeSeriesData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis 
                        dataKey="date" 
                        className="text-xs"
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis className="text-xs" tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Area
                        type="monotone"
                        dataKey="inputTokens"
                        stackId="1"
                        stroke="#3B82F6"
                        fill="#3B82F6"
                        fillOpacity={0.6}
                        name="Input Tokens"
                      />
                      <Area
                        type="monotone"
                        dataKey="outputTokens"
                        stackId="1"
                        stroke="#10B981"
                        fill="#10B981"
                        fillOpacity={0.6}
                        name="Output Tokens"
                      />
                      <Area
                        type="monotone"
                        dataKey="cacheTokens"
                        stackId="1"
                        stroke="#8B5CF6"
                        fill="#8B5CF6"
                        fillOpacity={0.6}
                        name="Cache Tokens"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Cost Trends */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5" />
                  <span>Cost Trends ({selectedTimeframe.label})</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsLineChart data={timeSeriesData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis 
                        dataKey="date" 
                        className="text-xs"
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis className="text-xs" tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="cost"
                        stroke="#EF4444"
                        strokeWidth={2}
                        dot={{ fill: "#EF4444", strokeWidth: 2, r: 4 }}
                        name="Daily Cost"
                      />
                    </RechartsLineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="providers" className="space-y-6">
            {/* Provider Breakdown Table */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Users className="h-5 w-5" />
                  <span>Provider Performance Analysis</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {providerBreakdown.map((provider, index) => (
                    <div key={index} className="flex items-center justify-between p-4 border border-border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div 
                          className="w-4 h-4 rounded-full" 
                          style={{ backgroundColor: provider.color }}
                        />
                        <div>
                          <div className="font-medium">{provider.provider}</div>
                          <div className="text-sm text-muted-foreground">{provider.model}</div>
                        </div>
                      </div>
                      <div className="grid grid-cols-4 gap-8 text-sm">
                        <div className="text-center">
                          <div className="font-medium">{provider.sessions}</div>
                          <div className="text-muted-foreground">Sessions</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium">{formatTokens(provider.tokens)}</div>
                          <div className="text-muted-foreground">Tokens</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium">{formatCurrency(provider.cost)}</div>
                          <div className="text-muted-foreground">Cost</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium">{formatPercentage(provider.percentage)}</div>
                          <div className="text-muted-foreground">Share</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="projects" className="space-y-6">
            {/* Session Distribution by Project */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart className="h-5 w-5" />
                  <span>Sessions by Project</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart data={projectAnalytics}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis 
                        dataKey="name" 
                        className="text-xs"
                        tick={{ fontSize: 10 }}
                        angle={-45}
                        textAnchor="end"
                        height={80}
                      />
                      <YAxis className="text-xs" tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar dataKey="sessions" fill="#10B981" radius={[4, 4, 0, 0]} name="Sessions" />
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Project Details Table */}
            <Card>
              <CardHeader>
                <CardTitle>Project Analytics Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {projectAnalytics.map((project, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border border-border rounded-lg">
                      <div className="flex-1">
                        <div className="font-medium truncate" title={project.path}>{project.name}</div>
                        <div className="text-xs text-muted-foreground truncate">{project.path}</div>
                      </div>
                      <div className="grid grid-cols-4 gap-6 text-sm">
                        <div className="text-center">
                          <div className="font-medium">{project.sessions}</div>
                          <div className="text-muted-foreground text-xs">Sessions</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium">{formatTokens(project.totalTokens)}</div>
                          <div className="text-muted-foreground text-xs">Tokens</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium">{formatCurrency(project.totalCost)}</div>
                          <div className="text-muted-foreground text-xs">Total Cost</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium">{formatDuration(project.avgSessionDuration)}</div>
                          <div className="text-muted-foreground text-xs">Avg Duration</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="cache" className="space-y-6">
            {/* Cache Performance */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Cache Hit Rate</div>
                      <div className="text-3xl font-bold text-green-500">{formatPercentage(cacheAnalytics.hitRate)}</div>
                    </div>
                    <Target className="h-8 w-8 text-green-500/20" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Total Reads</div>
                      <div className="text-3xl font-bold">{formatTokens(cacheAnalytics.totalReads)}</div>
                    </div>
                    <Database className="h-8 w-8 text-muted-foreground/20" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Cost Savings</div>
                      <div className="text-3xl font-bold text-green-500">{formatCurrency(cacheAnalytics.savings)}</div>
                    </div>
                    <DollarSign className="h-8 w-8 text-green-500/20" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Cache Hit/Miss Visualization */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="h-5 w-5" />
                  <span>Cache Performance</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPieChart>
                      <Pie
                        data={[
                          { name: 'Cache Hits', value: cacheAnalytics.hitRate, fill: '#10B981' },
                          { name: 'Cache Misses', value: cacheAnalytics.missRate, fill: '#EF4444' }
                        ]}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        nameKey="name"
                      >
                        <Cell fill="#10B981" />
                        <Cell fill="#EF4444" />
                      </Pie>
                      <Tooltip 
                        formatter={(value: any) => [`${value.toFixed(1)}%`, '']}
                        labelStyle={{ color: 'var(--foreground)' }}
                        contentStyle={{ 
                          backgroundColor: 'var(--card)',
                          border: '1px solid var(--border)',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
          </>
        )}
      </div>
    </div>
  );
}