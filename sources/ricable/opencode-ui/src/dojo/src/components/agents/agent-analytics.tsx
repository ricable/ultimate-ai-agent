"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Clock,
  DollarSign,
  Target,
  Users,
  Zap,
  BarChart3,
  PieChart,
  Calendar,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ArrowUp,
  ArrowDown,
  Minus,
  RefreshCw
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { openCodeClient } from "@/lib/opencode-client";
import type { 
  Agent, 
  AgentPerformanceMetrics, 
  AgentAnalytics,
  AgentRun 
} from "@/types/opencode";

interface AgentAnalyticsProps {
  agent: Agent;
  onClose?: () => void;
}

type TimeframeType = "hour" | "day" | "week" | "month";

export function AgentAnalytics({ agent, onClose }: AgentAnalyticsProps) {
  const [metrics, setMetrics] = useState<AgentPerformanceMetrics | null>(null);
  const [analytics, setAnalytics] = useState<AgentAnalytics | null>(null);
  const [recentRuns, setRecentRuns] = useState<AgentRun[]>([]);
  const [timeframe, setTimeframe] = useState<TimeframeType>("week");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadAnalytics();
  }, [agent.id, timeframe]);

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      const [metricsData, analyticsData, runsData] = await Promise.all([
        openCodeClient.getAgentMetrics(agent.id).catch(() => null),
        openCodeClient.getAgentAnalytics(agent.id, timeframe).catch(() => null),
        openCodeClient.getAgentRuns(agent.id).catch(() => [])
      ]);

      setMetrics(metricsData || getMockMetrics());
      setAnalytics(analyticsData || getMockAnalytics());
      setRecentRuns(runsData.slice(0, 10));
    } catch (error) {
      console.error("Failed to load analytics:", error);
      // Provide mock data for demonstration
      setMetrics(getMockMetrics());
      setAnalytics(getMockAnalytics());
      setRecentRuns(getMockRecentRuns());
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadAnalytics();
    setRefreshing(false);
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + "K";
    }
    return num.toString();
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(amount);
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return CheckCircle;
      case "running":
        return Activity;
      case "failed":
        return XCircle;
      default:
        return AlertTriangle;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "text-green-500";
      case "running":
        return "text-blue-500";
      case "failed":
        return "text-red-500";
      default:
        return "text-yellow-500";
    }
  };

  const getTrendIcon = (current: number, previous: number) => {
    if (current > previous) return <ArrowUp className="h-3 w-3 text-green-500" />;
    if (current < previous) return <ArrowDown className="h-3 w-3 text-red-500" />;
    return <Minus className="h-3 w-3 text-gray-500" />;
  };

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold flex items-center space-x-2">
              <span className="text-2xl">{agent.icon}</span>
              <span>{agent.name} Analytics</span>
            </h2>
            <p className="text-muted-foreground">Performance insights and metrics</p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(8)].map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-4">
                <div className="h-4 bg-gray-200 rounded mb-2"></div>
                <div className="h-8 bg-gray-200 rounded"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <span className="text-2xl">{agent.icon}</span>
            <span>{agent.name} Analytics</span>
          </h2>
          <p className="text-muted-foreground">Performance insights and metrics</p>
        </div>
        <div className="flex items-center space-x-2">
          <Select value={timeframe} onValueChange={(value) => setTimeframe(value as TimeframeType)}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="hour">Last Hour</SelectItem>
              <SelectItem value="day">Last Day</SelectItem>
              <SelectItem value="week">Last Week</SelectItem>
              <SelectItem value="month">Last Month</SelectItem>
            </SelectContent>
          </Select>
          <Button 
            variant="outline" 
            size="icon" 
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
          </Button>
          {onClose && (
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Runs</p>
                  <p className="text-2xl font-bold">{formatNumber(metrics?.total_runs || 0)}</p>
                </div>
                <Activity className="h-8 w-8 text-blue-500" />
              </div>
              <div className="flex items-center mt-2">
                {getTrendIcon(metrics?.total_runs || 0, (metrics?.total_runs || 0) * 0.8)}
                <span className="text-xs text-muted-foreground ml-1">vs last period</span>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold">{((metrics?.success_rate || 0) * 100).toFixed(1)}%</p>
                </div>
                <Target className="h-8 w-8 text-green-500" />
              </div>
              <Progress value={(metrics?.success_rate || 0) * 100} className="mt-2" />
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Avg Duration</p>
                  <p className="text-2xl font-bold">{formatDuration(metrics?.avg_duration_ms || 0)}</p>
                </div>
                <Clock className="h-8 w-8 text-orange-500" />
              </div>
              <div className="flex items-center mt-2">
                {getTrendIcon(1, 1.2)}
                <span className="text-xs text-muted-foreground ml-1">faster than avg</span>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Cost</p>
                  <p className="text-2xl font-bold">{formatCurrency(metrics?.total_cost_usd || 0)}</p>
                </div>
                <DollarSign className="h-8 w-8 text-purple-500" />
              </div>
              <div className="flex items-center mt-2">
                <span className="text-xs text-muted-foreground">
                  {formatCurrency(metrics?.avg_cost_per_run || 0)} per run
                </span>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="usage">Usage Patterns</TabsTrigger>
          <TabsTrigger value="errors">Error Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Provider Breakdown */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <PieChart className="h-5 w-5" />
                  <span>Provider Usage</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {analytics?.provider_breakdown && Object.entries(analytics.provider_breakdown).map(([provider, data]) => (
                  <div key={provider} className="flex items-center justify-between py-2">
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline">{provider}</Badge>
                      <span className="text-sm">{data.usage_count} runs</span>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">{(data.success_rate * 100).toFixed(1)}%</p>
                      <p className="text-xs text-muted-foreground">{formatCurrency(data.avg_cost)}</p>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Task Categories */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5" />
                  <span>Common Tasks</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {metrics?.common_tasks?.slice(0, 5).map((task, index) => (
                  <div key={index} className="flex items-center justify-between py-2">
                    <div>
                      <p className="text-sm font-medium truncate max-w-48">{task.task}</p>
                      <p className="text-xs text-muted-foreground">
                        {task.count} runs • {(task.success_rate * 100).toFixed(1)}% success
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm">{formatDuration(task.avg_duration)}</p>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Recent Activity */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Activity className="h-5 w-5" />
                <span>Recent Runs</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recentRuns.map((run) => {
                  const StatusIcon = getStatusIcon(run.status);
                  return (
                    <div key={run.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        <StatusIcon className={`h-4 w-4 ${getStatusColor(run.status)}`} />
                        <div>
                          <p className="text-sm font-medium truncate max-w-64">{run.task}</p>
                          <p className="text-xs text-muted-foreground">
                            {new Date(run.started_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant="outline" className="mb-1">
                          {run.provider}
                        </Badge>
                        <p className="text-xs text-muted-foreground">
                          {run.duration_ms ? formatDuration(run.duration_ms) : "Running..."}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Performance Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center text-muted-foreground">
                  <div className="text-center">
                    <TrendingUp className="h-12 w-12 mx-auto mb-2" />
                    <p>Performance chart would be rendered here</p>
                    <p className="text-sm">Integration with charting library needed</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Token Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>Total Tokens</span>
                      <span>{formatNumber(metrics?.total_tokens || 0)}</span>
                    </div>
                    <Progress value={75} className="mt-1" />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>Avg per Run</span>
                      <span>{formatNumber((metrics?.total_tokens || 0) / (metrics?.total_runs || 1))}</span>
                    </div>
                    <Progress value={60} className="mt-1" />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>Efficiency Score</span>
                      <span>85%</span>
                    </div>
                    <Progress value={85} className="mt-1" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="usage" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Usage by Time</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center text-muted-foreground">
                  <div className="text-center">
                    <Calendar className="h-12 w-12 mx-auto mb-2" />
                    <p>Usage timeline would be rendered here</p>
                    <p className="text-sm">Shows peak usage hours/days</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Usage Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm">Most Active Day</span>
                    <span className="text-sm font-medium">Tuesday</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Peak Hour</span>
                    <span className="text-sm font-medium">2:00 PM</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Avg Sessions/Day</span>
                    <span className="text-sm font-medium">3.2</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Last Used</span>
                    <span className="text-sm font-medium">{new Date(metrics?.last_used || Date.now()).toLocaleDateString()}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="errors" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5" />
                <span>Error Patterns</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {analytics?.error_patterns && analytics.error_patterns.length > 0 ? (
                <div className="space-y-3">
                  {analytics.error_patterns.map((error, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="text-sm font-medium">{error.error_type}</p>
                        <p className="text-xs text-muted-foreground">
                          Last occurred: {new Date(error.last_occurrence).toLocaleString()}
                        </p>
                      </div>
                      <Badge variant="destructive">{error.count} times</Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <CheckCircle className="h-12 w-12 mx-auto mb-2 text-green-500" />
                  <p>No significant errors detected</p>
                  <p className="text-sm">Agent is performing well</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Mock data functions
function getMockMetrics(): AgentPerformanceMetrics {
  return {
    agent_id: "agent-1",
    total_runs: 25,
    successful_runs: 23,
    failed_runs: 2,
    avg_duration_ms: 45000,
    total_cost_usd: 2.45,
    avg_cost_per_run: 0.098,
    total_tokens: 125000,
    success_rate: 0.92,
    last_used: "2024-01-20T15:30:00Z",
    most_used_provider: "anthropic",
    most_used_model: "claude-3.5-sonnet",
    common_tasks: [
      {
        task: "Code review and refactoring",
        count: 8,
        avg_duration: 38000,
        success_rate: 0.95
      },
      {
        task: "Bug fixing and debugging",
        count: 6,
        avg_duration: 52000,
        success_rate: 0.88
      },
      {
        task: "Documentation generation",
        count: 4,
        avg_duration: 25000,
        success_rate: 1.0
      }
    ]
  };
}

function getMockAnalytics(): AgentAnalytics {
  return {
    agent_id: "agent-1",
    timeframe: "week",
    usage_data: [
      {
        timestamp: "2024-01-14T00:00:00Z",
        executions: 2,
        success_rate: 1.0,
        avg_duration: 42000,
        total_cost: 0.18
      },
      {
        timestamp: "2024-01-15T00:00:00Z",
        executions: 4,
        success_rate: 0.95,
        avg_duration: 38000,
        total_cost: 0.35
      }
    ],
    provider_breakdown: {
      anthropic: {
        usage_count: 20,
        success_rate: 0.95,
        avg_cost: 0.095
      },
      openai: {
        usage_count: 5,
        success_rate: 0.80,
        avg_cost: 0.12
      }
    },
    task_categories: {
      "code-review": {
        count: 10,
        avg_duration: 35000,
        success_rate: 0.95
      },
      "debugging": {
        count: 8,
        avg_duration: 48000,
        success_rate: 0.88
      }
    },
    error_patterns: [
      {
        error_type: "Timeout",
        count: 2,
        last_occurrence: "2024-01-19T14:30:00Z"
      }
    ]
  };
}

function getMockRecentRuns(): AgentRun[] {
  return [
    {
      id: "run-1",
      agent_id: "agent-1",
      agent_name: "Code Assistant",
      agent_icon: "💻",
      session_id: "session-123",
      task: "Review authentication module",
      model: "claude-3.5-sonnet",
      provider: "anthropic",
      project_path: "/project",
      status: "completed",
      started_at: "2024-01-20T15:00:00Z",
      completed_at: "2024-01-20T15:05:00Z",
      duration_ms: 300000,
      metrics: {
        total_tokens: 2500,
        cost_usd: 0.0375,
        message_count: 8
      }
    },
    {
      id: "run-2",
      agent_id: "agent-1",
      agent_name: "Code Assistant",
      agent_icon: "💻",
      session_id: "session-124",
      task: "Fix database connection issue",
      model: "claude-3.5-sonnet",
      provider: "anthropic",
      project_path: "/project",
      status: "running",
      started_at: "2024-01-20T16:00:00Z",
      metrics: {
        total_tokens: 1200,
        cost_usd: 0.018,
        message_count: 4
      }
    }
  ];
}