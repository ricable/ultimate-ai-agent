"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  Square,
  Activity,
  Clock,
  DollarSign,
  Zap,
  Terminal,
  FileText,
  AlertCircle,
  CheckCircle,
  XCircle,
  Eye,
  Download,
  Share2,
  RefreshCw,
  Maximize,
  Minimize,
  Copy,
  Filter,
  Search,
  BarChart3,
  TrendingUp
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { openCodeClient } from "@/lib/opencode-client";
import type { 
  Agent,
  AgentRun,
  AgentRunStatus
} from "@/types/opencode";

interface AgentExecutionMonitorProps {
  agents?: Agent[];
  onClose?: () => void;
}

type FilterType = "all" | "running" | "completed" | "failed" | "pending";

export function AgentExecutionMonitor({ agents = [], onClose }: AgentExecutionMonitorProps) {
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<AgentRun | null>(null);
  const [runOutput, setRunOutput] = useState<string>("");
  const [showOutput, setShowOutput] = useState(false);
  const [filter, setFilter] = useState<FilterType>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    loadRuns();
    
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(loadRuns, 5000); // Refresh every 5 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const loadRuns = async () => {
    try {
      if (!refreshing) setLoading(true);
      const runsData = await openCodeClient.getAgentRuns();
      setRuns(runsData.sort((a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime()));
    } catch (error) {
      console.error("Failed to load agent runs:", error);
      // Provide mock data for demonstration
      setRuns(getMockRuns());
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadRuns();
  };

  const handleStopRun = async (runId: string) => {
    try {
      await openCodeClient.stopAgentRun(runId);
      await loadRuns(); // Refresh the list
      
      toast({
        title: "Success",
        description: "Agent execution stopped"
      });
    } catch (error) {
      console.error("Failed to stop run:", error);
      toast({
        title: "Error",
        description: "Failed to stop agent execution",
        variant: "destructive"
      });
    }
  };

  const handlePauseRun = async (runId: string) => {
    try {
      await openCodeClient.pauseAgentRun(runId);
      await loadRuns();
      
      toast({
        title: "Success",
        description: "Agent execution paused"
      });
    } catch (error) {
      console.error("Failed to pause run:", error);
      toast({
        title: "Error",
        description: "Failed to pause agent execution",
        variant: "destructive"
      });
    }
  };

  const handleResumeRun = async (runId: string) => {
    try {
      await openCodeClient.resumeAgentRun(runId);
      await loadRuns();
      
      toast({
        title: "Success",
        description: "Agent execution resumed"
      });
    } catch (error) {
      console.error("Failed to resume run:", error);
      toast({
        title: "Error",
        description: "Failed to resume agent execution",
        variant: "destructive"
      });
    }
  };

  const handleViewOutput = async (run: AgentRun) => {
    try {
      const output = await openCodeClient.getAgentRunOutput(run.id);
      setRunOutput(output);
      setSelectedRun(run);
      setShowOutput(true);
    } catch (error) {
      console.error("Failed to load output:", error);
      setRunOutput("Failed to load output");
      setSelectedRun(run);
      setShowOutput(true);
    }
  };

  const filteredRuns = runs.filter(run => {
    const matchesFilter = filter === "all" || run.status === filter;
    const matchesSearch = run.agent_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      run.task.toLowerCase().includes(searchQuery.toLowerCase()) ||
      run.provider.toLowerCase().includes(searchQuery.toLowerCase());
    
    return matchesFilter && matchesSearch;
  });

  const getStatusIcon = (status: AgentRunStatus) => {
    switch (status) {
      case "completed":
        return CheckCircle;
      case "running":
        return Activity;
      case "failed":
        return XCircle;
      case "paused":
        return Pause;
      case "cancelled":
        return Square;
      default:
        return Clock;
    }
  };

  const getStatusColor = (status: AgentRunStatus) => {
    switch (status) {
      case "completed":
        return "text-green-500";
      case "running":
        return "text-blue-500";
      case "failed":
        return "text-red-500";
      case "paused":
        return "text-yellow-500";
      case "cancelled":
        return "text-gray-500";
      default:
        return "text-orange-500";
    }
  };

  const formatDuration = (run: AgentRun) => {
    const start = new Date(run.started_at).getTime();
    const end = run.completed_at ? new Date(run.completed_at).getTime() : Date.now();
    const duration = end - start;
    
    const seconds = Math.floor(duration / 1000);
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

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(amount);
  };

  const getRunStats = () => {
    const total = runs.length;
    const running = runs.filter(r => r.status === "running").length;
    const completed = runs.filter(r => r.status === "completed").length;
    const failed = runs.filter(r => r.status === "failed").length;
    const totalCost = runs.reduce((sum, r) => sum + (r.metrics.cost_usd || 0), 0);
    
    return { total, running, completed, failed, totalCost };
  };

  const stats = getRunStats();

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold">Agent Execution Monitor</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
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
            <Activity className="h-6 w-6" />
            <span>Agent Execution Monitor</span>
          </h2>
          <p className="text-muted-foreground">Monitor and manage running agent executions</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="icon"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
          </Button>
          <Button
            variant={autoRefresh ? "default" : "outline"}
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? "Auto-refresh On" : "Auto-refresh Off"}
          </Button>
          {onClose && (
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
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
                  <p className="text-2xl font-bold">{stats.total}</p>
                </div>
                <BarChart3 className="h-8 w-8 text-blue-500" />
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
                  <p className="text-sm font-medium text-muted-foreground">Running</p>
                  <p className="text-2xl font-bold text-blue-500">{stats.running}</p>
                </div>
                <Activity className="h-8 w-8 text-blue-500" />
              </div>
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
                  <p className="text-sm font-medium text-muted-foreground">Completed</p>
                  <p className="text-2xl font-bold text-green-500">{stats.completed}</p>
                </div>
                <CheckCircle className="h-8 w-8 text-green-500" />
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
                  <p className="text-sm font-medium text-muted-foreground">Failed</p>
                  <p className="text-2xl font-bold text-red-500">{stats.failed}</p>
                </div>
                <XCircle className="h-8 w-8 text-red-500" />
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Cost</p>
                  <p className="text-2xl font-bold text-purple-500">{formatCurrency(stats.totalCost)}</p>
                </div>
                <DollarSign className="h-8 w-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search runs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        <Select value={filter} onValueChange={(value) => setFilter(value as FilterType)}>
          <SelectTrigger className="w-48">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Runs</SelectItem>
            <SelectItem value="running">Running</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="failed">Failed</SelectItem>
            <SelectItem value="pending">Pending</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Runs List */}
      {filteredRuns.length === 0 ? (
        <div className="text-center py-12">
          <Activity className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-2">No agent runs found</h3>
          <p className="text-muted-foreground">
            {runs.length === 0 ? "No agents have been executed yet" : "No runs match your current filters"}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <AnimatePresence>
            {filteredRuns.map((run) => {
              const StatusIcon = getStatusIcon(run.status);
              const canControl = run.status === "running" || run.status === "paused";
              
              return (
                <motion.div
                  key={run.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  layout
                >
                  <Card className="hover:shadow-lg transition-shadow">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className="text-2xl">{run.agent_icon}</div>
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-1">
                              <h3 className="font-semibold">{run.agent_name}</h3>
                              <Badge variant="outline">{run.provider}</Badge>
                              <Badge variant="outline">{run.model}</Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                              {run.task}
                            </p>
                            <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                              <span>Started: {new Date(run.started_at).toLocaleString()}</span>
                              <span>Duration: {formatDuration(run)}</span>
                              {run.metrics.cost_usd && (
                                <span>Cost: {formatCurrency(run.metrics.cost_usd)}</span>
                              )}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center space-x-2">
                          <div className="text-right mr-4">
                            <div className={`flex items-center space-x-1 ${getStatusColor(run.status)}`}>
                              <StatusIcon className="h-4 w-4" />
                              <span className="text-sm font-medium capitalize">{run.status}</span>
                            </div>
                            {run.status === "running" && run.metrics.message_count && (
                              <p className="text-xs text-muted-foreground">
                                {run.metrics.message_count} messages
                              </p>
                            )}
                          </div>

                          {/* Control Buttons */}
                          <div className="flex items-center space-x-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleViewOutput(run)}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>

                            {canControl && run.status === "running" && (
                              <>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handlePauseRun(run.id)}
                                >
                                  <Pause className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handleStopRun(run.id)}
                                >
                                  <Square className="h-4 w-4" />
                                </Button>
                              </>
                            )}

                            {run.status === "paused" && (
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => handleResumeRun(run.id)}
                              >
                                <Play className="h-4 w-4" />
                              </Button>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Progress Bar for Running Tasks */}
                      {run.status === "running" && (
                        <div className="mt-4">
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span>Execution Progress</span>
                            <span>Estimated completion in 2-5 minutes</span>
                          </div>
                          <Progress value={Math.min(75, (Date.now() - new Date(run.started_at).getTime()) / 1000 / 3)} />
                        </div>
                      )}

                      {/* Metrics Summary */}
                      {run.metrics && (
                        <div className="mt-4 grid grid-cols-3 md:grid-cols-5 gap-4 pt-4 border-t">
                          {run.metrics.total_tokens && (
                            <div className="text-center">
                              <p className="text-xs text-muted-foreground">Tokens</p>
                              <p className="text-sm font-medium">{run.metrics.total_tokens.toLocaleString()}</p>
                            </div>
                          )}
                          {run.metrics.message_count && (
                            <div className="text-center">
                              <p className="text-xs text-muted-foreground">Messages</p>
                              <p className="text-sm font-medium">{run.metrics.message_count}</p>
                            </div>
                          )}
                          {run.metrics.tool_executions && (
                            <div className="text-center">
                              <p className="text-xs text-muted-foreground">Tool Calls</p>
                              <p className="text-sm font-medium">{run.metrics.tool_executions}</p>
                            </div>
                          )}
                          {run.metrics.avg_response_time && (
                            <div className="text-center">
                              <p className="text-xs text-muted-foreground">Avg Response</p>
                              <p className="text-sm font-medium">{run.metrics.avg_response_time}ms</p>
                            </div>
                          )}
                          {run.metrics.cost_usd && (
                            <div className="text-center">
                              <p className="text-xs text-muted-foreground">Cost</p>
                              <p className="text-sm font-medium">{formatCurrency(run.metrics.cost_usd)}</p>
                            </div>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}

      {/* Output Dialog */}
      <Dialog open={showOutput} onOpenChange={setShowOutput}>
        <DialogContent className="max-w-4xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <Terminal className="h-5 w-5" />
              <span>Agent Output: {selectedRun?.agent_name}</span>
            </DialogTitle>
          </DialogHeader>
          
          <Tabs defaultValue="output" className="space-y-4">
            <TabsList>
              <TabsTrigger value="output">Live Output</TabsTrigger>
              <TabsTrigger value="logs">Execution Logs</TabsTrigger>
              <TabsTrigger value="metrics">Metrics</TabsTrigger>
            </TabsList>

            <TabsContent value="output">
              <ScrollArea className="h-96 w-full rounded-md border p-4">
                <pre className="text-sm whitespace-pre-wrap font-mono">
                  {runOutput || "No output available"}
                </pre>
              </ScrollArea>
              <div className="flex justify-end space-x-2 mt-4">
                <Button
                  variant="outline"
                  onClick={() => navigator.clipboard.writeText(runOutput)}
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Copy
                </Button>
                <Button variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="logs">
              <ScrollArea className="h-96 w-full rounded-md border p-4">
                <div className="space-y-2">
                  <div className="text-xs text-muted-foreground">
                    [INFO] Agent execution started at {selectedRun?.started_at}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    [INFO] Using model: {selectedRun?.model} via {selectedRun?.provider}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    [INFO] Task: {selectedRun?.task}
                  </div>
                  {selectedRun?.status === "completed" && (
                    <div className="text-xs text-green-600">
                      [SUCCESS] Execution completed successfully
                    </div>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="metrics">
              <div className="space-y-4">
                {selectedRun?.metrics && (
                  <div className="grid grid-cols-2 gap-4">
                    <Card>
                      <CardContent className="p-4">
                        <h4 className="font-medium mb-2">Token Usage</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Total Tokens:</span>
                            <span>{selectedRun.metrics.total_tokens?.toLocaleString() || "N/A"}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Input Tokens:</span>
                            <span>{selectedRun.metrics.input_tokens?.toLocaleString() || "N/A"}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Output Tokens:</span>
                            <span>{selectedRun.metrics.output_tokens?.toLocaleString() || "N/A"}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardContent className="p-4">
                        <h4 className="font-medium mb-2">Performance</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Messages:</span>
                            <span>{selectedRun.metrics.message_count || "N/A"}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Tool Executions:</span>
                            <span>{selectedRun.metrics.tool_executions || "N/A"}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Avg Response Time:</span>
                            <span>{selectedRun.metrics.avg_response_time ? `${selectedRun.metrics.avg_response_time}ms` : "N/A"}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Cost:</span>
                            <span>{selectedRun.metrics.cost_usd ? formatCurrency(selectedRun.metrics.cost_usd) : "N/A"}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Mock data function
function getMockRuns(): AgentRun[] {
  return [
    {
      id: "run-1",
      agent_id: "agent-1",
      agent_name: "Code Assistant",
      agent_icon: "💻",
      session_id: "session-123",
      task: "Review and refactor the authentication module for better security",
      model: "claude-3.5-sonnet",
      provider: "anthropic",
      project_path: "/home/user/project",
      status: "running",
      started_at: "2024-01-20T16:30:00Z",
      metrics: {
        total_tokens: 1500,
        input_tokens: 800,
        output_tokens: 700,
        cost_usd: 0.0225,
        message_count: 6,
        tool_executions: 2,
        errors_count: 0,
        avg_response_time: 1200
      }
    },
    {
      id: "run-2",
      agent_id: "agent-2",
      agent_name: "Data Analyst",
      agent_icon: "📊",
      session_id: "session-124",
      task: "Analyze sales data trends and create comprehensive visualizations",
      model: "gpt-4",
      provider: "openai",
      project_path: "/home/user/data-project",
      status: "completed",
      started_at: "2024-01-20T15:00:00Z",
      completed_at: "2024-01-20T15:12:00Z",
      duration_ms: 720000,
      metrics: {
        total_tokens: 3200,
        input_tokens: 1500,
        output_tokens: 1700,
        cost_usd: 0.048,
        message_count: 12,
        tool_executions: 5,
        errors_count: 0,
        avg_response_time: 950
      }
    },
    {
      id: "run-3",
      agent_id: "agent-1",
      agent_name: "Code Assistant",
      agent_icon: "💻",
      session_id: "session-125",
      task: "Debug the payment processing system",
      model: "claude-3.5-sonnet",
      provider: "anthropic",
      project_path: "/home/user/project",
      status: "failed",
      started_at: "2024-01-20T14:00:00Z",
      completed_at: "2024-01-20T14:08:00Z",
      duration_ms: 480000,
      error_message: "Timeout while waiting for API response",
      metrics: {
        total_tokens: 1200,
        input_tokens: 600,
        output_tokens: 600,
        cost_usd: 0.018,
        message_count: 4,
        tool_executions: 1,
        errors_count: 1,
        avg_response_time: 1800
      }
    }
  ];
}