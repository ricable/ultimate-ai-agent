"use client";

import React, { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Cpu,
  MemoryStick,
  Network,
  Zap,
  RefreshCw,
  Play,
  Square,
  Trash2,
  Settings,
  Info,
  TrendingUp,
  TrendingDown,
  Minus,
  ExternalLink,
  Terminal,
  Globe,
  Wrench,
  Eye,
  BarChart3
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { 
  Tooltip, 
  TooltipContent, 
  TooltipTrigger,
  TooltipProvider 
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

import { 
  MCPServerWithStatus,
  MCPServerStatus,
  MCPServerHealth,
  MCPServerMetrics,
  MCPTool,
  MCPResource
} from "@/lib/types/mcp";
import { cn } from "@/lib/utils";

// Utility functions
function formatUptime(uptimeMs?: number): string {
  if (!uptimeMs) return "N/A";
  
  const seconds = Math.floor(uptimeMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (days > 0) return `${days}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";
  
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

interface ServerDashboardProps {
  servers: MCPServerWithStatus[];
  onRefresh: () => void;
  onStartServer: (serverId: string) => void;
  onStopServer: (serverId: string) => void;
  onDeleteServer: (serverId: string) => void;
  onEditServer: (serverId: string) => void;
  refreshing?: boolean;
}

export function ServerDashboard({
  servers,
  onRefresh,
  onStartServer,
  onStopServer,
  onDeleteServer,
  onEditServer,
  refreshing = false
}: ServerDashboardProps) {
  const [selectedServer, setSelectedServer] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      onRefresh();
    }, 30000);

    return () => clearInterval(interval);
  }, [autoRefresh, onRefresh]);

  // Server statistics
  const stats = useMemo(() => {
    const total = servers.length;
    const connected = servers.filter(s => s.status.status === "connected").length;
    const disconnected = servers.filter(s => s.status.status === "disconnected").length;
    const errors = servers.filter(s => s.status.status === "error").length;
    const totalTools = servers.reduce((acc, s) => acc + s.tools.length, 0);
    const totalResources = servers.reduce((acc, s) => acc + s.resources.length, 0);
    const avgResponseTime = servers
      .filter(s => s.health.response_time !== undefined)
      .reduce((acc, s, _, arr) => acc + (s.health.response_time || 0) / arr.length, 0);

    return {
      total,
      connected,
      disconnected,
      errors,
      totalTools,
      totalResources,
      avgResponseTime: Math.round(avgResponseTime),
      uptime: servers
        .filter(s => s.status.uptime !== undefined)
        .reduce((acc, s, _, arr) => acc + (s.status.uptime || 0) / arr.length, 0)
    };
  }, [servers]);

  const getStatusIcon = (status: MCPServerStatus["status"]) => {
    switch (status) {
      case "connected":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "disconnected":
        return <XCircle className="h-4 w-4 text-gray-500" />;
      case "error":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "connecting":
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getStatusColor = (status: MCPServerStatus["status"]) => {
    switch (status) {
      case "connected":
        return "bg-green-500";
      case "disconnected":
        return "bg-gray-500";
      case "error":
        return "bg-red-500";
      case "connecting":
        return "bg-blue-500 animate-pulse";
      default:
        return "bg-yellow-500";
    }
  };

  const getHealthScore = (health: MCPServerHealth) => {
    if (!health.healthy) return 0;
    
    let score = 100;
    if (health.response_time && health.response_time > 5000) score -= 20;
    if (health.error_count && health.error_count > 10) score -= 30;
    if (health.availability_percentage && health.availability_percentage < 95) score -= 20;
    
    return Math.max(0, score);
  };

  const formatUptime = (uptime?: number) => {
    if (!uptime) return "N/A";
    
    const hours = Math.floor(uptime / 3600000);
    const minutes = Math.floor((uptime % 3600000) / 60000);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const formatBytes = (bytes?: number) => {
    if (!bytes) return "N/A";
    
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  return (
    <TooltipProvider>
      <div className="space-y-6">
        {/* Dashboard Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">MCP Server Dashboard</h2>
            <p className="text-muted-foreground">
              Monitor and manage your Model Context Protocol servers
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={cn(autoRefresh && "bg-blue-50 border-blue-200")}
            >
              <Activity className="h-4 w-4 mr-2" />
              Auto-refresh {autoRefresh ? "ON" : "OFF"}
            </Button>
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

        {/* Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Total Servers</p>
                  <p className="text-2xl font-bold">{stats.total}</p>
                </div>
                <div className="h-12 w-12 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Network className="h-6 w-6 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Connected</p>
                  <p className="text-2xl font-bold text-green-600">{stats.connected}</p>
                </div>
                <div className="h-12 w-12 bg-green-100 rounded-lg flex items-center justify-center">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Available Tools</p>
                  <p className="text-2xl font-bold">{stats.totalTools}</p>
                </div>
                <div className="h-12 w-12 bg-purple-100 rounded-lg flex items-center justify-center">
                  <Wrench className="h-6 w-6 text-purple-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Avg Response</p>
                  <p className="text-2xl font-bold">{stats.avgResponseTime}ms</p>
                </div>
                <div className="h-12 w-12 bg-orange-100 rounded-lg flex items-center justify-center">
                  <Clock className="h-6 w-6 text-orange-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Server List */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Servers</h3>
          
          {servers.length === 0 ? (
            <Card>
              <CardContent className="p-8 text-center">
                <Network className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-lg font-semibold mb-2">No servers configured</h3>
                <p className="text-muted-foreground">
                  Add your first MCP server to get started
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {servers.map((server) => (
                <Card key={server.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      {/* Server Info */}
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                          {server.type === "local" ? (
                            <Terminal className="h-5 w-5 text-blue-500" />
                          ) : (
                            <Globe className="h-5 w-5 text-green-500" />
                          )}
                          <div>
                            <div className="font-medium">{server.name}</div>
                            <div className="text-sm text-muted-foreground">
                              {server.description || "No description"}
                            </div>
                          </div>
                        </div>

                        {/* Status */}
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(server.status.status)}
                          <Badge 
                            variant="outline"
                            className={cn("text-white", getStatusColor(server.status.status))}
                          >
                            {server.status.status}
                          </Badge>
                        </div>

                        {/* Health Score */}
                        <div className="flex items-center space-x-2">
                          <div className="text-sm text-muted-foreground">Health</div>
                          <div className="flex items-center space-x-1">
                            <Progress 
                              value={getHealthScore(server.health)} 
                              className="w-16 h-2"
                            />
                            <span className="text-sm font-medium w-8">
                              {getHealthScore(server.health)}%
                            </span>
                          </div>
                        </div>

                        {/* Quick Stats */}
                        <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                          <Tooltip>
                            <TooltipTrigger>
                              <div className="flex items-center space-x-1">
                                <Wrench className="h-3 w-3" />
                                <span>{server.tools.length}</span>
                              </div>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>Available tools</p>
                            </TooltipContent>
                          </Tooltip>

                          <Tooltip>
                            <TooltipTrigger>
                              <div className="flex items-center space-x-1">
                                <Clock className="h-3 w-3" />
                                <span>{formatUptime(server.status.uptime)}</span>
                              </div>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>Uptime</p>
                            </TooltipContent>
                          </Tooltip>

                          {server.health.response_time && (
                            <Tooltip>
                              <TooltipTrigger>
                                <div className="flex items-center space-x-1">
                                  <Zap className="h-3 w-3" />
                                  <span>{server.health.response_time}ms</span>
                                </div>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>Response time</p>
                              </TooltipContent>
                            </Tooltip>
                          )}
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center space-x-2">
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button variant="ghost" size="sm">
                              <Eye className="h-4 w-4" />
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="max-w-4xl">
                            <DialogHeader>
                              <DialogTitle>Server Details: {server.name}</DialogTitle>
                            </DialogHeader>
                            <ServerDetailsView server={server} />
                          </DialogContent>
                        </Dialog>

                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={() => onEditServer(server.id)}
                        >
                          <Settings className="h-4 w-4" />
                        </Button>

                        {server.status.status === "connected" ? (
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => onStopServer(server.id)}
                          >
                            <Square className="h-4 w-4" />
                          </Button>
                        ) : (
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => onStartServer(server.id)}
                          >
                            <Play className="h-4 w-4" />
                          </Button>
                        )}

                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={() => onDeleteServer(server.id)}
                          className="text-red-600 hover:text-red-700"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>

                    {/* Error Message */}
                    {server.status.last_error && (
                      <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                        <div className="flex items-center space-x-2">
                          <AlertTriangle className="h-4 w-4" />
                          <span className="font-medium">Error:</span>
                          <span>{server.status.last_error}</span>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </TooltipProvider>
  );
}

// Server Details View Component
function ServerDetailsView({ server }: { server: MCPServerWithStatus }) {
  const [activeTab, setActiveTab] = useState<"overview" | "metrics" | "tools" | "resources">("overview");

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-muted p-1 rounded-lg">
        {["overview", "metrics", "tools", "resources"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as any)}
            className={cn(
              "px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
              activeTab === tab
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === "overview" && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Type:</span>
                    <Badge variant="outline">{server.type}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Enabled:</span>
                    <Badge variant={server.enabled ? "default" : "secondary"}>
                      {server.enabled ? "Yes" : "No"}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Auto-restart:</span>
                    <Badge variant={server.autoRestart ? "default" : "secondary"}>
                      {server.autoRestart ? "Yes" : "No"}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Timeout:</span>
                    <span className="text-sm">{server.timeout}ms</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Status</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Status:</span>
                    <Badge variant="outline">
                      {server.status.status}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">PID:</span>
                    <span className="text-sm">{server.status.process_id || "N/A"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Uptime:</span>
                    <span className="text-sm">{formatUptime(server.status.uptime)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Restarts:</span>
                    <span className="text-sm">{server.status.restart_count || 0}</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Connection Details */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Connection</CardTitle>
              </CardHeader>
              <CardContent>
                {server.type === "local" ? (
                  <div className="space-y-2">
                    <div>
                      <Label className="text-xs text-muted-foreground">Command</Label>
                      <div className="mt-1 p-2 bg-muted rounded text-sm font-mono">
                        {server.command?.join(" ")}
                      </div>
                    </div>
                    {server.workingDirectory && (
                      <div>
                        <Label className="text-xs text-muted-foreground">Working Directory</Label>
                        <div className="mt-1 p-2 bg-muted rounded text-sm font-mono">
                          {server.workingDirectory}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div>
                      <Label className="text-xs text-muted-foreground">URL</Label>
                      <div className="mt-1 p-2 bg-muted rounded text-sm font-mono">
                        {server.url}
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === "metrics" && (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Requests</p>
                      <p className="text-2xl font-bold">{server.metrics.requests_total}</p>
                    </div>
                    <BarChart3 className="h-8 w-8 text-blue-500" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Success Rate</p>
                      <p className="text-2xl font-bold">
                        {Math.round((server.metrics.requests_successful / server.metrics.requests_total) * 100)}%
                      </p>
                    </div>
                    <TrendingUp className="h-8 w-8 text-green-500" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Avg Response</p>
                      <p className="text-2xl font-bold">{server.metrics.avg_response_time}ms</p>
                    </div>
                    <Clock className="h-8 w-8 text-orange-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-xs text-muted-foreground">Memory Usage</Label>
                    <div className="mt-1 text-sm">
                      {formatBytes(server.status.memory_usage)}
                    </div>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">CPU Usage</Label>
                    <div className="mt-1 text-sm">
                      {server.status.cpu_usage?.toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-xs text-muted-foreground">Min Response Time</Label>
                    <div className="mt-1 text-sm">
                      {server.metrics.min_response_time}ms
                    </div>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Max Response Time</Label>
                    <div className="mt-1 text-sm">
                      {server.metrics.max_response_time}ms
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === "tools" && (
          <div className="space-y-4">
            {server.tools.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center">
                  <Wrench className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">No tools available</h3>
                  <p className="text-muted-foreground">
                    This server hasn&apos;t registered any tools yet
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-3">
                {server.tools.map((tool) => (
                  <Card key={tool.id}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <Wrench className="h-5 w-5 text-blue-500" />
                          <div>
                            <div className="font-medium">{tool.name}</div>
                            <div className="text-sm text-muted-foreground">
                              {tool.description}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge variant={tool.enabled ? "default" : "secondary"}>
                            {tool.enabled ? "Enabled" : "Disabled"}
                          </Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === "resources" && (
          <div className="space-y-4">
            {server.resources.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center">
                  <ExternalLink className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">No resources available</h3>
                  <p className="text-muted-foreground">
                    This server hasn&apos;t registered any resources yet
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-3">
                {server.resources.map((resource) => (
                  <Card key={resource.id}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <ExternalLink className="h-5 w-5 text-green-500" />
                          <div>
                            <div className="font-medium">{resource.name}</div>
                            <div className="text-sm text-muted-foreground">
                              {resource.description}
                            </div>
                            <div className="text-xs text-muted-foreground font-mono">
                              {resource.uri}
                            </div>
                          </div>
                        </div>
                        <Badge variant="outline">{resource.type}</Badge>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}