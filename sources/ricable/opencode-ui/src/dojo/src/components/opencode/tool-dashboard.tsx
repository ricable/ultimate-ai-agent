/**
 * Enhanced Tool Dashboard - Comprehensive tool execution monitoring and management
 * Features real-time monitoring, approval workflows, and advanced analytics
 */

"use client";

import React, { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Zap, 
  CheckCircle, 
  XCircle, 
  Clock,
  AlertCircle,
  AlertTriangle,
  Play,
  Square,
  Settings,
  RefreshCw,
  Filter,
  Package,
  Shield,
  Activity,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Bug,
  FileText,
  Search,
  Globe,
  Code,
  Terminal,
  Database,
  Eye,
  EyeOff,
  Download,
  Upload,
  Trash2,
  Plus,
  Minus,
  Pause,
  RotateCcw,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Layers,
  Cpu,
  MemoryStick,
  Network
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/lib/session-store';
import { ToolApprovalDialog } from '@/components/opencode/tool-approval-dialog';

interface EnhancedTool {
  id: string;
  name: string;
  description: string;
  category: 'file' | 'system' | 'network' | 'development' | 'ai' | 'mcp';
  enabled: boolean;
  requiresApproval: boolean;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  provider?: string; // MCP server name
  usage: {
    executions: number;
    lastUsed?: number;
    avgResponseTime: number;
    successRate: number;
  };
  permissions: string[];
  metadata?: Record<string, any>;
}

interface ToolExecutionDetails {
  id: string;
  toolId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  input: any;
  output?: any;
  error?: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  approvedBy?: string;
  riskAssessment?: {
    level: 'low' | 'medium' | 'high' | 'critical';
    concerns: string[];
    recommendations: string[];
  };
  sessionId: string;
  provider?: string;
}

interface ToolMetrics {
  totalExecutions: number;
  successfulExecutions: number;
  failedExecutions: number;
  pendingExecutions: number;
  avgResponseTime: number;
  topTools: { name: string; count: number }[];
  executionTrend: { timestamp: number; count: number }[];
  errorRate: number;
}

export const ToolDashboard: React.FC = () => {
  const {
    availableTools,
    toolExecutions,
    pendingApprovals,
    isLoadingTools,
    actions
  } = useSessionStore();
  
  const [selectedTab, setSelectedTab] = useState('overview');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showOnlyEnabled, setShowOnlyEnabled] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedExecution, setSelectedExecution] = useState<ToolExecutionDetails | null>(null);
  const [showApprovalDialog, setShowApprovalDialog] = useState(false);
  const [selectedApprovalRequest, setSelectedApprovalRequest] = useState<any>(null);
  const [executionHistory, setExecutionHistory] = useState<ToolExecutionDetails[]>([]);
  const [toolMetrics, setToolMetrics] = useState<ToolMetrics | null>(null);

  // Enhanced mock data for demonstration
  const enhancedTools: EnhancedTool[] = useMemo(() => [
    {
      id: 'file-glob',
      name: 'File Pattern Search',
      description: 'Find files matching glob patterns across your codebase',
      category: 'file',
      enabled: true,
      requiresApproval: false,
      riskLevel: 'low',
      usage: {
        executions: 156,
        lastUsed: Date.now() - 3600000,
        avgResponseTime: 120,
        successRate: 98.7
      },
      permissions: ['read:filesystem'],
      metadata: { version: '1.0.0', author: 'OpenCode' }
    },
    {
      id: 'bash-exec',
      name: 'Command Execution',
      description: 'Execute shell commands with security controls',
      category: 'system',
      enabled: true,
      requiresApproval: true,
      riskLevel: 'high',
      usage: {
        executions: 45,
        lastUsed: Date.now() - 1800000,
        avgResponseTime: 2300,
        successRate: 91.1
      },
      permissions: ['execute:shell', 'read:filesystem'],
      metadata: { version: '2.1.0', timeout: 30000 }
    },
    {
      id: 'mcp-puppeteer',
      name: 'Browser Automation',
      description: 'Control browsers for web automation tasks',
      category: 'mcp',
      enabled: true,
      requiresApproval: true,
      riskLevel: 'medium',
      provider: 'puppeteer',
      usage: {
        executions: 23,
        lastUsed: Date.now() - 7200000,
        avgResponseTime: 1800,
        successRate: 95.7
      },
      permissions: ['network:browser', 'execute:automation'],
      metadata: { mcpServer: 'puppeteer', version: '3.0.1' }
    },
    {
      id: 'ai-agent',
      name: 'AI Sub-Agent',
      description: 'Delegate tasks to specialized AI agents',
      category: 'ai',
      enabled: true,
      requiresApproval: false,
      riskLevel: 'low',
      usage: {
        executions: 89,
        lastUsed: Date.now() - 900000,
        avgResponseTime: 4500,
        successRate: 97.8
      },
      permissions: ['ai:subtask'],
      metadata: { maxTokens: 8000, timeout: 60000 }
    },
    {
      id: 'network-fetch',
      name: 'URL Fetcher',
      description: 'Fetch and process content from URLs',
      category: 'network',
      enabled: true,
      requiresApproval: false,
      riskLevel: 'medium',
      usage: {
        executions: 67,
        lastUsed: Date.now() - 2700000,
        avgResponseTime: 850,
        successRate: 93.4
      },
      permissions: ['network:http'],
      metadata: { timeout: 10000, followRedirects: true }
    }
  ], []);

  // Enhanced execution history
  const mockExecutions: ToolExecutionDetails[] = useMemo(() => [
    {
      id: 'exec-001',
      toolId: 'bash-exec',
      status: 'pending',
      input: { command: 'npm test', timeout: 30000 },
      startTime: Date.now() - 30000,
      sessionId: 'session-1',
      riskAssessment: {
        level: 'medium',
        concerns: ['Command execution', 'Potential file system changes'],
        recommendations: ['Review command before approval', 'Check for destructive operations']
      }
    },
    {
      id: 'exec-002',
      toolId: 'file-glob',
      status: 'completed',
      input: { pattern: '**/*.{ts,tsx}', path: '/src' },
      output: { files: ['src/index.ts', 'src/app.tsx'], count: 2 },
      startTime: Date.now() - 300000,
      endTime: Date.now() - 299800,
      duration: 200,
      sessionId: 'session-1'
    },
    {
      id: 'exec-003',
      toolId: 'mcp-puppeteer',
      status: 'running',
      input: { url: 'https://example.com', action: 'screenshot' },
      startTime: Date.now() - 5000,
      sessionId: 'session-2',
      provider: 'puppeteer'
    }
  ], []);

  // Calculate metrics
  const metrics: ToolMetrics = useMemo(() => {
    const total = enhancedTools.reduce((sum, tool) => sum + tool.usage.executions, 0);
    const successful = enhancedTools.reduce((sum, tool) => 
      sum + Math.floor(tool.usage.executions * (tool.usage.successRate / 100)), 0);
    const failed = total - successful;
    const pending = mockExecutions.filter(e => e.status === 'pending').length;
    const avgResponse = enhancedTools.reduce((sum, tool, _, arr) => 
      sum + tool.usage.avgResponseTime / arr.length, 0);
    
    return {
      totalExecutions: total,
      successfulExecutions: successful,
      failedExecutions: failed,
      pendingExecutions: pending,
      avgResponseTime: Math.round(avgResponse),
      topTools: enhancedTools
        .sort((a, b) => b.usage.executions - a.usage.executions)
        .slice(0, 5)
        .map(tool => ({ name: tool.name, count: tool.usage.executions })),
      executionTrend: Array.from({ length: 24 }, (_, i) => ({
        timestamp: Date.now() - (23 - i) * 3600000,
        count: Math.floor(Math.random() * 20) + 5
      })),
      errorRate: ((failed / total) * 100) || 0
    };
  }, [enhancedTools, mockExecutions]);

  useEffect(() => {
    actions.loadTools();
    actions.loadToolExecutions();
    setExecutionHistory(mockExecutions);
    setToolMetrics(metrics);
  }, []);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      actions.loadToolExecutions();
    }, 30000);

    return () => clearInterval(interval);
  }, [autoRefresh, actions]);

  // Filter tools based on search and category
  const filteredTools = useMemo(() => {
    return enhancedTools.filter(tool => {
      const matchesCategory = selectedCategory === 'all' || tool.category === selectedCategory;
      const matchesSearch = tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           tool.description.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesEnabled = !showOnlyEnabled || tool.enabled;
      return matchesCategory && matchesSearch && matchesEnabled;
    });
  }, [enhancedTools, selectedCategory, searchQuery, showOnlyEnabled]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'running':
        return <Play className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'cancelled':
        return <Square className="h-4 w-4 text-gray-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
      case 'running':
        return 'bg-blue-500';
      case 'pending':
        return 'bg-yellow-500';
      case 'cancelled':
        return 'bg-gray-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'high':
        return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'critical':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'file':
        return <FileText className="h-4 w-4" />;
      case 'system':
        return <Terminal className="h-4 w-4" />;
      case 'network':
        return <Globe className="h-4 w-4" />;
      case 'development':
        return <Code className="h-4 w-4" />;
      case 'ai':
        return <Zap className="h-4 w-4" />;
      case 'mcp':
        return <Package className="h-4 w-4" />;
      default:
        return <Settings className="h-4 w-4" />;
    }
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  const formatLastUsed = (timestamp?: number) => {
    if (!timestamp) return 'Never';
    const diff = Date.now() - timestamp;
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return `${Math.floor(diff / 86400000)}d ago`;
  };

  const handleApproveExecution = async (executionId: string, conditions?: any) => {
    try {
      setExecutionHistory(prev => prev.map(exec => 
        exec.id === executionId 
          ? { ...exec, status: 'running', approvedBy: 'user' }
          : exec
      ));
      
      // Simulate execution completion
      setTimeout(() => {
        setExecutionHistory(prev => prev.map(exec => 
          exec.id === executionId 
            ? { 
                ...exec, 
                status: 'completed', 
                endTime: Date.now(),
                duration: Date.now() - exec.startTime,
                output: { result: 'success', message: 'Operation completed successfully' }
              }
            : exec
        ));
      }, 2000);
      
      await actions.approveToolExecution(executionId);
    } catch (error) {
      console.error('Failed to approve tool execution:', error);
    }
  };

  const handleDenyExecution = async (executionId: string, reason: string) => {
    try {
      setExecutionHistory(prev => prev.map(exec => 
        exec.id === executionId 
          ? { ...exec, status: 'cancelled', error: reason }
          : exec
      ));
      
      await actions.cancelToolExecution(executionId);
    } catch (error) {
      console.error('Failed to deny tool execution:', error);
    }
  };

  const handleToolToggle = (toolId: string) => {
    // In real implementation, this would call an API
    console.log(`Toggle tool ${toolId}`);
  };

  const handleShowApprovalDialog = (execution: ToolExecutionDetails) => {
    const tool = enhancedTools.find(t => t.id === execution.toolId);
    if (tool && execution.riskAssessment) {
      setSelectedApprovalRequest({
        id: execution.id,
        toolId: tool.id,
        toolName: tool.name,
        description: tool.description,
        params: execution.input,
        riskLevel: execution.riskAssessment.level,
        securityWarnings: execution.riskAssessment.concerns,
        requestedAt: execution.startTime,
        estimatedDuration: tool.usage.avgResponseTime
      });
      setShowApprovalDialog(true);
    }
  };

  const categories = [
    { id: 'all', name: 'All Tools', icon: <Layers className="h-4 w-4" /> },
    { id: 'file', name: 'File Operations', icon: <FileText className="h-4 w-4" /> },
    { id: 'system', name: 'System Tools', icon: <Terminal className="h-4 w-4" /> },
    { id: 'network', name: 'Network', icon: <Globe className="h-4 w-4" /> },
    { id: 'development', name: 'Development', icon: <Code className="h-4 w-4" /> },
    { id: 'ai', name: 'AI Tools', icon: <Zap className="h-4 w-4" /> },
    { id: 'mcp', name: 'MCP Tools', icon: <Package className="h-4 w-4" /> }
  ];

  if (isLoadingTools) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center space-y-2">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <p className="text-muted-foreground">Loading tool dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <div className="flex flex-col h-full">
        {/* Enhanced Header */}
        <div className="p-6 border-b border-border bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                <Zap className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Enhanced Tool Dashboard</h1>
                <p className="text-muted-foreground">
                  Comprehensive tool monitoring with real-time analytics and security controls
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={cn(autoRefresh && "bg-blue-50 border-blue-200 dark:bg-blue-950")}
              >
                <Activity className="h-4 w-4 mr-2" />
                Auto-refresh {autoRefresh ? "ON" : "OFF"}
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  actions.loadTools();
                  actions.loadToolExecutions();
                }}
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
              <Button 
                variant="outline"
                onClick={() => {/* Navigate to MCP view */}}
              >
                <Package className="h-4 w-4 mr-2" />
                MCP Servers
              </Button>
              <Button variant="outline">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        </div>

        {/* Enhanced Stats Overview */}
        {toolMetrics && (
          <div className="px-6 py-4 bg-muted/30">
            <div className="grid grid-cols-5 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Total Executions</p>
                      <p className="text-2xl font-bold">{toolMetrics.totalExecutions}</p>
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
                      <p className="text-2xl font-bold text-green-600">
                        {((toolMetrics.successfulExecutions / toolMetrics.totalExecutions) * 100).toFixed(1)}%
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
                      <p className="text-2xl font-bold">{toolMetrics.avgResponseTime}ms</p>
                    </div>
                    <Clock className="h-8 w-8 text-orange-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Pending Approvals</p>
                      <p className="text-2xl font-bold text-yellow-600">{toolMetrics.pendingExecutions}</p>
                    </div>
                    <Shield className="h-8 w-8 text-yellow-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Error Rate</p>
                      <p className="text-2xl font-bold text-red-600">{toolMetrics.errorRate.toFixed(1)}%</p>
                    </div>
                    <AlertCircle className="h-8 w-8 text-red-500" />
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="overview">
                <Activity className="h-4 w-4 mr-2" />
                Overview
              </TabsTrigger>
              <TabsTrigger value="tools">
                <Zap className="h-4 w-4 mr-2" />
                Tools ({filteredTools.length})
              </TabsTrigger>
              <TabsTrigger value="executions">
                <Play className="h-4 w-4 mr-2" />
                Executions
                {executionHistory.filter(e => e.status === 'pending').length > 0 && (
                  <Badge className="ml-2" variant="destructive">
                    {executionHistory.filter(e => e.status === 'pending').length}
                  </Badge>
                )}
              </TabsTrigger>
              <TabsTrigger value="analytics">
                <BarChart3 className="h-4 w-4 mr-2" />
                Analytics
              </TabsTrigger>
              <TabsTrigger value="security">
                <Shield className="h-4 w-4 mr-2" />
                Security
              </TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Quick Actions */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Zap className="h-5 w-5" />
                      <span>Quick Actions</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Button className="w-full justify-start" variant="outline">
                      <Plus className="h-4 w-4 mr-2" />
                      Install New Tool
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <Package className="h-4 w-4 mr-2" />
                      Manage MCP Servers
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <Shield className="h-4 w-4 mr-2" />
                      Review Security Settings
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <Download className="h-4 w-4 mr-2" />
                      Export Tool Configurations
                    </Button>
                  </CardContent>
                </Card>

                {/* Recent Activity */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Activity className="h-5 w-5" />
                      <span>Recent Activity</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {executionHistory.slice(0, 5).map((execution) => {
                        const tool = enhancedTools.find(t => t.id === execution.toolId);
                        return (
                          <div key={execution.id} className="flex items-center justify-between py-2 border-b last:border-b-0">
                            <div className="flex items-center space-x-3">
                              {getStatusIcon(execution.status)}
                              <div>
                                <div className="font-medium text-sm">{tool?.name}</div>
                                <div className="text-xs text-muted-foreground">
                                  {formatDate(execution.startTime)}
                                </div>
                              </div>
                            </div>
                            <Badge variant="outline">{execution.status}</Badge>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              {/* Top Tools */}
              {toolMetrics && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <TrendingUp className="h-5 w-5" />
                      <span>Most Used Tools</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {toolMetrics.topTools.map((tool, index) => {
                        const toolDetails = enhancedTools.find(t => t.name === tool.name);
                        return (
                          <div key={tool.name} className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              <div className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-xs font-bold">
                                {index + 1}
                              </div>
                              <div className="flex items-center space-x-2">
                                {toolDetails && getCategoryIcon(toolDetails.category)}
                                <span className="font-medium">{tool.name}</span>
                              </div>
                            </div>
                            <div className="flex items-center space-x-2">
                              <span className="text-sm text-muted-foreground">{tool.count} executions</span>
                              <Progress value={(tool.count / toolMetrics.topTools[0].count) * 100} className="w-16" />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* Enhanced Tools Tab */}
            <TabsContent value="tools" className="space-y-6">
              {/* Filters */}
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between space-x-4">
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-2">
                        <Filter className="h-4 w-4" />
                        <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                          <SelectTrigger className="w-48">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {categories.map(cat => (
                              <SelectItem key={cat.id} value={cat.id}>
                                <div className="flex items-center space-x-2">
                                  {cat.icon}
                                  <span>{cat.name}</span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <Input
                        placeholder="Search tools..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-64"
                      />
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-2">
                        <label className="text-sm font-medium">Enabled only</label>
                        <Switch
                          checked={showOnlyEnabled}
                          onCheckedChange={setShowOnlyEnabled}
                        />
                      </div>
                      <Button variant="outline" size="sm">
                        <Plus className="h-4 w-4 mr-2" />
                        Add Tool
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Tools Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredTools.map((tool) => (
                  <motion.div
                    key={tool.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="group"
                  >
                    <Card className={cn(
                      "transition-all duration-200 hover:shadow-lg",
                      tool.enabled ? "border-green-200 bg-green-50/50 dark:bg-green-950/20" : "border-gray-200"
                    )}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            {getCategoryIcon(tool.category)}
                            <CardTitle className="text-base">{tool.name}</CardTitle>
                          </div>
                          <div className="flex items-center space-x-2">
                            {tool.provider && (
                              <Tooltip>
                                <TooltipTrigger>
                                  <Badge variant="outline" className="text-xs">
                                    {tool.provider}
                                  </Badge>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p>MCP Server: {tool.provider}</p>
                                </TooltipContent>
                              </Tooltip>
                            )}
                            <Switch
                              checked={tool.enabled}
                              onCheckedChange={() => handleToolToggle(tool.id)}
                            />
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <p className="text-sm text-muted-foreground">
                          {tool.description}
                        </p>
                        
                        <div className="flex items-center justify-between">
                          <Badge variant="outline">{tool.category}</Badge>
                          <Badge 
                            variant="outline"
                            className={getRiskColor(tool.riskLevel)}
                          >
                            {tool.riskLevel} risk
                          </Badge>
                        </div>

                        {tool.requiresApproval && (
                          <div className="flex items-center space-x-1 text-xs text-yellow-600">
                            <Shield className="h-3 w-3" />
                            <span>Requires approval</span>
                          </div>
                        )}

                        <Separator />

                        <div className="space-y-2">
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">Executions:</span>
                            <span className="font-medium">{tool.usage.executions}</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">Success Rate:</span>
                            <span className="font-medium text-green-600">{tool.usage.successRate}%</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">Avg Response:</span>
                            <span className="font-medium">{tool.usage.avgResponseTime}ms</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">Last Used:</span>
                            <span className="font-medium">{formatLastUsed(tool.usage.lastUsed)}</span>
                          </div>
                        </div>

                        <div className="flex items-center justify-between pt-2">
                          <div className="flex space-x-1">
                            <Tooltip>
                              <TooltipTrigger>
                                <Button variant="ghost" size="sm">
                                  <Eye className="h-3 w-3" />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>View Details</p>
                              </TooltipContent>
                            </Tooltip>
                            <Tooltip>
                              <TooltipTrigger>
                                <Button variant="ghost" size="sm">
                                  <Settings className="h-3 w-3" />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>Configure Tool</p>
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <Badge variant="secondary" className="text-xs">
                            v{tool.metadata?.version || '1.0.0'}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </TabsContent>

            {/* Enhanced Executions Tab */}
            <TabsContent value="executions" className="space-y-6">
              {/* Pending Approvals Section */}
              {executionHistory.filter(e => e.status === 'pending').length > 0 && (
                <Card className="border-yellow-200 bg-yellow-50 dark:bg-yellow-950/20">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-yellow-700 dark:text-yellow-400">
                      <AlertTriangle className="h-5 w-5" />
                      <span>Pending Approvals ({executionHistory.filter(e => e.status === 'pending').length})</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {executionHistory.filter(e => e.status === 'pending').map((execution) => {
                      const tool = enhancedTools.find(t => t.id === execution.toolId);
                      
                      return (
                        <motion.div
                          key={execution.id}
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="border rounded-lg p-4 bg-white dark:bg-gray-800 shadow-sm"
                        >
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center space-x-3">
                              <div className="p-2 bg-yellow-100 rounded-lg">
                                {tool && getCategoryIcon(tool.category)}
                              </div>
                              <div>
                                <div className="font-medium">{tool?.name || execution.toolId}</div>
                                <div className="text-sm text-muted-foreground">
                                  Requested {formatDate(execution.startTime)}
                                </div>
                              </div>
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              {execution.riskAssessment && (
                                <Badge 
                                  variant="outline"
                                  className={getRiskColor(execution.riskAssessment.level)}
                                >
                                  {execution.riskAssessment.level} risk
                                </Badge>
                              )}
                              <div className="flex space-x-2">
                                <Button
                                  size="sm"
                                  onClick={() => handleShowApprovalDialog(execution)}
                                  className="bg-green-600 hover:bg-green-700"
                                >
                                  <CheckCircle className="h-4 w-4 mr-1" />
                                  Review & Approve
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => handleDenyExecution(execution.id, 'Denied by user')}
                                >
                                  <XCircle className="h-4 w-4 mr-1" />
                                  Deny
                                </Button>
                              </div>
                            </div>
                          </div>
                          
                          {execution.riskAssessment && (
                            <div className="bg-yellow-50 dark:bg-yellow-950/40 p-3 rounded border">
                              <div className="text-sm font-medium mb-2">Security Assessment</div>
                              <div className="space-y-1">
                                {execution.riskAssessment.concerns.map((concern, idx) => (
                                  <div key={idx} className="text-xs text-muted-foreground flex items-center space-x-1">
                                    <AlertTriangle className="h-3 w-3 text-orange-500" />
                                    <span>{concern}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          <div className="mt-3">
                            <div className="text-sm font-medium mb-2">Input Parameters:</div>
                            <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded overflow-x-auto max-h-32">
                              {JSON.stringify(execution.input, null, 2)}
                            </pre>
                          </div>
                        </motion.div>
                      );
                    })}
                  </CardContent>
                </Card>
              )}

              {/* Execution History */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center space-x-2">
                      <Activity className="h-5 w-5" />
                      <span>Execution History</span>
                    </CardTitle>
                    <div className="flex items-center space-x-2">
                      <Select defaultValue="all">
                        <SelectTrigger className="w-32">
                          <SelectValue placeholder="Filter status" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Status</SelectItem>
                          <SelectItem value="completed">Completed</SelectItem>
                          <SelectItem value="running">Running</SelectItem>
                          <SelectItem value="failed">Failed</SelectItem>
                          <SelectItem value="cancelled">Cancelled</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button variant="outline" size="sm">
                        <Download className="h-4 w-4 mr-2" />
                        Export
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {executionHistory.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-32 text-center">
                      <Activity className="h-12 w-12 text-muted-foreground mb-4" />
                      <h3 className="text-lg font-medium mb-2">No executions yet</h3>
                      <p className="text-muted-foreground">
                        Tool executions will appear here when AI agents use tools
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {executionHistory.slice(0, 20).map((execution) => {
                        const tool = enhancedTools.find(t => t.id === execution.toolId);
                        
                        return (
                          <motion.div
                            key={execution.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="border rounded-lg p-4 hover:shadow-sm transition-shadow"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                {getStatusIcon(execution.status)}
                                <div className="flex items-center space-x-2">
                                  {tool && getCategoryIcon(tool.category)}
                                  <div>
                                    <div className="font-medium">
                                      {tool?.name || execution.toolId}
                                    </div>
                                    <div className="text-sm text-muted-foreground">
                                      {formatDate(execution.startTime)}
                                      {execution.duration && ` • ${formatDuration(execution.duration)}`}
                                      {execution.provider && ` • via ${execution.provider}`}
                                    </div>
                                  </div>
                                </div>
                              </div>
                              
                              <div className="flex items-center space-x-2">
                                <Badge 
                                  variant="outline"
                                  className={cn(
                                    execution.status === 'completed' && 'border-green-500 text-green-700',
                                    execution.status === 'failed' && 'border-red-500 text-red-700',
                                    execution.status === 'running' && 'border-blue-500 text-blue-700',
                                    execution.status === 'pending' && 'border-yellow-500 text-yellow-700'
                                  )}
                                >
                                  {execution.status}
                                </Badge>
                                
                                <Dialog>
                                  <DialogTrigger asChild>
                                    <Button variant="ghost" size="sm">
                                      <Eye className="h-4 w-4" />
                                    </Button>
                                  </DialogTrigger>
                                  <DialogContent className="max-w-3xl">
                                    <DialogHeader>
                                      <DialogTitle>Execution Details - {tool?.name}</DialogTitle>
                                    </DialogHeader>
                                    <div className="space-y-4">
                                      <div className="grid grid-cols-2 gap-4">
                                        <div>
                                          <div className="text-sm font-medium mb-1">Status</div>
                                          <div className="flex items-center space-x-2">
                                            {getStatusIcon(execution.status)}
                                            <Badge variant="outline">{execution.status}</Badge>
                                          </div>
                                        </div>
                                        <div>
                                          <div className="text-sm font-medium mb-1">Duration</div>
                                          <div className="text-sm">
                                            {execution.duration 
                                              ? formatDuration(execution.duration)
                                              : execution.status === 'running' ? 'Running...' : 'N/A'
                                            }
                                          </div>
                                        </div>
                                      </div>
                                      
                                      <div>
                                        <div className="text-sm font-medium mb-2">Input</div>
                                        <pre className="text-xs bg-muted p-3 rounded overflow-x-auto max-h-48">
                                          {JSON.stringify(execution.input, null, 2)}
                                        </pre>
                                      </div>
                                      
                                      {execution.output && (
                                        <div>
                                          <div className="text-sm font-medium mb-2">Output</div>
                                          <pre className="text-xs bg-muted p-3 rounded overflow-x-auto max-h-48">
                                            {JSON.stringify(execution.output, null, 2)}
                                          </pre>
                                        </div>
                                      )}
                                      
                                      {execution.error && (
                                        <div>
                                          <div className="text-sm font-medium mb-2">Error</div>
                                          <div className="text-sm text-red-600 bg-red-50 p-3 rounded">
                                            {execution.error}
                                          </div>
                                        </div>
                                      )}
                                    </div>
                                  </DialogContent>
                                </Dialog>
                              </div>
                            </div>
                            
                            {execution.error && (
                              <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                                <div className="flex items-center space-x-2">
                                  <XCircle className="h-4 w-4" />
                                  <span>{execution.error}</span>
                                </div>
                              </div>
                            )}
                          </motion.div>
                        );
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Analytics Tab */}
            <TabsContent value="analytics" className="space-y-6">
              {toolMetrics && (
                <div className="space-y-6">
                  {/* Performance Overview */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Execution Trend</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold mb-2">{toolMetrics.totalExecutions}</div>
                        <div className="text-xs text-muted-foreground">Total executions this month</div>
                        <div className="mt-3 h-16 bg-muted/30 rounded flex items-end space-x-1">
                          {toolMetrics.executionTrend.slice(-7).map((point, idx) => (
                            <div 
                              key={idx} 
                              className="bg-blue-500 rounded-t flex-1" 
                              style={{ height: `${(point.count / Math.max(...toolMetrics.executionTrend.map(p => p.count))) * 100}%` }}
                            />
                          ))}
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Response Times</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold mb-2">{toolMetrics.avgResponseTime}ms</div>
                        <div className="text-xs text-muted-foreground">Average response time</div>
                        <div className="mt-3 space-y-2">
                          {enhancedTools.slice(0, 3).map((tool) => (
                            <div key={tool.id} className="flex items-center justify-between text-xs">
                              <span className="truncate">{tool.name}</span>
                              <span className="font-medium">{tool.usage.avgResponseTime}ms</span>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Success Rate</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold mb-2 text-green-600">
                          {((toolMetrics.successfulExecutions / toolMetrics.totalExecutions) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-muted-foreground">Overall success rate</div>
                        <div className="mt-3">
                          <Progress value={(toolMetrics.successfulExecutions / toolMetrics.totalExecutions) * 100} className="h-2" />
                          <div className="flex justify-between text-xs text-muted-foreground mt-1">
                            <span>{toolMetrics.successfulExecutions} success</span>
                            <span>{toolMetrics.failedExecutions} failed</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Tool Performance Breakdown */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Tool Performance Breakdown</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {enhancedTools.map((tool) => (
                          <div key={tool.id} className="flex items-center justify-between py-3 border-b last:border-b-0">
                            <div className="flex items-center space-x-3">
                              {getCategoryIcon(tool.category)}
                              <div>
                                <div className="font-medium">{tool.name}</div>
                                <div className="text-sm text-muted-foreground">{tool.usage.executions} executions</div>
                              </div>
                            </div>
                            <div className="flex items-center space-x-4 text-sm">
                              <div className="text-center">
                                <div className="font-medium text-green-600">{tool.usage.successRate}%</div>
                                <div className="text-xs text-muted-foreground">Success</div>
                              </div>
                              <div className="text-center">
                                <div className="font-medium">{tool.usage.avgResponseTime}ms</div>
                                <div className="text-xs text-muted-foreground">Avg Time</div>
                              </div>
                              <div className="text-center">
                                <div className="font-medium">{formatLastUsed(tool.usage.lastUsed)}</div>
                                <div className="text-xs text-muted-foreground">Last Used</div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </TabsContent>

            {/* Security Tab */}
            <TabsContent value="security" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Security Overview */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Shield className="h-5 w-5" />
                      <span>Security Overview</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Tools requiring approval:</span>
                      <Badge variant="outline">
                        {enhancedTools.filter(t => t.requiresApproval).length} / {enhancedTools.length}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">High/Critical risk tools:</span>
                      <Badge variant="outline" className="text-red-600">
                        {enhancedTools.filter(t => ['high', 'critical'].includes(t.riskLevel)).length}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">MCP tools:</span>
                      <Badge variant="outline">
                        {enhancedTools.filter(t => t.category === 'mcp').length}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Pending approvals:</span>
                      <Badge variant="outline" className="text-yellow-600">
                        {executionHistory.filter(e => e.status === 'pending').length}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                {/* Risk Assessment */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <AlertTriangle className="h-5 w-5" />
                      <span>Risk Assessment</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {['low', 'medium', 'high', 'critical'].map((level) => {
                        const count = enhancedTools.filter(t => t.riskLevel === level).length;
                        const percentage = (count / enhancedTools.length) * 100;
                        
                        return (
                          <div key={level} className="space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-sm capitalize">{level} Risk</span>
                              <span className="text-sm font-medium">{count} tools</span>
                            </div>
                            <Progress value={percentage} className="h-2" />
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Tools by Risk Level */}
              <Card>
                <CardHeader>
                  <CardTitle>Tools by Risk Level</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {['critical', 'high', 'medium', 'low'].map((riskLevel) => {
                      const toolsInLevel = enhancedTools.filter(t => t.riskLevel === riskLevel);
                      if (toolsInLevel.length === 0) return null;
                      
                      return (
                        <div key={riskLevel}>
                          <h4 className="font-medium mb-3 flex items-center space-x-2">
                            <Badge variant="outline" className={getRiskColor(riskLevel)}>
                              {riskLevel.toUpperCase()} RISK
                            </Badge>
                            <span>({toolsInLevel.length} tools)</span>
                          </h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {toolsInLevel.map((tool) => (
                              <div key={tool.id} className="flex items-center justify-between p-3 border rounded">
                                <div className="flex items-center space-x-2">
                                  {getCategoryIcon(tool.category)}
                                  <div>
                                    <div className="font-medium text-sm">{tool.name}</div>
                                    <div className="text-xs text-muted-foreground">{tool.category}</div>
                                  </div>
                                </div>
                                <div className="flex items-center space-x-2">
                                  {tool.requiresApproval && (
                                    <Tooltip>
                                      <TooltipTrigger>
                                        <Shield className="h-4 w-4 text-yellow-500" />
                                      </TooltipTrigger>
                                      <TooltipContent>
                                        <p>Requires approval</p>
                                      </TooltipContent>
                                    </Tooltip>
                                  )}
                                  <Switch checked={tool.enabled} disabled />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

          </Tabs>
        </div>

        {/* Tool Approval Dialog */}
        <ToolApprovalDialog
          request={selectedApprovalRequest}
          open={showApprovalDialog}
          onApprove={handleApproveExecution}
          onDeny={handleDenyExecution}
          onClose={() => {
            setShowApprovalDialog(false);
            setSelectedApprovalRequest(null);
          }}
        />
      </div>
    </TooltipProvider>

  );
};