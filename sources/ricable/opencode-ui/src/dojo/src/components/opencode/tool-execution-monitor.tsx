/**
 * Tool Execution Monitor - Real-time monitoring of tool executions with enhanced security
 * Provides comprehensive oversight of all tool activities across sessions
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
  Pause,
  Play,
  Square,
  Eye,
  EyeOff,
  Filter,
  Search,
  Download,
  RefreshCw,
  Shield,
  Zap,
  BarChart3,
  Terminal,
  Globe,
  FileText,
  Code,
  Package,
  User,
  Calendar,
  Timer,
  Cpu,
  MemoryStick,
  HardDrive,
  NetworkIcon,
  AlertCircleIcon,
  TrendingUp,
  TrendingDown,
  Maximize2,
  Minimize2
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
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
  TooltipProvider, 
  TooltipTrigger 
} from '@/components/ui/tooltip';
import { Alert, AlertDescription } from '@/components/ui/alert';

import { cn } from '@/lib/utils';
import { useSessionStore } from '@/lib/session-store';

interface ToolExecution {
  id: string;
  toolId: string;
  toolName: string;
  category: 'file' | 'system' | 'network' | 'development' | 'ai' | 'mcp';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timeout';
  priority: 'low' | 'medium' | 'high' | 'critical';
  sessionId: string;
  sessionName?: string;
  userId?: string;
  input: any;
  output?: any;
  error?: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  approvedBy?: string;
  approvedAt?: number;
  resources: {
    cpuUsage?: number;
    memoryUsage?: number;
    diskUsage?: number;
    networkUsage?: number;
  };
  security: {
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    requiresApproval: boolean;
    permissions: string[];
    sandboxed: boolean;
  };
  metadata: {
    provider?: string;
    version?: string;
    environment?: string;
    tags?: string[];
  };
}

interface ExecutionMetrics {
  totalExecutions: number;
  activeExecutions: number;
  completedExecutions: number;
  failedExecutions: number;
  avgExecutionTime: number;
  successRate: number;
  resourceUtilization: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
  securityEvents: number;
  approvalsPending: number;
}

const getStatusIcon = (status: ToolExecution['status']) => {
  switch (status) {
    case 'pending':
      return <Clock className="h-4 w-4 text-yellow-500" />;
    case 'running':
      return <Play className="h-4 w-4 text-blue-500 animate-pulse" />;
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case 'failed':
      return <XCircle className="h-4 w-4 text-red-500" />;
    case 'cancelled':
      return <Square className="h-4 w-4 text-gray-500" />;
    case 'timeout':
      return <AlertTriangle className="h-4 w-4 text-orange-500" />;
    default:
      return <AlertCircleIcon className="h-4 w-4 text-gray-500" />;
  }
};

const getStatusColor = (status: ToolExecution['status']) => {
  switch (status) {
    case 'pending':
      return 'border-yellow-500 text-yellow-700 bg-yellow-50';
    case 'running':
      return 'border-blue-500 text-blue-700 bg-blue-50';
    case 'completed':
      return 'border-green-500 text-green-700 bg-green-50';
    case 'failed':
      return 'border-red-500 text-red-700 bg-red-50';
    case 'cancelled':
      return 'border-gray-500 text-gray-700 bg-gray-50';
    case 'timeout':
      return 'border-orange-500 text-orange-700 bg-orange-50';
    default:
      return 'border-gray-500 text-gray-700 bg-gray-50';
  }
};

const getPriorityColor = (priority: ToolExecution['priority']) => {
  switch (priority) {
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
      return <Activity className="h-4 w-4" />;
  }
};

const formatDuration = (ms?: number) => {
  if (!ms) return 'N/A';
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
};

const formatTimestamp = (timestamp: number) => {
  return new Date(timestamp).toLocaleString();
};

export const ToolExecutionMonitor: React.FC = () => {
  const { actions } = useSessionStore();
  
  const [executions, setExecutions] = useState<ToolExecution[]>([]);
  const [metrics, setMetrics] = useState<ExecutionMetrics | null>(null);
  const [selectedExecution, setSelectedExecution] = useState<ToolExecution | null>(null);
  const [filters, setFilters] = useState({
    status: 'all',
    category: 'all',
    priority: 'all',
    session: 'all',
    timeRange: '24h'
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showDetails, setShowDetails] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [sortBy, setSortBy] = useState<'startTime' | 'duration' | 'priority' | 'status'>('startTime');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Mock data for demonstration
  const mockExecutions: ToolExecution[] = useMemo(() => [
    {
      id: 'exec-001',
      toolId: 'bash-exec',
      toolName: 'Command Execution',
      category: 'system',
      status: 'running',
      priority: 'high',
      sessionId: 'session-1',
      sessionName: 'Main Development Session',
      userId: 'user-1',
      input: { command: 'npm test', workingDir: '/project', timeout: 60000 },
      startTime: Date.now() - 45000,
      resources: {
        cpuUsage: 15.3,
        memoryUsage: 128.5,
        diskUsage: 2.1,
        networkUsage: 0.5
      },
      security: {
        riskLevel: 'medium',
        requiresApproval: true,
        permissions: ['execute:shell', 'read:filesystem'],
        sandboxed: true
      },
      metadata: {
        environment: 'development',
        tags: ['testing', 'npm']
      }
    },
    {
      id: 'exec-002',
      toolId: 'file-glob',
      toolName: 'File Pattern Search',
      category: 'file',
      status: 'completed',
      priority: 'low',
      sessionId: 'session-1',
      sessionName: 'Main Development Session',
      userId: 'user-1',
      input: { pattern: '**/*.{ts,tsx}', path: '/src' },
      output: { files: ['src/index.ts', 'src/app.tsx', 'src/utils.ts'], count: 3 },
      startTime: Date.now() - 300000,
      endTime: Date.now() - 299500,
      duration: 500,
      resources: {
        cpuUsage: 2.1,
        memoryUsage: 45.2,
        diskUsage: 0.8,
        networkUsage: 0.0
      },
      security: {
        riskLevel: 'low',
        requiresApproval: false,
        permissions: ['read:filesystem'],
        sandboxed: false
      },
      metadata: {
        environment: 'development',
        tags: ['search', 'files']
      }
    },
    {
      id: 'exec-003',
      toolId: 'mcp-puppeteer',
      toolName: 'Browser Automation',
      category: 'mcp',
      status: 'pending',
      priority: 'medium',
      sessionId: 'session-2',
      sessionName: 'Testing Session',
      userId: 'user-1',
      input: { url: 'https://example.com', action: 'screenshot', options: { fullPage: true } },
      startTime: Date.now() - 10000,
      resources: {
        cpuUsage: 0,
        memoryUsage: 0,
        diskUsage: 0,
        networkUsage: 0
      },
      security: {
        riskLevel: 'medium',
        requiresApproval: true,
        permissions: ['network:browser', 'execute:automation'],
        sandboxed: true
      },
      metadata: {
        provider: 'puppeteer',
        version: '3.0.1',
        environment: 'testing',
        tags: ['browser', 'screenshot']
      }
    },
    {
      id: 'exec-004',
      toolId: 'ai-agent',
      toolName: 'AI Sub-Agent',
      category: 'ai',
      status: 'failed',
      priority: 'high',
      sessionId: 'session-1',
      sessionName: 'Main Development Session',
      userId: 'user-1',
      input: { task: 'Generate comprehensive test suite', context: 'React component testing' },
      error: 'Rate limit exceeded. Please try again later.',
      startTime: Date.now() - 600000,
      endTime: Date.now() - 580000,
      duration: 20000,
      resources: {
        cpuUsage: 8.7,
        memoryUsage: 256.8,
        diskUsage: 1.2,
        networkUsage: 15.4
      },
      security: {
        riskLevel: 'low',
        requiresApproval: false,
        permissions: ['ai:subtask'],
        sandboxed: true
      },
      metadata: {
        environment: 'development',
        tags: ['ai', 'testing', 'generation']
      }
    }
  ], []);

  const mockMetrics: ExecutionMetrics = useMemo(() => ({
    totalExecutions: 1247,
    activeExecutions: 3,
    completedExecutions: 1156,
    failedExecutions: 88,
    avgExecutionTime: 2340,
    successRate: 92.9,
    resourceUtilization: {
      cpu: 23.5,
      memory: 67.2,
      disk: 12.8,
      network: 5.1
    },
    securityEvents: 12,
    approvalsPending: 2
  }), []);

  useEffect(() => {
    setExecutions(mockExecutions);
    setMetrics(mockMetrics);
  }, [mockExecutions, mockMetrics]);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      // In real implementation, this would fetch fresh data
      console.log('Refreshing execution data...');
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Filter and sort executions
  const filteredExecutions = useMemo(() => {
    let filtered = executions.filter(execution => {
      const matchesStatus = filters.status === 'all' || execution.status === filters.status;
      const matchesCategory = filters.category === 'all' || execution.category === filters.category;
      const matchesPriority = filters.priority === 'all' || execution.priority === filters.priority;
      const matchesSession = filters.session === 'all' || execution.sessionId === filters.session;
      const matchesSearch = searchQuery === '' || 
        execution.toolName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        execution.sessionName?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        JSON.stringify(execution.input).toLowerCase().includes(searchQuery.toLowerCase());

      // Time range filter
      const now = Date.now();
      const timeRanges = {
        '1h': 3600000,
        '24h': 86400000,
        '7d': 604800000,
        '30d': 2592000000,
        'all': Infinity
      };
      const timeLimit = timeRanges[filters.timeRange as keyof typeof timeRanges];
      const matchesTimeRange = (now - execution.startTime) <= timeLimit;

      return matchesStatus && matchesCategory && matchesPriority && matchesSession && matchesSearch && matchesTimeRange;
    });

    // Sort executions
    filtered.sort((a, b) => {
      let aValue: any, bValue: any;
      
      switch (sortBy) {
        case 'startTime':
          aValue = a.startTime;
          bValue = b.startTime;
          break;
        case 'duration':
          aValue = a.duration || 0;
          bValue = b.duration || 0;
          break;
        case 'priority':
          const priorityOrder = { low: 1, medium: 2, high: 3, critical: 4 };
          aValue = priorityOrder[a.priority];
          bValue = priorityOrder[b.priority];
          break;
        case 'status':
          aValue = a.status;
          bValue = b.status;
          break;
        default:
          aValue = a.startTime;
          bValue = b.startTime;
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [executions, filters, searchQuery, sortBy, sortOrder]);




  const handleExecutionAction = (action: string, executionId: string) => {
    switch (action) {
      case 'approve':
        console.log(`Approving execution ${executionId}`);
        setExecutions(prev => prev.map(exec => 
          exec.id === executionId ? { ...exec, status: 'running', approvedBy: 'user', approvedAt: Date.now() } : exec
        ));
        break;
      case 'deny':
        console.log(`Denying execution ${executionId}`);
        setExecutions(prev => prev.map(exec => 
          exec.id === executionId ? { ...exec, status: 'cancelled', error: 'Denied by user' } : exec
        ));
        break;
      case 'cancel':
        console.log(`Cancelling execution ${executionId}`);
        setExecutions(prev => prev.map(exec => 
          exec.id === executionId ? { ...exec, status: 'cancelled' } : exec
        ));
        break;
      case 'retry':
        console.log(`Retrying execution ${executionId}`);
        setExecutions(prev => prev.map(exec => 
          exec.id === executionId ? { 
            ...exec, 
            status: 'pending', 
            error: undefined,
            startTime: Date.now()
          } : exec
        ));
        break;
    }
  };

  return (
    <TooltipProvider>
      <div className={cn(
        "flex flex-col transition-all duration-300",
        isExpanded ? "h-screen fixed inset-0 bg-background z-50" : "h-full"
      )}>
        {/* Header */}
        <div className="p-6 border-b border-border bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                <Activity className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Tool Execution Monitor</h1>
                <p className="text-muted-foreground">
                  Real-time monitoring and control of tool executions
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
                <RefreshCw className={cn("h-4 w-4 mr-2", autoRefresh && "animate-spin")} />
                Auto-refresh {autoRefresh ? "ON" : "OFF"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsExpanded(!isExpanded)}
              >
                {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        </div>

        {/* Metrics Overview */}
        {metrics && (
          <div className="px-6 py-4 bg-muted/30">
            <div className="grid grid-cols-6 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Active</p>
                      <p className="text-2xl font-bold text-blue-600">{metrics.activeExecutions}</p>
                    </div>
                    <Activity className="h-8 w-8 text-blue-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Success Rate</p>
                      <p className="text-2xl font-bold text-green-600">{metrics.successRate}%</p>
                    </div>
                    <TrendingUp className="h-8 w-8 text-green-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Avg Duration</p>
                      <p className="text-2xl font-bold">{formatDuration(metrics.avgExecutionTime)}</p>
                    </div>
                    <Timer className="h-8 w-8 text-orange-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">CPU Usage</p>
                      <p className="text-2xl font-bold">{metrics.resourceUtilization.cpu}%</p>
                    </div>
                    <Cpu className="h-8 w-8 text-purple-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Memory</p>
                      <p className="text-2xl font-bold">{metrics.resourceUtilization.memory}%</p>
                    </div>
                    <MemoryStick className="h-8 w-8 text-indigo-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Pending Approvals</p>
                      <p className="text-2xl font-bold text-yellow-600">{metrics.approvalsPending}</p>
                    </div>
                    <Shield className="h-8 w-8 text-yellow-500" />
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* Filters and Controls */}
        <div className="px-6 py-4 border-b border-border">
          <div className="flex items-center justify-between space-x-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Search className="h-4 w-4" />
                <Input
                  placeholder="Search executions..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-64"
                />
              </div>
              
              <Select value={filters.status} onValueChange={(value) => setFilters(prev => ({ ...prev, status: value }))}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="running">Running</SelectItem>
                  <SelectItem value="completed">Completed</SelectItem>
                  <SelectItem value="failed">Failed</SelectItem>
                  <SelectItem value="cancelled">Cancelled</SelectItem>
                </SelectContent>
              </Select>
              
              <Select value={filters.category} onValueChange={(value) => setFilters(prev => ({ ...prev, category: value }))}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Categories</SelectItem>
                  <SelectItem value="file">File</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                  <SelectItem value="network">Network</SelectItem>
                  <SelectItem value="development">Development</SelectItem>
                  <SelectItem value="ai">AI</SelectItem>
                  <SelectItem value="mcp">MCP</SelectItem>
                </SelectContent>
              </Select>
              
              <Select value={filters.timeRange} onValueChange={(value) => setFilters(prev => ({ ...prev, timeRange: value }))}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1h">Last Hour</SelectItem>
                  <SelectItem value="24h">Last 24h</SelectItem>
                  <SelectItem value="7d">Last 7 days</SelectItem>
                  <SelectItem value="30d">Last 30 days</SelectItem>
                  <SelectItem value="all">All Time</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
              <Button variant="outline" size="sm">
                <Filter className="h-4 w-4 mr-2" />
                Advanced Filters
              </Button>
            </div>
          </div>
        </div>

        {/* Execution List */}
        <div className="flex-1 overflow-auto p-6">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">
                Executions ({filteredExecutions.length})
              </h3>
              <div className="flex items-center space-x-2">
                <Select value={sortBy} onValueChange={(value: any) => setSortBy(value)}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="startTime">Start Time</SelectItem>
                    <SelectItem value="duration">Duration</SelectItem>
                    <SelectItem value="priority">Priority</SelectItem>
                    <SelectItem value="status">Status</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                >
                  {sortOrder === 'asc' ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                </Button>
              </div>
            </div>

            {filteredExecutions.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center">
                  <Activity className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">No executions found</h3>
                  <p className="text-muted-foreground">
                    Try adjusting your filters or search query
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-2">
                {filteredExecutions.map((execution) => (
                  <motion.div
                    key={execution.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="group"
                  >
                    <Card className="hover:shadow-md transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            {/* Status and Category */}
                            <div className="flex items-center space-x-2">
                              {getStatusIcon(execution.status)}
                              {getCategoryIcon(execution.category)}
                            </div>
                            
                            {/* Tool Information */}
                            <div>
                              <div className="font-medium">{execution.toolName}</div>
                              <div className="text-sm text-muted-foreground">
                                {execution.sessionName || execution.sessionId} • {formatTimestamp(execution.startTime)}
                              </div>
                            </div>
                            
                            {/* Priority and Risk */}
                            <div className="flex items-center space-x-2">
                              <Badge variant="outline" className={getPriorityColor(execution.priority)}>
                                {execution.priority.toUpperCase()}
                              </Badge>
                              <Badge variant="outline" className={getRiskColor(execution.security.riskLevel)}>
                                {execution.security.riskLevel} risk
                              </Badge>
                            </div>
                            
                            {/* Duration */}
                            <div className="text-sm">
                              <div className="flex items-center space-x-1">
                                <Timer className="h-3 w-3" />
                                <span>{formatDuration(execution.duration)}</span>
                              </div>
                            </div>
                            
                            {/* Resource Usage */}
                            {execution.status === 'running' && (
                              <div className="flex items-center space-x-2 text-xs">
                                <Tooltip>
                                  <TooltipTrigger>
                                    <div className="flex items-center space-x-1">
                                      <Cpu className="h-3 w-3" />
                                      <span>{execution.resources.cpuUsage?.toFixed(1)}%</span>
                                    </div>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p>CPU Usage</p>
                                  </TooltipContent>
                                </Tooltip>
                                
                                <Tooltip>
                                  <TooltipTrigger>
                                    <div className="flex items-center space-x-1">
                                      <MemoryStick className="h-3 w-3" />
                                      <span>{execution.resources.memoryUsage?.toFixed(1)}MB</span>
                                    </div>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p>Memory Usage</p>
                                  </TooltipContent>
                                </Tooltip>
                              </div>
                            )}
                          </div>
                          
                          {/* Actions */}
                          <div className="flex items-center space-x-2">
                            <Badge variant="outline" className={getStatusColor(execution.status)}>
                              {execution.status}
                            </Badge>
                            
                            {execution.status === 'pending' && execution.security.requiresApproval && (
                              <div className="flex space-x-1">
                                <Button
                                  size="sm"
                                  onClick={() => handleExecutionAction('approve', execution.id)}
                                  className="bg-green-600 hover:bg-green-700"
                                >
                                  <CheckCircle className="h-3 w-3 mr-1" />
                                  Approve
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => handleExecutionAction('deny', execution.id)}
                                >
                                  <XCircle className="h-3 w-3 mr-1" />
                                  Deny
                                </Button>
                              </div>
                            )}
                            
                            {execution.status === 'running' && (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleExecutionAction('cancel', execution.id)}
                              >
                                <Square className="h-3 w-3 mr-1" />
                                Cancel
                              </Button>
                            )}
                            
                            {execution.status === 'failed' && (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleExecutionAction('retry', execution.id)}
                              >
                                <RefreshCw className="h-3 w-3 mr-1" />
                                Retry
                              </Button>
                            )}
                            
                            <Dialog>
                              <DialogTrigger asChild>
                                <Button variant="ghost" size="sm">
                                  <Eye className="h-4 w-4" />
                                </Button>
                              </DialogTrigger>
                              <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
                                <DialogHeader>
                                  <DialogTitle>Execution Details - {execution.toolName}</DialogTitle>
                                </DialogHeader>
                                <ExecutionDetailsDialog execution={execution} />
                              </DialogContent>
                            </Dialog>
                          </div>
                        </div>
                        
                        {/* Error Display */}
                        {execution.error && (
                          <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                            <div className="flex items-center space-x-2">
                              <XCircle className="h-4 w-4" />
                              <span className="font-medium">Error:</span>
                              <span>{execution.error}</span>
                            </div>
                          </div>
                        )}
                        
                        {/* Security Warnings */}
                        {execution.security.riskLevel === 'high' && execution.status === 'pending' && (
                          <Alert className="mt-3 border-orange-200 bg-orange-50">
                            <AlertTriangle className="h-4 w-4 text-orange-600" />
                            <AlertDescription className="text-orange-800">
                              <span className="font-medium">High Risk Execution:</span> This tool requires elevated permissions and should be carefully reviewed before approval.
                            </AlertDescription>
                          </Alert>
                        )}
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
};

// Execution Details Dialog Component
function ExecutionDetailsDialog({ execution }: { execution: ToolExecution }) {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="input">Input</TabsTrigger>
          <TabsTrigger value="output">Output</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Execution Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Tool:</span>
                  <span className="text-sm font-medium">{execution.toolName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Category:</span>
                  <div className="flex items-center space-x-1">
                    {getCategoryIcon(execution.category)}
                    <span className="text-sm capitalize">{execution.category}</span>
                  </div>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Status:</span>
                  <Badge variant="outline" className={getStatusColor(execution.status)}>
                    {execution.status}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Priority:</span>
                  <Badge variant="outline" className={getPriorityColor(execution.priority)}>
                    {execution.priority}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Duration:</span>
                  <span className="text-sm">{formatDuration(execution.duration)}</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Resource Usage</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>CPU:</span>
                    <span>{execution.resources.cpuUsage?.toFixed(1)}%</span>
                  </div>
                  <Progress value={execution.resources.cpuUsage || 0} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Memory:</span>
                    <span>{execution.resources.memoryUsage?.toFixed(1)}MB</span>
                  </div>
                  <Progress value={(execution.resources.memoryUsage || 0) / 10} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Network:</span>
                    <span>{execution.resources.networkUsage?.toFixed(1)}KB/s</span>
                  </div>
                  <Progress value={execution.resources.networkUsage || 0} className="h-2" />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="input" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Input Parameters</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="text-xs bg-muted p-3 rounded overflow-x-auto max-h-96">
                {JSON.stringify(execution.input, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="output" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Output Results</CardTitle>
            </CardHeader>
            <CardContent>
              {execution.output ? (
                <pre className="text-xs bg-muted p-3 rounded overflow-x-auto max-h-96">
                  {JSON.stringify(execution.output, null, 2)}
                </pre>
              ) : execution.error ? (
                <div className="text-sm text-red-600 bg-red-50 p-3 rounded">
                  {execution.error}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground italic">
                  No output available yet
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Security Assessment</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Risk Level:</span>
                  <Badge variant="outline" className={getRiskColor(execution.security.riskLevel)}>
                    {execution.security.riskLevel}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Requires Approval:</span>
                  <Badge variant={execution.security.requiresApproval ? "destructive" : "secondary"}>
                    {execution.security.requiresApproval ? "Yes" : "No"}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Sandboxed:</span>
                  <Badge variant={execution.security.sandboxed ? "default" : "destructive"}>
                    {execution.security.sandboxed ? "Yes" : "No"}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Permissions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-1">
                  {execution.security.permissions.map((permission, idx) => (
                    <Badge key={idx} variant="outline" className="text-xs">
                      {permission}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}