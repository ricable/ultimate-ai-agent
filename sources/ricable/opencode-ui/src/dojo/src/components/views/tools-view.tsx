"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { 
  Wrench, 
  File, 
  Terminal, 
  Plug, 
  Settings, 
  Play, 
  Pause, 
  CheckCircle, 
  XCircle, 
  Clock, 
  AlertTriangle,
  Eye,
  Filter,
  RefreshCw,
  ChevronRight,
  Shield,
  Zap,
  FileText,
  Search,
  Globe,
  Code,
  GitBranch,
  Database,
  Bug
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useSessionStore } from "@/lib/session-store";
import { cn } from "@/lib/utils";
import { OpenCodeView } from "@/types/opencode";

interface ToolsViewProps {
  onViewChange: (view: OpenCodeView) => void;
}

interface BuiltInTool {
  id: string;
  name: string;
  description: string;
  category: "file" | "system" | "development" | "network";
  icon: React.ReactNode;
  enabled: boolean;
  requiresApproval: boolean;
  riskLevel: "low" | "medium" | "high";
  usage: {
    today: number;
    total: number;
    avgResponseTime: number;
  };
  permissions: string[];
  examples: string[];
}

interface ToolExecution {
  id: string;
  toolId: string;
  toolName: string;
  sessionId: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  input: any;
  output?: any;
  error?: string;
  startTime: number;
  endTime?: number;
  approvedBy?: string;
  riskAssessment?: {
    level: "low" | "medium" | "high";
    concerns: string[];
    recommendations: string[];
  };
}

// Built-in OpenCode tools
const builtInTools: BuiltInTool[] = [
  {
    id: "glob",
    name: "File Pattern Search",
    description: "Find files matching glob patterns",
    category: "file",
    icon: <Search className="h-4 w-4" />,
    enabled: true,
    requiresApproval: false,
    riskLevel: "low",
    usage: { today: 12, total: 145, avgResponseTime: 120 },
    permissions: ["read:filesystem"],
    examples: ["**/*.ts", "src/**/*.{js,jsx}", "*.md"]
  },
  {
    id: "grep",
    name: "Content Search",
    description: "Search file contents using patterns",
    category: "file",
    icon: <FileText className="h-4 w-4" />,
    enabled: true,
    requiresApproval: false,
    riskLevel: "low",
    usage: { today: 8, total: 89, avgResponseTime: 250 },
    permissions: ["read:filesystem"],
    examples: ["function.*export", "TODO|FIXME", "import.*react"]
  },
  {
    id: "ls",
    name: "List Directory",
    description: "List files and directories",
    category: "file",
    icon: <File className="h-4 w-4" />,
    enabled: true,
    requiresApproval: false,
    riskLevel: "low",
    usage: { today: 25, total: 312, avgResponseTime: 80 },
    permissions: ["read:filesystem"],
    examples: ["/src", "./components", "/etc"]
  },
  {
    id: "view",
    name: "View File",
    description: "Read file contents with optional range",
    category: "file",
    icon: <Eye className="h-4 w-4" />,
    enabled: true,
    requiresApproval: false,
    riskLevel: "low",
    usage: { today: 18, total: 201, avgResponseTime: 150 },
    permissions: ["read:filesystem"],
    examples: ["src/index.ts", "package.json", "README.md"]
  },
  {
    id: "write",
    name: "Write File",
    description: "Create or overwrite file contents",
    category: "file",
    icon: <FileText className="h-4 w-4" />,
    enabled: true,
    requiresApproval: true,
    riskLevel: "medium",
    usage: { today: 6, total: 67, avgResponseTime: 200 },
    permissions: ["write:filesystem"],
    examples: ["src/component.tsx", "config.json", "README.md"]
  },
  {
    id: "edit",
    name: "Edit File",
    description: "Apply targeted edits to files",
    category: "file",
    icon: <Code className="h-4 w-4" />,
    enabled: true,
    requiresApproval: true,
    riskLevel: "medium",
    usage: { today: 15, total: 178, avgResponseTime: 300 },
    permissions: ["write:filesystem"],
    examples: ["Replace function", "Add import", "Fix typo"]
  },
  {
    id: "patch",
    name: "Apply Patch",
    description: "Apply diff patches to files",
    category: "file",
    icon: <GitBranch className="h-4 w-4" />,
    enabled: true,
    requiresApproval: true,
    riskLevel: "high",
    usage: { today: 2, total: 23, avgResponseTime: 450 },
    permissions: ["write:filesystem"],
    examples: ["git diff", "unified diff", "context diff"]
  },
  {
    id: "bash",
    name: "Execute Command",
    description: "Run shell commands with timeout",
    category: "system",
    icon: <Terminal className="h-4 w-4" />,
    enabled: true,
    requiresApproval: true,
    riskLevel: "high",
    usage: { today: 4, total: 45, avgResponseTime: 1200 },
    permissions: ["execute:shell"],
    examples: ["npm test", "git status", "ls -la"]
  },
  {
    id: "fetch",
    name: "Fetch URL",
    description: "Fetch data from URLs with format conversion",
    category: "network",
    icon: <Globe className="h-4 w-4" />,
    enabled: true,
    requiresApproval: false,
    riskLevel: "medium",
    usage: { today: 3, total: 28, avgResponseTime: 800 },
    permissions: ["network:http"],
    examples: ["https://api.github.com", "https://docs.example.com", "local file"]
  },
  {
    id: "agent",
    name: "Sub-Agent",
    description: "Run sub-tasks with specialized AI agent",
    category: "system",
    icon: <Zap className="h-4 w-4" />,
    enabled: true,
    requiresApproval: false,
    riskLevel: "low",
    usage: { today: 1, total: 12, avgResponseTime: 2500 },
    permissions: ["ai:subtask"],
    examples: ["Analyze code", "Generate tests", "Write documentation"]
  },
  {
    id: "diagnostics",
    name: "LSP Diagnostics",
    description: "Get language server diagnostics",
    category: "development",
    icon: <Bug className="h-4 w-4" />,
    enabled: true,
    requiresApproval: false,
    riskLevel: "low",
    usage: { today: 5, total: 34, avgResponseTime: 400 },
    permissions: ["lsp:diagnostics"],
    examples: ["src/index.ts", "Current file", "All files"]
  }
];

// Mock tool executions
const mockExecutions: ToolExecution[] = [
  {
    id: "exec-1",
    toolId: "bash",
    toolName: "Execute Command",
    sessionId: "session-1",
    status: "pending",
    input: { command: "npm test", timeout: 30000 },
    startTime: Date.now() - 30000,
    riskAssessment: {
      level: "medium",
      concerns: ["Command execution", "File system access"],
      recommendations: ["Review command before approval", "Check for destructive operations"]
    }
  },
  {
    id: "exec-2",
    toolId: "write",
    toolName: "Write File",
    sessionId: "session-1",
    status: "completed",
    input: { filePath: "src/component.tsx", content: "// Component code..." },
    output: { success: true, bytesWritten: 1024 },
    startTime: Date.now() - 300000,
    endTime: Date.now() - 295000,
    approvedBy: "user"
  },
  {
    id: "exec-3",
    toolId: "glob",
    toolName: "File Pattern Search",
    sessionId: "session-1",
    status: "completed",
    input: { pattern: "**/*.ts" },
    output: { files: ["src/index.ts", "src/utils.ts"], count: 2 },
    startTime: Date.now() - 600000,
    endTime: Date.now() - 599000
  }
];

export function ToolsView({ onViewChange }: ToolsViewProps) {
  const [tools, setTools] = useState<BuiltInTool[]>(builtInTools);
  const [executions, setExecutions] = useState<ToolExecution[]>(mockExecutions);
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [showOnlyEnabled, setShowOnlyEnabled] = useState(false);
  const [selectedExecution, setSelectedExecution] = useState<ToolExecution | null>(null);
  const [showSettingsDialog, setShowSettingsDialog] = useState(false);
  
  const { actions } = useSessionStore();

  const categories = [
    { id: "all", name: "All Tools", icon: <Wrench className="h-4 w-4" /> },
    { id: "file", name: "File Operations", icon: <File className="h-4 w-4" /> },
    { id: "system", name: "System Tools", icon: <Terminal className="h-4 w-4" /> },
    { id: "development", name: "Development", icon: <Code className="h-4 w-4" /> },
    { id: "network", name: "Network", icon: <Globe className="h-4 w-4" /> }
  ];

  const filteredTools = tools.filter(tool => {
    const matchesCategory = selectedCategory === "all" || tool.category === selectedCategory;
    const matchesSearch = tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         tool.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesEnabled = !showOnlyEnabled || tool.enabled;
    return matchesCategory && matchesSearch && matchesEnabled;
  });

  const pendingExecutions = executions.filter(e => e.status === "pending");
  const recentExecutions = executions.slice(0, 10);

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
        return <XCircle className="h-4 w-4 text-gray-500" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low':
        return 'text-green-600 bg-green-50';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50';
      case 'high':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const handleToolToggle = (toolId: string) => {
    setTools(prev => prev.map(tool => 
      tool.id === toolId ? { ...tool, enabled: !tool.enabled } : tool
    ));
  };

  const handleApproveExecution = async (executionId: string) => {
    setExecutions(prev => prev.map(exec => 
      exec.id === executionId 
        ? { ...exec, status: 'running', approvedBy: 'user' }
        : exec
    ));
    
    // Simulate execution completion
    setTimeout(() => {
      setExecutions(prev => prev.map(exec => 
        exec.id === executionId 
          ? { ...exec, status: 'completed', endTime: Date.now() }
          : exec
      ));
    }, 2000);
  };

  const handleDenyExecution = async (executionId: string) => {
    setExecutions(prev => prev.map(exec => 
      exec.id === executionId 
        ? { ...exec, status: 'cancelled' }
        : exec
    ));
  };

  const totalUsageToday = tools.reduce((sum, tool) => sum + tool.usage.today, 0);
  const enabledTools = tools.filter(t => t.enabled).length;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onViewChange('welcome')}
            >
              <ChevronRight className="h-4 w-4 rotate-180" />
            </Button>
            <div className="flex items-center space-x-2">
              <Wrench className="h-6 w-6 text-blue-500" />
              <h1 className="text-2xl font-bold">Tools Dashboard</h1>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              onClick={() => onViewChange('mcp')}
            >
              <Plug className="h-4 w-4 mr-2" />
              MCP Servers
            </Button>
            <Button 
              variant="outline"
              onClick={() => setShowSettingsDialog(true)}
            >
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
          </div>
        </div>
        <p className="text-muted-foreground mt-2">
          Monitor and manage OpenCode&apos;s built-in tools and MCP integrations
        </p>
      </div>

      {/* Stats */}
      <div className="px-6 py-4 bg-muted/50">
        <div className="grid grid-cols-4 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold">{enabledTools}/{tools.length}</div>
            <div className="text-sm text-muted-foreground">Tools Enabled</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{totalUsageToday}</div>
            <div className="text-sm text-muted-foreground">Executions Today</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{pendingExecutions.length}</div>
            <div className="text-sm text-muted-foreground">Pending Approvals</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{tools.filter(t => t.requiresApproval).length}</div>
            <div className="text-sm text-muted-foreground">Require Approval</div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        <Tabs defaultValue="tools" className="h-full">
          <div className="px-6 py-4 border-b">
            <TabsList>
              <TabsTrigger value="tools">Available Tools</TabsTrigger>
              <TabsTrigger value="executions">
                Executions
                {pendingExecutions.length > 0 && (
                  <Badge className="ml-2" variant="destructive">
                    {pendingExecutions.length}
                  </Badge>
                )}
              </TabsTrigger>
              <TabsTrigger value="permissions">Permissions</TabsTrigger>
              <TabsTrigger value="monitor">Monitor</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="tools" className="p-6 space-y-6">
            {/* Filters */}
            <div className="flex items-center justify-between">
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
              <div className="flex items-center space-x-2">
                <label className="text-sm">Show enabled only</label>
                <Switch
                  checked={showOnlyEnabled}
                  onCheckedChange={setShowOnlyEnabled}
                />
              </div>
            </div>

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
                    "transition-all duration-200",
                    tool.enabled ? "border-green-200 bg-green-50/50" : "border-gray-200",
                    "hover:shadow-md"
                  )}>
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {tool.icon}
                          <CardTitle className="text-base">{tool.name}</CardTitle>
                        </div>
                        <Switch
                          checked={tool.enabled}
                          onCheckedChange={() => handleToolToggle(tool.id)}
                        />
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

                      <div className="text-xs text-muted-foreground">
                        <div>Today: {tool.usage.today} executions</div>
                        <div>Total: {tool.usage.total} executions</div>
                        <div>Avg: {tool.usage.avgResponseTime}ms</div>
                      </div>

                      <div className="space-y-1">
                        <div className="text-xs font-medium">Examples:</div>
                        <div className="flex flex-wrap gap-1">
                          {tool.examples.slice(0, 2).map((example, idx) => (
                            <Badge key={idx} variant="secondary" className="text-xs">
                              {example}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="executions" className="p-6 space-y-6">
            {/* Pending Approvals */}
            {pendingExecutions.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                  <AlertTriangle className="h-5 w-5 text-yellow-500" />
                  <span>Pending Approvals ({pendingExecutions.length})</span>
                </h3>
                <div className="space-y-3">
                  {pendingExecutions.map((execution) => (
                    <Card key={execution.id} className="border-yellow-200 bg-yellow-50">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            {getStatusIcon(execution.status)}
                            <div>
                              <div className="font-medium">{execution.toolName}</div>
                              <div className="text-sm text-muted-foreground">
                                Requested {new Date(execution.startTime).toLocaleString()}
                              </div>
                            </div>
                          </div>
                          <div className="flex space-x-2">
                            <Button
                              size="sm"
                              onClick={() => handleApproveExecution(execution.id)}
                            >
                              <CheckCircle className="h-4 w-4 mr-1" />
                              Approve
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleDenyExecution(execution.id)}
                            >
                              <XCircle className="h-4 w-4 mr-1" />
                              Deny
                            </Button>
                          </div>
                        </div>
                        
                        {execution.riskAssessment && (
                          <div className="bg-white p-3 rounded border">
                            <div className="text-sm font-medium mb-2">Risk Assessment</div>
                            <div className="space-y-2">
                              <div className="flex items-center space-x-2">
                                <Badge 
                                  variant="outline"
                                  className={getRiskColor(execution.riskAssessment.level)}
                                >
                                  {execution.riskAssessment.level} risk
                                </Badge>
                              </div>
                              <div>
                                <div className="text-xs font-medium mb-1">Concerns:</div>
                                <ul className="text-xs text-muted-foreground space-y-1">
                                  {execution.riskAssessment.concerns.map((concern, idx) => (
                                    <li key={idx}>• {concern}</li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          </div>
                        )}
                        
                        <div className="mt-3">
                          <div className="text-sm font-medium mb-2">Input:</div>
                          <pre className="text-xs bg-white p-2 rounded border overflow-x-auto">
                            {JSON.stringify(execution.input, null, 2)}
                          </pre>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}

            {/* Recent Executions */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Recent Executions</h3>
              <div className="space-y-2">
                {recentExecutions.map((execution) => (
                  <Card key={execution.id}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          {getStatusIcon(execution.status)}
                          <div>
                            <div className="font-medium">{execution.toolName}</div>
                            <div className="text-sm text-muted-foreground">
                              {execution.endTime 
                                ? `Completed in ${execution.endTime - execution.startTime}ms`
                                : `Started ${new Date(execution.startTime).toLocaleString()}`
                              }
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline">{execution.status}</Badge>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setSelectedExecution(execution)}
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="permissions" className="p-6">
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-4">Tool Permissions Matrix</h3>
                <div className="bg-card rounded-lg border p-4">
                  <div className="text-center py-8 text-muted-foreground">
                    <Shield className="h-12 w-12 mx-auto mb-4" />
                    <h4 className="font-medium mb-2">Permission Management</h4>
                    <p>Fine-grained permission control for tools and resources</p>
                    <p className="text-sm mt-2">Coming in next update</p>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="monitor" className="p-6">
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-4">Real-time Monitoring</h3>
                <div className="bg-card rounded-lg border p-4">
                  <div className="text-center py-8 text-muted-foreground">
                    <Database className="h-12 w-12 mx-auto mb-4" />
                    <h4 className="font-medium mb-2">Live Tool Monitoring</h4>
                    <p>Real-time tool execution monitoring and analytics</p>
                    <p className="text-sm mt-2">Feature in development</p>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* Tool Settings Dialog */}
      <Dialog open={showSettingsDialog} onOpenChange={setShowSettingsDialog}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <Settings className="h-5 w-5" />
              <span>Tool Configuration Settings</span>
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-6 py-4">
            <Tabs defaultValue="general" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="general">General</TabsTrigger>
                <TabsTrigger value="permissions">Permissions</TabsTrigger>
                <TabsTrigger value="security">Security</TabsTrigger>
                <TabsTrigger value="performance">Performance</TabsTrigger>
              </TabsList>
              
              <TabsContent value="general" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Default Tool Behavior</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto-enable new tools</Label>
                        <p className="text-sm text-muted-foreground">Automatically enable newly discovered tools</p>
                      </div>
                      <Switch defaultChecked />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Show usage statistics</Label>
                        <p className="text-sm text-muted-foreground">Display tool usage metrics in the interface</p>
                      </div>
                      <Switch defaultChecked />
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Default timeout (seconds)</Label>
                      <Input type="number" defaultValue="30" className="max-w-24" />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="permissions" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Tool Permissions</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>File system access</Label>
                          <p className="text-sm text-muted-foreground">Allow tools to read and write files</p>
                        </div>
                        <Select defaultValue="restricted">
                          <SelectTrigger className="w-32">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="disabled">Disabled</SelectItem>
                            <SelectItem value="restricted">Restricted</SelectItem>
                            <SelectItem value="full">Full Access</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Network access</Label>
                          <p className="text-sm text-muted-foreground">Allow tools to make network requests</p>
                        </div>
                        <Select defaultValue="restricted">
                          <SelectTrigger className="w-32">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="disabled">Disabled</SelectItem>
                            <SelectItem value="restricted">Restricted</SelectItem>
                            <SelectItem value="full">Full Access</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>System commands</Label>
                          <p className="text-sm text-muted-foreground">Allow tools to execute system commands</p>
                        </div>
                        <Select defaultValue="approval">
                          <SelectTrigger className="w-32">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="disabled">Disabled</SelectItem>
                            <SelectItem value="approval">Needs Approval</SelectItem>
                            <SelectItem value="full">Full Access</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="security" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Security Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Require approval for high-risk tools</Label>
                        <p className="text-sm text-muted-foreground">Always prompt before executing potentially dangerous tools</p>
                      </div>
                      <Switch defaultChecked />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Sandbox tool execution</Label>
                        <p className="text-sm text-muted-foreground">Run tools in isolated environment when possible</p>
                      </div>
                      <Switch defaultChecked />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Log all tool executions</Label>
                        <p className="text-sm text-muted-foreground">Keep detailed logs of tool usage for security auditing</p>
                      </div>
                      <Switch defaultChecked />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="performance" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Performance Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Maximum concurrent tool executions</Label>
                      <Input type="number" defaultValue="5" className="max-w-24" />
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Tool execution timeout (seconds)</Label>
                      <Input type="number" defaultValue="120" className="max-w-24" />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Cache tool results</Label>
                        <p className="text-sm text-muted-foreground">Cache results to improve performance for repeated operations</p>
                      </div>
                      <Switch defaultChecked />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>

            <div className="flex justify-end space-x-3 pt-4 border-t">
              <Button variant="outline" onClick={() => setShowSettingsDialog(false)}>
                Cancel
              </Button>
              <Button onClick={() => {
                // Save settings (placeholder)
                console.log('Saving tool settings');
                setShowSettingsDialog(false);
              }}>
                Save Settings
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Execution Details Modal */}
      {selectedExecution && (
        <Dialog open={!!selectedExecution} onOpenChange={() => setSelectedExecution(null)}>
          <DialogContent className="max-w-3xl">
            <DialogHeader>
              <DialogTitle>Execution Details - {selectedExecution.toolName}</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm font-medium mb-1">Status</div>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(selectedExecution.status)}
                    <Badge variant="outline">{selectedExecution.status}</Badge>
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium mb-1">Duration</div>
                  <div className="text-sm">
                    {selectedExecution.endTime 
                      ? `${selectedExecution.endTime - selectedExecution.startTime}ms`
                      : 'Running...'
                    }
                  </div>
                </div>
              </div>
              
              <div>
                <div className="text-sm font-medium mb-2">Input</div>
                <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
                  {JSON.stringify(selectedExecution.input, null, 2)}
                </pre>
              </div>
              
              {selectedExecution.output && (
                <div>
                  <div className="text-sm font-medium mb-2">Output</div>
                  <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
                    {JSON.stringify(selectedExecution.output, null, 2)}
                  </pre>
                </div>
              )}
              
              {selectedExecution.error && (
                <div>
                  <div className="text-sm font-medium mb-2">Error</div>
                  <div className="text-sm text-red-600 bg-red-50 p-3 rounded">
                    {selectedExecution.error}
                  </div>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}