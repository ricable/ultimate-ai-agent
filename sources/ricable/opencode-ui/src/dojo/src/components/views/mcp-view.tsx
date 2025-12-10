"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { 
  Plug, 
  Plus, 
  RefreshCw, 
  Download, 
  Upload,
  ChevronRight,
  Settings,
  Trash2,
  PlayCircle,
  StopCircle,
  CheckCircle,
  XCircle,
  AlertCircle,
  ExternalLink,
  Copy,
  Eye,
  EyeOff,
  Package,
  Activity,
  TestTube,
  FileDown,
  FileUp,
  Monitor
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { useSessionStore } from "@/lib/session-store";
import { OpenCodeView } from "@/types/opencode";
import { cn } from "@/lib/utils";

// Enhanced MCP Components
import { ServerConfigForm } from "@/components/mcp/server-config-form";
import { ServerDashboard } from "@/components/mcp/server-dashboard";
import { ServerHealthMonitor } from "@/components/mcp/server-health-monitor";
import { ServerTemplates } from "@/components/mcp/server-templates";
import { 
  MCPServerConfig, 
  MCPServerWithStatus, 
  MCPServerTemplate,
  MCPServerTestResult 
} from "@/lib/types/mcp";

interface MCPViewProps {
  onViewChange: (view: OpenCodeView) => void;
}

// Legacy interface for backward compatibility
interface MCPServer {
  id: string;
  name: string;
  type: "stdio" | "sse";
  status: "connected" | "disconnected" | "error" | "connecting";
  url?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  tools: string[];
  lastConnected?: number;
  errorMessage?: string;
}

// Mock MCP servers based on the screenshot
const mockMCPServers: MCPServer[] = [
  {
    id: "puppeteer",
    name: "puppeteer",
    type: "stdio",
    status: "connected",
    command: "npx puppeteer-mcp-server",
    args: [],
    env: {},
    tools: ["puppeteer_navigate", "puppeteer_screenshot", "puppeteer_click", "puppeteer_fill"],
    lastConnected: Date.now() - 3600000
  },
  {
    id: "consult7",
    name: "consult7",
    type: "stdio",
    status: "connected",
    command: "uvx consult7-google AizaSyDCpyDc95PTImwGg-Myr58Uz3GN7f-o8",
    args: [],
    env: {},
    tools: ["consultation", "code_analysis", "documentation_search"],
    lastConnected: Date.now() - 1800000
  }
];

export function MCPView({ onViewChange }: MCPViewProps) {
  const [servers, setServers] = useState<MCPServerWithStatus[]>([]);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showTemplatesDialog, setShowTemplatesDialog] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showImportExportDialog, setShowImportExportDialog] = useState(false);
  const [selectedServer, setSelectedServer] = useState<MCPServerConfig | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<"dashboard" | "health" | "templates" | "logs">("dashboard");
  const [loading, setLoading] = useState(false);

  // Mock enhanced server data
  const mockEnhancedServers: MCPServerWithStatus[] = mockMCPServers.map(server => ({
    id: server.id,
    name: server.name,
    description: `Enhanced ${server.name} server with advanced features`,
    type: server.type === "stdio" ? "local" : "remote",
    enabled: true,
    command: server.command ? [server.command, ...(server.args || [])] : undefined,
    url: server.url,
    environment: server.env,
    timeout: 30000,
    retryAttempts: 3,
    retryDelay: 1000,
    autoRestart: true,
    category: "automation",
    tags: ["automation", "tools"],
    created_at: Date.now() - Math.random() * 86400000,
    updated_at: Date.now(),
    status: {
      id: server.id,
      status: server.status,
      connected_at: server.lastConnected,
      last_ping: Date.now() - Math.random() * 60000,
      uptime: Math.random() * 86400000,
      restart_count: Math.floor(Math.random() * 5),
      process_id: Math.floor(Math.random() * 10000),
      memory_usage: Math.random() * 100000000,
      cpu_usage: Math.random() * 10,
      tools_count: server.tools.length,
      resources_count: Math.floor(Math.random() * 5)
    },
    health: {
      id: server.id,
      healthy: server.status === "connected",
      response_time: Math.random() * 2000,
      last_health_check: Date.now() - Math.random() * 300000,
      error_count: Math.floor(Math.random() * 10),
      warning_count: Math.floor(Math.random() * 5),
      performance_score: 70 + Math.random() * 30,
      availability_percentage: 90 + Math.random() * 10
    },
    metrics: {
      id: server.id,
      requests_total: Math.floor(Math.random() * 1000),
      requests_successful: Math.floor(Math.random() * 900),
      requests_failed: Math.floor(Math.random() * 100),
      avg_response_time: Math.random() * 1000,
      min_response_time: Math.random() * 100,
      max_response_time: Math.random() * 5000,
      bytes_sent: Math.random() * 1000000,
      bytes_received: Math.random() * 1000000,
      last_24h: {
        requests: Math.floor(Math.random() * 100),
        errors: Math.floor(Math.random() * 10),
        avg_response_time: Math.random() * 1000
      },
      last_7d: {
        requests: Math.floor(Math.random() * 500),
        errors: Math.floor(Math.random() * 50),
        avg_response_time: Math.random() * 1000
      }
    },
    tools: server.tools.map((tool, idx) => ({
      id: `${server.id}-tool-${idx}`,
      server_id: server.id,
      name: tool,
      description: `${tool} tool for ${server.name}`,
      enabled: true,
      category: "automation",
      usage_count: Math.floor(Math.random() * 100),
      last_used: Date.now() - Math.random() * 86400000,
      error_count: Math.floor(Math.random() * 5),
      avg_execution_time: Math.random() * 5000
    })),
    resources: Array.from({ length: Math.floor(Math.random() * 5) }, (_, idx) => ({
      id: `${server.id}-resource-${idx}`,
      server_id: server.id,
      name: `Resource ${idx + 1}`,
      description: `Resource ${idx + 1} for ${server.name}`,
      type: "file",
      uri: `/resources/${server.id}/resource-${idx}`,
      last_accessed: Date.now() - Math.random() * 86400000,
      access_count: Math.floor(Math.random() * 50)
    }))
  })) as MCPServerWithStatus[];

  useEffect(() => {
    setServers(mockEnhancedServers);
  }, []);
  
  const getStatusIcon = (status: MCPServer['status']) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'disconnected':
        return <XCircle className="h-4 w-4 text-gray-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'connecting':
        return <AlertCircle className="h-4 w-4 text-yellow-500 animate-pulse" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: MCPServer['status']) => {
    switch (status) {
      case 'connected':
        return 'bg-green-500';
      case 'disconnected':
        return 'bg-gray-500';
      case 'error':
        return 'bg-red-500';
      case 'connecting':
        return 'bg-yellow-500';
      default:
        return 'bg-gray-500';
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    // Simulate API call to refresh server data
    await new Promise(resolve => setTimeout(resolve, 1000));
    // In real implementation, this would call the OpenCode API
    setServers(mockEnhancedServers);
    setRefreshing(false);
  };

  const handleStartServer = async (serverId: string) => {
    setServers(prev => prev.map(s => 
      s.id === serverId ? { 
        ...s, 
        status: { ...s.status, status: 'connecting' }
      } : s
    ));
    
    // Simulate connection
    setTimeout(() => {
      setServers(prev => prev.map(s => 
        s.id === serverId ? { 
          ...s, 
          status: { 
            ...s.status, 
            status: 'connected', 
            connected_at: Date.now(),
            last_ping: Date.now()
          }
        } : s
      ));
    }, 2000);
  };

  const handleStopServer = async (serverId: string) => {
    setServers(prev => prev.map(s => 
      s.id === serverId ? { 
        ...s, 
        status: { ...s.status, status: 'disconnected' }
      } : s
    ));
  };

  const handleDeleteServer = async (serverId: string) => {
    setServers(prev => prev.filter(s => s.id !== serverId));
  };

  const handleEditServer = (serverId: string) => {
    const server = servers.find(s => s.id === serverId);
    if (server) {
      setSelectedServer(server);
      setShowEditDialog(true);
    }
  };

  const handleSaveServer = async (config: MCPServerConfig) => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      if (selectedServer) {
        // Update existing server
        setServers(prev => prev.map(s => 
          s.id === config.id ? { ...s, ...config } : s
        ));
      } else {
        // Add new server
        const newServer: MCPServerWithStatus = {
          ...config,
          status: {
            id: config.id,
            status: 'disconnected',
            tools_count: 0,
            resources_count: 0
          },
          health: {
            id: config.id,
            healthy: false
          },
          metrics: {
            id: config.id,
            requests_total: 0,
            requests_successful: 0,
            requests_failed: 0,
            avg_response_time: 0,
            min_response_time: 0,
            max_response_time: 0,
            bytes_sent: 0,
            bytes_received: 0,
            last_24h: { requests: 0, errors: 0, avg_response_time: 0 },
            last_7d: { requests: 0, errors: 0, avg_response_time: 0 }
          },
          tools: [],
          resources: []
        };
        setServers(prev => [...prev, newServer]);
      }
      
      setShowAddDialog(false);
      setShowEditDialog(false);
      setSelectedServer(null);
    } catch (error) {
      console.error('Failed to save server:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTestServer = async (config: Partial<MCPServerConfig>): Promise<MCPServerTestResult> => {
    // Simulate connection test
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const success = Math.random() > 0.3; // 70% success rate for demo
    
    return {
      success,
      status: success ? 'reachable' : 'unreachable',
      response_time: success ? Math.random() * 1000 : undefined,
      error: success ? undefined : 'Connection refused or server not responding',
      details: success ? {
        tools_discovered: Math.floor(Math.random() * 10),
        resources_discovered: Math.floor(Math.random() * 5),
        capabilities: ['tools', 'resources', 'notifications']
      } : undefined,
      timestamp: Date.now()
    };
  };

  const handleInstallTemplate = async (template: MCPServerTemplate) => {
    setLoading(true);
    try {
      // Simulate template installation
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const config: MCPServerConfig = {
        ...template.config,
        id: template.id,
        name: template.name,
        description: template.description,
        type: template.config.type || 'local',
        enabled: template.config.enabled ?? true,
        created_at: Date.now(),
        updated_at: Date.now()
      } as MCPServerConfig;
      
      await handleSaveServer(config);
      setShowTemplatesDialog(false);
    } catch (error) {
      console.error('Failed to install template:', error);
    } finally {
      setLoading(false);
    }
  };

  const getInstalledServerIds = () => {
    return servers.map(s => s.id);
  };

  const connectedCount = servers.filter(s => s.status.status === 'connected').length;
  const totalCount = servers.length;

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
              <Plug className="h-6 w-6 text-blue-500" />
              <h1 className="text-2xl font-bold">MCP Servers</h1>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              onClick={handleRefresh}
              disabled={refreshing}
            >
              <RefreshCw className={cn("h-4 w-4 mr-2", refreshing && "animate-spin")} />
              Refresh
            </Button>
            <Button 
              variant="outline"
              onClick={() => setShowTemplatesDialog(true)}
            >
              <Package className="h-4 w-4 mr-2" />
              Templates
            </Button>
            <Button 
              variant="outline"
              onClick={() => setShowImportExportDialog(true)}
            >
              <Download className="h-4 w-4 mr-2" />
              Import/Export
            </Button>
            <Button onClick={() => setShowAddDialog(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Add Server
            </Button>
          </div>
        </div>
        <p className="text-muted-foreground mt-2">
          Manage Model Context Protocol servers
        </p>
      </div>

      {/* Stats Bar */}
      <div className="px-6 py-4 bg-muted/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <Plug className="h-4 w-4" />
              <span className="text-sm font-medium">Servers</span>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-muted-foreground">
              Configured Servers
            </span>
            <div className="text-right">
              <div className="text-lg font-bold">{connectedCount} servers configured</div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="px-6 border-b border-border">
        <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)}>
          <TabsList className="grid w-full grid-cols-4 max-w-2xl">
            <TabsTrigger value="dashboard" className="flex items-center space-x-2">
              <Monitor className="h-4 w-4" />
              <span>Dashboard</span>
            </TabsTrigger>
            <TabsTrigger value="health" className="flex items-center space-x-2">
              <Activity className="h-4 w-4" />
              <span>Health</span>
            </TabsTrigger>
            <TabsTrigger value="templates" className="flex items-center space-x-2">
              <Package className="h-4 w-4" />
              <span>Templates</span>
            </TabsTrigger>
            <TabsTrigger value="logs" className="flex items-center space-x-2">
              <FileDown className="h-4 w-4" />
              <span>Logs</span>
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <Tabs value={activeTab} className="h-full">
          <TabsContent value="dashboard" className="h-full mt-0">
            <ServerDashboard
              servers={servers}
              onRefresh={handleRefresh}
              onStartServer={handleStartServer}
              onStopServer={handleStopServer}
              onDeleteServer={handleDeleteServer}
              onEditServer={handleEditServer}
              refreshing={refreshing}
            />
          </TabsContent>

          <TabsContent value="health" className="h-full mt-0">
            <ServerHealthMonitor
              servers={servers}
              onRefresh={handleRefresh}
              onConfigureThresholds={(thresholds) => {
                console.log('Configure thresholds:', thresholds);
                // In real implementation, this would save thresholds to OpenCode config
              }}
              refreshing={refreshing}
            />
          </TabsContent>

          <TabsContent value="templates" className="h-full mt-0">
            <ServerTemplates
              onInstallTemplate={handleInstallTemplate}
              installedServers={getInstalledServerIds()}
            />
          </TabsContent>

          <TabsContent value="logs" className="h-full mt-0">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="h-5 w-5" />
                  <span>Server Logs</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12 text-muted-foreground">
                  <Activity className="h-12 w-12 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Logs Coming Soon</h3>
                  <p>Real-time server logs and debugging information will be available here</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Dialogs */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
          <DialogHeader>
            <DialogTitle>Add MCP Server</DialogTitle>
          </DialogHeader>
          <ServerConfigForm
            onSave={handleSaveServer}
            onTest={handleTestServer}
            onCancel={() => setShowAddDialog(false)}
            loading={loading}
          />
        </DialogContent>
      </Dialog>

      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
          <DialogHeader>
            <DialogTitle>Edit MCP Server</DialogTitle>
          </DialogHeader>
          <ServerConfigForm
            server={selectedServer || undefined}
            onSave={handleSaveServer}
            onTest={handleTestServer}
            onCancel={() => {
              setShowEditDialog(false);
              setSelectedServer(null);
            }}
            loading={loading}
          />
        </DialogContent>
      </Dialog>

      <Dialog open={showImportExportDialog} onOpenChange={setShowImportExportDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Import/Export MCP Server Configurations</DialogTitle>
          </DialogHeader>
          <div className="space-y-6 py-4">
            <Tabs defaultValue="export">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="export">Export</TabsTrigger>
                <TabsTrigger value="import">Import</TabsTrigger>
              </TabsList>
              
              <TabsContent value="export" className="space-y-4">
                <div className="space-y-2">
                  <Label>Export Format</Label>
                  <Select defaultValue="json">
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="json">JSON Configuration</SelectItem>
                      <SelectItem value="yaml">YAML Configuration</SelectItem>
                      <SelectItem value="backup">Full Backup (ZIP)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label>Select Servers to Export</Label>
                  <div className="max-h-40 overflow-y-auto border rounded p-3 space-y-2">
                    {servers.map(server => (
                      <div key={server.id} className="flex items-center space-x-2">
                        <input type="checkbox" defaultChecked />
                        <span className="text-sm">{server.name}</span>
                        <Badge variant="outline" className={cn(
                          server.status.status === 'connected' ? 'text-green-600' : 'text-gray-600'
                        )}>
                          {server.status.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
                
                <Button 
                  className="w-full"
                  onClick={() => {
                    // Simulate export
                    const config = { servers: servers.map(s => ({ id: s.id, name: s.name, type: s.type })) };
                    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'mcp-servers-config.json';
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  <FileDown className="h-4 w-4 mr-2" />
                  Export Configuration
                </Button>
              </TabsContent>
              
              <TabsContent value="import" className="space-y-4">
                <div className="space-y-2">
                  <Label>Import Configuration File</Label>
                  <Input 
                    type="file" 
                    accept=".json,.yaml,.yml,.zip"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        // Handle file import (placeholder)
                        console.log('Importing file:', file.name);
                      }
                    }}
                  />
                </div>
                
                <div className="p-4 bg-blue-50 rounded border border-blue-200">
                  <div className="flex items-start space-x-2">
                    <FileUp className="h-5 w-5 text-blue-600 mt-0.5" />
                    <div>
                      <div className="font-medium text-blue-900">Import Guidelines</div>
                      <ul className="text-sm text-blue-700 mt-1 space-y-1">
                        <li>• Supported formats: JSON, YAML, ZIP backup files</li>
                        <li>• Existing servers with same ID will be updated</li>
                        <li>• New servers will be added to your configuration</li>
                        <li>• Invalid configurations will be skipped with warnings</li>
                      </ul>
                    </div>
                  </div>
                </div>
                
                <Button 
                  className="w-full" 
                  disabled
                  onClick={() => {
                    // Placeholder for import functionality
                    console.log('Import functionality will be implemented');
                  }}
                >
                  <FileUp className="h-4 w-4 mr-2" />
                  Import Configuration
                </Button>
              </TabsContent>
            </Tabs>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

