"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Save, 
  TestTube, 
  AlertCircle, 
  CheckCircle, 
  XCircle,
  Info,
  Eye,
  EyeOff,
  Copy,
  FileText,
  Settings,
  Zap,
  Globe,
  Terminal,
  Plus,
  Minus,
  HelpCircle
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Tooltip, 
  TooltipContent, 
  TooltipTrigger, 
  TooltipProvider 
} from "@/components/ui/tooltip";

import { 
  MCPServerConfig, 
  MCPServerFormData, 
  MCPServerValidation,
  MCPServerTestResult,
  MCP_SERVER_CATEGORIES 
} from "@/lib/types/mcp";
import { cn } from "@/lib/utils";

interface ServerConfigFormProps {
  server?: MCPServerConfig;
  onSave: (config: MCPServerConfig) => Promise<void>;
  onTest: (config: Partial<MCPServerConfig>) => Promise<MCPServerTestResult>;
  onCancel: () => void;
  loading?: boolean;
}

export function ServerConfigForm({ 
  server, 
  onSave, 
  onTest, 
  onCancel,
  loading = false 
}: ServerConfigFormProps) {
  const [formData, setFormData] = useState<MCPServerFormData>({
    name: server?.name || "",
    description: server?.description || "",
    type: server?.type || "local",
    enabled: server?.enabled ?? true,
    
    // Local server fields
    command: server?.command?.join(" ") || "",
    args: "",
    environment: server?.environment ? JSON.stringify(server.environment, null, 2) : "",
    workingDirectory: server?.workingDirectory || "",
    
    // Remote server fields
    url: server?.url || "",
    headers: server?.headers ? JSON.stringify(server.headers, null, 2) : "",
    authType: server?.authentication?.type || "none",
    authToken: server?.authentication?.token || "",
    authUsername: server?.authentication?.username || "",
    authPassword: server?.authentication?.password || "",
    
    // Advanced fields
    timeout: server?.timeout || 30000,
    retryAttempts: server?.retryAttempts || 3,
    retryDelay: server?.retryDelay || 1000,
    autoRestart: server?.autoRestart ?? true,
    
    // Metadata
    tags: server?.tags?.join(", ") || "",
    category: server?.category || "custom",
    version: server?.version || "1.0.0",
    author: server?.author || "",
    documentation: server?.documentation || ""
  });

  const [validation, setValidation] = useState<MCPServerValidation>({
    valid: true,
    errors: [],
    warnings: []
  });

  const [testResult, setTestResult] = useState<MCPServerTestResult | null>(null);
  const [testing, setTesting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [activeTab, setActiveTab] = useState("basic");

  // Validation logic
  useEffect(() => {
    validateForm();
  }, [formData]);

  const validateForm = () => {
    const errors: MCPServerValidation["errors"] = [];
    const warnings: MCPServerValidation["warnings"] = [];

    // Basic validation
    if (!formData.name.trim()) {
      errors.push({ field: "name", message: "Server name is required", code: "REQUIRED" });
    }

    if (formData.type === "local") {
      if (!formData.command.trim()) {
        errors.push({ field: "command", message: "Command is required for local servers", code: "REQUIRED" });
      }
    } else if (formData.type === "remote") {
      if (!formData.url.trim()) {
        errors.push({ field: "url", message: "URL is required for remote servers", code: "REQUIRED" });
      } else {
        try {
          new URL(formData.url);
        } catch {
          errors.push({ field: "url", message: "Invalid URL format", code: "INVALID_FORMAT" });
        }
      }
    }

    // JSON validation
    if (formData.environment) {
      try {
        JSON.parse(formData.environment);
      } catch {
        errors.push({ field: "environment", message: "Invalid JSON format", code: "INVALID_JSON" });
      }
    }

    if (formData.headers) {
      try {
        JSON.parse(formData.headers);
      } catch {
        errors.push({ field: "headers", message: "Invalid JSON format", code: "INVALID_JSON" });
      }
    }

    // Warnings
    if (formData.timeout < 5000) {
      warnings.push({ field: "timeout", message: "Timeout below 5 seconds may cause connection issues", code: "LOW_TIMEOUT" });
    }

    if (formData.retryAttempts > 10) {
      warnings.push({ field: "retryAttempts", message: "High retry attempts may cause delays", code: "HIGH_RETRIES" });
    }

    setValidation({
      valid: errors.length === 0,
      errors,
      warnings
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validation.valid) return;

    try {
      const config: MCPServerConfig = {
        id: server?.id || `mcp-${Date.now()}`,
        name: formData.name.trim(),
        description: formData.description.trim() || undefined,
        type: formData.type,
        enabled: formData.enabled,
        
        // Local server config
        ...(formData.type === "local" && {
          command: formData.command.trim().split(/\s+/),
          environment: formData.environment ? JSON.parse(formData.environment) : undefined,
          workingDirectory: formData.workingDirectory.trim() || undefined,
        }),
        
        // Remote server config
        ...(formData.type === "remote" && {
          url: formData.url.trim(),
          headers: formData.headers ? JSON.parse(formData.headers) : undefined,
          ...(formData.authType !== "none" && {
            authentication: {
              type: formData.authType as any,
              token: formData.authToken || undefined,
              username: formData.authUsername || undefined,
              password: formData.authPassword || undefined,
            }
          })
        }),
        
        // Advanced config
        timeout: formData.timeout,
        retryAttempts: formData.retryAttempts,
        retryDelay: formData.retryDelay,
        autoRestart: formData.autoRestart,
        
        // Metadata
        tags: formData.tags ? formData.tags.split(",").map(t => t.trim()).filter(Boolean) : undefined,
        category: formData.category,
        version: formData.version.trim() || undefined,
        author: formData.author.trim() || undefined,
        documentation: formData.documentation.trim() || undefined,
        
        // Timestamps
        created_at: server?.created_at || Date.now(),
        updated_at: Date.now()
      };

      await onSave(config);
    } catch (error) {
      console.error("Failed to save server config:", error);
    }
  };

  const handleTest = async () => {
    if (!validation.valid) return;

    setTesting(true);
    setTestResult(null);

    try {
      const config: Partial<MCPServerConfig> = {
        name: formData.name.trim(),
        type: formData.type,
        
        ...(formData.type === "local" && {
          command: formData.command.trim().split(/\s+/),
          environment: formData.environment ? JSON.parse(formData.environment) : undefined,
          workingDirectory: formData.workingDirectory.trim() || undefined,
        }),
        
        ...(formData.type === "remote" && {
          url: formData.url.trim(),
          headers: formData.headers ? JSON.parse(formData.headers) : undefined,
          ...(formData.authType !== "none" && {
            authentication: {
              type: formData.authType as any,
              token: formData.authToken || undefined,
              username: formData.authUsername || undefined,
              password: formData.authPassword || undefined,
            }
          })
        }),
        
        timeout: formData.timeout,
      };

      const result = await onTest(config);
      setTestResult(result);
    } catch (error) {
      setTestResult({
        success: false,
        status: "unreachable",
        error: error instanceof Error ? error.message : "Unknown error",
        timestamp: Date.now()
      });
    } finally {
      setTesting(false);
    }
  };

  const getFieldError = (field: string) => {
    return validation.errors.find(e => e.field === field);
  };

  const getFieldWarning = (field: string) => {
    return validation.warnings.find(w => w.field === field);
  };

  return (
    <TooltipProvider>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">
              {server ? "Edit Server" : "Add Server"}
            </h3>
            <p className="text-sm text-muted-foreground">
              Configure your MCP server connection and settings
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              type="button"
              variant="outline"
              onClick={handleTest}
              disabled={!validation.valid || testing}
            >
              <TestTube className={cn("h-4 w-4 mr-2", testing && "animate-pulse")} />
              {testing ? "Testing..." : "Test Connection"}
            </Button>
          </div>
        </div>

        {/* Test Result */}
        <AnimatePresence>
          {testResult && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              <Alert className={cn(
                testResult.success ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"
              )}>
                {testResult.success ? (
                  <CheckCircle className="h-4 w-4 text-green-600" />
                ) : (
                  <XCircle className="h-4 w-4 text-red-600" />
                )}
                <AlertDescription>
                  <div className="space-y-2">
                    <div className="font-medium">
                      {testResult.success ? "Connection successful!" : "Connection failed"}
                    </div>
                    {testResult.response_time && (
                      <div className="text-sm">
                        Response time: {testResult.response_time}ms
                      </div>
                    )}
                    {testResult.error && (
                      <div className="text-sm text-red-600">
                        {testResult.error}
                      </div>
                    )}
                    {testResult.details && (
                      <div className="text-sm space-y-1">
                        {testResult.details.tools_discovered !== undefined && (
                          <div>Tools discovered: {testResult.details.tools_discovered}</div>
                        )}
                        {testResult.details.resources_discovered !== undefined && (
                          <div>Resources discovered: {testResult.details.resources_discovered}</div>
                        )}
                      </div>
                    )}
                  </div>
                </AlertDescription>
              </Alert>
            </motion.div>
          )}
        </AnimatePresence>

        <form onSubmit={handleSubmit} className="space-y-6">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="basic">Basic</TabsTrigger>
              <TabsTrigger value="connection">Connection</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
              <TabsTrigger value="metadata">Metadata</TabsTrigger>
            </TabsList>

            <TabsContent value="basic" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Settings className="h-5 w-5" />
                    <span>Basic Configuration</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Server Name *</Label>
                    <Input
                      id="name"
                      value={formData.name}
                      onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                      placeholder="e.g., puppeteer-server"
                      className={cn(getFieldError("name") && "border-red-500")}
                    />
                    {getFieldError("name") && (
                      <p className="text-sm text-red-600">{getFieldError("name")?.message}</p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="description">Description</Label>
                    <Textarea
                      id="description"
                      value={formData.description}
                      onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                      placeholder="Brief description of what this server does"
                      rows={3}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="type">Server Type *</Label>
                    <Select 
                      value={formData.type} 
                      onValueChange={(value: "local" | "remote") => 
                        setFormData(prev => ({ ...prev, type: value }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="local">
                          <div className="flex items-center space-x-2">
                            <Terminal className="h-4 w-4" />
                            <span>Local (Command)</span>
                          </div>
                        </SelectItem>
                        <SelectItem value="remote">
                          <div className="flex items-center space-x-2">
                            <Globe className="h-4 w-4" />
                            <span>Remote (URL)</span>
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      id="enabled"
                      checked={formData.enabled}
                      onCheckedChange={(checked) => 
                        setFormData(prev => ({ ...prev, enabled: checked }))
                      }
                    />
                    <Label htmlFor="enabled">Enable server on startup</Label>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="connection" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Zap className="h-5 w-5" />
                    <span>Connection Settings</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {formData.type === "local" ? (
                    <>
                      <div className="space-y-2">
                        <Label htmlFor="command">Command *</Label>
                        <Input
                          id="command"
                          value={formData.command}
                          onChange={(e) => setFormData(prev => ({ ...prev, command: e.target.value }))}
                          placeholder="e.g., npx puppeteer-mcp-server"
                          className={cn(getFieldError("command") && "border-red-500")}
                        />
                        {getFieldError("command") && (
                          <p className="text-sm text-red-600">{getFieldError("command")?.message}</p>
                        )}
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="workingDirectory">Working Directory</Label>
                        <Input
                          id="workingDirectory"
                          value={formData.workingDirectory}
                          onChange={(e) => setFormData(prev => ({ ...prev, workingDirectory: e.target.value }))}
                          placeholder="/path/to/working/directory"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="environment">Environment Variables (JSON)</Label>
                        <Textarea
                          id="environment"
                          value={formData.environment}
                          onChange={(e) => setFormData(prev => ({ ...prev, environment: e.target.value }))}
                          placeholder='{"API_KEY": "your-key", "DEBUG": "true"}'
                          rows={4}
                          className={cn(getFieldError("environment") && "border-red-500")}
                        />
                        {getFieldError("environment") && (
                          <p className="text-sm text-red-600">{getFieldError("environment")?.message}</p>
                        )}
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="space-y-2">
                        <Label htmlFor="url">Server URL *</Label>
                        <Input
                          id="url"
                          value={formData.url}
                          onChange={(e) => setFormData(prev => ({ ...prev, url: e.target.value }))}
                          placeholder="https://example.com/mcp"
                          className={cn(getFieldError("url") && "border-red-500")}
                        />
                        {getFieldError("url") && (
                          <p className="text-sm text-red-600">{getFieldError("url")?.message}</p>
                        )}
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="authType">Authentication</Label>
                        <Select 
                          value={formData.authType} 
                          onValueChange={(value) => 
                            setFormData(prev => ({ ...prev, authType: value as any }))
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">No Authentication</SelectItem>
                            <SelectItem value="bearer">Bearer Token</SelectItem>
                            <SelectItem value="basic">Basic Auth</SelectItem>
                            <SelectItem value="api-key">API Key</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      {formData.authType === "bearer" && (
                        <div className="space-y-2">
                          <Label htmlFor="authToken">Bearer Token</Label>
                          <div className="relative">
                            <Input
                              id="authToken"
                              type={showPassword ? "text" : "password"}
                              value={formData.authToken}
                              onChange={(e) => setFormData(prev => ({ ...prev, authToken: e.target.value }))}
                              placeholder="Enter bearer token"
                            />
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="absolute right-2 top-1/2 -translate-y-1/2"
                              onClick={() => setShowPassword(!showPassword)}
                            >
                              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                            </Button>
                          </div>
                        </div>
                      )}

                      {formData.authType === "basic" && (
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label htmlFor="authUsername">Username</Label>
                            <Input
                              id="authUsername"
                              value={formData.authUsername}
                              onChange={(e) => setFormData(prev => ({ ...prev, authUsername: e.target.value }))}
                              placeholder="Username"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="authPassword">Password</Label>
                            <div className="relative">
                              <Input
                                id="authPassword"
                                type={showPassword ? "text" : "password"}
                                value={formData.authPassword}
                                onChange={(e) => setFormData(prev => ({ ...prev, authPassword: e.target.value }))}
                                placeholder="Password"
                              />
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="absolute right-2 top-1/2 -translate-y-1/2"
                                onClick={() => setShowPassword(!showPassword)}
                              >
                                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                              </Button>
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="space-y-2">
                        <Label htmlFor="headers">Custom Headers (JSON)</Label>
                        <Textarea
                          id="headers"
                          value={formData.headers}
                          onChange={(e) => setFormData(prev => ({ ...prev, headers: e.target.value }))}
                          placeholder='{"X-Custom-Header": "value"}'
                          rows={3}
                          className={cn(getFieldError("headers") && "border-red-500")}
                        />
                        {getFieldError("headers") && (
                          <p className="text-sm text-red-600">{getFieldError("headers")?.message}</p>
                        )}
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="advanced" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Advanced Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="timeout">
                        Timeout (ms)
                        <Tooltip>
                          <TooltipTrigger>
                            <HelpCircle className="h-3 w-3 ml-1 inline" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Maximum time to wait for server response</p>
                          </TooltipContent>
                        </Tooltip>
                      </Label>
                      <Input
                        id="timeout"
                        type="number"
                        value={formData.timeout}
                        onChange={(e) => setFormData(prev => ({ ...prev, timeout: parseInt(e.target.value) || 0 }))}
                        min={1000}
                        max={300000}
                      />
                      {getFieldWarning("timeout") && (
                        <p className="text-sm text-yellow-600">{getFieldWarning("timeout")?.message}</p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="retryAttempts">
                        Retry Attempts
                        <Tooltip>
                          <TooltipTrigger>
                            <HelpCircle className="h-3 w-3 ml-1 inline" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Number of retry attempts on connection failure</p>
                          </TooltipContent>
                        </Tooltip>
                      </Label>
                      <Input
                        id="retryAttempts"
                        type="number"
                        value={formData.retryAttempts}
                        onChange={(e) => setFormData(prev => ({ ...prev, retryAttempts: parseInt(e.target.value) || 0 }))}
                        min={0}
                        max={20}
                      />
                      {getFieldWarning("retryAttempts") && (
                        <p className="text-sm text-yellow-600">{getFieldWarning("retryAttempts")?.message}</p>
                      )}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="retryDelay">Retry Delay (ms)</Label>
                    <Input
                      id="retryDelay"
                      type="number"
                      value={formData.retryDelay}
                      onChange={(e) => setFormData(prev => ({ ...prev, retryDelay: parseInt(e.target.value) || 0 }))}
                      min={100}
                      max={60000}
                    />
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      id="autoRestart"
                      checked={formData.autoRestart}
                      onCheckedChange={(checked) => 
                        setFormData(prev => ({ ...prev, autoRestart: checked }))
                      }
                    />
                    <Label htmlFor="autoRestart">Auto-restart on failure</Label>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="metadata" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <FileText className="h-5 w-5" />
                    <span>Metadata & Documentation</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="category">Category</Label>
                    <Select 
                      value={formData.category} 
                      onValueChange={(value) => 
                        setFormData(prev => ({ ...prev, category: value }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {MCP_SERVER_CATEGORIES.map(category => (
                          <SelectItem key={category} value={category}>
                            {category.charAt(0).toUpperCase() + category.slice(1).replace("-", " ")}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="tags">Tags (comma-separated)</Label>
                    <Input
                      id="tags"
                      value={formData.tags}
                      onChange={(e) => setFormData(prev => ({ ...prev, tags: e.target.value }))}
                      placeholder="automation, web, testing"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="version">Version</Label>
                      <Input
                        id="version"
                        value={formData.version}
                        onChange={(e) => setFormData(prev => ({ ...prev, version: e.target.value }))}
                        placeholder="1.0.0"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="author">Author</Label>
                      <Input
                        id="author"
                        value={formData.author}
                        onChange={(e) => setFormData(prev => ({ ...prev, author: e.target.value }))}
                        placeholder="Your name"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="documentation">Documentation URL</Label>
                    <Input
                      id="documentation"
                      value={formData.documentation}
                      onChange={(e) => setFormData(prev => ({ ...prev, documentation: e.target.value }))}
                      placeholder="https://github.com/example/docs"
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Validation Summary */}
          {(validation.errors.length > 0 || validation.warnings.length > 0) && (
            <Card>
              <CardContent className="pt-6">
                {validation.errors.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2 text-red-600">
                      <AlertCircle className="h-4 w-4" />
                      <span className="font-medium">Errors ({validation.errors.length})</span>
                    </div>
                    <ul className="text-sm text-red-600 space-y-1 ml-6">
                      {validation.errors.map((error, idx) => (
                        <li key={idx}>• {error.message}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {validation.warnings.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2 text-yellow-600">
                      <AlertCircle className="h-4 w-4" />
                      <span className="font-medium">Warnings ({validation.warnings.length})</span>
                    </div>
                    <ul className="text-sm text-yellow-600 space-y-1 ml-6">
                      {validation.warnings.map((warning, idx) => (
                        <li key={idx}>• {warning.message}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Form Actions */}
          <div className="flex justify-end space-x-2">
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
            <Button 
              type="submit" 
              disabled={!validation.valid || loading}
              className="min-w-24"
            >
              {loading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Saving...</span>
                </div>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  {server ? "Update" : "Create"} Server
                </>
              )}
            </Button>
          </div>
        </form>
      </div>
    </TooltipProvider>
  );
}