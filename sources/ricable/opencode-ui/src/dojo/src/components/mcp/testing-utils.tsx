"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  TestTube,
  Play,
  Square,
  CheckCircle,
  XCircle,
  AlertCircle,
  Clock,
  Zap,
  Activity,
  BarChart3,
  RefreshCw,
  Settings,
  Terminal,
  Globe,
  Download,
  Save,
  Eye,
  EyeOff
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
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
  MCPServerTestResult,
  MCPServerWithStatus 
} from "@/lib/types/mcp";
import { cn } from "@/lib/utils";

interface TestSuite {
  id: string;
  name: string;
  description: string;
  tests: Test[];
  enabled: boolean;
}

interface Test {
  id: string;
  name: string;
  description: string;
  type: "connection" | "tool" | "resource" | "performance" | "stress";
  config: any;
  enabled: boolean;
}

interface TestResult {
  id: string;
  test_id: string;
  status: "running" | "passed" | "failed" | "skipped";
  started_at: number;
  completed_at?: number;
  duration?: number;
  message?: string;
  details?: any;
  metrics?: {
    response_time?: number;
    memory_usage?: number;
    cpu_usage?: number;
    throughput?: number;
  };
}

interface TestingUtilsProps {
  servers: MCPServerWithStatus[];
  onRunTest: (serverId: string, testId: string) => Promise<TestResult>;
  onRunSuite: (serverId: string, suiteId: string) => Promise<TestResult[]>;
}

const DEFAULT_TEST_SUITES: TestSuite[] = [
  {
    id: "basic-connectivity",
    name: "Basic Connectivity",
    description: "Test basic server connection and health",
    enabled: true,
    tests: [
      {
        id: "connection-test",
        name: "Connection Test",
        description: "Verify server is reachable and responsive",
        type: "connection",
        enabled: true,
        config: { timeout: 5000 }
      },
      {
        id: "health-check",
        name: "Health Check",
        description: "Check server health endpoint",
        type: "connection",
        enabled: true,
        config: { endpoint: "/health" }
      },
      {
        id: "capabilities-discovery",
        name: "Capabilities Discovery",
        description: "Discover available tools and resources",
        type: "connection",
        enabled: true,
        config: {}
      }
    ]
  },
  {
    id: "tool-functionality",
    name: "Tool Functionality",
    description: "Test individual tool operations",
    enabled: true,
    tests: [
      {
        id: "tool-listing",
        name: "Tool Listing",
        description: "Verify tools can be listed",
        type: "tool",
        enabled: true,
        config: {}
      },
      {
        id: "tool-execution",
        name: "Tool Execution",
        description: "Test tool execution with sample inputs",
        type: "tool",
        enabled: true,
        config: { timeout: 10000 }
      },
      {
        id: "tool-error-handling",
        name: "Error Handling",
        description: "Test error handling with invalid inputs",
        type: "tool",
        enabled: true,
        config: {}
      }
    ]
  },
  {
    id: "performance-benchmarks",
    name: "Performance Benchmarks",
    description: "Measure server performance characteristics",
    enabled: false,
    tests: [
      {
        id: "response-time",
        name: "Response Time",
        description: "Measure average response time under normal load",
        type: "performance",
        enabled: true,
        config: { requests: 100, concurrency: 10 }
      },
      {
        id: "throughput",
        name: "Throughput Test",
        description: "Measure maximum requests per second",
        type: "performance",
        enabled: true,
        config: { duration: 60000, max_concurrency: 50 }
      },
      {
        id: "memory-usage",
        name: "Memory Usage",
        description: "Monitor memory consumption during operations",
        type: "performance",
        enabled: true,
        config: { monitor_duration: 300000 }
      }
    ]
  },
  {
    id: "stress-testing",
    name: "Stress Testing",
    description: "Test server behavior under extreme conditions",
    enabled: false,
    tests: [
      {
        id: "high-concurrency",
        name: "High Concurrency",
        description: "Test with many simultaneous connections",
        type: "stress",
        enabled: true,
        config: { connections: 1000, duration: 120000 }
      },
      {
        id: "large-payloads",
        name: "Large Payloads",
        description: "Test with large request/response payloads",
        type: "stress",
        enabled: true,
        config: { payload_size: 10485760 } // 10MB
      },
      {
        id: "endurance",
        name: "Endurance Test",
        description: "Long-running test to check for memory leaks",
        type: "stress",
        enabled: true,
        config: { duration: 3600000 } // 1 hour
      }
    ]
  }
];

export function TestingUtils({ 
  servers, 
  onRunTest, 
  onRunSuite 
}: TestingUtilsProps) {
  const [selectedServer, setSelectedServer] = useState<string>("");
  const [testSuites, setTestSuites] = useState<TestSuite[]>(DEFAULT_TEST_SUITES);
  const [testResults, setTestResults] = useState<Record<string, TestResult[]>>({});
  const [runningTests, setRunningTests] = useState<Set<string>>(new Set());
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [autoSave, setAutoSave] = useState(true);

  // Initialize results for each server
  useEffect(() => {
    const initialResults: Record<string, TestResult[]> = {};
    servers.forEach(server => {
      if (!testResults[server.id]) {
        initialResults[server.id] = [];
      }
    });
    if (Object.keys(initialResults).length > 0) {
      setTestResults(prev => ({ ...prev, ...initialResults }));
    }
  }, [servers]);

  const handleRunTest = async (testId: string, suiteId?: string) => {
    if (!selectedServer) return;

    const testKey = `${selectedServer}-${testId}`;
    setRunningTests(prev => new Set(prev).add(testKey));

    try {
      const result = await onRunTest(selectedServer, testId);
      
      setTestResults(prev => ({
        ...prev,
        [selectedServer]: [
          ...(prev[selectedServer] || []).filter(r => r.test_id !== testId),
          result
        ]
      }));
    } catch (error) {
      console.error(`Test ${testId} failed:`, error);
    } finally {
      setRunningTests(prev => {
        const next = new Set(prev);
        next.delete(testKey);
        return next;
      });
    }
  };

  const handleRunSuite = async (suiteId: string) => {
    if (!selectedServer) return;

    const suite = testSuites.find(s => s.id === suiteId);
    if (!suite) return;

    const enabledTests = suite.tests.filter(t => t.enabled);
    
    for (const test of enabledTests) {
      await handleRunTest(test.id, suiteId);
    }
  };

  const handleRunAllSuites = async () => {
    if (!selectedServer) return;

    const enabledSuites = testSuites.filter(s => s.enabled);
    
    for (const suite of enabledSuites) {
      await handleRunSuite(suite.id);
    }
  };

  const getTestResult = (testId: string) => {
    if (!selectedServer) return null;
    return testResults[selectedServer]?.find(r => r.test_id === testId);
  };

  const isTestRunning = (testId: string) => {
    const testKey = `${selectedServer}-${testId}`;
    return runningTests.has(testKey);
  };

  const getTestIcon = (result?: TestResult | null, running?: boolean) => {
    if (running) return <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />;
    if (!result) return <Clock className="h-4 w-4 text-gray-500" />;
    
    switch (result.status) {
      case "passed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "skipped":
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getTestBadgeColor = (result?: TestResult | null, running?: boolean) => {
    if (running) return "bg-blue-500";
    if (!result) return "bg-gray-500";
    
    switch (result.status) {
      case "passed":
        return "bg-green-500";
      case "failed":
        return "bg-red-500";
      case "skipped":
        return "bg-yellow-500";
      default:
        return "bg-gray-500";
    }
  };

  const getSuiteProgress = (suite: TestSuite) => {
    if (!selectedServer) return 0;
    
    const results = testResults[selectedServer] || [];
    const suiteResults = results.filter(r => 
      suite.tests.some(t => t.id === r.test_id && t.enabled)
    );
    const enabledTestsCount = suite.tests.filter(t => t.enabled).length;
    
    if (enabledTestsCount === 0) return 100;
    return (suiteResults.length / enabledTestsCount) * 100;
  };

  const getSuiteStatus = (suite: TestSuite) => {
    if (!selectedServer) return "pending";
    
    const results = testResults[selectedServer] || [];
    const suiteResults = results.filter(r => 
      suite.tests.some(t => t.id === r.test_id && t.enabled)
    );
    
    if (suiteResults.length === 0) return "pending";
    
    const hasFailures = suiteResults.some(r => r.status === "failed");
    const allCompleted = suiteResults.length === suite.tests.filter(t => t.enabled).length;
    
    if (hasFailures) return "failed";
    if (allCompleted) return "passed";
    return "running";
  };

  const exportResults = () => {
    if (!selectedServer) return;
    
    const results = testResults[selectedServer] || [];
    const exportData = {
      server_id: selectedServer,
      server_name: servers.find(s => s.id === selectedServer)?.name,
      timestamp: Date.now(),
      results,
      summary: {
        total_tests: results.length,
        passed: results.filter(r => r.status === "passed").length,
        failed: results.filter(r => r.status === "failed").length,
        skipped: results.filter(r => r.status === "skipped").length
      }
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mcp-test-results-${selectedServer}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <TooltipProvider>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">MCP Testing Suite</h2>
            <p className="text-muted-foreground">
              Comprehensive testing and validation for MCP servers
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <Settings className="h-4 w-4 mr-2" />
              {showAdvanced ? "Hide" : "Show"} Advanced
            </Button>
            <Button
              variant="outline"
              onClick={exportResults}
              disabled={!selectedServer || !testResults[selectedServer]?.length}
            >
              <Download className="h-4 w-4 mr-2" />
              Export Results
            </Button>
          </div>
        </div>

        {/* Server Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TestTube className="h-5 w-5" />
              <span>Server Selection</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Target Server</Label>
                <Select value={selectedServer} onValueChange={setSelectedServer}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a server to test" />
                  </SelectTrigger>
                  <SelectContent>
                    {servers.map(server => (
                      <SelectItem key={server.id} value={server.id}>
                        <div className="flex items-center space-x-2">
                          {server.type === "local" ? (
                            <Terminal className="h-4 w-4" />
                          ) : (
                            <Globe className="h-4 w-4" />
                          )}
                          <span>{server.name}</span>
                          <Badge variant={server.status.status === "connected" ? "default" : "secondary"}>
                            {server.status.status}
                          </Badge>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Quick Actions</Label>
                <div className="flex space-x-2">
                  <Button
                    onClick={handleRunAllSuites}
                    disabled={!selectedServer}
                    className="flex-1"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Run All Tests
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setTestResults(prev => ({ ...prev, [selectedServer]: [] }))}
                    disabled={!selectedServer}
                  >
                    Clear Results
                  </Button>
                </div>
              </div>
            </div>

            {/* Advanced Options */}
            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-4 pt-4 border-t"
                >
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="auto-save"
                        checked={autoSave}
                        onCheckedChange={setAutoSave}
                      />
                      <Label htmlFor="auto-save">Auto-save results</Label>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
        </Card>

        {/* Test Suites */}
        {selectedServer && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Test Suites</h3>
            
            {testSuites.map(suite => {
              const progress = getSuiteProgress(suite);
              const status = getSuiteStatus(suite);
              
              return (
                <Card key={suite.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <Switch
                          checked={suite.enabled}
                          onCheckedChange={(checked) => {
                            setTestSuites(prev => prev.map(s => 
                              s.id === suite.id ? { ...s, enabled: checked } : s
                            ));
                          }}
                        />
                        <div>
                          <CardTitle className="text-base">{suite.name}</CardTitle>
                          <p className="text-sm text-muted-foreground">{suite.description}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-3">
                        <div className="text-right">
                          <div className="text-sm font-medium">
                            Progress: {Math.round(progress)}%
                          </div>
                          <Progress value={progress} className="w-24 h-2" />
                        </div>
                        
                        <Badge 
                          variant="outline"
                          className={cn("text-white", {
                            "bg-green-500": status === "passed",
                            "bg-red-500": status === "failed",
                            "bg-blue-500": status === "running",
                            "bg-gray-500": status === "pending"
                          })}
                        >
                          {status}
                        </Badge>
                        
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleRunSuite(suite.id)}
                          disabled={!suite.enabled}
                        >
                          <Play className="h-4 w-4 mr-2" />
                          Run Suite
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  
                  <CardContent>
                    <div className="space-y-3">
                      {suite.tests.map(test => {
                        const result = getTestResult(test.id);
                        const running = isTestRunning(test.id);
                        
                        return (
                          <div key={test.id} className="flex items-center justify-between p-3 border rounded">
                            <div className="flex items-center space-x-3">
                              <Switch
                                checked={test.enabled}
                                onCheckedChange={(checked) => {
                                  setTestSuites(prev => prev.map(s => 
                                    s.id === suite.id ? {
                                      ...s,
                                      tests: s.tests.map(t => 
                                        t.id === test.id ? { ...t, enabled: checked } : t
                                      )
                                    } : s
                                  ));
                                }}
                              />
                              
                              {getTestIcon(result, running)}
                              
                              <div>
                                <div className="font-medium">{test.name}</div>
                                <div className="text-sm text-muted-foreground">{test.description}</div>
                              </div>
                            </div>
                            
                            <div className="flex items-center space-x-3">
                              {result && (
                                <div className="text-right text-sm">
                                  {result.duration && (
                                    <div>Duration: {result.duration}ms</div>
                                  )}
                                  {result.metrics?.response_time && (
                                    <div>Response: {Math.round(result.metrics.response_time)}ms</div>
                                  )}
                                </div>
                              )}
                              
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleRunTest(test.id, suite.id)}
                                disabled={!test.enabled || running}
                              >
                                {running ? (
                                  <RefreshCw className="h-4 w-4 animate-spin" />
                                ) : (
                                  <Play className="h-4 w-4" />
                                )}
                              </Button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}

        {/* Results Summary */}
        {selectedServer && testResults[selectedServer]?.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <BarChart3 className="h-5 w-5" />
                <span>Test Results Summary</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { 
                    label: "Total Tests", 
                    value: testResults[selectedServer].length,
                    icon: TestTube,
                    color: "text-blue-600"
                  },
                  { 
                    label: "Passed", 
                    value: testResults[selectedServer].filter(r => r.status === "passed").length,
                    icon: CheckCircle,
                    color: "text-green-600"
                  },
                  { 
                    label: "Failed", 
                    value: testResults[selectedServer].filter(r => r.status === "failed").length,
                    icon: XCircle,
                    color: "text-red-600"
                  },
                  { 
                    label: "Avg Response", 
                    value: Math.round(
                      testResults[selectedServer]
                        .filter(r => r.metrics?.response_time)
                        .reduce((acc, r) => acc + (r.metrics?.response_time || 0), 0) /
                      testResults[selectedServer].filter(r => r.metrics?.response_time).length || 0
                    ) + "ms",
                    icon: Zap,
                    color: "text-orange-600"
                  }
                ].map((stat, idx) => (
                  <div key={idx} className="text-center">
                    <div className={cn("text-2xl font-bold", stat.color)}>
                      {stat.value}
                    </div>
                    <div className="text-sm text-muted-foreground flex items-center justify-center space-x-1">
                      <stat.icon className="h-4 w-4" />
                      <span>{stat.label}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </TooltipProvider>
  );
}