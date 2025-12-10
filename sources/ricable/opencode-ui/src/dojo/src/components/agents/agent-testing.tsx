"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  TestTube,
  Play,
  Pause,
  Square,
  Plus,
  Edit,
  Trash2,
  CheckCircle,
  XCircle,
  AlertCircle,
  Clock,
  Target,
  Activity,
  BarChart3,
  FileText,
  Code,
  Settings,
  Download,
  Upload,
  RefreshCw,
  Eye,
  Copy
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { openCodeClient } from "@/lib/opencode-client";
import type { 
  Agent,
  AgentTestCase,
  AgentTestResult,
  AgentTestSuite,
  ValidationRule
} from "@/types/opencode";

interface AgentTestingProps {
  agent: Agent;
  onClose?: () => void;
}

type TestModalType = "create" | "edit" | "view" | "run" | null;

export function AgentTesting({ agent, onClose }: AgentTestingProps) {
  const [testSuite, setTestSuite] = useState<AgentTestSuite | null>(null);
  const [testCases, setTestCases] = useState<AgentTestCase[]>([]);
  const [testResults, setTestResults] = useState<AgentTestResult[]>([]);
  const [selectedTest, setSelectedTest] = useState<AgentTestCase | null>(null);
  const [selectedResult, setSelectedResult] = useState<AgentTestResult | null>(null);
  const [activeModal, setActiveModal] = useState<TestModalType>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [activeTab, setActiveTab] = useState("tests");
  const { toast } = useToast();

  // New test form state
  const [newTest, setNewTest] = useState<Partial<AgentTestCase>>({
    name: "",
    description: "",
    input_task: "",
    expected_outputs: [""],
    validation_rules: [],
    timeout_ms: 300000,
    enabled: true
  });

  useEffect(() => {
    loadTestData();
  }, [agent.id]);

  const loadTestData = async () => {
    try {
      setLoading(true);
      const [resultsData] = await Promise.all([
        openCodeClient.getAgentTestResults(agent.id).catch(() => [])
      ]);

      setTestResults(resultsData);
      
      // Create mock test suite and cases for demonstration
      setTestSuite(getMockTestSuite());
      setTestCases(getMockTestCases());
    } catch (error) {
      console.error("Failed to load test data:", error);
      // Provide mock data
      setTestSuite(getMockTestSuite());
      setTestCases(getMockTestCases());
      setTestResults(getMockTestResults());
    } finally {
      setLoading(false);
    }
  };

  const handleCreateTest = async () => {
    try {
      if (!newTest.name || !newTest.input_task) {
        toast({
          title: "Validation Error",
          description: "Please fill in all required fields",
          variant: "destructive"
        });
        return;
      }

      const testToCreate = {
        ...newTest,
        expected_outputs: newTest.expected_outputs?.filter(output => output.trim()) || [],
        validation_rules: newTest.validation_rules || []
      } as Omit<AgentTestCase, 'id'>;

      const createdTest = await openCodeClient.createAgentTest(agent.id, testToCreate);
      setTestCases(prev => [...prev, createdTest]);
      setActiveModal(null);
      resetNewTestForm();
      
      toast({
        title: "Success",
        description: "Test case created successfully"
      });
    } catch (error) {
      console.error("Failed to create test:", error);
      toast({
        title: "Error",
        description: "Failed to create test case",
        variant: "destructive"
      });
    }
  };

  const handleRunTest = async (testCase: AgentTestCase) => {
    try {
      setRunning(true);
      const result = await openCodeClient.runAgentTest(agent.id, testCase.id);
      setTestResults(prev => [result, ...prev]);
      
      toast({
        title: "Success",
        description: `Test "${testCase.name}" completed`
      });
    } catch (error) {
      console.error("Failed to run test:", error);
      toast({
        title: "Error",
        description: "Failed to run test",
        variant: "destructive"
      });
    } finally {
      setRunning(false);
    }
  };

  const handleRunAllTests = async () => {
    try {
      setRunning(true);
      const suite = await openCodeClient.runAgentTestSuite(agent.id);
      setTestSuite(suite);
      await loadTestData(); // Reload to get updated results
      
      toast({
        title: "Success",
        description: "Test suite completed"
      });
    } catch (error) {
      console.error("Failed to run test suite:", error);
      toast({
        title: "Error",
        description: "Failed to run test suite",
        variant: "destructive"
      });
    } finally {
      setRunning(false);
    }
  };

  const resetNewTestForm = () => {
    setNewTest({
      name: "",
      description: "",
      input_task: "",
      expected_outputs: [""],
      validation_rules: [],
      timeout_ms: 300000,
      enabled: true
    });
  };

  const addExpectedOutput = () => {
    setNewTest(prev => ({
      ...prev,
      expected_outputs: [...(prev.expected_outputs || []), ""]
    }));
  };

  const updateExpectedOutput = (index: number, value: string) => {
    setNewTest(prev => ({
      ...prev,
      expected_outputs: prev.expected_outputs?.map((output, i) => i === index ? value : output) || []
    }));
  };

  const removeExpectedOutput = (index: number) => {
    setNewTest(prev => ({
      ...prev,
      expected_outputs: prev.expected_outputs?.filter((_, i) => i !== index) || []
    }));
  };

  const addValidationRule = () => {
    setNewTest(prev => ({
      ...prev,
      validation_rules: [
        ...(prev.validation_rules || []),
        { type: "contains", value: "", description: "" }
      ]
    }));
  };

  const updateValidationRule = (index: number, field: keyof ValidationRule, value: string) => {
    setNewTest(prev => ({
      ...prev,
      validation_rules: prev.validation_rules?.map((rule, i) => 
        i === index ? { ...rule, [field]: value } : rule
      ) || []
    }));
  };

  const removeValidationRule = (index: number) => {
    setNewTest(prev => ({
      ...prev,
      validation_rules: prev.validation_rules?.filter((_, i) => i !== index) || []
    }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "passed":
        return CheckCircle;
      case "failed":
        return XCircle;
      case "skipped":
        return AlertCircle;
      default:
        return Clock;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "passed":
        return "text-green-500";
      case "failed":
        return "text-red-500";
      case "skipped":
        return "text-yellow-500";
      default:
        return "text-gray-500";
    }
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <TestTube className="h-6 w-6" />
            <span>Testing {agent.name}</span>
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
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
            <TestTube className="h-6 w-6" />
            <span>Testing {agent.name}</span>
          </h2>
          <p className="text-muted-foreground">Automated testing and validation</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            onClick={handleRunAllTests}
            disabled={running}
          >
            <Play className="h-4 w-4 mr-2" />
            {running ? "Running..." : "Run All Tests"}
          </Button>
          <Button onClick={() => setActiveModal("create")}>
            <Plus className="h-4 w-4 mr-2" />
            New Test
          </Button>
          {onClose && (
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </div>

      {/* Test Suite Overview */}
      {testSuite && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Tests</p>
                  <p className="text-2xl font-bold">{testSuite.total_tests}</p>
                </div>
                <TestTube className="h-8 w-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Passed</p>
                  <p className="text-2xl font-bold text-green-500">{testSuite.passed_tests}</p>
                </div>
                <CheckCircle className="h-8 w-8 text-green-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Failed</p>
                  <p className="text-2xl font-bold text-red-500">{testSuite.failed_tests}</p>
                </div>
                <XCircle className="h-8 w-8 text-red-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Pass Rate</p>
                  <p className="text-2xl font-bold">{testSuite.pass_rate.toFixed(1)}%</p>
                </div>
                <Target className="h-8 w-8 text-orange-500" />
              </div>
              <Progress value={testSuite.pass_rate} className="mt-2" />
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="tests">Test Cases</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
          <TabsTrigger value="coverage">Coverage</TabsTrigger>
        </TabsList>

        <TabsContent value="tests" className="space-y-4">
          {testCases.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-12"
            >
              <TestTube className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No test cases created</h3>
              <p className="text-muted-foreground mb-6">
                Create your first test case to validate agent behavior
              </p>
              <Button onClick={() => setActiveModal("create")}>
                <Plus className="h-4 w-4 mr-2" />
                Create Test Case
              </Button>
            </motion.div>
          ) : (
            <div className="space-y-4">
              <AnimatePresence>
                {testCases.map((testCase) => (
                  <motion.div
                    key={testCase.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                  >
                    <Card>
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded-full ${testCase.enabled ? "bg-green-100" : "bg-gray-100"}`}>
                              <TestTube className={`h-4 w-4 ${testCase.enabled ? "text-green-600" : "text-gray-400"}`} />
                            </div>
                            <div>
                              <CardTitle className="text-base">{testCase.name}</CardTitle>
                              <p className="text-sm text-muted-foreground">{testCase.description}</p>
                            </div>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Badge variant={testCase.enabled ? "default" : "secondary"}>
                              {testCase.enabled ? "Enabled" : "Disabled"}
                            </Badge>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon">
                                  <Settings className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem onClick={() => handleRunTest(testCase)}>
                                  <Play className="h-4 w-4 mr-2" />
                                  Run Test
                                </DropdownMenuItem>
                                <DropdownMenuItem onClick={() => {
                                  setSelectedTest(testCase);
                                  setActiveModal("view");
                                }}>
                                  <Eye className="h-4 w-4 mr-2" />
                                  View Details
                                </DropdownMenuItem>
                                <DropdownMenuItem onClick={() => {
                                  setSelectedTest(testCase);
                                  setActiveModal("edit");
                                }}>
                                  <Edit className="h-4 w-4 mr-2" />
                                  Edit
                                </DropdownMenuItem>
                                <DropdownMenuItem>
                                  <Copy className="h-4 w-4 mr-2" />
                                  Duplicate
                                </DropdownMenuItem>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem className="text-destructive">
                                  <Trash2 className="h-4 w-4 mr-2" />
                                  Delete
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <div className="bg-muted rounded-lg p-3 mb-3">
                          <p className="text-sm font-medium mb-1">Input Task:</p>
                          <p className="text-sm text-muted-foreground">{testCase.input_task}</p>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">
                            {testCase.expected_outputs.length} expected output(s)
                          </span>
                          <span className="text-muted-foreground">
                            Timeout: {formatDuration(testCase.timeout_ms)}
                          </span>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
        </TabsContent>

        <TabsContent value="results" className="space-y-4">
          {testResults.length === 0 ? (
            <div className="text-center py-12">
              <BarChart3 className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No test results yet</h3>
              <p className="text-muted-foreground">Run some tests to see results here</p>
            </div>
          ) : (
            <div className="space-y-4">
              {testResults.map((result) => {
                const StatusIcon = getStatusIcon(result.status);
                return (
                  <Card key={result.test_name + result.timestamp}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <StatusIcon className={`h-5 w-5 ${getStatusColor(result.status)}`} />
                          <div>
                            <p className="font-medium">{result.test_name}</p>
                            <p className="text-sm text-muted-foreground">{result.test_description}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge variant={result.status === "passed" ? "default" : result.status === "failed" ? "destructive" : "secondary"}>
                            {result.status}
                          </Badge>
                          <p className="text-sm text-muted-foreground mt-1">
                            {formatDuration(result.duration_ms)}
                          </p>
                        </div>
                      </div>
                      {result.error_message && (
                        <Alert className="mt-3">
                          <AlertCircle className="h-4 w-4" />
                          <AlertDescription>{result.error_message}</AlertDescription>
                        </Alert>
                      )}
                      <div className="mt-3 pt-3 border-t">
                        <p className="text-xs text-muted-foreground">
                          Executed: {new Date(result.timestamp).toLocaleString()}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </TabsContent>

        <TabsContent value="coverage" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Test Coverage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Function Coverage</span>
                      <span>85%</span>
                    </div>
                    <Progress value={85} />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Feature Coverage</span>
                      <span>72%</span>
                    </div>
                    <Progress value={72} />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Edge Cases</span>
                      <span>45%</span>
                    </div>
                    <Progress value={45} />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Coverage Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-start space-x-2">
                    <AlertCircle className="h-4 w-4 text-yellow-500 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium">Error Handling</p>
                      <p className="text-xs text-muted-foreground">Add tests for timeout scenarios</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-2">
                    <AlertCircle className="h-4 w-4 text-yellow-500 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium">Performance Tests</p>
                      <p className="text-xs text-muted-foreground">Include load testing for large tasks</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-2">
                    <AlertCircle className="h-4 w-4 text-yellow-500 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium">Integration Tests</p>
                      <p className="text-xs text-muted-foreground">Test with different providers</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Create Test Modal */}
      <Dialog open={activeModal === "create"} onOpenChange={(open) => !open && setActiveModal(null)}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Create Test Case</DialogTitle>
          </DialogHeader>
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="name">Test Name *</Label>
                <Input
                  id="name"
                  value={newTest.name}
                  onChange={(e) => setNewTest(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Enter test name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="timeout">Timeout (seconds)</Label>
                <Input
                  id="timeout"
                  type="number"
                  value={(newTest.timeout_ms || 300000) / 1000}
                  onChange={(e) => setNewTest(prev => ({ ...prev, timeout_ms: parseInt(e.target.value) * 1000 }))}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={newTest.description}
                onChange={(e) => setNewTest(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Describe what this test validates"
                rows={2}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="input_task">Input Task *</Label>
              <Textarea
                id="input_task"
                value={newTest.input_task}
                onChange={(e) => setNewTest(prev => ({ ...prev, input_task: e.target.value }))}
                placeholder="The task to give to the agent"
                rows={3}
              />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Expected Outputs</Label>
                <Button type="button" variant="outline" size="sm" onClick={addExpectedOutput}>
                  <Plus className="h-3 w-3 mr-1" />
                  Add
                </Button>
              </div>
              {newTest.expected_outputs?.map((output, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <Textarea
                    value={output}
                    onChange={(e) => updateExpectedOutput(index, e.target.value)}
                    placeholder="Expected output or behavior"
                    rows={2}
                    className="flex-1"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => removeExpectedOutput(index)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Validation Rules</Label>
                <Button type="button" variant="outline" size="sm" onClick={addValidationRule}>
                  <Plus className="h-3 w-3 mr-1" />
                  Add Rule
                </Button>
              </div>
              {newTest.validation_rules?.map((rule, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <Select
                    value={rule.type}
                    onValueChange={(value) => updateValidationRule(index, "type", value)}
                  >
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="contains">Contains</SelectItem>
                      <SelectItem value="matches">Matches</SelectItem>
                      <SelectItem value="not_contains">Not Contains</SelectItem>
                      <SelectItem value="json_schema">JSON Schema</SelectItem>
                      <SelectItem value="custom">Custom</SelectItem>
                    </SelectContent>
                  </Select>
                  <Input
                    value={rule.value}
                    onChange={(e) => updateValidationRule(index, "value", e.target.value)}
                    placeholder="Rule value"
                    className="flex-1"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => removeValidationRule(index)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                checked={newTest.enabled}
                onCheckedChange={(checked) => setNewTest(prev => ({ ...prev, enabled: checked }))}
              />
              <Label>Enable this test</Label>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setActiveModal(null)}>
                Cancel
              </Button>
              <Button onClick={handleCreateTest}>
                Create Test
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Mock data functions
function getMockTestSuite(): AgentTestSuite {
  return {
    id: "suite-1",
    name: "Agent Test Suite",
    description: "Comprehensive testing for the agent",
    agent_id: "agent-1",
    tests: [],
    last_run: "2024-01-20T15:30:00Z",
    pass_rate: 85.5,
    total_tests: 8,
    passed_tests: 7,
    failed_tests: 1
  };
}

function getMockTestCases(): AgentTestCase[] {
  return [
    {
      id: "test-1",
      name: "Code Review Test",
      description: "Tests the agent's ability to review code and provide feedback",
      input_task: "Review this JavaScript function for potential issues and improvements",
      expected_outputs: [
        "Identifies potential bugs",
        "Suggests improvements",
        "Follows coding standards"
      ],
      validation_rules: [
        {
          type: "contains",
          value: "function",
          description: "Should mention function analysis"
        }
      ],
      timeout_ms: 300000,
      enabled: true
    },
    {
      id: "test-2",
      name: "Documentation Generation",
      description: "Tests the agent's ability to generate documentation",
      input_task: "Generate documentation for the provided API endpoints",
      expected_outputs: [
        "Clear API documentation",
        "Parameter descriptions",
        "Example usage"
      ],
      validation_rules: [
        {
          type: "contains",
          value: "API",
          description: "Should mention API in documentation"
        }
      ],
      timeout_ms: 240000,
      enabled: true
    }
  ];
}

function getMockTestResults(): AgentTestResult[] {
  return [
    {
      agent_id: "agent-1",
      test_name: "Code Review Test",
      test_description: "Tests code review capabilities",
      input_task: "Review this JavaScript function",
      expected_outcome: "Identifies issues and suggests improvements",
      actual_outcome: "Successfully identified 3 issues and provided 5 improvement suggestions",
      status: "passed",
      duration_ms: 45000,
      metrics: {
        total_tokens: 2500,
        cost_usd: 0.0375,
        message_count: 3
      },
      timestamp: "2024-01-20T15:30:00Z"
    },
    {
      agent_id: "agent-1",
      test_name: "Documentation Generation",
      test_description: "Tests documentation generation",
      input_task: "Generate API documentation",
      expected_outcome: "Creates comprehensive API docs",
      actual_outcome: "Generated partial documentation",
      status: "failed",
      duration_ms: 120000,
      error_message: "Documentation incomplete - missing parameter descriptions",
      metrics: {
        total_tokens: 1800,
        cost_usd: 0.027,
        message_count: 2
      },
      timestamp: "2024-01-20T14:15:00Z"
    }
  ];
}