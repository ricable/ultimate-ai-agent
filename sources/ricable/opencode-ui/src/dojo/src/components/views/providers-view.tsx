"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { 
  ArrowLeft,
  Globe,
  Activity,
  DollarSign,
  Settings,
  BarChart3,
  Shield,
  Target,
  Plus,
  Trash2,
  ArrowUpDown
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { ProviderDashboard } from "@/components/opencode/provider-dashboard";
import { ProviderHealth } from "@/components/opencode/provider-health";
import { ProviderCosts } from "@/components/opencode/provider-costs";
import { ProviderSelector } from "@/components/opencode/provider-selector";
import { OpenCodeView } from "@/types/opencode";

interface ProvidersViewProps {
  onViewChange: (view: OpenCodeView) => void;
}

export function ProvidersView({ onViewChange }: ProvidersViewProps) {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [showRoutingDialog, setShowRoutingDialog] = useState(false);
  const [routingRules, setRoutingRules] = useState([
    {
      id: 1,
      name: "High-Priority Tasks",
      condition: "priority === 'high'",
      primaryProvider: "claude-3-opus",
      fallbackProvider: "gpt-4",
      enabled: true
    },
    {
      id: 2,
      name: "Code Generation",
      condition: "task_type === 'code'",
      primaryProvider: "claude-3-sonnet",
      fallbackProvider: "gpt-3.5-turbo",
      enabled: true
    },
    {
      id: 3,
      name: "Cost-Optimized",
      condition: "budget === 'low'",
      primaryProvider: "gpt-3.5-turbo",
      fallbackProvider: "ollama-llama2",
      enabled: false
    }
  ]);

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Enhanced Header */}
      <div className="border-b border-border bg-gradient-to-r from-background to-muted/20">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <Button 
                variant="ghost" 
                size="icon"
                onClick={() => onViewChange("welcome")}
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <div>
                <h1 className="text-3xl font-bold flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600">
                    <Globe className="h-6 w-6 text-white" />
                  </div>
                  <span>Multi-Provider Command Center</span>
                </h1>
                <p className="text-muted-foreground mt-1">
                  Comprehensive management for 75+ AI providers with intelligent routing and real-time monitoring
                </p>
              </div>
            </div>
            
            {/* Quick Provider Selector */}
            <div className="flex items-center space-x-4">
              <div className="max-w-sm">
                <ProviderSelector />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content with Tabs */}
      <div className="flex-1 overflow-hidden">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
          <div className="px-6 pt-4">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="dashboard" className="flex items-center space-x-2">
                <Globe className="h-4 w-4" />
                <span>Provider Dashboard</span>
              </TabsTrigger>
              <TabsTrigger value="health" className="flex items-center space-x-2">
                <Activity className="h-4 w-4" />
                <span>Health Monitor</span>
              </TabsTrigger>
              <TabsTrigger value="costs" className="flex items-center space-x-2">
                <DollarSign className="h-4 w-4" />
                <span>Cost Management</span>
              </TabsTrigger>
              <TabsTrigger value="routing" className="flex items-center space-x-2">
                <Target className="h-4 w-4" />
                <span>Smart Routing</span>
              </TabsTrigger>
            </TabsList>
          </div>

          <div className="flex-1 overflow-auto">
            <TabsContent value="dashboard" className="h-full m-0">
              <ProviderDashboard />
            </TabsContent>
            
            <TabsContent value="health" className="h-full m-0 p-6">
              <ProviderHealth />
            </TabsContent>
            
            <TabsContent value="costs" className="h-full m-0 p-6">
              <ProviderCosts />
            </TabsContent>
            
            <TabsContent value="routing" className="h-full m-0 p-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <div className="text-center py-12">
                  <Target className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-xl font-semibold mb-2">Intelligent Provider Routing</h3>
                  <p className="text-muted-foreground max-w-md mx-auto">
                    Configure advanced routing rules, failover strategies, and task-specific provider selection 
                    to optimize performance, cost, and reliability across your entire AI workflow.
                  </p>
                  <div className="mt-6 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
                      <div className="p-4 rounded-lg border bg-card">
                        <BarChart3 className="h-8 w-8 text-blue-500 mx-auto mb-2" />
                        <h4 className="font-medium">Performance-Based</h4>
                        <p className="text-sm text-muted-foreground">Route based on response times and uptime</p>
                      </div>
                      <div className="p-4 rounded-lg border bg-card">
                        <DollarSign className="h-8 w-8 text-green-500 mx-auto mb-2" />
                        <h4 className="font-medium">Cost-Optimized</h4>
                        <p className="text-sm text-muted-foreground">Minimize costs while maintaining quality</p>
                      </div>
                      <div className="p-4 rounded-lg border bg-card">
                        <Shield className="h-8 w-8 text-purple-500 mx-auto mb-2" />
                        <h4 className="font-medium">Failover Ready</h4>
                        <p className="text-sm text-muted-foreground">Automatic fallback when providers are down</p>
                      </div>
                    </div>
                    <Button 
                      className="mt-4"
                      onClick={() => setShowRoutingDialog(true)}
                    >
                      <Settings className="h-4 w-4 mr-2" />
                      Configure Routing Rules
                    </Button>
                  </div>
                </div>
              </motion.div>
            </TabsContent>
          </div>
        </Tabs>
      </div>

      {/* Routing Configuration Dialog */}
      <Dialog open={showRoutingDialog} onOpenChange={setShowRoutingDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <Target className="h-5 w-5" />
              <span>Configure Provider Routing Rules</span>
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-6 py-4">
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                Define intelligent routing rules to automatically select the best provider for different types of tasks.
              </p>
              <Button size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Add Rule
              </Button>
            </div>

            <div className="space-y-4">
              {routingRules.map((rule) => (
                <Card key={rule.id} className={rule.enabled ? "border-green-200 bg-green-50/30" : "border-gray-200"}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <Switch
                          checked={rule.enabled}
                          onCheckedChange={(checked) => {
                            setRoutingRules(prev => prev.map(r => 
                              r.id === rule.id ? { ...r, enabled: checked } : r
                            ));
                          }}
                        />
                        <CardTitle className="text-lg">{rule.name}</CardTitle>
                        <Badge variant={rule.enabled ? "default" : "secondary"}>
                          {rule.enabled ? "Active" : "Disabled"}
                        </Badge>
                      </div>
                      <Button variant="ghost" size="sm">
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="space-y-2">
                        <Label>Condition</Label>
                        <Input 
                          value={rule.condition}
                          onChange={(e) => {
                            setRoutingRules(prev => prev.map(r => 
                              r.id === rule.id ? { ...r, condition: e.target.value } : r
                            ));
                          }}
                          placeholder="e.g., task_type === 'code'"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Primary Provider</Label>
                        <Select 
                          value={rule.primaryProvider}
                          onValueChange={(value) => {
                            setRoutingRules(prev => prev.map(r => 
                              r.id === rule.id ? { ...r, primaryProvider: value } : r
                            ));
                          }}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                            <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                            <SelectItem value="gpt-4">GPT-4</SelectItem>
                            <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                            <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
                            <SelectItem value="ollama-llama2">Ollama Llama 2</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label>Fallback Provider</Label>
                        <Select 
                          value={rule.fallbackProvider}
                          onValueChange={(value) => {
                            setRoutingRules(prev => prev.map(r => 
                              r.id === rule.id ? { ...r, fallbackProvider: value } : r
                            ));
                          }}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                            <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                            <SelectItem value="gpt-4">GPT-4</SelectItem>
                            <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                            <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
                            <SelectItem value="ollama-llama2">Ollama Llama 2</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                      <ArrowUpDown className="h-4 w-4" />
                      <span>
                        When <code className="bg-muted px-1 rounded">{rule.condition}</code> is true, 
                        use <strong>{rule.primaryProvider}</strong> with <strong>{rule.fallbackProvider}</strong> as backup
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <div className="p-4 bg-blue-50 rounded border border-blue-200">
              <div className="flex items-start space-x-2">
                <Shield className="h-5 w-5 text-blue-600 mt-0.5" />
                <div>
                  <div className="font-medium text-blue-900">Routing Priority</div>
                  <p className="text-sm text-blue-700 mt-1">
                    Rules are evaluated in order from top to bottom. The first matching rule will be applied.
                    Use drag and drop to reorder rules by priority.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex justify-end space-x-3 pt-4 border-t">
              <Button variant="outline" onClick={() => setShowRoutingDialog(false)}>
                Cancel
              </Button>
              <Button onClick={() => {
                // Save routing rules (placeholder)
                console.log('Saving routing rules:', routingRules);
                setShowRoutingDialog(false);
              }}>
                Save Configuration
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}