"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Bot, Plus, ArrowLeft } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { OpenCodeView } from "@/types/opencode";

interface AgentsViewProps {
  onViewChange: (view: OpenCodeView) => void;
}

export function AgentsView({ onViewChange }: AgentsViewProps) {
  const { toast } = useToast();
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [agentForm, setAgentForm] = useState({
    name: '',
    description: '',
    role: '',
    capabilities: [] as string[],
    provider: 'claude-3-sonnet'
  });

  const handleCreateAgent = async () => {
    if (!agentForm.name.trim()) {
      toast({ title: "Error", description: "Agent name is required", variant: "destructive" });
      return;
    }

    setIsCreating(true);
    try {
      // Simulate API call to create agent
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      toast({ title: "Success", description: `Agent "${agentForm.name}" has been created successfully`, variant: "success" });
      
      // Reset form and close dialog
      setAgentForm({
        name: '',
        description: '',
        role: '',
        capabilities: [],
        provider: 'claude-3-sonnet'
      });
      setShowCreateDialog(false);
      
      // In a real implementation, this would refresh the agents list
    } catch (error) {
      toast({ title: "Error", description: "Failed to create agent. Please try again.", variant: "destructive" });
    } finally {
      setIsCreating(false);
    }
  };

  const openCreateDialog = () => {
    setShowCreateDialog(true);
  };

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button 
              variant="ghost" 
              size="icon"
              onClick={() => onViewChange("welcome")}
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-2xl font-bold">AI Agents</h1>
              <p className="text-sm text-muted-foreground">
                Create and manage custom AI agents
              </p>
            </div>
          </div>
          <Button onClick={openCreateDialog}>
            <Plus className="h-4 w-4 mr-2" />
            Create Agent
          </Button>
        </div>
      </div>

      <div className="p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center py-12"
        >
          <Bot className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-2">No agents configured</h3>
          <p className="text-muted-foreground mb-6">
            Create your first AI agent to get started with specialized coding assistance
          </p>
          <Button onClick={openCreateDialog}>
            <Plus className="h-4 w-4 mr-2" />
            Create Agent
          </Button>
        </motion.div>
      </div>

      {/* Create Agent Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Create New AI Agent</DialogTitle>
          </DialogHeader>
          <div className="space-y-6 py-4">
            <div className="space-y-2">
              <Label htmlFor="agent-name">Agent Name *</Label>
              <Input
                id="agent-name"
                placeholder="e.g., Code Reviewer, Test Generator"
                value={agentForm.name}
                onChange={(e) => setAgentForm(prev => ({ ...prev, name: e.target.value }))}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="agent-description">Description</Label>
              <Textarea
                id="agent-description"
                placeholder="Describe what this agent will do and its purpose"
                value={agentForm.description}
                onChange={(e) => setAgentForm(prev => ({ ...prev, description: e.target.value }))}
                rows={3}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="agent-role">Role/Specialty</Label>
              <Select value={agentForm.role} onValueChange={(value) => setAgentForm(prev => ({ ...prev, role: value }))}>
                <SelectTrigger>
                  <SelectValue placeholder="Select agent specialization" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="code-reviewer">Code Reviewer</SelectItem>
                  <SelectItem value="test-generator">Test Generator</SelectItem>
                  <SelectItem value="documentation">Documentation Writer</SelectItem>
                  <SelectItem value="refactoring">Code Refactoring</SelectItem>
                  <SelectItem value="debugging">Bug Fixer</SelectItem>
                  <SelectItem value="architecture">Architecture Advisor</SelectItem>
                  <SelectItem value="general">General Purpose</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="agent-provider">AI Provider</Label>
              <Select value={agentForm.provider} onValueChange={(value) => setAgentForm(prev => ({ ...prev, provider: value }))}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                  <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                  <SelectItem value="gpt-4">GPT-4</SelectItem>
                  <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                  <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
                  <SelectItem value="llama-2">Llama 2 (Local)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex justify-end space-x-3 pt-4">
              <Button 
                variant="outline" 
                onClick={() => setShowCreateDialog(false)}
                disabled={isCreating}
              >
                Cancel
              </Button>
              <Button 
                onClick={handleCreateAgent}
                disabled={isCreating}
              >
                {isCreating ? "Creating..." : "Create Agent"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}