"use client";

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Settings, 
  Save, 
  RefreshCw,
  Download,
  Upload,
  Copy,
  Check,
  AlertCircle,
  User,
  Server,
  Palette,
  Shield,
  Keyboard
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { cn } from '@/lib/utils';

export const SettingsPanel = () => {
  const [config, setConfig] = useState({
    theme: 'opencode',
    model: 'anthropic/claude-sonnet-4-20250514',
    autoshare: false,
    autoupdate: true,
    provider: {
      anthropic: { apiKey: '', disabled: false },
      openai: { apiKey: '', disabled: false }
    },
    agents: {
      primary: { model: 'claude-3.7-sonnet', maxTokens: 5000 },
      task: { model: 'claude-3.7-sonnet', maxTokens: 5000 },
      title: { model: 'claude-3.7-sonnet', maxTokens: 80 }
    }
  });

  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [isSaving, setIsSaving] = useState(false);

  const handleSaveConfig = async () => {
    setIsSaving(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      console.log('Config saved:', config);
    } catch (error) {
      console.error('Failed to save config:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleValidateConfig = () => {
    const errors = [];
    if (!config.model) errors.push('Model is required');
    if (!config.provider.anthropic.apiKey && !config.provider.openai.apiKey) {
      errors.push('At least one provider API key is required');
    }
    setValidationErrors(errors);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Settings</h1>
            <p className="text-muted-foreground">
              Configure OpenCode behavior and preferences
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              onClick={handleValidateConfig}
            >
              <AlertCircle className="h-4 w-4 mr-2" />
              Validate
            </Button>
            <Button
              onClick={handleSaveConfig}
              disabled={isSaving}
            >
              {isSaving ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Save className="h-4 w-4 mr-2" />
              )}
              {isSaving ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </div>

        {validationErrors.length > 0 && (
          <Alert className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              <ul className="list-disc list-inside">
                {validationErrors.map((error, index) => (
                  <li key={index}>{error}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 p-6">
        <Tabs defaultValue="general" className="w-full">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="general">
              <Settings className="h-4 w-4 mr-2" />
              General
            </TabsTrigger>
            <TabsTrigger value="providers">
              <Server className="h-4 w-4 mr-2" />
              Providers
            </TabsTrigger>
            <TabsTrigger value="agents">
              <User className="h-4 w-4 mr-2" />
              Agents
            </TabsTrigger>
            <TabsTrigger value="appearance">
              <Palette className="h-4 w-4 mr-2" />
              Appearance
            </TabsTrigger>
            <TabsTrigger value="security">
              <Shield className="h-4 w-4 mr-2" />
              Security
            </TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>General Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="default-model">Default Model</Label>
                  <Select 
                    value={config.model} 
                    onValueChange={(value) => setConfig({...config, model: value})}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="anthropic/claude-sonnet-4-20250514">Claude Sonnet 4</SelectItem>
                      <SelectItem value="openai/gpt-4">GPT-4</SelectItem>
                      <SelectItem value="google/gemini-2.5-flash">Gemini 2.5 Flash</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Auto-share Sessions</Label>
                    <div className="text-sm text-muted-foreground">
                      Automatically generate shareable links for sessions
                    </div>
                  </div>
                  <Switch 
                    checked={config.autoshare}
                    onCheckedChange={(checked) => setConfig({...config, autoshare: checked})}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Auto-update</Label>
                    <div className="text-sm text-muted-foreground">
                      Automatically update OpenCode when new versions are available
                    </div>
                  </div>
                  <Switch 
                    checked={config.autoupdate}
                    onCheckedChange={(checked) => setConfig({...config, autoupdate: checked})}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="providers" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Provider Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>Anthropic</Label>
                    <Badge variant={config.provider.anthropic.apiKey ? "default" : "secondary"}>
                      {config.provider.anthropic.apiKey ? "Connected" : "Not configured"}
                    </Badge>
                  </div>
                  <Input
                    type="password"
                    placeholder="Enter Anthropic API key"
                    value={config.provider.anthropic.apiKey}
                    onChange={(e) => setConfig({
                      ...config,
                      provider: {
                        ...config.provider,
                        anthropic: { ...config.provider.anthropic, apiKey: e.target.value }
                      }
                    })}
                  />
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>OpenAI</Label>
                    <Badge variant={config.provider.openai.apiKey ? "default" : "secondary"}>
                      {config.provider.openai.apiKey ? "Connected" : "Not configured"}
                    </Badge>
                  </div>
                  <Input
                    type="password"
                    placeholder="Enter OpenAI API key"
                    value={config.provider.openai.apiKey}
                    onChange={(e) => setConfig({
                      ...config,
                      provider: {
                        ...config.provider,
                        openai: { ...config.provider.openai, apiKey: e.target.value }
                      }
                    })}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="agents" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Agent Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <Label>Primary Agent</Label>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="primary-model">Model</Label>
                      <Select 
                        value={config.agents.primary.model}
                        onValueChange={(value) => setConfig({
                          ...config,
                          agents: {
                            ...config.agents,
                            primary: { ...config.agents.primary, model: value }
                          }
                        })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="claude-3.7-sonnet">Claude 3.7 Sonnet</SelectItem>
                          <SelectItem value="gpt-4">GPT-4</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="primary-tokens">Max Tokens</Label>
                      <Input
                        type="number"
                        value={config.agents.primary.maxTokens}
                        onChange={(e) => setConfig({
                          ...config,
                          agents: {
                            ...config.agents,
                            primary: { ...config.agents.primary, maxTokens: parseInt(e.target.value) }
                          }
                        })}
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="appearance" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Theme Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Theme</Label>
                  <Select 
                    value={config.theme}
                    onValueChange={(value) => setConfig({...config, theme: value})}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="opencode">OpenCode</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="light">Light</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="security" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Security Settings</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground">
                  Security settings will be available in future versions.
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};