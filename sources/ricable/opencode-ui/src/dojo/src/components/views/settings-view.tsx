"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { 
  Settings, 
  ArrowLeft, 
  Save, 
  User,
  Shield,
  Code,
  Palette,
  Terminal,
  Globe
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { useSessionStore } from "@/lib/session-store";
import { OpenCodeView } from "@/types/opencode";

interface SettingsViewProps {
  onViewChange: (view: OpenCodeView) => void;
}

export function SettingsView({ onViewChange }: SettingsViewProps) {
  const { config, actions } = useSessionStore();
  const [localConfig, setLocalConfig] = useState(config);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    actions.loadConfig();
  }, []);

  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  const handleConfigChange = (path: string, value: any) => {
    if (!localConfig) return;
    
    const newConfig = { ...localConfig };
    const keys = path.split('.');
    let current = newConfig;
    
    for (let i = 0; i < keys.length - 1; i++) {
      if (!(current as any)[keys[i]]) (current as any)[keys[i]] = {};
      current = (current as any)[keys[i]];
    }
    
    (current as any)[keys[keys.length - 1]] = value;
    setLocalConfig(newConfig);
    setHasChanges(true);
  };

  const handleSave = async () => {
    if (localConfig) {
      try {
        await actions.updateConfig(localConfig);
        setHasChanges(false);
      } catch (error) {
        console.error('Failed to save settings:', error);
      }
    }
  };

  if (!localConfig) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      {/* Header matching Claudia's design */}
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
              <h1 className="text-2xl font-bold">Settings</h1>
              <p className="text-sm text-muted-foreground">Configure OpenCode preferences</p>
            </div>
          </div>
          <Button 
            onClick={handleSave} 
            disabled={!hasChanges}
            className="flex items-center space-x-2"
          >
            <Save className="h-4 w-4" />
            <span>Save Settings</span>
          </Button>
        </div>
      </div>

      <div className="p-6">
        <Tabs defaultValue="general" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 max-w-lg">
            <TabsTrigger value="general" className="flex items-center space-x-1">
              <User className="h-3 w-3" />
              <span>General</span>
            </TabsTrigger>
            <TabsTrigger value="permissions" className="flex items-center space-x-1">
              <Shield className="h-3 w-3" />
              <span>Permissions</span>
            </TabsTrigger>
            <TabsTrigger value="environment" className="flex items-center space-x-1">
              <Terminal className="h-3 w-3" />
              <span>Environment</span>
            </TabsTrigger>
            <TabsTrigger value="advanced" className="flex items-center space-x-1">
              <Code className="h-3 w-3" />
              <span>Advanced</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>General Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="theme">Theme</Label>
                    <Select
                      value={localConfig.theme}
                      onValueChange={(value) => handleConfigChange('theme', value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="opencode">OpenCode</SelectItem>
                        <SelectItem value="dark">Dark</SelectItem>
                        <SelectItem value="light">Light</SelectItem>
                        <SelectItem value="system">System</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="model">Default Model</Label>
                    <Select
                      value={localConfig.model}
                      onValueChange={(value) => handleConfigChange('model', value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="anthropic/claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</SelectItem>
                        <SelectItem value="openai/gpt-4o">GPT-4o</SelectItem>
                        <SelectItem value="google/gemini-2.0-flash-exp">Gemini 2.0 Flash</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-base">Include &quot;Co-authored by Claude&quot;</Label>
                    <p className="text-sm text-muted-foreground">
                      Add Claude attribution to git commits and pull requests
                    </p>
                  </div>
                  <Switch
                    checked={localConfig.autoshare}
                    onCheckedChange={(checked) => handleConfigChange('autoshare', checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-base">Verbose Output</Label>
                    <p className="text-sm text-muted-foreground">
                      Show full bash and command outputs
                    </p>
                  </div>
                  <Switch
                    checked={localConfig.autoupdate}
                    onCheckedChange={(checked) => handleConfigChange('autoupdate', checked)}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="chat-retention">Chat Transcript Retention (days)</Label>
                  <Input
                    id="chat-retention"
                    type="number"
                    value="30"
                    className="w-32"
                  />
                  <p className="text-sm text-muted-foreground">
                    How long to retain chat transcripts locally (default: 30 days)
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="permissions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Security & Permissions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-muted-foreground">
                  <Shield className="h-12 w-12 mx-auto mb-2" />
                  <p>Permission settings coming soon</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="environment" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>OpenCode Installation</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Select OpenCode Installation</Label>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                        <div>
                          <div className="font-medium">System PATH</div>
                          <div className="text-sm text-muted-foreground">v1.0.35</div>
                          <div className="text-xs text-muted-foreground font-mono">
                            /Users/cedric/.nvm/versions/node/v22.17.0/bin/opencode
                          </div>
                        </div>
                      </div>
                      <Badge variant="secondary">Selected</Badge>
                    </div>

                    <div className="flex items-center justify-between p-3 border rounded-lg opacity-60">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
                        <div>
                          <div className="font-medium">PATH</div>
                          <div className="text-sm text-muted-foreground">v1.0.35</div>
                          <div className="text-xs text-muted-foreground font-mono">
                            opencode
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
                  <div className="flex items-center space-x-2 text-yellow-800 dark:text-yellow-200">
                    <div className="w-4 h-4 bg-yellow-500 rounded-full flex items-center justify-center">
                      <span className="text-white text-xs">!</span>
                    </div>
                    <span className="text-sm font-medium">
                      OpenCode binary path has been changed. Remember to save your settings.
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Advanced Configuration</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-muted-foreground">
                  <Code className="h-12 w-12 mx-auto mb-2" />
                  <p>Advanced settings coming soon</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}