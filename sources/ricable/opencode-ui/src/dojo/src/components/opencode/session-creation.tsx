/**
 * Session Creation - Enhanced session creation with templates and provider configuration
 * Replicates Claudia's new session UI with OpenCode multi-provider support
 */

"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus,
  FolderOpen,
  Settings,
  Zap,
  Code,
  Database,
  Globe,
  Terminal,
  FileText,
  Sparkles,
  Layers,
  ArrowRight,
  Check,
  AlertCircle,
  Info,
  Loader2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/lib/session-store';
import { SessionConfig, Provider } from '@/lib/opencode-client';
import { SessionTemplate } from '@/types/opencode';

interface SessionCreationProps {
  /**
   * Whether the dialog is open
   */
  open: boolean;
  /**
   * Callback when dialog should close
   */
  onOpenChange: (open: boolean) => void;
  /**
   * Optional initial project path
   */
  initialPath?: string;
  /**
   * Callback when session is created
   */
  onSessionCreated?: (sessionId: string) => void;
}

interface UISessionTemplate {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  category: 'development' | 'analysis' | 'learning' | 'custom';
  config: Partial<SessionConfig>;
  systemPrompt?: string;
  suggestedTools?: string[];
  suggestedProviders?: string[];
}

const sessionTemplates: UISessionTemplate[] = [
  {
    id: 'general-coding',
    name: 'General Coding',
    description: 'General purpose coding assistant for any programming task',
    icon: <Code className="h-6 w-6" />,
    category: 'development',
    config: {
      max_tokens: 8000,
      temperature: 0.7
    },
    systemPrompt: 'You are a helpful coding assistant. Help with code review, debugging, feature implementation, and general programming questions.',
    suggestedTools: ['file_read', 'file_write', 'bash', 'web_search'],
    suggestedProviders: ['anthropic', 'openai', 'ollama']
  },
  {
    id: 'code-review',
    name: 'Code Review',
    description: 'Focused on reviewing code quality, security, and best practices',
    icon: <FileText className="h-6 w-6" />,
    category: 'analysis',
    config: {
      max_tokens: 4000,
      temperature: 0.3
    },
    systemPrompt: 'You are a senior code reviewer. Focus on code quality, security vulnerabilities, performance issues, and adherence to best practices.',
    suggestedTools: ['file_read', 'grep', 'static_analysis'],
    suggestedProviders: ['anthropic', 'google', 'ollama']
  },
  {
    id: 'debug-session',
    name: 'Debug Session',
    description: 'Specialized for debugging and troubleshooting issues',
    icon: <Zap className="h-6 w-6" />,
    category: 'development',
    config: {
      max_tokens: 6000,
      temperature: 0.4
    },
    systemPrompt: 'You are a debugging expert. Help identify bugs, analyze error messages, suggest fixes, and guide through troubleshooting steps.',
    suggestedTools: ['bash', 'file_read', 'log_analysis', 'debugger'],
    suggestedProviders: ['anthropic', 'openai', 'ollama']
  },
  {
    id: 'architecture-design',
    name: 'Architecture Design',
    description: 'System design and architectural discussions',
    icon: <Layers className="h-6 w-6" />,
    category: 'analysis',
    config: {
      max_tokens: 8000,
      temperature: 0.6
    },
    systemPrompt: 'You are a software architect. Help with system design, technology choices, scalability considerations, and architectural patterns.',
    suggestedTools: ['diagram_generation', 'web_search', 'documentation'],
    suggestedProviders: ['anthropic', 'google']
  },
  {
    id: 'learning-session',
    name: 'Learning Session',
    description: 'Educational focused session for learning new concepts',
    icon: <Sparkles className="h-6 w-6" />,
    category: 'learning',
    config: {
      max_tokens: 6000,
      temperature: 0.8
    },
    systemPrompt: 'You are a patient teacher. Explain concepts clearly, provide examples, and help the user learn step by step.',
    suggestedTools: ['web_search', 'documentation', 'example_generator'],
    suggestedProviders: ['anthropic', 'openai', 'google', 'ollama']
  },
  {
    id: 'database-work',
    name: 'Database Work',
    description: 'Database design, queries, and optimization',
    icon: <Database className="h-6 w-6" />,
    category: 'development',
    config: {
      max_tokens: 5000,
      temperature: 0.5
    },
    systemPrompt: 'You are a database expert. Help with SQL queries, database design, optimization, and troubleshooting.',
    suggestedTools: ['sql_executor', 'schema_analyzer', 'query_optimizer'],
    suggestedProviders: ['anthropic', 'openai']
  },
  {
    id: 'local-development',
    name: 'Local Development',
    description: 'Privacy-focused local development with offline models',
    icon: <Database className="h-6 w-6" />,
    category: 'development',
    config: {
      max_tokens: 8000,
      temperature: 0.7
    },
    systemPrompt: 'You are a local coding assistant running on the user\'s machine. Help with development tasks while maintaining privacy and working offline.',
    suggestedTools: ['file_read', 'file_write', 'bash', 'grep'],
    suggestedProviders: ['ollama', 'llamacpp', 'lmstudio']
  },
  {
    id: 'experiments',
    name: 'Experiments & Prototyping',
    description: 'Quick experiments with local models for testing ideas',
    icon: <Zap className="h-6 w-6" />,
    category: 'development',
    config: {
      max_tokens: 4000,
      temperature: 0.9
    },
    systemPrompt: 'You are an experimental coding assistant. Help with rapid prototyping, testing ideas, and creative solutions.',
    suggestedTools: ['file_write', 'bash', 'web_search'],
    suggestedProviders: ['ollama', 'groq', 'lmstudio']
  }
];

interface ProviderCardProps {
  provider: Provider;
  isSelected: boolean;
  onSelect: () => void;
  disabled?: boolean;
}

const ProviderCard: React.FC<ProviderCardProps> = ({ provider, isSelected, onSelect, disabled }) => {
  const getProviderColor = (providerId: string) => {
    const colors = {
      'anthropic': 'border-orange-500 bg-orange-50 dark:bg-orange-950',
      'openai': 'border-green-500 bg-green-50 dark:bg-green-950', 
      'google': 'border-blue-500 bg-blue-50 dark:bg-blue-950',
      'groq': 'border-purple-500 bg-purple-50 dark:bg-purple-950',
      'ollama': 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950',
      'llamacpp': 'border-gray-500 bg-gray-50 dark:bg-gray-950',
      'lmstudio': 'border-indigo-500 bg-indigo-50 dark:bg-indigo-950',
      'localai': 'border-teal-500 bg-teal-50 dark:bg-teal-950',
      'textgen-webui': 'border-pink-500 bg-pink-50 dark:bg-pink-950',
    };
    return colors[providerId as keyof typeof colors] || 'border-gray-500 bg-gray-50 dark:bg-gray-950';
  };

  return (
    <Card 
      className={cn(
        "cursor-pointer transition-all hover:shadow-md",
        isSelected && "ring-2 ring-primary",
        disabled && "opacity-50 cursor-not-allowed",
        isSelected && getProviderColor(provider.id)
      )}
      onClick={disabled ? undefined : onSelect}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-medium">{provider.name}</h4>
          <div className="flex items-center space-x-2">
            <div className={cn(
              "h-2 w-2 rounded-full",
              provider.status === 'online' ? 'bg-green-500' : 'bg-red-500'
            )} />
            {provider.authenticated && (
              <Check className="h-4 w-4 text-green-500" />
            )}
          </div>
        </div>
        
        <p className="text-sm text-muted-foreground mb-3">
          {provider.description}
        </p>
        
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{provider.models.length} models</span>
          <span>${provider.cost_per_1k_tokens.toFixed(4)}/1K tokens</span>
          <span>{provider.avg_response_time}ms avg</span>
        </div>
      </CardContent>
    </Card>
  );
};

interface TemplateCardProps {
  template: UISessionTemplate;
  isSelected: boolean;
  onSelect: () => void;
}

const TemplateCard: React.FC<TemplateCardProps> = ({ template, isSelected, onSelect }) => {
  return (
    <Card 
      className={cn(
        "cursor-pointer transition-all hover:shadow-md",
        isSelected && "ring-2 ring-primary bg-primary/5"
      )}
      onClick={onSelect}
    >
      <CardContent className="p-4">
        <div className="flex items-start space-x-3">
          <div className={cn(
            "p-2 rounded-lg",
            isSelected ? "bg-primary text-primary-foreground" : "bg-muted"
          )}>
            {template.icon}
          </div>
          <div className="flex-1 min-w-0">
            <h4 className="font-medium mb-1">{template.name}</h4>
            <p className="text-sm text-muted-foreground line-clamp-2">
              {template.description}
            </p>
            <div className="flex items-center space-x-2 mt-2">
              <Badge variant="outline" className="text-xs">
                {template.category}
              </Badge>
              {template.suggestedProviders && (
                <span className="text-xs text-muted-foreground">
                  {template.suggestedProviders.length} providers
                </span>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export const SessionCreation: React.FC<SessionCreationProps> = ({
  open,
  onOpenChange,
  initialPath,
  onSessionCreated
}) => {
  const { providers, actions } = useSessionStore();
  
  // Form state
  const [currentTab, setCurrentTab] = useState('template');
  const [selectedTemplate, setSelectedTemplate] = useState<UISessionTemplate | null>(null);
  const [sessionName, setSessionName] = useState('');
  const [projectPath, setProjectPath] = useState(initialPath || '');
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [enabledTools, setEnabledTools] = useState<string[]>([]);
  const [maxTokens, setMaxTokens] = useState(8000);
  const [temperature, setTemperature] = useState(0.7);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState('');
  const [showDirectoryPicker, setShowDirectoryPicker] = useState(false);
  const [tempDirectoryPath, setTempDirectoryPath] = useState('');

  // Get available providers (authenticated only)
  const availableProviders = providers.filter(p => p.authenticated && p.status === 'online');

  const resetForm = useCallback(() => {
    setCurrentTab('template');
    setSelectedTemplate(null);
    setSessionName('');
    setProjectPath(initialPath || '');
    setSelectedProvider('');
    setSelectedModel('');
    setSystemPrompt('');
    setEnabledTools([]);
    setMaxTokens(8000);
    setTemperature(0.7);
    setIsCreating(false);
    setError('');
    setShowDirectoryPicker(false);
    setTempDirectoryPath('');
  }, [initialPath]);

  // Reset form when dialog opens/closes
  useEffect(() => {
    if (open) {
      resetForm();
    }
  }, [open, initialPath, resetForm]);

  // Update form when template is selected
  useEffect(() => {
    if (selectedTemplate) {
      setSystemPrompt(selectedTemplate.systemPrompt || '');
      setMaxTokens(selectedTemplate.config.max_tokens || 8000);
      setTemperature(selectedTemplate.config.temperature || 0.7);
      setEnabledTools(selectedTemplate.suggestedTools || []);
      
      // Auto-select first suggested provider if available
      if (selectedTemplate.suggestedProviders) {
        const suggestedProvider = availableProviders.find(p => 
          selectedTemplate.suggestedProviders!.includes(p.id)
        );
        if (suggestedProvider) {
          setSelectedProvider(suggestedProvider.id);
        }
      }
    }
  }, [selectedTemplate, availableProviders]);

  // Update available models when provider changes
  useEffect(() => {
    setSelectedModel('');
  }, [selectedProvider]);

  const handleSelectDirectory = () => {
    setTempDirectoryPath(projectPath);
    setShowDirectoryPicker(true);
  };

  const validateDirectoryPath = (path: string): string | null => {
    if (!path.trim()) {
      return 'Directory path cannot be empty';
    }
    
    // Basic path validation
    if (path.includes('//')) {
      return 'Invalid path format';
    }
    
    // Check for potentially dangerous paths
    const dangerousPaths = ['/', '/System', '/etc', '/usr/bin', '/bin'];
    if (dangerousPaths.some(dangerous => path.startsWith(dangerous))) {
      return 'Cannot use system directories';
    }
    
    return null;
  };

  const handleConfirmDirectory = () => {
    const error = validateDirectoryPath(tempDirectoryPath);
    if (error) {
      setError(error);
      return;
    }
    
    setProjectPath(tempDirectoryPath.trim());
    setShowDirectoryPicker(false);
    setTempDirectoryPath('');
    setError('');
  };

  const handleCancelDirectory = () => {
    setShowDirectoryPicker(false);
    setTempDirectoryPath('');
    setError('');
  };

  const getCurrentProvider = () => {
    return providers.find(p => p.id === selectedProvider);
  };

  const getAvailableModels = () => {
    const provider = getCurrentProvider();
    return provider?.models || [];
  };

  const validateForm = () => {
    if (!projectPath.trim()) {
      setError('Project directory is required');
      return false;
    }
    if (!selectedProvider) {
      setError('Please select a provider');
      return false;
    }
    if (!selectedModel) {
      setError('Please select a model');
      return false;
    }
    setError('');
    return true;
  };

  const handleCreateSession = async () => {
    if (!validateForm()) return;

    setIsCreating(true);
    try {
      const config: SessionConfig = {
        name: sessionName.trim() || undefined,
        project_path: projectPath.trim(),
        provider: selectedProvider,
        model: selectedModel,
        max_tokens: maxTokens,
        temperature: temperature,
        system_prompt: systemPrompt.trim() || undefined,
        tools_enabled: enabledTools.length > 0,
        enabled_tools: enabledTools
      };

      const session = await actions.createSession(config);
      onSessionCreated?.(session.id);
      onOpenChange(false);
      resetForm();
    } catch (error) {
      console.error('Failed to create session:', error);
      setError(error instanceof Error ? error.message : 'Failed to create session');
    } finally {
      setIsCreating(false);
    }
  };

  const canProceed = () => {
    switch (currentTab) {
      case 'template':
        return selectedTemplate !== null;
      case 'provider':
        return selectedProvider && selectedModel;
      case 'config':
        return projectPath.trim() !== '';
      default:
        return false;
    }
  };

  const nextStep = () => {
    if (currentTab === 'template') setCurrentTab('provider');
    else if (currentTab === 'provider') setCurrentTab('config');
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-4xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle>Create New Session</DialogTitle>
          <DialogDescription>
            Start a new AI coding session with your preferred configuration
          </DialogDescription>
        </DialogHeader>

        <Tabs value={currentTab} onValueChange={setCurrentTab} className="flex-1">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="template" className="flex items-center space-x-2">
              <Sparkles className="h-4 w-4" />
              <span>Template</span>
            </TabsTrigger>
            <TabsTrigger value="provider" disabled={!selectedTemplate}>
              <div className="flex items-center space-x-2">
                <Globe className="h-4 w-4" />
                <span>Provider</span>
              </div>
            </TabsTrigger>
            <TabsTrigger value="config" disabled={!selectedProvider || !selectedModel}>
              <div className="flex items-center space-x-2">
                <Settings className="h-4 w-4" />
                <span>Configure</span>
              </div>
            </TabsTrigger>
          </TabsList>

          <div className="mt-6 max-h-[60vh] overflow-auto">
            {/* Template Selection */}
            <TabsContent value="template" className="space-y-4">
              <div>
                <h3 className="text-lg font-medium mb-2">Choose a Template</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Select a pre-configured template that matches your workflow
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {sessionTemplates.map((template) => (
                  <TemplateCard
                    key={template.id}
                    template={template}
                    isSelected={selectedTemplate?.id === template.id}
                    onSelect={() => setSelectedTemplate(template)}
                  />
                ))}
              </div>
            </TabsContent>

            {/* Provider Selection */}
            <TabsContent value="provider" className="space-y-4">
              <div>
                <h3 className="text-lg font-medium mb-2">Select Provider & Model</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Choose your AI provider and model for this session
                </p>
              </div>

              {availableProviders.length === 0 ? (
                <Card className="p-6 text-center">
                  <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h4 className="font-medium mb-2">No Providers Available</h4>
                  <p className="text-sm text-muted-foreground mb-4">
                    Please authenticate at least one provider in settings
                  </p>
                  <Button onClick={() => actions.setCurrentView('settings')}>
                    <Settings className="h-4 w-4 mr-2" />
                    Go to Settings
                  </Button>
                </Card>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {availableProviders.map((provider) => (
                      <ProviderCard
                        key={provider.id}
                        provider={provider}
                        isSelected={selectedProvider === provider.id}
                        onSelect={() => setSelectedProvider(provider.id)}
                      />
                    ))}
                  </div>

                  {selectedProvider && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-3"
                    >
                      <Label>Model</Label>
                      <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                          {getAvailableModels().map((model) => (
                            <SelectItem key={model} value={model}>
                              {model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </motion.div>
                  )}
                </div>
              )}
            </TabsContent>

            {/* Configuration */}
            <TabsContent value="config" className="space-y-4">
              <div>
                <h3 className="text-lg font-medium mb-2">Session Configuration</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Configure your session settings and project details
                </p>
              </div>

              <div className="space-y-4">
                {/* Basic Settings */}
                <div className="space-y-3">
                  <div>
                    <Label htmlFor="session-name">Session Name (Optional)</Label>
                    <Input
                      id="session-name"
                      placeholder="e.g., Refactor authentication module"
                      value={sessionName}
                      onChange={(e) => setSessionName(e.target.value)}
                    />
                  </div>

                  <div>
                    <Label htmlFor="project-path">Project Directory *</Label>
                    <div className="flex space-x-2">
                      <Input
                        id="project-path"
                        placeholder="/path/to/your/project"
                        value={projectPath}
                        onChange={(e) => setProjectPath(e.target.value)}
                        className="flex-1"
                      />
                      <Button
                        variant="outline"
                        onClick={handleSelectDirectory}
                        className="px-3"
                      >
                        <FolderOpen className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="system-prompt">System Prompt</Label>
                    <Textarea
                      id="system-prompt"
                      placeholder="Optional custom instructions for the AI..."
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      rows={3}
                    />
                  </div>
                </div>

                {/* Advanced Settings */}
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="max-tokens">Max Tokens</Label>
                      <Input
                        id="max-tokens"
                        type="number"
                        min={1000}
                        max={32000}
                        value={maxTokens}
                        onChange={(e) => setMaxTokens(Number(e.target.value))}
                      />
                    </div>
                    <div>
                      <Label htmlFor="temperature">Temperature</Label>
                      <Input
                        id="temperature"
                        type="number"
                        min={0}
                        max={2}
                        step={0.1}
                        value={temperature}
                        onChange={(e) => setTemperature(Number(e.target.value))}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>
          </div>
        </Tabs>

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center space-x-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg"
          >
            <AlertCircle className="h-4 w-4 text-destructive" />
            <span className="text-sm text-destructive">{error}</span>
          </motion.div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-4 border-t">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isCreating}
          >
            Cancel
          </Button>

          <div className="flex space-x-2">
            {currentTab !== 'config' && (
              <Button
                onClick={nextStep}
                disabled={!canProceed()}
              >
                Next
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            )}
            
            {currentTab === 'config' && (
              <Button
                onClick={handleCreateSession}
                disabled={!validateForm() || isCreating}
              >
                {isCreating ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    Create Session
                  </>
                )}
              </Button>
            )}
          </div>
        </div>
      </DialogContent>

      {/* Directory Picker Dialog */}
      <Dialog open={showDirectoryPicker} onOpenChange={setShowDirectoryPicker}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Select Project Directory</DialogTitle>
            <DialogDescription>
              Enter the path to your project directory. This is where OpenCode will look for your project files.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label htmlFor="directory-path">Directory Path</Label>
              <Input
                id="directory-path"
                placeholder="/Users/username/projects/my-project"
                value={tempDirectoryPath}
                onChange={(e) => setTempDirectoryPath(e.target.value)}
                className="font-mono text-sm"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleConfirmDirectory();
                  } else if (e.key === 'Escape') {
                    e.preventDefault();
                    handleCancelDirectory();
                  }
                }}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Tip: Use absolute paths like /Users/username/projects/my-project
              </p>
            </div>

            {/* Common directory suggestions */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Quick Select:</Label>
              <div className="flex flex-wrap gap-2">
                {[
                  '/Users/' + (typeof window !== 'undefined' ? (process.env.USER || 'username') : 'username'),
                  '/Users/' + (typeof window !== 'undefined' ? (process.env.USER || 'username') : 'username') + '/Documents',
                  '/Users/' + (typeof window !== 'undefined' ? (process.env.USER || 'username') : 'username') + '/Desktop',
                  '/Users/' + (typeof window !== 'undefined' ? (process.env.USER || 'username') : 'username') + '/Projects'
                ].map((suggestedPath) => (
                  <Button
                    key={suggestedPath}
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs font-mono"
                    onClick={() => setTempDirectoryPath(suggestedPath)}
                  >
                    {suggestedPath.replace('/Users/' + (typeof window !== 'undefined' ? (process.env.USER || 'username') : 'username'), '~')}
                  </Button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex items-center justify-end space-x-2 pt-4">
            <Button
              variant="outline"
              onClick={handleCancelDirectory}
            >
              Cancel
            </Button>
            <Button
              onClick={handleConfirmDirectory}
              disabled={!tempDirectoryPath.trim()}
            >
              <Check className="h-4 w-4 mr-2" />
              Select Directory
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Dialog>
  );
};