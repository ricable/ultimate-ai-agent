/**
 * Enhanced Provider Selector - Intelligent routing and quick provider switching
 * Supports hotkey switching, cost estimation, and automatic provider selection
 */

"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronDown, 
  Zap, 
  DollarSign, 
  Clock, 
  Brain, 
  Cpu,
  Command,
  Star,
  CheckCircle,
  AlertTriangle,
  Activity,
  Settings,
  Shuffle,
  Target,
  TrendingUp
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/lib/session-store';

interface TaskContext {
  type: 'coding' | 'analysis' | 'creative' | 'chat' | 'vision' | 'reasoning';
  complexity: 'simple' | 'medium' | 'complex';
  tokenEstimate: number;
  priority: 'speed' | 'quality' | 'cost';
}

const PROVIDER_LOGOS = {
  'anthropic': '🧠',
  'openai': '🤖', 
  'google': '🔍',
  'groq': '⚡',
  'ollama': '🦙',
  'llamacpp': '🔧',
  'lmstudio': '🎬',
  'localai': '🏠',
  'textgen-webui': '🌐',
  'cohere': '🔗',
  'mistral': '🌪️',
  'together': '🤝',
  'fireworks': '🎆',
  'perplexity': '🔎',
  'deepseek': '🤿',
  'huggingface': '🤗',
  'replicate': '🔄',
  'xai': '❌',
  'ai21': '🌊'
};

const PROVIDER_SPECIALTIES = {
  'anthropic': ['reasoning', 'analysis', 'coding'],
  'openai': ['coding', 'creative', 'vision', 'reasoning'], 
  'google': ['analysis', 'vision', 'creative'],
  'groq': ['chat', 'coding'], // Fast inference
  'ollama': ['coding', 'chat', 'local'], // Local inference
  'llamacpp': ['coding', 'chat', 'local'], // Local C++ implementation
  'lmstudio': ['coding', 'chat', 'local'], // User-friendly local
  'localai': ['coding', 'chat', 'local'], // OpenAI compatible local
  'textgen-webui': ['coding', 'chat', 'local'], // Community favorite
  'cohere': ['search', 'embedding', 'analysis'],
  'mistral': ['coding', 'multilingual'],
  'together': ['coding', 'open-source'],
  'fireworks': ['coding', 'fast-inference'],
  'perplexity': ['search', 'reasoning'],
  'deepseek': ['coding', 'reasoning', 'math']
};

const HOTKEYS = {
  '1': 'anthropic',
  '2': 'openai', 
  '3': 'google',
  '4': 'groq'
};

export const ProviderSelector: React.FC = () => {
  const { 
    providers, 
    activeProvider, 
    providerHealth, 
    providerMetrics,
    actions 
  } = useSessionStore();
  
  const [isOpen, setIsOpen] = useState(false);
  const [smartRouting, setSmartRouting] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [taskContext, setTaskContext] = useState<TaskContext>({
    type: 'coding',
    complexity: 'medium',
    tokenEstimate: 1000,
    priority: 'quality'
  });
  const [costEstimate, setCostEstimate] = useState<number>(0);

  // Hotkey handling
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        const key = e.key;
        if (HOTKEYS[key as keyof typeof HOTKEYS]) {
          e.preventDefault();
          const providerId = HOTKEYS[key as keyof typeof HOTKEYS];
          const provider = providers.find(p => p.id === providerId);
          if (provider && provider.authenticated) {
            actions.setActiveProvider(providerId);
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [providers, actions]);

  // Cost estimation
  useEffect(() => {
    if (activeProvider) {
      const provider = providers.find(p => p.id === activeProvider);
      if (provider) {
        const tokensInK = taskContext.tokenEstimate / 1000;
        const estimate = provider.cost_per_1k_tokens * tokensInK;
        setCostEstimate(estimate);
      }
    }
  }, [activeProvider, taskContext, providers]);

  const getRecommendedProvider = (context: TaskContext) => {
    if (!smartRouting) return null;

    const authenticatedProviders = providers.filter(p => p.authenticated);
    const onlineProviders = authenticatedProviders.filter(p => {
      const health = providerHealth.find(h => h.provider_id === p.id);
      return health?.status === 'online';
    });

    if (onlineProviders.length === 0) return null;

    // Score providers based on context
    const scoredProviders = onlineProviders.map(provider => {
      let score = 0;
      const health = providerHealth.find(h => h.provider_id === provider.id);
      const metrics = providerMetrics.find(m => m.provider_id === provider.id);

      // Task type compatibility
      const specialties = PROVIDER_SPECIALTIES[provider.id as keyof typeof PROVIDER_SPECIALTIES] || [];
      if (specialties.includes(context.type)) score += 30;

      // Priority-based scoring
      switch (context.priority) {
        case 'speed':
          score += Math.max(0, 30 - (provider.avg_response_time / 100));
          break;
        case 'cost':
          score += Math.max(0, 30 - (provider.cost_per_1k_tokens * 1000));
          break;
        case 'quality':
          // Favor foundation models for quality
          if (['anthropic', 'openai'].includes(provider.id)) score += 25;
          break;
      }

      // Health and reliability
      if (health) {
        score += (health.uptime || 0) * 0.2;
        score += Math.max(0, 20 - (health.response_time / 100));
      }

      // Success rate
      if (metrics) {
        score += (1 - (metrics.error_rate || 0)) * 20;
      }

      return { provider, score };
    });

    // Return highest scoring provider
    const best = scoredProviders.sort((a, b) => b.score - a.score)[0];
    return best?.provider || null;
  };

  const getProviderStatus = (provider: any) => {
    const health = providerHealth.find(h => h.provider_id === provider.id);
    if (!provider.authenticated) return 'unauthenticated';
    if (!health) return 'unknown';
    return health.status;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'text-green-500';
      case 'degraded':
        return 'text-yellow-500';
      case 'offline':
      case 'error':
        return 'text-red-500';
      case 'unauthenticated':
        return 'text-gray-500';
      default:
        return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="h-3 w-3" />;
      case 'degraded':
        return <AlertTriangle className="h-3 w-3" />;
      case 'offline':
      case 'error':
        return <AlertTriangle className="h-3 w-3" />;
      case 'unauthenticated':
        return <Settings className="h-3 w-3" />;
      default:
        return <Activity className="h-3 w-3" />;
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 4,
      maximumFractionDigits: 4
    }).format(amount);
  };

  const currentProvider = providers.find(p => p.id === activeProvider);
  const recommendedProvider = getRecommendedProvider(taskContext);

  return (
    <TooltipProvider>
      <div className="relative">
        <Card className={cn(
          "transition-all duration-200 cursor-pointer hover:shadow-md",
          isOpen && "ring-2 ring-primary"
        )}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between" onClick={() => setIsOpen(!isOpen)}>
              <div className="flex items-center space-x-3">
                {currentProvider ? (
                  <>
                    <div className="relative">
                      <span className="text-2xl">
                        {PROVIDER_LOGOS[currentProvider.id as keyof typeof PROVIDER_LOGOS] || '🤖'}
                      </span>
                      <div className={cn(
                        "absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-white",
                        getStatusColor(getProviderStatus(currentProvider)).replace('text-', 'bg-')
                      )} />
                    </div>
                    <div>
                      <div className="font-semibold">{currentProvider.name}</div>
                      <div className="text-sm text-muted-foreground flex items-center space-x-2">
                        <span>Model: {selectedModel || currentProvider.models[0]}</span>
                        <Badge variant="outline" className="text-xs">
                          {formatCurrency(costEstimate)} est.
                        </Badge>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="flex items-center space-x-2">
                    <Cpu className="h-5 w-5 text-muted-foreground" />
                    <span className="text-muted-foreground">Select Provider</span>
                  </div>
                )}
              </div>
              
              <div className="flex items-center space-x-2">
                {smartRouting && recommendedProvider && recommendedProvider.id !== activeProvider && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        className="h-8 w-8 p-0 text-blue-500 hover:bg-blue-50 rounded-md border border-transparent hover:border-blue-200 transition-colors inline-flex items-center justify-center"
                        onClick={(e) => {
                          e.stopPropagation();
                          actions.setActiveProvider(recommendedProvider.id);
                        }}
                      >
                        <Brain className="h-4 w-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>AI recommends: {recommendedProvider.name}</p>
                    </TooltipContent>
                  </Tooltip>
                )}
                
                <ChevronDown className={cn(
                  "h-4 w-4 transition-transform",
                  isOpen && "rotate-180"
                )} />
              </div>
            </div>
            
            {/* Quick Stats */}
            {currentProvider && (
              <div className="mt-3 grid grid-cols-3 gap-4 text-xs">
                <div className="text-center">
                  <div className="font-medium">{currentProvider.avg_response_time}ms</div>
                  <div className="text-muted-foreground">Response</div>
                </div>
                <div className="text-center">
                  <div className="font-medium">{formatCurrency(currentProvider.cost_per_1k_tokens)}</div>
                  <div className="text-muted-foreground">Per 1K</div>
                </div>
                <div className="text-center">
                  <div className="font-medium">
                    {providerHealth.find(h => h.provider_id === currentProvider.id)?.uptime?.toFixed(1) || '0.0'}%
                  </div>
                  <div className="text-muted-foreground">Uptime</div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="absolute top-full left-0 right-0 z-50 mt-2"
            >
              <Card className="shadow-lg border-2">
                <CardContent className="p-4 space-y-4">
                  {/* Smart Routing Controls */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label className="flex items-center space-x-2">
                        <Brain className="h-4 w-4" />
                        <span>Intelligent Routing</span>
                      </Label>
                      <Switch
                        checked={smartRouting}
                        onCheckedChange={setSmartRouting}
                      />
                    </div>
                    
                    {smartRouting && (
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <Label>Task Type</Label>
                          <Select value={taskContext.type} onValueChange={(value: any) => 
                            setTaskContext(prev => ({ ...prev, type: value }))
                          }>
                            <SelectTrigger className="h-8">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="coding">Coding</SelectItem>
                              <SelectItem value="analysis">Analysis</SelectItem>
                              <SelectItem value="creative">Creative</SelectItem>
                              <SelectItem value="chat">Chat</SelectItem>
                              <SelectItem value="vision">Vision</SelectItem>
                              <SelectItem value="reasoning">Reasoning</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <div>
                          <Label>Priority</Label>
                          <Select value={taskContext.priority} onValueChange={(value: any) => 
                            setTaskContext(prev => ({ ...prev, priority: value }))
                          }>
                            <SelectTrigger className="h-8">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="speed">Speed</SelectItem>
                              <SelectItem value="quality">Quality</SelectItem>
                              <SelectItem value="cost">Cost</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Provider Grid */}
                  <div className="space-y-2">
                    <Label>Available Providers</Label>
                    <div className="grid grid-cols-1 gap-2">
                      {providers.map((provider) => {
                        const status = getProviderStatus(provider);
                        const health = providerHealth.find(h => h.provider_id === provider.id);
                        const isRecommended = recommendedProvider?.id === provider.id;
                        const hotkey = Object.keys(HOTKEYS).find(key => HOTKEYS[key as keyof typeof HOTKEYS] === provider.id);
                        
                        return (
                          <motion.div
                            key={provider.id}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                          >
                            <Card 
                              className={cn(
                                "cursor-pointer transition-all duration-200 hover:shadow-sm",
                                activeProvider === provider.id && "ring-2 ring-primary bg-primary/5",
                                isRecommended && "ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-950/20",
                                !provider.authenticated && "opacity-50"
                              )}
                              onClick={() => {
                                if (provider.authenticated) {
                                  actions.setActiveProvider(provider.id);
                                  setIsOpen(false);
                                }
                              }}
                            >
                              <CardContent className="p-3">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center space-x-3">
                                    <div className="relative">
                                      <span className="text-xl">
                                        {PROVIDER_LOGOS[provider.id as keyof typeof PROVIDER_LOGOS] || '🤖'}
                                      </span>
                                      <div className={cn(
                                        "absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full",
                                        getStatusColor(status).replace('text-', 'bg-')
                                      )} />
                                    </div>
                                    
                                    <div>
                                      <div className="flex items-center space-x-2">
                                        <span className="font-medium">{provider.name}</span>
                                        {isRecommended && (
                                          <Badge variant="secondary" className="text-xs bg-blue-100 text-blue-700">
                                            <Star className="h-3 w-3 mr-1" />
                                            Recommended
                                          </Badge>
                                        )}
                                        {hotkey && (
                                          <Badge variant="outline" className="text-xs">
                                            <Command className="h-2 w-2 mr-1" />
                                            {hotkey}
                                          </Badge>
                                        )}
                                      </div>
                                      <div className="text-xs text-muted-foreground">
                                        {provider.models.length} models • {status}
                                      </div>
                                    </div>
                                  </div>
                                  
                                  <div className="flex items-center space-x-3 text-xs">
                                    <div className="flex items-center space-x-1 text-muted-foreground">
                                      <Clock className="h-3 w-3" />
                                      <span>{provider.avg_response_time}ms</span>
                                    </div>
                                    <div className="flex items-center space-x-1 text-muted-foreground">
                                      <DollarSign className="h-3 w-3" />
                                      <span>{formatCurrency(provider.cost_per_1k_tokens)}</span>
                                    </div>
                                    {health && (
                                      <div className={cn("flex items-center space-x-1", getStatusColor(status))}>
                                        {getStatusIcon(status)}
                                        <span>{health.uptime?.toFixed(1)}%</span>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Model Selection */}
                  {currentProvider && (
                    <div className="space-y-2">
                      <Label>Model Selection</Label>
                      <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select model..." />
                        </SelectTrigger>
                        <SelectContent>
                          {currentProvider.models.map(model => (
                            <SelectItem key={model} value={model}>
                              <div className="flex items-center justify-between w-full">
                                <span>{model}</span>
                                <div className="flex items-center space-x-2 text-xs text-muted-foreground ml-4">
                                  <Clock className="h-3 w-3" />
                                  <span>{currentProvider.avg_response_time}ms</span>
                                </div>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  {/* Cost Estimation */}
                  <div className="space-y-2 pt-2 border-t border-border">
                    <div className="flex items-center justify-between text-sm">
                      <span>Estimated Cost</span>
                      <span className="font-mono">{formatCurrency(costEstimate)}</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Based on {taskContext.tokenEstimate} tokens
                    </div>
                  </div>

                  {/* Quick Actions */}
                  <div className="flex items-center space-x-2 pt-2 border-t border-border">
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-1"
                      onClick={() => {
                        // Random provider selection
                        const availableProviders = providers.filter(p => p.authenticated && getProviderStatus(p) === 'online');
                        if (availableProviders.length > 0) {
                          const randomProvider = availableProviders[Math.floor(Math.random() * availableProviders.length)];
                          actions.setActiveProvider(randomProvider.id);
                        }
                      }}
                    >
                      <Shuffle className="h-4 w-4 mr-2" />
                      Random
                    </Button>
                    
                    {recommendedProvider && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1"
                        onClick={() => {
                          actions.setActiveProvider(recommendedProvider.id);
                          setIsOpen(false);
                        }}
                      >
                        <Target className="h-4 w-4 mr-2" />
                        Use Recommended
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Overlay to close dropdown */}
        {isOpen && (
          <div 
            className="fixed inset-0 z-40" 
            onClick={() => setIsOpen(false)}
          />
        )}
      </div>
    </TooltipProvider>
  );
};