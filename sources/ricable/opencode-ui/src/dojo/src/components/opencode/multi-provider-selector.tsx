import React, { useEffect, useState } from "react";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2, Cpu, Check, Zap, Wifi, WifiOff } from "lucide-react";
import { cn } from "@/lib/utils";

// OpenCode provider configuration
interface Provider {
  id: string;
  name: string;
  type: 'remote' | 'local';
  models: string[];
  status: 'active' | 'inactive' | 'error';
  costPerToken?: number;
  description?: string;
  apiKey?: boolean;
  endpoint?: string;
}

interface MultiProviderSelectorProps {
  /**
   * Currently selected provider and model
   */
  selectedProvider?: string | null;
  selectedModel?: string | null;
  /**
   * Callback when a provider and model are selected
   */
  onSelect: (provider: Provider, model: string) => void;
  /**
   * Optional className for styling
   */
  className?: string;
  /**
   * Whether to show a save button (for settings page)
   */
  showSaveButton?: boolean;
  /**
   * Callback when save button is clicked
   */
  onSave?: () => void;
  /**
   * Whether the save operation is in progress
   */
  isSaving?: boolean;
}

/**
 * MultiProviderSelector component for selecting OpenCode AI providers and models
 * Adapted from Claudia's ClaudeVersionSelector for multi-provider support
 * 
 * @example
 * <MultiProviderSelector
 *   selectedProvider="anthropic"
 *   selectedModel="claude-3.5-sonnet-20241022"
 *   onSelect={(provider, model) => setSelection(provider, model)}
 * />
 */
export const MultiProviderSelector: React.FC<MultiProviderSelectorProps> = ({
  selectedProvider,
  selectedModel,
  onSelect,
  className,
  showSaveButton = false,
  onSave,
  isSaving = false,
}) => {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedProviderObj, setSelectedProviderObj] = useState<Provider | null>(null);
  const [currentModel, setCurrentModel] = useState<string | null>(null);

  // Mock providers - in real implementation, this would come from OpenCode API
  const mockProviders: Provider[] = [
    {
      id: 'anthropic',
      name: 'Anthropic',
      type: 'remote',
      models: ['claude-3.5-sonnet-20241022', 'claude-3.5-haiku-20241022', 'claude-3-opus-20240229'],
      status: 'active',
      costPerToken: 0.00001,
      description: 'Advanced reasoning and coding capabilities',
      apiKey: true
    },
    {
      id: 'openai',
      name: 'OpenAI',
      type: 'remote',
      models: ['gpt-4o-2024-08-06', 'gpt-4o-mini', 'o1-preview', 'o1-mini'],
      status: 'active',
      costPerToken: 0.000015,
      description: 'GPT-4 family with latest improvements',
      apiKey: true
    },
    {
      id: 'groq',
      name: 'Groq',
      type: 'remote',
      models: ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
      status: 'active',
      costPerToken: 0.0000008,
      description: 'Ultra-fast inference with competitive models',
      apiKey: true
    },
    {
      id: 'google',
      name: 'Google',
      type: 'remote',
      models: ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'],
      status: 'active',
      costPerToken: 0.000002,
      description: 'Latest Gemini models with multimodal capabilities',
      apiKey: true
    },
    {
      id: 'ollama',
      name: 'Ollama',
      type: 'local',
      models: ['llama3.2:3b', 'llama3.2:1b', 'qwen2.5-coder:7b', 'deepseek-coder-v2:16b'],
      status: 'active',
      costPerToken: 0,
      description: 'Local models with zero cost and privacy',
      endpoint: 'http://localhost:11434'
    },
    {
      id: 'lm-studio',
      name: 'LM Studio',
      type: 'local',
      models: ['custom-model', 'llama-3.1-8b', 'codellama-7b'],
      status: 'inactive',
      costPerToken: 0,
      description: 'Local model management and inference',
      endpoint: 'http://localhost:1234'
    },
    {
      id: 'llama-cpp',
      name: 'Llama.cpp',
      type: 'local',
      models: ['llama-3.1-8b.gguf', 'codellama-13b.gguf'],
      status: 'inactive',
      costPerToken: 0,
      description: 'High-performance CPU/GPU inference',
      endpoint: 'http://localhost:8080'
    }
  ];

  useEffect(() => {
    loadProviders();
  }, []);

  useEffect(() => {
    // Update selected provider when selectedProvider changes
    if (selectedProvider && providers.length > 0) {
      const found = providers.find(p => p.id === selectedProvider);
      if (found) {
        setSelectedProviderObj(found);
        setCurrentModel(selectedModel || null);
      }
    }
  }, [selectedProvider, selectedModel, providers]);

  const loadProviders = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // In real implementation, this would fetch from OpenCode API
      setProviders(mockProviders);
      
      // Auto-select first active provider if none selected
      if (!selectedProvider) {
        const firstActive = mockProviders.find(p => p.status === 'active');
        if (firstActive && firstActive.models.length > 0) {
          setSelectedProviderObj(firstActive);
          setCurrentModel(firstActive.models[0]);
          onSelect(firstActive, firstActive.models[0]);
        }
      }
    } catch (err) {
      console.error("Failed to load providers:", err);
      setError(err instanceof Error ? err.message : "Failed to load providers");
    } finally {
      setLoading(false);
    }
  };

  const handleProviderSelect = (provider: Provider) => {
    setSelectedProviderObj(provider);
    const defaultModel = provider.models[0];
    setCurrentModel(defaultModel);
    onSelect(provider, defaultModel);
  };

  const handleModelSelect = (model: string) => {
    if (selectedProviderObj) {
      setCurrentModel(model);
      onSelect(selectedProviderObj, model);
    }
  };

  const getProviderIcon = (provider: Provider) => {
    if (provider.type === 'local') {
      return provider.status === 'active' ? (
        <Cpu className="w-4 h-4 text-green-500" />
      ) : (
        <Cpu className="w-4 h-4 text-muted-foreground" />
      );
    }
    return provider.status === 'active' ? (
      <Wifi className="w-4 h-4 text-blue-500" />
    ) : (
      <WifiOff className="w-4 h-4 text-muted-foreground" />
    );
  };

  const getStatusBadge = (provider: Provider) => {
    switch (provider.status) {
      case 'active':
        return <Badge variant="default" className="text-xs bg-green-100 text-green-700">Active</Badge>;
      case 'inactive':
        return <Badge variant="secondary" className="text-xs">Inactive</Badge>;
      case 'error':
        return <Badge variant="destructive" className="text-xs">Error</Badge>;
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className={cn("flex items-center justify-center py-8", className)}>
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <Card className={cn("p-4", className)}>
        <div className="text-sm text-destructive">{error}</div>
      </Card>
    );
  }

  if (providers.length === 0) {
    return (
      <Card className={cn("p-4", className)}>
        <div className="text-sm text-muted-foreground">
          No AI providers configured. Please check your OpenCode configuration.
        </div>
      </Card>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Provider Selection */}
      <div>
        <Label className="text-sm font-medium mb-3 block">
          Select AI Provider
        </Label>
        <RadioGroup
          value={selectedProviderObj?.id}
          onValueChange={(value: string) => {
            const provider = providers.find(p => p.id === value);
            if (provider) {
              handleProviderSelect(provider);
            }
          }}
        >
          <div className="grid gap-3 md:grid-cols-2">
            {providers.map((provider) => (
              <Card
                key={provider.id}
                className={cn(
                  "relative cursor-pointer transition-colors",
                  selectedProviderObj?.id === provider.id
                    ? "border-primary ring-1 ring-primary/20"
                    : "hover:border-muted-foreground/50",
                  provider.status !== 'active' && "opacity-60"
                )}
                onClick={() => provider.status === 'active' && handleProviderSelect(provider)}
              >
                <div className="flex items-start p-4">
                  <RadioGroupItem
                    value={provider.id}
                    id={provider.id}
                    className="mt-1"
                    disabled={provider.status !== 'active'}
                  />
                  <div className="ml-3 flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      {getProviderIcon(provider)}
                      <span className="font-medium text-sm">
                        {provider.name}
                      </span>
                      {getStatusBadge(provider)}
                      {provider.type === 'local' && (
                        <Badge variant="outline" className="text-xs">
                          <Zap className="w-3 h-3 mr-1" />
                          Local
                        </Badge>
                      )}
                      {selectedProvider === provider.id && (
                        <Badge variant="default" className="text-xs">
                          <Check className="w-3 h-3 mr-1" />
                          Selected
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mb-2">
                      {provider.description}
                    </p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span>{provider.models.length} models</span>
                      {provider.costPerToken !== undefined && (
                        <span>
                          {provider.costPerToken === 0 ? 'Free' : `$${provider.costPerToken.toFixed(6)}/token`}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </RadioGroup>
      </div>

      {/* Model Selection */}
      {selectedProviderObj && selectedProviderObj.models.length > 0 && (
        <div>
          <Label className="text-sm font-medium mb-3 block">
            Select Model for {selectedProviderObj.name}
          </Label>
          <Select value={currentModel || undefined} onValueChange={handleModelSelect}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Choose a model..." />
            </SelectTrigger>
            <SelectContent>
              {selectedProviderObj.models.map((model) => (
                <SelectItem key={model} value={model}>
                  <div className="flex items-center gap-2">
                    <span>{model}</span>
                    {selectedProviderObj.type === 'local' && (
                      <Badge variant="outline" className="text-xs">Local</Badge>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Configuration Summary */}
      {selectedProviderObj && currentModel && (
        <Card className="p-4 bg-muted/30">
          <div className="text-sm space-y-2">
            <div className="flex items-center gap-2">
              <span className="font-medium">Selected Configuration:</span>
              <Badge variant="default" className="text-xs">
                {selectedProviderObj.name} / {currentModel}
              </Badge>
            </div>
            <div className="grid grid-cols-2 gap-4 text-xs text-muted-foreground">
              <div>Type: {selectedProviderObj.type}</div>
              <div>
                Cost: {selectedProviderObj.costPerToken === 0 ? 'Free' : `$${selectedProviderObj.costPerToken?.toFixed(6)}/token`}
              </div>
              {selectedProviderObj.endpoint && (
                <div className="col-span-2">Endpoint: {selectedProviderObj.endpoint}</div>
              )}
            </div>
          </div>
        </Card>
      )}

      {showSaveButton && onSave && (
        <div className="flex justify-end">
          <Button
            onClick={onSave}
            disabled={!selectedProviderObj || !currentModel || isSaving}
            size="sm"
          >
            {isSaving ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              "Save Selection"
            )}
          </Button>
        </div>
      )}
    </div>
  );
};