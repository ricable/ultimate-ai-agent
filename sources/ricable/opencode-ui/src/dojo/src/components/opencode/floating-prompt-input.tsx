import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Maximize2,
  Minimize2,
  ChevronUp,
  Sparkles,
  Zap,
  Square,
  Brain,
  Bot,
  Cpu,
  Orbit
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Popover } from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { ImagePreview } from "./image-preview";
import { FilePicker } from "./file-picker";

interface FloatingPromptInputProps {
  /**
   * Callback when prompt is sent
   */
  onSend: (prompt: string, provider: string, model: string, thinkingMode?: ThinkingMode) => void;
  /**
   * Whether the input is loading
   */
  isLoading?: boolean;
  /**
   * Whether the input is disabled
   */
  disabled?: boolean;
  /**
   * Available providers
   */
  providers: Provider[];
  /**
   * Default provider to select
   */
  defaultProvider?: string;
  /**
   * Default model to select
   */
  defaultModel?: string;
  /**
   * Project path for file picker
   */
  projectPath?: string;
  /**
   * Optional className for styling
   */
  className?: string;
  /**
   * Callback when cancel is clicked (only during loading)
   */
  onCancel?: () => void;
  /**
   * Real-time cost estimation
   */
  estimatedCost?: number;
  /**
   * Token count for current prompt
   */
  tokenCount?: number;
}

export interface FloatingPromptInputRef {
  addImage: (imagePath: string) => void;
  focus: () => void;
  clear: () => void;
}

/**
 * Provider information
 */
interface Provider {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  models: Model[];
  cost_per_token?: number;
  max_tokens?: number;
  is_local?: boolean;
  status: "available" | "authenticated" | "error" | "disabled";
}

/**
 * Model information
 */
interface Model {
  id: string;
  name: string;
  description: string;
  max_tokens: number;
  cost_per_input_token?: number;
  cost_per_output_token?: number;
  capabilities: string[];
}

/**
 * Thinking mode type definition
 */
type ThinkingMode = "auto" | "think" | "think_hard" | "think_harder" | "ultrathink";

/**
 * Thinking mode configuration
 */
type ThinkingModeConfig = {
  id: ThinkingMode;
  name: string;
  description: string;
  level: number; // 0-4 for visual indicator
  phrase?: string; // The phrase to append
};

const THINKING_MODES: ThinkingModeConfig[] = [
  {
    id: "auto",
    name: "Auto",
    description: "Let the model decide",
    level: 0
  },
  {
    id: "think",
    name: "Think",
    description: "Basic reasoning",
    level: 1,
    phrase: "think"
  },
  {
    id: "think_hard",
    name: "Think Hard",
    description: "Deeper analysis",
    level: 2,
    phrase: "think hard"
  },
  {
    id: "think_harder",
    name: "Think Harder",
    description: "Extensive reasoning",
    level: 3,
    phrase: "think harder"
  },
  {
    id: "ultrathink",
    name: "Ultrathink",
    description: "Maximum computation",
    level: 4,
    phrase: "ultrathink"
  }
];

/**
 * Provider icons mapping
 */
const getProviderIcon = (providerId: string): React.ReactNode => {
  switch (providerId.toLowerCase()) {
    case "anthropic":
      return <Brain className="h-4 w-4" />;
    case "openai":
      return <Sparkles className="h-4 w-4" />;
    case "google":
      return <Orbit className="h-4 w-4" />;
    case "groq":
      return <Zap className="h-4 w-4" />;
    case "ollama":
    case "local":
      return <Cpu className="h-4 w-4" />;
    default:
      return <Bot className="h-4 w-4" />;
  }
};

/**
 * ThinkingModeIndicator component - Shows visual indicator bars for thinking level
 */
const ThinkingModeIndicator: React.FC<{ level: number }> = ({ level }) => {
  return (
    <div className="flex items-center gap-0.5">
      {[1, 2, 3, 4].map((i) => (
        <div
          key={i}
          className={cn(
            "w-1 h-3 rounded-full transition-colors",
            i <= level ? "bg-blue-500" : "bg-muted"
          )}
        />
      ))}
    </div>
  );
};

/**
 * FileEntry interface for file picker integration
 */
interface FileEntry {
  name: string;
  path: string;
  is_directory: boolean;
  size: number;
  extension?: string;
}

/**
 * FloatingPromptInput component - Multi-provider prompt input with model picker
 * 
 * @example
 * const promptRef = useRef<FloatingPromptInputRef>(null);
 * <FloatingPromptInput
 *   ref={promptRef}
 *   onSend={(prompt, provider, model) => console.log('Send:', prompt, provider, model)}
 *   providers={availableProviders}
 *   isLoading={false}
 * />
 */
const FloatingPromptInputInner = (
  {
    onSend,
    isLoading = false,
    disabled = false,
    providers = [],
    defaultProvider,
    defaultModel,
    projectPath,
    className,
    onCancel,
    estimatedCost = 0,
    tokenCount = 0,
  }: FloatingPromptInputProps,
  ref: React.Ref<FloatingPromptInputRef>,
) => {
  const [prompt, setPrompt] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<string>(
    defaultProvider || providers[0]?.id || ""
  );
  const [selectedModel, setSelectedModel] = useState<string>(
    defaultModel || providers[0]?.models[0]?.id || ""
  );
  const [selectedThinkingMode, setSelectedThinkingMode] = useState<ThinkingMode>("auto");
  const [isExpanded, setIsExpanded] = useState(false);
  const [providerPickerOpen, setProviderPickerOpen] = useState(false);
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [thinkingModePickerOpen, setThinkingModePickerOpen] = useState(false);
  const [showFilePicker, setShowFilePicker] = useState(false);
  const [filePickerQuery, setFilePickerQuery] = useState("");
  const [cursorPosition, setCursorPosition] = useState(0);
  const [embeddedImages, setEmbeddedImages] = useState<string[]>([]);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const expandedTextareaRef = useRef<HTMLTextAreaElement>(null);

  // Find current provider and model
  const currentProvider = providers.find(p => p.id === selectedProvider);
  const currentModel = currentProvider?.models.find(m => m.id === selectedModel);

  // Update model when provider changes
  useEffect(() => {
    if (currentProvider && !currentProvider.models.find(m => m.id === selectedModel)) {
      setSelectedModel(currentProvider.models[0]?.id || "");
    }
  }, [selectedProvider, currentProvider, selectedModel]);

  // Expose imperative methods
  React.useImperativeHandle(
    ref,
    () => ({
      addImage: (imagePath: string) => {
        setPrompt(currentPrompt => {
          const existingPaths = extractImagePaths(currentPrompt);
          if (existingPaths.includes(imagePath)) {
            return currentPrompt; // Image already added
          }

          const mention = `@${imagePath}`;
          const newPrompt = currentPrompt + (currentPrompt.endsWith(' ') || currentPrompt === '' ? '' : ' ') + mention + ' ';

          // Focus the textarea
          setTimeout(() => {
            const target = isExpanded ? expandedTextareaRef.current : textareaRef.current;
            target?.focus();
            target?.setSelectionRange(newPrompt.length, newPrompt.length);
          }, 0);

          return newPrompt;
        });
      },
      focus: () => {
        const target = isExpanded ? expandedTextareaRef.current : textareaRef.current;
        target?.focus();
      },
      clear: () => {
        setPrompt("");
        setEmbeddedImages([]);
      }
    }),
    [isExpanded]
  );

  // Helper function to check if a file is an image
  const isImageFile = (path: string): boolean => {
    const ext = path.split('.').pop()?.toLowerCase();
    return ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'ico', 'bmp'].includes(ext || '');
  };

  // Extract image paths from prompt text
  const extractImagePaths = (text: string): string[] => {
    const regex = /@([^\s]+)/g;
    const matches = Array.from(text.matchAll(regex));
    const pathsSet = new Set<string>();

    for (const match of matches) {
      const path = match[1];
      const fullPath = path.startsWith('/') ? path : (projectPath ? `${projectPath}/${path}` : path);
      if (isImageFile(fullPath)) {
        pathsSet.add(fullPath);
      }
    }

    return Array.from(pathsSet);
  };

  // Update embedded images when prompt changes
  useEffect(() => {
    const imagePaths = extractImagePaths(prompt);
    setEmbeddedImages(imagePaths);
  }, [prompt, projectPath]);

  useEffect(() => {
    // Focus the appropriate textarea when expanded state changes
    if (isExpanded && expandedTextareaRef.current) {
      expandedTextareaRef.current.focus();
    } else if (!isExpanded && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isExpanded]);

  const handleSend = () => {
    if (prompt.trim() && !isLoading && !disabled && currentProvider && currentModel) {
      let finalPrompt = prompt.trim();
      
      // Append thinking phrase if not auto mode
      const thinkingMode = THINKING_MODES.find(m => m.id === selectedThinkingMode);
      if (thinkingMode && thinkingMode.phrase) {
        finalPrompt = `${finalPrompt}\n\n${thinkingMode.phrase}.`;
      }
      
      onSend(finalPrompt, selectedProvider, selectedModel, selectedThinkingMode);
      setPrompt("");
      setEmbeddedImages([]);
    }
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const newCursorPosition = e.target.selectionStart || 0;

    // Check if @ was just typed
    if (projectPath?.trim() && newValue.length > prompt.length && newValue[newCursorPosition - 1] === '@') {
      setShowFilePicker(true);
      setFilePickerQuery("");
      setCursorPosition(newCursorPosition);
    }

    // Check if we're typing after @ (for search query)
    if (showFilePicker && newCursorPosition >= cursorPosition) {
      // Find the @ position before cursor
      let atPosition = -1;
      for (let i = newCursorPosition - 1; i >= 0; i--) {
        if (newValue[i] === '@') {
          atPosition = i;
          break;
        }
        // Stop if we hit whitespace (new word)
        if (newValue[i] === ' ' || newValue[i] === '\n') {
          break;
        }
      }

      if (atPosition !== -1) {
        const query = newValue.substring(atPosition + 1, newCursorPosition);
        setFilePickerQuery(query);
      } else {
        // @ was removed or cursor moved away
        setShowFilePicker(false);
        setFilePickerQuery("");
      }
    }

    setPrompt(newValue);
    setCursorPosition(newCursorPosition);
  };

  const handleFileSelect = (entry: FileEntry) => {
    const textarea = isExpanded ? expandedTextareaRef.current : textareaRef.current;
    if (textarea) {
      // Replace the @ and partial query with the selected path
      const beforeAt = prompt.substring(0, cursorPosition - 1);
      const afterCursor = prompt.substring(cursorPosition + filePickerQuery.length);
      const relativePath = entry.path.startsWith(projectPath || '')
        ? entry.path.slice((projectPath || '').length + 1)
        : entry.path;

      const newPrompt = `${beforeAt}@${relativePath} ${afterCursor}`;
      setPrompt(newPrompt);
      setShowFilePicker(false);
      setFilePickerQuery("");

      // Focus back on textarea and set cursor position after the inserted path
      setTimeout(() => {
        textarea.focus();
        const newCursorPos = beforeAt.length + relativePath.length + 2; // +2 for @ and space
        textarea.setSelectionRange(newCursorPos, newCursorPos);
      }, 0);
    }
  };

  const handleFilePickerClose = () => {
    setShowFilePicker(false);
    setFilePickerQuery("");
    // Return focus to textarea
    setTimeout(() => {
      const target = isExpanded ? expandedTextareaRef.current : textareaRef.current;
      target?.focus();
    }, 0);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showFilePicker && e.key === 'Escape') {
      e.preventDefault();
      setShowFilePicker(false);
      setFilePickerQuery("");
      return;
    }

    if (e.key === "Enter" && !e.shiftKey && !isExpanded && !showFilePicker) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleRemoveImage = (index: number) => {
    // Remove the corresponding @mention from the prompt
    const imagePath = embeddedImages[index];
    const patterns = [
      new RegExp(`@${imagePath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s?`, 'g'),
      new RegExp(`@${imagePath.replace(projectPath + '/', '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s?`, 'g')
    ];

    let newPrompt = prompt;
    for (const pattern of patterns) {
      newPrompt = newPrompt.replace(pattern, '');
    }

    setPrompt(newPrompt.trim());
  };

  const formatCost = (cost: number): string => {
    if (cost === 0) return "Free";
    if (cost < 0.01) return `<$0.01`;
    return `$${cost.toFixed(3)}`;
  };

  const canSend = prompt.trim() && !isLoading && !disabled && currentProvider && currentModel;

  return (
    <>
      {/* Expanded Modal */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
            onClick={() => setIsExpanded(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-background border border-border rounded-lg shadow-lg w-full max-w-2xl p-4 space-y-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium">Compose your prompt</h3>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsExpanded(false)}
                  className="h-8 w-8"
                >
                  <Minimize2 className="h-4 w-4" />
                </Button>
              </div>

              {/* Image previews in expanded mode */}
              {embeddedImages.length > 0 && (
                <ImagePreview
                  images={embeddedImages}
                  onRemove={handleRemoveImage}
                  className="border-t border-border pt-2"
                />
              )}

              <Textarea
                ref={expandedTextareaRef}
                value={prompt}
                onChange={handleTextChange}
                placeholder="Type your prompt here..."
                className="min-h-[200px] resize-none"
                disabled={isLoading || disabled}
              />

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  {/* Provider Selector */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Provider:</span>
                    <Popover
                      trigger={
                        <Button variant="outline" size="sm" className="gap-2">
                          {currentProvider ? getProviderIcon(currentProvider.id) : <Bot className="h-4 w-4" />}
                          {currentProvider?.name || "Select Provider"}
                        </Button>
                      }
                      content={
                        <div className="w-[280px] p-1">
                          {providers.map((provider) => (
                            <button
                              key={provider.id}
                              onClick={() => {
                                setSelectedProvider(provider.id);
                                setProviderPickerOpen(false);
                              }}
                              className={cn(
                                "w-full flex items-start gap-3 p-3 rounded-md transition-colors text-left",
                                "hover:bg-accent",
                                selectedProvider === provider.id && "bg-accent"
                              )}
                            >
                              <div className="mt-0.5">{getProviderIcon(provider.id)}</div>
                              <div className="flex-1 space-y-1">
                                <div className="flex items-center gap-2">
                                  <span className="font-medium text-sm">{provider.name}</span>
                                  {provider.is_local && (
                                    <Badge variant="secondary" className="text-xs">Local</Badge>
                                  )}
                                  <Badge 
                                    variant={provider.status === "authenticated" ? "default" : "outline"}
                                    className="text-xs"
                                  >
                                    {provider.status}
                                  </Badge>
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  {provider.description}
                                </div>
                              </div>
                            </button>
                          ))}
                        </div>
                      }
                      open={providerPickerOpen}
                      onOpenChange={setProviderPickerOpen}
                      align="start"
                      side="top"
                    />
                  </div>

                  {/* Model Selector */}
                  {currentProvider && currentProvider.models.length > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">Model:</span>
                      <Popover
                        trigger={
                          <Button variant="outline" size="sm" className="gap-2">
                            {currentModel?.name || "Select Model"}
                          </Button>
                        }
                        content={
                          <div className="w-[300px] p-1">
                            {currentProvider.models.map((model) => (
                              <button
                                key={model.id}
                                onClick={() => {
                                  setSelectedModel(model.id);
                                  setModelPickerOpen(false);
                                }}
                                className={cn(
                                  "w-full flex items-start gap-3 p-3 rounded-md transition-colors text-left",
                                  "hover:bg-accent",
                                  selectedModel === model.id && "bg-accent"
                                )}
                              >
                                <div className="flex-1 space-y-1">
                                  <div className="font-medium text-sm">{model.name}</div>
                                  <div className="text-xs text-muted-foreground">
                                    {model.description}
                                  </div>
                                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                    <span>Max: {model.max_tokens.toLocaleString()} tokens</span>
                                    {model.cost_per_input_token && (
                                      <span>
                                        ${(model.cost_per_input_token * 1000).toFixed(3)}/1k tokens
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </button>
                            ))}
                          </div>
                        }
                        open={modelPickerOpen}
                        onOpenChange={setModelPickerOpen}
                        align="start"
                        side="top"
                      />
                    </div>
                  )}

                  {/* Thinking Mode Selector */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Thinking:</span>
                    <Popover
                      trigger={
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="outline"
                                size="sm"
                                className="gap-2"
                              >
                                <Brain className="h-4 w-4" />
                                <ThinkingModeIndicator 
                                  level={THINKING_MODES.find(m => m.id === selectedThinkingMode)?.level || 0} 
                                />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p className="font-medium">{THINKING_MODES.find(m => m.id === selectedThinkingMode)?.name || "Auto"}</p>
                              <p className="text-xs text-muted-foreground">{THINKING_MODES.find(m => m.id === selectedThinkingMode)?.description}</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      }
                      content={
                        <div className="w-[280px] p-1">
                          {THINKING_MODES.map((mode) => (
                            <button
                              key={mode.id}
                              onClick={() => {
                                setSelectedThinkingMode(mode.id);
                                setThinkingModePickerOpen(false);
                              }}
                              className={cn(
                                "w-full flex items-start gap-3 p-3 rounded-md transition-colors text-left",
                                "hover:bg-accent",
                                selectedThinkingMode === mode.id && "bg-accent"
                              )}
                            >
                              <Brain className="h-4 w-4 mt-0.5" />
                              <div className="flex-1 space-y-1">
                                <div className="font-medium text-sm">
                                  {mode.name}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  {mode.description}
                                </div>
                              </div>
                              <ThinkingModeIndicator level={mode.level} />
                            </button>
                          ))}
                        </div>
                      }
                      open={thinkingModePickerOpen}
                      onOpenChange={setThinkingModePickerOpen}
                      align="start"
                      side="top"
                    />
                  </div>
                </div>

                <Button
                  onClick={handleSend}
                  disabled={!canSend}
                  size="default"
                  className="min-w-[60px]"
                >
                  {isLoading ? (
                    <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Fixed Position Input Bar */}
      <div
        className={cn(
          "fixed bottom-0 left-0 right-0 z-40 bg-background border-t border-border",
          className
        )}
      >
        <div className="max-w-5xl mx-auto">
          {/* Image previews */}
          {embeddedImages.length > 0 && (
            <ImagePreview
              images={embeddedImages}
              onRemove={handleRemoveImage}
              className="border-b border-border"
            />
          )}

          <div className="p-4">
            <div className="flex items-end gap-3">
              {/* Provider Picker */}
              <Popover
                trigger={
                  <Button
                    variant="outline"
                    size="default"
                    disabled={isLoading || disabled}
                    className="gap-2 min-w-[140px] justify-start"
                  >
                    {currentProvider ? getProviderIcon(currentProvider.id) : <Bot className="h-4 w-4" />}
                    <span className="flex-1 text-left truncate">
                      {currentProvider?.name || "Provider"}
                    </span>
                    <ChevronUp className="h-4 w-4 opacity-50" />
                  </Button>
                }
                content={
                  <div className="w-[280px] p-1">
                    {providers.map((provider) => (
                      <button
                        key={provider.id}
                        onClick={() => {
                          setSelectedProvider(provider.id);
                          setProviderPickerOpen(false);
                        }}
                        className={cn(
                          "w-full flex items-start gap-3 p-3 rounded-md transition-colors text-left",
                          "hover:bg-accent",
                          selectedProvider === provider.id && "bg-accent"
                        )}
                      >
                        <div className="mt-0.5">{getProviderIcon(provider.id)}</div>
                        <div className="flex-1 space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-sm">{provider.name}</span>
                            {provider.is_local && (
                              <Badge variant="secondary" className="text-xs">Local</Badge>
                            )}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {provider.description}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                }
                open={providerPickerOpen}
                onOpenChange={setProviderPickerOpen}
                align="start"
                side="top"
              />

              {/* Model Picker */}
              {currentProvider && currentProvider.models.length > 0 && (
                <Popover
                  trigger={
                    <Button
                      variant="outline"
                      size="default"
                      disabled={isLoading || disabled}
                      className="gap-2 min-w-[120px] justify-start"
                    >
                      <span className="flex-1 text-left truncate">
                        {currentModel?.name || "Model"}
                      </span>
                      <ChevronUp className="h-4 w-4 opacity-50" />
                    </Button>
                  }
                  content={
                    <div className="w-[300px] p-1">
                      {currentProvider.models.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => {
                            setSelectedModel(model.id);
                            setModelPickerOpen(false);
                          }}
                          className={cn(
                            "w-full flex items-start gap-3 p-3 rounded-md transition-colors text-left",
                            "hover:bg-accent",
                            selectedModel === model.id && "bg-accent"
                          )}
                        >
                          <div className="flex-1 space-y-1">
                            <div className="font-medium text-sm">{model.name}</div>
                            <div className="text-xs text-muted-foreground">
                              {model.description}
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  }
                  open={modelPickerOpen}
                  onOpenChange={setModelPickerOpen}
                  align="start"
                  side="top"
                />
              )}

              {/* Thinking Mode Picker */}
              <Popover
                trigger={
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="outline"
                          size="default"
                          disabled={isLoading || disabled}
                          className="gap-2"
                        >
                          <Brain className="h-4 w-4" />
                          <ThinkingModeIndicator 
                            level={THINKING_MODES.find(m => m.id === selectedThinkingMode)?.level || 0} 
                          />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="font-medium">{THINKING_MODES.find(m => m.id === selectedThinkingMode)?.name || "Auto"}</p>
                        <p className="text-xs text-muted-foreground">{THINKING_MODES.find(m => m.id === selectedThinkingMode)?.description}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                }
                content={
                  <div className="w-[280px] p-1">
                    {THINKING_MODES.map((mode) => (
                      <button
                        key={mode.id}
                        onClick={() => {
                          setSelectedThinkingMode(mode.id);
                          setThinkingModePickerOpen(false);
                        }}
                        className={cn(
                          "w-full flex items-start gap-3 p-3 rounded-md transition-colors text-left",
                          "hover:bg-accent",
                          selectedThinkingMode === mode.id && "bg-accent"
                        )}
                      >
                        <Brain className="h-4 w-4 mt-0.5" />
                        <div className="flex-1 space-y-1">
                          <div className="font-medium text-sm">
                            {mode.name}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {mode.description}
                          </div>
                        </div>
                        <ThinkingModeIndicator level={mode.level} />
                      </button>
                    ))}
                  </div>
                }
                open={thinkingModePickerOpen}
                onOpenChange={setThinkingModePickerOpen}
                align="start"
                side="top"
              />

              {/* Prompt Input */}
              <div className="flex-1 relative">
                <Textarea
                  ref={textareaRef}
                  value={prompt}
                  onChange={handleTextChange}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask anything..."
                  disabled={isLoading || disabled}
                  className="min-h-[44px] max-h-[120px] resize-none pr-10"
                  rows={1}
                />

                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsExpanded(true)}
                  disabled={isLoading || disabled}
                  className="absolute right-1 bottom-1 h-8 w-8"
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>

                {/* File Picker */}
                <AnimatePresence>
                  {showFilePicker && projectPath && projectPath.trim() && (
                    <FilePicker
                      basePath={projectPath.trim()}
                      onSelect={handleFileSelect}
                      onClose={handleFilePickerClose}
                      initialQuery={filePickerQuery}
                    />
                  )}
                </AnimatePresence>
              </div>

              {/* Send/Stop Button */}
              <Button
                onClick={isLoading ? onCancel : handleSend}
                disabled={isLoading ? false : !canSend}
                variant={isLoading ? "destructive" : "default"}
                size="default"
                className="min-w-[60px]"
              >
                {isLoading ? (
                  <>
                    <Square className="h-4 w-4 mr-1" />
                    Stop
                  </>
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>

            {/* Info Bar */}
            <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
              <div className="flex items-center gap-4">
                <span>Press Enter to send, Shift+Enter for new line</span>
                {projectPath?.trim() && <span>@ to mention files</span>}
                {tokenCount > 0 && <span>{tokenCount.toLocaleString()} tokens</span>}
                {estimatedCost > 0 && <span>Est. cost: {formatCost(estimatedCost)}</span>}
              </div>
              {currentProvider && currentModel && (
                <span className="flex items-center gap-1">
                  {getProviderIcon(currentProvider.id)}
                  {currentModel.name}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export const FloatingPromptInput = React.forwardRef<
  FloatingPromptInputRef,
  FloatingPromptInputProps
>(FloatingPromptInputInner);

FloatingPromptInput.displayName = 'FloatingPromptInput';