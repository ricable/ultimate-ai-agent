import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { StopCircle, Clock, Hash, DollarSign, Bot, Cpu, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ExecutionControlBarProps {
  isExecuting: boolean;
  onStop: () => void;
  totalTokens?: number;
  elapsedTime?: number; // in seconds
  estimatedCost?: number;
  provider?: string;
  model?: string;
  className?: string;
  /**
   * Additional execution stats
   */
  stats?: {
    messagesCount?: number;
    toolExecutions?: number;
    filesModified?: number;
  };
}

/**
 * Get provider icon based on provider name
 */
const getProviderIcon = (provider?: string) => {
  if (!provider) return <Bot className="h-3.5 w-3.5" />;
  
  switch (provider.toLowerCase()) {
    case "anthropic":
      return <Brain className="h-3.5 w-3.5" />;
    case "openai":
      return <Bot className="h-3.5 w-3.5" />;
    case "google":
      return <Bot className="h-3.5 w-3.5" />;
    case "groq":
      return <Bot className="h-3.5 w-3.5" />;
    case "ollama":
    case "local":
      return <Cpu className="h-3.5 w-3.5" />;
    default:
      return <Bot className="h-3.5 w-3.5" />;
  }
};

/**
 * Floating control bar shown during OpenCode execution
 * Provides stop functionality and real-time multi-provider statistics
 */
export const ExecutionControlBar: React.FC<ExecutionControlBarProps> = ({ 
  isExecuting, 
  onStop, 
  totalTokens = 0,
  elapsedTime = 0,
  estimatedCost = 0,
  provider,
  model,
  stats,
  className 
}) => {
  // Format elapsed time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins > 0) {
      return `${mins}m ${secs.toFixed(0)}s`;
    }
    return `${secs.toFixed(1)}s`;
  };

  // Format token count
  const formatTokens = (tokens: number) => {
    if (tokens >= 1000000) {
      return `${(tokens / 1000000).toFixed(1)}M`;
    }
    if (tokens >= 1000) {
      return `${(tokens / 1000).toFixed(1)}k`;
    }
    return tokens.toString();
  };

  // Format cost
  const formatCost = (cost: number) => {
    if (cost === 0) return "Free";
    if (cost < 0.01) return "<$0.01";
    return `$${cost.toFixed(3)}`;
  };

  // Format model name for display
  const formatModelName = (modelName?: string) => {
    if (!modelName) return "";
    // Truncate long model names
    if (modelName.length > 20) {
      return modelName.substring(0, 17) + "...";
    }
    return modelName;
  };

  return (
    <AnimatePresence>
      {isExecuting && (
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className={cn(
            "fixed bottom-6 left-1/2 -translate-x-1/2 z-50",
            "bg-background/95 backdrop-blur-md border rounded-full shadow-lg",
            "px-6 py-3 flex items-center gap-4 max-w-[90vw]",
            className
          )}
        >
          {/* Rotating symbol indicator */}
          <div className="relative flex items-center justify-center">
            <div className="animate-spin h-4 w-4 border-2 border-primary border-t-transparent rounded-full"></div>
          </div>

          {/* Status text with provider info */}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Executing</span>
            {provider && (
              <div className="flex items-center gap-1">
                {getProviderIcon(provider)}
                <Badge variant="secondary" className="text-xs px-1.5 py-0.5">
                  {provider}
                </Badge>
                {model && (
                  <span className="text-xs text-muted-foreground">
                    {formatModelName(model)}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Divider */}
          <div className="h-4 w-px bg-border" />

          {/* Core Stats */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            {/* Time */}
            <div className="flex items-center gap-1.5">
              <Clock className="h-3.5 w-3.5" />
              <span>{formatTime(elapsedTime)}</span>
            </div>

            {/* Tokens */}
            {totalTokens > 0 && (
              <div className="flex items-center gap-1.5">
                <Hash className="h-3.5 w-3.5" />
                <span>{formatTokens(totalTokens)}</span>
              </div>
            )}

            {/* Cost */}
            {estimatedCost > 0 && (
              <div className="flex items-center gap-1.5">
                <DollarSign className="h-3.5 w-3.5" />
                <span>{formatCost(estimatedCost)}</span>
              </div>
            )}
          </div>

          {/* Additional Stats (if provided) */}
          {stats && (stats.messagesCount || stats.toolExecutions || stats.filesModified) && (
            <>
              {/* Divider */}
              <div className="h-4 w-px bg-border" />
              
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                {stats.messagesCount && stats.messagesCount > 0 && (
                  <span>{stats.messagesCount} msg{stats.messagesCount !== 1 ? 's' : ''}</span>
                )}
                {stats.toolExecutions && stats.toolExecutions > 0 && (
                  <span>{stats.toolExecutions} tool{stats.toolExecutions !== 1 ? 's' : ''}</span>
                )}
                {stats.filesModified && stats.filesModified > 0 && (
                  <span>{stats.filesModified} file{stats.filesModified !== 1 ? 's' : ''}</span>
                )}
              </div>
            </>
          )}

          {/* Divider */}
          <div className="h-4 w-px bg-border" />

          {/* Stop button */}
          <Button
            size="sm"
            variant="destructive"
            onClick={onStop}
            className="gap-2 px-3 py-1 h-7"
          >
            <StopCircle className="h-3.5 w-3.5" />
            Stop
          </Button>
        </motion.div>
      )}
    </AnimatePresence>
  );
};