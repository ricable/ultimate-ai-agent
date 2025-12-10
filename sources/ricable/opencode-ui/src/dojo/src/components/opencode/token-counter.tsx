import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Hash, DollarSign, TrendingUp, TrendingDown, Minus, ChevronUp, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface TokenCounterProps {
  /**
   * Input tokens used
   */
  inputTokens: number;
  /**
   * Output tokens generated
   */
  outputTokens: number;
  /**
   * Total cost for this session/interaction
   */
  cost?: number;
  /**
   * Cost breakdown per provider (for multi-provider sessions)
   */
  costBreakdown?: Array<{
    provider: string;
    cost: number;
    tokens: number;
    percentage: number;
  }>;
  /**
   * Whether to show the counter
   */
  show?: boolean;
  /**
   * Whether to show detailed breakdown initially
   */
  expanded?: boolean;
  /**
   * Optional className for styling
   */
  className?: string;
  /**
   * Position of the counter
   */
  position?: "bottom-right" | "bottom-left" | "top-right" | "top-left";
}

/**
 * Enhanced TokenCounter component for OpenCode - Real-time token and cost tracking
 * Supports multi-provider cost breakdown and detailed analytics
 * 
 * @example
 * <TokenCounter 
 *   inputTokens={1234} 
 *   outputTokens={567}
 *   cost={0.045}
 *   costBreakdown={providerBreakdown}
 *   show={true} 
 * />
 */
export const TokenCounter: React.FC<TokenCounterProps> = ({
  inputTokens,
  outputTokens,
  cost = 0,
  costBreakdown = [],
  show = true,
  expanded = false,
  className,
  position = "bottom-right"
}) => {
  const [isExpanded, setIsExpanded] = useState(expanded);
  
  const totalTokens = inputTokens + outputTokens;
  
  if (!show || totalTokens === 0) return null;

  // Format numbers with appropriate suffixes
  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}k`;
    }
    return num.toLocaleString();
  };

  // Format cost
  const formatCost = (costValue: number): string => {
    if (costValue === 0) return "Free";
    if (costValue < 0.001) return "<$0.001";
    if (costValue < 0.01) return `$${costValue.toFixed(4)}`;
    if (costValue < 1) return `$${costValue.toFixed(3)}`;
    return `$${costValue.toFixed(2)}`;
  };

  // Get position classes
  const getPositionClasses = () => {
    switch (position) {
      case "bottom-left":
        return "bottom-20 left-4";
      case "top-right":
        return "top-4 right-4";
      case "top-left":
        return "top-4 left-4";
      case "bottom-right":
      default:
        return "bottom-20 right-4";
    }
  };

  // Calculate token ratio for visual indicator
  const tokenRatio = totalTokens > 0 ? (outputTokens / totalTokens) * 100 : 50;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      className={cn(
        "fixed z-30",
        getPositionClasses(),
        "bg-background/95 backdrop-blur-sm",
        "border border-border rounded-lg shadow-lg",
        "min-w-[200px] max-w-[300px]",
        className
      )}
    >
      {/* Main Counter Display */}
      <div className="p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Hash className="h-4 w-4 text-muted-foreground" />
            <div className="flex flex-col">
              <span className="font-mono text-sm font-medium">
                {formatNumber(totalTokens)}
              </span>
              <span className="text-xs text-muted-foreground">tokens</span>
            </div>
          </div>

          {cost > 0 && (
            <div className="flex items-center gap-1">
              <DollarSign className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="font-mono text-sm font-medium">
                {formatCost(cost)}
              </span>
            </div>
          )}

          {/* Expand/Collapse Button */}
          {(inputTokens > 0 || outputTokens > 0 || costBreakdown.length > 0) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="h-6 w-6 p-0"
            >
              {isExpanded ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronUp className="h-3 w-3" />
              )}
            </Button>
          )}
        </div>

        {/* Token ratio bar */}
        {totalTokens > 0 && (
          <div className="mt-2 space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Input</span>
              <span>Output</span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div 
                className="h-full bg-green-500 transition-all duration-300"
                style={{ width: `${100 - tokenRatio}%` }}
              />
              <div 
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${tokenRatio}%`, marginTop: '-6px' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Expanded Details */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-border"
          >
            <div className="p-3 space-y-3">
              {/* Token Breakdown */}
              {(inputTokens > 0 || outputTokens > 0) && (
                <div className="space-y-2">
                  <h4 className="text-xs font-medium text-muted-foreground">Token Breakdown</h4>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <TrendingDown className="h-3 w-3 text-green-500" />
                      <span className="text-xs">Input</span>
                    </div>
                    <span className="font-mono text-xs">{formatNumber(inputTokens)}</span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-3 w-3 text-blue-500" />
                      <span className="text-xs">Output</span>
                    </div>
                    <span className="font-mono text-xs">{formatNumber(outputTokens)}</span>
                  </div>
                </div>
              )}

              {/* Cost Breakdown by Provider */}
              {costBreakdown.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-medium text-muted-foreground">Cost by Provider</h4>
                  
                  {costBreakdown.map((item, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="text-xs px-1.5 py-0.5">
                          {item.provider}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {item.percentage.toFixed(1)}%
                        </span>
                      </div>
                      <span className="font-mono text-xs">{formatCost(item.cost)}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Total Summary */}
              {cost > 0 && (
                <div className="pt-2 border-t border-border">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium">Total Cost</span>
                    <span className="font-mono text-sm font-medium">{formatCost(cost)}</span>
                  </div>
                  
                  {totalTokens > 0 && (
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-xs text-muted-foreground">Cost per 1k tokens</span>
                      <span className="font-mono text-xs text-muted-foreground">
                        {formatCost((cost / totalTokens) * 1000)}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};