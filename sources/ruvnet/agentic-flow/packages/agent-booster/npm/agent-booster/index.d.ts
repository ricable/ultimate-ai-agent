/**
 * Agent Booster - TypeScript Definitions
 * Auto-detecting Native/WASM prompt optimization library
 */

/**
 * Runtime type indicator
 */
export type RuntimeType = 'native' | 'wasm' | null;

/**
 * Optimization options for prompt transformations
 */
export interface OptimizationOptions {
  /**
   * Strategy to use for optimization
   * @default 'balanced'
   */
  strategy?: 'aggressive' | 'balanced' | 'conservative';

  /**
   * Maximum tokens in optimized output
   */
  maxTokens?: number;

  /**
   * Preserve specific patterns or keywords
   */
  preserve?: string[];

  /**
   * Target model or API (for model-specific optimizations)
   */
  targetModel?: string;

  /**
   * Custom optimization rules
   */
  customRules?: Record<string, any>;
}

/**
 * Analysis results for a prompt
 */
export interface AnalysisResult {
  /**
   * Original prompt token count
   */
  originalTokens: number;

  /**
   * Estimated optimized token count
   */
  optimizedTokens: number;

  /**
   * Potential token savings
   */
  savings: number;

  /**
   * Savings percentage
   */
  savingsPercent: number;

  /**
   * Detected patterns in the prompt
   */
  patterns: string[];

  /**
   * Optimization suggestions
   */
  suggestions: string[];

  /**
   * Complexity score (0-100)
   */
  complexity: number;
}

/**
 * Version information
 */
export interface VersionInfo {
  /**
   * NPM package version
   */
  version: string;

  /**
   * Currently loaded runtime
   */
  runtime: RuntimeType;

  /**
   * Core library version
   */
  coreVersion: string;
}

/**
 * Applies prompt optimization transformations
 * @param prompt - The input prompt to optimize
 * @param options - Optimization options
 * @returns Promise resolving to optimized prompt
 * @throws Error if no runtime is available
 */
export function apply(prompt: string, options?: OptimizationOptions): Promise<string>;

/**
 * Applies prompt optimization in batch mode
 * @param prompts - Array of prompts to optimize
 * @param options - Optimization options
 * @returns Promise resolving to array of optimized prompts
 * @throws Error if no runtime is available
 */
export function batchApply(prompts: string[], options?: OptimizationOptions): Promise<string[]>;

/**
 * Analyzes a prompt and returns optimization metrics
 * @param prompt - The prompt to analyze
 * @returns Promise resolving to analysis results
 * @throws Error if no runtime is available
 */
export function analyze(prompt: string): Promise<AnalysisResult>;

/**
 * Gets version information including runtime type
 * @returns Version information object
 */
export function getVersion(): VersionInfo;

/**
 * Gets the currently loaded runtime type
 * @returns 'native', 'wasm', or null if not yet initialized
 */
export function getRuntime(): RuntimeType;

/**
 * Explicitly initializes Agent Booster (auto-called on first use)
 * @returns The loaded runtime instance
 * @throws Error if no runtime is available
 */
export function initialize(): any;
