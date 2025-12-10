/**
 * Multi-Model Router for Titan Council
 *
 * Routes requests to specialized LLM models based on capability:
 * - DeepSeek-R1: Logical analysis and mathematical reasoning
 * - Gemini 1.5 Pro: Context retrieval and historical analysis
 * - Claude 3.7 Sonnet: Strategic synthesis and decision-making
 *
 * Integrates with agentic-flow QUIC transport for low-latency routing.
 */

import { EventEmitter } from 'events';

/**
 * Model endpoint configuration
 */
export interface ModelEndpoint {
  /** Unique identifier for the model */
  model_id: string;

  /** API endpoint URL */
  endpoint_url: string;

  /** Reference to API key (env var name or secret store key) */
  api_key_ref: string;

  /** Optional: Model-specific configuration */
  config?: {
    temperature?: number;
    max_tokens?: number;
    timeout_ms?: number;
  };

  /** Health status */
  healthy?: boolean;

  /** Last health check timestamp */
  last_health_check?: number;

  /** Current load (number of active requests) */
  current_load?: number;
}

/**
 * Capability types for routing decisions
 */
export enum Capability {
  LOGICAL_ANALYSIS = 'logical_analysis',
  MATHEMATICAL_REASONING = 'mathematical_reasoning',
  CHAOS_DETECTION = 'chaos_detection',
  CONTEXT_RETRIEVAL = 'context_retrieval',
  HISTORICAL_ANALYSIS = 'historical_analysis',
  PATTERN_MATCHING = 'pattern_matching',
  STRATEGIC_SYNTHESIS = 'strategic_synthesis',
  DECISION_MAKING = 'decision_making',
  PARAMETER_OPTIMIZATION = 'parameter_optimization'
}

/**
 * Intent classification for routing
 */
export interface Intent {
  type: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  context?: Record<string, any>;
}

/**
 * Routing result
 */
export interface RoutingResult {
  model_id: string;
  endpoint: ModelEndpoint;
  reason: string;
  fallback_chain?: string[];
}

/**
 * Multi-Model Router Configuration
 */
export interface RouterConfig {
  /** Enable load balancing across endpoints */
  enable_load_balancing?: boolean;

  /** Enable automatic failover */
  enable_failover?: boolean;

  /** Health check interval in milliseconds */
  health_check_interval_ms?: number;

  /** Maximum retries for failed requests */
  max_retries?: number;

  /** QUIC transport configuration */
  quic_config?: {
    enabled: boolean;
    endpoint?: string;
    max_concurrent_streams?: number;
  };
}

/**
 * Multi-Model Router
 *
 * Intelligent routing system that directs requests to the most appropriate
 * LLM model based on capability requirements, load balancing, and health status.
 */
export class MultiModelRouter extends EventEmitter {
  private endpoints: Map<string, ModelEndpoint>;
  private capabilityMap: Map<Capability, string[]>;
  private config: RouterConfig;
  private healthCheckTimer?: NodeJS.Timeout;

  constructor(config?: RouterConfig) {
    super();

    this.config = {
      enable_load_balancing: true,
      enable_failover: true,
      health_check_interval_ms: 30000, // 30 seconds
      max_retries: 3,
      quic_config: {
        enabled: true,
        max_concurrent_streams: 100
      },
      ...config
    };

    this.endpoints = new Map();
    this.capabilityMap = new Map();

    this._initializeDefaultEndpoints();
    this._initializeCapabilityMap();

    if (this.config.health_check_interval_ms) {
      this._startHealthChecks();
    }
  }

  /**
   * Initialize default model endpoints for the Council
   */
  private _initializeDefaultEndpoints(): void {
    // DeepSeek-R1: The Analyst (Logical Reasoning)
    this.registerEndpoint({
      model_id: 'deepseek-r1-distill',
      endpoint_url: process.env.DEEPSEEK_ENDPOINT || 'https://api.deepseek.com/v1',
      api_key_ref: 'DEEPSEEK_API_KEY',
      config: {
        temperature: 0.1, // Low temperature for precise logic
        max_tokens: 4096,
        timeout_ms: 10000
      },
      healthy: true,
      current_load: 0
    });

    // Gemini 1.5 Pro: The Historian (Context & Memory)
    this.registerEndpoint({
      model_id: 'gemini-1.5-pro',
      endpoint_url: process.env.GEMINI_ENDPOINT || 'https://generativelanguage.googleapis.com/v1',
      api_key_ref: 'GEMINI_API_KEY',
      config: {
        temperature: 0.7, // Medium temperature for creative retrieval
        max_tokens: 8192,
        timeout_ms: 15000
      },
      healthy: true,
      current_load: 0
    });

    // Claude 3.7 Sonnet: The Strategist (Synthesis & Strategy)
    this.registerEndpoint({
      model_id: 'claude-3-7-sonnet',
      endpoint_url: process.env.CLAUDE_ENDPOINT || 'https://api.anthropic.com/v1',
      api_key_ref: 'ANTHROPIC_API_KEY',
      config: {
        temperature: 0.5, // Balanced temperature for strategic thinking
        max_tokens: 16384,
        timeout_ms: 20000
      },
      healthy: true,
      current_load: 0
    });

    this.emit('endpoints_initialized', {
      count: this.endpoints.size,
      models: Array.from(this.endpoints.keys())
    });
  }

  /**
   * Initialize capability to model mapping
   */
  private _initializeCapabilityMap(): void {
    // Logical Analysis & Mathematical Reasoning -> DeepSeek
    this.capabilityMap.set(Capability.LOGICAL_ANALYSIS, ['deepseek-r1-distill']);
    this.capabilityMap.set(Capability.MATHEMATICAL_REASONING, ['deepseek-r1-distill']);
    this.capabilityMap.set(Capability.CHAOS_DETECTION, ['deepseek-r1-distill']);

    // Context & History -> Gemini
    this.capabilityMap.set(Capability.CONTEXT_RETRIEVAL, ['gemini-1.5-pro']);
    this.capabilityMap.set(Capability.HISTORICAL_ANALYSIS, ['gemini-1.5-pro']);
    this.capabilityMap.set(Capability.PATTERN_MATCHING, ['gemini-1.5-pro']);

    // Strategy & Synthesis -> Claude
    this.capabilityMap.set(Capability.STRATEGIC_SYNTHESIS, ['claude-3-7-sonnet']);
    this.capabilityMap.set(Capability.DECISION_MAKING, ['claude-3-7-sonnet']);
    this.capabilityMap.set(Capability.PARAMETER_OPTIMIZATION, ['claude-3-7-sonnet']);
  }

  /**
   * Register a new model endpoint
   */
  public registerEndpoint(endpoint: ModelEndpoint): void {
    this.endpoints.set(endpoint.model_id, {
      ...endpoint,
      healthy: endpoint.healthy ?? true,
      current_load: endpoint.current_load ?? 0,
      last_health_check: Date.now()
    });

    this.emit('endpoint_registered', endpoint.model_id);
  }

  /**
   * Route a request based on intent and capability
   *
   * @param intent - The intent of the request
   * @param capability - Required capability for handling the request
   * @returns Routing result with selected endpoint
   */
  public route_by_capability(intent: Intent, capability: Capability): RoutingResult {
    const candidateModelIds = this.capabilityMap.get(capability);

    if (!candidateModelIds || candidateModelIds.length === 0) {
      throw new Error(`No models registered for capability: ${capability}`);
    }

    // Get healthy candidates
    const healthyCandidates = candidateModelIds
      .map(id => this.endpoints.get(id))
      .filter((ep): ep is ModelEndpoint => ep !== undefined && ep.healthy === true);

    if (healthyCandidates.length === 0) {
      // Attempt fallback routing
      return this._fallback_route(intent, capability, candidateModelIds);
    }

    // Apply load balancing if enabled and multiple candidates available
    let selectedEndpoint: ModelEndpoint;

    if (this.config.enable_load_balancing && healthyCandidates.length > 1) {
      selectedEndpoint = this.load_balance(healthyCandidates);
    } else {
      selectedEndpoint = healthyCandidates[0];
    }

    // Increment load counter
    selectedEndpoint.current_load = (selectedEndpoint.current_load || 0) + 1;

    this.emit('request_routed', {
      model_id: selectedEndpoint.model_id,
      capability,
      intent_type: intent.type,
      current_load: selectedEndpoint.current_load
    });

    return {
      model_id: selectedEndpoint.model_id,
      endpoint: selectedEndpoint,
      reason: `Capability match: ${capability}`,
      fallback_chain: this._get_fallback_chain(capability)
    };
  }

  /**
   * Load balance across multiple healthy endpoints
   *
   * Uses least-connections algorithm to distribute load evenly.
   *
   * @param candidates - Array of healthy endpoint candidates
   * @returns Selected endpoint with lowest load
   */
  public load_balance(candidates: ModelEndpoint[]): ModelEndpoint {
    if (candidates.length === 0) {
      throw new Error('No candidates available for load balancing');
    }

    if (candidates.length === 1) {
      return candidates[0];
    }

    // Least-connections load balancing
    const sorted = [...candidates].sort((a, b) => {
      const loadA = a.current_load || 0;
      const loadB = b.current_load || 0;
      return loadA - loadB;
    });

    const selected = sorted[0];

    this.emit('load_balanced', {
      selected: selected.model_id,
      load: selected.current_load,
      candidates: candidates.map(c => ({
        id: c.model_id,
        load: c.current_load
      }))
    });

    return selected;
  }

  /**
   * Fallback routing when primary endpoints are unavailable
   */
  private _fallback_route(
    intent: Intent,
    capability: Capability,
    unavailableModelIds: string[]
  ): RoutingResult {
    if (!this.config.enable_failover) {
      throw new Error(
        `All models for capability ${capability} are unavailable and failover is disabled`
      );
    }

    // Build fallback chain based on capability similarity
    const fallbackChain = this._get_fallback_chain(capability);

    for (const fallbackModelId of fallbackChain) {
      const endpoint = this.endpoints.get(fallbackModelId);

      if (endpoint && endpoint.healthy && !unavailableModelIds.includes(fallbackModelId)) {
        this.emit('fallback_activated', {
          original_capability: capability,
          fallback_model: fallbackModelId,
          reason: 'Primary endpoints unavailable'
        });

        endpoint.current_load = (endpoint.current_load || 0) + 1;

        return {
          model_id: fallbackModelId,
          endpoint,
          reason: `Fallback routing (primary models unavailable)`,
          fallback_chain: fallbackChain
        };
      }
    }

    throw new Error(
      `All endpoints unavailable for capability ${capability} and fallback chain exhausted`
    );
  }

  /**
   * Get fallback chain for a capability
   */
  private _get_fallback_chain(capability: Capability): string[] {
    // Define fallback priorities based on capability type
    const fallbackMap: Record<string, string[]> = {
      // Logical/Math -> Claude (strategic reasoning) -> Gemini (pattern matching)
      [Capability.LOGICAL_ANALYSIS]: ['claude-3-7-sonnet', 'gemini-1.5-pro'],
      [Capability.MATHEMATICAL_REASONING]: ['claude-3-7-sonnet', 'gemini-1.5-pro'],
      [Capability.CHAOS_DETECTION]: ['claude-3-7-sonnet', 'gemini-1.5-pro'],

      // Context/History -> Claude (synthesis) -> DeepSeek (analysis)
      [Capability.CONTEXT_RETRIEVAL]: ['claude-3-7-sonnet', 'deepseek-r1-distill'],
      [Capability.HISTORICAL_ANALYSIS]: ['claude-3-7-sonnet', 'deepseek-r1-distill'],
      [Capability.PATTERN_MATCHING]: ['claude-3-7-sonnet', 'deepseek-r1-distill'],

      // Strategy/Synthesis -> Gemini (context) -> DeepSeek (logic)
      [Capability.STRATEGIC_SYNTHESIS]: ['gemini-1.5-pro', 'deepseek-r1-distill'],
      [Capability.DECISION_MAKING]: ['gemini-1.5-pro', 'deepseek-r1-distill'],
      [Capability.PARAMETER_OPTIMIZATION]: ['gemini-1.5-pro', 'deepseek-r1-distill']
    };

    return fallbackMap[capability] || [];
  }

  /**
   * Release a request (decrement load counter)
   */
  public release_request(model_id: string): void {
    const endpoint = this.endpoints.get(model_id);

    if (endpoint && endpoint.current_load && endpoint.current_load > 0) {
      endpoint.current_load -= 1;

      this.emit('request_released', {
        model_id,
        remaining_load: endpoint.current_load
      });
    }
  }

  /**
   * Perform health check on an endpoint
   */
  private async _check_endpoint_health(endpoint: ModelEndpoint): Promise<boolean> {
    try {
      // TODO: Implement actual health check via agentic-flow QUIC transport
      // For now, simulate a health check

      // In production, this would:
      // 1. Send a lightweight ping via QUIC
      // 2. Verify response within timeout
      // 3. Update endpoint health status

      const healthy = Math.random() > 0.05; // 95% uptime simulation

      endpoint.healthy = healthy;
      endpoint.last_health_check = Date.now();

      if (!healthy) {
        this.emit('endpoint_unhealthy', {
          model_id: endpoint.model_id,
          endpoint_url: endpoint.endpoint_url
        });
      }

      return healthy;
    } catch (error) {
      endpoint.healthy = false;
      endpoint.last_health_check = Date.now();

      this.emit('health_check_failed', {
        model_id: endpoint.model_id,
        error: error instanceof Error ? error.message : String(error)
      });

      return false;
    }
  }

  /**
   * Start periodic health checks
   */
  private _startHealthChecks(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }

    this.healthCheckTimer = setInterval(async () => {
      const endpoints = Array.from(this.endpoints.values());

      for (const endpoint of endpoints) {
        await this._check_endpoint_health(endpoint);
      }

      const healthyCount = endpoints.filter(e => e.healthy).length;

      this.emit('health_check_complete', {
        total: endpoints.length,
        healthy: healthyCount,
        unhealthy: endpoints.length - healthyCount
      });
    }, this.config.health_check_interval_ms);
  }

  /**
   * Stop health checks and cleanup
   */
  public shutdown(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = undefined;
    }

    this.emit('shutdown');
    this.removeAllListeners();
  }

  /**
   * Get router statistics
   */
  public get_stats(): {
    total_endpoints: number;
    healthy_endpoints: number;
    total_load: number;
    endpoints: Array<{
      model_id: string;
      healthy: boolean;
      current_load: number;
      last_health_check?: number;
    }>;
  } {
    const endpoints = Array.from(this.endpoints.values());

    return {
      total_endpoints: endpoints.length,
      healthy_endpoints: endpoints.filter(e => e.healthy).length,
      total_load: endpoints.reduce((sum, e) => sum + (e.current_load || 0), 0),
      endpoints: endpoints.map(e => ({
        model_id: e.model_id,
        healthy: e.healthy || false,
        current_load: e.current_load || 0,
        last_health_check: e.last_health_check
      }))
    };
  }

  /**
   * Get endpoint by model ID
   */
  public get_endpoint(model_id: string): ModelEndpoint | undefined {
    return this.endpoints.get(model_id);
  }

  /**
   * Get all registered capabilities
   */
  public get_capabilities(): Capability[] {
    return Array.from(this.capabilityMap.keys());
  }

  /**
   * Get models for a specific capability
   */
  public get_models_for_capability(capability: Capability): string[] {
    return this.capabilityMap.get(capability) || [];
  }
}

/**
 * Singleton instance for global router access
 */
let routerInstance: MultiModelRouter | null = null;

/**
 * Get or create the global router instance
 */
export function getRouter(config?: RouterConfig): MultiModelRouter {
  if (!routerInstance) {
    routerInstance = new MultiModelRouter(config);
  }
  return routerInstance;
}

/**
 * Reset the global router instance (primarily for testing)
 */
export function resetRouter(): void {
  if (routerInstance) {
    routerInstance.shutdown();
    routerInstance = null;
  }
}

export default MultiModelRouter;
