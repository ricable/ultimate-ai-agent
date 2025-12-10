/**
 * MLPatternStorage - Phase 2 ML Development Memory Coordination
 * Advanced pattern storage with AgentDB integration for RL, causal inference, and DSPy components
 */

import { EventEmitter } from 'events';
import { AgentDBMemoryManager } from '../agentdb/AgentDBMemoryManager';
import { CognitiveConsciousnessCore } from '../cognitive/CognitiveConsciousnessCore';

// Core ML Pattern Interfaces
export interface TrainingEpisode {
  episode_id: string;
  timestamp: number;
  algorithm: 'reinforcement_learning' | 'causal_inference' | 'dspy_optimization';
  domain: 'mobility' | 'energy' | 'coverage' | 'capacity' | 'performance';
  input_state: Float32Array;
  actions_taken: PolicyAction[];
  rewards: number[];
  outcome: EpisodeOutcome;
  performance_metrics: PerformanceMetrics;
  causal_factors?: CausalFactor[];
  temporal_signature: TemporalSignature;
  cross_agent_applicable: boolean;
  success_rate: number;
  confidence: number;
}

export interface CausalInsight {
  insight_id: string;
  type: 'handover_success' | 'energy_efficiency' | 'coverage_optimization' | 'capacity_management';
  target_variable: string;
  causal_factors: CausalFactor[];
  intervention_recommendations: InterventionRecommendation[];
  confidence: number;
  temporal_validity: TimeWindow;
  cross_agent_transferability: number;
  learned_from: string[];
}

export interface OptimizationPattern {
  pattern_id: string;
  name: string;
  category: PatternCategory;
  vector_signature: Float32Array;
  success_rate: number;
  improvement_magnitude: number;
  applicable_domains: string[];
  required_capabilities: string[];
  temporal_patterns: TemporalPattern[];
  causal_relationships: CausalRelationship[];
  reinforcement_score: number;
  adaptation_count: number;
  last_success: number;
}

export interface MemoryCoordinatorConfig {
  agentdb_config: {
    swarmId: string;
    syncProtocol: 'QUIC';
    persistenceEnabled: boolean;
    crossAgentLearning: boolean;
    patternRecognition: boolean;
  };
  vector_config: {
    dimensions: number;
    quantizationBits: number; // 32x memory reduction = 5 bits (32/5)
    indexingMethod: 'HNSW'; // Hierarchical Navigable Small World
    similarityThreshold: number;
  };
  ml_config: {
    learningRate: number;
    patternRetention: number; // days
    crossAgentTransferThreshold: number;
    reinforcementInterval: number; // minutes
  };
  performance_config: {
    memoryQuotaGB: number;
    syncLatencyTarget: number; // <1ms
    searchSpeedTarget: number; // queries/second
    autoOptimizationEnabled: boolean;
  };
}

export interface MemoryStats {
  totalPatterns: number;
  rlEpisodes: number;
  causalInsights: number;
  optimizationPatterns: number;
  crossAgentShared: number;
  memoryUsageGB: number;
  searchLatency: number;
  syncLatency: number;
  learningRate: number;
  compressionRatio: number;
}

export class MLPatternStorage extends EventEmitter {
  private config: MemoryCoordinatorConfig;
  private agentDB: AgentDBMemoryManager;
  private cognitiveCore: CognitiveConsciousnessCore;
  private vectorIndex: HNSWIndex;
  private quantizationEngine: QuantizationEngine;
  private crossAgentCoordinator: CrossAgentMemoryCoordinator;
  private performanceMonitor: MemoryPerformanceMonitor;
  private learningEngine: AdaptiveLearningEngine;

  // Memory stores
  private trainingEpisodes: Map<string, TrainingEpisode> = new Map();
  private causalInsights: Map<string, CausalInsight> = new Map();
  private optimizationPatterns: Map<string, OptimizationPattern> = new Map();
  private agentMemories: Map<string, AgentMemorySpace> = new Map();

  // Performance tracking
  private memoryStats: MemoryStats;
  private activeConnections: Map<string, any> = new Map();
  private syncIntervals: Map<string, NodeJS.Timeout> = new Map();

  constructor(config: MemoryCoordinatorConfig) {
    super();
    this.config = config;

    this.initializeMemoryStats();
    this.setupComponents();
  }

  /**
   * Initialize ML pattern storage system
   */
  async initialize(): Promise<void> {
    console.log('🧠 Initializing ML Pattern Storage for Phase 2...');

    try {
      // Phase 1: Initialize AgentDB with ML-specific configuration
      await this.initializeAgentDB();

      // Phase 2: Setup vector indexing with HNSW and quantization
      await this.initializeVectorIndexing();

      // Phase 3: Initialize cognitive consciousness for ML optimization
      await this.initializeCognitiveCore();

      // Phase 4: Setup cross-agent memory coordination
      await this.initializeCrossAgentCoordination();

      // Phase 5: Initialize performance monitoring
      await this.initializePerformanceMonitoring();

      // Phase 6: Enable QUIC synchronization for <1ms sync
      await this.enableQUICSynchronization();

      console.log('✅ ML Pattern Storage initialized successfully');
      this.emit('initialized', this.getStats());

    } catch (error) {
      console.error('❌ ML Pattern Storage initialization failed:', error);
      throw error;
    }
  }

  /**
   * Store RL training episode with automatic pattern extraction
   */
  async storeRLTrainingEpisode(episode: TrainingEpisode): Promise<string> {
    const startTime = performance.now();

    try {
      // Quantize vector for 32x memory reduction
      const quantizedVector = await this.quantizationEngine.quantize(episode.input_state);

      // Extract learning patterns
      const patterns = await this.extractLearningPatterns(episode);

      // Store in AgentDB with metadata
      await this.agentDB.store(`rl_episode_${episode.episode_id}`, {
        ...episode,
        quantized_vector: quantizedVector,
        extracted_patterns: patterns,
        cross_agent_applicable: episode.cross_agent_applicable,
        storage_timestamp: Date.now()
      }, {
        tags: ['rl', 'training', episode.domain, episode.algorithm],
        shared: episode.cross_agent_applicable,
        priority: episode.success_rate > 0.8 ? 'high' : 'medium'
      });

      // Store in local memory for fast access
      this.trainingEpisodes.set(episode.episode_id, episode);

      // Add to HNSW vector index for fast similarity search
      await this.vectorIndex.add(episode.episode_id, quantizedVector, {
        type: 'rl_episode',
        domain: episode.domain,
        success_rate: episode.success_rate
      });

      // Share with cross-agent coordinator if applicable
      if (episode.cross_agent_applicable) {
        await this.crossAgentCoordinator.sharePattern(episode.episode_id, {
          type: 'rl_episode',
          pattern: episode,
          confidence: episode.confidence,
          transferability: this.calculateTransferability(episode)
        });
      }

      // Update learning engine
      await this.learningEngine.updateFromEpisode(episode);

      // Update memory stats
      this.updateMemoryStats('rl_episode', 1);

      const storeTime = performance.now() - startTime;
      console.log(`💾 RL episode stored in ${storeTime.toFixed(2)}ms`);

      this.emit('episode_stored', { episodeId: episode.episode_id, storeTime });
      return episode.episode_id;

    } catch (error) {
      console.error('❌ Failed to store RL episode:', error);
      throw error;
    }
  }

  /**
   * Retrieve causal insights with contextual matching
   */
  async retrieveCausalInsights(query: CausalQuery): Promise<CausalInsight[]> {
    const startTime = performance.now();

    try {
      // Vectorize query for similarity search
      const queryVector = await this.vectorizeCausalQuery(query);

      // Search in HNSW index for similar patterns
      const candidates = await this.vectorIndex.search(queryVector, {
        k: 20,
        filter: { type: 'causal_insight', domain: query.domain },
        threshold: this.config.vector_config.similarityThreshold
      });

      // Detailed matching and ranking
      const insights: CausalInsight[] = [];
      for (const candidate of candidates) {
        const insight = this.causalInsights.get(candidate.id);
        if (insight && this.isInsightApplicable(insight, query)) {
          insights.push({
            ...insight,
            relevance_score: candidate.score,
            contextual_match: await this.calculateContextualMatch(insight, query)
          });
        }
      }

      // Sort by relevance and contextual match
      insights.sort((a, b) =>
        (b.relevance_score || 0) * 0.7 + (b.contextual_match || 0) * 0.3 -
        ((a.relevance_score || 0) * 0.7 + (a.contextual_match || 0) * 0.3)
      );

      const retrieveTime = performance.now() - startTime;
      console.log(`🔍 Retrieved ${insights.length} causal insights in ${retrieveTime.toFixed(2)}ms`);

      return insights.slice(0, query.limit || 10);

    } catch (error) {
      console.error('❌ Failed to retrieve causal insights:', error);
      return [];
    }
  }

  /**
   * Share optimization patterns between agents
   */
  async shareOptimizationPatterns(patterns: OptimizationPattern[]): Promise<void> {
    console.log(`📚 Sharing ${patterns.length} optimization patterns across agents...`);

    try {
      for (const pattern of patterns) {
        // Quantize pattern for efficient storage and transfer
        const quantizedPattern = await this.quantizationEngine.quantizePattern(pattern);

        // Store in AgentDB with cross-agent sharing enabled
        await this.agentDB.store(`pattern_${pattern.pattern_id}`, {
          ...pattern,
          quantized_signature: quantizedPattern,
          shared_timestamp: Date.now(),
          sharing_agent: this.config.agentdb_config.swarmId
        }, {
          tags: ['optimization', 'shared', pattern.category, ...pattern.applicable_domains],
          shared: true,
          priority: pattern.success_rate > 0.85 ? 'high' : 'medium'
        });

        // Update cross-agent coordinator
        await this.crossAgentCoordinator.broadcastPattern(pattern.pattern_id, {
          type: 'optimization_pattern',
          pattern: quantizedPattern,
          source_agent: this.config.agentdb_config.swarmId,
          confidence: pattern.success_rate,
          recommended_for: pattern.applicable_domains
        });
      }

      this.updateMemoryStats('shared_patterns', patterns.length);
      console.log(`✅ Shared ${patterns.length} patterns across agents`);

      this.emit('patterns_shared', { count: patterns.length });

    } catch (error) {
      console.error('❌ Failed to share optimization patterns:', error);
      throw error;
    }
  }

  /**
   * Learn from execution outcomes and update patterns
   */
  async learnFromExecution(outcomes: ExecutionOutcome[]): Promise<void> {
    console.log(`🎓 Learning from ${outcomes.length} execution outcomes...`);

    try {
      let patternsUpdated = 0;
      let newPatternsCreated = 0;

      for (const outcome of outcomes) {
        // Find similar patterns in memory
        const similarPatterns = await this.findSimilarPatterns(outcome);

        if (similarPatterns.length > 0) {
          // Update existing patterns based on outcome
          for (const pattern of similarPatterns) {
            const updateResult = await this.updatePatternFromOutcome(pattern, outcome);
            if (updateResult.updated) patternsUpdated++;
          }
        } else {
          // Create new pattern from successful outcome
          if (outcome.success && outcome.improvement > 0.1) {
            const newPattern = await this.createPatternFromOutcome(outcome);
            await this.storeOptimizationPattern(newPattern);
            newPatternsCreated++;
          }
        }

        // Store execution outcome for future learning
        await this.storeExecutionOutcome(outcome);
      }

      // Trigger memory consolidation if needed
      if (patternsUpdated > 0 || newPatternsCreated > 0) {
        await this.consolidateMemory();
      }

      console.log(`📈 Learning complete: ${patternsUpdated} patterns updated, ${newPatternsCreated} created`);

      this.emit('learning_complete', {
        outcomesProcessed: outcomes.length,
        patternsUpdated,
        newPatternsCreated
      });

    } catch (error) {
      console.error('❌ Failed to learn from execution outcomes:', error);
      throw error;
    }
  }

  /**
   * Get comprehensive memory statistics
   */
  async getStatistics(): Promise<MemoryStats> {
    try {
      // Get AgentDB statistics
      const agentDBStats = await this.agentDB.getStatistics();

      // Get vector index statistics
      const indexStats = await this.vectorIndex.getStats();

      // Get cross-agent statistics
      const crossAgentStats = await this.crossAgentCoordinator.getStats();

      return {
        ...this.memoryStats,
        totalPatterns: this.trainingEpisodes.size + this.causalInsights.size + this.optimizationPatterns.size,
        rlEpisodes: this.trainingEpisodes.size,
        causalInsights: this.causalInsights.size,
        optimizationPatterns: this.optimizationPatterns.size,
        crossAgentShared: crossAgentStats.sharedPatterns,
        memoryUsageGB: agentDBStats.performance.memoryUsage / 1024,
        searchLatency: indexStats.averageSearchTime,
        syncLatency: agentDBStats.performance.syncLatency,
        compressionRatio: this.config.vector_config.quantizationBits / 32,
        learningRate: this.config.ml_config.learningRate
      };

    } catch (error) {
      console.error('❌ Failed to get memory statistics:', error);
      return this.memoryStats;
    }
  }

  /**
   * Optimize memory performance and cleanup
   */
  async optimizeMemory(): Promise<void> {
    console.log('⚡ Optimizing memory performance...');

    try {
      // Optimize vector index
      await this.vectorIndex.optimize();

      // Consolidate old patterns
      await this.consolidateMemory();

      // Optimize quantization
      await this.quantizationEngine.optimize();

      // Update AgentDB performance
      await this.agentDB.updatePerformanceMetrics();

      console.log('✅ Memory optimization complete');
      this.emit('memory_optimized', await this.getStatistics());

    } catch (error) {
      console.error('❌ Memory optimization failed:', error);
      throw error;
    }
  }

  // Private helper methods

  private initializeMemoryStats(): void {
    this.memoryStats = {
      totalPatterns: 0,
      rlEpisodes: 0,
      causalInsights: 0,
      optimizationPatterns: 0,
      crossAgentShared: 0,
      memoryUsageGB: 0,
      searchLatency: 0,
      syncLatency: 0,
      learningRate: this.config.ml_config.learningRate,
      compressionRatio: this.config.vector_config.quantizationBits / 32
    };
  }

  private setupComponents(): void {
    this.agentDB = new AgentDBMemoryManager(this.config.agentdb_config);
    this.vectorIndex = new HNSWIndex(this.config.vector_config);
    this.quantizationEngine = new QuantizationEngine(this.config.vector_config.quantizationBits);
    this.crossAgentCoordinator = new CrossAgentMemoryCoordinator();
    this.performanceMonitor = new MemoryPerformanceMonitor(this.config.performance_config);
    this.learningEngine = new AdaptiveLearningEngine(this.config.ml_config);

    this.cognitiveCore = new CognitiveConsciousnessCore({
      level: 'maximum',
      temporalExpansion: 1000,
      strangeLoopOptimization: true,
      autonomousAdaptation: true
    });
  }

  private async initializeAgentDB(): Promise<void> {
    await this.agentDB.initialize();
    console.log('✅ AgentDB initialized with ML configuration');
  }

  private async initializeVectorIndexing(): Promise<void> {
    await this.vectorIndex.initialize();
    console.log(`✅ HNSW vector index initialized with ${this.config.vector_config.dimensions}D vectors`);
  }

  private async initializeCognitiveCore(): Promise<void> {
    await this.cognitiveCore.initialize();
    console.log('✅ Cognitive consciousness core initialized for ML optimization');
  }

  private async initializeCrossAgentCoordination(): Promise<void> {
    await this.crossAgentCoordinator.initialize({
      swarmId: this.config.agentdb_config.swarmId,
      supportedAgents: ['ml-developer', 'ml-researcher', 'ml-analyst'],
      transferThreshold: this.config.ml_config.crossAgentTransferThreshold
    });
    console.log('✅ Cross-agent memory coordinator initialized');
  }

  private async initializePerformanceMonitoring(): Promise<void> {
    await this.performanceMonitor.initialize();
    console.log('✅ Performance monitoring initialized');
  }

  private async enableQUICSynchronization(): Promise<void> {
    await this.agentDB.enableQUICSynchronization();
    console.log('⚡ QUIC synchronization enabled for <1ms sync');
  }

  private async extractLearningPatterns(episode: TrainingEpisode): Promise<any[]> {
    // Extract patterns from successful episodes
    if (episode.success_rate > 0.8) {
      return [{
        type: 'successful_strategy',
        domain: episode.domain,
        actions: episode.actions_taken,
        rewards: episode.rewards,
        confidence: episode.confidence
      }];
    }
    return [];
  }

  private calculateTransferability(episode: TrainingEpisode): number {
    // Calculate how transferable this episode is to other agents
    let transferability = episode.cross_agent_applicable ? 0.5 : 0.1;

    if (episode.success_rate > 0.9) transferability += 0.3;
    if (episode.confidence > 0.8) transferability += 0.2;

    return Math.min(1.0, transferability);
  }

  private async vectorizeCausalQuery(query: CausalQuery): Promise<Float32Array> {
    // Convert causal query to vector for similarity search
    const features = [
      ...query.target_variables.map(v => this.hashToFloat(v)),
      ...query.context_factors.map(f => this.hashToFloat(f)),
      query.temporal_window?.start || 0,
      query.temporal_window?.end || 0,
      query.confidence_threshold || 0.5
    ];

    return new Float32Array(features.slice(0, this.config.vector_config.dimensions));
  }

  private hashToFloat(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash + str.charCodeAt(i)) & 0xffffffff;
    }
    return Math.abs(hash) / 0xffffffff;
  }

  private isInsightApplicable(insight: CausalInsight, query: CausalQuery): boolean {
    // Check if insight is applicable to the query
    return insight.confidence >= (query.confidence_threshold || 0.5) &&
           this.isWithinTimeWindow(insight.temporal_validity, query.temporal_window) &&
           insight.target_variable === query.target_variable;
  }

  private isWithinTimeWindow(insightWindow: TimeWindow, queryWindow?: TimeWindow): boolean {
    if (!queryWindow) return true;

    return insightWindow.start <= queryWindow.end &&
           insightWindow.end >= queryWindow.start;
  }

  private async calculateContextualMatch(insight: CausalInsight, query: CausalQuery): Promise<number> {
    // Calculate contextual match between insight and query
    const contextMatches = query.context_factors.filter(factor =>
      insight.causal_factors.some(cf => cf.factor.includes(factor))
    ).length;

    return contextMatches / Math.max(query.context_factors.length, 1);
  }

  private async findSimilarPatterns(outcome: ExecutionOutcome): Promise<OptimizationPattern[]> {
    const patterns: OptimizationPattern[] = [];

    for (const pattern of this.optimizationPatterns.values()) {
      if (pattern.category === outcome.category &&
          pattern.applicable_domains.includes(outcome.domain)) {
        patterns.push(pattern);
      }
    }

    return patterns;
  }

  private async updatePatternFromOutcome(pattern: OptimizationPattern, outcome: ExecutionOutcome): Promise<{updated: boolean}> {
    // Update pattern based on execution outcome
    pattern.adaptation_count++;
    pattern.last_success = Date.now();

    if (outcome.success) {
      pattern.success_rate = (pattern.success_rate * 0.8) + (1.0 * 0.2);
      pattern.reinforcement_score += outcome.improvement;
    } else {
      pattern.success_rate = pattern.success_rate * 0.9;
    }

    // Update in storage
    await this.agentDB.store(`pattern_${pattern.pattern_id}`, pattern, {
      tags: ['optimization', pattern.category],
      shared: true
    });

    return { updated: true };
  }

  private async createPatternFromOutcome(outcome: ExecutionOutcome): Promise<OptimizationPattern> {
    return {
      pattern_id: `pattern_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: `Generated from ${outcome.task_type}`,
      category: outcome.category,
      vector_signature: outcome.input_vector,
      success_rate: outcome.success ? 1.0 : 0.0,
      improvement_magnitude: outcome.improvement,
      applicable_domains: [outcome.domain],
      required_capabilities: outcome.required_capabilities,
      temporal_patterns: outcome.temporal_patterns || [],
      causal_relationships: outcome.causal_relationships || [],
      reinforcement_score: outcome.improvement,
      adaptation_count: 1,
      last_success: Date.now()
    };
  }

  private async storeOptimizationPattern(pattern: OptimizationPattern): Promise<void> {
    this.optimizationPatterns.set(pattern.pattern_id, pattern);

    const quantizedPattern = await this.quantizationEngine.quantizePattern(pattern);

    await this.agentDB.store(`pattern_${pattern.pattern_id}`, {
      ...pattern,
      quantized_signature: quantizedPattern
    }, {
      tags: ['optimization', 'new', pattern.category],
      shared: true,
      priority: pattern.success_rate > 0.8 ? 'high' : 'medium'
    });

    await this.vectorIndex.add(pattern.pattern_id, quantizedPattern, {
      type: 'optimization_pattern',
      category: pattern.category,
      success_rate: pattern.success_rate
    });
  }

  private async storeExecutionOutcome(outcome: ExecutionOutcome): Promise<void> {
    await this.agentDB.store(`outcome_${outcome.outcome_id}`, outcome, {
      tags: ['execution', 'outcome', outcome.domain],
      shared: false
    });
  }

  private async consolidateMemory(): Promise<void> {
    // Remove old, low-performing patterns
    const maxAge = Date.now() - (this.config.ml_config.patternRetention * 24 * 60 * 60 * 1000);
    let removedCount = 0;

    for (const [id, pattern] of this.optimizationPatterns) {
      if (pattern.last_success < maxAge && pattern.success_rate < 0.5) {
        this.optimizationPatterns.delete(id);
        await this.vectorIndex.remove(id);
        removedCount++;
      }
    }

    if (removedCount > 0) {
      console.log(`🧹 Consolidated memory: removed ${removedCount} old patterns`);
    }
  }

  private updateMemoryStats(type: string, count: number): void {
    switch (type) {
      case 'rl_episode':
        this.memoryStats.rlEpisodes += count;
        break;
      case 'shared_patterns':
        this.memoryStats.crossAgentShared += count;
        break;
    }
    this.memoryStats.totalPatterns = this.trainingEpisodes.size + this.causalInsights.size + this.optimizationPatterns.size;
  }

  private async storeOptimizationPattern(pattern: OptimizationPattern): Promise<void> {
    this.optimizationPatterns.set(pattern.pattern_id, pattern);

    const quantizedPattern = await this.quantizationEngine.quantizePattern(pattern);

    await this.agentDB.store(`pattern_${pattern.pattern_id}`, {
      ...pattern,
      quantized_signature: quantizedPattern
    }, {
      tags: ['optimization', 'new', pattern.category],
      shared: true,
      priority: pattern.success_rate > 0.8 ? 'high' : 'medium'
    });

    await this.vectorIndex.add(pattern.pattern_id, quantizedPattern, {
      type: 'optimization_pattern',
      category: pattern.category,
      success_rate: pattern.success_rate
    });
  }

  private getStats(): MemoryStats {
    return this.memoryStats;
  }
}

// Supporting class definitions (simplified for brevity)
class HNSWIndex {
  async initialize(): Promise<void> {}
  async add(id: string, vector: Float32Array, metadata: any): Promise<void> {}
  async search(query: Float32Array, options: any): Promise<any[]> { return []; }
  async remove(id: string): Promise<void> {}
  async optimize(): Promise<void> {}
  async getStats(): Promise<any> { return { averageSearchTime: 0 }; }
}

class QuantizationEngine {
  constructor(private bits: number) {}
  async quantize(vector: Float32Array): Promise<Float32Array> { return vector; }
  async quantizePattern(pattern: OptimizationPattern): Promise<OptimizationPattern> { return pattern; }
  async optimize(): Promise<void> {}
}

class CrossAgentMemoryCoordinator {
  async initialize(config: any): Promise<void> {}
  async sharePattern(id: string, pattern: any): Promise<void> {}
  async broadcastPattern(id: string, pattern: any): Promise<void> {}
  async getStats(): Promise<any> { return { sharedPatterns: 0 }; }
}

class MemoryPerformanceMonitor {
  constructor(private config: any) {}
  async initialize(): Promise<void> {}
}

class AdaptiveLearningEngine {
  constructor(private config: any) {}
  async updateFromEpisode(episode: TrainingEpisode): Promise<void> {}
}

// Supporting type definitions
export interface CausalQuery {
  target_variable: string;
  context_factors: string[];
  temporal_window?: TimeWindow;
  confidence_threshold?: number;
  domain?: string;
  limit?: number;
}

export interface TimeWindow {
  start: number;
  end: number;
}

export interface PolicyAction {
  action_id: string;
  type: string;
  parameters: Record<string, number>;
  timestamp: number;
  outcome: string;
}

export interface EpisodeOutcome {
  success: boolean;
  total_reward: number;
  final_state: Float32Array;
  completion_time: number;
  error_count: number;
}

export interface PerformanceMetrics {
  improvement_percentage: number;
  efficiency_gain: number;
  latency_reduction: number;
  accuracy_improvement: number;
}

export interface CausalFactor {
  factor: string;
  influence: number;
  confidence: number;
  temporal_delay: number;
}

export interface TemporalSignature {
  time_of_day: number;
  day_of_week: number;
  seasonal_factor: number;
  trend_duration: number;
}

export interface PatternCategory {
  type: string;
  subcategory: string;
  complexity: 'low' | 'medium' | 'high';
}

export interface TemporalPattern {
  period: 'hourly' | 'daily' | 'weekly' | 'seasonal';
  strength: number;
  phase: number;
}

export interface CausalRelationship {
  cause: string;
  effect: string;
  strength: number;
  delay: number;
}

export interface InterventionRecommendation {
  intervention: string;
  expected_effect: number;
  confidence: number;
  implementation_cost: number;
}

export interface AgentMemorySpace {
  agentId: string;
  agentType: string;
  memoryPatterns: string[];
  sharedPatterns: string[];
  lastSync: number;
}

export interface ExecutionOutcome {
  outcome_id: string;
  task_type: string;
  domain: string;
  category: PatternCategory;
  success: boolean;
  improvement: number;
  input_vector: Float32Array;
  required_capabilities: string[];
  temporal_patterns?: TemporalPattern[];
  causal_relationships?: CausalRelationship[];
}

export default MLPatternStorage;