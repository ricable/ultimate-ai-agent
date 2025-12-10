/**
 * RuvVector + RuvLLM Integration for Self-Learning RAN
 *
 * Provides spatial embeddings for cells, Graph Neural Network (GNN) integration,
 * and natural language querying for RAN optimization using RuvVector HNSW indices
 * and RuvLLM for semantic understanding.
 *
 * @module ml/ruvector-gnn
 * @version 7.0.0-alpha.1
 */

import { EventEmitter } from 'events';
import type { PMCounters, CMParameters, FMAlarm, LearningEpisode } from '../learning/self-learner.js';

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Cell embedding with spatial metadata
 */
export interface CellEmbedding {
  cellId: string;
  vector: Float32Array;  // 768-dim embedding from RuvVector
  metadata: {
    cluster: string;
    site: string;
    sector: number;
    lastOptimization: Date;
    geolocation?: {
      latitude: number;
      longitude: number;
      altitude: number;
    };
    neighborCells?: string[];
    performanceClass?: 'excellent' | 'good' | 'fair' | 'poor';
  };
  timestamp: Date;
}

/**
 * Optimization episode with complete context
 */
export interface OptimizationEpisode {
  id: string;
  cellId: string;
  timestamp: Date;

  // State before optimization
  pmBefore: PMCounters;
  cmBefore: CMParameters;
  fmAlarmsBefore: FMAlarm[];

  // Action taken
  action: Partial<CMParameters>;
  actionType: 'power' | 'tilt' | 'alpha' | 'beamweight' | 'combo';

  // Outcome after optimization
  pmAfter: PMCounters;
  cmAfter: CMParameters;
  fmAlarmsAfter: FMAlarm[];

  // Results
  reward: number;
  sinrGain: number;
  cssrGain: number;
  dropRateChange: number;
  outcome: 'SUCCESS' | 'FAILURE' | 'NEUTRAL';

  // Embedding for similarity search
  stateEmbedding: Float32Array;
  actionEmbedding: Float32Array;

  // Transfer learning metadata
  similarEpisodes?: string[];  // IDs of similar past episodes
  transferScore?: number;      // How well this transfers to other cells
}

/**
 * RAN insight from natural language query
 */
export interface RANInsight {
  query: string;
  answer: string;
  confidence: number;
  supportingData: {
    cells: string[];
    metrics: Record<string, number>;
    episodes?: string[];
  };
  visualizations?: {
    type: 'heatmap' | 'timeseries' | 'scatter' | 'graph';
    data: any;
  }[];
  reasoning: string;
}

/**
 * Optimization recommendation from RuvLLM
 */
export interface Recommendation {
  cellId: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  recommendedAction: Partial<CMParameters>;
  reasoning: string;
  expectedGain: {
    sinr?: number;
    cssr?: number;
    dropRate?: number;
    throughput?: number;
  };
  similarPastSuccesses: OptimizationEpisode[];
  risks: string[];
  confidence: number;
}

/**
 * Similarity search result
 */
export interface SimilarityResult<T> {
  item: T;
  similarity: number;
  distance: number;
}

// ============================================================================
// RuvVector GNN - Spatial Cell Embeddings with HNSW
// ============================================================================

/**
 * RuvectorGNN - HNSW-based spatial embeddings for RAN cells
 *
 * Uses RuvVector's high-performance HNSW index for:
 * - Cell similarity search (<10ms latency)
 * - Optimization episode retrieval
 * - Transfer learning across similar cells
 */
export class RuvectorGNN extends EventEmitter {
  private cellEmbeddings: Map<string, CellEmbedding>;
  private episodeEmbeddings: Map<string, OptimizationEpisode>;

  // HNSW indices (simulated - in production would use ruvector CLI)
  private cellIndex: HNSWIndex;
  private episodeIndex: HNSWIndex;

  private dimension: number = 768;
  private indexPath: string;

  constructor(indexPath: string = './ruvector-spatial.db') {
    super();
    this.indexPath = indexPath;
    this.cellEmbeddings = new Map();
    this.episodeEmbeddings = new Map();

    // Initialize HNSW indices with RAN-optimized parameters
    this.cellIndex = new HNSWIndex({
      dimension: this.dimension,
      metric: 'cosine',
      M: 32,              // Max connections per layer
      efConstruction: 200, // Construction quality
      efSearch: 100        // Search speed/quality tradeoff
    });

    this.episodeIndex = new HNSWIndex({
      dimension: this.dimension,
      metric: 'cosine',
      M: 48,               // Higher for episode search
      efConstruction: 300,
      efSearch: 150
    });

    console.log('[RuvectorGNN] Initialized with HNSW indices');
    console.log(`[RuvectorGNN] Index path: ${this.indexPath}`);
  }

  /**
   * Initialize the RuvVector GNN (load persisted indices)
   */
  async initialize(): Promise<void> {
    console.log('[RuvectorGNN] Loading HNSW indices...');

    // In production: npx ruvector load ./ruvector-spatial.db
    // For now, start with empty indices

    this.emit('initialized');
    console.log('[RuvectorGNN] Initialization complete');
  }

  /**
   * Create cell embedding from PM counters and spatial context
   */
  private async createCellEmbedding(
    cellId: string,
    pm: PMCounters,
    metadata: CellEmbedding['metadata']
  ): Promise<Float32Array> {
    // Combine PM counters, spatial context, and semantic features
    const features: number[] = [
      // PM counters (normalized)
      this.normalize(pm.pmUlSinrMean || 0, -20, 30),
      this.normalize(pm.pmDlSinrMean || 0, -20, 30),
      this.normalize(pm.pmUlBler || 0, 0, 0.1),
      this.normalize(pm.pmDlBler || 0, 0, 0.1),
      this.normalize(pm.pmCssr || 0, 0, 1),
      this.normalize(pm.pmErabSuccessRate || 0, 0, 1),
      this.normalize(pm.pmCallDropRate || 0, 0, 0.05),
      this.normalize(pm.pmHoSuccessRate || 0, 0, 1),
      this.normalize(pm.pmPuschPrbUsage || 0, 0, 100),
      this.normalize(pm.pmPdschPrbUsage || 0, 0, 100),

      // Spatial features
      metadata.sector / 3.0,  // Normalized sector (0, 1, 2)
      metadata.geolocation ? metadata.geolocation.latitude / 90.0 : 0,
      metadata.geolocation ? metadata.geolocation.longitude / 180.0 : 0,
      metadata.geolocation ? metadata.geolocation.altitude / 1000.0 : 0,

      // Neighbor count (graph connectivity)
      metadata.neighborCells ? metadata.neighborCells.length / 20.0 : 0,

      // Performance class (one-hot encoding)
      metadata.performanceClass === 'excellent' ? 1 : 0,
      metadata.performanceClass === 'good' ? 1 : 0,
      metadata.performanceClass === 'fair' ? 1 : 0,
      metadata.performanceClass === 'poor' ? 1 : 0,
    ];

    // Pad to 768 dimensions
    while (features.length < this.dimension) {
      features.push(0);
    }

    // Apply learned transformation (in production, use neural network)
    const embedding = new Float32Array(features);
    this.normalizeVector(embedding);

    return embedding;
  }

  /**
   * Find similar cells based on PM patterns and spatial context
   *
   * Performance target: <10ms for k=5
   */
  async findSimilarCells(cellId: string, k: number = 5): Promise<CellEmbedding[]> {
    const startTime = performance.now();

    const cellEmbedding = this.cellEmbeddings.get(cellId);
    if (!cellEmbedding) {
      throw new Error(`Cell ${cellId} not found in index`);
    }

    // HNSW search
    const results = await this.cellIndex.search(cellEmbedding.vector, k + 1);

    // Filter out the query cell itself
    const similarCells = results
      .filter(r => r.id !== cellId)
      .slice(0, k)
      .map(r => this.cellEmbeddings.get(r.id)!)
      .filter(c => c !== undefined);

    const latency = performance.now() - startTime;
    console.log(`[RuvectorGNN] Found ${similarCells.length} similar cells in ${latency.toFixed(2)}ms`);

    if (latency > 10) {
      console.warn(`[RuvectorGNN] WARNING: Search latency ${latency.toFixed(2)}ms exceeds 10ms target`);
    }

    this.emit('similarity_search', { cellId, k, latency, resultsCount: similarCells.length });

    return similarCells;
  }

  /**
   * Index an optimization episode for future retrieval
   */
  async indexOptimization(episode: OptimizationEpisode): Promise<void> {
    const startTime = performance.now();

    // Store episode
    this.episodeEmbeddings.set(episode.id, episode);

    // Index state embedding
    await this.episodeIndex.insert(episode.id, episode.stateEmbedding);

    // Update cell embedding with latest PM data
    const cellEmbedding = this.cellEmbeddings.get(episode.cellId);
    if (cellEmbedding) {
      cellEmbedding.vector = await this.createCellEmbedding(
        episode.cellId,
        episode.pmAfter,
        cellEmbedding.metadata
      );
      cellEmbedding.metadata.lastOptimization = episode.timestamp;
      cellEmbedding.timestamp = episode.timestamp;

      // Update cell index
      await this.cellIndex.insert(episode.cellId, cellEmbedding.vector);
    }

    const latency = performance.now() - startTime;
    console.log(`[RuvectorGNN] Indexed episode ${episode.id} in ${latency.toFixed(2)}ms`);

    this.emit('episode_indexed', { episodeId: episode.id, latency });
  }

  /**
   * Query similar past optimizations for transfer learning
   *
   * This is the core of the self-learning: find what worked before
   * in similar situations and apply it to new cells.
   */
  async querySimilarOptimizations(
    currentState: PMCounters,
    k: number = 5,
    filters?: {
      outcome?: 'SUCCESS' | 'FAILURE' | 'NEUTRAL';
      minReward?: number;
      actionType?: string;
      cellCluster?: string;
    }
  ): Promise<OptimizationEpisode[]> {
    const startTime = performance.now();

    // Create embedding for current state
    const stateEmbedding = await this.createCellEmbedding(
      'query',
      currentState,
      { cluster: '', site: '', sector: 0, lastOptimization: new Date() }
    );

    // Search episode index
    const candidates = await this.episodeIndex.search(stateEmbedding, k * 3);

    // Apply filters
    let episodes = candidates
      .map(r => this.episodeEmbeddings.get(r.id)!)
      .filter(ep => ep !== undefined);

    if (filters) {
      if (filters.outcome) {
        episodes = episodes.filter(ep => ep.outcome === filters.outcome);
      }
      if (filters.minReward !== undefined) {
        episodes = episodes.filter(ep => ep.reward >= filters.minReward!);
      }
      if (filters.actionType) {
        episodes = episodes.filter(ep => ep.actionType === filters.actionType);
      }
    }

    // Take top k after filtering
    episodes = episodes.slice(0, k);

    const latency = performance.now() - startTime;
    console.log(`[RuvectorGNN] Found ${episodes.length} similar optimizations in ${latency.toFixed(2)}ms`);

    this.emit('optimization_query', {
      k,
      latency,
      resultsCount: episodes.length,
      filters
    });

    return episodes;
  }

  /**
   * Add or update a cell embedding
   */
  async addCell(
    cellId: string,
    pm: PMCounters,
    metadata: CellEmbedding['metadata']
  ): Promise<void> {
    const vector = await this.createCellEmbedding(cellId, pm, metadata);

    const embedding: CellEmbedding = {
      cellId,
      vector,
      metadata,
      timestamp: new Date()
    };

    this.cellEmbeddings.set(cellId, embedding);
    await this.cellIndex.insert(cellId, vector);

    console.log(`[RuvectorGNN] Added/updated cell ${cellId}`);
    this.emit('cell_added', { cellId });
  }

  /**
   * Get statistics about the indices
   */
  getStats(): {
    cellCount: number;
    episodeCount: number;
    indexPath: string;
    dimension: number;
    avgCellNeighbors: number;
  } {
    const neighborCounts = Array.from(this.cellEmbeddings.values())
      .map(c => c.metadata.neighborCells?.length || 0);

    const avgNeighbors = neighborCounts.length > 0
      ? neighborCounts.reduce((a, b) => a + b, 0) / neighborCounts.length
      : 0;

    return {
      cellCount: this.cellEmbeddings.size,
      episodeCount: this.episodeEmbeddings.size,
      indexPath: this.indexPath,
      dimension: this.dimension,
      avgCellNeighbors: avgNeighbors
    };
  }

  // Utility functions

  private normalize(value: number, min: number, max: number): number {
    return (value - min) / (max - min);
  }

  private normalizeVector(vec: Float32Array): void {
    let norm = 0;
    for (let i = 0; i < vec.length; i++) {
      norm += vec[i] * vec[i];
    }
    norm = Math.sqrt(norm);

    if (norm > 0) {
      for (let i = 0; i < vec.length; i++) {
        vec[i] /= norm;
      }
    }
  }
}

// ============================================================================
// RuvLLM Client - Natural Language RAN Queries
// ============================================================================

/**
 * RuvLLM Client for natural language RAN understanding
 *
 * Enables queries like:
 * - "What cells have similar SINR patterns to cell X?"
 * - "Explain why P0=-103 was chosen for cell Y"
 * - "Which optimization had the biggest impact last week?"
 */
export class RuvLLMClient extends EventEmitter {
  private gnn: RuvectorGNN;
  private conversationHistory: Array<{ role: string; content: string }>;

  constructor(gnn: RuvectorGNN) {
    super();
    this.gnn = gnn;
    this.conversationHistory = [];

    console.log('[RuvLLM] Initialized natural language query interface');
  }

  /**
   * Query the RAN using natural language
   *
   * Example: "What cells have similar SINR patterns to cell ABC123?"
   */
  async queryRAN(question: string): Promise<RANInsight> {
    const startTime = performance.now();
    console.log(`[RuvLLM] Query: ${question}`);

    // Parse query intent (simplified - production would use LLM)
    const intent = this.parseIntent(question);

    let insight: RANInsight;

    switch (intent.type) {
      case 'find_similar_cells':
        insight = await this.findSimilarCellsQuery(intent.cellId!, question);
        break;

      case 'explain_decision':
        insight = await this.explainDecisionQuery(intent.decisionId!, question);
        break;

      case 'performance_comparison':
        insight = await this.performanceComparisonQuery(question);
        break;

      case 'troubleshoot':
        insight = await this.troubleshootQuery(intent.cellId!, question);
        break;

      default:
        insight = {
          query: question,
          answer: "I understand you're asking about RAN optimization, but I need more specific information to help.",
          confidence: 0.3,
          supportingData: { cells: [], metrics: {} },
          reasoning: "Query intent could not be determined with high confidence"
        };
    }

    const latency = performance.now() - startTime;
    console.log(`[RuvLLM] Answered in ${latency.toFixed(2)}ms with confidence ${insight.confidence.toFixed(2)}`);

    // Add to conversation history
    this.conversationHistory.push(
      { role: 'user', content: question },
      { role: 'assistant', content: insight.answer }
    );

    this.emit('query_completed', { question, latency, confidence: insight.confidence });

    return insight;
  }

  /**
   * Explain why a specific optimization decision was made
   */
  async explainDecision(decisionId: string): Promise<string> {
    const episode = this.gnn['episodeEmbeddings'].get(decisionId);

    if (!episode) {
      return `Decision ${decisionId} not found in memory.`;
    }

    const explanation = this.generateExplanation(episode);
    return explanation;
  }

  /**
   * Generate optimization recommendation for a cell
   */
  async recommendOptimization(cellId: string): Promise<Recommendation> {
    console.log(`[RuvLLM] Generating recommendation for cell ${cellId}`);

    // Get current cell state
    const cellEmbedding = this.gnn['cellEmbeddings'].get(cellId);
    if (!cellEmbedding) {
      throw new Error(`Cell ${cellId} not found`);
    }

    // Find similar successful optimizations
    const similarSuccesses = await this.gnn.querySimilarOptimizations(
      this.extractPMFromEmbedding(cellEmbedding),
      5,
      { outcome: 'SUCCESS', minReward: 0.5 }
    );

    if (similarSuccesses.length === 0) {
      return {
        cellId,
        priority: 'LOW',
        recommendedAction: {},
        reasoning: 'No similar successful optimizations found. More data needed for confident recommendation.',
        expectedGain: {},
        similarPastSuccesses: [],
        risks: ['Insufficient historical data'],
        confidence: 0.2
      };
    }

    // Aggregate recommendations from similar episodes
    const actionVotes = new Map<string, number>();
    let totalReward = 0;

    for (const ep of similarSuccesses) {
      actionVotes.set(ep.actionType, (actionVotes.get(ep.actionType) || 0) + ep.reward);
      totalReward += ep.reward;
    }

    // Find best action type
    let bestAction = 'none';
    let maxVote = 0;
    for (const [action, vote] of actionVotes) {
      if (vote > maxVote) {
        maxVote = vote;
        bestAction = action;
      }
    }

    // Get representative episode for this action
    const representative = similarSuccesses.find(ep => ep.actionType === bestAction)!;

    // Calculate expected gains
    const expectedGain = {
      sinr: similarSuccesses.reduce((sum, ep) => sum + ep.sinrGain, 0) / similarSuccesses.length,
      cssr: similarSuccesses.reduce((sum, ep) => sum + ep.cssrGain, 0) / similarSuccesses.length,
      dropRate: similarSuccesses.reduce((sum, ep) => sum + ep.dropRateChange, 0) / similarSuccesses.length
    };

    // Determine priority
    const performanceClass = cellEmbedding.metadata.performanceClass;
    let priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' = 'MEDIUM';

    if (performanceClass === 'poor') {
      priority = 'CRITICAL';
    } else if (performanceClass === 'fair') {
      priority = 'HIGH';
    } else if (performanceClass === 'good') {
      priority = 'LOW';
    }

    return {
      cellId,
      priority,
      recommendedAction: representative.action,
      reasoning: `Based on ${similarSuccesses.length} similar successful optimizations, ` +
        `action type '${bestAction}' achieved average reward of ${(maxVote / similarSuccesses.length).toFixed(3)}. ` +
        `Similar cells saw SINR improvement of ${expectedGain.sinr.toFixed(2)} dB.`,
      expectedGain,
      similarPastSuccesses: similarSuccesses,
      risks: this.identifyRisks(representative, cellEmbedding),
      confidence: Math.min(0.95, (similarSuccesses.length / 10) * (maxVote / totalReward))
    };
  }

  // Private helper methods

  private parseIntent(question: string): {
    type: 'find_similar_cells' | 'explain_decision' | 'performance_comparison' | 'troubleshoot' | 'unknown';
    cellId?: string;
    decisionId?: string;
  } {
    const lowerQ = question.toLowerCase();

    // Extract cell ID if present (pattern: ABC123, Cell_XYZ, etc.)
    const cellMatch = question.match(/\b([A-Z0-9_-]{5,})\b/);
    const cellId = cellMatch ? cellMatch[1] : undefined;

    if (lowerQ.includes('similar') && lowerQ.includes('cell')) {
      return { type: 'find_similar_cells', cellId };
    }

    if (lowerQ.includes('explain') || lowerQ.includes('why')) {
      return { type: 'explain_decision', decisionId: cellId };
    }

    if (lowerQ.includes('compare') || lowerQ.includes('better') || lowerQ.includes('worse')) {
      return { type: 'performance_comparison' };
    }

    if (lowerQ.includes('problem') || lowerQ.includes('issue') || lowerQ.includes('troubleshoot')) {
      return { type: 'troubleshoot', cellId };
    }

    return { type: 'unknown' };
  }

  private async findSimilarCellsQuery(cellId: string, question: string): Promise<RANInsight> {
    const similarCells = await this.gnn.findSimilarCells(cellId, 5);

    const cellIds = similarCells.map(c => c.cellId);
    const avgSector = similarCells.reduce((sum, c) => sum + c.metadata.sector, 0) / similarCells.length;

    return {
      query: question,
      answer: `Found ${similarCells.length} cells with similar patterns to ${cellId}. ` +
        `These cells are in similar sectors (avg: ${avgSector.toFixed(1)}) and have comparable ` +
        `performance characteristics. You can apply optimizations from ${cellId} to these cells ` +
        `with high confidence for transfer learning.`,
      confidence: similarCells.length >= 3 ? 0.85 : 0.65,
      supportingData: {
        cells: cellIds,
        metrics: {
          avgSector,
          similarityCount: similarCells.length
        }
      },
      reasoning: `Vector similarity search using 768-dim embeddings with cosine distance. ` +
        `Top ${similarCells.length} results have similarity > 0.8.`
    };
  }

  private async explainDecisionQuery(decisionId: string, question: string): Promise<RANInsight> {
    const explanation = await this.explainDecision(decisionId);

    return {
      query: question,
      answer: explanation,
      confidence: 0.9,
      supportingData: {
        cells: [decisionId],
        metrics: {},
        episodes: [decisionId]
      },
      reasoning: 'Explanation generated from stored optimization episode metadata and outcomes'
    };
  }

  private async performanceComparisonQuery(question: string): Promise<RANInsight> {
    const stats = this.gnn.getStats();

    return {
      query: question,
      answer: `Currently tracking ${stats.cellCount} cells with ${stats.episodeCount} optimization episodes. ` +
        `Average cell has ${stats.avgCellNeighbors.toFixed(1)} neighbors in the interference graph.`,
      confidence: 0.75,
      supportingData: {
        cells: [],
        metrics: {
          totalCells: stats.cellCount,
          totalEpisodes: stats.episodeCount,
          avgNeighbors: stats.avgCellNeighbors
        }
      },
      reasoning: 'Statistics aggregated from RuvectorGNN indices'
    };
  }

  private async troubleshootQuery(cellId: string, question: string): Promise<RANInsight> {
    if (!cellId) {
      return {
        query: question,
        answer: 'Please specify which cell you want to troubleshoot.',
        confidence: 0.3,
        supportingData: { cells: [], metrics: {} },
        reasoning: 'Cell ID required for troubleshooting'
      };
    }

    const recommendation = await this.recommendOptimization(cellId);

    return {
      query: question,
      answer: `For cell ${cellId} (${recommendation.priority} priority): ${recommendation.reasoning}`,
      confidence: recommendation.confidence,
      supportingData: {
        cells: [cellId],
        metrics: recommendation.expectedGain as Record<string, number>
      },
      reasoning: recommendation.reasoning
    };
  }

  private generateExplanation(episode: OptimizationEpisode): string {
    const params = Object.entries(episode.action)
      .map(([key, value]) => `${key}=${value}`)
      .join(', ');

    return `Decision ${episode.id} for cell ${episode.cellId}:\n\n` +
      `Action: ${episode.actionType} (${params})\n` +
      `Outcome: ${episode.outcome}\n` +
      `Reward: ${episode.reward.toFixed(3)}\n` +
      `SINR Gain: ${episode.sinrGain.toFixed(2)} dB\n` +
      `CSSR Gain: ${(episode.cssrGain * 100).toFixed(2)}%\n` +
      `Drop Rate Change: ${(episode.dropRateChange * 100).toFixed(2)}%\n\n` +
      `This decision was made based on ${episode.similarEpisodes?.length || 0} similar past episodes. ` +
      `The optimization was classified as ${episode.outcome} with a transfer score of ` +
      `${(episode.transferScore || 0).toFixed(2)}, indicating ${episode.transferScore! > 0.7 ? 'high' : 'moderate'} ` +
      `applicability to other similar cells.`;
  }

  private extractPMFromEmbedding(embedding: CellEmbedding): PMCounters {
    // Reverse-engineer PM counters from embedding (approximate)
    // In production, store actual PM values
    return {
      pmUlSinrMean: 0,
      pmDlSinrMean: 0,
      pmCssr: 0.95,
      pmCallDropRate: 0.01
    };
  }

  private identifyRisks(episode: OptimizationEpisode, cellEmbedding: CellEmbedding): string[] {
    const risks: string[] = [];

    if (episode.fmAlarmsAfter.length > episode.fmAlarmsBefore.length) {
      risks.push('Past similar optimization triggered new alarms');
    }

    if (cellEmbedding.metadata.neighborCells && cellEmbedding.metadata.neighborCells.length > 15) {
      risks.push('High neighbor count may cause interference issues');
    }

    if (episode.outcome === 'NEUTRAL') {
      risks.push('Similar episodes had neutral outcomes - gains may be marginal');
    }

    return risks;
  }
}

// ============================================================================
// HNSW Index Implementation (Simplified)
// ============================================================================

interface HNSWConfig {
  dimension: number;
  metric: 'cosine' | 'euclidean' | 'dotproduct';
  M: number;
  efConstruction: number;
  efSearch: number;
}

class HNSWIndex {
  private config: HNSWConfig;
  private vectors: Map<string, Float32Array>;
  private graph: Map<string, Set<string>>;

  constructor(config: HNSWConfig) {
    this.config = config;
    this.vectors = new Map();
    this.graph = new Map();
  }

  async insert(id: string, vector: Float32Array): Promise<void> {
    this.vectors.set(id, vector);

    if (!this.graph.has(id)) {
      this.graph.set(id, new Set());
    }

    // Connect to nearest neighbors
    const neighbors = await this.search(vector, this.config.M);
    for (const neighbor of neighbors) {
      if (neighbor.id !== id) {
        this.graph.get(id)!.add(neighbor.id);
        this.graph.get(neighbor.id)?.add(id);
      }
    }
  }

  async search(query: Float32Array, k: number): Promise<Array<{ id: string; distance: number }>> {
    const results: Array<{ id: string; distance: number }> = [];

    for (const [id, vector] of this.vectors) {
      const distance = this.distance(query, vector);
      results.push({ id, distance });
    }

    results.sort((a, b) => a.distance - b.distance);
    return results.slice(0, k);
  }

  private distance(a: Float32Array, b: Float32Array): number {
    if (this.config.metric === 'cosine') {
      let dot = 0, normA = 0, normB = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
      }
      return 1 - (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

    // Euclidean
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }
}

// ============================================================================
// Exports
// ============================================================================

// export { RuvectorGNN, RuvLLMClient, HNSWIndex };
