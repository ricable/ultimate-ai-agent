/**
 * AgentDB Reflexion Memory Integration
 *
 * Integrates RuvVector embeddings with AgentDB for persistent reflexion memory.
 * Stores successful optimizations and enables transfer learning across cells
 * using vector similarity search.
 *
 * AgentDB provides:
 * - SQLite backend for persistent storage
 * - HNSW vector indices for fast similarity search
 * - Reflexion memory for learning from past episodes
 *
 * @module ml/agentdb-reflexion
 * @version 7.0.0-alpha.1
 */

import { EventEmitter } from 'events';
import type { OptimizationEpisode, CellEmbedding } from './ruvector-gnn.js';
import type { PMCounters, CMParameters } from '../learning/self-learner.js';

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Memory entry for AgentDB storage
 */
export interface MemoryEntry {
  id: string;
  type: 'optimization_episode' | 'cell_state' | 'failure' | 'success_pattern';
  embedding: number[];  // Vector embedding for similarity search
  metadata: Record<string, any>;
  timestamp: Date;
  tags?: string[];
  outcome?: 'success' | 'failure' | 'neutral';
}

/**
 * Query result from AgentDB
 */
export interface MemoryQueryResult {
  entry: MemoryEntry;
  similarity: number;
  distance: number;
}

/**
 * Transfer learning result
 */
export interface TransferLearningResult {
  sourceCellId: string;
  targetCellId: string;
  similarEpisodes: OptimizationEpisode[];
  recommendedAction: Partial<CMParameters>;
  confidence: number;
  expectedReward: number;
  transferScore: number;  // 0-1, how well this transfers
}

/**
 * Reflexion memory statistics
 */
export interface ReflexionStats {
  totalEpisodes: number;
  successfulEpisodes: number;
  failedEpisodes: number;
  avgReward: number;
  successRate: number;
  topActions: Array<{ action: string; count: number; avgReward: number }>;
  memoryUtilization: number;  // 0-1
}

/**
 * AgentDB configuration
 */
export interface AgentDBConfig {
  dbPath: string;
  maxMemorySize: number;
  embeddingDim: number;
  hnswM: number;
  hnswEfConstruction: number;
  hnswEfSearch: number;
}

// ============================================================================
// AgentDB Reflexion Memory Manager
// ============================================================================

/**
 * Manages reflexion memory using AgentDB for persistent storage
 * and RuvVector for fast similarity search.
 */
export class AgentDBReflexion extends EventEmitter {
  private config: AgentDBConfig;
  private memory: Map<string, MemoryEntry>;  // In-memory cache
  private episodeIndex: Map<string, OptimizationEpisode>;

  // Simulated AgentDB interface (in production, use actual agentdb@alpha CLI)
  private db: AgentDBInterface;

  constructor(config?: Partial<AgentDBConfig>) {
    super();

    this.config = {
      dbPath: config?.dbPath || './titan-ran.db',
      maxMemorySize: config?.maxMemorySize || 100000,
      embeddingDim: config?.embeddingDim || 768,
      hnswM: config?.hnswM || 32,
      hnswEfConstruction: config?.hnswEfConstruction || 200,
      hnswEfSearch: config?.hnswEfSearch || 100
    };

    this.memory = new Map();
    this.episodeIndex = new Map();

    this.db = new AgentDBInterface(this.config);

    console.log('[AgentDB] Initialized reflexion memory');
    console.log(`[AgentDB] Database: ${this.config.dbPath}`);
  }

  /**
   * Initialize the AgentDB connection and load memory
   */
  async initialize(): Promise<void> {
    console.log('[AgentDB] Connecting to database...');

    await this.db.connect();
    await this.db.createSchema();
    await this.loadMemoryFromDB();

    this.emit('initialized');
    console.log('[AgentDB] Initialization complete');
  }

  /**
   * Store a successful optimization episode in reflexion memory
   *
   * This is the core learning mechanism: save what worked so we can
   * apply it to similar situations in the future.
   */
  async storeOptimization(episode: OptimizationEpisode): Promise<void> {
    const startTime = performance.now();

    // Convert to memory entry
    const entry: MemoryEntry = {
      id: episode.id,
      type: 'optimization_episode',
      embedding: Array.from(episode.stateEmbedding),
      metadata: {
        cellId: episode.cellId,
        actionType: episode.actionType,
        action: episode.action,
        reward: episode.reward,
        sinrGain: episode.sinrGain,
        cssrGain: episode.cssrGain,
        dropRateChange: episode.dropRateChange,
        pmBefore: episode.pmBefore,
        pmAfter: episode.pmAfter,
        cmBefore: episode.cmBefore,
        cmAfter: episode.cmAfter
      },
      timestamp: episode.timestamp,
      tags: [
        episode.actionType,
        `outcome:${episode.outcome}`,
        `reward:${episode.reward > 0.5 ? 'high' : episode.reward > 0 ? 'medium' : 'low'}`
      ],
      outcome: episode.outcome.toLowerCase() as 'success' | 'failure' | 'neutral'
    };

    // Store in memory cache
    this.memory.set(entry.id, entry);
    this.episodeIndex.set(episode.id, episode);

    // Persist to AgentDB
    await this.db.insert(entry);

    const latency = performance.now() - startTime;
    console.log(`[AgentDB] Stored episode ${episode.id} (${episode.outcome}) in ${latency.toFixed(2)}ms`);

    this.emit('episode_stored', {
      episodeId: episode.id,
      outcome: episode.outcome,
      reward: episode.reward,
      latency
    });

    // Prune old memories if needed
    if (this.memory.size > this.config.maxMemorySize) {
      await this.pruneMemory();
    }
  }

  /**
   * Query for similar successful optimizations (transfer learning)
   *
   * This is how we transfer knowledge from one cell to another:
   * "This cell is in a similar state to cells we optimized before.
   *  Let's try what worked for them."
   */
  async queryForTransferLearning(
    currentState: PMCounters,
    currentStateEmbedding: Float32Array,
    topK: number = 5,
    filters?: {
      outcome?: 'success' | 'failure' | 'neutral';
      minReward?: number;
      actionType?: string;
      maxAge?: number;  // days
    }
  ): Promise<MemoryQueryResult[]> {
    const startTime = performance.now();

    // Query AgentDB vector index
    const results = await this.db.similaritySearch(
      Array.from(currentStateEmbedding),
      topK * 2  // Get more candidates for filtering
    );

    // Apply filters
    let filtered = results;

    if (filters) {
      if (filters.outcome) {
        filtered = filtered.filter(r => r.entry.outcome === filters.outcome);
      }

      if (filters.minReward !== undefined) {
        filtered = filtered.filter(r =>
          r.entry.metadata.reward !== undefined &&
          r.entry.metadata.reward >= filters.minReward!
        );
      }

      if (filters.actionType) {
        filtered = filtered.filter(r => r.entry.metadata.actionType === filters.actionType);
      }

      if (filters.maxAge) {
        const maxDate = new Date(Date.now() - filters.maxAge * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(r => r.entry.timestamp >= maxDate);
      }
    }

    // Take top k after filtering
    filtered = filtered.slice(0, topK);

    const latency = performance.now() - startTime;
    console.log(`[AgentDB] Found ${filtered.length} similar episodes in ${latency.toFixed(2)}ms`);

    this.emit('transfer_learning_query', {
      resultsCount: filtered.length,
      latency,
      filters
    });

    return filtered;
  }

  /**
   * Get transfer learning recommendation for a specific cell
   */
  async recommendTransferLearning(
    sourceCellId: string,
    targetCellId: string,
    targetState: PMCounters,
    targetStateEmbedding: Float32Array
  ): Promise<TransferLearningResult> {
    console.log(`[AgentDB] Computing transfer learning: ${sourceCellId} -> ${targetCellId}`);

    // Query for similar successful episodes
    const similarMemories = await this.queryForTransferLearning(
      targetState,
      targetStateEmbedding,
      5,
      { outcome: 'success', minReward: 0.3, maxAge: 90 }
    );

    if (similarMemories.length === 0) {
      return {
        sourceCellId,
        targetCellId,
        similarEpisodes: [],
        recommendedAction: {},
        confidence: 0.1,
        expectedReward: 0,
        transferScore: 0
      };
    }

    // Extract episodes
    const similarEpisodes = similarMemories
      .map(m => this.episodeIndex.get(m.entry.id))
      .filter(ep => ep !== undefined) as OptimizationEpisode[];

    // Aggregate actions (vote)
    const actionVotes = new Map<string, { count: number; totalReward: number; action: Partial<CMParameters> }>();

    for (const ep of similarEpisodes) {
      const key = ep.actionType;
      if (!actionVotes.has(key)) {
        actionVotes.set(key, { count: 0, totalReward: 0, action: ep.action });
      }

      const vote = actionVotes.get(key)!;
      vote.count++;
      vote.totalReward += ep.reward;
    }

    // Find best action
    let bestAction: Partial<CMParameters> = {};
    let maxScore = 0;

    for (const [actionType, vote] of actionVotes) {
      const score = vote.totalReward / vote.count;
      if (score > maxScore) {
        maxScore = score;
        bestAction = vote.action;
      }
    }

    // Calculate expected reward
    const expectedReward = similarEpisodes.reduce((sum, ep) => sum + ep.reward, 0) / similarEpisodes.length;

    // Calculate transfer score based on similarity
    const avgSimilarity = similarMemories.reduce((sum, m) => sum + m.similarity, 0) / similarMemories.length;
    const transferScore = avgSimilarity * Math.min(1, similarEpisodes.length / 5);

    const confidence = Math.min(0.95, transferScore * (1 + expectedReward));

    console.log(`[AgentDB] Transfer learning: confidence=${confidence.toFixed(2)}, expectedReward=${expectedReward.toFixed(3)}`);

    return {
      sourceCellId,
      targetCellId,
      similarEpisodes,
      recommendedAction: bestAction,
      confidence,
      expectedReward,
      transferScore
    };
  }

  /**
   * Get reflexion statistics
   */
  async getReflexionStats(): Promise<ReflexionStats> {
    const episodes = Array.from(this.episodeIndex.values());

    const successful = episodes.filter(e => e.outcome === 'SUCCESS').length;
    const failed = episodes.filter(e => e.outcome === 'FAILURE').length;

    const totalReward = episodes.reduce((sum, e) => sum + e.reward, 0);
    const avgReward = episodes.length > 0 ? totalReward / episodes.length : 0;

    const successRate = episodes.length > 0 ? successful / episodes.length : 0;

    // Count actions
    const actionCounts = new Map<string, { count: number; totalReward: number }>();

    for (const ep of episodes) {
      if (!actionCounts.has(ep.actionType)) {
        actionCounts.set(ep.actionType, { count: 0, totalReward: 0 });
      }

      const ac = actionCounts.get(ep.actionType)!;
      ac.count++;
      ac.totalReward += ep.reward;
    }

    const topActions = Array.from(actionCounts.entries())
      .map(([action, data]) => ({
        action,
        count: data.count,
        avgReward: data.totalReward / data.count
      }))
      .sort((a, b) => b.avgReward - a.avgReward)
      .slice(0, 5);

    const memoryUtilization = this.memory.size / this.config.maxMemorySize;

    return {
      totalEpisodes: episodes.length,
      successfulEpisodes: successful,
      failedEpisodes: failed,
      avgReward,
      successRate,
      topActions,
      memoryUtilization
    };
  }

  /**
   * Clear all reflexion memory (dangerous!)
   */
  async clearMemory(): Promise<void> {
    console.log('[AgentDB] WARNING: Clearing all reflexion memory');

    await this.db.clear();
    this.memory.clear();
    this.episodeIndex.clear();

    this.emit('memory_cleared');
  }

  /**
   * Export memory to JSON for backup
   */
  async exportMemory(): Promise<string> {
    const entries = Array.from(this.memory.values());
    return JSON.stringify(entries, null, 2);
  }

  /**
   * Import memory from JSON backup
   */
  async importMemory(json: string): Promise<void> {
    const entries: MemoryEntry[] = JSON.parse(json);

    console.log(`[AgentDB] Importing ${entries.length} memory entries`);

    for (const entry of entries) {
      this.memory.set(entry.id, entry);
      await this.db.insert(entry);
    }

    console.log('[AgentDB] Import complete');
    this.emit('memory_imported', { count: entries.length });
  }

  // Private methods

  private async loadMemoryFromDB(): Promise<void> {
    console.log('[AgentDB] Loading memory from database...');

    const entries = await this.db.loadAll();

    for (const entry of entries) {
      this.memory.set(entry.id, entry);

      // Reconstruct episode if type is optimization_episode
      if (entry.type === 'optimization_episode') {
        const episode: OptimizationEpisode = {
          id: entry.id,
          cellId: entry.metadata.cellId,
          timestamp: entry.timestamp,
          pmBefore: entry.metadata.pmBefore,
          cmBefore: entry.metadata.cmBefore,
          fmAlarmsBefore: [],
          action: entry.metadata.action,
          actionType: entry.metadata.actionType,
          pmAfter: entry.metadata.pmAfter,
          cmAfter: entry.metadata.cmAfter,
          fmAlarmsAfter: [],
          reward: entry.metadata.reward,
          sinrGain: entry.metadata.sinrGain,
          cssrGain: entry.metadata.cssrGain,
          dropRateChange: entry.metadata.dropRateChange,
          outcome: entry.outcome?.toUpperCase() as 'SUCCESS' | 'FAILURE' | 'NEUTRAL',
          stateEmbedding: new Float32Array(entry.embedding),
          actionEmbedding: new Float32Array(768)
        };

        this.episodeIndex.set(episode.id, episode);
      }
    }

    console.log(`[AgentDB] Loaded ${entries.length} memory entries`);
  }

  private async pruneMemory(): Promise<void> {
    console.log('[AgentDB] Pruning old memories...');

    // Keep only successful episodes and recent failures
    const entries = Array.from(this.memory.values());
    const sortedByDate = entries.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    // Keep successful + top 20% of entries by date
    const toKeep = sortedByDate.slice(0, Math.floor(this.config.maxMemorySize * 0.8));

    // Always keep successful episodes
    const successful = entries.filter(e => e.outcome === 'success');
    for (const entry of successful) {
      if (!toKeep.includes(entry)) {
        toKeep.push(entry);
      }
    }

    // Remove pruned entries
    const toRemove = entries.filter(e => !toKeep.includes(e));

    for (const entry of toRemove) {
      this.memory.delete(entry.id);
      this.episodeIndex.delete(entry.id);
      await this.db.delete(entry.id);
    }

    console.log(`[AgentDB] Pruned ${toRemove.length} entries`);
    this.emit('memory_pruned', { removedCount: toRemove.length });
  }
}

// ============================================================================
// AgentDB Interface (Simulated)
// ============================================================================

/**
 * Simulated AgentDB interface
 * In production, this would call: npx agentdb@alpha <command>
 */
class AgentDBInterface {
  private config: AgentDBConfig;
  private storage: Map<string, MemoryEntry>;
  private vectorIndex: Map<string, number[]>;

  constructor(config: AgentDBConfig) {
    this.config = config;
    this.storage = new Map();
    this.vectorIndex = new Map();
  }

  async connect(): Promise<void> {
    console.log(`[AgentDB Interface] Connecting to ${this.config.dbPath}`);
    // In production: exec('npx agentdb@alpha connect --db ' + this.config.dbPath)
  }

  async createSchema(): Promise<void> {
    console.log('[AgentDB Interface] Creating schema');
    // In production: exec('npx agentdb@alpha migrate --latest')
  }

  async insert(entry: MemoryEntry): Promise<void> {
    this.storage.set(entry.id, entry);
    this.vectorIndex.set(entry.id, entry.embedding);

    // In production: exec('npx agentdb@alpha insert ...')
  }

  async similaritySearch(query: number[], k: number): Promise<MemoryQueryResult[]> {
    // Simple cosine similarity search
    const results: MemoryQueryResult[] = [];

    for (const [id, embedding] of this.vectorIndex) {
      const similarity = this.cosineSimilarity(query, embedding);
      const distance = 1 - similarity;

      const entry = this.storage.get(id);
      if (entry) {
        results.push({ entry, similarity, distance });
      }
    }

    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, k);

    // In production: exec('npx agentdb@alpha search --vector [...] --topk ' + k)
  }

  async loadAll(): Promise<MemoryEntry[]> {
    return Array.from(this.storage.values());

    // In production: exec('npx agentdb@alpha export --json')
  }

  async delete(id: string): Promise<void> {
    this.storage.delete(id);
    this.vectorIndex.delete(id);

    // In production: exec('npx agentdb@alpha delete --id ' + id)
  }

  async clear(): Promise<void> {
    this.storage.clear();
    this.vectorIndex.clear();

    // In production: exec('npx agentdb@alpha clear --confirm')
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0, normA = 0, normB = 0;

    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

// ============================================================================
// Exports
// ============================================================================

// export {
//   AgentDBReflexion,
//   type MemoryEntry,
//   type MemoryQueryResult,
//   type TransferLearningResult,
//   type ReflexionStats,
//   type AgentDBConfig
// };
