/**
 * ReflexionMemory - Episodic Replay Memory System
 *
 * Implements reflexion-style episodic replay for agent self-improvement.
 * Stores self-critiques and outcomes, retrieves relevant past experiences.
 *
 * Based on: "Reflexion: Language Agents with Verbal Reinforcement Learning"
 * https://arxiv.org/abs/2303.11366
 */

// Database type from db-fallback
type Database = any;
import { EmbeddingService } from './EmbeddingService.js';
import type { VectorBackend } from '../backends/VectorBackend.js';
import type { LearningBackend } from '../backends/LearningBackend.js';
import type { GraphBackend, GraphNode } from '../backends/GraphBackend.js';
import type { GraphDatabaseAdapter } from '../backends/graph/GraphDatabaseAdapter.js';
import { NodeIdMapper } from '../utils/NodeIdMapper.js';

export interface Episode {
  id?: number;
  ts?: number;
  sessionId: string;
  task: string;
  input?: string;
  output?: string;
  critique?: string;
  reward: number;
  success: boolean;
  latencyMs?: number;
  tokensUsed?: number;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface EpisodeWithEmbedding extends Episode {
  embedding?: Float32Array;
  similarity?: number;
}

export interface ReflexionQuery {
  task: string;
  currentState?: string;
  k?: number; // Top-k to retrieve
  minReward?: number;
  onlyFailures?: boolean;
  onlySuccesses?: boolean;
  timeWindowDays?: number;
}

export class ReflexionMemory {
  private db: Database;
  private embedder: EmbeddingService;
  private vectorBackend?: VectorBackend;
  private learningBackend?: LearningBackend;
  private graphBackend?: GraphBackend;

  constructor(
    db: Database,
    embedder: EmbeddingService,
    vectorBackend?: VectorBackend,
    learningBackend?: LearningBackend,
    graphBackend?: GraphBackend
  ) {
    this.db = db;
    this.embedder = embedder;
    this.vectorBackend = vectorBackend;
    this.learningBackend = learningBackend;
    this.graphBackend = graphBackend;
  }

  /**
   * Store a new episode with its critique and outcome
   */
  async storeEpisode(episode: Episode): Promise<number> {
    // Use GraphDatabaseAdapter if available (AgentDB v2)
    if (this.graphBackend && 'storeEpisode' in this.graphBackend) {
      // GraphDatabaseAdapter has specialized storeEpisode method
      const graphAdapter = this.graphBackend as any as GraphDatabaseAdapter;

      // Generate embedding for the task
      const taskEmbedding = await this.embedder.embed(episode.task);

      // Create episode node using GraphDatabaseAdapter
      const nodeId = await graphAdapter.storeEpisode({
        id: episode.id ? `episode-${episode.id}` : `episode-${Date.now()}-${Math.random()}`,
        sessionId: episode.sessionId,
        task: episode.task,
        reward: episode.reward,
        success: episode.success,
        input: episode.input,
        output: episode.output,
        critique: episode.critique,
        createdAt: episode.ts ? episode.ts * 1000 : Date.now(),
        tokensUsed: episode.tokensUsed,
        latencyMs: episode.latencyMs
      }, taskEmbedding);

      // Return a numeric ID (parse from string ID)
      const numericId = parseInt(nodeId.split('-').pop() || '0', 36);

      // Register mapping for later use by CausalMemoryGraph
      NodeIdMapper.getInstance().register(numericId, nodeId);

      return numericId;
    }

    // Use generic GraphBackend if available
    if (this.graphBackend) {
      // Generate embedding for the task
      const taskEmbedding = await this.embedder.embed(episode.task);

      // Create episode node ID
      const nodeId = await this.graphBackend.createNode(
        ['Episode'],
        {
          sessionId: episode.sessionId,
          task: episode.task,
          input: episode.input || '',
          output: episode.output || '',
          critique: episode.critique || '',
          reward: episode.reward,
          success: episode.success,
          latencyMs: episode.latencyMs || 0,
          tokensUsed: episode.tokensUsed || 0,
          tags: episode.tags ? JSON.stringify(episode.tags) : '[]',
          metadata: episode.metadata ? JSON.stringify(episode.metadata) : '{}',
          createdAt: Date.now()
        }
      );

      // Store embedding using vectorBackend if available
      if (this.vectorBackend && taskEmbedding) {
        this.vectorBackend.insert(nodeId, taskEmbedding, {
          type: 'episode',
          sessionId: episode.sessionId
        });
      }

      // Return a numeric ID (parse from string ID)
      const numericId = parseInt(nodeId.split('-').pop() || '0', 36);

      // Register mapping for later use by CausalMemoryGraph
      NodeIdMapper.getInstance().register(numericId, nodeId);

      return numericId;
    }

    // Fallback to SQLite (v1 compatibility)
    const stmt = this.db.prepare(`
      INSERT INTO episodes (
        session_id, task, input, output, critique, reward, success,
        latency_ms, tokens_used, tags, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const tags = episode.tags ? JSON.stringify(episode.tags) : null;
    const metadata = episode.metadata ? JSON.stringify(episode.metadata) : null;

    const result = stmt.run(
      episode.sessionId,
      episode.task,
      episode.input || null,
      episode.output || null,
      episode.critique || null,
      episode.reward,
      episode.success ? 1 : 0,
      episode.latencyMs || null,
      episode.tokensUsed || null,
      tags,
      metadata
    );

    const episodeId = result.lastInsertRowid as number;

    // Generate and store embedding
    const text = this.buildEpisodeText(episode);
    const embedding = await this.embedder.embed(text);

    // Use vector backend if available (150x faster retrieval)
    if (this.vectorBackend) {
      this.vectorBackend.insert(episodeId.toString(), embedding);
    }

    // Also store in SQL for fallback
    this.storeEmbedding(episodeId, embedding);

    // Create graph node for episode if graph backend available
    if (this.graphBackend) {
      await this.createEpisodeGraphNode(episodeId, episode, embedding);
    }

    // Add training sample if learning backend available
    if (this.learningBackend && episode.success !== undefined) {
      this.learningBackend.addSample({
        embedding,
        label: episode.success ? 1 : 0,
        weight: Math.abs(episode.reward),
        context: {
          task: episode.task,
          sessionId: episode.sessionId,
          latencyMs: episode.latencyMs,
          tokensUsed: episode.tokensUsed
        }
      });
    }

    return episodeId;
  }

  /**
   * Retrieve relevant past episodes for a new task attempt
   */
  async retrieveRelevant(query: ReflexionQuery): Promise<EpisodeWithEmbedding[]> {
    const {
      task,
      currentState = '',
      k = 5,
      minReward,
      onlyFailures = false,
      onlySuccesses = false,
      timeWindowDays
    } = query;

    // Generate query embedding
    const queryText = currentState ? `${task}\n${currentState}` : task;
    let queryEmbedding = await this.embedder.embed(queryText);

    // Enhance query with GNN if learning backend available
    if (this.learningBackend) {
      queryEmbedding = await this.enhanceQueryWithGNN(queryEmbedding, k);
    }

    // Use GraphDatabaseAdapter if available (AgentDB v2)
    if (this.graphBackend && 'searchSimilarEpisodes' in this.graphBackend) {
      const graphAdapter = this.graphBackend as any as GraphDatabaseAdapter;

      // Search using vector similarity
      const results = await graphAdapter.searchSimilarEpisodes(queryEmbedding, k * 3);

      // Apply filters
      const filtered = results.filter((ep: any) => {
        if (minReward !== undefined && ep.reward < minReward) return false;
        if (onlyFailures && ep.success) return false;
        if (onlySuccesses && !ep.success) return false;
        if (timeWindowDays && ep.createdAt < (Date.now() - timeWindowDays * 86400000)) return false;
        return true;
      });

      // Convert to EpisodeWithEmbedding format
      const episodes: EpisodeWithEmbedding[] = filtered.slice(0, k).map((ep: any) => ({
        id: parseInt(ep.id.split('-').pop() || '0', 36),
        sessionId: ep.sessionId,
        task: ep.task,
        input: ep.input,
        output: ep.output,
        critique: ep.critique,
        reward: ep.reward,
        success: ep.success,
        latencyMs: ep.latencyMs,
        tokensUsed: ep.tokensUsed,
        ts: Math.floor(ep.createdAt / 1000)
      }));

      return episodes;
    }

    // Use generic GraphBackend if available
    if (this.graphBackend && 'execute' in this.graphBackend) {
      // Build Cypher query with filters
      let cypherQuery = 'MATCH (e:Episode) WHERE 1=1';

      if (minReward !== undefined) {
        cypherQuery += ` AND e.reward >= ${minReward}`;
      }
      if (onlyFailures) {
        cypherQuery += ` AND e.success = false`;
      }
      if (onlySuccesses) {
        cypherQuery += ` AND e.success = true`;
      }
      if (timeWindowDays) {
        const cutoff = Date.now() - timeWindowDays * 86400000;
        cypherQuery += ` AND e.createdAt >= ${cutoff}`;
      }

      cypherQuery += ` RETURN e LIMIT ${k * 3}`;

      const result = await this.graphBackend.execute(cypherQuery);

      // Convert to EpisodeWithEmbedding format
      const episodes: EpisodeWithEmbedding[] = result.rows.map((row: any) => {
        const node = row.e;
        return {
          id: parseInt(node.id.split('-').pop() || '0', 36),
          sessionId: node.properties.sessionId,
          task: node.properties.task,
          input: node.properties.input,
          output: node.properties.output,
          critique: node.properties.critique,
          reward: typeof node.properties.reward === 'string' ? parseFloat(node.properties.reward) : node.properties.reward,
          success: typeof node.properties.success === 'string' ? node.properties.success === 'true' : node.properties.success,
          latencyMs: node.properties.latencyMs,
          tokensUsed: node.properties.tokensUsed,
          tags: node.properties.tags ? JSON.parse(node.properties.tags) : [],
          metadata: node.properties.metadata ? JSON.parse(node.properties.metadata) : {},
          ts: Math.floor(node.properties.createdAt / 1000)
        };
      });

      return episodes.slice(0, k);
    }

    // Use optimized vector backend if available (150x faster)
    if (this.vectorBackend) {
      // Get candidates from vector backend
      const searchResults = this.vectorBackend.search(queryEmbedding, k * 3, {
        threshold: 0.0
      });

      // Fetch full episode data from DB
      const episodeIds = searchResults.map(r => parseInt(r.id));
      if (episodeIds.length === 0) {
        return [];
      }

      const placeholders = episodeIds.map(() => '?').join(',');
      const stmt = this.db.prepare(`
        SELECT * FROM episodes
        WHERE id IN (${placeholders})
      `);

      const rows = stmt.all(...episodeIds) as any[];
      const episodeMap = new Map(rows.map(r => [r.id.toString(), r]));

      // Map results back with similarity scores and apply filters
      const episodes: EpisodeWithEmbedding[] = [];

      for (const result of searchResults) {
        const row = episodeMap.get(result.id);
        if (!row) continue;

        // Apply additional filters
        if (minReward !== undefined && row.reward < minReward) continue;
        if (onlyFailures && row.success === 1) continue;
        if (onlySuccesses && row.success === 0) continue;
        if (timeWindowDays && row.ts < (Date.now() / 1000 - timeWindowDays * 86400)) continue;

        episodes.push({
          id: row.id,
          ts: row.ts,
          sessionId: row.session_id,
          task: row.task,
          input: row.input,
          output: row.output,
          critique: row.critique,
          reward: row.reward,
          success: row.success === 1,
          latencyMs: row.latency_ms,
          tokensUsed: row.tokens_used,
          tags: row.tags ? JSON.parse(row.tags) : undefined,
          metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
          similarity: result.similarity
        });

        if (episodes.length >= k) break;
      }

      return episodes;
    }

    // Fallback to SQL-based similarity search
    const filters: string[] = [];
    const params: any[] = [];

    if (minReward !== undefined) {
      filters.push('e.reward >= ?');
      params.push(minReward);
    }

    if (onlyFailures) {
      filters.push('e.success = 0');
    }

    if (onlySuccesses) {
      filters.push('e.success = 1');
    }

    if (timeWindowDays) {
      filters.push('e.ts > strftime("%s", "now") - ?');
      params.push(timeWindowDays * 86400);
    }

    const whereClause = filters.length > 0 ? `WHERE ${filters.join(' AND ')}` : '';

    const stmt = this.db.prepare(`
      SELECT
        e.*,
        ee.embedding
      FROM episodes e
      JOIN episode_embeddings ee ON e.id = ee.episode_id
      ${whereClause}
      ORDER BY e.reward DESC
    `);

    const rows = stmt.all(...params) as any[];

    // Calculate similarities manually
    const episodes: EpisodeWithEmbedding[] = rows.map(row => {
      const embedding = this.deserializeEmbedding(row.embedding);
      const similarity = this.cosineSimilarity(queryEmbedding, embedding);

      return {
        id: row.id,
        ts: row.ts,
        sessionId: row.session_id,
        task: row.task,
        input: row.input,
        output: row.output,
        critique: row.critique,
        reward: row.reward,
        success: row.success === 1,
        latencyMs: row.latency_ms,
        tokensUsed: row.tokens_used,
        tags: row.tags ? JSON.parse(row.tags) : undefined,
        metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
        embedding,
        similarity
      };
    });

    // Sort by similarity and return top-k
    episodes.sort((a, b) => (b.similarity || 0) - (a.similarity || 0));
    return episodes.slice(0, k);
  }

  /**
   * Get statistics for a task
   */
  getTaskStats(task: string, timeWindowDays?: number): {
    totalAttempts: number;
    successRate: number;
    avgReward: number;
    avgLatency: number;
    improvementTrend: number;
  } {
    const windowFilter = timeWindowDays
      ? `AND ts > strftime('%s', 'now') - ${timeWindowDays * 86400}`
      : '';

    const stmt = this.db.prepare(`
      SELECT
        COUNT(*) as total,
        AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
        AVG(reward) as avg_reward,
        AVG(latency_ms) as avg_latency
      FROM episodes
      WHERE task = ? ${windowFilter}
    `);

    const stats = stmt.get(task) as any;

    // Calculate improvement trend (recent vs older)
    const trendStmt = this.db.prepare(`
      SELECT
        AVG(CASE
          WHEN ts > strftime('%s', 'now') - ${7 * 86400} THEN reward
        END) as recent_reward,
        AVG(CASE
          WHEN ts <= strftime('%s', 'now') - ${7 * 86400} THEN reward
        END) as older_reward
      FROM episodes
      WHERE task = ? ${windowFilter}
    `);

    const trend = trendStmt.get(task) as any;
    const improvementTrend = trend.recent_reward && trend.older_reward
      ? (trend.recent_reward - trend.older_reward) / trend.older_reward
      : 0;

    return {
      totalAttempts: stats.total || 0,
      successRate: stats.success_rate || 0,
      avgReward: stats.avg_reward || 0,
      avgLatency: stats.avg_latency || 0,
      improvementTrend
    };
  }

  /**
   * Build critique summary from similar failed episodes
   */
  async getCritiqueSummary(query: ReflexionQuery): Promise<string> {
    const failures = await this.retrieveRelevant({
      ...query,
      onlyFailures: true,
      k: 3
    });

    if (failures.length === 0) {
      return 'No prior failures found for this task.';
    }

    const critiques = failures
      .filter(ep => ep.critique)
      .map((ep, i) => `${i + 1}. ${ep.critique} (reward: ${ep.reward.toFixed(2)})`)
      .join('\n');

    return `Prior failures and lessons learned:\n${critiques}`;
  }

  /**
   * Get successful strategies for a task
   */
  async getSuccessStrategies(query: ReflexionQuery): Promise<string> {
    const successes = await this.retrieveRelevant({
      ...query,
      onlySuccesses: true,
      minReward: 0.7,
      k: 3
    });

    if (successes.length === 0) {
      return 'No successful strategies found for this task.';
    }

    const strategies = successes
      .map((ep, i) => {
        const approach = ep.output?.substring(0, 200) || 'No output recorded';
        return `${i + 1}. Approach (reward ${ep.reward.toFixed(2)}): ${approach}...`;
      })
      .join('\n');

    return `Successful strategies:\n${strategies}`;
  }

  /**
   * Get recent episodes for a session
   */
  async getRecentEpisodes(sessionId: string, limit: number = 10): Promise<Episode[]> {
    const stmt = this.db.prepare(`
      SELECT * FROM episodes
      WHERE session_id = ?
      ORDER BY ts DESC
      LIMIT ?
    `);

    const rows = stmt.all(sessionId, limit) as any[];

    return rows.map(row => ({
      id: row.id,
      ts: row.ts,
      sessionId: row.session_id,
      task: row.task,
      input: row.input,
      output: row.output,
      critique: row.critique,
      reward: row.reward,
      success: row.success === 1,
      latencyMs: row.latency_ms,
      tokensUsed: row.tokens_used,
      tags: row.tags ? JSON.parse(row.tags) : undefined,
      metadata: row.metadata ? JSON.parse(row.metadata) : undefined
    }));
  }

  /**
   * Prune low-quality episodes based on TTL and quality threshold
   */
  pruneEpisodes(config: {
    minReward?: number;
    maxAgeDays?: number;
    keepMinPerTask?: number;
  }): number {
    const { minReward = 0.3, maxAgeDays = 30, keepMinPerTask = 5 } = config;

    // Keep high-reward episodes and minimum per task
    const stmt = this.db.prepare(`
      DELETE FROM episodes
      WHERE id IN (
        SELECT id FROM (
          SELECT
            id,
            reward,
            ts,
            ROW_NUMBER() OVER (PARTITION BY task ORDER BY reward DESC) as rank
          FROM episodes
          WHERE reward < ?
            AND ts < strftime('%s', 'now') - ?
        ) WHERE rank > ?
      )
    `);

    const result = stmt.run(minReward, maxAgeDays * 86400, keepMinPerTask);
    return result.changes;
  }

  // ========================================================================
  // Private Helper Methods
  // ========================================================================

  private buildEpisodeText(episode: Episode): string {
    const parts = [episode.task];
    if (episode.critique) parts.push(episode.critique);
    if (episode.output) parts.push(episode.output);
    return parts.join('\n');
  }

  private storeEmbedding(episodeId: number, embedding: Float32Array): void {
    const stmt = this.db.prepare(`
      INSERT INTO episode_embeddings (episode_id, embedding)
      VALUES (?, ?)
    `);

    stmt.run(episodeId, this.serializeEmbedding(embedding));
  }

  private serializeEmbedding(embedding: Float32Array): Buffer {
    // Handle empty/null embeddings
    if (!embedding || !embedding.buffer) {
      return Buffer.alloc(0);
    }
    return Buffer.from(embedding.buffer);
  }

  private deserializeEmbedding(buffer: Buffer): Float32Array {
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
  }

  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  // ========================================================================
  // GNN and Graph Integration Methods
  // ========================================================================

  /**
   * Create graph node for episode with relationships
   */
  private async createEpisodeGraphNode(
    episodeId: number,
    episode: Episode,
    embedding: Float32Array
  ): Promise<void> {
    if (!this.graphBackend) return;

    // Create episode node
    const nodeId = await this.graphBackend.createNode(
      ['Episode', episode.success ? 'Success' : 'Failure'],
      {
        episodeId,
        sessionId: episode.sessionId,
        task: episode.task,
        reward: episode.reward,
        success: episode.success,
        timestamp: episode.ts || Date.now(),
        latencyMs: episode.latencyMs,
        tokensUsed: episode.tokensUsed
      }
    );

    // Find similar episodes using graph vector search
    const similarEpisodes = await this.graphBackend.vectorSearch(embedding, 5, nodeId);

    // Create similarity relationships to similar episodes
    for (const similar of similarEpisodes) {
      if (similar.id !== nodeId && similar.properties.episodeId !== episodeId) {
        await this.graphBackend.createRelationship(
          nodeId,
          similar.id,
          'SIMILAR_TO',
          {
            similarity: this.cosineSimilarity(
              embedding,
              similar.embedding || new Float32Array()
            ),
            createdAt: Date.now()
          }
        );
      }
    }

    // Create session relationship
    const sessionNodes = await this.graphBackend.execute(
      'MATCH (s:Session {sessionId: $sessionId}) RETURN s',
      { sessionId: episode.sessionId }
    );

    let sessionNodeId: string;
    if (sessionNodes.rows.length === 0) {
      // Create session node if doesn't exist
      sessionNodeId = await this.graphBackend.createNode(
        ['Session'],
        {
          sessionId: episode.sessionId,
          startTime: episode.ts || Date.now()
        }
      );
    } else {
      sessionNodeId = sessionNodes.rows[0].s.id;
    }

    await this.graphBackend.createRelationship(
      nodeId,
      sessionNodeId,
      'BELONGS_TO_SESSION',
      { timestamp: episode.ts || Date.now() }
    );

    // If episode has critique, create causal relationship to previous failures
    if (episode.critique && !episode.success) {
      const previousFailures = await this.graphBackend.execute(
        `MATCH (e:Episode:Failure {sessionId: $sessionId})
         WHERE e.timestamp < $timestamp
         RETURN e
         ORDER BY e.timestamp DESC
         LIMIT 3`,
        { sessionId: episode.sessionId, timestamp: episode.ts || Date.now() }
      );

      for (const prevFailure of previousFailures.rows) {
        await this.graphBackend.createRelationship(
          nodeId,
          prevFailure.e.id,
          'LEARNED_FROM',
          {
            critique: episode.critique,
            improvementAttempt: true
          }
        );
      }
    }
  }

  /**
   * Enhance query embedding using GNN attention mechanism
   */
  private async enhanceQueryWithGNN(
    queryEmbedding: Float32Array,
    k: number
  ): Promise<Float32Array> {
    if (!this.learningBackend || !this.vectorBackend) {
      return queryEmbedding;
    }

    try {
      // Get initial neighbors
      const initialResults = this.vectorBackend.search(queryEmbedding, k * 2, {
        threshold: 0.0
      });

      if (initialResults.length === 0) {
        return queryEmbedding;
      }

      // Fetch neighbor embeddings
      const neighborEmbeddings: Float32Array[] = [];
      const weights: number[] = [];

      const episodeIds = initialResults.map(r => r.id);
      const placeholders = episodeIds.map(() => '?').join(',');
      const episodes = this.db.prepare(`
        SELECT ee.embedding, e.reward
        FROM episode_embeddings ee
        JOIN episodes e ON e.id = ee.episode_id
        WHERE ee.episode_id IN (${placeholders})
      `).all(...episodeIds) as any[];

      for (const ep of episodes) {
        const embedding = this.deserializeEmbedding(ep.embedding);
        neighborEmbeddings.push(embedding);
        // Use reward as weight (higher reward = more important)
        weights.push(Math.max(0.1, ep.reward));
      }

      // Enhance query using GNN
      const enhanced = this.learningBackend.enhance(
        queryEmbedding,
        neighborEmbeddings,
        weights
      );

      return enhanced;
    } catch (error) {
      console.warn('[ReflexionMemory] GNN enhancement failed:', error);
      return queryEmbedding;
    }
  }

  /**
   * Get graph-based episode relationships
   */
  async getEpisodeRelationships(episodeId: number): Promise<{
    similar: number[];
    session: string;
    learnedFrom: number[];
  }> {
    if (!this.graphBackend) {
      return { similar: [], session: '', learnedFrom: [] };
    }

    const result = await this.graphBackend.execute(
      `MATCH (e:Episode {episodeId: $episodeId})
       OPTIONAL MATCH (e)-[:SIMILAR_TO]->(similar:Episode)
       OPTIONAL MATCH (e)-[:BELONGS_TO_SESSION]->(s:Session)
       OPTIONAL MATCH (e)-[:LEARNED_FROM]->(learned:Episode)
       RETURN e, collect(DISTINCT similar.episodeId) as similar,
              s.sessionId as session,
              collect(DISTINCT learned.episodeId) as learnedFrom`,
      { episodeId }
    );

    if (result.rows.length === 0) {
      return { similar: [], session: '', learnedFrom: [] };
    }

    const row = result.rows[0];
    return {
      similar: (row.similar || []).filter((id: any) => id != null),
      session: row.session || '',
      learnedFrom: (row.learnedFrom || []).filter((id: any) => id != null)
    };
  }

  /**
   * Train GNN model on accumulated samples
   */
  async trainGNN(options?: { epochs?: number }): Promise<void> {
    if (!this.learningBackend) {
      console.warn('[ReflexionMemory] No learning backend available for training');
      return;
    }

    const stats = this.learningBackend.getStats();
    if (stats.samplesCollected < 10) {
      console.warn('[ReflexionMemory] Not enough samples for training (need at least 10)');
      return;
    }

    const result = await this.learningBackend.train(options);
    console.log('[ReflexionMemory] GNN training complete:', {
      epochs: result.epochs,
      finalLoss: result.finalLoss.toFixed(4),
      improvement: `${(result.improvement * 100).toFixed(1)}%`,
      duration: `${result.duration}ms`
    });
  }

  /**
   * Get learning backend statistics
   */
  getLearningStats() {
    if (!this.learningBackend) {
      return null;
    }
    return this.learningBackend.getStats();
  }

  /**
   * Get graph backend statistics
   */
  getGraphStats() {
    if (!this.graphBackend) {
      return null;
    }
    return this.graphBackend.getStats();
  }
}
