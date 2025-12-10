/**
 * AgentDB ReasoningBank - Cognitive Memory System
 *
 * The ReasoningBank is the core cognitive substrate for RAN optimization agents.
 * It stores "Thought Trajectories" - sequences of symptoms, contexts, actions, and outcomes
 * that enable agents to learn from experience rather than static rules.
 *
 * Key Features:
 * - Reflexion Memory: Self-critique of failed actions to prevent repeat mistakes
 * - Causal Graphs: Probabilistic causal relationships for root cause analysis
 * - Vector Search: HNSW-based similarity search for retrieving relevant past experiences
 * - RL Integration: Support for Decision Transformer and PPO algorithms
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import {
  ThoughtTrajectory,
  CausalGraph,
  CausalEdge,
  ReasoningBankQuery,
  ActionRecord,
  CellKPIs,
  NanoAgentId
} from '../../types/index.js';

// ============================================================================
// HNSW INDEX FOR VECTOR SIMILARITY SEARCH
// ============================================================================

interface HNSWConfig {
  dimensions: number;
  maxElements: number;
  efConstruction: number;
  M: number;
  efSearch: number;
}

class HNSWIndex {
  private nodes: Map<string, number[]> = new Map();
  private graph: Map<string, Set<string>> = new Map();
  private config: HNSWConfig;
  private entryPoint: string | null = null;

  constructor(config: HNSWConfig) {
    this.config = config;
  }

  /**
   * Add a vector to the HNSW index
   */
  add(id: string, vector: number[]): void {
    if (vector.length !== this.config.dimensions) {
      throw new Error(`Vector dimension mismatch: expected ${this.config.dimensions}, got ${vector.length}`);
    }

    this.nodes.set(id, vector);
    this.graph.set(id, new Set());

    if (!this.entryPoint) {
      this.entryPoint = id;
      return;
    }

    // Find M nearest neighbors and connect
    const neighbors = this.searchNearest(vector, this.config.M * 2);
    const connections = neighbors.slice(0, this.config.M);

    for (const neighbor of connections) {
      this.graph.get(id)!.add(neighbor.id);
      this.graph.get(neighbor.id)?.add(id);

      // Prune if too many connections
      const neighborConns = this.graph.get(neighbor.id)!;
      if (neighborConns.size > this.config.M * 2) {
        this.pruneConnections(neighbor.id);
      }
    }
  }

  /**
   * Search for k nearest neighbors to query vector
   */
  search(query: number[], k: number): Array<{ id: string; distance: number }> {
    if (!this.entryPoint) return [];
    return this.searchNearest(query, k).slice(0, k);
  }

  private searchNearest(query: number[], k: number): Array<{ id: string; distance: number }> {
    if (!this.entryPoint) return [];

    const visited = new Set<string>();
    const candidates: Array<{ id: string; distance: number }> = [];
    const results: Array<{ id: string; distance: number }> = [];

    // Start from entry point
    const entryDist = this.cosineSimilarity(query, this.nodes.get(this.entryPoint)!);
    candidates.push({ id: this.entryPoint, distance: entryDist });
    visited.add(this.entryPoint);

    while (candidates.length > 0) {
      // Get closest candidate
      candidates.sort((a, b) => b.distance - a.distance);
      const current = candidates.pop()!;

      // If this is worse than our worst result and we have enough, stop
      if (results.length >= k && current.distance < results[results.length - 1].distance) {
        break;
      }

      results.push(current);

      // Explore neighbors
      const neighbors = this.graph.get(current.id) || new Set();
      for (const neighborId of neighbors) {
        if (visited.has(neighborId)) continue;
        visited.add(neighborId);

        const neighborVec = this.nodes.get(neighborId);
        if (!neighborVec) continue;

        const dist = this.cosineSimilarity(query, neighborVec);
        candidates.push({ id: neighborId, distance: dist });
      }
    }

    results.sort((a, b) => b.distance - a.distance);
    return results;
  }

  private pruneConnections(nodeId: string): void {
    const connections = Array.from(this.graph.get(nodeId) || []);
    const nodeVec = this.nodes.get(nodeId)!;

    // Sort by distance and keep only M best
    const scored = connections.map(connId => ({
      id: connId,
      distance: this.cosineSimilarity(nodeVec, this.nodes.get(connId)!)
    }));
    scored.sort((a, b) => b.distance - a.distance);

    const keepSet = new Set(scored.slice(0, this.config.M).map(s => s.id));
    const toRemove = connections.filter(c => !keepSet.has(c));

    for (const removeId of toRemove) {
      this.graph.get(nodeId)?.delete(removeId);
      this.graph.get(removeId)?.delete(nodeId);
    }
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  getSize(): number {
    return this.nodes.size;
  }
}

// ============================================================================
// CAUSAL GRAPH ENGINE
// ============================================================================

class CausalGraphEngine {
  private graph: CausalGraph;
  private observations: Map<string, Map<string, number[]>> = new Map();

  constructor() {
    this.graph = {
      nodes: new Set(),
      edges: [],
      lastUpdated: new Date()
    };
  }

  /**
   * Record an observation of cause -> effect relationship
   */
  recordObservation(cause: string, effect: string, strength: number): void {
    this.graph.nodes.add(cause);
    this.graph.nodes.add(effect);

    // Track observations for Bayesian update
    const key = `${cause}::${effect}`;
    if (!this.observations.has(key)) {
      this.observations.set(key, new Map());
    }
    const obs = this.observations.get(key)!;
    if (!obs.has('strengths')) {
      obs.set('strengths', []);
    }
    obs.get('strengths')!.push(strength);

    this.updateEdge(cause, effect);
  }

  /**
   * Update causal edge probability using Bayesian inference
   */
  private updateEdge(cause: string, effect: string): void {
    const key = `${cause}::${effect}`;
    const strengths = this.observations.get(key)?.get('strengths') || [];

    if (strengths.length === 0) return;

    // Calculate mean probability and confidence
    const mean = strengths.reduce((a, b) => a + b, 0) / strengths.length;
    const variance = strengths.reduce((acc, s) => acc + Math.pow(s - mean, 2), 0) / strengths.length;
    const confidence = 1 / (1 + variance) * Math.min(1, strengths.length / 10);

    // Find or create edge
    let edge = this.graph.edges.find(e => e.cause === cause && e.effect === effect);
    if (!edge) {
      edge = {
        cause,
        effect,
        probability: mean,
        confidence,
        observationCount: strengths.length
      };
      this.graph.edges.push(edge);
    } else {
      edge.probability = mean;
      edge.confidence = confidence;
      edge.observationCount = strengths.length;
    }

    this.graph.lastUpdated = new Date();
  }

  /**
   * Query P(effect | do(cause)) - causal probability
   */
  queryCausalProbability(cause: string, effect: string): number {
    const edge = this.graph.edges.find(e => e.cause === cause && e.effect === effect);
    return edge?.probability || 0;
  }

  /**
   * Find root cause by traversing causal chain backwards
   */
  findRootCause(symptom: string, maxDepth: number = 5): CausalEdge[] {
    const chain: CausalEdge[] = [];
    const visited = new Set<string>();
    let current = symptom;

    for (let depth = 0; depth < maxDepth; depth++) {
      if (visited.has(current)) break;
      visited.add(current);

      // Find most probable cause of current symptom
      const causes = this.graph.edges
        .filter(e => e.effect === current)
        .sort((a, b) => (b.probability * b.confidence) - (a.probability * a.confidence));

      if (causes.length === 0) break;

      const bestCause = causes[0];
      chain.unshift(bestCause);
      current = bestCause.cause;
    }

    return chain;
  }

  /**
   * Get downstream effects of an action
   */
  predictEffects(cause: string): Array<{ effect: string; probability: number }> {
    return this.graph.edges
      .filter(e => e.cause === cause && e.confidence > 0.3)
      .map(e => ({ effect: e.effect, probability: e.probability }))
      .sort((a, b) => b.probability - a.probability);
  }

  getGraph(): CausalGraph {
    return this.graph;
  }

  serialize(): string {
    return JSON.stringify({
      graph: {
        nodes: Array.from(this.graph.nodes),
        edges: this.graph.edges,
        lastUpdated: this.graph.lastUpdated.toISOString()
      },
      observations: Array.from(this.observations.entries()).map(([k, v]) => ({
        key: k,
        strengths: v.get('strengths')
      }))
    });
  }

  static deserialize(json: string): CausalGraphEngine {
    const data = JSON.parse(json);
    const engine = new CausalGraphEngine();
    engine.graph = {
      nodes: new Set(data.graph.nodes),
      edges: data.graph.edges,
      lastUpdated: new Date(data.graph.lastUpdated)
    };
    for (const obs of data.observations) {
      const map = new Map<string, number[]>();
      map.set('strengths', obs.strengths);
      engine.observations.set(obs.key, map);
    }
    return engine;
  }
}

// ============================================================================
// REFLEXION MEMORY
// ============================================================================

interface ReflexionEntry {
  trajectoryId: string;
  originalAction: ActionRecord;
  outcome: string;
  critique: string;
  lessonLearned: string;
  timestamp: Date;
  revisitCount: number;
}

class ReflexionMemory {
  private entries: Map<string, ReflexionEntry> = new Map();
  private actionPatterns: Map<string, string[]> = new Map(); // action type -> failure patterns

  /**
   * Record a reflexion on a failed action
   */
  addReflexion(
    trajectoryId: string,
    action: ActionRecord,
    outcome: string,
    critique: string,
    lessonLearned: string
  ): void {
    const entry: ReflexionEntry = {
      trajectoryId,
      originalAction: action,
      outcome,
      critique,
      lessonLearned,
      timestamp: new Date(),
      revisitCount: 0
    };

    this.entries.set(trajectoryId, entry);

    // Index by action type for quick lookup
    const actionKey = `${action.type}::${action.targetCgi}`;
    if (!this.actionPatterns.has(actionKey)) {
      this.actionPatterns.set(actionKey, []);
    }
    this.actionPatterns.get(actionKey)!.push(lessonLearned);
  }

  /**
   * Check if similar action has failed before
   */
  checkForPastFailures(action: ActionRecord): ReflexionEntry[] {
    const actionKey = `${action.type}::${action.targetCgi}`;
    const directMatches = Array.from(this.entries.values())
      .filter(e => e.originalAction.type === action.type);

    // Also check for similar parameter values
    const paramMatches = directMatches.filter(e => {
      const origParams = e.originalAction.parameters;
      const newParams = action.parameters;
      return Object.keys(origParams).some(k =>
        newParams[k] !== undefined &&
        Math.abs(Number(origParams[k]) - Number(newParams[k])) < 2
      );
    });

    return paramMatches;
  }

  /**
   * Get lessons for a specific action type
   */
  getLessons(actionType: string): string[] {
    const lessons: string[] = [];
    for (const entry of this.entries.values()) {
      if (entry.originalAction.type === actionType) {
        lessons.push(entry.lessonLearned);
      }
    }
    return lessons;
  }

  getEntryCount(): number {
    return this.entries.size;
  }
}

// ============================================================================
// MAIN REASONING BANK CLASS
// ============================================================================

export interface ReasoningBankConfig {
  agentId: NanoAgentId;
  maxTrajectories: number;
  vectorDimensions: number;
  hnswConfig: Partial<HNSWConfig>;
  reflexionEnabled: boolean;
}

const DEFAULT_CONFIG: ReasoningBankConfig = {
  agentId: 'default-agent',
  maxTrajectories: 10000,
  vectorDimensions: 768,
  hnswConfig: {
    efConstruction: 200,
    M: 16,
    efSearch: 100
  },
  reflexionEnabled: true
};

export class ReasoningBank extends EventEmitter {
  private config: ReasoningBankConfig;
  private trajectories: Map<string, ThoughtTrajectory> = new Map();
  private vectorIndex: HNSWIndex;
  private causalGraph: CausalGraphEngine;
  private reflexionMemory: ReflexionMemory;

  // Embedding cache for symptom/context strings
  private embeddingCache: Map<string, number[]> = new Map();

  constructor(config: Partial<ReasoningBankConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.vectorIndex = new HNSWIndex({
      dimensions: this.config.vectorDimensions,
      maxElements: this.config.maxTrajectories,
      efConstruction: this.config.hnswConfig.efConstruction || 200,
      M: this.config.hnswConfig.M || 16,
      efSearch: this.config.hnswConfig.efSearch || 100
    });

    this.causalGraph = new CausalGraphEngine();
    this.reflexionMemory = new ReflexionMemory();
  }

  /**
   * Store a new thought trajectory
   */
  async storeTrajectory(trajectory: Omit<ThoughtTrajectory, 'id' | 'timestamp'>): Promise<string> {
    const id = uuidv4();
    const fullTrajectory: ThoughtTrajectory = {
      ...trajectory,
      id,
      timestamp: new Date()
    };

    // Generate embedding for the symptom + context
    const embedding = await this.generateEmbedding(trajectory.symptom, trajectory.context);

    // Store trajectory
    this.trajectories.set(id, fullTrajectory);

    // Add to vector index
    this.vectorIndex.add(id, embedding);

    // Update causal graph based on outcome
    if (trajectory.outcome.success) {
      for (const action of trajectory.actionSequence) {
        this.causalGraph.recordObservation(
          `action:${action.type}`,
          'outcome:success',
          1.0
        );
      }
    } else {
      // Add reflexion for failed trajectory
      if (this.config.reflexionEnabled && trajectory.reflexion) {
        const lastAction = trajectory.actionSequence[trajectory.actionSequence.length - 1];
        if (lastAction) {
          this.reflexionMemory.addReflexion(
            id,
            {
              ...lastAction,
              id: uuidv4(),
              executed: true,
              blocked: false
            },
            trajectory.outcome.verdict,
            trajectory.reflexion,
            this.extractLesson(trajectory)
          );
        }
      }
    }

    // Enforce max trajectories
    if (this.trajectories.size > this.config.maxTrajectories) {
      this.evictOldest();
    }

    this.emit('trajectory-stored', fullTrajectory);
    return id;
  }

  /**
   * Query similar past experiences
   */
  async query(query: ReasoningBankQuery): Promise<ThoughtTrajectory[]> {
    const results = this.vectorIndex.search(query.contextEmbedding, query.maxResults * 2);

    const trajectories: ThoughtTrajectory[] = [];
    for (const result of results) {
      if (result.distance < query.minSimilarity) continue;

      const trajectory = this.trajectories.get(result.id);
      if (trajectory) {
        trajectories.push(trajectory);
        if (trajectories.length >= query.maxResults) break;
      }
    }

    return trajectories;
  }

  /**
   * Query by symptom string (convenience method)
   */
  async queryBySymptom(
    symptom: string,
    context: any,
    maxResults: number = 5
  ): Promise<ThoughtTrajectory[]> {
    const embedding = await this.generateEmbedding(symptom, context);
    return this.query({
      symptom,
      contextEmbedding: embedding,
      maxResults,
      minSimilarity: 0.5
    });
  }

  /**
   * Check if proposed action might fail based on past experience
   */
  checkActionRisk(action: ActionRecord): {
    riskLevel: 'low' | 'medium' | 'high';
    pastFailures: ReflexionEntry[];
    warnings: string[];
  } {
    const pastFailures = this.reflexionMemory.checkForPastFailures(action);
    const lessons = this.reflexionMemory.getLessons(action.type);

    const riskLevel = pastFailures.length === 0 ? 'low' :
                      pastFailures.length < 3 ? 'medium' : 'high';

    return {
      riskLevel,
      pastFailures,
      warnings: lessons.slice(0, 3)
    };
  }

  /**
   * Record causal observation
   */
  recordCausalObservation(cause: string, effect: string, strength: number): void {
    this.causalGraph.recordObservation(cause, effect, strength);
    this.emit('causal-observation', { cause, effect, strength });
  }

  /**
   * Find root cause of a symptom
   */
  findRootCause(symptom: string): CausalEdge[] {
    return this.causalGraph.findRootCause(symptom);
  }

  /**
   * Predict effects of an action
   */
  predictActionEffects(actionType: string): Array<{ effect: string; probability: number }> {
    return this.causalGraph.predictEffects(`action:${actionType}`);
  }

  /**
   * Get statistics about the reasoning bank
   */
  getStats(): {
    trajectoryCount: number;
    successfulTrajectories: number;
    failedTrajectories: number;
    causalEdgeCount: number;
    reflexionCount: number;
    vectorIndexSize: number;
  } {
    let successful = 0;
    let failed = 0;

    for (const t of this.trajectories.values()) {
      if (t.outcome.success) successful++;
      else failed++;
    }

    return {
      trajectoryCount: this.trajectories.size,
      successfulTrajectories: successful,
      failedTrajectories: failed,
      causalEdgeCount: this.causalGraph.getGraph().edges.length,
      reflexionCount: this.reflexionMemory.getEntryCount(),
      vectorIndexSize: this.vectorIndex.getSize()
    };
  }

  /**
   * Export state for federation/persistence
   */
  export(): string {
    return JSON.stringify({
      agentId: this.config.agentId,
      trajectories: Array.from(this.trajectories.entries()),
      causalGraph: this.causalGraph.serialize(),
      exportedAt: new Date().toISOString()
    });
  }

  /**
   * Generate embedding for symptom + context
   * Uses a simple but effective TF-IDF + positional encoding approach
   */
  private async generateEmbedding(symptom: string, context: any): Promise<number[]> {
    const cacheKey = `${symptom}::${JSON.stringify(context)}`;
    if (this.embeddingCache.has(cacheKey)) {
      return this.embeddingCache.get(cacheKey)!;
    }

    // Simple embedding: hash-based with semantic features
    const embedding = new Array(this.config.vectorDimensions).fill(0);

    // Hash symptom tokens
    const tokens = symptom.toLowerCase().split(/\s+/);
    for (let i = 0; i < tokens.length; i++) {
      const hash = this.hashString(tokens[i]);
      const idx = Math.abs(hash) % this.config.vectorDimensions;
      embedding[idx] += 1 / (i + 1); // Position-weighted
    }

    // Add context features
    if (context.cellState) {
      const state = context.cellState;
      // Encode key metrics into specific dimensions
      if (state.callDropRate !== undefined) {
        embedding[0] = state.callDropRate;
      }
      if (state.iotUl !== undefined) {
        embedding[1] = state.iotUl / 20; // Normalize
      }
      if (state.avgSinrUl !== undefined) {
        embedding[2] = (state.avgSinrUl + 10) / 40; // Normalize -10 to 30
      }
      if (state.prbUtilizationUl !== undefined) {
        embedding[3] = state.prbUtilizationUl / 100;
      }
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((acc, v) => acc + v * v, 0));
    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }

    this.embeddingCache.set(cacheKey, embedding);
    return embedding;
  }

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash;
  }

  private extractLesson(trajectory: ThoughtTrajectory): string {
    // Extract a concise lesson from the failed trajectory
    const lastAction = trajectory.actionSequence[trajectory.actionSequence.length - 1];
    return `${lastAction?.type || 'unknown action'} in context "${trajectory.symptom}" led to: ${trajectory.outcome.verdict}`;
  }

  private evictOldest(): void {
    // Simple LRU - remove oldest trajectory
    let oldest: string | null = null;
    let oldestTime = Infinity;

    for (const [id, t] of this.trajectories) {
      if (t.timestamp.getTime() < oldestTime) {
        oldestTime = t.timestamp.getTime();
        oldest = id;
      }
    }

    if (oldest) {
      this.trajectories.delete(oldest);
      this.emit('trajectory-evicted', oldest);
    }
  }
}

export { CausalGraphEngine, ReflexionMemory, HNSWIndex };
