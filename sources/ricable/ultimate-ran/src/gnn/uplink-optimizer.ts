/**
 * GNN-based Uplink Power Control Optimizer
 *
 * Optimizes P0 Nominal PUSCH and Alpha FPC parameters using Graph Attention Networks (GAT).
 * Implements multi-head attention for neighbor cell interference modeling and Q-learning
 * for self-learning optimization with ruvector embeddings and agentdb episode memory.
 *
 * 3GPP Compliance:
 * - P0 Nominal PUSCH: -130 to -70 dBm (TS 38.213 Section 7.1.1)
 * - Alpha: 0 to 1 (Fractional Power Control factor, TS 38.213 Section 7.1.1)
 * - SINR targets: 0-30 dB typical range
 *
 * @module gnn/uplink-optimizer
 * @agent Agent 1: GNN P0/Alpha Optimizer for Ericsson RAN
 */

import { EventEmitter } from 'events';
import type { PMCounters, CMParameters } from '../learning/self-learner.js';

// ============================================================
// Constants & Configuration
// ============================================================

/** P0 Nominal PUSCH valid range (3GPP TS 38.213) */
const P0_MIN = -130; // dBm
const P0_MAX = -70;  // dBm

/** Alpha FPC factor valid range */
const ALPHA_MIN = 0.0;
const ALPHA_MAX = 1.0;

/** Target SINR for uplink (dB) */
const TARGET_SINR_MIN = 10;
const TARGET_SINR_MAX = 25;

/** Vector embedding dimension (matching ruvector) */
const EMBEDDING_DIM = 768;

/** Number of attention heads for GAT */
const NUM_ATTENTION_HEADS = 8;

/** GAT hidden layer dimensions */
const GAT_HIDDEN_DIM = 128;

// ============================================================
// Type Definitions
// ============================================================

/**
 * Uplink GNN Optimizer interface
 */
export interface UplinkGNNOptimizer {
  /**
   * Optimize P0/Alpha parameters for a cell using GNN
   *
   * @param cellId - Target cell identifier
   * @param pmData - Current PM counters
   * @returns Optimized parameters with confidence and predicted gain
   */
  optimizeP0Alpha(cellId: string, pmData: PMCounters): Promise<{
    p0: number;
    alpha: number;
    confidence: number;
    predictedSinrGain: number;
  }>;
}

/**
 * Cell node in the interference graph
 */
export interface CellNode {
  /** Cell identifier */
  cellId: string;

  /** Node features: [SINR, RSRP, PRB usage, CQI] */
  features: number[];

  /** Current P0 setting */
  p0?: number;

  /** Current Alpha setting */
  alpha?: number;

  /** Geographic coordinates (optional) */
  coordinates?: { latitude: number; longitude: number; altitude?: number };

  /** Vector embedding (768-dim) */
  embedding?: number[];
}

/**
 * Edge between neighboring cells with interference coupling
 */
export interface InterferenceEdge {
  /** Source cell ID */
  fromCell: string;

  /** Target cell ID */
  toCell: string;

  /** Edge features: [distance, overlap_pct, interference_coupling] */
  features: number[];

  /** Physical distance in meters */
  distance: number;

  /** Coverage overlap percentage (0-1) */
  overlapPct: number;

  /** Interference coupling loss in dB */
  interferenceCoupling: number;
}

/**
 * Graph representation of RAN cells
 */
export interface CellGraph {
  /** Map of cell ID to node */
  nodes: Map<string, CellNode>;

  /** Adjacency list: cell ID -> list of neighbor edges */
  edges: Map<string, InterferenceEdge[]>;

  /** Graph metadata */
  metadata: {
    timestamp: number;
    nodeCount: number;
    edgeCount: number;
  };
}

/**
 * GAT attention head output
 */
interface AttentionHead {
  /** Attention weights for each neighbor */
  weights: Map<string, number>;

  /** Transformed feature vector */
  features: number[];
}

/**
 * Optimization episode for Q-learning
 */
export interface OptimizationEpisode {
  /** Unique episode identifier */
  id: string;

  /** Target cell ID */
  cellId: string;

  /** PM counters before optimization */
  pmBefore: PMCounters;

  /** PM counters after optimization */
  pmAfter: PMCounters;

  /** Optimization action taken */
  action: {
    p0: number;
    alpha: number;
  };

  /** Reward (SINR improvement) */
  reward: number;

  /** Episode timestamp */
  timestamp: number;

  /** 768-dimensional embedding for ruvector */
  embedding: number[];

  /** Neighbor context used in decision */
  neighborContext?: {
    cellIds: string[];
    avgSinr: number;
    maxInterference: number;
  };
}

/**
 * GNN optimizer configuration
 */
export interface GNNOptimizerConfig {
  /** Learning rate for Q-learning */
  learningRate: number;

  /** Discount factor (gamma) for future rewards */
  discountFactor: number;

  /** Exploration rate (epsilon) for epsilon-greedy */
  explorationRate: number;

  /** Batch size for experience replay */
  batchSize: number;

  /** Maximum number of episodes to store */
  maxEpisodes: number;

  /** Number of GAT attention heads */
  numHeads: number;

  /** GAT hidden dimension */
  hiddenDim: number;

  /** Path to agentdb database */
  agentDbPath: string;

  /** Path to ruvector database */
  ruvectorDbPath: string;

  /** Enable transfer learning from similar episodes */
  enableTransferLearning: boolean;
}

// ============================================================
// Graph Attention Network (GAT) Implementation
// ============================================================

/**
 * Multi-head Graph Attention Network for cell interference modeling
 *
 * Implements the attention mechanism from "Graph Attention Networks" (Veličković et al., 2018)
 * with multi-head attention for robust feature learning across interference patterns.
 */
class GraphAttentionNetwork {
  private numHeads: number;
  private hiddenDim: number;
  private inputDim: number;

  // Attention parameters (in practice, these would be learned via backprop)
  private weightMatrices: number[][][]; // [head][inputDim][hiddenDim]
  private attentionVectors: number[][]; // [head][2*hiddenDim]

  /**
   * Initialize GAT with multi-head attention
   *
   * @param inputDim - Input feature dimension (4: SINR, RSRP, PRB, CQI)
   * @param hiddenDim - Hidden layer dimension
   * @param numHeads - Number of attention heads
   */
  constructor(inputDim: number = 4, hiddenDim: number = GAT_HIDDEN_DIM, numHeads: number = NUM_ATTENTION_HEADS) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.numHeads = numHeads;

    // Initialize weight matrices with Xavier initialization
    this.weightMatrices = this.initializeWeights(numHeads, inputDim, hiddenDim);
    this.attentionVectors = this.initializeAttentionVectors(numHeads, hiddenDim);

    console.log(`[GAT] Initialized with ${numHeads} attention heads, input_dim=${inputDim}, hidden_dim=${hiddenDim}`);
  }

  /**
   * Initialize weight matrices using Xavier initialization
   */
  private initializeWeights(numHeads: number, inputDim: number, hiddenDim: number): number[][][] {
    const weights: number[][][] = [];
    const limit = Math.sqrt(6 / (inputDim + hiddenDim));

    for (let h = 0; h < numHeads; h++) {
      const headWeights: number[][] = [];
      for (let i = 0; i < inputDim; i++) {
        const row: number[] = [];
        for (let j = 0; j < hiddenDim; j++) {
          row.push((Math.random() * 2 - 1) * limit);
        }
        headWeights.push(row);
      }
      weights.push(headWeights);
    }

    return weights;
  }

  /**
   * Initialize attention vectors
   */
  private initializeAttentionVectors(numHeads: number, hiddenDim: number): number[][] {
    const vectors: number[][] = [];
    const limit = Math.sqrt(6 / (2 * hiddenDim));

    for (let h = 0; h < numHeads; h++) {
      const vec: number[] = [];
      for (let i = 0; i < 2 * hiddenDim; i++) {
        vec.push((Math.random() * 2 - 1) * limit);
      }
      vectors.push(vec);
    }

    return vectors;
  }

  /**
   * Apply graph attention to a target cell and its neighbors
   *
   * @param targetNode - Target cell node
   * @param neighborNodes - Array of neighbor cell nodes
   * @param neighborEdges - Edges to neighbors with interference features
   * @returns Aggregated node embedding with attention
   */
  forward(
    targetNode: CellNode,
    neighborNodes: CellNode[],
    neighborEdges: InterferenceEdge[]
  ): number[] {
    const headOutputs: number[][] = [];

    // Process each attention head
    for (let h = 0; h < this.numHeads; h++) {
      const headOutput = this.computeAttentionHead(
        h,
        targetNode,
        neighborNodes,
        neighborEdges
      );
      headOutputs.push(headOutput);
    }

    // Concatenate multi-head outputs
    return this.concatenateHeads(headOutputs);
  }

  /**
   * Compute attention for a single head
   */
  private computeAttentionHead(
    headIdx: number,
    targetNode: CellNode,
    neighborNodes: CellNode[],
    neighborEdges: InterferenceEdge[]
  ): number[] {
    // Transform target node features
    const targetFeatures = this.transformFeatures(targetNode.features, headIdx);

    // Compute attention coefficients for each neighbor
    const attentionScores: number[] = [];
    const transformedNeighbors: number[][] = [];

    for (let i = 0; i < neighborNodes.length; i++) {
      const neighbor = neighborNodes[i];
      const edge = neighborEdges[i];

      // Transform neighbor features
      const neighborFeatures = this.transformFeatures(neighbor.features, headIdx);
      transformedNeighbors.push(neighborFeatures);

      // Compute attention coefficient with edge features
      const score = this.computeAttentionScore(
        headIdx,
        targetFeatures,
        neighborFeatures,
        edge.features
      );
      attentionScores.push(score);
    }

    // Apply softmax to attention scores
    const attentionWeights = this.softmax(attentionScores);

    // Aggregate neighbor features using attention weights
    const aggregated = new Array(this.hiddenDim).fill(0);

    for (let i = 0; i < transformedNeighbors.length; i++) {
      const weight = attentionWeights[i];
      const neighborFeats = transformedNeighbors[i];

      for (let j = 0; j < this.hiddenDim; j++) {
        aggregated[j] += weight * neighborFeats[j];
      }
    }

    // Add target node's own features (residual connection)
    for (let j = 0; j < this.hiddenDim; j++) {
      aggregated[j] = 0.5 * aggregated[j] + 0.5 * targetFeatures[j];
    }

    // Apply LeakyReLU activation
    return aggregated.map(x => x > 0 ? x : 0.2 * x);
  }

  /**
   * Transform input features using weight matrix
   */
  private transformFeatures(features: number[], headIdx: number): number[] {
    const W = this.weightMatrices[headIdx];
    const output = new Array(this.hiddenDim).fill(0);

    for (let i = 0; i < this.hiddenDim; i++) {
      for (let j = 0; j < Math.min(features.length, this.inputDim); j++) {
        output[i] += features[j] * W[j][i];
      }
    }

    return output;
  }

  /**
   * Compute attention score between target and neighbor
   *
   * e_ij = LeakyReLU(a^T [W*h_i || W*h_j || edge_features])
   */
  private computeAttentionScore(
    headIdx: number,
    targetFeatures: number[],
    neighborFeatures: number[],
    edgeFeatures: number[]
  ): number {
    const a = this.attentionVectors[headIdx];

    // Concatenate [target || neighbor] features
    const concat = [...targetFeatures, ...neighborFeatures];

    // Add edge features (distance, overlap, coupling) at the end
    const fullConcat = [...concat, ...edgeFeatures.slice(0, 3)];

    // Compute attention score: a^T * concat
    let score = 0;
    for (let i = 0; i < Math.min(fullConcat.length, a.length); i++) {
      score += a[i] * fullConcat[i];
    }

    // LeakyReLU
    return score > 0 ? score : 0.2 * score;
  }

  /**
   * Softmax normalization
   */
  private softmax(scores: number[]): number[] {
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(e => e / sumExp);
  }

  /**
   * Concatenate multi-head outputs
   */
  private concatenateHeads(headOutputs: number[][]): number[] {
    // Average pooling across heads for final output
    const numFeatures = headOutputs[0].length;
    const pooled = new Array(numFeatures).fill(0);

    for (const headOutput of headOutputs) {
      for (let i = 0; i < numFeatures; i++) {
        pooled[i] += headOutput[i] / headOutputs.length;
      }
    }

    return pooled;
  }

  /**
   * Update weights using gradient descent (simplified online learning)
   *
   * In production, this would use full backpropagation.
   * Here we use a simplified reward-based update.
   */
  updateWeights(reward: number, learningRate: number): void {
    // Simple gradient ascent based on reward signal
    const scale = learningRate * reward;

    for (let h = 0; h < this.numHeads; h++) {
      for (let i = 0; i < this.inputDim; i++) {
        for (let j = 0; j < this.hiddenDim; j++) {
          // Small random perturbation scaled by reward
          const perturbation = (Math.random() * 2 - 1) * scale;
          this.weightMatrices[h][i][j] += perturbation;
        }
      }
    }
  }
}

// ============================================================
// GNN-based Uplink Optimizer
// ============================================================

/**
 * GNN-based P0/Alpha optimizer with Q-learning and transfer learning
 *
 * Implements:
 * - Graph Attention Networks (GAT) for neighbor interference modeling
 * - Q-learning for policy optimization
 * - Ruvector integration for episode similarity search
 * - AgentDB reflexion storage for experience replay
 */
class GNNUplinkOptimizer extends EventEmitter implements UplinkGNNOptimizer {
  private config: GNNOptimizerConfig;
  private gat: GraphAttentionNetwork;
  private cellGraph: CellGraph;
  private episodes: Map<string, OptimizationEpisode>;
  private qTable: Map<string, Map<string, number>>; // state -> action -> Q-value
  private episodeEmbeddings: Map<string, number[]>; // episode ID -> embedding

  /**
   * Initialize GNN uplink optimizer
   *
   * @param config - Optimizer configuration
   */
  constructor(config?: Partial<GNNOptimizerConfig>) {
    super();

    this.config = {
      learningRate: config?.learningRate || 0.01,
      discountFactor: config?.discountFactor || 0.95,
      explorationRate: config?.explorationRate || 0.15,
      batchSize: config?.batchSize || 32,
      maxEpisodes: config?.maxEpisodes || 10000,
      numHeads: config?.numHeads || NUM_ATTENTION_HEADS,
      hiddenDim: config?.hiddenDim || GAT_HIDDEN_DIM,
      agentDbPath: config?.agentDbPath || './titan-ran.db',
      ruvectorDbPath: config?.ruvectorDbPath || './ruvector-spatial.db',
      enableTransferLearning: config?.enableTransferLearning !== false
    };

    // Initialize GAT
    this.gat = new GraphAttentionNetwork(4, this.config.hiddenDim, this.config.numHeads);

    // Initialize cell graph
    this.cellGraph = {
      nodes: new Map(),
      edges: new Map(),
      metadata: {
        timestamp: Date.now(),
        nodeCount: 0,
        edgeCount: 0
      }
    };

    this.episodes = new Map();
    this.qTable = new Map();
    this.episodeEmbeddings = new Map();

    console.log('[GNN-Optimizer] Initialized with config:', {
      numHeads: this.config.numHeads,
      hiddenDim: this.config.hiddenDim,
      learningRate: this.config.learningRate,
      transferLearning: this.config.enableTransferLearning
    });
  }

  /**
   * Add a cell node to the graph
   *
   * @param node - Cell node to add
   */
  addCellNode(node: CellNode): void {
    this.cellGraph.nodes.set(node.cellId, node);

    if (!this.cellGraph.edges.has(node.cellId)) {
      this.cellGraph.edges.set(node.cellId, []);
    }

    this.cellGraph.metadata.nodeCount = this.cellGraph.nodes.size;
    this.cellGraph.metadata.timestamp = Date.now();

    this.emit('cell_added', { cellId: node.cellId });
  }

  /**
   * Add interference edge between cells
   *
   * @param edge - Interference edge to add
   */
  addInterferenceEdge(edge: InterferenceEdge): void {
    // Add edge to adjacency list
    if (!this.cellGraph.edges.has(edge.fromCell)) {
      this.cellGraph.edges.set(edge.fromCell, []);
    }

    this.cellGraph.edges.get(edge.fromCell)!.push(edge);

    // Add reverse edge for undirected graph
    const reverseEdge: InterferenceEdge = {
      fromCell: edge.toCell,
      toCell: edge.fromCell,
      features: [...edge.features],
      distance: edge.distance,
      overlapPct: edge.overlapPct,
      interferenceCoupling: edge.interferenceCoupling
    };

    if (!this.cellGraph.edges.has(edge.toCell)) {
      this.cellGraph.edges.set(edge.toCell, []);
    }

    this.cellGraph.edges.get(edge.toCell)!.push(reverseEdge);

    this.cellGraph.metadata.edgeCount = Array.from(this.cellGraph.edges.values())
      .reduce((sum, edges) => sum + edges.length, 0) / 2; // Divide by 2 for undirected

    this.emit('edge_added', { fromCell: edge.fromCell, toCell: edge.toCell });
  }

  /**
   * Optimize P0/Alpha parameters using GNN and Q-learning
   *
   * @param cellId - Target cell identifier
   * @param pmData - Current PM counters
   * @returns Optimized P0/Alpha parameters with confidence
   */
  async optimizeP0Alpha(cellId: string, pmData: PMCounters): Promise<{
    p0: number;
    alpha: number;
    confidence: number;
    predictedSinrGain: number;
  }> {
    const startTime = Date.now();

    // Get cell node from graph
    const targetNode = this.cellGraph.nodes.get(cellId);

    if (!targetNode) {
      throw new Error(`Cell ${cellId} not found in graph. Call addCellNode() first.`);
    }

    // Update node features from PM data
    this.updateNodeFeatures(targetNode, pmData);

    // Get neighbors and edges
    const { neighbors, edges } = this.getNeighborContext(cellId);

    if (neighbors.length === 0) {
      console.warn(`[GNN-Optimizer] Cell ${cellId} has no neighbors, using fallback optimization`);
      return this.fallbackOptimization(cellId, pmData);
    }

    // Run GAT forward pass
    const gnnEmbedding = this.gat.forward(targetNode, neighbors, edges);

    // Query similar past episodes for transfer learning
    let transferKnowledge: OptimizationEpisode | null = null;

    if (this.config.enableTransferLearning) {
      transferKnowledge = await this.querySimilarEpisode(gnnEmbedding);
    }

    // Get Q-learning action (epsilon-greedy)
    const state = this.encodeState(pmData);
    const action = this.selectAction(state, gnnEmbedding, transferKnowledge);

    // Predict SINR gain
    const currentSinr = pmData.pmUlSinrMean || 0;
    const predictedSinrGain = this.predictSinrGain(
      currentSinr,
      action.p0,
      action.alpha,
      neighbors,
      edges
    );

    // Calculate confidence based on Q-value and transfer learning
    const confidence = this.calculateConfidence(state, action, transferKnowledge);

    const latency = Date.now() - startTime;

    console.log(`[GNN-Optimizer] Optimized ${cellId} in ${latency}ms: P0=${action.p0.toFixed(1)} dBm, Alpha=${action.alpha.toFixed(3)}, predicted_gain=${predictedSinrGain.toFixed(2)} dB`);

    this.emit('optimization_complete', {
      cellId,
      p0: action.p0,
      alpha: action.alpha,
      confidence,
      predictedSinrGain,
      latency,
      neighbors: neighbors.length
    });

    return {
      p0: action.p0,
      alpha: action.alpha,
      confidence,
      predictedSinrGain
    };
  }

  /**
   * Update node features from PM data
   */
  private updateNodeFeatures(node: CellNode, pmData: PMCounters): void {
    // Extract key features: [SINR, RSRP, PRB usage, CQI]
    node.features = [
      pmData.pmUlSinrMean || 0,           // SINR in dB
      pmData.pmUlRssi || -100,            // RSRP approximation
      (pmData.pmPuschPrbUsage || 0) * 100, // PRB usage as percentage
      this.estimateCQI(pmData)            // Estimated CQI
    ];
  }

  /**
   * Estimate CQI from SINR (3GPP mapping)
   */
  private estimateCQI(pmData: PMCounters): number {
    const sinr = pmData.pmUlSinrMean || 0;

    // Simplified CQI mapping (0-15 range)
    if (sinr < -5) return 1;
    if (sinr < 0) return 4;
    if (sinr < 5) return 7;
    if (sinr < 10) return 10;
    if (sinr < 15) return 12;
    if (sinr < 20) return 14;
    return 15;
  }

  /**
   * Get neighbor context for a cell
   */
  private getNeighborContext(cellId: string): {
    neighbors: CellNode[];
    edges: InterferenceEdge[];
  } {
    const neighborEdges = this.cellGraph.edges.get(cellId) || [];
    const neighbors: CellNode[] = [];
    const edges: InterferenceEdge[] = [];

    for (const edge of neighborEdges) {
      const neighbor = this.cellGraph.nodes.get(edge.toCell);
      if (neighbor) {
        neighbors.push(neighbor);
        edges.push(edge);
      }
    }

    return { neighbors, edges };
  }

  /**
   * Encode PM state for Q-learning
   */
  private encodeState(pmData: PMCounters): string {
    // Discretize continuous state into bins
    const sinrBin = Math.floor((pmData.pmUlSinrMean || 0) / 5);
    const blerBin = Math.floor((pmData.pmUlBler || 0) * 20);
    const prbBin = Math.floor((pmData.pmPuschPrbUsage || 0) * 10);

    return `s_${sinrBin}_${blerBin}_${prbBin}`;
  }

  /**
   * Select action using epsilon-greedy Q-learning
   */
  private selectAction(
    state: string,
    gnnEmbedding: number[],
    transferKnowledge: OptimizationEpisode | null
  ): { p0: number; alpha: number; actionKey: string } {
    // Epsilon-greedy exploration
    if (Math.random() < this.config.explorationRate) {
      return this.exploreAction();
    }

    // Transfer learning: use similar episode's action with high probability
    if (transferKnowledge && Math.random() < 0.7) {
      console.log(`[GNN-Optimizer] Using transfer learning from episode ${transferKnowledge.id}`);
      return {
        p0: transferKnowledge.action.p0,
        alpha: transferKnowledge.action.alpha,
        actionKey: this.encodeAction(transferKnowledge.action.p0, transferKnowledge.action.alpha)
      };
    }

    // Exploit: select best action from Q-table
    return this.exploitAction(state, gnnEmbedding);
  }

  /**
   * Explore: random action within valid bounds
   */
  private exploreAction(): { p0: number; alpha: number; actionKey: string } {
    const p0 = P0_MIN + Math.random() * (P0_MAX - P0_MIN);
    const alpha = ALPHA_MIN + Math.random() * (ALPHA_MAX - ALPHA_MIN);

    return {
      p0,
      alpha,
      actionKey: this.encodeAction(p0, alpha)
    };
  }

  /**
   * Exploit: best action from Q-table
   */
  private exploitAction(state: string, gnnEmbedding: number[]): {
    p0: number;
    alpha: number;
    actionKey: string;
  } {
    const stateActions = this.qTable.get(state);

    if (!stateActions || stateActions.size === 0) {
      // No Q-values yet, use GNN embedding to guide initial action
      return this.gnnGuidedAction(gnnEmbedding);
    }

    // Find action with highest Q-value
    let bestAction = '';
    let bestQ = -Infinity;

    for (const [action, qValue] of stateActions.entries()) {
      if (qValue > bestQ) {
        bestQ = qValue;
        bestAction = action;
      }
    }

    const { p0, alpha } = this.decodeAction(bestAction);

    return { p0, alpha, actionKey: bestAction };
  }

  /**
   * Use GNN embedding to guide action selection
   */
  private gnnGuidedAction(gnnEmbedding: number[]): {
    p0: number;
    alpha: number;
    actionKey: string;
  } {
    // Use GNN embedding to compute P0 and Alpha
    // Take weighted sum of embedding features (simplified)
    const embeddingSum = gnnEmbedding.reduce((sum, val) => sum + val, 0);
    const embeddingAvg = embeddingSum / gnnEmbedding.length;

    // Map embedding to P0 range
    const p0Normalized = Math.tanh(embeddingAvg); // [-1, 1]
    const p0 = P0_MIN + (p0Normalized + 1) / 2 * (P0_MAX - P0_MIN);

    // Map embedding variance to Alpha
    const embeddingVariance = gnnEmbedding.reduce((sum, val) =>
      sum + Math.pow(val - embeddingAvg, 2), 0) / gnnEmbedding.length;
    const alpha = Math.min(ALPHA_MAX, Math.max(ALPHA_MIN, embeddingVariance));

    return {
      p0,
      alpha,
      actionKey: this.encodeAction(p0, alpha)
    };
  }

  /**
   * Encode action as string key
   */
  private encodeAction(p0: number, alpha: number): string {
    const p0Bin = Math.round(p0);
    const alphaBin = Math.round(alpha * 100);
    return `a_${p0Bin}_${alphaBin}`;
  }

  /**
   * Decode action from string key
   */
  private decodeAction(actionKey: string): { p0: number; alpha: number } {
    const parts = actionKey.split('_');
    const p0 = parseInt(parts[1]);
    const alpha = parseInt(parts[2]) / 100;

    return { p0, alpha };
  }

  /**
   * Predict SINR gain from P0/Alpha change
   *
   * Uses simplified path loss model with neighbor interference
   */
  private predictSinrGain(
    currentSinr: number,
    p0: number,
    alpha: number,
    neighbors: CellNode[],
    edges: InterferenceEdge[]
  ): number {
    // Simplified SINR prediction:
    // SINR_new = P_rx - (I + N)
    // where P_rx depends on P0 and alpha (Fractional Power Control)

    // Estimate received power change
    const p0Delta = p0 - (currentSinr - 10); // Rough approximation
    const alphaEffect = alpha * 3; // Alpha contributes ~3 dB at full FPC

    // Estimate interference from neighbors
    let interferenceDb = -100; // Start with thermal noise floor

    for (let i = 0; i < neighbors.length; i++) {
      const neighbor = neighbors[i];
      const edge = edges[i];

      // Neighbor interference contribution
      const neighborSinr = neighbor.features[0] || 0;
      const coupling = edge.interferenceCoupling || 60; // dB coupling loss
      const neighborInterference = neighborSinr - coupling;

      // Sum interference in linear domain
      interferenceDb = 10 * Math.log10(
        Math.pow(10, interferenceDb / 10) + Math.pow(10, neighborInterference / 10)
      );
    }

    // Predicted SINR with new P0/Alpha
    const predictedSinr = currentSinr + p0Delta * alpha - interferenceDb / 10;

    return predictedSinr - currentSinr;
  }

  /**
   * Calculate confidence score for optimization
   */
  private calculateConfidence(
    state: string,
    action: { p0: number; alpha: number; actionKey: string },
    transferKnowledge: OptimizationEpisode | null
  ): number {
    let confidence = 0.5; // Base confidence

    // Increase confidence if we have Q-value history
    const stateActions = this.qTable.get(state);
    if (stateActions) {
      const qValue = stateActions.get(action.actionKey) || 0;
      confidence += Math.min(0.3, qValue / 10);
    }

    // Increase confidence if transfer learning provides good match
    if (transferKnowledge) {
      // Assuming similarity was already computed to find this episode
      confidence += 0.2;
    }

    return Math.min(1.0, Math.max(0.0, confidence));
  }

  /**
   * Query similar episode using ruvector embeddings
   *
   * In production, this would call: npx ruvector query ./ruvector-spatial.db
   */
  private async querySimilarEpisode(queryEmbedding: number[]): Promise<OptimizationEpisode | null> {
    // Expand query embedding to 768 dimensions
    const fullEmbedding = this.expandEmbedding(queryEmbedding);

    // Search for most similar episode in memory
    let bestMatch: OptimizationEpisode | null = null;
    let bestSimilarity = -1;

    for (const [episodeId, embedding] of this.episodeEmbeddings.entries()) {
      const similarity = this.cosineSimilarity(fullEmbedding, embedding);

      if (similarity > bestSimilarity && similarity > 0.85) {
        bestSimilarity = similarity;
        const episode = this.episodes.get(episodeId);
        if (episode && episode.reward > 0) {
          bestMatch = episode;
        }
      }
    }

    if (bestMatch) {
      console.log(`[GNN-Optimizer] Found similar episode ${bestMatch.id} with similarity ${bestSimilarity.toFixed(3)}`);
    }

    return bestMatch;
  }

  /**
   * Expand GNN embedding to 768 dimensions for ruvector
   */
  private expandEmbedding(gnnEmbedding: number[]): number[] {
    const fullEmbedding = new Array(EMBEDDING_DIM).fill(0);

    // Repeat and interpolate GNN embedding to fill 768 dimensions
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      const sourceIdx = i % gnnEmbedding.length;
      fullEmbedding[i] = gnnEmbedding[sourceIdx];
    }

    // Normalize
    const norm = Math.sqrt(fullEmbedding.reduce((sum, v) => sum + v * v, 0));
    if (norm > 0) {
      for (let i = 0; i < EMBEDDING_DIM; i++) {
        fullEmbedding[i] /= norm;
      }
    }

    return fullEmbedding;
  }

  /**
   * Cosine similarity between two vectors
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Fallback optimization when no neighbors available
   */
  private fallbackOptimization(cellId: string, pmData: PMCounters): {
    p0: number;
    alpha: number;
    confidence: number;
    predictedSinrGain: number;
  } {
    const currentSinr = pmData.pmUlSinrMean || 0;

    // Simple heuristic: if SINR too low, increase P0 and alpha
    let p0 = -100; // Default P0
    let alpha = 0.8; // Default alpha

    if (currentSinr < TARGET_SINR_MIN) {
      p0 = -95; // Increase P0 to boost power
      alpha = 0.9; // High alpha for aggressive FPC
    } else if (currentSinr > TARGET_SINR_MAX) {
      p0 = -105; // Decrease P0 to save power
      alpha = 0.6; // Lower alpha
    }

    return {
      p0,
      alpha,
      confidence: 0.3, // Low confidence without neighbor context
      predictedSinrGain: (p0 + 100) * alpha / 10
    };
  }

  /**
   * Store optimization episode to agentdb reflexion and ruvector
   *
   * @param episode - Optimization episode to store
   */
  async storeEpisode(episode: OptimizationEpisode): Promise<void> {
    // Store episode in memory
    this.episodes.set(episode.id, episode);
    this.episodeEmbeddings.set(episode.id, episode.embedding);

    // Trim if exceeds max
    if (this.episodes.size > this.config.maxEpisodes) {
      const oldestId = this.episodes.keys().next().value;
      if (oldestId) {
        this.episodes.delete(oldestId);
        this.episodeEmbeddings.delete(oldestId);
      }
    }

    // Update Q-table with episode reward
    await this.updateQTable(episode);

    // Update GAT weights based on reward
    this.gat.updateWeights(episode.reward, this.config.learningRate);

    // Store to agentdb reflexion (simulated)
    await this.storeToAgentDB(episode);

    // Store embedding to ruvector (simulated)
    await this.storeToRuvector(episode);

    console.log(`[GNN-Optimizer] Stored episode ${episode.id} with reward ${episode.reward.toFixed(3)}`);

    this.emit('episode_stored', {
      episodeId: episode.id,
      reward: episode.reward,
      cellId: episode.cellId
    });
  }

  /**
   * Update Q-table with episode
   */
  private async updateQTable(episode: OptimizationEpisode): Promise<void> {
    const state = this.encodeState(episode.pmBefore);
    const nextState = this.encodeState(episode.pmAfter);
    const action = this.encodeAction(episode.action.p0, episode.action.alpha);

    // Initialize Q-table entries if needed
    if (!this.qTable.has(state)) {
      this.qTable.set(state, new Map());
    }
    if (!this.qTable.has(nextState)) {
      this.qTable.set(nextState, new Map());
    }

    const stateActions = this.qTable.get(state)!;
    const nextStateActions = this.qTable.get(nextState)!;

    // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    const currentQ = stateActions.get(action) || 0;
    const maxNextQ = nextStateActions.size > 0
      ? Math.max(...Array.from(nextStateActions.values()))
      : 0;

    const newQ = currentQ + this.config.learningRate * (
      episode.reward + this.config.discountFactor * maxNextQ - currentQ
    );

    stateActions.set(action, newQ);

    console.log(`[Q-Learning] Updated Q(${state}, ${action}) = ${newQ.toFixed(3)} (reward: ${episode.reward.toFixed(3)})`);
  }

  /**
   * Store episode to agentdb reflexion table
   *
   * Production implementation would use:
   * await agentdb.reflexion.store({ episode, state, action, reward, nextState })
   */
  private async storeToAgentDB(episode: OptimizationEpisode): Promise<void> {
    // Simulated agentdb storage
    const reflexionEntry = {
      episode: `p0_opt_${episode.cellId}_${episode.timestamp}`,
      state: episode.pmBefore,
      action: episode.action,
      reward: episode.reward,
      nextState: episode.pmAfter,
      timestamp: episode.timestamp,
      metadata: {
        cellId: episode.cellId,
        neighborCount: episode.neighborContext?.cellIds.length || 0,
        avgSinr: episode.neighborContext?.avgSinr || 0
      }
    };

    // In production: await agentdb.reflexion.store(reflexionEntry);
    console.log(`[AgentDB] Stored reflexion entry: ${reflexionEntry.episode}`);
  }

  /**
   * Store embedding to ruvector database
   *
   * Production implementation would use:
   * npx ruvector insert ./ruvector-spatial.db --id episode.id --vector episode.embedding
   */
  private async storeToRuvector(episode: OptimizationEpisode): Promise<void> {
    // Simulated ruvector storage
    console.log(`[Ruvector] Stored embedding for episode ${episode.id} (${EMBEDDING_DIM} dims)`);

    // In production, would execute:
    // await exec(`npx ruvector insert ${this.config.ruvectorDbPath} --id ${episode.id} --vector ${episode.embedding.join(',')}`);
  }

  /**
   * Create optimization episode from PM data before/after
   *
   * @param cellId - Cell identifier
   * @param pmBefore - PM counters before optimization
   * @param pmAfter - PM counters after optimization
   * @param action - P0/Alpha action taken
   * @returns Complete episode with embedding
   */
  createEpisode(
    cellId: string,
    pmBefore: PMCounters,
    pmAfter: PMCounters,
    action: { p0: number; alpha: number }
  ): OptimizationEpisode {
    // Calculate SINR improvement as reward
    const sinrBefore = pmBefore.pmUlSinrMean || 0;
    const sinrAfter = pmAfter.pmUlSinrMean || 0;
    const sinrDelta = sinrAfter - sinrBefore;

    // Reward function: primary = SINR gain, penalty for BLER increase
    const blerDelta = (pmAfter.pmUlBler || 0) - (pmBefore.pmUlBler || 0);
    const reward = sinrDelta - blerDelta * 5.0;

    // Get neighbor context
    const { neighbors, edges } = this.getNeighborContext(cellId);
    const neighborContext = {
      cellIds: neighbors.map(n => n.cellId),
      avgSinr: neighbors.length > 0
        ? neighbors.reduce((sum, n) => sum + (n.features[0] || 0), 0) / neighbors.length
        : 0,
      maxInterference: edges.length > 0
        ? Math.max(...edges.map(e => e.interferenceCoupling))
        : 0
    };

    // Create episode embedding (combine PM features + action)
    const embedding = this.createEpisodeEmbedding(pmBefore, pmAfter, action, neighborContext);

    const episode: OptimizationEpisode = {
      id: `p0_opt_${cellId}_${Date.now()}`,
      cellId,
      pmBefore,
      pmAfter,
      action,
      reward,
      timestamp: Date.now(),
      embedding,
      neighborContext
    };

    return episode;
  }

  /**
   * Create 768-dimensional embedding for episode
   */
  private createEpisodeEmbedding(
    pmBefore: PMCounters,
    pmAfter: PMCounters,
    action: { p0: number; alpha: number },
    neighborContext: { cellIds: string[]; avgSinr: number; maxInterference: number }
  ): number[] {
    // Feature vector for embedding
    const features: number[] = [
      // PM before (8 features)
      pmBefore.pmUlSinrMean || 0,
      pmBefore.pmUlBler || 0,
      pmBefore.pmUlRssi || 0,
      pmBefore.pmPuschPrbUsage || 0,
      pmBefore.pmCssr || 0,
      pmBefore.pmCallDropRate || 0,
      pmBefore.pmHoSuccessRate || 0,
      pmBefore.pmDlSinrMean || 0,

      // PM after (8 features)
      pmAfter.pmUlSinrMean || 0,
      pmAfter.pmUlBler || 0,
      pmAfter.pmUlRssi || 0,
      pmAfter.pmPuschPrbUsage || 0,
      pmAfter.pmCssr || 0,
      pmAfter.pmCallDropRate || 0,
      pmAfter.pmHoSuccessRate || 0,
      pmAfter.pmDlSinrMean || 0,

      // Action (2 features)
      action.p0,
      action.alpha,

      // Neighbor context (2 features)
      neighborContext.avgSinr,
      neighborContext.maxInterference
    ];

    // Expand to 768 dimensions by repeating and transforming
    const embedding = new Array(EMBEDDING_DIM).fill(0);

    for (let i = 0; i < EMBEDDING_DIM; i++) {
      const featureIdx = i % features.length;
      const layerIdx = Math.floor(i / features.length);

      // Apply non-linear transformation
      embedding[i] = Math.tanh(features[featureIdx] * (1 + layerIdx * 0.1));
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
    if (norm > 0) {
      for (let i = 0; i < EMBEDDING_DIM; i++) {
        embedding[i] /= norm;
      }
    }

    return embedding;
  }

  /**
   * Get optimizer statistics
   */
  getStats(): {
    cellCount: number;
    edgeCount: number;
    episodeCount: number;
    qTableSize: number;
    avgReward: number;
  } {
    const rewards = Array.from(this.episodes.values()).map(e => e.reward);
    const avgReward = rewards.length > 0
      ? rewards.reduce((sum, r) => sum + r, 0) / rewards.length
      : 0;

    return {
      cellCount: this.cellGraph.nodes.size,
      edgeCount: this.cellGraph.metadata.edgeCount,
      episodeCount: this.episodes.size,
      qTableSize: Array.from(this.qTable.values()).reduce((sum, actions) => sum + actions.size, 0),
      avgReward
    };
  }
}

// ============================================================
// Exports
// ============================================================

export {
  GraphAttentionNetwork,
  GNNUplinkOptimizer,
  GNNUplinkOptimizer as UplinkOptimizer, // Alias for backward compatibility
  P0_MIN,
  P0_MAX,
  ALPHA_MIN,
  ALPHA_MAX,
  TARGET_SINR_MIN,
  TARGET_SINR_MAX,
  EMBEDDING_DIM,
  NUM_ATTENTION_HEADS,
  GAT_HIDDEN_DIM
};
