/**
 * Graph Attention Network (GAT) for RAN Interference Modeling
 *
 * Implements multi-head attention mechanism for learning neighbor interference
 * patterns in cellular networks. Uses edge attention weights based on coupling
 * loss (RSRP, path loss, interference) to predict network-wide impact of
 * parameter changes.
 *
 * Key Features:
 * - Multi-head attention for diverse interference patterns
 * - Edge features: RSRP, path loss, coupling loss, azimuth
 * - Node features: PM counters, CM parameters, cell metadata
 * - Message passing for neighborhood aggregation
 *
 * @module ml/attention-gnn
 * @version 7.0.0-alpha.1
 */

import { EventEmitter } from 'events';
import type { PMCounters, CMParameters } from '../learning/self-learner.js';

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Cell node in the interference graph
 */
export interface CellNode {
  id: string;
  features: Float32Array;  // Node feature vector
  pm: PMCounters;
  cm: CMParameters;
  metadata: {
    sector: number;
    azimuth: number;  // degrees
    beamwidth: number;
    height: number;  // meters
    latitude: number;
    longitude: number;
  };
}

/**
 * Edge representing interference between cells
 */
export interface InterferenceEdge {
  source: string;  // Cell ID
  target: string;  // Cell ID
  features: Float32Array;  // Edge feature vector
  metadata: {
    rsrp: number;      // dBm - Reference Signal Received Power
    pathLoss: number;  // dB - Free space + shadowing
    couplingLoss: number;  // dB - Total coupling loss
    distance: number;  // meters
    azimuthDiff: number;  // degrees - relative azimuth
    interferenceRank: number;  // 1 = strongest interferer
  };
  attentionWeights?: number[];  // Per-head attention weights
}

/**
 * Interference graph for GAT
 */
export interface InterferenceGraph {
  nodes: Map<string, CellNode>;
  edges: Map<string, InterferenceEdge[]>;  // Adjacency list
  neighborhoodDepth: number;
}

/**
 * GAT layer configuration
 */
export interface GATConfig {
  numHeads: number;          // Number of attention heads
  hiddenDim: number;         // Hidden dimension per head
  nodeFeatureDim: number;    // Input node feature dimension
  edgeFeatureDim: number;    // Input edge feature dimension
  outputDim: number;         // Output dimension
  dropout: number;           // Dropout probability
  leakyReluAlpha: number;    // LeakyReLU negative slope
  attentionDropout: number;  // Attention dropout
}

/**
 * Attention output for interpretability
 */
export interface AttentionOutput {
  cellId: string;
  neighborAttentions: Map<string, number[]>;  // neighbor -> per-head attention
  aggregatedFeatures: Float32Array;
  topNeighbors: Array<{
    cellId: string;
    avgAttention: number;
    rsrp: number;
  }>;
}

/**
 * Graph propagation result
 */
export interface PropagationResult {
  affectedCells: Map<string, {
    originalPM: PMCounters;
    predictedPM: PMCounters;
    impactScore: number;  // 0-1, higher = more affected
    propagationDepth: number;
  }>;
  totalImpactScore: number;
  propagationTime: number;  // milliseconds
}

// ============================================================================
// Graph Attention Network Implementation
// ============================================================================

/**
 * Multi-head Graph Attention Network for RAN interference modeling
 *
 * Uses attention mechanism to learn which neighbors matter most for
 * predicting the impact of parameter changes.
 */
export class GraphAttentionNetwork extends EventEmitter {
  private config: GATConfig;
  private graph: InterferenceGraph;

  // Learnable parameters (in production, these would be trained)
  private W_node!: Float32Array[];      // Node transformation weights [numHeads]
  private W_edge!: Float32Array[];      // Edge transformation weights [numHeads]
  private a_src!: Float32Array[];       // Source attention weights [numHeads]
  private a_tgt!: Float32Array[];       // Target attention weights [numHeads]
  private W_out!: Float32Array;         // Output projection weights

  private trainingMode: boolean = false;
  private attentionCache: Map<string, AttentionOutput>;

  constructor(config?: Partial<GATConfig>) {
    super();

    // RAN-optimized defaults
    this.config = {
      numHeads: config?.numHeads || 8,
      hiddenDim: config?.hiddenDim || 64,
      nodeFeatureDim: config?.nodeFeatureDim || 128,
      edgeFeatureDim: config?.edgeFeatureDim || 32,
      outputDim: config?.outputDim || 128,
      dropout: config?.dropout || 0.1,
      leakyReluAlpha: config?.leakyReluAlpha || 0.2,
      attentionDropout: config?.attentionDropout || 0.1
    };

    this.graph = {
      nodes: new Map(),
      edges: new Map(),
      neighborhoodDepth: 2
    };

    this.attentionCache = new Map();

    // Initialize weights (random initialization)
    this.initializeWeights();

    console.log('[GAT] Initialized Graph Attention Network');
    console.log(`[GAT] Heads: ${this.config.numHeads}, Hidden: ${this.config.hiddenDim}`);
  }

  /**
   * Initialize learnable parameters
   */
  private initializeWeights(): void {
    const { numHeads, hiddenDim, nodeFeatureDim, edgeFeatureDim, outputDim } = this.config;

    this.W_node = [];
    this.W_edge = [];
    this.a_src = [];
    this.a_tgt = [];

    for (let h = 0; h < numHeads; h++) {
      // Xavier initialization
      this.W_node.push(this.randomMatrix(nodeFeatureDim, hiddenDim));
      this.W_edge.push(this.randomMatrix(edgeFeatureDim, hiddenDim));
      this.a_src.push(this.randomMatrix(hiddenDim, 1));
      this.a_tgt.push(this.randomMatrix(hiddenDim, 1));
    }

    this.W_out = this.randomMatrix(numHeads * hiddenDim, outputDim);
  }

  /**
   * Add a cell to the interference graph
   */
  addCell(cell: CellNode): void {
    // Create node features from PM/CM parameters
    const features = this.createNodeFeatures(cell);
    cell.features = features;

    this.graph.nodes.set(cell.id, cell);

    if (!this.graph.edges.has(cell.id)) {
      this.graph.edges.set(cell.id, []);
    }

    console.log(`[GAT] Added cell ${cell.id} to graph`);
    this.emit('cell_added', { cellId: cell.id });
  }

  /**
   * Add an interference edge between two cells
   */
  addInterferenceEdge(edge: InterferenceEdge): void {
    // Create edge features
    const features = this.createEdgeFeatures(edge);
    edge.features = features;

    // Add bidirectional edge
    if (!this.graph.edges.has(edge.source)) {
      this.graph.edges.set(edge.source, []);
    }
    if (!this.graph.edges.has(edge.target)) {
      this.graph.edges.set(edge.target, []);
    }

    this.graph.edges.get(edge.source)!.push(edge);

    // Add reverse edge
    const reverseEdge: InterferenceEdge = {
      source: edge.target,
      target: edge.source,
      features: edge.features,
      metadata: { ...edge.metadata }
    };
    this.graph.edges.get(edge.target)!.push(reverseEdge);

    console.log(`[GAT] Added edge ${edge.source} -> ${edge.target} (RSRP: ${edge.metadata.rsrp.toFixed(1)} dBm)`);
  }

  /**
   * Compute multi-head attention for a cell
   *
   * Returns attention weights and aggregated neighbor features
   */
  computeAttention(cellId: string): AttentionOutput {
    const node = this.graph.nodes.get(cellId);
    if (!node) {
      throw new Error(`Cell ${cellId} not found in graph`);
    }

    const neighbors = this.graph.edges.get(cellId) || [];
    const neighborAttentions = new Map<string, number[]>();

    // Multi-head attention
    const headOutputs: Float32Array[] = [];

    for (let h = 0; h < this.config.numHeads; h++) {
      const { aggregated, attentions } = this.computeHeadAttention(
        node,
        neighbors,
        h
      );

      headOutputs.push(aggregated);

      // Store attention weights
      for (let i = 0; i < neighbors.length; i++) {
        const neighborId = neighbors[i].target;
        if (!neighborAttentions.has(neighborId)) {
          neighborAttentions.set(neighborId, []);
        }
        neighborAttentions.get(neighborId)!.push(attentions[i]);
      }
    }

    // Concatenate head outputs
    const concatenated = this.concatenate(headOutputs);

    // Output projection
    const aggregatedFeatures = this.matmul(concatenated, this.W_out);

    // Find top neighbors by average attention
    const topNeighbors = Array.from(neighborAttentions.entries())
      .map(([neighborId, attentions]) => {
        const avgAttention = attentions.reduce((a, b) => a + b, 0) / attentions.length;
        const edge = neighbors.find(e => e.target === neighborId)!;
        return {
          cellId: neighborId,
          avgAttention,
          rsrp: edge.metadata.rsrp
        };
      })
      .sort((a, b) => b.avgAttention - a.avgAttention)
      .slice(0, 5);

    const output: AttentionOutput = {
      cellId,
      neighborAttentions,
      aggregatedFeatures,
      topNeighbors
    };

    // Cache for efficiency
    this.attentionCache.set(cellId, output);

    return output;
  }

  /**
   * Compute attention for a single head
   */
  private computeHeadAttention(
    node: CellNode,
    neighbors: InterferenceEdge[],
    headIdx: number
  ): { aggregated: Float32Array; attentions: number[] } {
    if (neighbors.length === 0) {
      return {
        aggregated: new Float32Array(this.config.hiddenDim),
        attentions: []
      };
    }

    // Transform source node features
    const h_src = this.matmul(node.features, this.W_node[headIdx]);

    // Compute attention scores for each neighbor
    const attentionScores: number[] = [];

    for (const edge of neighbors) {
      const neighborNode = this.graph.nodes.get(edge.target)!;
      const h_tgt = this.matmul(neighborNode.features, this.W_node[headIdx]);

      // Edge-aware attention
      const h_edge = this.matmul(edge.features, this.W_edge[headIdx]);

      // Attention coefficient: e_ij = LeakyReLU(a^T [W h_i || W h_j || W e_ij])
      const combined = this.concatenate([h_src, h_tgt, h_edge]);
      const e_ij = this.leakyReLU(this.dot(combined, this.a_src[headIdx]));

      attentionScores.push(e_ij);
    }

    // Softmax normalization
    const attentions = this.softmax(attentionScores);

    // Weighted aggregation of neighbor features
    const aggregated = new Float32Array(this.config.hiddenDim);

    for (let i = 0; i < neighbors.length; i++) {
      const neighborNode = this.graph.nodes.get(neighbors[i].target)!;
      const h_tgt = this.matmul(neighborNode.features, this.W_node[headIdx]);

      for (let j = 0; j < this.config.hiddenDim; j++) {
        aggregated[j] += attentions[i] * h_tgt[j];
      }
    }

    // Apply activation
    for (let j = 0; j < aggregated.length; j++) {
      aggregated[j] = this.elu(aggregated[j]);
    }

    return { aggregated, attentions };
  }

  /**
   * Predict network-wide impact of parameter change using graph propagation
   *
   * This is the key method for "what-if" analysis:
   * "If I change P0 on cell X, what happens to neighbors?"
   */
  async predictPropagation(
    cellId: string,
    parameterChange: Partial<CMParameters>,
    maxDepth: number = 2
  ): Promise<PropagationResult> {
    const startTime = performance.now();

    const affectedCells = new Map<string, {
      originalPM: PMCounters;
      predictedPM: PMCounters;
      impactScore: number;
      propagationDepth: number;
    }>();

    // BFS propagation with attention-weighted impact
    const visited = new Set<string>();
    const queue: Array<{ cellId: string; depth: number; incomingImpact: number }> = [
      { cellId, depth: 0, incomingImpact: 1.0 }
    ];

    while (queue.length > 0) {
      const { cellId: currentId, depth, incomingImpact } = queue.shift()!;

      if (visited.has(currentId) || depth > maxDepth) {
        continue;
      }

      visited.add(currentId);

      const node = this.graph.nodes.get(currentId);
      if (!node) continue;

      // Compute attention to determine which neighbors are affected
      const attention = this.computeAttention(currentId);

      // Predict PM changes based on parameter change and neighbor impact
      const originalPM = node.pm;
      const predictedPM = this.predictPMChange(
        originalPM,
        parameterChange,
        incomingImpact,
        depth === 0  // Is source cell
      );

      const impactScore = this.calculateImpactScore(originalPM, predictedPM);

      affectedCells.set(currentId, {
        originalPM,
        predictedPM,
        impactScore,
        propagationDepth: depth
      });

      // Propagate to neighbors using attention weights
      const neighbors = this.graph.edges.get(currentId) || [];

      for (const edge of neighbors) {
        const neighborAttention = attention.neighborAttentions.get(edge.target);
        if (!neighborAttention) continue;

        const avgAttention = neighborAttention.reduce((a, b) => a + b, 0) / neighborAttention.length;

        // Attention weight determines how much impact propagates
        const propagatedImpact = incomingImpact * avgAttention * this.getCouplingFactor(edge);

        // Only propagate if impact is significant
        if (propagatedImpact > 0.05 && !visited.has(edge.target)) {
          queue.push({
            cellId: edge.target,
            depth: depth + 1,
            incomingImpact: propagatedImpact
          });
        }
      }
    }

    const totalImpactScore = Array.from(affectedCells.values())
      .reduce((sum, cell) => sum + cell.impactScore, 0);

    const propagationTime = performance.now() - startTime;

    console.log(`[GAT] Predicted impact on ${affectedCells.size} cells in ${propagationTime.toFixed(2)}ms`);
    console.log(`[GAT] Total impact score: ${totalImpactScore.toFixed(3)}`);

    this.emit('propagation_complete', {
      cellId,
      affectedCount: affectedCells.size,
      totalImpactScore,
      propagationTime
    });

    return {
      affectedCells,
      totalImpactScore,
      propagationTime
    };
  }

  /**
   * Build interference graph from cell measurements
   */
  buildGraphFromMeasurements(
    cells: CellNode[],
    measurements: Array<{
      source: string;
      target: string;
      rsrp: number;
      rsrq: number;
    }>
  ): void {
    console.log(`[GAT] Building graph from ${cells.length} cells and ${measurements.length} measurements`);

    // Add all cells
    for (const cell of cells) {
      this.addCell(cell);
    }

    // Add edges from measurements
    for (const meas of measurements) {
      const sourceNode = this.graph.nodes.get(meas.source);
      const targetNode = this.graph.nodes.get(meas.target);

      if (!sourceNode || !targetNode) continue;

      // Calculate path loss and distance
      const distance = this.haversineDistance(
        sourceNode.metadata.latitude,
        sourceNode.metadata.longitude,
        targetNode.metadata.latitude,
        targetNode.metadata.longitude
      );

      const pathLoss = this.calculatePathLoss(distance, 2100);  // 2.1 GHz

      // Coupling loss = path loss - antenna gains
      const couplingLoss = pathLoss + 10;  // Simplified

      // Azimuth difference
      const azimuthDiff = Math.abs(sourceNode.metadata.azimuth - targetNode.metadata.azimuth);

      const edge: InterferenceEdge = {
        source: meas.source,
        target: meas.target,
        features: new Float32Array(),  // Will be filled in addInterferenceEdge
        metadata: {
          rsrp: meas.rsrp,
          pathLoss,
          couplingLoss,
          distance,
          azimuthDiff,
          interferenceRank: 0  // Will be calculated
        }
      };

      this.addInterferenceEdge(edge);
    }

    // Calculate interference ranks
    this.calculateInterferenceRanks();

    console.log(`[GAT] Graph built: ${this.graph.nodes.size} nodes, ${Array.from(this.graph.edges.values()).reduce((sum, edges) => sum + edges.length, 0)} edges`);
  }

  /**
   * Get graph statistics
   */
  getStats(): {
    nodeCount: number;
    edgeCount: number;
    avgDegree: number;
    maxDegree: number;
    attentionCacheSize: number;
  } {
    const degrees: number[] = [];

    for (const edges of this.graph.edges.values()) {
      degrees.push(edges.length);
    }

    const avgDegree = degrees.length > 0
      ? degrees.reduce((a, b) => a + b, 0) / degrees.length
      : 0;

    const maxDegree = degrees.length > 0 ? Math.max(...degrees) : 0;

    return {
      nodeCount: this.graph.nodes.size,
      edgeCount: Array.from(this.graph.edges.values()).reduce((sum, e) => sum + e.length, 0),
      avgDegree,
      maxDegree,
      attentionCacheSize: this.attentionCache.size
    };
  }

  // ============================================================================
  // Private Helper Methods
  // ============================================================================

  private createNodeFeatures(cell: CellNode): Float32Array {
    const features: number[] = [
      // PM counters (normalized)
      this.normalize(cell.pm.pmUlSinrMean || 0, -20, 30),
      this.normalize(cell.pm.pmDlSinrMean || 0, -20, 30),
      this.normalize(cell.pm.pmUlBler || 0, 0, 0.1),
      this.normalize(cell.pm.pmDlBler || 0, 0, 0.1),
      this.normalize(cell.pm.pmCssr || 0, 0, 1),
      this.normalize(cell.pm.pmCallDropRate || 0, 0, 0.05),
      this.normalize(cell.pm.pmHoSuccessRate || 0, 0, 1),
      this.normalize(cell.pm.pmPuschPrbUsage || 0, 0, 100),

      // CM parameters
      this.normalize(cell.cm.p0NominalPUSCH || -103, -130, -80),
      this.normalize(cell.cm.alpha || 0.8, 0, 1),
      this.normalize(cell.cm.electricalTilt || 0, 0, 15),
      this.normalize(cell.cm.txPower || 40, 0, 46),

      // Cell metadata
      this.normalize(cell.metadata.sector, 0, 2),
      this.normalize(cell.metadata.azimuth, 0, 360),
      this.normalize(cell.metadata.beamwidth, 0, 120),
      this.normalize(cell.metadata.height, 0, 100),
    ];

    // Pad to nodeFeatureDim
    while (features.length < this.config.nodeFeatureDim) {
      features.push(0);
    }

    return new Float32Array(features.slice(0, this.config.nodeFeatureDim));
  }

  private createEdgeFeatures(edge: InterferenceEdge): Float32Array {
    const features: number[] = [
      this.normalize(edge.metadata.rsrp, -140, -60),
      this.normalize(edge.metadata.pathLoss, 60, 160),
      this.normalize(edge.metadata.couplingLoss, 60, 160),
      this.normalize(edge.metadata.distance, 0, 10000),
      this.normalize(edge.metadata.azimuthDiff, 0, 180),
      this.normalize(edge.metadata.interferenceRank, 1, 20),
    ];

    // Pad to edgeFeatureDim
    while (features.length < this.config.edgeFeatureDim) {
      features.push(0);
    }

    return new Float32Array(features.slice(0, this.config.edgeFeatureDim));
  }

  private predictPMChange(
    originalPM: PMCounters,
    paramChange: Partial<CMParameters>,
    incomingImpact: number,
    isSource: boolean
  ): PMCounters {
    // Simplified prediction model (in production, use trained GNN)
    const predictedPM: PMCounters = { ...originalPM };

    if (isSource) {
      // Direct impact on source cell
      if (paramChange.p0NominalPUSCH !== undefined) {
        const p0Delta = (paramChange.p0NominalPUSCH - (originalPM as any).p0 || -103);
        predictedPM.pmUlSinrMean = (originalPM.pmUlSinrMean || 0) + p0Delta * 0.1;
      }

      if (paramChange.electricalTilt !== undefined) {
        const tiltDelta = paramChange.electricalTilt - 0;
        predictedPM.pmDlSinrMean = (originalPM.pmDlSinrMean || 0) - tiltDelta * 0.3;
      }
    } else {
      // Neighbor impact (interference)
      predictedPM.pmUlSinrMean = (originalPM.pmUlSinrMean || 0) - incomingImpact * 0.5;
      predictedPM.pmDlSinrMean = (originalPM.pmDlSinrMean || 0) - incomingImpact * 0.5;
    }

    return predictedPM;
  }

  private calculateImpactScore(original: PMCounters, predicted: PMCounters): number {
    const sinrDelta = Math.abs((predicted.pmUlSinrMean || 0) - (original.pmUlSinrMean || 0));
    const cssrDelta = Math.abs((predicted.pmCssr || 0) - (original.pmCssr || 0));
    const dropDelta = Math.abs((predicted.pmCallDropRate || 0) - (original.pmCallDropRate || 0));

    return (sinrDelta * 0.4 + cssrDelta * 100 * 0.3 + dropDelta * 100 * 0.3) / 10;
  }

  private getCouplingFactor(edge: InterferenceEdge): number {
    // Stronger coupling = more impact propagates
    const rsrpNorm = this.normalize(edge.metadata.rsrp, -140, -60);
    return rsrpNorm;
  }

  private calculateInterferenceRanks(): void {
    for (const [cellId, edges] of this.graph.edges) {
      // Sort by RSRP (strongest first)
      edges.sort((a, b) => b.metadata.rsrp - a.metadata.rsrp);

      // Assign ranks
      edges.forEach((edge, idx) => {
        edge.metadata.interferenceRank = idx + 1;
      });
    }
  }

  private haversineDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const R = 6371e3;  // Earth radius in meters
    const φ1 = lat1 * Math.PI / 180;
    const φ2 = lat2 * Math.PI / 180;
    const Δφ = (lat2 - lat1) * Math.PI / 180;
    const Δλ = (lon2 - lon1) * Math.PI / 180;

    const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
      Math.cos(φ1) * Math.cos(φ2) *
      Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
  }

  private calculatePathLoss(distance: number, frequency: number): number {
    // Free space path loss (Friis equation)
    const c = 3e8;  // Speed of light
    const fspl = 20 * Math.log10(distance) + 20 * Math.log10(frequency * 1e6) + 20 * Math.log10(4 * Math.PI / c);
    return fspl;
  }

  // Linear algebra utilities

  private randomMatrix(rows: number, cols: number): Float32Array {
    const size = rows * cols;
    const matrix = new Float32Array(size);
    const scale = Math.sqrt(2.0 / (rows + cols));  // Xavier initialization

    for (let i = 0; i < size; i++) {
      matrix[i] = (Math.random() * 2 - 1) * scale;
    }

    return matrix;
  }

  private matmul(vec: Float32Array, matrix: Float32Array): Float32Array {
    // Assume matrix is stored row-major
    const outDim = Math.floor(matrix.length / vec.length);
    const result = new Float32Array(outDim);

    for (let i = 0; i < outDim; i++) {
      let sum = 0;
      for (let j = 0; j < vec.length; j++) {
        sum += vec[j] * matrix[i * vec.length + j];
      }
      result[i] = sum;
    }

    return result;
  }

  private dot(vec1: Float32Array, vec2: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < vec1.length; i++) {
      sum += vec1[i] * vec2[i];
    }
    return sum;
  }

  private concatenate(vecs: Float32Array[]): Float32Array {
    const totalLength = vecs.reduce((sum, v) => sum + v.length, 0);
    const result = new Float32Array(totalLength);

    let offset = 0;
    for (const vec of vecs) {
      result.set(vec, offset);
      offset += vec.length;
    }

    return result;
  }

  private softmax(scores: number[]): number[] {
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(e => e / sumExp);
  }

  private leakyReLU(x: number): number {
    return x > 0 ? x : this.config.leakyReluAlpha * x;
  }

  private elu(x: number, alpha: number = 1.0): number {
    return x > 0 ? x : alpha * (Math.exp(x) - 1);
  }

  private normalize(value: number, min: number, max: number): number {
    return (value - min) / (max - min);
  }
}

// ============================================================================
// Exports
// ============================================================================

// export {
//   GraphAttentionNetwork,
//   type CellNode,
//   type InterferenceEdge,
//   type InterferenceGraph,
//   type GATConfig,
//   type AttentionOutput,
//   type PropagationResult
// };
