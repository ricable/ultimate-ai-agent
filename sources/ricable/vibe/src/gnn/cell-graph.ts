/**
 * Graph Neural Network for Cell Neighbor Relations and SINR Analysis
 * Uses ruvector GNN capabilities for graph-based analysis
 */

import type {
  CellGraph,
  CellGraphNode,
  CellGraphEdge,
  NeighborRelation,
  CellKPISnapshot,
} from '../models/ran-kpi.js';

// ============================================================================
// CELL GRAPH BUILDER
// ============================================================================

export interface CellFeatureConfig {
  includeAccessibility: boolean;
  includeRetainability: boolean;
  includeRadioQuality: boolean;
  includeMobility: boolean;
  includeUplinkInterference: boolean;
  includeUplinkPowerControl: boolean;
}

export const DEFAULT_FEATURE_CONFIG: CellFeatureConfig = {
  includeAccessibility: true,
  includeRetainability: true,
  includeRadioQuality: true,
  includeMobility: true,
  includeUplinkInterference: true,
  includeUplinkPowerControl: true,
};

export class CellGraphBuilder {
  private featureConfig: CellFeatureConfig;

  constructor(featureConfig: Partial<CellFeatureConfig> = {}) {
    this.featureConfig = { ...DEFAULT_FEATURE_CONFIG, ...featureConfig };
  }

  /**
   * Build a cell graph from KPI snapshots and neighbor relations
   */
  buildGraph(
    cellSnapshots: Map<string, CellKPISnapshot>,
    neighborRelations: NeighborRelation[]
  ): CellGraph {
    const nodes: CellGraphNode[] = [];
    const edges: CellGraphEdge[] = [];

    // Build nodes
    for (const [cellId, snapshot] of cellSnapshots) {
      const features = this.extractNodeFeatures(snapshot);
      nodes.push({
        id: cellId,
        cellId,
        features,
        position: snapshot.cell.latitude && snapshot.cell.longitude
          ? [snapshot.cell.latitude, snapshot.cell.longitude]
          : undefined,
      });
    }

    // Build edges from neighbor relations
    for (const relation of neighborRelations) {
      if (
        cellSnapshots.has(relation.sourceCellId) &&
        cellSnapshots.has(relation.targetCellId)
      ) {
        const edgeFeatures = this.extractEdgeFeatures(relation);
        edges.push({
          source: relation.sourceCellId,
          target: relation.targetCellId,
          relationshipType: relation.relationshipType,
          features: edgeFeatures,
          weight: this.calculateEdgeWeight(relation),
        });
      }
    }

    // Build adjacency matrix
    const nodeIds = nodes.map(n => n.id);
    const adjacencyMatrix = this.buildAdjacencyMatrix(nodeIds, edges);

    return {
      nodes,
      edges,
      adjacencyMatrix,
      metadata: {
        timestamp: new Date(),
        numCells: nodes.length,
        numRelations: edges.length,
      },
    };
  }

  /**
   * Extract feature vector from cell KPI snapshot
   */
  private extractNodeFeatures(snapshot: CellKPISnapshot): number[] {
    const features: number[] = [];

    if (this.featureConfig.includeAccessibility) {
      features.push(
        this.normalize(snapshot.accessibility.rrcSetupSuccessRate, 90, 100),
        this.normalize(snapshot.accessibility.erabSetupSuccessRate, 90, 100),
        this.normalize(snapshot.accessibility.initialContextSetupSuccessRate, 90, 100)
      );
    }

    if (this.featureConfig.includeRetainability) {
      features.push(
        1 - this.normalize(snapshot.retainability.erabDropRate, 0, 5), // Invert: lower is better
        this.normalize(snapshot.retainability.dataSessionRetainability, 95, 100)
      );
    }

    if (this.featureConfig.includeRadioQuality) {
      features.push(
        this.normalize(snapshot.radioQuality.dlAvgCqi, 0, 15),
        this.normalize(snapshot.radioQuality.ulSinrAvg, -5, 30),
        this.normalize(snapshot.radioQuality.rsrpAvg, -140, -80),
        this.normalize(snapshot.radioQuality.rsrqAvg, -25, 0),
        this.normalize(snapshot.radioQuality.dlSpectralEfficiency, 0, 10)
      );
    }

    if (this.featureConfig.includeMobility) {
      features.push(
        this.normalize(snapshot.mobility.intraFreqHoSuccessRate, 90, 100),
        this.normalize(snapshot.mobility.interFreqHoSuccessRate, 90, 100),
        1 - this.normalize(snapshot.mobility.pingPongHo, 0, 50) // Invert: lower is better
      );
    }

    if (this.featureConfig.includeUplinkInterference) {
      features.push(
        1 - this.normalize(snapshot.uplinkInterference.iotAvg, 0, 20), // Invert: lower is better
        1 - this.normalize(snapshot.uplinkInterference.highInterferencePrbRatio, 0, 50)
      );
    }

    if (this.featureConfig.includeUplinkPowerControl) {
      features.push(
        this.normalize(snapshot.uplinkPowerControl.p0NominalPusch, -126, 24),
        this.normalize(snapshot.uplinkPowerControl.alpha, 0, 1),
        this.normalize(snapshot.uplinkPowerControl.powerHeadroomAvg, -20, 40),
        1 - this.normalize(snapshot.uplinkPowerControl.powerLimitedUeRatio, 0, 50),
        1 - this.normalize(snapshot.uplinkPowerControl.negativePowerHeadroomRatio, 0, 50)
      );
    }

    return features;
  }

  /**
   * Extract edge features from neighbor relation
   */
  private extractEdgeFeatures(relation: NeighborRelation): number[] {
    return [
      this.normalize(relation.sourceSinr, -5, 30),
      this.normalize(relation.targetSinr, -5, 30),
      relation.targetSinr - relation.sourceSinr, // SINR delta (normalized)
      this.normalize(relation.sourceRsrp, -140, -80),
      this.normalize(relation.targetRsrp, -140, -80),
      relation.targetRsrp - relation.sourceRsrp, // RSRP delta
      this.normalize(relation.hoSuccessRate, 0, 100),
      this.normalize(relation.a3Offset, -24, 24),
      this.normalize(relation.hysteresis, 0, 10),
      this.normalize(relation.timeToTrigger, 0, 5120),
      relation.relationshipType === 'intra-freq' ? 1 : 0,
      relation.relationshipType === 'inter-freq' ? 1 : 0,
      relation.relationshipType === 'inter-rat' ? 1 : 0,
    ];
  }

  /**
   * Calculate edge weight based on handover success and signal quality
   */
  private calculateEdgeWeight(relation: NeighborRelation): number {
    const hoWeight = relation.hoSuccessRate / 100;
    const sinrWeight = Math.max(0, (relation.targetSinr + 5) / 35);
    const rsrpWeight = Math.max(0, (relation.targetRsrp + 140) / 60);

    return (hoWeight * 0.5 + sinrWeight * 0.3 + rsrpWeight * 0.2);
  }

  /**
   * Build adjacency matrix from edges
   */
  private buildAdjacencyMatrix(nodeIds: string[], edges: CellGraphEdge[]): number[][] {
    const n = nodeIds.length;
    const matrix: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));

    const idToIndex = new Map<string, number>();
    nodeIds.forEach((id, idx) => idToIndex.set(id, idx));

    for (const edge of edges) {
      const sourceIdx = idToIndex.get(edge.source);
      const targetIdx = idToIndex.get(edge.target);

      if (sourceIdx !== undefined && targetIdx !== undefined) {
        matrix[sourceIdx][targetIdx] = edge.weight;
        matrix[targetIdx][sourceIdx] = edge.weight; // Symmetric for undirected
      }
    }

    return matrix;
  }

  /**
   * Normalize value to [0, 1] range
   */
  private normalize(value: number, min: number, max: number): number {
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
  }
}

// ============================================================================
// GNN LAYER IMPLEMENTATION
// ============================================================================

export interface GNNConfig {
  inputDim: number;
  hiddenDim: number;
  outputDim: number;
  numLayers: number;
  numHeads: number; // For multi-head attention
  dropout: number;
  aggregationType: 'mean' | 'sum' | 'max' | 'attention';
}

export const DEFAULT_GNN_CONFIG: GNNConfig = {
  inputDim: 20,
  hiddenDim: 64,
  outputDim: 32,
  numLayers: 3,
  numHeads: 4,
  dropout: 0.1,
  aggregationType: 'attention',
};

/**
 * Graph Neural Network Layer for Cell Analysis
 * Implements message passing with multi-head attention
 */
export class GNNLayer {
  private config: GNNConfig;
  private weights: {
    W_query: number[][];
    W_key: number[][];
    W_value: number[][];
    W_out: number[][];
  };

  constructor(config: Partial<GNNConfig> = {}) {
    this.config = { ...DEFAULT_GNN_CONFIG, ...config };
    this.weights = this.initializeWeights();
  }

  /**
   * Initialize layer weights (Xavier initialization)
   */
  private initializeWeights(): typeof this.weights {
    const { inputDim, hiddenDim, outputDim, numHeads } = this.config;
    const headDim = hiddenDim / numHeads;

    return {
      W_query: this.xavierInit(inputDim, hiddenDim),
      W_key: this.xavierInit(inputDim, hiddenDim),
      W_value: this.xavierInit(inputDim, hiddenDim),
      W_out: this.xavierInit(hiddenDim, outputDim),
    };
  }

  private xavierInit(fanIn: number, fanOut: number): number[][] {
    const std = Math.sqrt(2 / (fanIn + fanOut));
    return Array(fanIn).fill(null).map(() =>
      Array(fanOut).fill(null).map(() => (Math.random() * 2 - 1) * std)
    );
  }

  /**
   * Forward pass through GNN layer
   */
  forward(
    nodeFeatures: number[][],
    adjacencyMatrix: number[][],
    edgeFeatures?: number[][][]
  ): number[][] {
    const numNodes = nodeFeatures.length;

    // Compute queries, keys, values for attention
    const queries = this.matmul(nodeFeatures, this.weights.W_query);
    const keys = this.matmul(nodeFeatures, this.weights.W_key);
    const values = this.matmul(nodeFeatures, this.weights.W_value);

    // Multi-head attention aggregation
    const aggregated = this.multiHeadAttention(
      queries,
      keys,
      values,
      adjacencyMatrix,
      edgeFeatures
    );

    // Output projection
    const output = this.matmul(aggregated, this.weights.W_out);

    // Residual connection and normalization
    return this.layerNorm(this.addResidual(nodeFeatures, output));
  }

  /**
   * Multi-head attention mechanism
   */
  private multiHeadAttention(
    queries: number[][],
    keys: number[][],
    values: number[][],
    adjacencyMatrix: number[][],
    edgeFeatures?: number[][][]
  ): number[][] {
    const numNodes = queries.length;
    const hiddenDim = this.config.hiddenDim;
    const numHeads = this.config.numHeads;
    const headDim = hiddenDim / numHeads;

    const result: number[][] = Array(numNodes).fill(null).map(() => Array(hiddenDim).fill(0));

    // For each head
    for (let h = 0; h < numHeads; h++) {
      const startIdx = h * headDim;
      const endIdx = (h + 1) * headDim;

      for (let i = 0; i < numNodes; i++) {
        // Attention scores
        const scores: number[] = [];
        for (let j = 0; j < numNodes; j++) {
          if (adjacencyMatrix[i][j] > 0) {
            // Query-key dot product
            let score = 0;
            for (let k = startIdx; k < endIdx; k++) {
              score += queries[i][k] * keys[j][k];
            }
            score /= Math.sqrt(headDim);

            // Add edge features if available
            if (edgeFeatures && edgeFeatures[i] && edgeFeatures[i][j]) {
              score += edgeFeatures[i][j].reduce((a, b) => a + b, 0) * 0.1;
            }

            scores.push(score);
          } else {
            scores.push(-Infinity);
          }
        }

        // Softmax
        const maxScore = Math.max(...scores.filter(s => isFinite(s)));
        const expScores = scores.map(s => isFinite(s) ? Math.exp(s - maxScore) : 0);
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const attentionWeights = expScores.map(s => s / (sumExp || 1));

        // Weighted sum of values
        for (let k = startIdx; k < endIdx; k++) {
          for (let j = 0; j < numNodes; j++) {
            result[i][k] += attentionWeights[j] * values[j][k];
          }
        }
      }
    }

    return result;
  }

  /**
   * Matrix multiplication
   */
  private matmul(a: number[][], b: number[][]): number[][] {
    const m = a.length;
    const n = b[0].length;
    const k = b.length;

    const result: number[][] = Array(m).fill(null).map(() => Array(n).fill(0));

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        for (let l = 0; l < k; l++) {
          result[i][j] += a[i][l] * b[l][j];
        }
      }
    }

    return result;
  }

  /**
   * Add residual connection with dimension matching
   */
  private addResidual(original: number[][], transformed: number[][]): number[][] {
    const m = original.length;
    const n = transformed[0].length;
    const result: number[][] = Array(m).fill(null).map(() => Array(n).fill(0));

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n && j < original[0].length; j++) {
        result[i][j] = transformed[i][j] + original[i][j];
      }
      for (let j = original[0].length; j < n; j++) {
        result[i][j] = transformed[i][j];
      }
    }

    return result;
  }

  /**
   * Layer normalization
   */
  private layerNorm(x: number[][]): number[][] {
    const eps = 1e-6;
    const result: number[][] = [];

    for (const row of x) {
      const mean = row.reduce((a, b) => a + b, 0) / row.length;
      const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
      const std = Math.sqrt(variance + eps);
      result.push(row.map(v => (v - mean) / std));
    }

    return result;
  }
}

// ============================================================================
// CELL GNN MODEL
// ============================================================================

export class CellGNN {
  private layers: GNNLayer[];
  private config: GNNConfig;

  constructor(config: Partial<GNNConfig> = {}) {
    this.config = { ...DEFAULT_GNN_CONFIG, ...config };
    this.layers = [];

    // Build layers
    for (let i = 0; i < this.config.numLayers; i++) {
      const inputDim = i === 0 ? this.config.inputDim : this.config.hiddenDim;
      const outputDim = i === this.config.numLayers - 1 ? this.config.outputDim : this.config.hiddenDim;

      this.layers.push(new GNNLayer({
        ...this.config,
        inputDim,
        outputDim,
      }));
    }
  }

  /**
   * Forward pass through entire GNN
   */
  forward(graph: CellGraph): number[][] {
    let nodeFeatures = graph.nodes.map(n => n.features);
    const adjacencyMatrix = graph.adjacencyMatrix || this.buildAdjacencyFromEdges(graph);

    // Build edge feature tensor
    const edgeFeatures = this.buildEdgeFeatureTensor(graph);

    // Pass through all layers
    for (const layer of this.layers) {
      nodeFeatures = layer.forward(nodeFeatures, adjacencyMatrix, edgeFeatures);
    }

    return nodeFeatures;
  }

  /**
   * Get cell embeddings for downstream tasks
   */
  getEmbeddings(graph: CellGraph): Map<string, number[]> {
    const embeddings = this.forward(graph);
    const result = new Map<string, number[]>();

    graph.nodes.forEach((node, idx) => {
      result.set(node.cellId, embeddings[idx]);
    });

    return result;
  }

  /**
   * Predict optimal power control parameters using GNN embeddings
   */
  predictPowerControlParams(graph: CellGraph): Map<string, { p0: number; alpha: number }> {
    const embeddings = this.forward(graph);
    const predictions = new Map<string, { p0: number; alpha: number }>();

    graph.nodes.forEach((node, idx) => {
      const embedding = embeddings[idx];

      // Use embedding to predict optimal P0 and alpha
      // Simple linear projection (in practice, would use trained weights)
      const p0Raw = embedding.slice(0, Math.floor(embedding.length / 2)).reduce((a, b) => a + b, 0);
      const alphaRaw = embedding.slice(Math.floor(embedding.length / 2)).reduce((a, b) => a + b, 0);

      // Map to valid ranges
      const p0 = Math.round(Math.max(-126, Math.min(24, p0Raw * 10 - 96)));
      const alpha = Math.max(0, Math.min(1, (alphaRaw + 1) / 2));

      predictions.set(node.cellId, { p0, alpha });
    });

    return predictions;
  }

  /**
   * Identify anomalous cells based on GNN embeddings
   */
  detectAnomalousCells(graph: CellGraph, threshold: number = 2): string[] {
    const embeddings = this.forward(graph);
    const anomalous: string[] = [];

    // Compute mean and std of embeddings
    const allValues = embeddings.flat();
    const mean = allValues.reduce((a, b) => a + b, 0) / allValues.length;
    const variance = allValues.reduce((a, b) => a + (b - mean) ** 2, 0) / allValues.length;
    const std = Math.sqrt(variance);

    graph.nodes.forEach((node, idx) => {
      const embedding = embeddings[idx];
      const embeddingMean = embedding.reduce((a, b) => a + b, 0) / embedding.length;

      if (Math.abs(embeddingMean - mean) > threshold * std) {
        anomalous.push(node.cellId);
      }
    });

    return anomalous;
  }

  private buildAdjacencyFromEdges(graph: CellGraph): number[][] {
    const n = graph.nodes.length;
    const matrix: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));

    const idToIndex = new Map<string, number>();
    graph.nodes.forEach((node, idx) => idToIndex.set(node.id, idx));

    for (const edge of graph.edges) {
      const sourceIdx = idToIndex.get(edge.source);
      const targetIdx = idToIndex.get(edge.target);

      if (sourceIdx !== undefined && targetIdx !== undefined) {
        matrix[sourceIdx][targetIdx] = edge.weight;
        matrix[targetIdx][sourceIdx] = edge.weight;
      }
    }

    return matrix;
  }

  private buildEdgeFeatureTensor(graph: CellGraph): number[][][] {
    const n = graph.nodes.length;
    const tensor: number[][][] = Array(n).fill(null).map(() =>
      Array(n).fill(null).map(() => [])
    );

    const idToIndex = new Map<string, number>();
    graph.nodes.forEach((node, idx) => idToIndex.set(node.id, idx));

    for (const edge of graph.edges) {
      const sourceIdx = idToIndex.get(edge.source);
      const targetIdx = idToIndex.get(edge.target);

      if (sourceIdx !== undefined && targetIdx !== undefined) {
        tensor[sourceIdx][targetIdx] = edge.features;
        tensor[targetIdx][sourceIdx] = edge.features;
      }
    }

    return tensor;
  }
}

// ============================================================================
// SINR-BASED NEIGHBOR ANALYSIS
// ============================================================================

export class SINRNeighborAnalyzer {
  /**
   * Analyze SINR relationships in the cell graph
   */
  analyzeSINRRelationships(graph: CellGraph): Array<{
    sourceCellId: string;
    targetCellId: string;
    sinrDelta: number;
    recommendation: string;
    priority: 'high' | 'medium' | 'low';
  }> {
    const recommendations: Array<{
      sourceCellId: string;
      targetCellId: string;
      sinrDelta: number;
      recommendation: string;
      priority: 'high' | 'medium' | 'low';
    }> = [];

    for (const edge of graph.edges) {
      // Edge features: [sourceSinr, targetSinr, sinrDelta, ...]
      const sourceSinr = edge.features[0] * 35 - 5; // Denormalize
      const targetSinr = edge.features[1] * 35 - 5;
      const sinrDelta = targetSinr - sourceSinr;

      if (edge.relationshipType === 'intra-freq') {
        // Check for too early/late HO based on SINR
        if (sinrDelta > 10) {
          recommendations.push({
            sourceCellId: edge.source,
            targetCellId: edge.target,
            sinrDelta,
            recommendation: `Consider reducing A3 offset for ${edge.source}->${edge.target}: target SINR is ${sinrDelta.toFixed(1)}dB better`,
            priority: sinrDelta > 15 ? 'high' : 'medium',
          });
        } else if (sinrDelta < -3 && edge.weight < 0.5) {
          recommendations.push({
            sourceCellId: edge.source,
            targetCellId: edge.target,
            sinrDelta,
            recommendation: `Review neighbor relation ${edge.source}->${edge.target}: poor target SINR (${targetSinr.toFixed(1)}dB) causing HO failures`,
            priority: sinrDelta < -6 ? 'high' : 'medium',
          });
        }
      }

      // Check for interference issues between neighbors
      if (sourceSinr < 3 && targetSinr < 3) {
        recommendations.push({
          sourceCellId: edge.source,
          targetCellId: edge.target,
          sinrDelta,
          recommendation: `Both cells ${edge.source} and ${edge.target} have low SINR - check for mutual interference`,
          priority: 'high',
        });
      }
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { high: 0, medium: 1, low: 2 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });
  }

  /**
   * Find optimal handover parameters based on SINR
   */
  optimizeHandoverParams(
    relation: NeighborRelation,
    currentSnapshot: CellKPISnapshot
  ): {
    a3Offset: number;
    hysteresis: number;
    timeToTrigger: number;
    rationale: string;
  } {
    const sinrDelta = relation.targetSinr - relation.sourceSinr;
    const currentA3 = relation.a3Offset;
    const currentHyst = relation.hysteresis;
    const currentTtt = relation.timeToTrigger;

    let newA3 = currentA3;
    let newHyst = currentHyst;
    let newTtt = currentTtt;
    let rationale = '';

    // Adjust A3 based on SINR delta and HO success rate
    if (relation.hoSuccessRate < 95) {
      if (sinrDelta > 6) {
        // Target is much better - handover is likely too late
        newA3 = Math.max(-6, currentA3 - 1);
        newTtt = Math.max(40, currentTtt - 40);
        rationale = 'Reducing A3 offset and TTT: target SINR is significantly better, HO may be too late';
      } else if (sinrDelta < 0) {
        // Target is worse - handover might be too early
        newA3 = Math.min(6, currentA3 + 1);
        newHyst = Math.min(6, currentHyst + 0.5);
        rationale = 'Increasing A3 offset and hysteresis: target SINR is worse, HO may be too early';
      }
    }

    // Handle ping-pong scenarios
    const mobility = currentSnapshot.mobility;
    if (mobility.pingPongHo > 10) {
      newHyst = Math.min(6, currentHyst + 1);
      newTtt = Math.min(1024, currentTtt + 80);
      rationale = rationale
        ? rationale + '; Also increasing hysteresis/TTT for ping-pong reduction'
        : 'Increasing hysteresis and TTT to reduce ping-pong handovers';
    }

    if (!rationale) {
      rationale = 'Current parameters appear optimal';
    }

    return {
      a3Offset: newA3,
      hysteresis: newHyst,
      timeToTrigger: newTtt,
      rationale,
    };
  }
}

export default {
  CellGraphBuilder,
  GNNLayer,
  CellGNN,
  SINRNeighborAnalyzer,
  DEFAULT_FEATURE_CONFIG,
  DEFAULT_GNN_CONFIG,
};
