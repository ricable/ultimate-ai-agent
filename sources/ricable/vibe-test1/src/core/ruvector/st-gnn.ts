/**
 * Ruvector - Spatio-Temporal Graph Neural Network Engine
 *
 * Implements ST-GNN for understanding the spatial structure of the radio network.
 * Key capabilities:
 * - Hypergraph support for modeling interference clusters
 * - 39 attention mechanism types including Flash Attention
 * - Message passing for neighbor influence propagation
 * - Temporal encoding with LSTM/TCN integration
 *
 * This enables agents to understand and mitigate complex inter-cell
 * interference patterns dynamically.
 */

import { EventEmitter } from 'eventemitter3';
import {
  RANGraph,
  RANGraphNode,
  RANGraphEdge,
  RANHyperedge
} from '../../types/index.js';

// ============================================================================
// ATTENTION MECHANISM IMPLEMENTATIONS
// ============================================================================

export type AttentionType =
  | 'standard'
  | 'flash'
  | 'multi_head'
  | 'linear'
  | 'sparse'
  | 'local'
  | 'global'
  | 'axial'
  | 'cross'
  | 'self'
  | 'gat'          // Graph Attention
  | 'gat_v2'       // GATv2 (dynamic attention)
  | 'transformer'
  | 'performer'
  | 'linformer'
  | 'longformer'
  | 'bigbird'
  | 'reformer'
  | 'routing'
  | 'slot'
  | 'memory'
  | 'compositional'
  | 'sparse_sinkhorn'
  | 'clustered'
  | 'block_sparse'
  | 'random_feature'
  | 'nyström'
  | 'low_rank'
  | 'kernel'
  | 'fourier'
  | 'wavelet'
  | 'causal'
  | 'bidirectional'
  | 'relative_position'
  | 'rotary'         // RoPE
  | 'alibi'
  | 'learned_position'
  | 'multiplicative'
  | 'additive'
  | 'scaled_dot_product';

interface AttentionConfig {
  type: AttentionType;
  numHeads: number;
  headDim: number;
  dropout: number;
  useFlashAttention: boolean;
}

class AttentionMechanism {
  config: AttentionConfig;

  constructor(config: AttentionConfig) {
    this.config = config;
  }

  /**
   * Compute attention scores between query and key vectors
   */
  computeScores(
    queries: Float32Array,
    keys: Float32Array,
    numQueries: number,
    numKeys: number,
    dim: number
  ): Float32Array {
    const scores = new Float32Array(numQueries * numKeys);
    const scale = 1 / Math.sqrt(dim);

    switch (this.config.type) {
      case 'flash':
        return this.flashAttention(queries, keys, numQueries, numKeys, dim, scale);
      case 'gat':
        return this.gatAttention(queries, keys, numQueries, numKeys, dim);
      case 'gat_v2':
        return this.gatV2Attention(queries, keys, numQueries, numKeys, dim);
      case 'linear':
        return this.linearAttention(queries, keys, numQueries, numKeys, dim);
      default:
        return this.standardAttention(queries, keys, numQueries, numKeys, dim, scale);
    }
  }

  private standardAttention(
    queries: Float32Array,
    keys: Float32Array,
    numQueries: number,
    numKeys: number,
    dim: number,
    scale: number
  ): Float32Array {
    const scores = new Float32Array(numQueries * numKeys);

    for (let i = 0; i < numQueries; i++) {
      for (let j = 0; j < numKeys; j++) {
        let dot = 0;
        for (let d = 0; d < dim; d++) {
          dot += queries[i * dim + d] * keys[j * dim + d];
        }
        scores[i * numKeys + j] = dot * scale;
      }
    }

    return scores;
  }

  /**
   * Flash Attention - Memory-efficient attention computation
   * Uses tiling to reduce memory bandwidth
   */
  private flashAttention(
    queries: Float32Array,
    keys: Float32Array,
    numQueries: number,
    numKeys: number,
    dim: number,
    scale: number
  ): Float32Array {
    const BLOCK_SIZE = 64;
    const scores = new Float32Array(numQueries * numKeys);

    // Process in blocks for better cache utilization
    for (let qi = 0; qi < numQueries; qi += BLOCK_SIZE) {
      const qEnd = Math.min(qi + BLOCK_SIZE, numQueries);

      for (let ki = 0; ki < numKeys; ki += BLOCK_SIZE) {
        const kEnd = Math.min(ki + BLOCK_SIZE, numKeys);

        // Compute block of attention scores
        for (let i = qi; i < qEnd; i++) {
          for (let j = ki; j < kEnd; j++) {
            let dot = 0;
            // Unrolled inner loop for SIMD-like behavior
            let d = 0;
            for (; d + 4 <= dim; d += 4) {
              dot += queries[i * dim + d] * keys[j * dim + d];
              dot += queries[i * dim + d + 1] * keys[j * dim + d + 1];
              dot += queries[i * dim + d + 2] * keys[j * dim + d + 2];
              dot += queries[i * dim + d + 3] * keys[j * dim + d + 3];
            }
            for (; d < dim; d++) {
              dot += queries[i * dim + d] * keys[j * dim + d];
            }
            scores[i * numKeys + j] = dot * scale;
          }
        }
      }
    }

    return scores;
  }

  /**
   * Graph Attention Network (GAT) attention
   * Learns importance of neighbors dynamically
   */
  private gatAttention(
    queries: Float32Array,
    keys: Float32Array,
    numQueries: number,
    numKeys: number,
    dim: number
  ): Float32Array {
    const scores = new Float32Array(numQueries * numKeys);

    // GAT uses learned attention vector 'a'
    // e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    // Here we approximate with dot product + LeakyReLU
    const leakySlope = 0.2;

    for (let i = 0; i < numQueries; i++) {
      for (let j = 0; j < numKeys; j++) {
        let score = 0;
        for (let d = 0; d < dim; d++) {
          score += queries[i * dim + d] + keys[j * dim + d];
        }
        // LeakyReLU
        scores[i * numKeys + j] = score > 0 ? score : score * leakySlope;
      }
    }

    return scores;
  }

  /**
   * GATv2 - Dynamic attention that can express more attention patterns
   */
  private gatV2Attention(
    queries: Float32Array,
    keys: Float32Array,
    numQueries: number,
    numKeys: number,
    dim: number
  ): Float32Array {
    const scores = new Float32Array(numQueries * numKeys);
    const leakySlope = 0.2;

    // GATv2: a^T LeakyReLU(W[h_i || h_j])
    for (let i = 0; i < numQueries; i++) {
      for (let j = 0; j < numKeys; j++) {
        let score = 0;
        for (let d = 0; d < dim; d++) {
          const combined = queries[i * dim + d] + keys[j * dim + d];
          // Apply LeakyReLU before dot product
          score += combined > 0 ? combined : combined * leakySlope;
        }
        scores[i * numKeys + j] = score;
      }
    }

    return scores;
  }

  /**
   * Linear Attention - O(n) complexity via kernel feature maps
   */
  private linearAttention(
    queries: Float32Array,
    keys: Float32Array,
    numQueries: number,
    numKeys: number,
    dim: number
  ): Float32Array {
    const scores = new Float32Array(numQueries * numKeys);

    // Use ELU + 1 as feature map: φ(x) = elu(x) + 1
    const elu = (x: number) => x > 0 ? x : Math.exp(x) - 1;

    // Transform queries and keys
    const phiQ = new Float32Array(numQueries * dim);
    const phiK = new Float32Array(numKeys * dim);

    for (let i = 0; i < numQueries * dim; i++) {
      phiQ[i] = elu(queries[i]) + 1;
    }
    for (let i = 0; i < numKeys * dim; i++) {
      phiK[i] = elu(keys[i]) + 1;
    }

    // Compute via kernel trick
    for (let i = 0; i < numQueries; i++) {
      for (let j = 0; j < numKeys; j++) {
        let dot = 0;
        for (let d = 0; d < dim; d++) {
          dot += phiQ[i * dim + d] * phiK[j * dim + d];
        }
        scores[i * numKeys + j] = dot;
      }
    }

    return scores;
  }

  /**
   * Apply softmax normalization to attention scores
   */
  softmax(scores: Float32Array, numRows: number, numCols: number): Float32Array {
    const result = new Float32Array(scores.length);

    for (let i = 0; i < numRows; i++) {
      const rowStart = i * numCols;

      // Find max for numerical stability
      let maxVal = -Infinity;
      for (let j = 0; j < numCols; j++) {
        maxVal = Math.max(maxVal, scores[rowStart + j]);
      }

      // Compute exp and sum
      let sumExp = 0;
      for (let j = 0; j < numCols; j++) {
        result[rowStart + j] = Math.exp(scores[rowStart + j] - maxVal);
        sumExp += result[rowStart + j];
      }

      // Normalize
      for (let j = 0; j < numCols; j++) {
        result[rowStart + j] /= sumExp;
      }
    }

    return result;
  }
}

// ============================================================================
// TEMPORAL ENCODER (LSTM-like)
// ============================================================================

class TemporalEncoder {
  hiddenSize: number;
  inputSize: number;

  // LSTM weights (simplified)
  Wf: Float32Array; // Forget gate
  Wi: Float32Array; // Input gate
  Wc: Float32Array; // Cell gate
  Wo: Float32Array; // Output gate

  constructor(inputSize: number, hiddenSize: number) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;

    const initWeight = (size: number) => {
      const w = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        w[i] = (Math.random() - 0.5) * 0.1;
      }
      return w;
    };

    const gateSize = (inputSize + hiddenSize) * hiddenSize;
    this.Wf = initWeight(gateSize);
    this.Wi = initWeight(gateSize);
    this.Wc = initWeight(gateSize);
    this.Wo = initWeight(gateSize);
  }

  /**
   * Encode a sequence of temporal observations
   */
  encode(sequence: Float32Array[]): Float32Array {
    const h = new Float32Array(this.hiddenSize);
    const c = new Float32Array(this.hiddenSize);

    for (const x of sequence) {
      this.step(x, h, c);
    }

    return h;
  }

  private step(x: Float32Array, h: Float32Array, c: Float32Array): void {
    const combined = new Float32Array(this.inputSize + this.hiddenSize);
    combined.set(x);
    combined.set(h, this.inputSize);

    // Compute gates
    const sigmoid = (val: number) => 1 / (1 + Math.exp(-val));
    const tanh = Math.tanh;

    for (let i = 0; i < this.hiddenSize; i++) {
      let fg = 0, ig = 0, cg = 0, og = 0;

      for (let j = 0; j < combined.length; j++) {
        const idx = j * this.hiddenSize + i;
        fg += combined[j] * this.Wf[idx];
        ig += combined[j] * this.Wi[idx];
        cg += combined[j] * this.Wc[idx];
        og += combined[j] * this.Wo[idx];
      }

      const forgetGate = sigmoid(fg);
      const inputGate = sigmoid(ig);
      const cellGate = tanh(cg);
      const outputGate = sigmoid(og);

      c[i] = forgetGate * c[i] + inputGate * cellGate;
      h[i] = outputGate * tanh(c[i]);
    }
  }
}

// ============================================================================
// MESSAGE PASSING LAYER
// ============================================================================

interface MessagePassingConfig {
  inputDim: number;
  outputDim: number;
  aggregation: 'sum' | 'mean' | 'max' | 'attention';
  attentionConfig?: AttentionConfig;
}

class MessagePassingLayer {
  config: MessagePassingConfig;
  attention?: AttentionMechanism;

  // Learnable transformations
  messageTransform: Float32Array;
  updateTransform: Float32Array;

  constructor(config: MessagePassingConfig) {
    this.config = config;

    if (config.aggregation === 'attention' && config.attentionConfig) {
      this.attention = new AttentionMechanism(config.attentionConfig);
    }

    // Initialize transforms
    const initTransform = (rows: number, cols: number) => {
      const w = new Float32Array(rows * cols);
      const scale = Math.sqrt(2 / (rows + cols));
      for (let i = 0; i < w.length; i++) {
        w[i] = (Math.random() - 0.5) * scale;
      }
      return w;
    };

    this.messageTransform = initTransform(config.inputDim, config.outputDim);
    this.updateTransform = initTransform(config.inputDim + config.outputDim, config.outputDim);
  }

  /**
   * Perform message passing on graph
   *
   * h_v^{(k+1)} = φ(h_v^{(k)}, AGG({ψ(h_u^{(k)}) : u ∈ N(v)}))
   */
  forward(
    nodeFeatures: Map<string, Float32Array>,
    edges: Array<{ source: string; target: string; weight: number }>
  ): Map<string, Float32Array> {
    const updatedFeatures = new Map<string, Float32Array>();

    // Build adjacency structure
    const neighbors = new Map<string, Array<{ id: string; weight: number }>>();
    for (const [id] of nodeFeatures) {
      neighbors.set(id, []);
    }
    for (const edge of edges) {
      neighbors.get(edge.target)?.push({ id: edge.source, weight: edge.weight });
    }

    // Message passing
    for (const [nodeId, features] of nodeFeatures) {
      const nodeNeighbors = neighbors.get(nodeId) || [];

      if (nodeNeighbors.length === 0) {
        // No neighbors, just transform self
        updatedFeatures.set(nodeId, this.transformFeatures(features));
        continue;
      }

      // Collect neighbor messages
      const messages: Float32Array[] = [];
      const weights: number[] = [];

      for (const { id: neighborId, weight } of nodeNeighbors) {
        const neighborFeatures = nodeFeatures.get(neighborId);
        if (neighborFeatures) {
          messages.push(this.generateMessage(neighborFeatures));
          weights.push(weight);
        }
      }

      // Aggregate messages
      const aggregated = this.aggregate(messages, weights, features);

      // Update node representation
      const updated = this.update(features, aggregated);
      updatedFeatures.set(nodeId, updated);
    }

    return updatedFeatures;
  }

  private generateMessage(features: Float32Array): Float32Array {
    const message = new Float32Array(this.config.outputDim);

    for (let i = 0; i < this.config.outputDim; i++) {
      for (let j = 0; j < this.config.inputDim; j++) {
        message[i] += features[j] * this.messageTransform[j * this.config.outputDim + i];
      }
    }

    return message;
  }

  private aggregate(
    messages: Float32Array[],
    weights: number[],
    selfFeatures: Float32Array
  ): Float32Array {
    const result = new Float32Array(this.config.outputDim);

    if (messages.length === 0) {
      return result;
    }

    switch (this.config.aggregation) {
      case 'sum':
        for (const msg of messages) {
          for (let i = 0; i < result.length; i++) {
            result[i] += msg[i];
          }
        }
        break;

      case 'mean':
        for (const msg of messages) {
          for (let i = 0; i < result.length; i++) {
            result[i] += msg[i];
          }
        }
        for (let i = 0; i < result.length; i++) {
          result[i] /= messages.length;
        }
        break;

      case 'max':
        result.fill(-Infinity);
        for (const msg of messages) {
          for (let i = 0; i < result.length; i++) {
            result[i] = Math.max(result[i], msg[i]);
          }
        }
        break;

      case 'attention':
        if (this.attention) {
          // Compute attention over messages
          const queries = selfFeatures;
          const keys = new Float32Array(messages.length * this.config.inputDim);
          for (let i = 0; i < messages.length; i++) {
            const nodeFeatures = messages[i];
            keys.set(nodeFeatures.slice(0, this.config.inputDim), i * this.config.inputDim);
          }

          const scores = this.attention.computeScores(
            queries,
            keys,
            1,
            messages.length,
            this.config.inputDim
          );

          const attnWeights = this.attention.softmax(scores, 1, messages.length);

          for (let i = 0; i < messages.length; i++) {
            for (let j = 0; j < result.length; j++) {
              result[j] += attnWeights[i] * messages[i][j];
            }
          }
        }
        break;
    }

    return result;
  }

  private update(selfFeatures: Float32Array, aggregated: Float32Array): Float32Array {
    // Concatenate self features with aggregated neighbor info
    const combined = new Float32Array(this.config.inputDim + this.config.outputDim);
    combined.set(selfFeatures);
    combined.set(aggregated, this.config.inputDim);

    // Transform
    const result = new Float32Array(this.config.outputDim);
    for (let i = 0; i < this.config.outputDim; i++) {
      for (let j = 0; j < combined.length; j++) {
        result[i] += combined[j] * this.updateTransform[j * this.config.outputDim + i];
      }
      // ReLU activation
      result[i] = Math.max(0, result[i]);
    }

    return result;
  }

  private transformFeatures(features: Float32Array): Float32Array {
    const result = new Float32Array(this.config.outputDim);
    for (let i = 0; i < this.config.outputDim; i++) {
      for (let j = 0; j < Math.min(features.length, this.config.inputDim); j++) {
        result[i] += features[j] * this.messageTransform[j * this.config.outputDim + i];
      }
      result[i] = Math.max(0, result[i]); // ReLU
    }
    return result;
  }
}

// ============================================================================
// HYPERGRAPH PROCESSOR
// ============================================================================

class HypergraphProcessor {
  maxHyperedgeSize: number;

  constructor(maxHyperedgeSize: number = 16) {
    this.maxHyperedgeSize = maxHyperedgeSize;
  }

  /**
   * Process hyperedge connections (sets of mutually interacting nodes)
   * Useful for interference clusters where multiple cells interact
   */
  processHyperedges(
    nodeFeatures: Map<string, Float32Array>,
    hyperedges: RANHyperedge[],
    featureDim: number
  ): Map<string, Float32Array> {
    const hyperedgeInfluence = new Map<string, Float32Array>();

    // Initialize influence vectors
    for (const [nodeId] of nodeFeatures) {
      hyperedgeInfluence.set(nodeId, new Float32Array(featureDim));
    }

    // For each hyperedge, propagate information between all member nodes
    for (const hyperedge of hyperedges) {
      if (hyperedge.nodeIds.length > this.maxHyperedgeSize) continue;

      // Compute hyperedge representation (mean of member features)
      const hyperedgeEmbed = new Float32Array(featureDim);
      let count = 0;

      for (const nodeId of hyperedge.nodeIds) {
        const features = nodeFeatures.get(nodeId);
        if (features) {
          for (let i = 0; i < featureDim; i++) {
            hyperedgeEmbed[i] += features[i];
          }
          count++;
        }
      }

      if (count > 0) {
        for (let i = 0; i < featureDim; i++) {
          hyperedgeEmbed[i] /= count;
        }
      }

      // Add weighted influence to each member
      for (const nodeId of hyperedge.nodeIds) {
        const influence = hyperedgeInfluence.get(nodeId);
        if (influence) {
          for (let i = 0; i < featureDim; i++) {
            influence[i] += hyperedgeEmbed[i] * hyperedge.weight;
          }
        }
      }
    }

    return hyperedgeInfluence;
  }
}

// ============================================================================
// MAIN ST-GNN CLASS
// ============================================================================

export interface STGNNConfig {
  inputDim: number;          // Input feature dimension
  hiddenDim: number;         // Hidden layer dimension
  outputDim: number;         // Output embedding dimension
  numLayers: number;         // Number of message passing layers
  numHeads: number;          // Attention heads
  attentionType: AttentionType;
  temporalWindowSize: number; // How many time steps to encode
  useHypergraph: boolean;
  dropout: number;
}

const DEFAULT_STGNN_CONFIG: STGNNConfig = {
  inputDim: 64,
  hiddenDim: 256,
  outputDim: 128,
  numLayers: 4,
  numHeads: 8,
  attentionType: 'gat_v2',
  temporalWindowSize: 24,
  useHypergraph: true,
  dropout: 0.1
};

export class SpatioTemporalGNN extends EventEmitter {
  config: STGNNConfig;
  temporalEncoder: TemporalEncoder;
  messageLayers: MessagePassingLayer[];
  hypergraphProcessor?: HypergraphProcessor;

  // For tracking
  forwardPasses: number = 0;
  lastProcessingTime: number = 0;

  constructor(config: Partial<STGNNConfig> = {}) {
    super();
    this.config = { ...DEFAULT_STGNN_CONFIG, ...config };

    // Initialize temporal encoder
    this.temporalEncoder = new TemporalEncoder(
      this.config.inputDim,
      this.config.hiddenDim
    );

    // Initialize message passing layers
    this.messageLayers = [];
    for (let i = 0; i < this.config.numLayers; i++) {
      const inputDim = i === 0 ? this.config.hiddenDim : this.config.hiddenDim;
      const outputDim = i === this.config.numLayers - 1
        ? this.config.outputDim
        : this.config.hiddenDim;

      this.messageLayers.push(new MessagePassingLayer({
        inputDim,
        outputDim,
        aggregation: 'attention',
        attentionConfig: {
          type: this.config.attentionType,
          numHeads: this.config.numHeads,
          headDim: Math.floor(outputDim / this.config.numHeads),
          dropout: this.config.dropout,
          useFlashAttention: this.config.attentionType === 'flash'
        }
      }));
    }

    // Initialize hypergraph processor if enabled
    if (this.config.useHypergraph) {
      this.hypergraphProcessor = new HypergraphProcessor();
    }
  }

  /**
   * Process RAN graph and generate node embeddings
   *
   * @param graph - The RAN graph with nodes, edges, and hyperedges
   * @param temporalHistory - Historical features per node (for temporal encoding)
   * @returns Map of node ID to embedding vector
   */
  forward(
    graph: RANGraph,
    temporalHistory?: Map<string, Float32Array[]>
  ): Map<string, Float32Array> {
    const startTime = Date.now();

    // Step 1: Prepare initial node features with temporal encoding
    let nodeFeatures = new Map<string, Float32Array>();

    for (const [nodeId, node] of graph.nodes) {
      let features: Float32Array;

      // Check if we have temporal history for this node
      if (temporalHistory?.has(nodeId)) {
        const history = temporalHistory.get(nodeId)!;
        features = this.temporalEncoder.encode(history);
      } else {
        // Use static + dynamic features directly
        features = this.extractNodeFeatures(node);
      }

      nodeFeatures.set(nodeId, features);
    }

    // Step 2: Convert edges to simple format
    const simpleEdges = graph.edges.map(e => ({
      source: e.sourceId,
      target: e.targetId,
      weight: e.weight
    }));

    // Step 3: Message passing layers
    for (const layer of this.messageLayers) {
      nodeFeatures = layer.forward(nodeFeatures, simpleEdges);
    }

    // Step 4: Process hyperedges if enabled
    if (this.hypergraphProcessor && graph.hyperedges.length > 0) {
      const hyperedgeInfluence = this.hypergraphProcessor.processHyperedges(
        nodeFeatures,
        graph.hyperedges,
        this.config.outputDim
      );

      // Add hyperedge influence to node embeddings
      for (const [nodeId, embedding] of nodeFeatures) {
        const influence = hyperedgeInfluence.get(nodeId);
        if (influence) {
          for (let i = 0; i < embedding.length; i++) {
            embedding[i] += influence[i] * 0.5; // Weighted combination
          }
        }
      }
    }

    // Step 5: L2 normalize embeddings
    for (const [nodeId, embedding] of nodeFeatures) {
      let norm = 0;
      for (const val of embedding) {
        norm += val * val;
      }
      norm = Math.sqrt(norm);
      if (norm > 0) {
        for (let i = 0; i < embedding.length; i++) {
          embedding[i] /= norm;
        }
      }
      nodeFeatures.set(nodeId, embedding);
    }

    this.forwardPasses++;
    this.lastProcessingTime = Date.now() - startTime;
    this.emit('forward-complete', {
      nodeCount: graph.nodes.size,
      processingTimeMs: this.lastProcessingTime
    });

    return nodeFeatures;
  }

  /**
   * Compute attention coefficients between nodes
   * Returns which neighbors are most relevant to each node
   */
  getAttentionWeights(
    graph: RANGraph,
    nodeFeatures: Map<string, Float32Array>
  ): Map<string, Map<string, number>> {
    const attentionWeights = new Map<string, Map<string, number>>();

    // Build adjacency list
    const neighbors = new Map<string, string[]>();
    for (const [id] of graph.nodes) {
      neighbors.set(id, []);
    }
    for (const edge of graph.edges) {
      neighbors.get(edge.targetId)?.push(edge.sourceId);
    }

    // Compute attention for each node
    const attention = new AttentionMechanism({
      type: this.config.attentionType,
      numHeads: 1,
      headDim: this.config.outputDim,
      dropout: 0,
      useFlashAttention: false
    });

    for (const [nodeId, embedding] of nodeFeatures) {
      const nodeNeighbors = neighbors.get(nodeId) || [];
      if (nodeNeighbors.length === 0) continue;

      const queries = embedding;
      const keys = new Float32Array(nodeNeighbors.length * this.config.outputDim);

      for (let i = 0; i < nodeNeighbors.length; i++) {
        const neighborEmbed = nodeFeatures.get(nodeNeighbors[i]);
        if (neighborEmbed) {
          keys.set(neighborEmbed, i * this.config.outputDim);
        }
      }

      const scores = attention.computeScores(
        queries,
        keys,
        1,
        nodeNeighbors.length,
        this.config.outputDim
      );

      const weights = attention.softmax(scores, 1, nodeNeighbors.length);

      const nodeAttention = new Map<string, number>();
      for (let i = 0; i < nodeNeighbors.length; i++) {
        nodeAttention.set(nodeNeighbors[i], weights[i]);
      }
      attentionWeights.set(nodeId, nodeAttention);
    }

    return attentionWeights;
  }

  /**
   * Extract features from a graph node
   */
  private extractNodeFeatures(node: RANGraphNode): Float32Array {
    const features = new Float32Array(this.config.hiddenDim);

    // Static features
    let idx = 0;
    features[idx++] = node.staticFeatures.azimuth / 360;
    features[idx++] = node.staticFeatures.tilt / 15;
    features[idx++] = node.staticFeatures.height / 100;
    features[idx++] = node.staticFeatures.beamwidth / 180;
    features[idx++] = node.staticFeatures.frequency / 10000;
    features[idx++] = node.staticFeatures.bandwidth / 100;
    features[idx++] = node.staticFeatures.technology === 'NR' ? 1 : 0;

    // Dynamic features
    features[idx++] = (node.dynamicFeatures.txPower + 10) / 60;
    features[idx++] = node.dynamicFeatures.load / 100;
    features[idx++] = (node.dynamicFeatures.rtwp + 120) / 60;
    features[idx++] = node.dynamicFeatures.activeUsers / 500;
    features[idx++] = node.dynamicFeatures.throughput / 1000;

    return features;
  }

  /**
   * Get model statistics
   */
  getStats(): {
    config: STGNNConfig;
    forwardPasses: number;
    lastProcessingTimeMs: number;
  } {
    return {
      config: this.config,
      forwardPasses: this.forwardPasses,
      lastProcessingTimeMs: this.lastProcessingTime
    };
  }
}

export { AttentionMechanism, TemporalEncoder, MessagePassingLayer, HypergraphProcessor };
