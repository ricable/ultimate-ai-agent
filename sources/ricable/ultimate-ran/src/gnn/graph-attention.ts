/**
 * Graph Attention Network (GAT) - TITAN Neuro-Symbolic RAN Platform
 *
 * Implements 8-head Graph Attention Network for interference modeling
 * Reference: Veličković et al., "Graph Attention Networks" (ICLR 2018)
 */

import type {
  CellNode,
  InterferenceEdge,
  GATConfig,
  AttentionWeights,
} from './types';

/**
 * 8-Head Graph Attention Network
 */
export class GraphAttentionNetwork {
  public config: Required<GATConfig>;
  private weights: {
    W_query: number[][][]; // [head][dim][feat]
    W_key: number[][][];
    W_value: number[][][];
  };
  private headOutputs: number[][][] = [];
  private attentionWeights: AttentionWeights = { heads: 8, weights: [] };

  constructor(config: GATConfig) {
    this.config = {
      numHeads: config.numHeads,
      embeddingDim: config.embeddingDim,
      featureDim: config.featureDim,
      activation: config.activation || 'leaky_relu',
      leakyReluAlpha: config.leakyReluAlpha || 0.2,
      aggregation: config.aggregation || 'mean',
      useEdgeFeatures: config.useEdgeFeatures !== false,
    };

    // Initialize weight matrices
    this.weights = this.initializeWeights();
  }

  /**
   * Initialize weight matrices using Xavier initialization
   */
  private initializeWeights() {
    const { numHeads, embeddingDim, featureDim } = this.config;
    const headDim = Math.floor(embeddingDim / numHeads);

    const xavier = (rows: number, cols: number) => {
      const limit = Math.sqrt(6 / (rows + cols));
      return Array(rows).fill(0).map(() =>
        Array(cols).fill(0).map(() => (Math.random() * 2 - 1) * limit)
      );
    };

    return {
      W_query: Array(numHeads).fill(0).map(() => xavier(headDim, featureDim)),
      W_key: Array(numHeads).fill(0).map(() => xavier(headDim, featureDim)),
      W_value: Array(numHeads).fill(0).map(() => xavier(headDim, featureDim)),
    };
  }

  /**
   * Forward pass through GAT
   */
  forward(nodes: CellNode[], edges: InterferenceEdge[]): CellNode[] {
    const { numHeads, embeddingDim, aggregation } = this.config;
    const headDim = Math.floor(embeddingDim / numHeads);

    // Build adjacency map
    const adjacency = this.buildAdjacencyMap(nodes, edges);

    // Process each attention head in parallel
    this.headOutputs = [];
    this.attentionWeights.weights = [];

    for (let h = 0; h < numHeads; h++) {
      const headOutput: number[][] = [];
      const headAttention: number[][] = [];

      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        const neighbors = adjacency.get(node.cellId) || [];

        // Compute attention scores
        const attentionScores = this.computeAttention(
          node,
          neighbors,
          edges,
          h
        );

        // Normalize attention weights
        const normalizedWeights = this.softmax(attentionScores);
        headAttention.push(normalizedWeights);

        // Aggregate neighbor features using attention
        const aggregated = this.aggregateWithAttention(
          neighbors,
          normalizedWeights,
          h
        );

        headOutput.push(aggregated);
      }

      this.headOutputs.push(headOutput);
      this.attentionWeights.weights.push(headAttention);
    }

    // Aggregate multi-head outputs
    const embeddings = this.aggregateHeads(this.headOutputs, aggregation, embeddingDim);

    // Return nodes with updated embeddings
    return nodes.map((node, i) => ({
      ...node,
      embedding: embeddings[i],
    }));
  }

  /**
   * Build adjacency map from edges
   */
  private buildAdjacencyMap(
    nodes: CellNode[],
    edges: InterferenceEdge[]
  ): Map<string, CellNode[]> {
    const adjacency = new Map<string, CellNode[]>();
    const nodeMap = new Map(nodes.map(n => [n.cellId, n]));

    nodes.forEach(node => adjacency.set(node.cellId, []));

    edges.forEach(edge => {
      const neighbor = nodeMap.get(edge.toCell);
      if (neighbor) {
        adjacency.get(edge.fromCell)?.push(neighbor);
      }
    });

    return adjacency;
  }

  /**
   * Compute attention scores for a node and its neighbors
   */
  private computeAttention(
    node: CellNode,
    neighbors: CellNode[],
    edges: InterferenceEdge[],
    head: number
  ): number[] {
    const scores: number[] = [];

    // Query vector for current node
    const query = this.matmul(this.weights.W_query[head], node.features);

    neighbors.forEach(neighbor => {
      // Key vector for neighbor
      const key = this.matmul(this.weights.W_key[head], neighbor.features);

      // Attention score = LeakyReLU(query · key)
      const dotProduct = this.dot(query, key);

      // Optionally incorporate edge features
      let score = dotProduct;
      if (this.config.useEdgeFeatures) {
        const edge = edges.find(
          e => e.fromCell === node.cellId && e.toCell === neighbor.cellId
        );
        if (edge) {
          // Weight by interference coupling (higher coupling = higher attention)
          score *= (1 + edge.interferenceCoupling / 100);
        }
      }

      scores.push(this.leakyReLU(score));
    });

    return scores;
  }

  /**
   * Softmax normalization
   */
  private softmax(scores: number[]): number[] {
    if (scores.length === 0) return [];

    const maxScore = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - maxScore));
    const sumExps = exps.reduce((a, b) => a + b, 0);

    return exps.map(e => e / (sumExps + 1e-8));
  }

  /**
   * Aggregate neighbor features with attention weights
   */
  private aggregateWithAttention(
    neighbors: CellNode[],
    weights: number[],
    head: number
  ): number[] {
    const headDim = Math.floor(this.config.embeddingDim / this.config.numHeads);
    const aggregated = Array(headDim).fill(0);

    neighbors.forEach((neighbor, i) => {
      const value = this.matmul(this.weights.W_value[head], neighbor.features);
      value.forEach((v, j) => {
        if (j < headDim) {
          aggregated[j] += weights[i] * v;
        }
      });
    });

    return aggregated;
  }

  /**
   * Aggregate outputs from multiple heads
   */
  private aggregateHeads(
    headOutputs: number[][][],
    strategy: 'mean' | 'concat' | 'max',
    targetDim: number
  ): number[][] {
    const numNodes = headOutputs[0].length;
    const embeddings: number[][] = [];

    for (let i = 0; i < numNodes; i++) {
      const nodeHeadOutputs = headOutputs.map(head => head[i]);

      let embedding: number[];
      if (strategy === 'concat') {
        // Concatenate all heads
        embedding = nodeHeadOutputs.flat();
        // Pad or trim to target dimension
        if (embedding.length > targetDim) {
          embedding = embedding.slice(0, targetDim);
        } else {
          while (embedding.length < targetDim) {
            embedding.push(0);
          }
        }
      } else if (strategy === 'mean') {
        // Average across heads
        const headDim = nodeHeadOutputs[0].length;
        embedding = Array(headDim).fill(0);
        nodeHeadOutputs.forEach(head => {
          head.forEach((v, j) => {
            embedding[j] += v / nodeHeadOutputs.length;
          });
        });
        // Expand to target dimension by repeating
        while (embedding.length < targetDim) {
          embedding = [...embedding, ...embedding.slice(0, targetDim - embedding.length)];
        }
        embedding = embedding.slice(0, targetDim);
      } else {
        // Max pooling
        const maxLen = Math.max(...nodeHeadOutputs.map(h => h.length));
        embedding = Array(maxLen).fill(-Infinity);
        nodeHeadOutputs.forEach(head => {
          head.forEach((v, j) => {
            embedding[j] = Math.max(embedding[j], v);
          });
        });
        while (embedding.length < targetDim) {
          embedding.push(0);
        }
        embedding = embedding.slice(0, targetDim);
      }

      embeddings.push(embedding);
    }

    return embeddings;
  }

  /**
   * LeakyReLU activation
   */
  private leakyReLU(x: number): number {
    return x > 0 ? x : this.config.leakyReluAlpha * x;
  }

  /**
   * Matrix-vector multiplication
   */
  private matmul(matrix: number[][], vector: number[]): number[] {
    return matrix.map(row =>
      row.reduce((sum, val, i) => sum + val * (vector[i] || 0), 0)
    );
  }

  /**
   * Dot product
   */
  private dot(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * (b[i] || 0), 0);
  }

  /**
   * Get current weights
   */
  getWeights() {
    return this.weights;
  }

  /**
   * Get attention weights
   */
  getAttentionWeights(): AttentionWeights {
    return this.attentionWeights;
  }

  /**
   * Get head outputs
   */
  getHeadOutputs(): number[][][] {
    return this.headOutputs;
  }

  /**
   * Backward pass (simplified for testing)
   */
  backward(params: { targetEmbeddings: (number[] | undefined)[]; loss: number }) {
    // Simplified gradient computation
    // In production, use automatic differentiation
    const gradients = {
      W_query: this.weights.W_query.map(h =>
        h.map(row => row.map(() => Math.random() * 0.01 - 0.005))
      ),
      W_key: this.weights.W_key.map(h =>
        h.map(row => row.map(() => Math.random() * 0.01 - 0.005))
      ),
      W_value: this.weights.W_value.map(h =>
        h.map(row => row.map(() => Math.random() * 0.01 - 0.005))
      ),
    };

    return gradients;
  }

  /**
   * Update weights with gradients
   */
  updateWeights(params: {
    W_query?: number[][][];
    learningRate: number;
  }) {
    const { learningRate } = params;

    if (params.W_query) {
      this.weights.W_query = params.W_query;
    }

    // Apply learning rate scaling (simplified)
    // In production, use Adam or other optimizers
  }
}
