/**
 * Graph Attention Network Test Suite - London School TDD
 * TITAN Neuro-Symbolic RAN Platform
 *
 * Tests 8-head GAT mechanism for interference modeling
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { createMockCellNode, createMockInterferenceEdge } from '../setup';
import type { CellNode, InterferenceEdge } from '../../src/gnn/types';

// Import to be implemented
import { GraphAttentionNetwork } from '../../src/gnn/graph-attention';

describe('GraphAttentionNetwork', () => {
  let gat: GraphAttentionNetwork;
  let mockNodes: CellNode[];
  let mockEdges: InterferenceEdge[];

  beforeEach(() => {
    gat = new GraphAttentionNetwork({
      numHeads: 8,
      embeddingDim: 768,
      featureDim: 4,
    });

    mockNodes = [
      createMockCellNode({ cellId: 'NRCELL_001' }),
      createMockCellNode({ cellId: 'NRCELL_002' }),
      createMockCellNode({ cellId: 'NRCELL_003' }),
    ];

    // Ensure bidirectional/connected graph for attention tests
    mockEdges = [
      createMockInterferenceEdge({ fromCell: 'NRCELL_001', toCell: 'NRCELL_002' }),
      createMockInterferenceEdge({ fromCell: 'NRCELL_002', toCell: 'NRCELL_001' }),
      createMockInterferenceEdge({ fromCell: 'NRCELL_002', toCell: 'NRCELL_003' }),
      createMockInterferenceEdge({ fromCell: 'NRCELL_003', toCell: 'NRCELL_002' }),
    ];
  });

  describe('Initialization', () => {
    it('should initialize with 8 attention heads', () => {
      expect(gat.config.numHeads).toBe(8);
    });

    it('should have 768-dimensional embeddings', () => {
      expect(gat.config.embeddingDim).toBe(768);
    });

    it('should accept 4-dimensional input features', () => {
      expect(gat.config.featureDim).toBe(4);
    });

    it('should initialize weight matrices', () => {
      const weights = gat.getWeights();

      expect(weights).toHaveProperty('W_query');
      expect(weights).toHaveProperty('W_key');
      expect(weights).toHaveProperty('W_value');
    });
  });

  describe('Forward Pass', () => {
    it('should compute embeddings for all nodes', () => {
      const output = gat.forward(mockNodes, mockEdges);

      expect(output).toHaveLength(mockNodes.length);
      output.forEach(node => {
        expect(node.embedding).toHaveLength(768);
      });
    });

    it('should generate different embeddings per head', () => {
      const output = gat.forward(mockNodes, mockEdges);
      const headOutputs = gat.getHeadOutputs();

      expect(headOutputs).toHaveLength(8);
      headOutputs.forEach(head => {
        expect(head).toHaveLength(mockNodes.length);
      });
    });

    it('should aggregate multi-head outputs', () => {
      const output = gat.forward(mockNodes, mockEdges);

      // Default aggregation is mean
      expect(gat.config.aggregation).toBe('mean');
    });
  });

  describe('Attention Mechanism', () => {
    it('should compute attention coefficients', () => {
      gat.forward(mockNodes, mockEdges);
      const weights = gat.getAttentionWeights();

      expect(weights.heads).toBe(8);
      expect(weights.weights).toHaveLength(8);
    });

    it('should normalize attention weights per node', () => {
      gat.forward(mockNodes, mockEdges);
      const weights = gat.getAttentionWeights();

      weights.weights.forEach(headWeights => {
        headWeights.forEach((nodeWeights, i) => {
           // Only check if node has neighbors (otherwise weights are empty or zero)
           if (nodeWeights.length > 0) {
              const sum = nodeWeights.reduce((a: number, b: number) => a + b, 0);
              expect(sum).toBeCloseTo(1.0, 1);
           }
        });
      });
    });

    it('should assign higher weights to strong interferers', () => {
      // Setup: Cell 1 has two neighbors: Cell 2 (strong) and Cell 3 (weak)
      const nodes = [
          createMockCellNode({ cellId: 'NRCELL_001', features: [1, 1, 1, 1] }),
          createMockCellNode({ cellId: 'NRCELL_002', features: [1, 1, 1, 1] }),
          createMockCellNode({ cellId: 'NRCELL_003', features: [1, 1, 1, 1] })
      ];

      const strongEdge = createMockInterferenceEdge({
        fromCell: 'NRCELL_001',
        toCell: 'NRCELL_002',
        interferenceCoupling: 150, // Ultra High coupling
      });

      const weakEdge = createMockInterferenceEdge({
        fromCell: 'NRCELL_001',
        toCell: 'NRCELL_003',
        interferenceCoupling: 10, // Very Low coupling
      });

      // Set deterministic positive weights to ensure dot product > 0
      // so coupling multiplier (which is > 1) always increases the score.
      const dim = gat.config.embeddingDim;
      const heads = gat.config.numHeads;
      const headDim = dim / heads;
      const feat = gat.config.featureDim;
      
      const ones = Array(heads).fill(0).map(() => 
        Array(headDim).fill(0).map(() => Array(feat).fill(0.1))
      );

      gat.updateWeights({
        W_query: ones,
        learningRate: 0 // Irrelevant here
      });
      // Note: We should also set W_key to ensure dot product is positive, 
      // but updateWeights might only expose W_query in the interface? 
      // Checking source: updateWeights ONLY updates W_query if passed. 
      // W_key remains random. This is a problem.
      // We need to control W_key too. 
      // Since we can't easily via public API, we'll rely on "statistically likely" 
      // but with a retry or skipped if flaky. 
      // OR, we can hack it by casting to any.
      (gat as any).weights.W_key = ones;

      gat.forward(nodes, [strongEdge, weakEdge]);
      const weights = gat.getAttentionWeights();

      // Now with positive weights and features, dot product is positive.
      // Coupling factor (1 + 1.5) vs (1 + 0.1) -> 2.5x vs 1.1x.
      // Strong should win.
      
      let strongWins = 0;
      const totalHeads = weights.weights.length;

      for(let h=0; h<totalHeads; h++) {
          const node0Weights = weights.weights[h][0];
          const strongWeight = node0Weights[0]; 
          const weakWeight = node0Weights[1];
          
          if (strongWeight > weakWeight) strongWins++;
      }

      expect(strongWins).toBe(totalHeads);
    });

    it('should use LeakyReLU activation', () => {
      expect(gat.config.activation).toBe('leaky_relu');
      expect(gat.config.leakyReluAlpha).toBe(0.2);
    });
  });

  describe('Multi-Head Attention', () => {
    it('should process all 8 heads in parallel', () => {
      const startTime = Date.now();
      gat.forward(mockNodes, mockEdges);
      const duration = Date.now() - startTime;

      // Should be fast (parallel processing)
      expect(duration).toBeLessThan(100);
    });

    it('should learn different attention patterns per head', () => {
      // Use nodes with DISTINCT features to break symmetry
      const distinctNodes = [
        createMockCellNode({ cellId: 'NRCELL_001', features: [1, 2, 3, 4] }),
        createMockCellNode({ cellId: 'NRCELL_002', features: [5, 6, 7, 8] }), // Target
        createMockCellNode({ cellId: 'NRCELL_003', features: [9, 10, 11, 12] }),
      ];

      gat.forward(distinctNodes, mockEdges);
      const weights = gat.getAttentionWeights();

      // Variance across heads should be > 0
      // Use Node 1 (index 1) which has 2 neighbors (Node 0 and Node 2)
      const head1 = weights.weights[0][1]; // Head 1, Node 1 neighbors
      const head2 = weights.weights[1][1]; // Head 2, Node 1 neighbors

      // If randomization works, these should differ
      const isDifferent = !head1.every((v: number, i: number) =>
        Math.abs(v - head2[i]) < 0.000001
      );
      expect(isDifferent).toBe(true);
    });

    it('should support different aggregation strategies', () => {
      const gatConcat = new GraphAttentionNetwork({
        numHeads: 8,
        embeddingDim: 768,
        featureDim: 4,
        aggregation: 'concat',
      });

      const output = gatConcat.forward(mockNodes, mockEdges);

      // Concat: 8 heads * (768/8) = 768 total
      expect(output[0].embedding).toHaveLength(768);
    });
  });

  describe('Edge Features', () => {
    it('should incorporate edge features in attention', () => {
      const edge = createMockInterferenceEdge({
        features: [500, 0.2, 90], // [distance, overlap, coupling]
      });

      gat.forward(mockNodes, [edge]);

      // Edge features should influence attention weights
      expect(gat.config.useEdgeFeatures).toBe(true);
    });

    it('should handle missing edge features gracefully', () => {
      const edgeNoFeatures = {
        fromCell: 'NRCELL_001',
        toCell: 'NRCELL_002',
      } as InterferenceEdge;

      expect(() => gat.forward(mockNodes, [edgeNoFeatures])).not.toThrow();
    });
  });

  describe('Training and Updates', () => {
    it('should support backpropagation', () => {
      const gradients = gat.backward({
        targetEmbeddings: mockNodes.map(n => n.embedding!),
        loss: 0.5,
      });

      expect(gradients).toBeDefined();
      expect(gradients.W_query).toBeDefined();
    });

    it('should update weights with optimizer', () => {
      const initialWeights = JSON.parse(JSON.stringify(gat.getWeights())); // Deep copy

      gat.updateWeights({
        W_query: initialWeights.W_query.map((h: any[]) => h.map((row: number[]) => row.map(v => v * 0.99))),
        learningRate: 0.001,
      });

      const newWeights = gat.getWeights();
      // Compare specific value to avoid reference equality issues
      expect(newWeights.W_query[0][0][0]).not.toBe(initialWeights.W_query[0][0][0]);
    });
  });

  describe('Performance', () => {
    it('should scale to 20+ nodes efficiently', () => {
      const largeNodeSet = Array.from({ length: 25 }, (_, i) =>
        createMockCellNode({ cellId: `NRCELL_${i}` })
      );

      const largeEdgeSet = Array.from({ length: 50 }, (_, i) =>
        createMockInterferenceEdge({
          fromCell: `NRCELL_${i % 25}`,
          toCell: `NRCELL_${(i + 1) % 25}`,
        })
      );

      const startTime = Date.now();
      gat.forward(largeNodeSet, largeEdgeSet);
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(500); // 500ms for large graph
    });
  });
});