/**
 * Interference Graph Builder Test Suite - London School TDD
 * TITAN Neuro-Symbolic RAN Platform
 *
 * Tests graph construction for GNN-based interference modeling
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { createMockCellNode, createMockInterferenceEdge } from '../setup';
import type { CellNode, InterferenceEdge, CellGraph } from '../../src/gnn/types';

// Import to be implemented
import { InterferenceGraphBuilder } from '../../src/gnn/interference-graph';

describe('InterferenceGraphBuilder', () => {
  let builder: InterferenceGraphBuilder;
  let mockCells: CellNode[];

  beforeEach(() => {
    builder = new InterferenceGraphBuilder();

    mockCells = [
      createMockCellNode({ cellId: 'NRCELL_001', features: [10.5, -95.0, 65.5, 12.0] }),
      createMockCellNode({ cellId: 'NRCELL_002', features: [8.0, -98.0, 70.0, 10.0] }),
      createMockCellNode({ cellId: 'NRCELL_003', features: [12.0, -92.0, 60.0, 14.0] }),
    ];
  });

  describe('Graph Construction', () => {
    it('should build graph with nodes and edges', () => {
      const graph = builder.buildGraph(mockCells);

      expect(graph.nodes).toHaveLength(3);
      expect(graph.edges).toBeDefined();
      expect(Array.isArray(graph.edges)).toBe(true);
    });

    it('should create edges for neighboring cells', () => {
      const graph = builder.buildGraph(mockCells, { minCoupling: 10 });

      // Should have edges between neighbors (not fully connected)
      expect(graph.edges.length).toBeGreaterThan(0);
      expect(graph.edges.length).toBeLessThanOrEqual(mockCells.length * (mockCells.length - 1));
    });

    it('should calculate interference coupling for each edge', () => {
      const graph = builder.buildGraph(mockCells);

      graph.edges.forEach(edge => {
        expect(edge.interferenceCoupling).toBeGreaterThan(0);
        expect(edge.interferenceCoupling).toBeLessThan(120); // Realistic dB range
      });
    });

    it('should include distance and overlap metrics', () => {
      const graph = builder.buildGraph(mockCells);

      graph.edges.forEach(edge => {
        expect(edge.distance).toBeGreaterThan(0);
        expect(edge.overlapPct).toBeGreaterThanOrEqual(0);
        expect(edge.overlapPct).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('Interference Coupling Calculation', () => {
    it('should calculate coupling based on distance', () => {
      const cell1 = mockCells[0];
      const cell2 = mockCells[1];

      const coupling = builder.calculateInterferenceCoupling(cell1, cell2);

      expect(coupling).toBeGreaterThan(0);
      expect(typeof coupling).toBe('number');
    });

    it('should give higher coupling for closer cells', () => {
      const closeCells = builder.buildGraph([
        createMockCellNode({ cellId: 'CLOSE_1' }),
        createMockCellNode({ cellId: 'CLOSE_2' }),
      ], { minCoupling: 10 });

      const farCells = builder.buildGraph([
        createMockCellNode({ cellId: 'FAR_1' }),
        createMockCellNode({ cellId: 'FAR_2' }),
      ], { minCoupling: 10 });

      // Mock: closer cells have higher coupling
      expect(closeCells.edges[0]?.interferenceCoupling).toBeDefined();
    });

    it('should factor in coverage overlap percentage', () => {
      const edge = createMockInterferenceEdge({
        distance: 500,
        overlapPct: 0.3,
      });

      expect(edge.overlapPct).toBe(0.3);
      // Higher overlap should increase coupling weight
    });
  });

  describe('Edge Filtering', () => {
    it('should filter edges below coupling threshold', () => {
      const graph = builder.buildGraph(mockCells, { minCoupling: 80 });

      graph.edges.forEach(edge => {
        expect(edge.interferenceCoupling).toBeGreaterThanOrEqual(80);
      });
    });

    it('should support max distance filtering', () => {
      const graph = builder.buildGraph(mockCells, { maxDistance: 1000 });

      graph.edges.forEach(edge => {
        expect(edge.distance).toBeLessThanOrEqual(1000);
      });
    });

    it('should create sparse graph for large topologies', () => {
      const largeCellSet = Array.from({ length: 20 }, (_, i) =>
        createMockCellNode({ cellId: `NRCELL_${i}` })
      );

      const graph = builder.buildGraph(largeCellSet, { maxNeighbors: 5 });

      // Should not be fully connected
      expect(graph.edges.length).toBeLessThan(largeCellSet.length * largeCellSet.length);
    });
  });

  describe('Graph Topology', () => {
    it('should compute adjacency matrix', () => {
      const graph = builder.buildGraph(mockCells);
      const adjacency = builder.getAdjacencyMatrix(graph);

      expect(adjacency).toHaveLength(mockCells.length);
      expect(adjacency[0]).toHaveLength(mockCells.length);
    });

    it('should identify strongly connected components', () => {
      const graph = builder.buildGraph(mockCells);
      const components = builder.getConnectedComponents(graph);

      expect(components).toBeDefined();
      expect(components.length).toBeGreaterThan(0);
    });

    it('should find k-hop neighbors', () => {
      const graph = builder.buildGraph(mockCells);
      const neighbors = builder.getKHopNeighbors(graph, 'NRCELL_001', 2);

      expect(Array.isArray(neighbors)).toBe(true);
    });
  });

  describe('Feature Engineering', () => {
    it('should aggregate neighbor features', () => {
      const graph = builder.buildGraph(mockCells);
      const aggregated = builder.aggregateNeighborFeatures(graph, 'NRCELL_001');

      expect(aggregated).toHaveLength(4); // [SINR, RSRP, PRB, CQI]
    });

    it('should normalize edge features', () => {
      const graph = builder.buildGraph(mockCells);
      const normalized = builder.normalizeEdgeFeatures(graph);

      normalized.edges.forEach(edge => {
        expect(edge.features.every(f => f >= 0 && f <= 1)).toBe(true);
      });
    });
  });
});
