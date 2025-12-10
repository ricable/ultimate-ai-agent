/**
 * Interference Graph Builder - TITAN Neuro-Symbolic RAN Platform
 *
 * Constructs interference graphs for GNN-based optimization
 * Models inter-cell interference using distance, overlap, and coupling metrics
 */

import type {
  CellNode,
  InterferenceEdge,
  CellGraph,
  GraphBuilderOptions,
} from './types';

/**
 * Builds interference graphs from cell topology
 */
export class InterferenceGraphBuilder {
  private adjacencyCache: Map<string, number[][]> = new Map();

  /**
   * Build complete interference graph from cell nodes
   */
  buildGraph(
    cells: CellNode[],
    options: GraphBuilderOptions = {}
  ): CellGraph {
    const {
      minCoupling = 70, // dB threshold
      maxDistance = 2000, // meters
      maxNeighbors = 8,
    } = options;

    const edges: InterferenceEdge[] = [];

    // Create edges for all cell pairs within constraints
    for (let i = 0; i < cells.length; i++) {
      const neighborDistances: Array<{ index: number; distance: number; coupling: number }> = [];

      for (let j = 0; j < cells.length; j++) {
        if (i === j) continue;

        const distance = this.calculateDistance(cells[i], cells[j]);
        if (distance > maxDistance) continue;

        const coupling = this.calculateInterferenceCoupling(cells[i], cells[j]);
        if (coupling < minCoupling) continue;

        neighborDistances.push({ index: j, distance, coupling });
      }

      // Sort by coupling strength and take top N neighbors
      neighborDistances
        .sort((a, b) => b.coupling - a.coupling)
        .slice(0, maxNeighbors)
        .forEach(({ index, distance, coupling }) => {
          const overlap = this.calculateOverlap(cells[i], cells[index]);

          edges.push({
            fromCell: cells[i].cellId,
            toCell: cells[index].cellId,
            features: [distance, overlap, coupling],
            distance,
            overlapPct: overlap,
            interferenceCoupling: coupling,
          });
        });
    }

    return { nodes: cells, edges };
  }

  /**
   * Calculate interference coupling between two cells
   * Based on distance, overlap, and propagation characteristics
   */
  calculateInterferenceCoupling(cell1: CellNode, cell2: CellNode): number {
    const distance = this.calculateDistance(cell1, cell2);
    const overlap = this.calculateOverlap(cell1, cell2);

    // Path loss model (simplified 3GPP Urban Macro)
    // PL = 128.1 + 37.6 * log10(d_km)
    const distanceKm = distance / 1000;
    const pathLoss = 128.1 + 37.6 * Math.log10(Math.max(0.001, distanceKm));

    // Coupling loss = Path loss - Antenna gain - Overlap factor
    const antennaGain = 15; // dBi
    const overlapFactor = overlap * 10; // Higher overlap = stronger coupling

    const couplingLoss = pathLoss - antennaGain - overlapFactor;

    // Return coupling strength (inverse of loss)
    // Higher coupling = stronger interference
    return Math.max(0, 140 - couplingLoss);
  }

  /**
   * Calculate distance between cells (simplified)
   * In production, use actual GPS coordinates
   */
  private calculateDistance(cell1: CellNode, cell2: CellNode): number {
    // Mock distance based on cell ID hash
    const hash1 = this.hashCellId(cell1.cellId);
    const hash2 = this.hashCellId(cell2.cellId);

    // Simulate realistic distances: 200m to 2000m
    return 200 + Math.abs(hash1 - hash2) % 1800;
  }

  /**
   * Calculate coverage overlap percentage
   */
  private calculateOverlap(cell1: CellNode, cell2: CellNode): number {
    const distance = this.calculateDistance(cell1, cell2);

    // Simplified overlap model: inversely proportional to distance
    // Cells closer than 500m have high overlap
    if (distance < 500) return 0.3 + (500 - distance) / 1000;
    if (distance < 1000) return 0.15 + (1000 - distance) / 2000;
    return Math.max(0, 0.1 - (distance - 1000) / 10000);
  }

  /**
   * Hash cell ID to deterministic number
   */
  private hashCellId(cellId: string): number {
    let hash = 0;
    for (let i = 0; i < cellId.length; i++) {
      hash = (hash << 5) - hash + cellId.charCodeAt(i);
      hash |= 0; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Get adjacency matrix representation
   */
  getAdjacencyMatrix(graph: CellGraph): number[][] {
    const cacheKey = graph.nodes.map(n => n.cellId).join(',');
    if (this.adjacencyCache.has(cacheKey)) {
      return this.adjacencyCache.get(cacheKey)!;
    }

    const n = graph.nodes.length;
    const matrix = Array(n).fill(0).map(() => Array(n).fill(0));

    const nodeIndexMap = new Map<string, number>();
    graph.nodes.forEach((node, i) => nodeIndexMap.set(node.cellId, i));

    graph.edges.forEach(edge => {
      const i = nodeIndexMap.get(edge.fromCell)!;
      const j = nodeIndexMap.get(edge.toCell)!;
      matrix[i][j] = edge.interferenceCoupling;
    });

    this.adjacencyCache.set(cacheKey, matrix);
    return matrix;
  }

  /**
   * Find connected components in graph
   */
  getConnectedComponents(graph: CellGraph): string[][] {
    const visited = new Set<string>();
    const components: string[][] = [];

    const dfs = (nodeId: string, component: string[]) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      component.push(nodeId);

      graph.edges
        .filter(e => e.fromCell === nodeId || e.toCell === nodeId)
        .forEach(edge => {
          const neighbor = edge.fromCell === nodeId ? edge.toCell : edge.fromCell;
          dfs(neighbor, component);
        });
    };

    graph.nodes.forEach(node => {
      if (!visited.has(node.cellId)) {
        const component: string[] = [];
        dfs(node.cellId, component);
        components.push(component);
      }
    });

    return components;
  }

  /**
   * Get k-hop neighbors
   */
  getKHopNeighbors(graph: CellGraph, cellId: string, k: number): string[] {
    const neighbors = new Set<string>();
    let frontier = new Set([cellId]);

    for (let hop = 0; hop < k; hop++) {
      const nextFrontier = new Set<string>();

      frontier.forEach(nodeId => {
        graph.edges
          .filter(e => e.fromCell === nodeId || e.toCell === nodeId)
          .forEach(edge => {
            const neighbor = edge.fromCell === nodeId ? edge.toCell : edge.fromCell;
            if (neighbor !== cellId && !neighbors.has(neighbor)) {
              nextFrontier.add(neighbor);
              neighbors.add(neighbor);
            }
          });
      });

      frontier = nextFrontier;
    }

    return Array.from(neighbors);
  }

  /**
   * Aggregate features from neighbors
   */
  aggregateNeighborFeatures(graph: CellGraph, cellId: string): number[] {
    const node = graph.nodes.find(n => n.cellId === cellId);
    if (!node) return [0, 0, 0, 0];

    const neighbors = graph.edges
      .filter(e => e.fromCell === cellId || e.toCell === cellId)
      .map(e => e.fromCell === cellId ? e.toCell : e.fromCell);

    if (neighbors.length === 0) return node.features;

    const neighborNodes = graph.nodes.filter(n => neighbors.includes(n.cellId));

    // Mean aggregation
    const aggregated = node.features.map((_, i) => {
      const sum = neighborNodes.reduce((acc, n) => acc + n.features[i], 0);
      return sum / neighborNodes.length;
    });

    return aggregated;
  }

  /**
   * Normalize edge features to [0, 1]
   */
  normalizeEdgeFeatures(graph: CellGraph): CellGraph {
    const maxDistance = Math.max(...graph.edges.map(e => e.distance));
    const maxCoupling = Math.max(...graph.edges.map(e => e.interferenceCoupling));

    const normalizedEdges = graph.edges.map(edge => ({
      ...edge,
      features: [
        edge.distance / maxDistance,
        edge.overlapPct, // Already [0, 1]
        edge.interferenceCoupling / maxCoupling,
      ] as [number, number, number],
    }));

    return { ...graph, edges: normalizedEdges };
  }
}
