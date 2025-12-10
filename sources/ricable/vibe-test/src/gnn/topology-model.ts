/**
 * Topology Model - Graph-based representation of RAN topology
 * Implements ruvector patterns for spatial intelligence
 */

import { v4 as uuidv4 } from 'uuid';
import {
  CellEmbedding,
  CellState,
  GraphEdge,
  NetworkGraph,
  StaticFeatures,
  DynamicFeatures,
  NeighborRelation,
  CellConfiguration,
  CellMetrics,
} from '../core/types.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('TopologyModel');

/**
 * Compression level for tensor storage
 */
export type CompressionLevel = 'none' | 'half' | 'pq8' | 'pq4' | 'binary';

/**
 * Topology Model - Graph representation of the network
 */
export class TopologyModel {
  private graph: NetworkGraph;
  private embeddingDim: number;
  private compressionLevels: Map<string, CompressionLevel>;
  private hotCells: Set<string>;
  private updateQueue: string[];

  constructor(embeddingDim: number = 64) {
    this.embeddingDim = embeddingDim;
    this.graph = {
      nodes: new Map(),
      edges: [],
    };
    this.compressionLevels = new Map();
    this.hotCells = new Set();
    this.updateQueue = [];

    logger.info('Topology model initialized', { embeddingDim });
  }

  /**
   * Add a cell to the topology
   */
  addCell(cellState: CellState): CellEmbedding {
    const embedding = this.createEmbedding(cellState);
    this.graph.nodes.set(cellState.identity.cellId, embedding);

    // Add edges for neighbor relations
    for (const neighbor of cellState.neighbors) {
      this.addEdge(
        cellState.identity.cellId,
        neighbor.targetCellId,
        'neighbor',
        1 - neighbor.interferenceLevel / 100
      );
    }

    logger.debug('Cell added to topology', {
      cellId: cellState.identity.cellId,
      neighborCount: cellState.neighbors.length,
    });

    return embedding;
  }

  /**
   * Update cell state and re-compute embedding
   */
  updateCell(cellId: string, metrics: CellMetrics): void {
    const embedding = this.graph.nodes.get(cellId);
    if (!embedding) {
      logger.warn('Cell not found for update', { cellId });
      return;
    }

    // Update dynamic features
    embedding.dynamicFeatures = {
      load: metrics.prbUtilizationDl,
      interferenceLevel: metrics.interferenceLevel,
      throughput: metrics.throughputDl,
      userCount: metrics.activeUesDl,
    };
    embedding.lastUpdated = Date.now();

    // Re-compute embedding vector
    this.computeEmbeddingVector(embedding);

    // Mark as hot cell if high activity
    if (metrics.prbUtilizationDl > 0.7 || metrics.activeUesDl > 50) {
      this.hotCells.add(cellId);
      this.compressionLevels.set(cellId, 'none');
    } else if (this.hotCells.has(cellId)) {
      this.hotCells.delete(cellId);
      this.compressionLevels.set(cellId, 'pq4');
    }
  }

  /**
   * Add an edge between cells
   */
  addEdge(
    sourceId: string,
    targetId: string,
    edgeType: 'neighbor' | 'interferer' | 'handover',
    weight: number
  ): void {
    // Check for existing edge
    const existing = this.graph.edges.find(
      (e) => e.sourceId === sourceId && e.targetId === targetId && e.edgeType === edgeType
    );

    if (existing) {
      existing.weight = weight;
      return;
    }

    this.graph.edges.push({
      sourceId,
      targetId,
      edgeType,
      weight,
      attributes: {},
    });
  }

  /**
   * Get neighbors of a cell
   */
  getNeighbors(cellId: string, edgeType?: string): string[] {
    return this.graph.edges
      .filter(
        (e) =>
          (e.sourceId === cellId || e.targetId === cellId) &&
          (!edgeType || e.edgeType === edgeType)
      )
      .map((e) => (e.sourceId === cellId ? e.targetId : e.sourceId));
  }

  /**
   * Find k-nearest neighbors by embedding similarity
   */
  findKNearestNeighbors(cellId: string, k: number): { cellId: string; similarity: number }[] {
    const sourceEmbedding = this.graph.nodes.get(cellId);
    if (!sourceEmbedding) return [];

    const similarities: { cellId: string; similarity: number }[] = [];

    for (const [otherId, otherEmbedding] of this.graph.nodes) {
      if (otherId === cellId) continue;

      const similarity = this.cosineSimilarity(
        sourceEmbedding.vector,
        otherEmbedding.vector
      );
      similarities.push({ cellId: otherId, similarity });
    }

    return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, k);
  }

  /**
   * Detect handover loops using cycle detection
   */
  detectHandoverLoops(): string[][] {
    const loops: string[][] = [];
    const visited = new Set<string>();
    const recStack = new Set<string>();
    const path: string[] = [];

    const handoverEdges = this.graph.edges.filter((e) => e.edgeType === 'handover');

    // Build adjacency list for handover edges
    const adj = new Map<string, string[]>();
    for (const edge of handoverEdges) {
      if (!adj.has(edge.sourceId)) adj.set(edge.sourceId, []);
      adj.get(edge.sourceId)!.push(edge.targetId);
    }

    const dfs = (node: string): void => {
      visited.add(node);
      recStack.add(node);
      path.push(node);

      const neighbors = adj.get(node) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          dfs(neighbor);
        } else if (recStack.has(neighbor)) {
          // Found a cycle
          const cycleStart = path.indexOf(neighbor);
          if (cycleStart !== -1) {
            loops.push(path.slice(cycleStart));
          }
        }
      }

      path.pop();
      recStack.delete(node);
    };

    for (const [cellId] of this.graph.nodes) {
      if (!visited.has(cellId)) {
        dfs(cellId);
      }
    }

    if (loops.length > 0) {
      logger.warn('Handover loops detected', { loopCount: loops.length });
    }

    return loops;
  }

  /**
   * Detect PCI collisions using graph coloring analysis
   */
  detectPCICollisions(): PCICollision[] {
    const collisions: PCICollision[] = [];
    const pciMap = new Map<number, string[]>();

    // Group cells by PCI
    for (const [cellId, embedding] of this.graph.nodes) {
      // Assume PCI is encoded in static features
      const pci = Math.floor(embedding.staticFeatures.azimuth) % 504; // Simplified
      if (!pciMap.has(pci)) pciMap.set(pci, []);
      pciMap.get(pci)!.push(cellId);
    }

    // Check for collisions among neighbors
    for (const [cellId, embedding] of this.graph.nodes) {
      const neighbors = this.getNeighbors(cellId, 'neighbor');
      const cellPci = Math.floor(embedding.staticFeatures.azimuth) % 504;

      for (const neighborId of neighbors) {
        const neighborEmbedding = this.graph.nodes.get(neighborId);
        if (!neighborEmbedding) continue;

        const neighborPci = Math.floor(neighborEmbedding.staticFeatures.azimuth) % 504;

        if (cellPci === neighborPci) {
          // Check if collision already recorded
          const exists = collisions.some(
            (c) =>
              (c.cellId1 === cellId && c.cellId2 === neighborId) ||
              (c.cellId1 === neighborId && c.cellId2 === cellId)
          );

          if (!exists) {
            collisions.push({
              cellId1: cellId,
              cellId2: neighborId,
              pci: cellPci,
              severity: this.calculateCollisionSeverity(cellId, neighborId),
            });
          }
        }
      }
    }

    if (collisions.length > 0) {
      logger.warn('PCI collisions detected', { collisionCount: collisions.length });
    }

    return collisions;
  }

  /**
   * Optimize PCI assignment using graph coloring
   */
  optimizePCIAssignment(): Map<string, number> {
    const assignment = new Map<string, number>();
    const maxPCI = 504;

    // Order cells by degree (number of neighbors)
    const cellsByDegree = Array.from(this.graph.nodes.keys())
      .map((cellId) => ({
        cellId,
        degree: this.getNeighbors(cellId).length,
      }))
      .sort((a, b) => b.degree - a.degree);

    for (const { cellId } of cellsByDegree) {
      const neighborPCIs = new Set<number>();

      for (const neighborId of this.getNeighbors(cellId)) {
        const neighborPCI = assignment.get(neighborId);
        if (neighborPCI !== undefined) {
          neighborPCIs.add(neighborPCI);
          // Also avoid adjacent PCIs for reference signal orthogonality
          neighborPCIs.add((neighborPCI + 1) % maxPCI);
          neighborPCIs.add((neighborPCI - 1 + maxPCI) % maxPCI);
        }
      }

      // Find first available PCI
      let pci = 0;
      while (neighborPCIs.has(pci) && pci < maxPCI) {
        pci += 3; // Skip by 3 for better orthogonality
      }

      assignment.set(cellId, pci % maxPCI);
    }

    logger.info('PCI assignment optimized', { cellCount: assignment.size });
    return assignment;
  }

  /**
   * Find shortest path between cells (for handover path analysis)
   * Uses Dijkstra's algorithm
   */
  findShortestPath(sourceId: string, targetId: string): PathResult | null {
    if (!this.graph.nodes.has(sourceId) || !this.graph.nodes.has(targetId)) {
      return null;
    }

    const distances = new Map<string, number>();
    const previous = new Map<string, string>();
    const unvisited = new Set<string>();

    for (const [cellId] of this.graph.nodes) {
      distances.set(cellId, Infinity);
      unvisited.add(cellId);
    }
    distances.set(sourceId, 0);

    while (unvisited.size > 0) {
      // Find minimum distance node
      let minNode: string | null = null;
      let minDist = Infinity;
      for (const node of unvisited) {
        const dist = distances.get(node)!;
        if (dist < minDist) {
          minDist = dist;
          minNode = node;
        }
      }

      if (minNode === null || minDist === Infinity) break;
      if (minNode === targetId) break;

      unvisited.delete(minNode);

      // Update neighbors
      const neighbors = this.getNeighbors(minNode);
      for (const neighbor of neighbors) {
        if (!unvisited.has(neighbor)) continue;

        const edge = this.graph.edges.find(
          (e) =>
            (e.sourceId === minNode && e.targetId === neighbor) ||
            (e.targetId === minNode && e.sourceId === neighbor)
        );
        const weight = edge ? 1 / edge.weight : 1;
        const alt = minDist + weight;

        if (alt < distances.get(neighbor)!) {
          distances.set(neighbor, alt);
          previous.set(neighbor, minNode);
        }
      }
    }

    // Reconstruct path
    if (!previous.has(targetId) && sourceId !== targetId) {
      return null;
    }

    const path: string[] = [];
    let current: string | undefined = targetId;
    while (current) {
      path.unshift(current);
      current = previous.get(current);
    }

    return {
      path,
      totalCost: distances.get(targetId) || Infinity,
      hops: path.length - 1,
    };
  }

  /**
   * Create embedding for a cell
   */
  private createEmbedding(cellState: CellState): CellEmbedding {
    const staticFeatures: StaticFeatures = {
      azimuth: cellState.location.azimuth,
      height: cellState.location.altitude,
      beamwidth: 65, // Default
      technology: cellState.identity.technology === '5G' ? 1 : 0,
      sectorCount: 3, // Default
    };

    const dynamicFeatures: DynamicFeatures = {
      load: cellState.metrics.prbUtilizationDl,
      interferenceLevel: cellState.metrics.interferenceLevel,
      throughput: cellState.metrics.throughputDl,
      userCount: cellState.metrics.activeUesDl,
    };

    const embedding: CellEmbedding = {
      cellId: cellState.identity.cellId,
      vector: new Float32Array(this.embeddingDim),
      staticFeatures,
      dynamicFeatures,
      lastUpdated: Date.now(),
    };

    this.computeEmbeddingVector(embedding);
    return embedding;
  }

  /**
   * Compute embedding vector from features
   */
  private computeEmbeddingVector(embedding: CellEmbedding): void {
    const vector = embedding.vector;
    const s = embedding.staticFeatures;
    const d = embedding.dynamicFeatures;

    // Normalize and encode features into embedding
    // First half: static features (normalized)
    vector[0] = s.azimuth / 360;
    vector[1] = s.height / 100;
    vector[2] = s.beamwidth / 180;
    vector[3] = s.technology;
    vector[4] = s.sectorCount / 6;

    // Second quarter: dynamic features
    const offset = Math.floor(this.embeddingDim / 4);
    vector[offset] = d.load;
    vector[offset + 1] = (d.interferenceLevel + 120) / 40; // Normalize from [-120, -80]
    vector[offset + 2] = Math.min(1, d.throughput / 1000); // Normalize to 1 Gbps
    vector[offset + 3] = Math.min(1, d.userCount / 100);

    // Rest: learned features (would be from GNN in production)
    // For now, use hash-based pseudo-random features
    const seed = this.hashString(embedding.cellId);
    for (let i = offset + 4; i < this.embeddingDim; i++) {
      vector[i] = this.pseudoRandom(seed + i);
    }
  }

  /**
   * Cosine similarity between two vectors
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }

  /**
   * Calculate collision severity based on interference potential
   */
  private calculateCollisionSeverity(cellId1: string, cellId2: string): number {
    const emb1 = this.graph.nodes.get(cellId1);
    const emb2 = this.graph.nodes.get(cellId2);

    if (!emb1 || !emb2) return 0;

    // Higher severity if both cells have high load
    const loadFactor = (emb1.dynamicFeatures.load + emb2.dynamicFeatures.load) / 2;

    // Higher severity if cells are spatially close (similar embeddings)
    const similarity = this.cosineSimilarity(emb1.vector, emb2.vector);

    return loadFactor * 0.5 + similarity * 0.5;
  }

  /**
   * Simple hash function for strings
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash |= 0;
    }
    return Math.abs(hash);
  }

  /**
   * Pseudo-random number generator
   */
  private pseudoRandom(seed: number): number {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  }

  /**
   * Get the full network graph
   */
  getGraph(): NetworkGraph {
    return this.graph;
  }

  /**
   * Get cell embedding
   */
  getEmbedding(cellId: string): CellEmbedding | undefined {
    return this.graph.nodes.get(cellId);
  }

  /**
   * Get statistics about the topology
   */
  getStats(): TopologyStats {
    const nodes = this.graph.nodes.size;
    const edges = this.graph.edges.length;
    const avgDegree = nodes > 0 ? (2 * edges) / nodes : 0;

    const edgeTypes = new Map<string, number>();
    for (const edge of this.graph.edges) {
      edgeTypes.set(edge.edgeType, (edgeTypes.get(edge.edgeType) || 0) + 1);
    }

    return {
      nodeCount: nodes,
      edgeCount: edges,
      averageDegree: avgDegree,
      hotCellCount: this.hotCells.size,
      edgeTypeBreakdown: Object.fromEntries(edgeTypes),
    };
  }
}

/**
 * PCI Collision information
 */
export interface PCICollision {
  cellId1: string;
  cellId2: string;
  pci: number;
  severity: number;
}

/**
 * Path finding result
 */
export interface PathResult {
  path: string[];
  totalCost: number;
  hops: number;
}

/**
 * Topology statistics
 */
export interface TopologyStats {
  nodeCount: number;
  edgeCount: number;
  averageDegree: number;
  hotCellCount: number;
  edgeTypeBreakdown: Record<string, number>;
}

/**
 * Create a configured topology model instance
 */
export function createTopologyModel(embeddingDim?: number): TopologyModel {
  return new TopologyModel(embeddingDim);
}
