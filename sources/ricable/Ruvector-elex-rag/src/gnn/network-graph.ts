/**
 * Network Graph Construction for Ericsson RAN
 *
 * Builds graph representation of cellular network topology for GNN processing.
 * Handles:
 * - Cell node creation with feature vectors
 * - Interference edge construction
 * - Adjacency matrix generation
 * - Voronoi tessellation for physical adjacency
 */

import { v4 as uuidv4 } from 'uuid';
import type {
  NetworkGraph,
  CellNode,
  InterferenceEdge,
  CellConfiguration,
  PerformanceMetrics,
  PowerControlParams,
  AlphaValue,
} from '../core/types.js';
import { getConfig } from '../core/config.js';
import { logger } from '../utils/logger.js';

export interface GraphConstructionOptions {
  /** Maximum distance for edge creation (meters) */
  maxEdgeDistance: number;
  /** Minimum handover count for ANR-derived edges */
  minHandoverCount: number;
  /** Include virtual edges from SON relations */
  includeSonRelations: boolean;
  /** Feature normalization method */
  normalization: 'minmax' | 'zscore' | 'none';
}

const DEFAULT_OPTIONS: GraphConstructionOptions = {
  maxEdgeDistance: 5000, // 5km
  minHandoverCount: 10,
  includeSonRelations: true,
  normalization: 'zscore',
};

export class NetworkGraphBuilder {
  private options: GraphConstructionOptions;
  private config = getConfig();

  constructor(options: Partial<GraphConstructionOptions> = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /**
   * Build a network graph from cell configurations and metrics
   */
  buildGraph(
    cells: CellConfiguration[],
    metrics: Map<string, PerformanceMetrics>,
    neighborRelations?: Map<string, string[]>,
    handoverStats?: Map<string, Map<string, number>>
  ): NetworkGraph {
    const graphId = uuidv4();
    const clusterId = this.determineClusterId(cells);

    logger.info('Building network graph', {
      graphId,
      cellCount: cells.length,
      hasMetrics: metrics.size > 0,
      hasNeighborRelations: neighborRelations ? neighborRelations.size > 0 : false,
    });

    // Create cell nodes
    const nodes = new Map<string, CellNode>();
    for (const cell of cells) {
      const cellId = cell.ecgi || cell.ncgi || uuidv4();
      const nodeMetrics = metrics.get(cellId);

      const node = this.createCellNode(cell, nodeMetrics);
      nodes.set(cellId, node);
    }

    // Build edges
    const edges = this.buildEdges(
      nodes,
      neighborRelations,
      handoverStats
    );

    // Build adjacency list
    const adjacencyList = this.buildAdjacencyList(edges);

    const graph: NetworkGraph = {
      id: graphId,
      name: `Cluster_${clusterId}`,
      clusterId,
      nodes,
      edges,
      adjacencyList,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    logger.info('Network graph built', {
      graphId,
      nodeCount: nodes.size,
      edgeCount: edges.length,
    });

    return graph;
  }

  /**
   * Create a cell node with feature vector
   */
  private createCellNode(
    config: CellConfiguration,
    metrics?: PerformanceMetrics
  ): CellNode {
    const cellId = config.ecgi || config.ncgi || uuidv4();

    // Build feature vector
    // Features: [p0, alpha, tilt, azimuth, height, maxPower, bandwidth,
    //            sinrMean, sinrP5, sinrP95, load, throughput, activeUsers]
    const features = new Float32Array(13);

    // Configuration features
    features[0] = config.powerControl.pZeroNominalPusch;
    features[1] = config.powerControl.alpha;
    features[2] = config.antennaTilt;
    features[3] = config.azimuth;
    features[4] = config.height;
    features[5] = config.maxTxPower;
    features[6] = config.bandwidth;

    // Performance features (default to 0 if no metrics)
    if (metrics) {
      features[7] = metrics.pmPuschSinr.mean;
      features[8] = metrics.pmPuschSinr.p5;
      features[9] = metrics.pmPuschSinr.p95;
      features[10] = metrics.pmUlPrbUtilization;
      features[11] = metrics.pmUlThroughput;
      features[12] = metrics.pmActiveUsers;
    }

    return {
      id: cellId,
      config,
      metrics,
      coordinates: {
        latitude: 0, // Should be provided from MDT or site data
        longitude: 0,
      },
      features,
    };
  }

  /**
   * Build interference edges between cells
   */
  private buildEdges(
    nodes: Map<string, CellNode>,
    neighborRelations?: Map<string, string[]>,
    handoverStats?: Map<string, Map<string, number>>
  ): InterferenceEdge[] {
    const edges: InterferenceEdge[] = [];
    const edgeSet = new Set<string>();

    // Build edges from neighbor relations (ANR)
    if (neighborRelations) {
      for (const [sourceId, neighbors] of neighborRelations) {
        if (!nodes.has(sourceId)) continue;

        for (const targetId of neighbors) {
          if (!nodes.has(targetId)) continue;
          if (sourceId === targetId) continue;

          const edgeKey = this.getEdgeKey(sourceId, targetId);
          if (edgeSet.has(edgeKey)) continue;

          const handoverCount = handoverStats?.get(sourceId)?.get(targetId) || 0;
          if (handoverCount < this.options.minHandoverCount && !this.options.includeSonRelations) {
            continue;
          }

          const sourceNode = nodes.get(sourceId)!;
          const targetNode = nodes.get(targetId)!;
          const distance = this.calculateDistance(sourceNode, targetNode);

          if (distance > this.options.maxEdgeDistance) continue;

          const weight = this.calculateEdgeWeight(sourceNode, targetNode, handoverCount, distance);

          edges.push({
            source: sourceId,
            target: targetId,
            weight,
            distance,
            handoverCount,
            anrDerived: true,
          });

          edgeSet.add(edgeKey);
        }
      }
    }

    // Build edges from physical proximity (Voronoi-style)
    const nodeArray = Array.from(nodes.values());
    for (let i = 0; i < nodeArray.length; i++) {
      for (let j = i + 1; j < nodeArray.length; j++) {
        const source = nodeArray[i];
        const target = nodeArray[j];

        const edgeKey = this.getEdgeKey(source.id, target.id);
        if (edgeSet.has(edgeKey)) continue;

        const distance = this.calculateDistance(source, target);
        if (distance > this.options.maxEdgeDistance) continue;

        // Only add physical edges for nearby cells without ANR relation
        if (distance < this.options.maxEdgeDistance / 2) {
          const handoverCount = handoverStats?.get(source.id)?.get(target.id) || 0;
          const weight = this.calculateEdgeWeight(source, target, handoverCount, distance);

          edges.push({
            source: source.id,
            target: target.id,
            weight,
            distance,
            handoverCount,
            anrDerived: false,
          });

          edgeSet.add(edgeKey);
        }
      }
    }

    return edges;
  }

  /**
   * Calculate edge weight based on interference coupling
   */
  private calculateEdgeWeight(
    source: CellNode,
    target: CellNode,
    handoverCount: number,
    distance: number
  ): number {
    // Factors affecting interference coupling:
    // 1. Distance (inverse square law approximation)
    // 2. Antenna alignment (azimuth difference)
    // 3. Handover frequency (strong coupling indicator)
    // 4. Power levels

    // Distance factor (normalized)
    const maxDist = this.options.maxEdgeDistance;
    const distanceFactor = 1 - (distance / maxDist);

    // Azimuth factor (how much antennas point at each other)
    const azimuthDiff = Math.abs(source.config.azimuth - target.config.azimuth);
    const alignedAzimuth = azimuthDiff > 90 && azimuthDiff < 270;
    const azimuthFactor = alignedAzimuth ? 0.8 : 0.4;

    // Handover factor (normalized)
    const maxHandovers = 1000;
    const handoverFactor = Math.min(handoverCount / maxHandovers, 1);

    // Power factor
    const powerDiff = Math.abs(source.config.maxTxPower - target.config.maxTxPower);
    const powerFactor = 1 - (powerDiff / 20); // Normalize by 20dB range

    // Combined weight
    const weight =
      0.4 * distanceFactor +
      0.2 * azimuthFactor +
      0.3 * handoverFactor +
      0.1 * powerFactor;

    return Math.max(0.1, Math.min(1.0, weight));
  }

  /**
   * Calculate distance between two cells (Haversine formula)
   */
  private calculateDistance(source: CellNode, target: CellNode): number {
    const lat1 = source.coordinates.latitude;
    const lon1 = source.coordinates.longitude;
    const lat2 = target.coordinates.latitude;
    const lon2 = target.coordinates.longitude;

    // If coordinates are not set, use a default distance
    if (lat1 === 0 && lon1 === 0 && lat2 === 0 && lon2 === 0) {
      return 500; // Default 500m
    }

    const R = 6371000; // Earth's radius in meters
    const dLat = this.toRad(lat2 - lat1);
    const dLon = this.toRad(lon2 - lon1);

    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(this.toRad(lat1)) *
        Math.cos(this.toRad(lat2)) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
  }

  private toRad(deg: number): number {
    return deg * (Math.PI / 180);
  }

  /**
   * Build adjacency list for efficient graph traversal
   */
  private buildAdjacencyList(edges: InterferenceEdge[]): Map<string, string[]> {
    const adjacencyList = new Map<string, string[]>();

    for (const edge of edges) {
      // Add bidirectional edges
      if (!adjacencyList.has(edge.source)) {
        adjacencyList.set(edge.source, []);
      }
      if (!adjacencyList.has(edge.target)) {
        adjacencyList.set(edge.target, []);
      }

      adjacencyList.get(edge.source)!.push(edge.target);
      adjacencyList.get(edge.target)!.push(edge.source);
    }

    return adjacencyList;
  }

  /**
   * Get unique edge key (order independent)
   */
  private getEdgeKey(source: string, target: string): string {
    return [source, target].sort().join('->');
  }

  /**
   * Determine cluster ID from cells
   */
  private determineClusterId(cells: CellConfiguration[]): string {
    // Use common prefix from cell IDs
    if (cells.length === 0) return 'unknown';

    const ids = cells.map((c) => c.ecgi || c.ncgi || '');
    if (ids[0].length < 5) return uuidv4().substring(0, 8);

    // Find common prefix
    let prefix = ids[0];
    for (const id of ids) {
      while (!id.startsWith(prefix) && prefix.length > 0) {
        prefix = prefix.substring(0, prefix.length - 1);
      }
    }

    return prefix || uuidv4().substring(0, 8);
  }

  /**
   * Normalize feature vectors across all nodes
   */
  normalizeFeatures(graph: NetworkGraph): void {
    if (this.options.normalization === 'none') return;

    const featureDim = 13;
    const nodes = Array.from(graph.nodes.values());

    if (this.options.normalization === 'zscore') {
      // Calculate mean and std for each feature
      const means = new Float32Array(featureDim);
      const stds = new Float32Array(featureDim);

      // Calculate means
      for (const node of nodes) {
        for (let i = 0; i < featureDim; i++) {
          means[i] += node.features[i];
        }
      }
      for (let i = 0; i < featureDim; i++) {
        means[i] /= nodes.length;
      }

      // Calculate stds
      for (const node of nodes) {
        for (let i = 0; i < featureDim; i++) {
          stds[i] += Math.pow(node.features[i] - means[i], 2);
        }
      }
      for (let i = 0; i < featureDim; i++) {
        stds[i] = Math.sqrt(stds[i] / nodes.length);
        if (stds[i] === 0) stds[i] = 1; // Avoid division by zero
      }

      // Normalize
      for (const node of nodes) {
        for (let i = 0; i < featureDim; i++) {
          node.features[i] = (node.features[i] - means[i]) / stds[i];
        }
      }
    } else if (this.options.normalization === 'minmax') {
      // Calculate min and max for each feature
      const mins = new Float32Array(featureDim).fill(Infinity);
      const maxs = new Float32Array(featureDim).fill(-Infinity);

      for (const node of nodes) {
        for (let i = 0; i < featureDim; i++) {
          mins[i] = Math.min(mins[i], node.features[i]);
          maxs[i] = Math.max(maxs[i], node.features[i]);
        }
      }

      // Normalize
      for (const node of nodes) {
        for (let i = 0; i < featureDim; i++) {
          const range = maxs[i] - mins[i];
          if (range === 0) {
            node.features[i] = 0;
          } else {
            node.features[i] = (node.features[i] - mins[i]) / range;
          }
        }
      }
    }
  }

  /**
   * Get adjacency matrix for GNN
   */
  getAdjacencyMatrix(graph: NetworkGraph): Float32Array {
    const n = graph.nodes.size;
    const matrix = new Float32Array(n * n);
    const nodeIds = Array.from(graph.nodes.keys());
    const nodeIndexMap = new Map<string, number>();

    nodeIds.forEach((id, index) => {
      nodeIndexMap.set(id, index);
    });

    for (const edge of graph.edges) {
      const i = nodeIndexMap.get(edge.source);
      const j = nodeIndexMap.get(edge.target);

      if (i !== undefined && j !== undefined) {
        matrix[i * n + j] = edge.weight;
        matrix[j * n + i] = edge.weight; // Symmetric
      }
    }

    return matrix;
  }

  /**
   * Get feature matrix for GNN
   */
  getFeatureMatrix(graph: NetworkGraph): Float32Array {
    const n = graph.nodes.size;
    const featureDim = 13;
    const matrix = new Float32Array(n * featureDim);

    let i = 0;
    for (const node of graph.nodes.values()) {
      for (let j = 0; j < featureDim; j++) {
        matrix[i * featureDim + j] = node.features[j];
      }
      i++;
    }

    return matrix;
  }

  /**
   * Extract subgraph for a specific cluster
   */
  extractSubgraph(
    graph: NetworkGraph,
    centerNodeId: string,
    hops: number = 2
  ): NetworkGraph {
    const visited = new Set<string>();
    const queue: { nodeId: string; depth: number }[] = [
      { nodeId: centerNodeId, depth: 0 },
    ];

    // BFS to find all nodes within k hops
    while (queue.length > 0) {
      const { nodeId, depth } = queue.shift()!;

      if (visited.has(nodeId) || depth > hops) continue;
      visited.add(nodeId);

      const neighbors = graph.adjacencyList.get(nodeId) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          queue.push({ nodeId: neighbor, depth: depth + 1 });
        }
      }
    }

    // Create subgraph
    const subNodes = new Map<string, CellNode>();
    for (const nodeId of visited) {
      const node = graph.nodes.get(nodeId);
      if (node) {
        subNodes.set(nodeId, node);
      }
    }

    const subEdges = graph.edges.filter(
      (e) => visited.has(e.source) && visited.has(e.target)
    );

    return {
      id: `${graph.id}-sub-${centerNodeId}`,
      name: `Subgraph_${centerNodeId}`,
      clusterId: graph.clusterId,
      nodes: subNodes,
      edges: subEdges,
      adjacencyList: this.buildAdjacencyList(subEdges),
      createdAt: new Date(),
      updatedAt: new Date(),
    };
  }

  /**
   * Update graph with new metrics
   */
  updateGraphMetrics(
    graph: NetworkGraph,
    newMetrics: Map<string, PerformanceMetrics>
  ): void {
    for (const [cellId, metrics] of newMetrics) {
      const node = graph.nodes.get(cellId);
      if (node) {
        node.metrics = metrics;

        // Update feature vector with new metrics
        node.features[7] = metrics.pmPuschSinr.mean;
        node.features[8] = metrics.pmPuschSinr.p5;
        node.features[9] = metrics.pmPuschSinr.p95;
        node.features[10] = metrics.pmUlPrbUtilization;
        node.features[11] = metrics.pmUlThroughput;
        node.features[12] = metrics.pmActiveUsers;
      }
    }

    graph.updatedAt = new Date();
    this.normalizeFeatures(graph);
  }
}

export default NetworkGraphBuilder;
