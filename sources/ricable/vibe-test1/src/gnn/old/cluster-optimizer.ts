/**
 * Cluster-Based Multi-Cell Optimizer for Ericsson Uplink Optimization
 *
 * This module addresses the root cause of single-cell optimization limitations:
 * when cells are optimized independently, they trade interference for SINR,
 * creating local optima where individual cells improve at the expense of neighbors.
 *
 * Key capabilities:
 * - Groups interfering cells into clusters for joint optimization
 * - Prioritizes clusters containing critical cells
 * - Implements neighbor sacrifice strategies for critical cell recovery
 * - Evaluates network-wide impact of parameter changes
 */

import type { CellKPISnapshot, NeighborRelation } from '../models/ran-kpi.js';
import {
  SurrogateGraphBuilder,
  IssueCellDetector,
  type SurrogateGraph,
  type PowerControlParams,
  type CellOptimizationResult,
  type SurrogateModelConfig,
  DEFAULT_SURROGATE_CONFIG,
} from './network-surrogate-model.js';
import { SelfLearningUplinkGNN, type LearningMetrics } from './self-learning-uplink-gnn.js';

// ============================================================================
// TYPES
// ============================================================================

/**
 * Represents a cluster of interfering cells
 */
export interface InterferenceCluster {
  id: string;
  cellIds: string[];
  issueCellIds: string[];
  criticalCellIds: string[]; // Cells with SINR < 0
  totalInterferenceCoupling: number;
  priority: number;
}

/**
 * Result of cluster optimization
 */
export interface ClusterOptimizationResult {
  clusterId: string;
  cellResults: CellOptimizationResult[];
  clusterSINRBefore: number;
  clusterSINRAfter: number;
  clusterImprovement: number;
  networkImpact: number;
  iterationsTotal: number;
  strategyUsed: string;
}

/**
 * Joint parameter configuration for a cluster
 */
export type JointConfiguration = Map<string, PowerControlParams>;

// ============================================================================
// CLUSTER IDENTIFIER
// ============================================================================

/**
 * Identifies clusters of interfering cells for joint optimization
 *
 * Clusters are formed based on:
 * - Adjacency matrix coupling strength (>0.3 threshold)
 * - Presence of issue/critical cells
 * - Network topology
 */
export class ClusterIdentifier {
  private config: SurrogateModelConfig;
  private issueDetector: IssueCellDetector;

  constructor(config: Partial<SurrogateModelConfig> = {}) {
    this.config = { ...DEFAULT_SURROGATE_CONFIG, ...config };
    this.issueDetector = new IssueCellDetector(this.config);
  }

  /**
   * Identify clusters of interfering cells
   */
  identifyClusters(
    graph: SurrogateGraph,
    issueCellIds: string[],
    cellSnapshots: Map<string, CellKPISnapshot>,
    maxClusterSize: number = 5
  ): InterferenceCluster[] {
    const clusters: InterferenceCluster[] = [];
    const visited = new Set<string>();

    // Identify critical cells (SINR < 0)
    const criticalCellIds = this.identifyCriticalCells(graph, issueCellIds, cellSnapshots);

    // Start with critical cells as seeds, then issue cells
    const seedOrder = [
      ...criticalCellIds,
      ...issueCellIds.filter(id => !criticalCellIds.includes(id)),
    ];

    for (const seedCell of seedOrder) {
      if (visited.has(seedCell)) continue;

      const cluster = this.buildCluster(
        seedCell,
        graph,
        issueCellIds,
        criticalCellIds,
        visited,
        maxClusterSize
      );

      if (cluster.cellIds.length > 0) {
        clusters.push(cluster);
        cluster.cellIds.forEach(id => visited.add(id));
      }
    }

    // Sort by priority: critical cells first, then by coupling strength
    return clusters.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Identify cells with critical SINR (< 0 dB)
   */
  private identifyCriticalCells(
    graph: SurrogateGraph,
    issueCellIds: string[],
    cellSnapshots: Map<string, CellKPISnapshot>
  ): string[] {
    const criticalCells: string[] = [];

    for (const cellId of issueCellIds) {
      const snapshot = cellSnapshots.get(cellId);
      if (snapshot && snapshot.radioQuality && snapshot.radioQuality.ulSinrAvg < 0) {
        criticalCells.push(cellId);
      }
    }

    return criticalCells;
  }

  /**
   * Build a cluster starting from a seed cell
   */
  private buildCluster(
    seedCell: string,
    graph: SurrogateGraph,
    issueCellIds: string[],
    criticalCellIds: string[],
    visited: Set<string>,
    maxSize: number
  ): InterferenceCluster {
    const cellIds: string[] = [seedCell];
    const queue: string[] = [seedCell];
    const couplingThreshold = 0.3;

    while (queue.length > 0 && cellIds.length < maxSize) {
      const current = queue.shift()!;
      const currentIdx = graph.nodeIds.indexOf(current);

      if (currentIdx < 0) continue;

      // Find strongly coupled neighbors
      for (let j = 0; j < graph.nodeIds.length; j++) {
        const neighborId = graph.nodeIds[j];
        const coupling = graph.adjacencyMatrix[currentIdx][j];

        // Include neighbors that are:
        // 1. Not already in cluster
        // 2. Not already visited
        // 3. Strongly coupled (> 0.3) OR very strongly coupled (> 0.5) even if healthy
        const isIssueCell = issueCellIds.includes(neighborId);
        const shouldInclude =
          coupling > couplingThreshold &&
          !cellIds.includes(neighborId) &&
          !visited.has(neighborId) &&
          (isIssueCell || coupling > 0.5);

        if (shouldInclude) {
          cellIds.push(neighborId);
          queue.push(neighborId);
        }
      }
    }

    // Categorize cells in cluster
    const issueCellsInCluster = cellIds.filter(id => issueCellIds.includes(id));
    const criticalCellsInCluster = cellIds.filter(id => criticalCellIds.includes(id));
    const totalCoupling = this.calculateTotalCoupling(cellIds, graph);

    // Priority: critical cells weighted 20x, issue cells 10x, plus coupling
    const priority =
      criticalCellsInCluster.length * 20 +
      issueCellsInCluster.length * 10 +
      totalCoupling;

    return {
      id: `cluster_${seedCell}_${Date.now()}`,
      cellIds,
      issueCellIds: issueCellsInCluster,
      criticalCellIds: criticalCellsInCluster,
      totalInterferenceCoupling: totalCoupling,
      priority,
    };
  }

  /**
   * Calculate total interference coupling within a cluster
   */
  private calculateTotalCoupling(cellIds: string[], graph: SurrogateGraph): number {
    let total = 0;
    for (const cellId of cellIds) {
      const idx = graph.nodeIds.indexOf(cellId);
      if (idx < 0) continue;

      for (const otherId of cellIds) {
        if (cellId !== otherId) {
          const otherIdx = graph.nodeIds.indexOf(otherId);
          if (otherIdx >= 0) {
            total += graph.adjacencyMatrix[idx][otherIdx];
          }
        }
      }
    }
    return total / 2; // Each edge counted twice
  }
}

// ============================================================================
// CLUSTER OPTIMIZER
// ============================================================================

/**
 * Optimizes cells within a cluster jointly
 *
 * Key strategies:
 * 1. All balanced: Everyone uses nominal settings
 * 2. All conservative: Everyone uses low-interference settings
 * 3. Critical boosted: Critical cells get more power, others conservative
 * 4. Neighbor sacrifice: Healthy neighbors reduce power to help critical cells
 * 5. Gradient strategy: Power decreases outward from critical cells
 */
export class ClusterOptimizer {
  private config: SurrogateModelConfig;
  private gnn: SelfLearningUplinkGNN;
  private graphBuilder: SurrogateGraphBuilder;

  constructor(gnn: SelfLearningUplinkGNN, config: Partial<SurrogateModelConfig> = {}) {
    this.config = { ...DEFAULT_SURROGATE_CONFIG, ...config };
    this.gnn = gnn;
    this.graphBuilder = new SurrogateGraphBuilder(this.config);
  }

  /**
   * Optimize all cells in a cluster jointly
   */
  optimizeCluster(
    cluster: InterferenceCluster,
    graph: SurrogateGraph,
    cellSnapshots: Map<string, CellKPISnapshot>
  ): ClusterOptimizationResult {
    // Get baseline predictions
    const baselinePred = this.gnn.predict(graph);
    const clusterIndices = cluster.cellIds
      .map(id => graph.nodeIds.indexOf(id))
      .filter(idx => idx >= 0);

    // Calculate cluster baseline SINR
    const clusterSINRBefore =
      clusterIndices.length > 0
        ? clusterIndices.map(idx => baselinePred.sinr[idx]).reduce((a, b) => a + b, 0) /
          clusterIndices.length
        : 0;

    // Generate joint configuration candidates
    const jointCandidates = this.generateJointCandidates(cluster, graph);

    // Evaluate each candidate
    let bestJointConfig: JointConfiguration = new Map();
    let bestClusterSINR = clusterSINRBefore;
    let bestNetworkImpact = 0;
    let bestStrategyName = 'none';
    let iterations = 0;

    for (const { config: jointCandidate, name: strategyName } of jointCandidates) {
      iterations++;

      // Update graph with joint configuration
      const updatedGraph = this.graphBuilder.updateGraphParams(
        graph,
        jointCandidate,
        cellSnapshots
      );
      const pred = this.gnn.predict(updatedGraph);

      // Calculate cluster SINR with this configuration
      const candidateClusterSINR =
        clusterIndices.length > 0
          ? clusterIndices.map(idx => pred.sinr[idx]).reduce((a, b) => a + b, 0) /
            clusterIndices.length
          : 0;

      // Calculate impact on cells outside the cluster (network impact)
      const outsideIndices = graph.nodeIds
        .map((_, idx) => idx)
        .filter(idx => !clusterIndices.includes(idx));

      const baselineOutsideSINR =
        outsideIndices.length > 0
          ? outsideIndices.map(idx => baselinePred.sinr[idx]).reduce((a, b) => a + b, 0) /
            outsideIndices.length
          : 0;
      const candidateOutsideSINR =
        outsideIndices.length > 0
          ? outsideIndices.map(idx => pred.sinr[idx]).reduce((a, b) => a + b, 0) /
            outsideIndices.length
          : 0;
      const networkImpact = candidateOutsideSINR - baselineOutsideSINR;

      // Combined fitness: cluster improvement + network impact
      // Weight network impact less (0.3) since cluster optimization is the priority
      const fitness = candidateClusterSINR - clusterSINRBefore + networkImpact * 0.3;
      const currentFitness = bestClusterSINR - clusterSINRBefore + bestNetworkImpact * 0.3;

      if (fitness > currentFitness) {
        bestJointConfig = jointCandidate;
        bestClusterSINR = candidateClusterSINR;
        bestNetworkImpact = networkImpact;
        bestStrategyName = strategyName;
      }
    }

    // Build individual cell results
    const cellResults = this.buildCellResults(
      cluster,
      graph,
      bestJointConfig,
      baselinePred,
      cellSnapshots
    );

    return {
      clusterId: cluster.id,
      cellResults,
      clusterSINRBefore,
      clusterSINRAfter: bestClusterSINR,
      clusterImprovement: bestClusterSINR - clusterSINRBefore,
      networkImpact: bestNetworkImpact,
      iterationsTotal: iterations,
      strategyUsed: bestStrategyName,
    };
  }

  /**
   * Generate joint configuration candidates (all strategies)
   */
  private generateJointCandidates(
    cluster: InterferenceCluster,
    graph: SurrogateGraph
  ): Array<{ config: JointConfiguration; name: string }> {
    const candidates: Array<{ config: JointConfiguration; name: string }> = [];

    // Strategy 1: All balanced (nominal settings)
    candidates.push({
      name: 'all_balanced',
      config: new Map(cluster.cellIds.map(id => [id, { p0: -100, alpha: 0.8 }])),
    });

    // Strategy 2: All conservative (low interference)
    candidates.push({
      name: 'all_conservative',
      config: new Map(cluster.cellIds.map(id => [id, { p0: -105, alpha: 0.7 }])),
    });

    // Strategy 3: Issue cells boosted, others conservative
    candidates.push({
      name: 'issue_boosted',
      config: new Map(
        cluster.cellIds.map(id => [
          id,
          cluster.issueCellIds.includes(id)
            ? { p0: -96, alpha: 0.9 }
            : { p0: -102, alpha: 0.7 },
        ])
      ),
    });

    // Strategy 4: Critical cells boosted, others conservative
    if (cluster.criticalCellIds.length > 0) {
      candidates.push({
        name: 'critical_boosted',
        config: new Map(
          cluster.cellIds.map(id => [
            id,
            cluster.criticalCellIds.includes(id)
              ? { p0: -90, alpha: 1.0 }
              : { p0: -105, alpha: 0.6 },
          ])
        ),
      });
    }

    // Strategy 5: Neighbors SACRIFICE for critical cells (key strategy)
    if (cluster.criticalCellIds.length > 0) {
      candidates.push({
        name: 'neighbor_sacrifice',
        config: new Map(
          cluster.cellIds.map(id => [
            id,
            cluster.criticalCellIds.includes(id)
              ? { p0: -88, alpha: 1.0 } // Maximum power for critical
              : { p0: -110, alpha: 0.5 }, // Minimum power for neighbors
          ])
        ),
      });
    }

    // Strategy 6: Gradual power reduction from critical outward
    if (cluster.criticalCellIds.length > 0) {
      candidates.push({
        name: 'gradient_from_critical',
        config: this.generateGradientStrategy(cluster, graph),
      });
    }

    // Strategy 7: Ultra-conservative (everyone very low)
    candidates.push({
      name: 'ultra_conservative',
      config: new Map(cluster.cellIds.map(id => [id, { p0: -108, alpha: 0.6 }])),
    });

    // Strategy 8: Moderate boost for all issue cells
    candidates.push({
      name: 'moderate_all_issue',
      config: new Map(
        cluster.cellIds.map(id => [
          id,
          cluster.issueCellIds.includes(id)
            ? { p0: -98, alpha: 0.85 }
            : { p0: -100, alpha: 0.8 },
        ])
      ),
    });

    return candidates;
  }

  /**
   * Generate gradient strategy: power decreases as distance from critical cells increases
   */
  private generateGradientStrategy(
    cluster: InterferenceCluster,
    graph: SurrogateGraph
  ): JointConfiguration {
    const config = new Map<string, PowerControlParams>();

    // Calculate "distance" from critical cells using adjacency
    const distanceFromCritical = new Map<string, number>();

    for (const cellId of cluster.cellIds) {
      if (cluster.criticalCellIds.includes(cellId)) {
        distanceFromCritical.set(cellId, 0);
      } else {
        // Find minimum coupling to any critical cell
        const cellIdx = graph.nodeIds.indexOf(cellId);
        let maxCoupling = 0;

        for (const criticalId of cluster.criticalCellIds) {
          const criticalIdx = graph.nodeIds.indexOf(criticalId);
          if (cellIdx >= 0 && criticalIdx >= 0) {
            maxCoupling = Math.max(maxCoupling, graph.adjacencyMatrix[cellIdx][criticalIdx]);
          }
        }

        // Distance inversely proportional to coupling (high coupling = close)
        distanceFromCritical.set(cellId, maxCoupling > 0 ? 1 / maxCoupling : 10);
      }
    }

    // Assign parameters based on distance
    for (const cellId of cluster.cellIds) {
      const distance = distanceFromCritical.get(cellId) || 10;

      if (distance === 0) {
        // Critical cell: maximum power
        config.set(cellId, { p0: -88, alpha: 1.0 });
      } else if (distance < 2) {
        // Very close neighbor: sacrifice significantly
        config.set(cellId, { p0: -108, alpha: 0.6 });
      } else if (distance < 5) {
        // Medium distance: moderate reduction
        config.set(cellId, { p0: -104, alpha: 0.7 });
      } else {
        // Far: minimal change
        config.set(cellId, { p0: -100, alpha: 0.8 });
      }
    }

    return config;
  }

  /**
   * Build individual cell results from joint optimization
   */
  private buildCellResults(
    cluster: InterferenceCluster,
    graph: SurrogateGraph,
    jointConfig: JointConfiguration,
    baselinePred: { sinr: number[]; iot: number[] },
    cellSnapshots: Map<string, CellKPISnapshot>
  ): CellOptimizationResult[] {
    // Get optimized predictions
    const optimizedGraph = this.graphBuilder.updateGraphParams(
      graph,
      jointConfig,
      cellSnapshots
    );
    const optimizedPred = this.gnn.predict(optimizedGraph);

    return cluster.cellIds.map(cellId => {
      const cellIdx = graph.nodeIds.indexOf(cellId);
      const originalParams = graph.powerParams.get(cellId)!;
      const optimizedParams = jointConfig.get(cellId) || originalParams;

      const originalSINR = cellIdx >= 0 ? baselinePred.sinr[cellIdx] : 0;
      const optimizedSINR = cellIdx >= 0 ? optimizedPred.sinr[cellIdx] : 0;

      return {
        cellId,
        originalParams,
        optimizedParams,
        originalSINR,
        optimizedSINR,
        sinrImprovement: optimizedSINR - originalSINR,
        neighborImpact: 0, // Calculated at cluster level
        iterations: 1, // Joint optimization
        confidence: 0.75, // Cluster optimization has moderate confidence
        statusTransition: {
          before: this.getStatus(originalSINR),
          after: this.getStatus(optimizedSINR),
          scoreBefore: Math.round(originalSINR + 10),
          scoreAfter: Math.round(optimizedSINR + 10),
        },
      };
    });
  }

  /**
   * Get cell status based on SINR
   */
  private getStatus(sinr: number): 'healthy' | 'warning' | 'issue' | 'critical' {
    if (sinr < 0) return 'critical';
    if (sinr < 5) return 'issue';
    if (sinr < 10) return 'warning';
    return 'healthy';
  }
}

// ============================================================================
// NETWORK CLUSTER OPTIMIZER
// ============================================================================

/**
 * High-level interface for cluster-based network optimization
 *
 * Combines ClusterIdentifier and ClusterOptimizer to provide
 * a complete solution for multi-cell optimization.
 */
export class NetworkClusterOptimizer {
  private config: SurrogateModelConfig;
  private gnn: SelfLearningUplinkGNN;
  private graphBuilder: SurrogateGraphBuilder;
  private issueDetector: IssueCellDetector;
  private clusterIdentifier: ClusterIdentifier;
  private clusterOptimizer: ClusterOptimizer;

  constructor(gnn: SelfLearningUplinkGNN, config: Partial<SurrogateModelConfig> = {}) {
    this.config = { ...DEFAULT_SURROGATE_CONFIG, ...config };
    this.gnn = gnn;
    this.graphBuilder = new SurrogateGraphBuilder(this.config);
    this.issueDetector = new IssueCellDetector(this.config);
    this.clusterIdentifier = new ClusterIdentifier(this.config);
    this.clusterOptimizer = new ClusterOptimizer(gnn, this.config);
  }

  /**
   * Optimize the entire network using cluster-based approach
   */
  optimizeNetwork(
    cellSnapshots: Map<string, CellKPISnapshot>,
    neighborRelations: NeighborRelation[]
  ): {
    timestamp: Date;
    clusters: InterferenceCluster[];
    clusterResults: ClusterOptimizationResult[];
    cellResults: CellOptimizationResult[];
    metrics: {
      totalClusters: number;
      criticalClustersFixed: number;
      avgClusterImprovement: number;
      networkwideImpact: number;
    };
    recommendations: string[];
  } {
    const timestamp = new Date();

    // Build network graph
    const graph = this.graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    // Detect issue cells
    const issueCells = this.issueDetector.detectIssueCells(cellSnapshots, neighborRelations);
    const issueCellIds = issueCells.map(ic => ic.cellId);

    // Identify clusters
    const clusters = this.clusterIdentifier.identifyClusters(
      graph,
      issueCellIds,
      cellSnapshots,
      5 // Max cluster size
    );

    // Optimize each cluster
    const clusterResults: ClusterOptimizationResult[] = [];
    const allCellResults: CellOptimizationResult[] = [];

    for (const cluster of clusters) {
      const result = this.clusterOptimizer.optimizeCluster(cluster, graph, cellSnapshots);
      clusterResults.push(result);
      allCellResults.push(...result.cellResults);
    }

    // Calculate metrics
    const criticalClustersFixed = clusterResults.filter(
      r => r.clusterImprovement > 1.0 && r.cellResults.some(c => c.statusTransition.before === 'critical' && c.statusTransition.after !== 'critical')
    ).length;

    const avgClusterImprovement =
      clusterResults.length > 0
        ? clusterResults.reduce((sum, r) => sum + r.clusterImprovement, 0) / clusterResults.length
        : 0;

    const networkwideImpact =
      clusterResults.length > 0
        ? clusterResults.reduce((sum, r) => sum + r.networkImpact, 0) / clusterResults.length
        : 0;

    // Generate recommendations
    const recommendations = this.generateRecommendations(clusters, clusterResults);

    return {
      timestamp,
      clusters,
      clusterResults,
      cellResults: allCellResults,
      metrics: {
        totalClusters: clusters.length,
        criticalClustersFixed,
        avgClusterImprovement,
        networkwideImpact,
      },
      recommendations,
    };
  }

  /**
   * Generate recommendations from cluster optimization results
   */
  private generateRecommendations(
    clusters: InterferenceCluster[],
    results: ClusterOptimizationResult[]
  ): string[] {
    const recommendations: string[] = [];

    recommendations.push('[Cluster-Based Multi-Cell Optimization]');
    recommendations.push(`Identified ${clusters.length} interference clusters`);
    recommendations.push('');

    // Count critical cells
    const totalCriticalCells = clusters.reduce(
      (sum, c) => sum + c.criticalCellIds.length,
      0
    );
    if (totalCriticalCells > 0) {
      recommendations.push(`PRIORITY: ${totalCriticalCells} critical cells identified`);
    }

    // Best performing clusters
    const sortedResults = [...results].sort((a, b) => b.clusterImprovement - a.clusterImprovement);
    const topN = Math.min(3, sortedResults.length);

    if (topN > 0) {
      recommendations.push(`\nTop ${topN} cluster improvements:`);
      for (let i = 0; i < topN; i++) {
        const r = sortedResults[i];
        recommendations.push(
          `  ${i + 1}. ${r.clusterId.split('_')[1]}: +${r.clusterImprovement.toFixed(2)} dB ` +
          `(${r.strategyUsed}, ${r.cellResults.length} cells)`
        );
      }
    }

    // Neighbor sacrifice strategy success
    const sacrificeResults = results.filter(r => r.strategyUsed === 'neighbor_sacrifice');
    if (sacrificeResults.length > 0) {
      const avgSacrificeImprovement =
        sacrificeResults.reduce((sum, r) => sum + r.clusterImprovement, 0) /
        sacrificeResults.length;
      recommendations.push(
        `\nNeighbor sacrifice strategy: ${sacrificeResults.length} clusters, ` +
        `+${avgSacrificeImprovement.toFixed(2)} dB avg improvement`
      );
    }

    recommendations.push('\nDeployment notes:');
    recommendations.push('- Apply cluster changes atomically (all cells in cluster together)');
    recommendations.push('- Monitor neighbor cells for unexpected degradation');
    recommendations.push('- Critical cells should show improvement within 1 hour');

    return recommendations;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  ClusterIdentifier,
  ClusterOptimizer,
  NetworkClusterOptimizer,
};
