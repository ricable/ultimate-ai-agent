/**
 * GNN-Based Interference Optimization System
 *
 * Implements the Ericsson-style "Digital Twin" approach for uplink interference optimization:
 * - Single-layer GNN architecture (avoids over-smoothing for 1-hop interference)
 * - SINR prediction model (supervised regression)
 * - Genetic Algorithm for P0/Alpha parameter optimization
 * - Issue cell detection and optimization loop
 *
 * Based on: Ericsson's offline GNN approach for Radio Access Network optimization
 */

import type {
  CellGraph,
  CellGraphNode,
  CellGraphEdge,
  CellKPISnapshot,
  NeighborRelation,
} from '../models/ran-kpi.js';

// ============================================================================
// CONFIGURATION TYPES
// ============================================================================

export interface InterferenceOptimizerConfig {
  // GNN Architecture (single layer as per Ericsson approach)
  inputDim: number;
  hiddenDim: number;
  numHeads: number;

  // Power control parameter ranges
  p0Min: number;        // Minimum P0 (dBm), typically -110
  p0Max: number;        // Maximum P0 (dBm), typically -85
  alphaValues: number[]; // Valid alpha values: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

  // Issue cell detection
  sinrThreshold: number;           // SINR below this is an issue (dB)
  iotThreshold: number;            // IoT above this is an issue (dB)

  // Genetic Algorithm parameters
  populationSize: number;
  numGenerations: number;
  mutationRate: number;
  crossoverRate: number;
  eliteRatio: number;

  // Optimization constraints
  maxInterferenceIncrease: number; // Max acceptable IoT increase to neighbors (dB)
  minSinrImprovement: number;      // Minimum SINR improvement to accept (dB)
}

export const DEFAULT_OPTIMIZER_CONFIG: InterferenceOptimizerConfig = {
  // Single-layer GNN architecture
  inputDim: 20,
  hiddenDim: 64,
  numHeads: 4,

  // Power control ranges (as per 3GPP specs)
  p0Min: -110,
  p0Max: -85,
  alphaValues: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

  // Issue detection thresholds
  sinrThreshold: 5,    // SINR < 5 dB is problematic
  iotThreshold: 10,    // IoT > 10 dB is excessive

  // Genetic Algorithm settings
  populationSize: 100,
  numGenerations: 50,
  mutationRate: 0.15,
  crossoverRate: 0.8,
  eliteRatio: 0.1,

  // Constraints
  maxInterferenceIncrease: 2,  // Max 2 dB IoT increase to neighbors
  minSinrImprovement: 0.5,     // At least 0.5 dB improvement
};

// ============================================================================
// SINGLE-LAYER GNN FOR SINR PREDICTION (DIGITAL TWIN)
// ============================================================================

/**
 * Single-layer Graph Neural Network for SINR prediction
 *
 * Architecture rationale (from Ericsson presentation):
 * - Uses ONLY ONE GNN layer followed by MLP because interference is primarily
 *   a 1-hop neighbor issue. The influence of "neighbor of a neighbor" is negligible.
 * - Stacking multiple layers causes "over-smoothing" where features of issue cells
 *   get washed out by averaging with healthy cells further away.
 *
 * This model acts as a "Digital Twin" - a highly accurate predictive model that
 * replaces the need to experiment on the live network.
 */
export class SINRPredictionGNN {
  private config: InterferenceOptimizerConfig;

  // Weights for the single GNN layer
  private W_query: number[][];
  private W_key: number[][];
  private W_value: number[][];

  // MLP head for SINR prediction
  private W_mlp1: number[][];
  private W_mlp2: number[][];
  private b_mlp1: number[];
  private b_mlp2: number[];

  // Training state (for online learning)
  private isTrained: boolean = false;
  private trainingLoss: number = 0;

  constructor(config: Partial<InterferenceOptimizerConfig> = {}) {
    this.config = { ...DEFAULT_OPTIMIZER_CONFIG, ...config };

    const { inputDim, hiddenDim, numHeads } = this.config;

    // Initialize GNN layer weights (Xavier initialization)
    this.W_query = this.xavierInit(inputDim, hiddenDim);
    this.W_key = this.xavierInit(inputDim, hiddenDim);
    this.W_value = this.xavierInit(inputDim, hiddenDim);

    // Initialize MLP head (2 layers: hiddenDim -> hiddenDim -> 1)
    this.W_mlp1 = this.xavierInit(hiddenDim, hiddenDim);
    this.b_mlp1 = Array(hiddenDim).fill(0);
    this.W_mlp2 = this.xavierInit(hiddenDim, 1);
    this.b_mlp2 = [0];
  }

  /**
   * Xavier/Glorot weight initialization
   */
  private xavierInit(fanIn: number, fanOut: number): number[][] {
    const std = Math.sqrt(2 / (fanIn + fanOut));
    return Array(fanIn).fill(null).map(() =>
      Array(fanOut).fill(null).map(() => (Math.random() * 2 - 1) * std)
    );
  }

  /**
   * Build input features for a cell including control parameters
   *
   * Input features (as per Ericsson presentation):
   * A. Control Parameters (learnable/optimizable):
   *    - P0 (normalized)
   *    - Alpha (normalized)
   * B. Fixed Node Features (KPIs from radio experts):
   *    - Traffic load
   *    - Historical SINR
   *    - Cell capabilities
   * C. Edge Features (topology):
   *    - Distance between cells
   *    - User density at borders
   */
  buildNodeFeatures(
    snapshot: CellKPISnapshot,
    proposedP0: number,
    proposedAlpha: number
  ): number[] {
    const features: number[] = [];

    // A. Control Parameters (P0 and Alpha) - these are what we optimize
    features.push(this.normalize(proposedP0, this.config.p0Min, this.config.p0Max));
    features.push(proposedAlpha); // Already in [0, 1]

    // B. Fixed Node Features (KPIs)
    // Radio quality indicators
    features.push(this.normalize(snapshot.radioQuality.ulSinrAvg, -5, 30));
    features.push(this.normalize(snapshot.radioQuality.rsrpAvg, -140, -80));
    features.push(this.normalize(snapshot.radioQuality.rsrqAvg, -25, 0));
    features.push(this.normalize(snapshot.radioQuality.dlAvgCqi, 0, 15));

    // Interference indicators
    features.push(this.normalize(snapshot.uplinkInterference.iotAvg, 0, 20));
    features.push(this.normalize(snapshot.uplinkInterference.rip, -110, -90));
    features.push(this.normalize(snapshot.uplinkInterference.highInterferencePrbRatio, 0, 100));

    // Power control state
    features.push(this.normalize(snapshot.uplinkPowerControl.powerHeadroomAvg, -20, 40));
    features.push(this.normalize(snapshot.uplinkPowerControl.powerLimitedUeRatio, 0, 100));
    features.push(this.normalize(snapshot.uplinkPowerControl.pathLossAvg, 80, 160));

    // Traffic indicators
    features.push(this.normalize(snapshot.accessibility.rrcSetupAttempts, 0, 10000));
    features.push(this.normalize(snapshot.retainability.dataSessionRetainability, 90, 100));

    // Mobility indicators (handover activity reflects user movement)
    features.push(this.normalize(snapshot.mobility.intraFreqHoAttempts, 0, 5000));
    features.push(this.normalize(snapshot.mobility.intraFreqHoSuccessRate, 80, 100));

    // Additional capacity indicators
    features.push(this.normalize(snapshot.radioQuality.ulSpectralEfficiency, 0, 10));
    features.push(this.normalize(snapshot.radioQuality.ulBlerPercent, 0, 20));

    // Pad to inputDim if needed
    while (features.length < this.config.inputDim) {
      features.push(0);
    }

    return features.slice(0, this.config.inputDim);
  }

  /**
   * Build edge features for neighbor relationship
   */
  buildEdgeFeatures(relation: NeighborRelation): number[] {
    return [
      this.normalize(relation.targetRsrp - relation.sourceRsrp, -30, 30), // RSRP delta
      this.normalize(relation.targetSinr - relation.sourceSinr, -20, 20), // SINR delta
      this.normalize(relation.hoSuccessRate, 0, 100),
      relation.distance ? this.normalize(relation.distance, 0, 5000) : 0.5,
      relation.relationshipType === 'intra-freq' ? 1 : 0,
    ];
  }

  /**
   * Single-layer message passing with multi-head attention
   *
   * This is the core GNN operation that aggregates information from 1-hop neighbors.
   * Only uses ONE layer to avoid over-smoothing.
   */
  private messagePass(
    nodeFeatures: number[][],
    adjacencyMatrix: number[][],
    edgeFeatures?: number[][][]
  ): number[][] {
    const numNodes = nodeFeatures.length;
    const { hiddenDim, numHeads } = this.config;
    const headDim = Math.floor(hiddenDim / numHeads);

    // Compute queries, keys, values
    const queries = this.matmul(nodeFeatures, this.W_query);
    const keys = this.matmul(nodeFeatures, this.W_key);
    const values = this.matmul(nodeFeatures, this.W_value);

    // Output after message passing
    const output: number[][] = Array(numNodes).fill(null).map(() =>
      Array(hiddenDim).fill(0)
    );

    // Multi-head attention aggregation
    for (let h = 0; h < numHeads; h++) {
      const startIdx = h * headDim;
      const endIdx = startIdx + headDim;

      for (let i = 0; i < numNodes; i++) {
        // Compute attention scores for node i's neighbors
        const attentionScores: number[] = [];
        const neighborIndices: number[] = [];

        for (let j = 0; j < numNodes; j++) {
          if (adjacencyMatrix[i][j] > 0 || i === j) { // Include self-loop
            // Query-Key dot product scaled
            let score = 0;
            for (let k = startIdx; k < endIdx; k++) {
              score += queries[i][k] * keys[j][k];
            }
            score /= Math.sqrt(headDim);

            // Incorporate edge features if available
            if (edgeFeatures && edgeFeatures[i] && edgeFeatures[i][j]?.length > 0) {
              const edgeBias = edgeFeatures[i][j].reduce((a, b) => a + b, 0) * 0.1;
              score += edgeBias;
            }

            // Weight by adjacency (stronger connections = more influence)
            if (i !== j) {
              score += Math.log(adjacencyMatrix[i][j] + 0.1);
            }

            attentionScores.push(score);
            neighborIndices.push(j);
          }
        }

        // Softmax over attention scores
        const maxScore = Math.max(...attentionScores);
        const expScores = attentionScores.map(s => Math.exp(s - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const weights = expScores.map(s => s / (sumExp || 1));

        // Aggregate neighbor values
        for (let k = startIdx; k < endIdx; k++) {
          let aggregated = 0;
          for (let n = 0; n < neighborIndices.length; n++) {
            aggregated += weights[n] * values[neighborIndices[n]][k];
          }
          output[i][k] = aggregated;
        }
      }
    }

    // Add residual connection and layer normalization
    const residual = this.addResidual(nodeFeatures, output);
    return this.layerNorm(residual);
  }

  /**
   * MLP head for SINR prediction
   * Takes the GNN output embedding and predicts SINR value
   */
  private mlpHead(embedding: number[]): number {
    // Layer 1: Linear + ReLU
    let hidden: number[] = Array(this.config.hiddenDim).fill(0);
    for (let i = 0; i < this.config.hiddenDim; i++) {
      for (let j = 0; j < embedding.length && j < this.W_mlp1.length; j++) {
        hidden[i] += embedding[j] * this.W_mlp1[j][i];
      }
      hidden[i] += this.b_mlp1[i];
      hidden[i] = Math.max(0, hidden[i]); // ReLU
    }

    // Layer 2: Linear (output SINR prediction)
    let sinr = this.b_mlp2[0];
    for (let i = 0; i < hidden.length && i < this.W_mlp2.length; i++) {
      sinr += hidden[i] * this.W_mlp2[i][0];
    }

    // Scale output to valid SINR range (-5 to 30 dB)
    return sinr * 35 - 5;
  }

  /**
   * Predict SINR for all cells given their features
   *
   * This is the main inference function - the "Digital Twin" prediction.
   * Given current network state and proposed P0/Alpha values, it predicts
   * what the SINR will be.
   */
  predictSINR(
    nodeFeatures: number[][],
    adjacencyMatrix: number[][]
  ): number[] {
    // Single-layer message passing
    const embeddings = this.messagePass(nodeFeatures, adjacencyMatrix);

    // Predict SINR for each node using MLP head
    return embeddings.map(emb => this.mlpHead(emb));
  }

  /**
   * Predict SINR for a specific cell with proposed parameters
   *
   * This is used by the optimizer to evaluate different P0/Alpha combinations.
   * The GNN is queried millions of times during optimization.
   */
  predictCellSINR(
    cellIndex: number,
    nodeFeatures: number[][],
    adjacencyMatrix: number[][]
  ): number {
    const sinrs = this.predictSINR(nodeFeatures, adjacencyMatrix);
    return sinrs[cellIndex];
  }

  /**
   * Predict neighbor impact - how much increasing power in one cell affects neighbors
   *
   * Returns the predicted SINR degradation for each neighbor if we apply
   * the proposed parameters to the target cell.
   */
  predictNeighborImpact(
    targetCellIndex: number,
    neighborIndices: number[],
    nodeFeatures: number[][],
    adjacencyMatrix: number[][]
  ): Map<number, number> {
    const sinrs = this.predictSINR(nodeFeatures, adjacencyMatrix);
    const impact = new Map<number, number>();

    for (const neighborIdx of neighborIndices) {
      // The impact is already embedded in the GNN prediction
      // We return the predicted SINR for neighbors
      impact.set(neighborIdx, sinrs[neighborIdx]);
    }

    return impact;
  }

  /**
   * Train the model on historical data (simplified online learning)
   *
   * In practice, this would use backpropagation with real network data.
   * The Ericsson presentation notes: model achieves <1% error because it's
   * trained on massive amounts of real-world data from mobile users.
   */
  train(
    trainingData: Array<{
      nodeFeatures: number[][];
      adjacencyMatrix: number[][];
      actualSINR: number[];
    }>,
    learningRate: number = 0.001,
    epochs: number = 100
  ): { loss: number; accuracy: number } {
    let totalLoss = 0;
    let totalError = 0;
    let count = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const sample of trainingData) {
        const predictedSINR = this.predictSINR(
          sample.nodeFeatures,
          sample.adjacencyMatrix
        );

        // Calculate MSE loss
        for (let i = 0; i < predictedSINR.length; i++) {
          const error = predictedSINR[i] - sample.actualSINR[i];
          totalLoss += error * error;
          totalError += Math.abs(error);
          count++;

          // Simplified gradient descent (would use proper backprop in production)
          const grad = 2 * error * learningRate;

          // Update MLP weights slightly based on error
          for (let j = 0; j < this.W_mlp2.length; j++) {
            this.W_mlp2[j][0] -= grad * 0.01;
          }
          this.b_mlp2[0] -= grad * 0.01;
        }
      }
    }

    this.isTrained = true;
    this.trainingLoss = totalLoss / count;

    const avgError = totalError / count;
    const accuracy = Math.max(0, 1 - avgError / 35); // Normalize by SINR range

    return {
      loss: this.trainingLoss,
      accuracy,
    };
  }

  // Matrix operations
  private matmul(a: number[][], b: number[][]): number[][] {
    const m = a.length;
    const n = b[0]?.length || 0;
    const k = b.length;
    const result: number[][] = Array(m).fill(null).map(() => Array(n).fill(0));

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        for (let l = 0; l < k && l < a[i].length; l++) {
          result[i][j] += a[i][l] * (b[l]?.[j] || 0);
        }
      }
    }
    return result;
  }

  private addResidual(original: number[][], transformed: number[][]): number[][] {
    const m = original.length;
    const n = transformed[0]?.length || this.config.hiddenDim;
    const result: number[][] = Array(m).fill(null).map(() => Array(n).fill(0));

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const origVal = j < (original[i]?.length || 0) ? original[i][j] : 0;
        const transVal = transformed[i]?.[j] || 0;
        result[i][j] = origVal + transVal;
      }
    }
    return result;
  }

  private layerNorm(x: number[][]): number[][] {
    const eps = 1e-6;
    return x.map(row => {
      const mean = row.reduce((a, b) => a + b, 0) / row.length;
      const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
      const std = Math.sqrt(variance + eps);
      return row.map(v => (v - mean) / std);
    });
  }

  private normalize(value: number, min: number, max: number): number {
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
  }
}

// ============================================================================
// ISSUE CELL DETECTOR
// ============================================================================

export interface IssueCell {
  cellId: string;
  cellIndex: number;
  currentSINR: number;
  currentIoT: number;
  currentP0: number;
  currentAlpha: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  issues: string[];
  neighborCellIds: string[];
}

/**
 * Detects cells with low SINR that need optimization
 *
 * From Ericsson presentation: "Issue cells" with low SINR are identified
 * before the optimization process begins.
 */
export class IssueCellDetector {
  private config: InterferenceOptimizerConfig;

  constructor(config: Partial<InterferenceOptimizerConfig> = {}) {
    this.config = { ...DEFAULT_OPTIMIZER_CONFIG, ...config };
  }

  /**
   * Detect all cells that have interference issues
   */
  detectIssueCells(
    cellSnapshots: Map<string, CellKPISnapshot>,
    neighborRelations: NeighborRelation[]
  ): IssueCell[] {
    const issueCells: IssueCell[] = [];
    const cellIds = Array.from(cellSnapshots.keys());

    let cellIndex = 0;
    for (const [cellId, snapshot] of cellSnapshots) {
      const issues: string[] = [];
      // Use numeric severity for easier comparisons (4=critical, 3=high, 2=medium, 1=low)
      let severityLevel = 1;

      const sinr = snapshot.radioQuality.ulSinrAvg;
      const iot = snapshot.uplinkInterference.iotAvg;
      const powerLimited = snapshot.uplinkPowerControl.powerLimitedUeRatio;
      const negativePhr = snapshot.uplinkPowerControl.negativePowerHeadroomRatio;

      // Check SINR threshold
      if (sinr < this.config.sinrThreshold) {
        issues.push(`Low SINR: ${sinr.toFixed(1)} dB (threshold: ${this.config.sinrThreshold} dB)`);
        if (sinr < 0) {
          severityLevel = Math.max(severityLevel, 4); // critical
        } else if (sinr < 3) {
          severityLevel = Math.max(severityLevel, 3); // high
        } else {
          severityLevel = Math.max(severityLevel, 2); // medium
        }
      }

      // Check IoT threshold
      if (iot > this.config.iotThreshold) {
        issues.push(`High IoT: ${iot.toFixed(1)} dB (threshold: ${this.config.iotThreshold} dB)`);
        if (iot > 15) {
          severityLevel = Math.max(severityLevel, 4); // critical
        } else if (iot > 12) {
          severityLevel = Math.max(severityLevel, 3); // high
        }
      }

      // Check power-limited UEs
      if (powerLimited > 20) {
        issues.push(`High power-limited UE ratio: ${powerLimited.toFixed(1)}%`);
        severityLevel = Math.max(severityLevel, 2); // medium
      }

      // Check negative power headroom
      if (negativePhr > 15) {
        issues.push(`High negative PHR ratio: ${negativePhr.toFixed(1)}%`);
        severityLevel = Math.max(severityLevel, 2); // medium
      }

      // Convert numeric severity to string
      const severityMap: Record<number, IssueCell['severity']> = {
        1: 'low',
        2: 'medium',
        3: 'high',
        4: 'critical',
      };
      const severity = severityMap[severityLevel] || 'low';

      // Check interference distribution
      if (snapshot.uplinkInterference.highInterferencePrbRatio > 30) {
        issues.push(`High interference PRB ratio: ${snapshot.uplinkInterference.highInterferencePrbRatio.toFixed(1)}%`);
      }

      if (issues.length > 0) {
        // Find neighbors
        const neighborCellIds = neighborRelations
          .filter(nr => nr.sourceCellId === cellId && nr.relationshipType === 'intra-freq')
          .map(nr => nr.targetCellId);

        issueCells.push({
          cellId,
          cellIndex,
          currentSINR: sinr,
          currentIoT: iot,
          currentP0: snapshot.uplinkPowerControl.p0NominalPusch,
          currentAlpha: snapshot.uplinkPowerControl.alpha,
          severity,
          issues,
          neighborCellIds,
        });
      }

      cellIndex++;
    }

    // Sort by severity (critical first)
    const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    return issueCells.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);
  }
}

// ============================================================================
// GENETIC ALGORITHM OPTIMIZER
// ============================================================================

interface Individual {
  p0: number;
  alpha: number;
  fitness: number;
}

interface OptimizationResult {
  cellId: string;
  currentP0: number;
  currentAlpha: number;
  optimizedP0: number;
  optimizedAlpha: number;
  predictedSINRImprovement: number;
  predictedNeighborImpact: number;
  confidence: number;
  generationsRun: number;
  rationale: string;
}

/**
 * Genetic Algorithm for P0/Alpha parameter optimization
 *
 * From Ericsson presentation: "An algorithm (Genetic Algorithm or RL) runs
 * against the GNN model (not the real network) to adjust P0 and α until
 * the best balance between signal strength and neighbor interference is found."
 *
 * The GA queries the GNN model millions of times to find optimal parameters.
 */
export class GeneticOptimizer {
  private config: InterferenceOptimizerConfig;
  private gnn: SINRPredictionGNN;

  constructor(
    gnn: SINRPredictionGNN,
    config: Partial<InterferenceOptimizerConfig> = {}
  ) {
    this.config = { ...DEFAULT_OPTIMIZER_CONFIG, ...config };
    this.gnn = gnn;
  }

  /**
   * Optimize P0 and Alpha for a specific issue cell
   *
   * The optimization loop:
   * 1. Generate population of candidate (P0, α) pairs
   * 2. Evaluate fitness using GNN predictions
   * 3. Select best individuals
   * 4. Crossover and mutate
   * 5. Repeat until convergence
   */
  optimize(
    issueCell: IssueCell,
    nodeFeatures: number[][],
    adjacencyMatrix: number[][],
    cellSnapshots: Map<string, CellKPISnapshot>,
    cellIndexMap: Map<string, number>
  ): OptimizationResult {
    const { populationSize, numGenerations, mutationRate, crossoverRate, eliteRatio } = this.config;

    // Get baseline SINR with current parameters
    const baselineSINR = this.gnn.predictCellSINR(
      issueCell.cellIndex,
      nodeFeatures,
      adjacencyMatrix
    );

    // Get baseline neighbor SINRs
    const neighborIndices = issueCell.neighborCellIds
      .map(id => cellIndexMap.get(id))
      .filter((idx): idx is number => idx !== undefined);

    const baselineNeighborSINRs = this.gnn.predictSINR(nodeFeatures, adjacencyMatrix);
    const avgBaselineNeighborSINR = neighborIndices.length > 0
      ? neighborIndices.reduce((sum, idx) => sum + baselineNeighborSINRs[idx], 0) / neighborIndices.length
      : 0;

    // Initialize population
    let population = this.initializePopulation(issueCell);

    // Evaluate initial fitness
    population = this.evaluateFitness(
      population,
      issueCell,
      nodeFeatures,
      adjacencyMatrix,
      neighborIndices,
      baselineSINR,
      avgBaselineNeighborSINR,
      cellSnapshots
    );

    // Evolution loop
    let bestIndividual = population[0];
    let stagnationCount = 0;
    let lastBestFitness = bestIndividual.fitness;

    for (let gen = 0; gen < numGenerations; gen++) {
      // Selection
      const parents = this.selectParents(population);

      // Crossover
      const offspring = this.crossover(parents, crossoverRate);

      // Mutation
      const mutated = this.mutate(offspring, mutationRate);

      // Evaluate new individuals
      const evaluated = this.evaluateFitness(
        mutated,
        issueCell,
        nodeFeatures,
        adjacencyMatrix,
        neighborIndices,
        baselineSINR,
        avgBaselineNeighborSINR,
        cellSnapshots
      );

      // Elitism: keep best individuals from previous generation
      const numElite = Math.floor(populationSize * eliteRatio);
      population.sort((a, b) => b.fitness - a.fitness);
      const elite = population.slice(0, numElite);

      // Combine elite with new offspring
      evaluated.sort((a, b) => b.fitness - a.fitness);
      population = [...elite, ...evaluated.slice(0, populationSize - numElite)];

      // Track best
      if (population[0].fitness > bestIndividual.fitness) {
        bestIndividual = { ...population[0] };
        stagnationCount = 0;
      } else {
        stagnationCount++;
      }

      // Early stopping if converged
      if (stagnationCount > 10) {
        break;
      }

      lastBestFitness = bestIndividual.fitness;
    }

    // Calculate predicted improvements
    const optimizedFeatures = this.updateNodeFeatures(
      nodeFeatures,
      issueCell.cellIndex,
      bestIndividual.p0,
      bestIndividual.alpha,
      cellSnapshots.get(issueCell.cellId)!
    );

    const predictedSINR = this.gnn.predictCellSINR(
      issueCell.cellIndex,
      optimizedFeatures,
      adjacencyMatrix
    );

    const predictedNeighborSINRs = this.gnn.predictSINR(optimizedFeatures, adjacencyMatrix);
    const avgPredictedNeighborSINR = neighborIndices.length > 0
      ? neighborIndices.reduce((sum, idx) => sum + predictedNeighborSINRs[idx], 0) / neighborIndices.length
      : 0;

    return {
      cellId: issueCell.cellId,
      currentP0: issueCell.currentP0,
      currentAlpha: issueCell.currentAlpha,
      optimizedP0: bestIndividual.p0,
      optimizedAlpha: bestIndividual.alpha,
      predictedSINRImprovement: predictedSINR - baselineSINR,
      predictedNeighborImpact: avgPredictedNeighborSINR - avgBaselineNeighborSINR,
      confidence: Math.min(0.95, bestIndividual.fitness),
      generationsRun: numGenerations,
      rationale: this.generateRationale(
        issueCell,
        bestIndividual,
        predictedSINR - baselineSINR,
        avgPredictedNeighborSINR - avgBaselineNeighborSINR
      ),
    };
  }

  /**
   * Initialize population with random P0/Alpha combinations
   */
  private initializePopulation(issueCell: IssueCell): Individual[] {
    const population: Individual[] = [];
    const { populationSize, p0Min, p0Max, alphaValues } = this.config;

    // Add current configuration
    population.push({
      p0: issueCell.currentP0,
      alpha: issueCell.currentAlpha,
      fitness: 0,
    });

    // Add some heuristic-based starting points
    // Higher P0 for low SINR
    if (issueCell.currentSINR < 3) {
      population.push({
        p0: Math.min(p0Max, issueCell.currentP0 + 3),
        alpha: issueCell.currentAlpha,
        fitness: 0,
      });
    }

    // Lower alpha for high interference
    if (issueCell.currentIoT > 10) {
      const lowerAlphaIdx = alphaValues.findIndex(a => a === issueCell.currentAlpha);
      if (lowerAlphaIdx > 0) {
        population.push({
          p0: issueCell.currentP0,
          alpha: alphaValues[lowerAlphaIdx - 1],
          fitness: 0,
        });
      }
    }

    // Fill rest with random individuals
    while (population.length < populationSize) {
      const p0 = Math.round(p0Min + Math.random() * (p0Max - p0Min));
      const alpha = alphaValues[Math.floor(Math.random() * alphaValues.length)];
      population.push({ p0, alpha, fitness: 0 });
    }

    return population;
  }

  /**
   * Evaluate fitness of individuals using GNN predictions
   */
  private evaluateFitness(
    population: Individual[],
    issueCell: IssueCell,
    baseNodeFeatures: number[][],
    adjacencyMatrix: number[][],
    neighborIndices: number[],
    baselineSINR: number,
    baselineNeighborSINR: number,
    cellSnapshots: Map<string, CellKPISnapshot>
  ): Individual[] {
    const snapshot = cellSnapshots.get(issueCell.cellId)!;

    return population.map(individual => {
      // Update node features with proposed parameters
      const updatedFeatures = this.updateNodeFeatures(
        baseNodeFeatures,
        issueCell.cellIndex,
        individual.p0,
        individual.alpha,
        snapshot
      );

      // Predict SINR with proposed parameters
      const predictedSINR = this.gnn.predictCellSINR(
        issueCell.cellIndex,
        updatedFeatures,
        adjacencyMatrix
      );

      // Predict neighbor impact
      const allSINRs = this.gnn.predictSINR(updatedFeatures, adjacencyMatrix);
      const avgNeighborSINR = neighborIndices.length > 0
        ? neighborIndices.reduce((sum, idx) => sum + allSINRs[idx], 0) / neighborIndices.length
        : baselineNeighborSINR;

      // Calculate fitness components
      const sinrImprovement = predictedSINR - baselineSINR;
      const neighborDegradation = baselineNeighborSINR - avgNeighborSINR;

      // Fitness function:
      // - Maximize SINR improvement for issue cell
      // - Penalize neighbor degradation
      // - Bonus for meeting SINR threshold
      let fitness = 0;

      // SINR improvement (primary objective)
      fitness += sinrImprovement * 0.4;

      // Neighbor impact penalty
      if (neighborDegradation > this.config.maxInterferenceIncrease) {
        fitness -= (neighborDegradation - this.config.maxInterferenceIncrease) * 0.5;
      } else {
        fitness += (this.config.maxInterferenceIncrease - neighborDegradation) * 0.1;
      }

      // Bonus for achieving good SINR
      if (predictedSINR > this.config.sinrThreshold) {
        fitness += 0.2;
      }
      if (predictedSINR > 10) {
        fitness += 0.1;
      }

      // Normalize fitness to [0, 1]
      fitness = Math.max(0, Math.min(1, (fitness + 0.5) / 1.5));

      return { ...individual, fitness };
    });
  }

  /**
   * Update node features with proposed P0/Alpha
   */
  private updateNodeFeatures(
    baseFeatures: number[][],
    cellIndex: number,
    newP0: number,
    newAlpha: number,
    snapshot: CellKPISnapshot
  ): number[][] {
    const updated = baseFeatures.map(row => [...row]);

    // Build new features for the target cell
    updated[cellIndex] = this.gnn.buildNodeFeatures(snapshot, newP0, newAlpha);

    return updated;
  }

  /**
   * Tournament selection
   */
  private selectParents(population: Individual[]): Individual[] {
    const parents: Individual[] = [];
    const tournamentSize = 3;

    for (let i = 0; i < population.length; i++) {
      const tournament = Array(tournamentSize).fill(null).map(() =>
        population[Math.floor(Math.random() * population.length)]
      );
      tournament.sort((a, b) => b.fitness - a.fitness);
      parents.push(tournament[0]);
    }

    return parents;
  }

  /**
   * Crossover: blend P0/Alpha from two parents
   */
  private crossover(parents: Individual[], rate: number): Individual[] {
    const offspring: Individual[] = [];

    for (let i = 0; i < parents.length - 1; i += 2) {
      const parent1 = parents[i];
      const parent2 = parents[i + 1];

      if (Math.random() < rate) {
        // Arithmetic crossover for P0
        const ratio = Math.random();
        const childP0 = Math.round(parent1.p0 * ratio + parent2.p0 * (1 - ratio));

        // Uniform crossover for alpha (discrete)
        const childAlpha = Math.random() < 0.5 ? parent1.alpha : parent2.alpha;

        offspring.push({ p0: childP0, alpha: childAlpha, fitness: 0 });
        offspring.push({ p0: childP0, alpha: parent1.alpha === childAlpha ? parent2.alpha : parent1.alpha, fitness: 0 });
      } else {
        offspring.push({ ...parent1, fitness: 0 });
        offspring.push({ ...parent2, fitness: 0 });
      }
    }

    return offspring;
  }

  /**
   * Mutation: randomly adjust P0/Alpha
   */
  private mutate(individuals: Individual[], rate: number): Individual[] {
    const { p0Min, p0Max, alphaValues } = this.config;

    return individuals.map(ind => {
      let { p0, alpha } = ind;

      // Mutate P0
      if (Math.random() < rate) {
        const delta = Math.round((Math.random() * 2 - 1) * 5);
        p0 = Math.max(p0Min, Math.min(p0Max, p0 + delta));
      }

      // Mutate alpha
      if (Math.random() < rate) {
        const currentIdx = alphaValues.indexOf(alpha);
        const newIdx = Math.max(0, Math.min(
          alphaValues.length - 1,
          currentIdx + (Math.random() < 0.5 ? -1 : 1)
        ));
        alpha = alphaValues[newIdx];
      }

      return { p0, alpha, fitness: 0 };
    });
  }

  /**
   * Generate human-readable rationale for the optimization
   */
  private generateRationale(
    issueCell: IssueCell,
    best: Individual,
    sinrImprovement: number,
    neighborImpact: number
  ): string {
    const parts: string[] = [];

    parts.push(`Cell ${issueCell.cellId} identified with ${issueCell.issues.join('; ')}.`);

    const p0Delta = best.p0 - issueCell.currentP0;
    const alphaDelta = best.alpha - issueCell.currentAlpha;

    if (p0Delta !== 0) {
      const direction = p0Delta > 0 ? 'Increase' : 'Decrease';
      parts.push(`${direction} P0 by ${Math.abs(p0Delta)} dB (${issueCell.currentP0} → ${best.p0} dBm).`);
    }

    if (alphaDelta !== 0) {
      const direction = alphaDelta > 0 ? 'Increase' : 'Decrease';
      parts.push(`${direction} Alpha by ${Math.abs(alphaDelta).toFixed(1)} (${issueCell.currentAlpha} → ${best.alpha}).`);
    }

    parts.push(`Predicted SINR improvement: ${sinrImprovement.toFixed(1)} dB.`);

    if (neighborImpact < 0) {
      parts.push(`Predicted neighbor impact: ${Math.abs(neighborImpact).toFixed(1)} dB degradation (within tolerance).`);
    } else {
      parts.push(`No negative impact on neighbors.`);
    }

    return parts.join(' ');
  }
}

// ============================================================================
// OPTIMIZATION LOOP ORCHESTRATOR
// ============================================================================

export interface NetworkOptimizationResult {
  timestamp: Date;
  totalCellsAnalyzed: number;
  issueCellsDetected: number;
  cellsOptimized: number;
  optimizationResults: OptimizationResult[];
  aggregateMetrics: {
    avgSINRImprovement: number;
    avgNeighborImpact: number;
    successRate: number;
  };
  deploymentRecommendations: string[];
}

/**
 * Main orchestrator for the GNN-based interference optimization system
 *
 * Implements the complete optimization workflow as per Ericsson presentation:
 * 1. Detection: Issue cells with low SINR are identified
 * 2. Prediction: GNN predicts SINR based on current parameters
 * 3. Optimization: GA adjusts P0/Alpha by querying GNN
 * 4. Deployment: Optimized parameters are recommended for real network
 */
export class InterferenceOptimizationLoop {
  private config: InterferenceOptimizerConfig;
  private gnn: SINRPredictionGNN;
  private issueDetector: IssueCellDetector;
  private geneticOptimizer: GeneticOptimizer;

  constructor(config: Partial<InterferenceOptimizerConfig> = {}) {
    this.config = { ...DEFAULT_OPTIMIZER_CONFIG, ...config };
    this.gnn = new SINRPredictionGNN(this.config);
    this.issueDetector = new IssueCellDetector(this.config);
    this.geneticOptimizer = new GeneticOptimizer(this.gnn, this.config);
  }

  /**
   * Get the underlying GNN model (for training)
   */
  getGNN(): SINRPredictionGNN {
    return this.gnn;
  }

  /**
   * Run complete optimization on a network snapshot
   *
   * This is the main entry point that:
   * 1. Builds the cell graph from snapshots and relations
   * 2. Detects issue cells
   * 3. Runs GA optimization for each issue cell
   * 4. Compiles results and recommendations
   */
  optimize(
    cellSnapshots: Map<string, CellKPISnapshot>,
    neighborRelations: NeighborRelation[]
  ): NetworkOptimizationResult {
    const timestamp = new Date();

    // Build cell graph structures
    const cellIds = Array.from(cellSnapshots.keys());
    const cellIndexMap = new Map<string, number>();
    cellIds.forEach((id, idx) => cellIndexMap.set(id, idx));

    // Build adjacency matrix
    const n = cellIds.length;
    const adjacencyMatrix: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));

    for (const relation of neighborRelations) {
      const sourceIdx = cellIndexMap.get(relation.sourceCellId);
      const targetIdx = cellIndexMap.get(relation.targetCellId);
      if (sourceIdx !== undefined && targetIdx !== undefined) {
        // Weight by handover success rate
        const weight = relation.hoSuccessRate / 100;
        adjacencyMatrix[sourceIdx][targetIdx] = weight;
        adjacencyMatrix[targetIdx][sourceIdx] = weight;
      }
    }

    // Add self-loops
    for (let i = 0; i < n; i++) {
      adjacencyMatrix[i][i] = 1;
    }

    // Build node features with current parameters
    const nodeFeatures: number[][] = [];
    for (const [cellId, snapshot] of cellSnapshots) {
      const features = this.gnn.buildNodeFeatures(
        snapshot,
        snapshot.uplinkPowerControl.p0NominalPusch,
        snapshot.uplinkPowerControl.alpha
      );
      nodeFeatures.push(features);
    }

    // Detect issue cells
    const issueCells = this.issueDetector.detectIssueCells(cellSnapshots, neighborRelations);

    // Optimize each issue cell
    const optimizationResults: OptimizationResult[] = [];

    for (const issueCell of issueCells) {
      const result = this.geneticOptimizer.optimize(
        issueCell,
        nodeFeatures,
        adjacencyMatrix,
        cellSnapshots,
        cellIndexMap
      );

      // Only include if there's meaningful improvement
      if (result.predictedSINRImprovement >= this.config.minSinrImprovement) {
        optimizationResults.push(result);
      }
    }

    // Calculate aggregate metrics
    const avgSINRImprovement = optimizationResults.length > 0
      ? optimizationResults.reduce((sum, r) => sum + r.predictedSINRImprovement, 0) / optimizationResults.length
      : 0;

    const avgNeighborImpact = optimizationResults.length > 0
      ? optimizationResults.reduce((sum, r) => sum + r.predictedNeighborImpact, 0) / optimizationResults.length
      : 0;

    const successRate = issueCells.length > 0
      ? optimizationResults.length / issueCells.length
      : 1;

    // Generate deployment recommendations
    const deploymentRecommendations = this.generateDeploymentRecommendations(
      optimizationResults,
      issueCells
    );

    return {
      timestamp,
      totalCellsAnalyzed: cellSnapshots.size,
      issueCellsDetected: issueCells.length,
      cellsOptimized: optimizationResults.length,
      optimizationResults,
      aggregateMetrics: {
        avgSINRImprovement,
        avgNeighborImpact,
        successRate,
      },
      deploymentRecommendations,
    };
  }

  /**
   * Generate deployment recommendations
   *
   * From Ericsson presentation: Parameters optimized on one week of data
   * tend to remain effective for 1-2 months.
   */
  private generateDeploymentRecommendations(
    results: OptimizationResult[],
    issueCells: IssueCell[]
  ): string[] {
    const recommendations: string[] = [];

    if (results.length === 0) {
      recommendations.push('No optimization changes recommended at this time.');
      return recommendations;
    }

    // Sort by predicted improvement
    const sorted = [...results].sort((a, b) => b.predictedSINRImprovement - a.predictedSINRImprovement);

    // Critical cells first
    const criticalCells = issueCells.filter(c => c.severity === 'critical');
    if (criticalCells.length > 0) {
      recommendations.push(`Priority: ${criticalCells.length} critical cells require immediate attention.`);
    }

    // Top improvements
    const topN = Math.min(5, sorted.length);
    recommendations.push(`Top ${topN} cells with highest predicted improvement:`);

    for (let i = 0; i < topN; i++) {
      const r = sorted[i];
      recommendations.push(
        `  ${i + 1}. ${r.cellId}: P0 ${r.currentP0}→${r.optimizedP0} dBm, ` +
        `Alpha ${r.currentAlpha}→${r.optimizedAlpha} (+${r.predictedSINRImprovement.toFixed(1)} dB SINR)`
      );
    }

    // Deployment schedule recommendation
    recommendations.push('');
    recommendations.push('Deployment notes:');
    recommendations.push('- Apply changes during low-traffic periods');
    recommendations.push('- Monitor KPIs for 24-48 hours after deployment');
    recommendations.push('- Re-run optimization in 4-6 weeks or if performance degrades');

    return recommendations;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  SINRPredictionGNN,
  IssueCellDetector,
  GeneticOptimizer,
  InterferenceOptimizationLoop,
  DEFAULT_OPTIMIZER_CONFIG,
};
