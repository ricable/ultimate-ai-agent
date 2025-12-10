/**
 * GNN Type Definitions - TITAN Neuro-Symbolic RAN Platform
 *
 * Type definitions for Graph Neural Network-based uplink optimization
 */

/**
 * Cell Node in the interference graph
 * PRD Reference: Line 1245-1262 (interface specification)
 */
export interface CellNode {
  cellId: string;
  features: number[]; // [SINR, RSRP, PRB utilization, CQI]
  p0?: number; // Target received power: -130 to -70 dBm (3GPP TS 38.213)
  alpha?: number; // Pathloss compensation: 0 to 1
  embedding?: number[]; // 768-dimensional learned representation
}

/**
 * Interference edge between cells
 */
export interface InterferenceEdge {
  fromCell: string;
  toCell: string;
  features: [number, number, number]; // [distance_m, overlap_pct, coupling_dB]
  distance: number; // meters
  overlapPct: number; // 0 to 1
  interferenceCoupling: number; // dB
}

/**
 * Graph structure for GNN
 */
export interface CellGraph {
  nodes: CellNode[];
  edges: InterferenceEdge[];
}

/**
 * GNN optimization result
 */
export interface OptimizationResult {
  optimizedCells: CellNode[];
  predictedSINR: number[];
  actualSINR: number[];
  recommendations: string[] | Array<{
    message: string;
    impact: 'high' | 'medium' | 'low';
  }>;
  metadata: {
    rmse: number;
    attentionAggregation: 'mean' | 'concat' | 'max';
    optimizationMode: 'joint' | 'sequential';
    alphaStrategy: 'full-compensation' | 'partial-compensation';
    propagationSteps: number;
    convergence: boolean;
    cellAccuracy: Record<string, {
      rmse: number;
      confidence: number;
    }>;
    components?: number;
  };
}

/**
 * GAT configuration
 */
export interface GATConfig {
  numHeads: number; // 8 heads for multi-head attention
  embeddingDim: number; // 768 dimensions
  featureDim: number; // 4 input features
  activation?: 'leaky_relu' | 'relu' | 'elu';
  leakyReluAlpha?: number;
  aggregation?: 'mean' | 'concat' | 'max';
  useEdgeFeatures?: boolean;
}

/**
 * Attention weights structure
 */
export interface AttentionWeights {
  heads: number;
  weights: number[][][]; // [head][node][neighbor]
}

/**
 * P0/Alpha optimization result
 */
export interface ParameterOptimizationResult {
  p0: number;
  alpha: number;
  rationale: string;
  strategy: 'gradient' | 'rules' | 'hybrid';
  confidence: number;
}

/**
 * Parameter validation result
 */
export interface ValidationResult {
  valid: boolean;
  violations: string[];
}

/**
 * Graph builder options
 */
export interface GraphBuilderOptions {
  minCoupling?: number; // Minimum interference coupling (dB)
  maxDistance?: number; // Maximum cell distance (meters)
  maxNeighbors?: number; // Maximum neighbors per cell
}

/**
 * Optimization strategy options
 */
export interface OptimizationOptions {
  strategy?: 'gradient' | 'rules' | 'hybrid';
  maxIterations?: number;
  convergenceThreshold?: number;
  learningRate?: number;
}

/**
 * Historical outcome record
 */
export interface OptimizationOutcome {
  p0: number;
  alpha: number;
  sinrDelta: number; // SINR change after optimization
  timestamp?: number;
}

/**
 * Accuracy metrics
 */
export interface AccuracyMetrics {
  rmse: number; // Root Mean Square Error
  mae: number; // Mean Absolute Error
  r2?: number; // R-squared
}
