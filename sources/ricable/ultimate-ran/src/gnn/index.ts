/**
 * GNN Module - TITAN Neuro-Symbolic RAN Platform
 *
 * Centralized exports for Graph Neural Network components
 */

// Core GNN Optimizer
export {
  GraphAttentionNetwork,
  GNNUplinkOptimizer,
  P0_MIN,
  P0_MAX,
  ALPHA_MIN,
  ALPHA_MAX,
  TARGET_SINR_MIN,
  TARGET_SINR_MAX,
  EMBEDDING_DIM,
  NUM_ATTENTION_HEADS,
  GAT_HIDDEN_DIM
} from './uplink-optimizer.js';

// Graph Attention Network
export { GraphAttentionNetwork as GAT } from './graph-attention.js';

// Interference Graph Builder
export { InterferenceGraphBuilder } from './interference-graph.js';

// P0/Alpha Controller
export { P0AlphaController } from './p0-alpha-controller.js';

// Type Definitions
export type {
  CellNode,
  InterferenceEdge,
  CellGraph,
  OptimizationResult,
  GATConfig,
  AttentionWeights,
  ParameterOptimizationResult,
  ValidationResult,
  GraphBuilderOptions,
  OptimizationOptions,
  OptimizationOutcome,
  AccuracyMetrics
} from './types.js';
