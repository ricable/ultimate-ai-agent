/**
 * TITAN RAN Machine Learning Module
 *
 * Complete self-learning RAN optimization system combining:
 * - RuvVector: HNSW spatial embeddings for cells
 * - RuvLLM: Natural language RAN querying
 * - Graph Attention Network: Multi-head attention for interference
 * - AgentDB: Reflexion memory for transfer learning
 * - PydanticAI: Type-safe validation for ML outputs
 *
 * @module ml
 * @version 7.0.0-alpha.1
 */

// ============================================================================
// RuvVector + RuvLLM
// ============================================================================

export {
  RuvectorGNN,
  RuvLLMClient,
  type CellEmbedding,
  type OptimizationEpisode,
  type RANInsight,
  type Recommendation,
  type SimilarityResult
} from './ruvector-gnn.js';

// ============================================================================
// Graph Attention Network
// ============================================================================

export {
  GraphAttentionNetwork,
  type CellNode,
  type InterferenceEdge,
  type InterferenceGraph,
  type GATConfig,
  type AttentionOutput,
  type PropagationResult
} from './attention-gnn.js';

// ============================================================================
// AgentDB Reflexion Memory
// ============================================================================

export {
  AgentDBReflexion,
  type MemoryEntry,
  type MemoryQueryResult,
  type TransferLearningResult,
  type ReflexionStats,
  type AgentDBConfig
} from './agentdb-reflexion.js';

// ============================================================================
// PydanticAI Validation
// ============================================================================

export {
  Validator,
  ThreeGPPValidators,
  PhysicsValidators,
  CMParametersSchema,
  RecommendationSchema,
  cmParametersValidator,
  recommendationValidator,
  type ValidationError,
  type ValidationResult,
  type ValidationSchema,
  type SchemaField,
  type FieldValidator
} from './pydantic-validation.js';

// ============================================================================
// Integrated System
// ============================================================================

export {
  SelfLearningRANSystem,
  runExample
} from './integration-example.js';

// ============================================================================
// Quick Start Factory
// ============================================================================

import { SelfLearningRANSystem } from './integration-example.js';

/**
 * Create and initialize a complete self-learning RAN system
 *
 * Usage:
 * ```typescript
 * import { createSelfLearningRAN } from './ml';
 *
 * const ran = await createSelfLearningRAN();
 *
 * // Optimize a cell
 * const result = await ran.optimizeCell('ABC123');
 *
 * // Natural language query
 * const insight = await ran.query('What cells need optimization?');
 *
 * // Execute and learn
 * await ran.executeAndLearn('ABC123', action, expectedOutcome);
 * ```
 */
export async function createSelfLearningRAN(): Promise<SelfLearningRANSystem> {
  const system = new SelfLearningRANSystem();
  await system.initialize();
  return system;
}

/**
 * Module version
 */
export const VERSION = '7.0.0-alpha.1';

/**
 * Module metadata
 */
export const METADATA = {
  name: 'TITAN RAN Self-Learning ML',
  version: VERSION,
  components: [
    'RuvVector GNN',
    'RuvLLM',
    'Graph Attention Network',
    'AgentDB Reflexion',
    'PydanticAI Validation'
  ],
  capabilities: [
    'Spatial cell similarity search (<10ms)',
    'Transfer learning across cells',
    'Natural language RAN queries',
    'Multi-head attention for interference',
    'Network-wide propagation prediction',
    '3GPP + physics constraint validation',
    'Reflexion memory for continuous learning'
  ],
  performance: {
    vectorSearchLatency: '<10ms',
    embeddingDimension: 768,
    hnswM: 32,
    hnswEfConstruction: 200,
    hnswEfSearch: 100,
    attentionHeads: 8,
    maxMemorySize: 100000
  }
} as const;
