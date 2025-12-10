/**
 * ReasoningBank - Closed-loop memory system for AI agents
 * Based on arXiv:2509.25140 (Google DeepMind)
 */

export { ReasoningBankEngine } from './core/memory-engine.js';
export { ReasoningBankDB } from './core/database.js';
export { piiScrubber, PIIScrubber } from './utils/pii-scrubber.js';
export { createEmbeddingProvider, cosineSimilarity } from './utils/embeddings.js';

export type {
  Memory,
  PatternData,
  PatternEmbedding,
  TaskTrajectory,
  MattsRun,
  RetrievalOptions,
  ScoringWeights,
  JudgmentResult,
  ConsolidationOptions,
  ConsolidationStats,
  ReasoningBankConfig,
  TaskExecutionOptions,
  TaskResult,
  MemoryCandidate
} from './types/index.js';
