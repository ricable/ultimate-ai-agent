/**
 * RuVector Telecom RAG - Main Entry Point
 *
 * Cognitive Automation Platform for Ericsson RAN
 * Self-Learning RAG System for ELEX and 3GPP Documentation
 */

export { RAGPipeline } from './rag/rag-pipeline.js';
export { SelfLearningVectorStore } from './rag/vector-store.js';
export { ELEXParser } from './parsers/elex-parser.js';
export { ThreeGPPParser } from './parsers/threegpp-parser.js';
export { NetworkGraphBuilder } from './gnn/network-graph.js';
export { GNNInferenceEngine, BayesianGNN } from './gnn/gnn-engine.js';
export { SwarmController, OptimizerAgent, ValidatorAgent, AuditorAgent } from './agents/agent-swarm.js';
export { getConfig } from './core/config.js';
export { logger } from './utils/logger.js';
export * from './core/types.js';

import { startServer } from './server/index.js';

// Start server if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  startServer();
}
