/**
 * RuvSwarm - Swarm Orchestration Engine for RAN Optimization
 *
 * Ruv-swarm v1.0.20 provides swarm intelligence capabilities:
 * - Neural Routing (tiny-dancer) for task complexity routing
 * - Federated Learning with FedAvg, FedProx, FedNova
 * - Leader Election via Bully algorithm
 * - Pattern Propagation for sharing successful optimizations
 * - Strange Loops Consensus for model provenance
 */

export { SwarmOrchestrator } from './swarm-orchestrator.js';
export type {
  SwarmOrchestratorConfig,
  TaskComplexity,
  RoutingDecision,
  AggregationAlgorithm
} from './swarm-orchestrator.js';
