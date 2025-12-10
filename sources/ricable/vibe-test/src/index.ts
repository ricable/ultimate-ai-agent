/**
 * Neuro-Federated Swarm Intelligence for Ericsson RAN Optimization
 * Main entry point and public API
 */

// Core types
export * from './core/types.js';

// Orchestration
export { SwarmOrchestrator, createSwarm, SandboxManager } from './orchestration/swarm-orchestrator.js';
export {
  AgentFactory,
  BaseAgent,
  OptimizerAgent,
  HealerAgent,
  ConfiguratorAgent,
} from './orchestration/agent-runtime.js';

// Performance Management
export { ChaosAnalyzer, createChaosAnalyzer } from './performance/chaos-analyzer.js';
export { NeuralForecaster, createNeuralForecaster } from './performance/neural-forecaster.js';

// Fault Management
export { AnomalyDetector, createAnomalyDetector } from './fault/anomaly-detector.js';
export { NeuroSymbolicRCA, createNeuroSymbolicRCA } from './fault/neuro-symbolic-rca.js';

// Configuration Management
export { GOAPPlanner, createGOAPPlanner } from './configuration/goap-planner.js';
export { SafeExecutor, createSafeExecutor } from './configuration/safe-executor.js';

// Graph Neural Networks
export { TopologyModel, createTopologyModel } from './gnn/topology-model.js';

// Ericsson Integration
export {
  EricssonMOClient,
  createEricssonMOClient,
  STANDARD_PM_COUNTERS,
} from './ericsson/mo-client.js';

// Utilities
export { Logger, createLogger, LogLevel } from './utils/logger.js';
export * from './utils/math.js';

import { createSwarm } from './orchestration/swarm-orchestrator.js';
import { createChaosAnalyzer } from './performance/chaos-analyzer.js';
import { createNeuralForecaster } from './performance/neural-forecaster.js';
import { createAnomalyDetector } from './fault/anomaly-detector.js';
import { createNeuroSymbolicRCA } from './fault/neuro-symbolic-rca.js';
import { createGOAPPlanner } from './configuration/goap-planner.js';
import { createSafeExecutor } from './configuration/safe-executor.js';
import { createTopologyModel } from './gnn/topology-model.js';
import { createEricssonMOClient } from './ericsson/mo-client.js';
import { createLogger } from './utils/logger.js';

const logger = createLogger('NeuroFederatedSwarm');

/**
 * Complete Neuro-Federated Swarm system
 */
export interface NeuroFederatedSystem {
  swarm: Awaited<ReturnType<typeof createSwarm>>;
  chaosAnalyzer: ReturnType<typeof createChaosAnalyzer>;
  neuralForecaster: ReturnType<typeof createNeuralForecaster>;
  anomalyDetector: ReturnType<typeof createAnomalyDetector>;
  neuroSymbolicRCA: ReturnType<typeof createNeuroSymbolicRCA>;
  goapPlanner: ReturnType<typeof createGOAPPlanner>;
  safeExecutor: ReturnType<typeof createSafeExecutor>;
  topologyModel: ReturnType<typeof createTopologyModel>;
  enmClient: ReturnType<typeof createEricssonMOClient>;
}

/**
 * Initialize complete Neuro-Federated Swarm system
 *
 * Equivalent to running:
 * npx claude-flow@alpha init --force --sublinear --neural
 */
export async function initializeSystem(config?: {
  enmHost?: string;
  topology?: 'mesh' | 'ring' | 'star' | 'hierarchical';
  embeddingDim?: number;
}): Promise<NeuroFederatedSystem> {
  logger.info('Initializing Neuro-Federated Swarm system');

  // Initialize all components
  const [
    swarm,
    chaosAnalyzer,
    neuralForecaster,
    anomalyDetector,
    neuroSymbolicRCA,
    goapPlanner,
    safeExecutor,
    topologyModel,
    enmClient,
  ] = await Promise.all([
    createSwarm({
      topology: config?.topology || 'hierarchical',
      force: true,
      sublinear: true,
      neural: true,
    }),
    Promise.resolve(createChaosAnalyzer()),
    Promise.resolve(createNeuralForecaster()),
    Promise.resolve(createAnomalyDetector()),
    Promise.resolve(createNeuroSymbolicRCA()),
    Promise.resolve(createGOAPPlanner()),
    Promise.resolve(createSafeExecutor()),
    Promise.resolve(createTopologyModel(config?.embeddingDim)),
    Promise.resolve(createEricssonMOClient(config?.enmHost)),
  ]);

  logger.info('Neuro-Federated Swarm system initialized successfully');

  return {
    swarm,
    chaosAnalyzer,
    neuralForecaster,
    anomalyDetector,
    neuroSymbolicRCA,
    goapPlanner,
    safeExecutor,
    topologyModel,
    enmClient,
  };
}

/**
 * Shutdown the system gracefully
 */
export async function shutdownSystem(system: NeuroFederatedSystem): Promise<void> {
  logger.info('Shutting down Neuro-Federated Swarm system');

  await system.swarm.shutdown();
  await system.enmClient.disconnect();

  logger.info('System shutdown complete');
}
