/**
 * AI Integrations Index
 * Exports Claude Agent SDK and Google ADK integrations for TITAN Dashboard
 *
 * @module ui/integrations
 * @version 7.0.0-alpha.1
 */

// Claude Agent SDK Integration
export {
  ClaudeAgentIntegration,
  type ClaudeAgentConfig
} from './claude-agent-integration.js';

// Google Generative AI (Gemini) Integration
export {
  GoogleADKIntegration,
  type GoogleADKConfig
} from './google-adk-integration.js';

// AI Orchestrator (Hybrid Intelligence)
export {
  AIOrchestrator,
  type AIOrchestrationConfig,
  type OptimizationResult
} from './ai-orchestrator.js';

// AI Swarm Coordinator (Multi-Agent Consensus Reasoning)
export {
  AISwarmCoordinator,
  type SwarmConfig,
  type SwarmAgent,
  type ConsensusResult,
  type SwarmTask
} from './swarm-coordinator.js';

// Example demonstrating usage
export { runExamples } from './example.js';
