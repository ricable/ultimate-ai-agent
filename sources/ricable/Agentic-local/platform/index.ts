/**
 * HyperScale AI Agents Platform
 *
 * Main entry point for the sovereign edge-native AI infrastructure
 *
 * @module @edge-ai/hyperscale-platform
 */

// Core Components
export { QUADScheduler } from './core/quad/scheduler';
export { RuvLLMEngine, DEFAULT_MODELS } from './core/ruvllm/inference-engine';

// GPU Acceleration
export {
  MacSiliconAccelerator,
  LlamaEdgeMetalRunner,
} from './gpu/mac-silicon-accelerator';

// Federation & Scaling
export {
  HybridCloudFederationController,
  ClusterRegistry,
  PlacementScheduler,
  SpilloverController,
  GaiaNetworkIntegration,
  DEFAULT_LOCAL_CLUSTER,
  DEFAULT_EDGE_CLUSTERS,
} from './federation/hybrid-cloud-controller';

// Agent Orchestration
export {
  UnifiedOrchestrator,
  createOrchestrator,
  createCodingSwarm,
  createResearchSwarm,
} from './agents/unified-orchestrator';

// MLX Deep Council - Distributed Multi-Model Consensus
export {
  MLXDeepCouncil,
  createLocalCouncil,
  createDistributedCouncil,
  createThunderboltCouncil,
} from './council';

// Types
export type {
  // From QUAD Scheduler
  QDAGNode,
  TaskPayload,
  ResourceRequirements,
  AgentCapabilities,
  SchedulerConfig,
  DAGExecution,
} from './core/quad/scheduler';

export type {
  // From ruvllm
  ModelConfig,
  InferenceRequest,
  InferenceResponse,
  StreamChunk,
  BackendStatus,
} from './core/ruvllm/inference-engine';

export type {
  // From Federation
  ClusterConfig,
  ClusterCapabilities,
  WorkloadSpec,
  PlacementDecision,
  SpilloverConfig,
  FederationMetrics,
} from './federation/hybrid-cloud-controller';

export type {
  // From Orchestrator
  OrchestratorConfig,
  AgentDefinition,
  Agent,
  Swarm,
  SwarmDefinition,
  TaskRequest,
  TaskResult,
} from './agents/unified-orchestrator';

export type {
  // From MLX Deep Council
  CouncilMember,
  CouncilConfig,
  CouncilQuery,
  CouncilSession,
  IndividualResponse,
  PeerReview,
  ChairmanSynthesis,
  SessionMetrics,
} from './council';

// =============================================================================
// QUICK START API
// =============================================================================

/**
 * Initialize the HyperScale platform with sensible defaults
 */
export async function initializeHyperScale(config?: {
  enableGPU?: boolean;
  enableFederation?: boolean;
  maxAgents?: number;
}): Promise<{
  orchestrator: UnifiedOrchestrator;
  inference: RuvLLMEngine;
  federation?: HybridCloudFederationController;
  gpu?: MacSiliconAccelerator;
}> {
  console.log('🚀 Initializing HyperScale AI Agents Platform...');

  // Create inference engine
  const inference = new RuvLLMEngine({
    backends: ['llamaedge', 'mlx', 'gaia'],
    autoSelectBackend: true,
    enableCaching: true,
  });

  await inference.initialize();
  console.log('✅ Inference engine initialized');

  // Create GPU accelerator if enabled
  let gpu: MacSiliconAccelerator | undefined;
  if (config?.enableGPU !== false && process.platform === 'darwin') {
    try {
      gpu = new MacSiliconAccelerator();
      await gpu.initialize();
      console.log('✅ GPU acceleration enabled (Mac Silicon)');
    } catch (error) {
      console.warn('⚠️ GPU acceleration not available:', error);
    }
  }

  // Create federation controller if enabled
  let federation: HybridCloudFederationController | undefined;
  if (config?.enableFederation !== false) {
    federation = new HybridCloudFederationController();
    await federation.initialize();
    console.log('✅ Hybrid cloud federation enabled');
  }

  // Create orchestrator
  const orchestrator = new UnifiedOrchestrator({
    maxAgents: config?.maxAgents || 10000,
    enableGPU: !!gpu,
    enableFederation: !!federation,
    enableAutoScale: true,
  });

  await orchestrator.initialize();
  console.log('✅ Agent orchestrator initialized');

  console.log('🎉 HyperScale platform ready!');

  return {
    orchestrator,
    inference,
    federation,
    gpu,
  };
}

/**
 * Quick agent creation
 */
export function quickAgent(
  orchestrator: UnifiedOrchestrator,
  type: 'coder' | 'researcher' | 'analyst' = 'coder'
) {
  return orchestrator.createAgent({
    name: `quick-${type}-${Date.now()}`,
    type,
    capabilities: [],
  });
}

/**
 * Quick swarm creation
 */
export function quickSwarm(
  orchestrator: UnifiedOrchestrator,
  size: number = 5,
  type: 'coding' | 'research' = 'coding'
) {
  if (type === 'coding') {
    return createCodingSwarm(orchestrator, size);
  } else {
    return createResearchSwarm(orchestrator, size);
  }
}

/**
 * Quick chat with an agent
 */
export async function quickChat(
  orchestrator: UnifiedOrchestrator,
  message: string,
  options?: {
    model?: string;
    temperature?: number;
    stream?: boolean;
  }
) {
  const result = await orchestrator.executeTask({
    type: 'chat',
    input: {
      message,
      model: options?.model || 'qwen-coder-7b',
      temperature: options?.temperature || 0.7,
    },
    options: {
      stream: options?.stream,
    },
  });

  return result.output;
}

/**
 * Quick council query - uses multiple models to reach consensus
 *
 * Implements Karpathy's LLM Council pattern:
 * 1. Query multiple models for individual responses
 * 2. Each model reviews and ranks others' responses
 * 3. Chairman model synthesizes final consensus answer
 *
 * @example
 * const result = await quickCouncil("What is the best approach to implement a cache?");
 * console.log(result.finalResponse);
 */
export async function quickCouncil(
  query: string,
  options?: {
    models?: string[];
    requireConsensus?: boolean;
    minAgreement?: number;
  }
) {
  const council = createLocalCouncil({
    name: 'Quick Council',
    models: options?.models || [
      'mlx-community/Llama-3.2-3B-Instruct-4bit',
      'mlx-community/Mistral-7B-Instruct-v0.3-4bit',
      'mlx-community/Qwen2.5-7B-Instruct-4bit',
    ],
  });

  try {
    await council.initialize();
    console.log('Council initialized with', council.getStatus().members.length, 'members');

    const session = await council.query({
      content: query,
      requireConsensus: options?.requireConsensus ?? true,
      minAgreement: options?.minAgreement ?? 0.6,
    });

    return {
      finalResponse: session.chairmanSynthesis?.finalResponse || '',
      reasoning: session.chairmanSynthesis?.reasoning || '',
      confidence: session.chairmanSynthesis?.confidenceScore || 0,
      consensusReached: session.consensusReached,
      consensusStrength: session.metrics.consensusStrength,
      individualResponses: session.individualResponses.map(r => ({
        model: r.anonymousId,
        content: r.content,
        score: session.aggregatedScores.get(r.anonymousId) || 0,
      })),
      metrics: session.metrics,
    };
  } finally {
    await council.shutdown();
  }
}

// =============================================================================
// CLI INTERFACE
// =============================================================================

if (require.main === module) {
  // Run as CLI
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
HyperScale AI Agents Platform

Usage: npx ts-node platform/index.ts [command] [options]

Commands:
  init              Initialize the platform
  agent <type>      Create a quick agent
  swarm <size>      Launch an agent swarm
  chat <message>    Chat with an agent
  council <query>   Query the MLX Deep Council for consensus
  status            Show platform status

Options:
  --gpu             Enable GPU acceleration
  --no-federation   Disable cluster federation
  --model <name>    Specify LLM model

Examples:
  npx ts-node platform/index.ts init
  npx ts-node platform/index.ts agent coder
  npx ts-node platform/index.ts swarm 100
  npx ts-node platform/index.ts chat "Write a sorting algorithm"
  npx ts-node platform/index.ts council "What is the best database for this use case?"

MLX Deep Council:
  The council uses multiple LLM models to reach consensus on complex queries.
  Based on Karpathy's LLM Council pattern with three stages:
  1. Individual responses from each model
  2. Peer review and ranking
  3. Chairman synthesis

  For distributed council across multiple Macs:
  npx ts-node platform/council/council-launcher.ts --help
    `);
    process.exit(0);
  }

  const command = args[0];

  (async () => {
    // Handle council command separately (doesn't need full platform init)
    if (command === 'council') {
      const query = args.slice(1).join(' ');
      if (!query) {
        console.error('Please provide a query for the council');
        process.exit(1);
      }

      console.log('Querying MLX Deep Council...\n');
      const result = await quickCouncil(query);

      console.log('='.repeat(80));
      console.log('COUNCIL CONSENSUS');
      console.log('='.repeat(80));
      console.log(`\n${result.finalResponse}\n`);
      console.log('-'.repeat(80));
      console.log(`Consensus: ${result.consensusReached ? 'REACHED' : 'NOT REACHED'}`);
      console.log(`Confidence: ${(result.confidence * 100).toFixed(0)}%`);
      console.log(`Strength: ${(result.consensusStrength * 100).toFixed(0)}%`);

      if (result.reasoning) {
        console.log(`\nReasoning: ${result.reasoning}`);
      }

      console.log('\nIndividual Responses:');
      for (const resp of result.individualResponses) {
        console.log(`  - ${resp.model}: Score ${resp.score.toFixed(1)}/10`);
      }

      return;
    }

    const { orchestrator } = await initializeHyperScale({
      enableGPU: !args.includes('--no-gpu'),
      enableFederation: !args.includes('--no-federation'),
    });

    switch (command) {
      case 'init':
        console.log('Platform initialized successfully!');
        break;

      case 'agent':
        const agentType = (args[1] || 'coder') as 'coder' | 'researcher' | 'analyst';
        const agent = quickAgent(orchestrator, agentType);
        console.log(`Created agent: ${agent.id}`);
        break;

      case 'swarm':
        const size = parseInt(args[1] || '5', 10);
        const swarm = quickSwarm(orchestrator, size);
        console.log(`Created swarm: ${swarm.id} with ${size} agents`);
        break;

      case 'chat':
        const message = args.slice(1).join(' ') || 'Hello!';
        const response = await quickChat(orchestrator, message);
        console.log('Response:', response.response);
        break;

      case 'status':
        console.log('Platform Status:', orchestrator.getStatus());
        break;

      default:
        console.log('Unknown command. Use --help for usage.');
    }

    await orchestrator.shutdown();
  })().catch(console.error);
}
