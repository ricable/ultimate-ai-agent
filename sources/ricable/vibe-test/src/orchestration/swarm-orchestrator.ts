/**
 * Swarm Orchestrator - Hive-Mind coordination for distributed agents
 * Implements claude-flow patterns with MCP standardization
 */

import { v4 as uuidv4 } from 'uuid';
import {
  Agent,
  AgentRole,
  SwarmConfiguration,
  SwarmMetrics,
  SwarmState,
  SwarmTopology,
  Task,
  TaskPriority,
  TaskResult,
  SystemEvent,
  EventType,
  FederatedModel,
  GradientUpdate,
} from '../core/types.js';
import { AgentFactory, BaseAgent } from './agent-runtime.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('SwarmOrchestrator');

/**
 * Event emitter interface for swarm events
 */
export interface SwarmEventHandler {
  onEvent(event: SystemEvent): void;
}

/**
 * Swarm Orchestrator - Central nervous system of the Neuro-Federated architecture
 */
export class SwarmOrchestrator {
  private swarmId: string;
  private configuration: SwarmConfiguration;
  private agents: Map<string, Agent>;
  private queenAgent?: Agent;
  private eventHandlers: SwarmEventHandler[];
  private taskQueue: Task[];
  private isRunning: boolean;
  private metrics: SwarmMetrics;
  private federatedModels: Map<string, FederatedModel>;
  private gradientBuffer: GradientUpdate[];

  constructor(configuration: Partial<SwarmConfiguration> = {}) {
    this.swarmId = uuidv4();
    this.configuration = {
      topology: configuration.topology || 'hierarchical',
      maxAgents: configuration.maxAgents || 100,
      communicationProtocol: configuration.communicationProtocol || 'quic',
      federationEnabled: configuration.federationEnabled ?? true,
      sandboxed: configuration.sandboxed ?? true,
    };
    this.agents = new Map();
    this.eventHandlers = [];
    this.taskQueue = [];
    this.isRunning = false;
    this.metrics = this.initializeMetrics();
    this.federatedModels = new Map();
    this.gradientBuffer = [];

    logger.info(`Swarm orchestrator initialized`, {
      swarmId: this.swarmId,
      topology: this.configuration.topology,
    });
  }

  /**
   * Initialize the swarm with force cleanup
   * Equivalent to: npx claude-flow@alpha init --force --sublinear --neural
   */
  async initialize(options: {
    force?: boolean;
    sublinear?: boolean;
    neural?: boolean;
  } = {}): Promise<void> {
    logger.info(`Initializing swarm`, { options });

    if (options.force) {
      // Environment sanitization - clean up existing state
      await this.cleanup();
    }

    // Initialize solver integrations
    if (options.sublinear) {
      logger.info('Sublinear solver integration enabled');
      // In production, this would set up the sublinear-time-solver MCP server
    }

    // Initialize neural capabilities
    if (options.neural) {
      logger.info('Neural capabilities enabled (ruv-fann, ruvector bindings)');
      // In production, this would initialize neural inference bindings
    }

    // Spawn the Queen agent for hierarchical topology
    if (this.configuration.topology === 'hierarchical') {
      await this.spawnQueen();
    }

    this.isRunning = true;
    this.emitEvent('agent_spawned', { swarmId: this.swarmId, status: 'initialized' });

    logger.info(`Swarm initialized successfully`, { agentCount: this.agents.size });
  }

  /**
   * Spawn a new agent in the swarm
   */
  async spawnAgent(
    role: AgentRole,
    nodeId: string,
    optimized: boolean = false
  ): Promise<Agent> {
    if (this.agents.size >= this.configuration.maxAgents) {
      throw new Error(`Maximum agent limit reached: ${this.configuration.maxAgents}`);
    }

    const agent = AgentFactory.create(role, nodeId);

    if (optimized) {
      // Agent Booster - compile to optimized WASM (simulated)
      logger.info(`Agent optimization enabled (352x speedup)`, {
        agentId: agent.identity.id,
      });
    }

    this.agents.set(agent.identity.id, agent);
    this.emitEvent('agent_spawned', {
      agentId: agent.identity.id,
      role,
      nodeId,
    });

    logger.info(`Agent spawned`, {
      agentId: agent.identity.id,
      role,
      totalAgents: this.agents.size,
    });

    return agent;
  }

  /**
   * Spawn the Queen/Coordinator agent
   */
  private async spawnQueen(): Promise<void> {
    const queen = AgentFactory.create('queen', 'central-unit');
    this.queenAgent = queen;
    this.agents.set(queen.identity.id, queen);

    logger.info(`Queen agent spawned`, { queenId: queen.identity.id });
  }

  /**
   * Submit a task to the swarm for execution
   */
  async submitTask(task: Omit<Task, 'id' | 'status'>): Promise<string> {
    const fullTask: Task = {
      ...task,
      id: uuidv4(),
      status: 'pending',
    };

    this.taskQueue.push(fullTask);
    this.processTaskQueue();

    return fullTask.id;
  }

  /**
   * Process the task queue
   */
  private async processTaskQueue(): Promise<void> {
    if (this.taskQueue.length === 0) return;

    // Sort by priority
    this.taskQueue.sort((a, b) => {
      const priorityOrder: Record<TaskPriority, number> = {
        critical: 0,
        high: 1,
        medium: 2,
        low: 3,
      };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });

    const task = this.taskQueue.shift()!;
    task.status = 'in_progress';

    // Find suitable agent
    const agent = this.findSuitableAgent(task);
    if (!agent) {
      logger.warn(`No suitable agent found for task`, { taskId: task.id });
      task.status = 'failed';
      return;
    }

    try {
      const result = await agent.execute(task);
      task.status = result.success ? 'completed' : 'failed';

      this.updateMetrics(result);
      this.emitEvent('task_completed', { taskId: task.id, result });
    } catch (error) {
      task.status = 'failed';
      logger.error(`Task execution error`, { taskId: task.id, error: String(error) });
    }
  }

  /**
   * Find a suitable agent for a task based on capabilities
   */
  private findSuitableAgent(task: Task): Agent | undefined {
    const requiredCapabilities = task.constraints.requiredCapabilities;

    for (const agent of this.agents.values()) {
      if (agent.state === 'idle' || agent.state === 'active') {
        const hasCapabilities = requiredCapabilities.every((cap) =>
          agent.capabilities.includes(cap)
        );
        if (hasCapabilities) {
          return agent;
        }
      }
    }

    return undefined;
  }

  /**
   * Broadcast a message to all agents
   */
  async broadcast(message: Record<string, unknown>): Promise<void> {
    logger.debug(`Broadcasting message to swarm`, { agentCount: this.agents.size });

    const promises = Array.from(this.agents.values()).map(async (agent) => {
      if (agent.state !== 'terminated') {
        // In production, this would use QUIC for low-latency communication
        agent.memory.shortTerm.set('broadcast', message);
      }
    });

    await Promise.all(promises);
  }

  /**
   * Federated learning: Submit gradient updates
   */
  submitGradientUpdate(update: GradientUpdate): void {
    if (!this.configuration.federationEnabled) {
      logger.warn('Federation is disabled');
      return;
    }

    this.gradientBuffer.push(update);

    // Aggregate when we have enough updates
    if (this.gradientBuffer.length >= 5) {
      this.aggregateGradients();
    }
  }

  /**
   * Aggregate gradients using Federated Averaging
   */
  private aggregateGradients(): void {
    const updates = this.gradientBuffer.splice(0, this.gradientBuffer.length);
    if (updates.length === 0) return;

    const modelId = updates[0].modelId;
    const totalSamples = updates.reduce((sum, u) => sum + u.sampleCount, 0);

    // Weighted average of gradients
    const gradientLength = updates[0].gradients.length;
    const aggregated = new Float32Array(gradientLength);

    for (const update of updates) {
      const weight = update.sampleCount / totalSamples;
      for (let i = 0; i < gradientLength; i++) {
        aggregated[i] += update.gradients[i] * weight;
      }
    }

    // Get or create model
    let model = this.federatedModels.get(modelId);
    if (!model) {
      model = {
        id: modelId,
        version: 0,
        weights: new Float32Array(gradientLength),
        aggregatedFrom: [],
        timestamp: Date.now(),
      };
    }

    // Apply gradients (simple gradient descent)
    const learningRate = 0.01;
    for (let i = 0; i < gradientLength; i++) {
      model.weights[i] -= learningRate * aggregated[i];
    }

    model.version++;
    model.aggregatedFrom = updates.map((u) => u.agentId);
    model.timestamp = Date.now();

    this.federatedModels.set(modelId, model);

    logger.info(`Federated model updated`, {
      modelId,
      version: model.version,
      contributors: updates.length,
    });
  }

  /**
   * Get current swarm state
   */
  getState(): SwarmState {
    return {
      id: this.swarmId,
      configuration: this.configuration,
      agents: this.agents,
      queenId: this.queenAgent?.identity.id,
      status: this.isRunning ? 'running' : 'stopped',
      metrics: this.metrics,
    };
  }

  /**
   * Get swarm metrics
   */
  getMetrics(): SwarmMetrics {
    return { ...this.metrics };
  }

  /**
   * Register an event handler
   */
  onEvent(handler: SwarmEventHandler): void {
    this.eventHandlers.push(handler);
  }

  /**
   * Emit a system event
   */
  private emitEvent(type: EventType, payload: Record<string, unknown>): void {
    const event: SystemEvent = {
      id: uuidv4(),
      type,
      timestamp: Date.now(),
      source: this.swarmId,
      payload,
    };

    for (const handler of this.eventHandlers) {
      try {
        handler.onEvent(event);
      } catch (error) {
        logger.error(`Event handler error`, { error: String(error) });
      }
    }
  }

  /**
   * Update metrics after task completion
   */
  private updateMetrics(result: TaskResult): void {
    if (result.success) {
      this.metrics.tasksCompleted++;
    } else {
      this.metrics.tasksFailed++;
    }

    const total = this.metrics.tasksCompleted + this.metrics.tasksFailed;
    this.metrics.successRate = this.metrics.tasksCompleted / total;

    // Update running average of latency
    this.metrics.averageLatencyMs =
      (this.metrics.averageLatencyMs * (total - 1) + result.executionTimeMs) / total;
  }

  /**
   * Initialize metrics
   */
  private initializeMetrics(): SwarmMetrics {
    return {
      activeAgents: 0,
      tasksCompleted: 0,
      tasksFailed: 0,
      averageLatencyMs: 0,
      successRate: 1,
      uptime: Date.now(),
    };
  }

  /**
   * Cleanup and reset the swarm
   */
  async cleanup(): Promise<void> {
    logger.info(`Cleaning up swarm`, { swarmId: this.swarmId });

    // Terminate all agents
    const terminationPromises = Array.from(this.agents.values()).map((agent) =>
      agent.terminate()
    );
    await Promise.all(terminationPromises);

    this.agents.clear();
    this.queenAgent = undefined;
    this.taskQueue = [];
    this.federatedModels.clear();
    this.gradientBuffer = [];
    this.isRunning = false;
  }

  /**
   * Graceful shutdown
   */
  async shutdown(): Promise<void> {
    logger.info(`Shutting down swarm`, { swarmId: this.swarmId });

    this.isRunning = false;
    await this.cleanup();

    logger.info(`Swarm shutdown complete`);
  }

  /**
   * Get agent by ID
   */
  getAgent(agentId: string): Agent | undefined {
    return this.agents.get(agentId);
  }

  /**
   * Get all agents of a specific role
   */
  getAgentsByRole(role: AgentRole): Agent[] {
    return Array.from(this.agents.values()).filter(
      (agent) => agent.identity.role === role
    );
  }

  /**
   * Terminate a specific agent
   */
  async terminateAgent(agentId: string): Promise<boolean> {
    const agent = this.agents.get(agentId);
    if (!agent) return false;

    await agent.terminate();
    this.agents.delete(agentId);

    this.emitEvent('agent_terminated', { agentId });
    logger.info(`Agent terminated`, { agentId });

    return true;
  }
}

/**
 * Sandbox Manager - Manages e2b sandbox environments
 */
export class SandboxManager {
  private activeSandboxes: Map<string, SandboxInstance>;

  constructor() {
    this.activeSandboxes = new Map();
  }

  /**
   * Create a new sandbox for an agent
   * Startup time ~150ms as per e2b specs
   */
  async createSandbox(agentId: string, config: SandboxConfig): Promise<SandboxInstance> {
    const startTime = performance.now();

    const sandbox: SandboxInstance = {
      id: uuidv4(),
      agentId,
      createdAt: Date.now(),
      status: 'running',
      resourceLimits: config.resourceLimits,
    };

    // Simulate e2b sandbox startup (~150ms)
    await new Promise((resolve) => setTimeout(resolve, 150));

    this.activeSandboxes.set(sandbox.id, sandbox);

    const startupTime = performance.now() - startTime;
    logger.debug(`Sandbox created`, { sandboxId: sandbox.id, startupTimeMs: startupTime });

    return sandbox;
  }

  /**
   * Terminate a sandbox
   */
  async terminateSandbox(sandboxId: string): Promise<void> {
    const sandbox = this.activeSandboxes.get(sandboxId);
    if (!sandbox) return;

    sandbox.status = 'terminated';
    this.activeSandboxes.delete(sandboxId);

    logger.debug(`Sandbox terminated`, { sandboxId });
  }

  /**
   * Get active sandbox count
   */
  getActiveSandboxCount(): number {
    return this.activeSandboxes.size;
  }
}

export interface SandboxConfig {
  resourceLimits: {
    cpuPercent: number;
    memoryMb: number;
    timeoutMs: number;
  };
}

export interface SandboxInstance {
  id: string;
  agentId: string;
  createdAt: number;
  status: 'running' | 'paused' | 'terminated';
  resourceLimits: SandboxConfig['resourceLimits'];
}

/**
 * Create and initialize a swarm with standard configuration
 */
export async function createSwarm(
  options: {
    topology?: SwarmTopology;
    force?: boolean;
    sublinear?: boolean;
    neural?: boolean;
  } = {}
): Promise<SwarmOrchestrator> {
  const swarm = new SwarmOrchestrator({
    topology: options.topology || 'hierarchical',
    federationEnabled: true,
    sandboxed: true,
  });

  await swarm.initialize({
    force: options.force ?? true,
    sublinear: options.sublinear ?? true,
    neural: options.neural ?? true,
  });

  return swarm;
}
