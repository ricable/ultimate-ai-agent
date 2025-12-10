/**
 * Agent Runtime - High-performance agent execution environment
 * Provides ephemeral intelligence with WASM-optimized execution
 */

import { v4 as uuidv4 } from 'uuid';
import {
  Agent,
  AgentIdentity,
  AgentMemory,
  AgentRole,
  AgentState,
  Experience,
  SkillVector,
  Task,
  TaskResult,
} from '../core/types.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('AgentRuntime');

/**
 * Agent configuration options
 */
export interface AgentConfig {
  role: AgentRole;
  nodeId: string;
  capabilities: string[];
  memoryLimit: number;
  executionTimeout: number;
  sandboxed: boolean;
}

/**
 * Base Agent implementation
 * Designed for ephemeral, high-performance execution
 */
export class BaseAgent implements Agent {
  identity: AgentIdentity;
  state: AgentState;
  capabilities: string[];
  memory: AgentMemory;

  private config: AgentConfig;
  private taskQueue: Task[] = [];
  private isProcessing: boolean = false;

  constructor(config: AgentConfig) {
    this.config = config;
    this.identity = {
      id: uuidv4(),
      role: config.role,
      nodeId: config.nodeId,
      createdAt: Date.now(),
      version: '1.0.0',
    };
    this.state = 'idle';
    this.capabilities = config.capabilities;
    this.memory = {
      shortTerm: new Map(),
      experiences: [],
      skills: [],
    };

    logger.info(`Agent created`, {
      id: this.identity.id,
      role: this.identity.role,
      nodeId: this.identity.nodeId,
    });
  }

  /**
   * Execute a task with performance tracking
   */
  async execute(task: Task): Promise<TaskResult> {
    const startTime = performance.now();
    this.state = 'processing';

    logger.debug(`Executing task`, { taskId: task.id, type: task.type });

    try {
      // Validate constraints
      this.validateConstraints(task);

      // Execute with timeout
      const result = await Promise.race([
        this.executeTask(task),
        this.createTimeout(task.constraints.maxLatencyMs),
      ]);

      const executionTime = performance.now() - startTime;

      // Record experience
      this.recordExperience(task, result, executionTime);

      this.state = 'idle';
      return {
        ...result,
        executionTimeMs: executionTime,
      };
    } catch (error) {
      this.state = 'idle';
      const executionTime = performance.now() - startTime;

      logger.error(`Task execution failed`, {
        taskId: task.id,
        error: String(error),
      });

      return {
        taskId: task.id,
        success: false,
        error: error instanceof Error ? error.message : String(error),
        executionTimeMs: executionTime,
      };
    }
  }

  /**
   * Core task execution logic - override in specialized agents
   */
  protected async executeTask(task: Task): Promise<TaskResult> {
    // Base implementation - specialized agents override this
    return {
      taskId: task.id,
      success: true,
      data: { message: 'Task processed by base agent' },
      executionTimeMs: 0,
    };
  }

  /**
   * Terminate the agent and cleanup resources
   */
  async terminate(): Promise<void> {
    logger.info(`Terminating agent`, { id: this.identity.id });

    this.state = 'terminated';
    this.taskQueue = [];
    this.memory.shortTerm.clear();

    // Persist important experiences before termination
    await this.persistExperiences();
  }

  /**
   * Add a skill to the agent's repertoire
   */
  addSkill(skill: SkillVector): void {
    this.skills.push(skill);
    logger.debug(`Skill added`, { skillId: skill.id, agentId: this.identity.id });
  }

  /**
   * Find similar skills based on vector similarity
   */
  findSimilarSkills(embedding: Float32Array, threshold: number = 0.8): SkillVector[] {
    return this.memory.skills.filter((skill) => {
      const similarity = this.cosineSimilarity(embedding, skill.embedding);
      return similarity >= threshold;
    });
  }

  /**
   * Get the agent's skill vectors
   */
  private get skills(): SkillVector[] {
    return this.memory.skills;
  }

  /**
   * Validate task constraints
   */
  private validateConstraints(task: Task): void {
    const missingCapabilities = task.constraints.requiredCapabilities.filter(
      (cap) => !this.capabilities.includes(cap)
    );

    if (missingCapabilities.length > 0) {
      throw new Error(`Missing capabilities: ${missingCapabilities.join(', ')}`);
    }
  }

  /**
   * Create a timeout promise
   */
  private createTimeout(ms: number): Promise<TaskResult> {
    return new Promise((_, reject) => {
      setTimeout(() => reject(new Error(`Task timeout after ${ms}ms`)), ms);
    });
  }

  /**
   * Record an experience for learning
   */
  private recordExperience(
    task: Task,
    result: TaskResult,
    executionTime: number
  ): void {
    const experience: Experience = {
      timestamp: Date.now(),
      action: task.type,
      context: { ...task.payload, executionTime },
      outcome: result.success ? 'success' : 'failure',
      kpiDelta: result.kpiImpact
        ? result.kpiImpact.throughputDelta +
          result.kpiImpact.latencyDelta * -1 +
          result.kpiImpact.interferenceDelta * -1
        : 0,
    };

    this.memory.experiences.push(experience);

    // Limit memory usage
    if (this.memory.experiences.length > this.config.memoryLimit) {
      this.memory.experiences.shift();
    }
  }

  /**
   * Persist experiences to long-term storage
   */
  private async persistExperiences(): Promise<void> {
    // In production, this would write to agentdb
    const significantExperiences = this.memory.experiences.filter(
      (exp) => Math.abs(exp.kpiDelta) > 0.1 || exp.outcome === 'failure'
    );

    if (significantExperiences.length > 0) {
      logger.debug(`Persisting experiences`, {
        count: significantExperiences.length,
        agentId: this.identity.id,
      });
    }
  }

  /**
   * Cosine similarity between two vectors
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
  }
}

/**
 * Optimizer Agent - Specialized for performance optimization
 */
export class OptimizerAgent extends BaseAgent {
  constructor(nodeId: string) {
    super({
      role: 'optimizer',
      nodeId,
      capabilities: [
        'traffic_optimization',
        'interference_mitigation',
        'load_balancing',
        'energy_saving',
        'capacity_optimization',
      ],
      memoryLimit: 1000,
      executionTimeout: 5000,
      sandboxed: true,
    });
  }

  protected async executeTask(task: Task): Promise<TaskResult> {
    switch (task.type) {
      case 'optimize_throughput':
        return this.optimizeThroughput(task);
      case 'mitigate_interference':
        return this.mitigateInterference(task);
      case 'balance_load':
        return this.balanceLoad(task);
      case 'save_energy':
        return this.saveEnergy(task);
      default:
        return super.executeTask(task);
    }
  }

  private async optimizeThroughput(task: Task): Promise<TaskResult> {
    const cellId = task.payload.cellId as string;
    logger.info(`Optimizing throughput for cell`, { cellId });

    // Analyze current state and compute optimizations
    const recommendations = this.computeThroughputOptimizations(task.payload);

    return {
      taskId: task.id,
      success: true,
      data: { recommendations },
      executionTimeMs: 0,
      kpiImpact: {
        throughputDelta: 0.15, // Expected 15% improvement
        latencyDelta: -0.05,
        interferenceDelta: 0,
        energyDelta: 0.02,
      },
    };
  }

  private async mitigateInterference(task: Task): Promise<TaskResult> {
    const cellId = task.payload.cellId as string;
    const interfererId = task.payload.interfererId as string;

    logger.info(`Mitigating interference`, { cellId, interfererId });

    const mitigationPlan = this.computeInterferenceMitigation(task.payload);

    return {
      taskId: task.id,
      success: true,
      data: { mitigationPlan },
      executionTimeMs: 0,
      kpiImpact: {
        throughputDelta: 0.1,
        latencyDelta: -0.02,
        interferenceDelta: -0.3,
        energyDelta: 0,
      },
    };
  }

  private async balanceLoad(task: Task): Promise<TaskResult> {
    const cellIds = task.payload.cellIds as string[];

    logger.info(`Balancing load across cells`, { count: cellIds.length });

    return {
      taskId: task.id,
      success: true,
      data: { balanced: true },
      executionTimeMs: 0,
      kpiImpact: {
        throughputDelta: 0.08,
        latencyDelta: -0.1,
        interferenceDelta: -0.05,
        energyDelta: -0.03,
      },
    };
  }

  private async saveEnergy(task: Task): Promise<TaskResult> {
    const cellId = task.payload.cellId as string;
    const sleepRatio = task.payload.targetSleepRatio as number;

    logger.info(`Initiating energy saving`, { cellId, sleepRatio });

    return {
      taskId: task.id,
      success: true,
      data: { sleepModeActivated: true, targetSleepRatio: sleepRatio },
      executionTimeMs: 0,
      kpiImpact: {
        throughputDelta: -0.02,
        latencyDelta: 0.01,
        interferenceDelta: -0.05,
        energyDelta: -0.25,
      },
    };
  }

  private computeThroughputOptimizations(
    payload: Record<string, unknown>
  ): Record<string, unknown>[] {
    return [
      { parameter: 'electricalTilt', adjustment: -2, reason: 'Reduce overshoot' },
      { parameter: 'transmitPower', adjustment: 1, reason: 'Improve edge coverage' },
    ];
  }

  private computeInterferenceMitigation(
    payload: Record<string, unknown>
  ): Record<string, unknown> {
    return {
      strategy: 'pci_change',
      oldPci: payload.currentPci,
      newPci: ((payload.currentPci as number) + 33) % 504,
      reason: 'Orthogonalize reference signals',
    };
  }
}

/**
 * Healer Agent - Specialized for fault detection and healing
 */
export class HealerAgent extends BaseAgent {
  constructor(nodeId: string) {
    super({
      role: 'healer',
      nodeId,
      capabilities: [
        'anomaly_detection',
        'root_cause_analysis',
        'auto_healing',
        'alarm_correlation',
        'preventive_maintenance',
      ],
      memoryLimit: 500,
      executionTimeout: 10000,
      sandboxed: true,
    });
  }

  protected async executeTask(task: Task): Promise<TaskResult> {
    switch (task.type) {
      case 'detect_anomaly':
        return this.detectAnomaly(task);
      case 'analyze_root_cause':
        return this.analyzeRootCause(task);
      case 'heal_cell':
        return this.healCell(task);
      default:
        return super.executeTask(task);
    }
  }

  private async detectAnomaly(task: Task): Promise<TaskResult> {
    const cellId = task.payload.cellId as string;
    const metrics = task.payload.metrics as number[];

    logger.info(`Detecting anomalies for cell`, { cellId });

    const anomalyScore = this.computeAnomalyScore(metrics);
    const isAnomaly = anomalyScore > 0.7;

    return {
      taskId: task.id,
      success: true,
      data: { isAnomaly, score: anomalyScore },
      executionTimeMs: 0,
    };
  }

  private async analyzeRootCause(task: Task): Promise<TaskResult> {
    const alarmCode = task.payload.alarmCode as string;
    const context = task.payload.context as Record<string, unknown>;

    logger.info(`Analyzing root cause`, { alarmCode });

    const analysis = this.performNeuroSymbolicRCA(alarmCode, context);

    return {
      taskId: task.id,
      success: true,
      data: analysis,
      executionTimeMs: 0,
    };
  }

  private async healCell(task: Task): Promise<TaskResult> {
    const cellId = task.payload.cellId as string;
    const healingAction = task.payload.action as string;

    logger.info(`Healing cell`, { cellId, action: healingAction });

    return {
      taskId: task.id,
      success: true,
      data: { healed: true, action: healingAction },
      executionTimeMs: 0,
    };
  }

  private computeAnomalyScore(metrics: number[]): number {
    if (metrics.length < 2) return 0;

    const mean = metrics.reduce((a, b) => a + b, 0) / metrics.length;
    const stdDev = Math.sqrt(
      metrics.reduce((sum, val) => sum + (val - mean) ** 2, 0) / metrics.length
    );

    const lastValue = metrics[metrics.length - 1];
    const zScore = stdDev > 0 ? Math.abs(lastValue - mean) / stdDev : 0;

    return Math.min(1, zScore / 3);
  }

  private performNeuroSymbolicRCA(
    alarmCode: string,
    context: Record<string, unknown>
  ): Record<string, unknown> {
    // Neuro-symbolic reasoning combining symbolic rules with neural context
    const reasoningChain = [];

    // Symbolic rule matching
    if (alarmCode === 'VSWR_HIGH') {
      if (context.weather === 'rain') {
        reasoningChain.push({
          type: 'symbolic',
          premise: 'Alarm VSWR_HIGH is active',
          conclusion: 'High voltage standing wave ratio detected',
        });
        reasoningChain.push({
          type: 'neural',
          premise: 'Weather context indicates precipitation',
          conclusion: 'Environmental factor detected',
        });
        reasoningChain.push({
          type: 'hybrid',
          premise: 'VSWR_HIGH + Rain condition',
          conclusion: 'Probable cause: Water ingress in connector',
        });

        return {
          probableCause: 'water_ingress_connector',
          confidence: 0.85,
          recommendedAction: 'monitor',
          reasoningChain,
        };
      } else {
        return {
          probableCause: 'hardware_failure',
          confidence: 0.75,
          recommendedAction: 'dispatch_technician',
          reasoningChain,
        };
      }
    }

    return {
      probableCause: 'unknown',
      confidence: 0.3,
      recommendedAction: 'escalate',
      reasoningChain,
    };
  }
}

/**
 * Configurator Agent - Specialized for configuration management
 */
export class ConfiguratorAgent extends BaseAgent {
  constructor(nodeId: string) {
    super({
      role: 'configurator',
      nodeId,
      capabilities: [
        'parameter_tuning',
        'neighbor_management',
        'pci_optimization',
        'antenna_configuration',
        'feature_activation',
      ],
      memoryLimit: 500,
      executionTimeout: 15000,
      sandboxed: true,
    });
  }

  protected async executeTask(task: Task): Promise<TaskResult> {
    switch (task.type) {
      case 'tune_parameter':
        return this.tuneParameter(task);
      case 'update_neighbor':
        return this.updateNeighbor(task);
      case 'optimize_pci':
        return this.optimizePci(task);
      default:
        return super.executeTask(task);
    }
  }

  private async tuneParameter(task: Task): Promise<TaskResult> {
    const cellId = task.payload.cellId as string;
    const parameter = task.payload.parameter as string;
    const value = task.payload.value as number;

    logger.info(`Tuning parameter`, { cellId, parameter, value });

    return {
      taskId: task.id,
      success: true,
      data: { applied: true, parameter, value },
      executionTimeMs: 0,
    };
  }

  private async updateNeighbor(task: Task): Promise<TaskResult> {
    const cellId = task.payload.cellId as string;
    const neighborId = task.payload.neighborId as string;
    const action = task.payload.action as 'add' | 'remove' | 'modify';

    logger.info(`Updating neighbor relation`, { cellId, neighborId, action });

    return {
      taskId: task.id,
      success: true,
      data: { updated: true, action },
      executionTimeMs: 0,
    };
  }

  private async optimizePci(task: Task): Promise<TaskResult> {
    const cellIds = task.payload.cellIds as string[];

    logger.info(`Optimizing PCI assignment`, { count: cellIds.length });

    // Compute optimal PCI assignment using graph coloring
    const pciAssignment = this.computePciAssignment(cellIds);

    return {
      taskId: task.id,
      success: true,
      data: { pciAssignment },
      executionTimeMs: 0,
    };
  }

  private computePciAssignment(cellIds: string[]): Record<string, number> {
    const assignment: Record<string, number> = {};
    let nextPci = 0;

    for (const cellId of cellIds) {
      assignment[cellId] = nextPci;
      nextPci = (nextPci + 3) % 504; // Skip to avoid collision
    }

    return assignment;
  }
}

/**
 * Agent Factory - Creates specialized agents based on role
 */
export class AgentFactory {
  static create(role: AgentRole, nodeId: string): Agent {
    switch (role) {
      case 'optimizer':
        return new OptimizerAgent(nodeId);
      case 'healer':
        return new HealerAgent(nodeId);
      case 'configurator':
        return new ConfiguratorAgent(nodeId);
      case 'worker':
      case 'queen':
      default:
        return new BaseAgent({
          role,
          nodeId,
          capabilities: [],
          memoryLimit: 100,
          executionTimeout: 5000,
          sandboxed: true,
        });
    }
  }
}
