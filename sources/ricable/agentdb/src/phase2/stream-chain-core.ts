/**
 * Stream-Chain Core Infrastructure for Multi-Agent ML Workflows
 * Phase 2: Reinforcement Learning and Causal Inference Coordination
 */

import { EventEmitter } from 'events';
import { Transform, Readable, Writable } from 'stream';
import { WebSocket } from 'ws';
import { AgentDB } from '../agentdb/agentdb-core';
import { TemporalReasoningCore } from '../temporal/temporal-core';

// Core Stream Pipeline Interfaces
export interface StreamPipeline {
  id: string;
  name: string;
  type: StreamType;
  status: StreamStatus;
  throughput: number;
  latency: number;
  errorRate: number;
  createdAt: Date;
  updatedAt: Date;
}

export enum StreamType {
  ML_TRAINING = 'ml_training',
  CAUSAL_INFERENCE = 'causal_inference',
  MULTI_AGENT = 'multi_agent',
  REAL_TIME_OPTIMIZATION = 'real_time_optimization',
  MEMORY_COORDINATION = 'memory_coordination',
  PERFORMANCE_MONITORING = 'performance_monitoring'
}

export enum StreamStatus {
  INITIALIZING = 'initializing',
  RUNNING = 'running',
  PAUSED = 'paused',
  STOPPED = 'stopped',
  ERROR = 'error'
}

export interface StreamStep {
  id: string;
  name: string;
  type: StepType;
  processor: StreamProcessor;
  dependencies: string[];
  parallelism: number;
  retryPolicy: RetryPolicy;
  performance: StepPerformance;
}

export enum StepType {
  TRANSFORM = 'transform',
  FILTER = 'filter',
  AGGREGATE = 'aggregate',
  VALIDATE = 'validate',
  STORE = 'store',
  DISTRIBUTE = 'distribute',
  MONITOR = 'monitor'
}

export interface StreamProcessor {
  process(data: any, context: StreamContext): Promise<any>;
  initialize?(config: any): Promise<void>;
  cleanup?(): Promise<void>;
  healthCheck?(): Promise<boolean>;
}

export interface StreamContext {
  pipelineId: string;
  stepId: string;
  agentId: string;
  timestamp: Date;
  correlationId: string;
  metadata: Map<string, any>;
  memory: AgentDB;
  temporal: TemporalReasoningCore;
}

export interface RetryPolicy {
  maxAttempts: number;
  backoffMs: number;
  maxBackoffMs: number;
  retryableErrors: string[];
}

export interface StepPerformance {
  avgProcessingTime: number;
  throughput: number;
  errorRate: number;
  lastProcessed: Date;
  totalProcessed: number;
}

// Main Stream-Chain Builder
export class StreamChain {
  private static instance: StreamChain;
  private pipelines: Map<string, StreamPipeline> = new Map();
  private steps: Map<string, Map<string, StreamStep>> = new Map(); // pipelineId -> stepId -> step
  private eventBus: EventEmitter;
  private agentDB: AgentDB;
  private temporalCore: TemporalReasoningCore;

  constructor(agentDB: AgentDB, temporalCore: TemporalReasoningCore) {
    this.agentDB = agentDB;
    this.temporalCore = temporalCore;
    this.eventBus = new EventEmitter();
    this.setupEventHandlers();
  }

  static getInstance(agentDB: AgentDB, temporalCore: TemporalReasoningCore): StreamChain {
    if (!StreamChain.instance) {
      StreamChain.instance = new StreamChain(agentDB, temporalCore);
    }
    return StreamChain.instance;
  }

  // Builder Pattern for Pipeline Construction
  static builder(): StreamPipelineBuilder {
    return new StreamPipelineBuilder();
  }

  // Sequential Pipeline Creation
  static sequential(...processors: StreamProcessor[]): StreamPipeline {
    const builder = new StreamPipelineBuilder();
    processors.forEach((processor, index) => {
      builder.addStep(`step-${index}`, processor);
    });
    return builder.build();
  }

  // Parallel Pipeline Creation
  static parallel(pipelines: StreamPipeline[]): StreamPipeline {
    const builder = new StreamPipelineBuilder();
    builder.setType(StreamType.MULTI_AGENT);
    pipelines.forEach((pipeline, index) => {
      builder.addSubPipeline(`parallel-${index}`, pipeline);
    });
    return builder.build();
  }

  // Pipeline Management
  async createPipeline(config: PipelineConfig): Promise<StreamPipeline> {
    const pipeline: StreamPipeline = {
      id: config.id || this.generateId(),
      name: config.name,
      type: config.type,
      status: StreamStatus.INITIALIZING,
      throughput: 0,
      latency: 0,
      errorRate: 0,
      createdAt: new Date(),
      updatedAt: new Date()
    };

    this.pipelines.set(pipeline.id, pipeline);
    this.steps.set(pipeline.id, new Map());

    // Initialize steps
    for (const stepConfig of config.steps) {
      await this.addStep(pipeline.id, stepConfig);
    }

    this.eventBus.emit('pipeline:created', pipeline);
    return pipeline;
  }

  async addStep(pipelineId: string, stepConfig: StepConfig): Promise<StreamStep> {
    const step: StreamStep = {
      id: stepConfig.id || this.generateId(),
      name: stepConfig.name,
      type: stepConfig.type,
      processor: stepConfig.processor,
      dependencies: stepConfig.dependencies || [],
      parallelism: stepConfig.parallelism || 1,
      retryPolicy: stepConfig.retryPolicy || this.getDefaultRetryPolicy(),
      performance: {
        avgProcessingTime: 0,
        throughput: 0,
        errorRate: 0,
        lastProcessed: new Date(),
        totalProcessed: 0
      }
    };

    const pipelineSteps = this.steps.get(pipelineId);
    if (pipelineSteps) {
      pipelineSteps.set(step.id, step);
      await step.processor.initialize?.(stepConfig.config);
      this.eventBus.emit('step:added', { pipelineId, step });
    }

    return step;
  }

  async executePipeline(pipelineId: string, inputData: any): Promise<any> {
    const pipeline = this.pipelines.get(pipelineId);
    if (!pipeline) {
      throw new Error(`Pipeline ${pipelineId} not found`);
    }

    const steps = this.steps.get(pipelineId);
    if (!steps || steps.size === 0) {
      throw new Error(`No steps found for pipeline ${pipelineId}`);
    }

    pipeline.status = StreamStatus.RUNNING;
    pipeline.updatedAt = new Date();

    try {
      const context: StreamContext = {
        pipelineId,
        stepId: '',
        agentId: this.getCurrentAgentId(),
        timestamp: new Date(),
        correlationId: this.generateCorrelationId(),
        metadata: new Map(),
        memory: this.agentDB,
        temporal: this.temporalCore
      };

      const result = await this.executeStepsSequentially(
        Array.from(steps.values()),
        inputData,
        context
      );

      pipeline.status = StreamStatus.RUNNING;
      this.eventBus.emit('pipeline:completed', { pipelineId, result });

      return result;
    } catch (error) {
      pipeline.status = StreamStatus.ERROR;
      pipeline.errorRate = Math.min(pipeline.errorRate + 0.01, 1.0);
      this.eventBus.emit('pipeline:error', { pipelineId, error });
      throw error;
    }
  }

  private async executeStepsSequentially(
    steps: StreamStep[],
    data: any,
    context: StreamContext
  ): Promise<any> {
    let currentData = data;

    for (const step of steps) {
      if (!this.canExecuteStep(step, context)) {
        continue;
      }

      const startTime = Date.now();
      context.stepId = step.id;

      try {
        currentData = await this.executeStepWithRetry(step, currentData, context);

        // Update step performance
        const processingTime = Date.now() - startTime;
        this.updateStepPerformance(step, processingTime, true);

        // Store intermediate results in AgentDB
        await this.storeIntermediateResult(step, currentData, context);

      } catch (error) {
        const processingTime = Date.now() - startTime;
        this.updateStepPerformance(step, processingTime, false);
        throw error;
      }
    }

    return currentData;
  }

  private async executeStepWithRetry(
    step: StreamStep,
    data: any,
    context: StreamContext
  ): Promise<any> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= step.retryPolicy.maxAttempts; attempt++) {
      try {
        return await step.processor.process(data, context);
      } catch (error) {
        lastError = error as Error;

        if (!this.isRetryableError(error, step.retryPolicy)) {
          throw error;
        }

        if (attempt < step.retryPolicy.maxAttempts) {
          const backoff = Math.min(
            step.retryPolicy.backoffMs * Math.pow(2, attempt - 1),
            step.retryPolicy.maxBackoffMs
          );
          await this.sleep(backoff);
        }
      }
    }

    throw lastError;
  }

  private canExecuteStep(step: StreamStep, context: StreamContext): boolean {
    // Check if all dependencies are satisfied
    // This is a simplified implementation - in practice, you'd track completion status
    return true;
  }

  private async storeIntermediateResult(
    step: StreamStep,
    data: any,
    context: StreamContext
  ): Promise<void> {
    try {
      const resultKey = `pipeline:${context.pipelineId}:step:${step.id}:result:${context.correlationId}`;
      await this.agentDB.store(resultKey, {
        data,
        timestamp: context.timestamp,
        agentId: context.agentId,
        processingTime: step.performance.avgProcessingTime
      });
    } catch (error) {
      console.warn('Failed to store intermediate result:', error);
    }
  }

  private updateStepPerformance(
    step: StreamStep,
    processingTime: number,
    success: boolean
  ): void {
    step.performance.totalProcessed++;
    step.performance.avgProcessingTime =
      (step.performance.avgProcessingTime * (step.performance.totalProcessed - 1) + processingTime) /
      step.performance.totalProcessed;

    if (!success) {
      step.performance.errorRate =
        (step.performance.errorRate * (step.performance.totalProcessed - 1) + 1) /
        step.performance.totalProcessed;
    } else {
      step.performance.errorRate =
        (step.performance.errorRate * (step.performance.totalProcessed - 1)) /
        step.performance.totalProcessed;
    }

    step.performance.lastProcessed = new Date();
    step.performance.throughput = 1000 / processingTime; // items per second
  }

  private isRetryableError(error: Error, retryPolicy: RetryPolicy): boolean {
    return retryPolicy.retryableErrors.some(errorType =>
      error.name.includes(errorType) || error.message.includes(errorType)
    );
  }

  private setupEventHandlers(): void {
    this.eventBus.on('pipeline:created', (pipeline) => {
      console.log(`Pipeline created: ${pipeline.name} (${pipeline.id})`);
    });

    this.eventBus.on('step:added', ({ pipelineId, step }) => {
      console.log(`Step added to pipeline ${pipelineId}: ${step.name}`);
    });

    this.eventBus.on('pipeline:completed', ({ pipelineId, result }) => {
      console.log(`Pipeline completed: ${pipelineId}`);
    });

    this.eventBus.on('pipeline:error', ({ pipelineId, error }) => {
      console.error(`Pipeline error: ${pipelineId}`, error);
    });
  }

  // Utility Methods
  private generateId(): string {
    return Math.random().toString(36).substr(2, 9);
  }

  private generateCorrelationId(): string {
    return `corr_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
  }

  private getCurrentAgentId(): string {
    return 'agent_' + process.env.AGENT_ID || 'unknown';
  }

  private getDefaultRetryPolicy(): RetryPolicy {
    return {
      maxAttempts: 3,
      backoffMs: 1000,
      maxBackoffMs: 10000,
      retryableErrors: ['NetworkError', 'TimeoutError', 'TemporaryError']
    };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Pipeline Monitoring and Management
  getPipeline(pipelineId: string): StreamPipeline | undefined {
    return this.pipelines.get(pipelineId);
  }

  getAllPipelines(): StreamPipeline[] {
    return Array.from(this.pipelines.values());
  }

  getPipelineSteps(pipelineId: string): StreamStep[] {
    const steps = this.steps.get(pipelineId);
    return steps ? Array.from(steps.values()) : [];
  }

  async pausePipeline(pipelineId: string): Promise<void> {
    const pipeline = this.pipelines.get(pipelineId);
    if (pipeline) {
      pipeline.status = StreamStatus.PAUSED;
      pipeline.updatedAt = new Date();
      this.eventBus.emit('pipeline:paused', { pipelineId });
    }
  }

  async resumePipeline(pipelineId: string): Promise<void> {
    const pipeline = this.pipelines.get(pipelineId);
    if (pipeline) {
      pipeline.status = StreamStatus.RUNNING;
      pipeline.updatedAt = new Date();
      this.eventBus.emit('pipeline:resumed', { pipelineId });
    }
  }

  async stopPipeline(pipelineId: string): Promise<void> {
    const pipeline = this.pipelines.get(pipelineId);
    if (pipeline) {
      pipeline.status = StreamStatus.STOPPED;
      pipeline.updatedAt = new Date();
      this.eventBus.emit('pipeline:stopped', { pipelineId });
    }
  }

  // Performance Analytics
  getPipelinePerformance(pipelineId: string): PipelinePerformance | undefined {
    const pipeline = this.pipelines.get(pipelineId);
    const steps = this.steps.get(pipelineId);

    if (!pipeline || !steps) {
      return undefined;
    }

    const stepPerformances = Array.from(steps.values()).map(step => step.performance);
    const avgStepLatency = stepPerformances.reduce((sum, perf) => sum + perf.avgProcessingTime, 0) / stepPerformances.length;
    const totalThroughput = stepPerformances.reduce((sum, perf) => sum + perf.throughput, 0);
    const avgErrorRate = stepPerformances.reduce((sum, perf) => sum + perf.errorRate, 0) / stepPerformances.length;

    return {
      pipelineId,
      pipelineName: pipeline.name,
      status: pipeline.status,
      totalSteps: steps.size,
      avgStepLatency,
      totalThroughput,
      avgErrorRate,
      lastUpdated: pipeline.updatedAt
    };
  }
}

// Supporting Interfaces
export interface PipelineConfig {
  id?: string;
  name: string;
  type: StreamType;
  steps: StepConfig[];
}

export interface StepConfig {
  id?: string;
  name: string;
  type: StepType;
  processor: StreamProcessor;
  dependencies?: string[];
  parallelism?: number;
  retryPolicy?: RetryPolicy;
  config?: any;
}

export interface PipelinePerformance {
  pipelineId: string;
  pipelineName: string;
  status: StreamStatus;
  totalSteps: number;
  avgStepLatency: number;
  totalThroughput: number;
  avgErrorRate: number;
  lastUpdated: Date;
}

// Pipeline Builder
export class StreamPipelineBuilder {
  private config: Partial<PipelineConfig> = {};
  private stepConfigs: StepConfig[] = [];

  setId(id: string): StreamPipelineBuilder {
    this.config.id = id;
    return this;
  }

  setName(name: string): StreamPipelineBuilder {
    this.config.name = name;
    return this;
  }

  setType(type: StreamType): StreamPipelineBuilder {
    this.config.type = type;
    return this;
  }

  addStep(id: string, processor: StreamProcessor, type?: StepType): StreamPipelineBuilder {
    this.stepConfigs.push({
      id,
      name: id,
      type: type || StepType.TRANSFORM,
      processor
    });
    return this;
  }

  addSubPipeline(id: string, pipeline: StreamPipeline): StreamPipelineBuilder {
    // Convert pipeline to sub-pipeline processor
    const processor: StreamProcessor = {
      process: async (data, context) => {
        return await StreamChain.getInstance(context.memory, context.temporal)
          .executePipeline(pipeline.id, data);
      }
    };

    this.stepConfigs.push({
      id,
      name: id,
      type: StepType.DISTRIBUTE,
      processor
    });
    return this;
  }

  build(): StreamPipeline {
    if (!this.config.name || !this.config.type) {
      throw new Error('Pipeline name and type are required');
    }

    const finalConfig: PipelineConfig = {
      ...this.config as PipelineConfig,
      steps: this.stepConfigs
    };

    // Return a placeholder pipeline - actual creation happens via StreamChain.createPipeline
    return {
      id: this.config.id || '',
      name: finalConfig.name,
      type: finalConfig.type,
      status: StreamStatus.INITIALIZING,
      throughput: 0,
      latency: 0,
      errorRate: 0,
      createdAt: new Date(),
      updatedAt: new Date()
    };
  }
}

// Stream Transform Utilities
export class StreamMap {
  static transform<T, R>(mapper: (item: T, context: StreamContext) => Promise<R>): StreamProcessor {
    return {
      process: async (data: T[], context: StreamContext): Promise<R[]> => {
        if (!Array.isArray(data)) {
          return await mapper(data, context);
        }

        const results = await Promise.all(
          data.map(item => mapper(item, context))
        );
        return results;
      }
    };
  }
}

export class StreamFilter {
  static filter<T>(predicate: (item: T, context: StreamContext) => Promise<boolean>): StreamProcessor {
    return {
      process: async (data: T[], context: StreamContext): Promise<T[]> => {
        if (!Array.isArray(data)) {
          return (await predicate(data, context)) ? [data] : [];
        }

        const results = await Promise.all(
          data.map(async item => ({
            item,
            keep: await predicate(item, context)
          }))
        );

        return results.filter(result => result.keep).map(result => result.item);
      }
    };
  }
}

export class StreamFlatMap {
  static flatMap<T, R>(mapper: (item: T, context: StreamContext) => Promise<R[]>): StreamProcessor {
    return {
      process: async (data: T[], context: StreamContext): Promise<R[]> => {
        if (!Array.isArray(data)) {
          return await mapper(data, context);
        }

        const results = await Promise.all(
          data.map(item => mapper(item, context))
        );

        return results.flat();
      }
    };
  }
}

export class StreamReduce {
  static reduce<T, R>(
    reducer: (acc: R, item: T, context: StreamContext) => Promise<R>,
    initialValue: R
  ): StreamProcessor {
    return {
      process: async (data: T[], context: StreamContext): Promise<R> => {
        if (!Array.isArray(data)) {
          return await reducer(initialValue, data, context);
        }

        let accumulator = initialValue;
        for (const item of data) {
          accumulator = await reducer(accumulator, item, context);
        }
        return accumulator;
      }
    };
  }
}

export default StreamChain;