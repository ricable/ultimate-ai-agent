/**
 * Distributed Training Coordinator for Swarm-Based ML Training
 *
 * Coordinates distributed reinforcement learning, causal inference, and DSPy
 * optimization across multiple agents with hierarchical topology management.
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';

// ============================================================================
// Interfaces
// ============================================================================

export interface DistributedTrainingConfig {
  maxWorkers: number;
  synchronizationStrategy: 'synchronous' | 'asynchronous' | 'semi_synchronous';
  aggregationStrategy: 'federated' | 'parameter_server' | 'peer_to_peer';
  batchSize: number;
  learningRate: number;
  communicationFrequency: number;
  compressionEnabled: boolean;
  checkpointInterval: number;
  faultToleranceEnabled: boolean;
}

export interface TrainingJob {
  id: string;
  type: TrainingJobType;
  config: TrainingJobConfig;
  status: TrainingJobStatus;
  assignedWorkers: string[];
  progress: TrainingProgress;
  startTime: Date;
  endTime?: Date;
  metrics: TrainingMetrics;
}

export enum TrainingJobType {
  REINFORCEMENT_LEARNING = 'reinforcement_learning',
  CAUSAL_INFERENCE = 'causal_inference',
  DSPY_OPTIMIZATION = 'dspy_optimization',
  HYBRID_TRAINING = 'hybrid_training'
}

export enum TrainingJobStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export interface TrainingJobConfig {
  algorithm: string;
  hyperparameters: Record<string, any>;
  dataset: DatasetConfig;
  environment: EnvironmentConfig;
  distributedConfig: DistributedTrainingConfig;
  resources: ResourceRequirements;
}

export interface DatasetConfig {
  type: 'ran_metrics' | 'causal_data' | 'dspy_examples';
  source: string;
  partitions: number;
  validationSplit: number;
  preprocessing: PreprocessingConfig;
}

export interface EnvironmentConfig {
  type: 'simulation' | 'emulation' | 'production';
  parameters: Record<string, any>;
  constraints: EnvironmentConstraints;
}

export interface ResourceRequirements {
  cpu: number;
  memory: number;
  gpu?: number;
  storage: number;
  networkBandwidth: number;
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  stepsCompleted: number;
  totalSteps: number;
  currentLoss: number;
  bestLoss: number;
  convergenceMetrics: ConvergenceMetrics;
}

export interface TrainingMetrics {
  loss: number[];
  accuracy: number[];
  throughput: number[];
  latency: number[];
  resourceUtilization: ResourceUtilization[];
  communicationOverhead: CommunicationMetrics;
}

export interface WorkerNode {
  id: string;
  address: string;
  status: WorkerStatus;
  capabilities: WorkerCapabilities;
  currentLoad: WorkerLoad;
  performance: WorkerPerformance;
  lastHeartbeat: Date;
}

export enum WorkerStatus {
  AVAILABLE = 'available',
  BUSY = 'busy',
  MAINTENANCE = 'maintenance',
  FAILED = 'failed'
}

export interface WorkerCapabilities {
  supportedAlgorithms: string[];
  maxMemory: number;
  hasGPU: boolean;
  gpuMemory?: number;
  supportedDataTypes: string[];
  maxConcurrency: number;
}

export interface WorkerLoad {
  currentJobs: number;
  maxJobs: number;
  cpuUtilization: number;
  memoryUtilization: number;
  gpuUtilization?: number;
  networkUtilization: number;
}

export interface WorkerPerformance {
  avgResponseTime: number;
  avgThroughput: number;
  errorRate: number;
  reliabilityScore: number;
  lastUpdated: Date;
}

export interface GradientAggregation {
  gradients: Map<string, number[]>;
  metadata: AggregationMetadata;
  timestamp: Date;
  sourceWorker: string;
  compressionRatio?: number;
}

export interface AggregationMetadata {
  algorithmVersion: string;
  modelVersion: string;
  stepNumber: number;
  batchSize: number;
  learningRate: number;
  weightDecay: number;
}

export interface ParameterSync {
  parameters: Map<string, number[]>;
  version: number;
  timestamp: Date;
  sourceNodeId: string;
  targetNodes: string[];
  checksum: string;
}

export interface ConvergenceMetrics {
  gradientNorm: number;
  parameterChanges: number[];
  lossImprovement: number;
  validationScore: number;
  earlyStoppingPatience: number;
}

export interface CommunicationMetrics {
  bytesTransmitted: number;
  bytesReceived: number;
  messagesSent: number;
  messagesReceived: number;
  averageLatency: number;
  compressionRatio: number;
  retransmissions: number;
}

// ============================================================================
// Distributed Training Coordinator
// ============================================================================

export class DistributedTrainingCoordinator extends EventEmitter {
  private workers: Map<string, WorkerNode>;
  private trainingJobs: Map<string, TrainingJob>;
  private topologyManager: TopologyManager;
  private loadBalancer: LoadBalancer;
  private gradientAggregator: GradientAggregator;
  private parameterSyncer: ParameterSyncer;
  private checkpointManager: CheckpointManager;
  private faultToleranceManager: FaultToleranceManager;
  private metricsCollector: MetricsCollector;
  private logger: Logger;
  private config: DistributedTrainingConfig;

  constructor(config: DistributedTrainingConfig, logger: Logger) {
    super();
    this.config = config;
    this.logger = logger;
    this.workers = new Map();
    this.trainingJobs = new Map();

    this.initializeComponents();
    this.setupEventHandlers();
  }

  private initializeComponents(): void {
    this.topologyManager = new HierarchicalTopologyManager(this.config);
    this.loadBalancer = new AdaptiveLoadBalancer();
    this.gradientAggregator = new FederatedGradientAggregator();
    this.parameterSyncer = new ParameterSyncManager();
    this.checkpointManager = new DistributedCheckpointManager();
    this.faultToleranceManager = new FaultToleranceManager();
    this.metricsCollector = new TrainingMetricsCollector();
  }

  private setupEventHandlers(): void {
    this.topologyManager.on('topology_changed', (topology: Topology) => {
      this.logger.info('Training topology changed:', topology);
      this.emit('topology_changed', topology);
    });

    this.loadBalancer.on('rebalance_required', (reason: string) => {
      this.logger.info(`Load rebalance required: ${reason}`);
      this.rebalanceWorkers();
    });

    this.gradientAggregator.on('aggregation_complete', (aggregation: GradientAggregation) => {
      this.handleGradientAggregation(aggregation);
    });

    this.faultToleranceManager.on('worker_failed', (workerId: string) => {
      this.handleWorkerFailure(workerId);
    });
  }

  // ============================================================================
  // Public API
  // ============================================================================

  /**
   * Submit a distributed training job
   *
   * @param jobConfig Training job configuration
   * @returns Training job ID
   */
  public async submitTrainingJob(jobConfig: TrainingJobConfig): Promise<string> {
    const jobId = this.generateJobId();
    const job: TrainingJob = {
      id: jobId,
      type: this.determineJobType(jobConfig),
      config: jobConfig,
      status: TrainingJobStatus.PENDING,
      assignedWorkers: [],
      progress: {
        epoch: 0,
        totalEpochs: jobConfig.hyperparameters.epochs || 100,
        stepsCompleted: 0,
        totalSteps: 0,
        currentLoss: Infinity,
        bestLoss: Infinity,
        convergenceMetrics: this.initializeConvergenceMetrics()
      },
      startTime: new Date(),
      metrics: this.initializeTrainingMetrics()
    };

    this.trainingJobs.set(jobId, job);

    try {
      // Schedule job execution
      await this.scheduleTrainingJob(job);
      this.logger.info(`Training job submitted successfully: ${jobId}`);
      return jobId;
    } catch (error) {
      this.trainingJobs.delete(jobId);
      this.logger.error(`Failed to submit training job: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get training job status and progress
   *
   * @param jobId Training job ID
   * @returns Training job details
   */
  public async getJobStatus(jobId: string): Promise<TrainingJob> {
    const job = this.trainingJobs.get(jobId);
    if (!job) {
      throw new Error(`Training job not found: ${jobId}`);
    }

    // Update progress from workers
    await this.updateJobProgress(job);

    return { ...job };
  }

  /**
   * Cancel a running training job
   *
   * @param jobId Training job ID
   */
  public async cancelTrainingJob(jobId: string): Promise<void> {
    const job = this.trainingJobs.get(jobId);
    if (!job) {
      throw new Error(`Training job not found: ${jobId}`);
    }

    if (job.status !== TrainingJobStatus.RUNNING) {
      throw new Error(`Cannot cancel job in status: ${job.status}`);
    }

    // Notify all assigned workers to stop
    await this.notifyWorkersToStop(job);

    // Update job status
    job.status = TrainingJobStatus.CANCELLED;
    job.endTime = new Date();

    this.logger.info(`Training job cancelled: ${jobId}`);
    this.emit('job_cancelled', job);
  }

  /**
   * Register a new worker node
   *
   * @param workerInfo Worker node information
   * @returns Worker ID
   */
  public async registerWorker(workerInfo: WorkerRegistrationInfo): Promise<string> {
    const workerId = this.generateWorkerId();
    const worker: WorkerNode = {
      id: workerId,
      address: workerInfo.address,
      status: WorkerStatus.AVAILABLE,
      capabilities: workerInfo.capabilities,
      currentLoad: {
        currentJobs: 0,
        maxJobs: workerInfo.capabilities.maxConcurrency,
        cpuUtilization: 0,
        memoryUtilization: 0,
        networkUtilization: 0
      },
      performance: {
        avgResponseTime: 0,
        avgThroughput: 0,
        errorRate: 0,
        reliabilityScore: 1.0,
        lastUpdated: new Date()
      },
      lastHeartbeat: new Date()
    };

    this.workers.set(workerId, worker);
    await this.topologyManager.addWorker(worker);

    this.logger.info(`Worker registered successfully: ${workerId}`);
    this.emit('worker_registered', worker);

    return workerId;
  }

  /**
   * Unregister a worker node
   *
   * @param workerId Worker ID
   */
  public async unregisterWorker(workerId: string): Promise<void> {
    const worker = this.workers.get(workerId);
    if (!worker) {
      throw new Error(`Worker not found: ${workerId}`);
    }

    // Check if worker has active jobs
    if (worker.currentLoad.currentJobs > 0) {
      await this.migrateWorkerJobs(worker);
    }

    this.workers.delete(workerId);
    await this.topologyManager.removeWorker(workerId);

    this.logger.info(`Worker unregistered: ${workerId}`);
    this.emit('worker_unregistered', workerId);
  }

  /**
   * Get comprehensive training metrics
   *
   * @param jobId Training job ID (optional)
   * @returns Training metrics
   */
  public getTrainingMetrics(jobId?: string): Promise<TrainingMetrics> {
    if (jobId) {
      const job = this.trainingJobs.get(jobId);
      if (!job) {
        throw new Error(`Training job not found: ${jobId}`);
      }
      return Promise.resolve(job.metrics);
    }

    return this.metricsCollector.getAggregatedMetrics();
  }

  /**
   * Optimize training configuration based on current performance
   *
   * @param jobId Training job ID
   */
  public async optimizeTraining(jobId: string): Promise<OptimizationRecommendation[]> {
    const job = this.trainingJobs.get(jobId);
    if (!job) {
      throw new Error(`Training job not found: ${jobId}`);
    }

    const optimizer = new TrainingOptimizer();
    const recommendations = await optimizer.generateRecommendations(job);

    // Apply auto-optimizations if enabled
    if (this.config.autoOptimization) {
      await this.applyOptimizations(job, recommendations.filter(r => r.autoApply));
    }

    return recommendations;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private async scheduleTrainingJob(job: TrainingJob): Promise<void> {
    // Determine optimal workers for this job
    const selectedWorkers = await this.selectOptimalWorkers(job);

    // Assign workers to job
    job.assignedWorkers = selectedWorkers.map(w => w.id);

    // Update worker loads
    selectedWorkers.forEach(worker => {
      worker.currentLoad.currentJobs++;
      worker.status = WorkerStatus.BUSY;
    });

    // Create training topology
    const topology = await this.topologyManager.createTrainingTopology(job);

    // Initialize parameter synchronization
    await this.parameterSyncer.initializeJob(job, topology);

    // Start training on assigned workers
    await this.startTrainingOnWorkers(job, topology);

    // Update job status
    job.status = TrainingJobStatus.RUNNING;

    this.emit('job_started', job);
  }

  private async selectOptimalWorkers(job: TrainingJob): Promise<WorkerNode[]> {
    const availableWorkers = Array.from(this.workers.values())
      .filter(w => w.status === WorkerStatus.AVAILABLE)
      .filter(w => this.isWorkerCompatible(w, job.config));

    return this.loadBalancer.selectWorkers(
      availableWorkers,
      job.config.distributedConfig.maxWorkers,
      job.config.resources
    );
  }

  private isWorkerCompatible(worker: WorkerNode, config: TrainingJobConfig): boolean {
    // Check algorithm compatibility
    if (!worker.capabilities.supportedAlgorithms.includes(config.algorithm)) {
      return false;
    }

    // Check resource requirements
    if (worker.capabilities.maxMemory < config.resources.memory) {
      return false;
    }

    if (config.resources.gpu && !worker.capabilities.hasGPU) {
      return false;
    }

    return true;
  }

  private async startTrainingOnWorkers(job: TrainingJob, topology: Topology): Promise<void> {
    const promises = job.assignedWorkers.map(async (workerId, index) => {
      const worker = this.workers.get(workerId);
      const workerConfig = this.generateWorkerConfig(job, index, topology);

      return this.sendTrainingCommand(worker, workerConfig);
    });

    await Promise.all(promises);
  }

  private generateWorkerConfig(job: TrainingJob, workerIndex: number, topology: Topology): WorkerTrainingConfig {
    return {
      jobId: job.id,
      workerIndex,
      totalWorkers: job.assignedWorkers.length,
      config: job.config,
      topology: topology.getWorkerTopology(workerIndex),
      peers: topology.getWorkerPeers(workerIndex),
      syncStrategy: this.config.synchronizationStrategy,
      compressionEnabled: this.config.compressionEnabled
    };
  }

  private async sendTrainingCommand(worker: WorkerNode, config: WorkerTrainingConfig): Promise<void> {
    const commandClient = new WorkerCommandClient(worker.address);
    await commandClient.startTraining(config);
  }

  private async updateJobProgress(job: TrainingJob): Promise<void> {
    const progressPromises = job.assignedWorkers.map(async (workerId) => {
      const worker = this.workers.get(workerId);
      const progressClient = new WorkerProgressClient(worker.address);
      return progressClient.getProgress(job.id);
    });

    const workerProgresses = await Promise.all(progressPromises);

    // Aggregate progress from all workers
    job.progress = this.aggregateWorkerProgress(workerProgresses);
    job.metrics = this.aggregateWorkerMetrics(workerProgresses);
  }

  private aggregateWorkerProgress(progresses: WorkerProgress[]): TrainingProgress {
    const totalSteps = progresses.reduce((sum, p) => sum + p.stepsCompleted, 0);
    const avgLoss = progresses.reduce((sum, p) => sum + p.currentLoss, 0) / progresses.length;
    const minLoss = Math.min(...progresses.map(p => p.bestLoss));

    return {
      epoch: Math.max(...progresses.map(p => p.epoch)),
      totalEpochs: progresses[0]?.totalEpochs || 0,
      stepsCompleted: totalSteps,
      totalSteps: progresses[0]?.totalSteps || 0,
      currentLoss: avgLoss,
      bestLoss: minLoss,
      convergenceMetrics: this.aggregateConvergenceMetrics(progresses)
    };
  }

  private handleGradientAggregation(aggregation: GradientAggregation): void {
    // Update global model with aggregated gradients
    this.parameterSyncer.updateGlobalModel(aggregation);

    // Distribute updated parameters to workers
    this.parameterSyncer.broadcastParameters(aggregation.metadata);

    this.emit('gradient_aggregated', aggregation);
  }

  private async handleWorkerFailure(workerId: string): Promise<void> {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    this.logger.error(`Worker failure detected: ${workerId}`);

    // Migrate running jobs to other workers
    await this.migrateWorkerJobs(worker);

    // Update worker status
    worker.status = WorkerStatus.FAILED;

    // Emit failure event
    this.emit('worker_failed', workerId);
  }

  private async migrateWorkerJobs(failedWorker: WorkerNode): Promise<void> {
    const affectedJobs = Array.from(this.trainingJobs.values())
      .filter(job => job.assignedWorkers.includes(failedWorker.id));

    for (const job of affectedJobs) {
      await this.migrateJob(job, failedWorker);
    }
  }

  private async migrateJob(job: TrainingJob, failedWorker: WorkerNode): Promise<void> {
    // Remove failed worker from job
    job.assignedWorkers = job.assignedWorkers.filter(id => id !== failedWorker.id);

    // Select replacement worker
    const replacementWorkers = await this.selectOptimalWorkers(job);

    if (replacementWorkers.length > 0) {
      const replacementWorker = replacementWorkers[0];
      job.assignedWorkers.push(replacementWorker.id);

      // Resume training on replacement worker
      await this.resumeTrainingOnWorker(job, replacementWorker, failedWorker.id);
    }
  }

  private async resumeTrainingOnWorker(
    job: TrainingJob,
    worker: WorkerNode,
    failedWorkerId: string
  ): Promise<void> {
    // Get latest checkpoint
    const checkpoint = await this.checkpointManager.getLatestCheckpoint(job.id);

    // Generate worker config with checkpoint info
    const workerConfig = this.generateWorkerConfig(job, 0, null); // Simplified
    workerConfig.checkpoint = checkpoint;
    workerConfig.failedWorkerId = failedWorkerId;

    await this.sendTrainingCommand(worker, workerConfig);
  }

  private async notifyWorkersToStop(job: TrainingJob): Promise<void> {
    const promises = job.assignedWorkers.map(async (workerId) => {
      const worker = this.workers.get(workerId);
      const commandClient = new WorkerCommandClient(worker.address);
      await commandClient.stopTraining(job.id);
    });

    await Promise.all(promises);
  }

  private async rebalanceWorkers(): Promise<void> {
    // Implement load rebalancing logic
    this.logger.info('Rebalancing worker loads...');
  }

  private determineJobType(config: TrainingJobConfig): TrainingJobType {
    if (config.algorithm.includes('rl') || config.algorithm.includes('reinforcement')) {
      return TrainingJobType.REINFORCEMENT_LEARNING;
    } else if (config.algorithm.includes('causal') || config.algorithm.includes('gpcm')) {
      return TrainingJobType.CAUSAL_INFERENCE;
    } else if (config.algorithm.includes('dspy') || config.algorithm.includes('program')) {
      return TrainingJobType.DSPY_OPTIMIZATION;
    } else {
      return TrainingJobType.HYBRID_TRAINING;
    }
  }

  private initializeConvergenceMetrics(): ConvergenceMetrics {
    return {
      gradientNorm: Infinity,
      parameterChanges: [],
      lossImprovement: 0,
      validationScore: 0,
      earlyStoppingPatience: 10
    };
  }

  private initializeTrainingMetrics(): TrainingMetrics {
    return {
      loss: [],
      accuracy: [],
      throughput: [],
      latency: [],
      resourceUtilization: [],
      communicationOverhead: {
        bytesTransmitted: 0,
        bytesReceived: 0,
        messagesSent: 0,
        messagesReceived: 0,
        averageLatency: 0,
        compressionRatio: 1,
        retransmissions: 0
      }
    };
  }

  private generateJobId(): string {
    return `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateWorkerId(): string {
    return `worker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // ============================================================================
  // Lifecycle Management
  // ============================================================================

  /**
   * Start the distributed training coordinator
   */
  public async start(): Promise<void> {
    this.logger.info('Starting Distributed Training Coordinator...');

    // Start all components
    await this.topologyManager.start();
    await this.loadBalancer.start();
    await this.gradientAggregator.start();
    await this.parameterSyncer.start();
    await this.checkpointManager.start();
    await this.faultToleranceManager.start();
    await this.metricsCollector.start();

    this.logger.info('Distributed Training Coordinator started successfully');
  }

  /**
   * Stop the distributed training coordinator
   */
  public async stop(): Promise<void> {
    this.logger.info('Stopping Distributed Training Coordinator...');

    // Stop all running jobs
    await this.stopAllJobs();

    // Stop all components
    await this.metricsCollector.stop();
    await this.faultToleranceManager.stop();
    await this.checkpointManager.stop();
    await this.parameterSyncer.stop();
    await this.gradientAggregator.stop();
    await this.loadBalancer.stop();
    await this.topologyManager.stop();

    this.logger.info('Distributed Training Coordinator stopped');
  }

  private async stopAllJobs(): Promise<void> {
    const runningJobs = Array.from(this.trainingJobs.values())
      .filter(job => job.status === TrainingJobStatus.RUNNING);

    const stopPromises = runningJobs.map(job => this.cancelTrainingJob(job.id));
    await Promise.all(stopPromises);
  }
}

// ============================================================================
// Supporting Types
// ============================================================================

export interface WorkerRegistrationInfo {
  address: string;
  capabilities: WorkerCapabilities;
}

export interface WorkerTrainingConfig {
  jobId: string;
  workerIndex: number;
  totalWorkers: number;
  config: TrainingJobConfig;
  topology: WorkerTopology;
  peers: string[];
  syncStrategy: string;
  compressionEnabled: boolean;
  checkpoint?: Checkpoint;
  failedWorkerId?: string;
}

export interface WorkerTopology {
  rank: number;
  neighbors: string[];
  communicationPattern: 'ring' | 'all_reduce' | 'parameter_server';
}

export interface Checkpoint {
  id: string;
  step: number;
  epoch: number;
  modelPath: string;
  optimizerPath: string;
  timestamp: Date;
}

export interface Topology {
  id: string;
  type: string;
  workers: string[];
  connections: Connection[];
  getWorkerTopology(workerIndex: number): WorkerTopology;
  getWorkerPeers(workerIndex: number): string[];
}

export interface Connection {
  source: string;
  target: string;
  weight: number;
  latency: number;
  bandwidth: number;
}

export interface WorkerProgress {
  workerId: string;
  stepsCompleted: number;
  totalSteps: number;
  epoch: number;
  totalEpochs: number;
  currentLoss: number;
  bestLoss: number;
  convergenceMetrics: ConvergenceMetrics;
  metrics: TrainingMetrics;
}

export interface OptimizationRecommendation {
  type: 'hyperparameter' | 'topology' | 'resource' | 'algorithm';
  description: string;
  currentValue: any;
  recommendedValue: any;
  expectedImprovement: number;
  autoApply: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

export interface EnvironmentConstraints {
  maxLatency: number;
  maxMemoryUsage: number;
  maxCpuUsage: number;
  maxNetworkBandwidth: number;
}

export interface PreprocessingConfig {
  normalization: boolean;
  featureScaling: boolean;
  dimensionalityReduction: boolean;
  outlierRemoval: boolean;
  missingValueHandling: string;
}

// Abstract interfaces for supporting classes
export abstract class TopologyManager extends EventEmitter {
  abstract addWorker(worker: WorkerNode): Promise<void>;
  abstract removeWorker(workerId: string): Promise<void>;
  abstract createTrainingTopology(job: TrainingJob): Promise<Topology>;
  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
}

export abstract class LoadBalancer extends EventEmitter {
  abstract selectWorkers(workers: WorkerNode[], count: number, requirements: ResourceRequirements): WorkerNode[];
  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
}

export abstract class GradientAggregator extends EventEmitter {
  abstract aggregateGradients(gradients: GradientAggregation[]): Promise<GradientAggregation>;
  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
}

export abstract class ParameterSyncer {
  abstract initializeJob(job: TrainingJob, topology: Topology): Promise<void>;
  abstract updateGlobalModel(aggregation: GradientAggregation): Promise<void>;
  abstract broadcastParameters(metadata: AggregationMetadata): Promise<void>;
  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
}

export abstract class CheckpointManager {
  abstract saveCheckpoint(jobId: string, data: any): Promise<Checkpoint>;
  abstract getLatestCheckpoint(jobId: string): Promise<Checkpoint>;
  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
}

export abstract class FaultToleranceManager extends EventEmitter {
  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
}

export abstract class MetricsCollector {
  abstract getAggregatedMetrics(): Promise<TrainingMetrics>;
  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
}

// Client interfaces for worker communication
export abstract class WorkerCommandClient {
  constructor(address: string) {}
  abstract startTraining(config: WorkerTrainingConfig): Promise<void>;
  abstract stopTraining(jobId: string): Promise<void>;
}

export abstract class WorkerProgressClient {
  constructor(address: string) {}
  abstract getProgress(jobId: string): Promise<WorkerProgress>;
}

export abstract class TrainingOptimizer {
  abstract generateRecommendations(job: TrainingJob): Promise<OptimizationRecommendation[]>;
}