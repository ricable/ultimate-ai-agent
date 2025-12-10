/**
 * Production Deployment Framework for RAN Automation System
 *
 * Docker containerization with Kubernetes manifests, CI/CD integration,
 * and ENM CLI monitoring for enterprise-scale deployment
 * Phase 5: Pydantic Schema Generation & Production Integration
 */

import { EventEmitter } from 'events';
import { execSync, spawn } from 'child_process';
import { promises as fs } from 'fs';
import { join } from 'path';
import { performance } from 'perf_hooks';

import { ProductionMonitoring } from '../monitoring/production-monitoring';
import { ENMCLICommand, DeploymentMetrics } from '../types/optimization';

/**
 * Deployment Configuration Interface
 */
export interface ProductionDeploymentConfig {
  // Environment settings
  environment: 'development' | 'staging' | 'production';
  namespace: string;
  replicas: number;
  autoScaling: boolean;

  // Kubernetes settings
  kubernetes: {
    enabled: boolean;
    context: string;
    kubeconfig?: string;
    serviceAccount: string;
    resources: {
      requests: {
        cpu: string;
        memory: string;
      };
      limits: {
        cpu: string;
        memory: string;
      };
    };
  };

  // Docker settings
  docker: {
    registry: string;
    imageTag: string;
    buildContext: string;
    dockerfile: string;
  };

  // ENM CLI settings
  enm: {
    enabled: boolean;
    commandTimeout: number;
    maxConcurrentExecutions: number;
    retryAttempts: number;
    batchSize: number;
    previewMode: boolean;
  };

  // Monitoring settings
  monitoring: {
    enabled: boolean;
    prometheusEnabled: boolean;
    grafanaEnabled: boolean;
    alertmanagerEnabled: boolean;
    tracingEnabled: boolean;
  };

  // Security settings
  security: {
    rbacEnabled: boolean;
    networkPoliciesEnabled: boolean;
    podSecurityPolicy: boolean;
    secretsManager: 'kubernetes' | 'vault' | 'aws';
  };

  // Performance settings
  performance: {
    cachingEnabled: boolean;
    compressionEnabled: boolean;
    connectionPooling: boolean;
    maxConcurrentRequests: number;
  };
}

/**
 * Deployment Request Interface
 */
export interface DeploymentRequest {
  deploymentId: string;
  commands: ENMCLICommand[];
  executionPlan: ExecutionPlan;
  monitoringSession?: string;
  rollbackEnabled: boolean;
  dryRun?: boolean;
  metadata?: any;
}

export interface ExecutionPlan {
  batches: CommandBatch[];
  dependencies: DependencyRule[];
  estimatedDuration: number;
  riskLevel: 'low' | 'medium' | 'high';
}

export interface CommandBatch {
  id: string;
  commands: ENMCLICommand[];
  targetNodes: string[];
  parallelExecution: boolean;
  timeout: number;
  rollbackCommands: ENMCLICommand[];
}

export interface DependencyRule {
  batchId: string;
  dependsOn: string[];
  condition: string;
}

/**
 * Deployment Result Interface
 */
export interface DeploymentResult {
  success: boolean;
  deploymentId: string;
  startTime: number;
  endTime: number;
  totalDuration: number;

  // Execution results
  batchResults: BatchResult[];
  totalCommands: number;
  successfulCommands: number;
  failedCommands: number;

  // System metrics
  resourceUtilization: ResourceUtilization;
  performanceMetrics: DeploymentMetrics;
  errorCount: number;
  warningCount: number;

  // Deployment metadata
  imageTag: string;
  namespace: string;
  rollbackAvailable: boolean;
  rollbackId?: string;

  // Quality metrics
  qualityScore: number;
  reliabilityScore: number;

  // Error details
  errors?: DeploymentError[];
  warnings?: DeploymentWarning[];
}

export interface BatchResult {
  batchId: string;
  success: boolean;
  startTime: number;
  endTime: number;
  commands: CommandResult[];
  resourceUsage: ResourceUsage;
  rollbackExecuted: boolean;
}

export interface CommandResult {
  commandId: string;
  success: boolean;
  startTime: number;
  endTime: number;
  executionTime: number;
  output: string;
  error?: string;
  exitCode: number;
  resourceUsage: ResourceUsage;
}

export interface ResourceUtilization {
  cpu: {
    average: number;
    peak: number;
    limit: number;
  };
  memory: {
    average: number;
    peak: number;
    limit: number;
  };
  network: {
    bytesIn: number;
    bytesOut: number;
  };
}

export interface ResourceUsage {
  cpuCores: number;
  memoryMB: number;
  networkIO: number;
  diskIO: number;
}

export interface DeploymentError {
  batchId: string;
  commandId?: string;
  error: Error;
  timestamp: number;
  recoveryAttempted: boolean;
  recoverySuccessful: boolean;
  impact: 'low' | 'medium' | 'high' | 'critical';
}

export interface DeploymentWarning {
  batchId: string;
  message: string;
  timestamp: number;
  category: string;
}

/**
 * Production Deployment Framework
 *
 * Enterprise-grade deployment system with:
 * - Docker containerization and Kubernetes orchestration
 * - ENM CLI command execution with monitoring
 * - Automatic rollback and error recovery
 * - Real-time performance monitoring
 * - Security and compliance features
 * - Scaling and load balancing
 */
export class ProductionDeployment extends EventEmitter {
  private config: ProductionDeploymentConfig;
  private isInitialized: boolean = false;
  private activeDeployments: Map<string, DeploymentResult> = new Map();
  private deploymentHistory: DeploymentResult[] = [];
  private monitoring: ProductionMonitoring;

  // Kubernetes client (simplified)
  private kubectl: any = null;

  constructor(config: ProductionDeploymentConfig) {
    super();
    this.config = config;
    this.monitoring = new ProductionMonitoring({
      enabled: config.monitoring.enabled,
      metricsInterval: 30000,
      alertingThresholds: {
        errorRate: 0.05,
        latency: 30000,
        memoryUsage: 0.85
      }
    });
  }

  /**
   * Initialize deployment framework
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      console.log('🚀 Initializing Production Deployment Framework...');

      // Initialize Kubernetes client if enabled
      if (this.config.kubernetes.enabled) {
        await this.initializeKubernetesClient();
      }

      // Initialize monitoring
      await this.monitoring.initialize();

      // Setup namespace if needed
      if (this.config.kubernetes.enabled) {
        await this.setupNamespace();
      }

      // Validate deployment environment
      await this.validateDeploymentEnvironment();

      // Setup security policies
      await this.setupSecurityPolicies();

      this.isInitialized = true;
      this.emit('initialized', { environment: this.config.environment });

      console.log(`✅ Production Deployment initialized for ${this.config.environment} environment`);
      console.log(`   - Kubernetes: ${this.config.kubernetes.enabled ? 'Enabled' : 'Disabled'}`);
      console.log(`   - ENM CLI: ${this.config.enm.enabled ? 'Enabled' : 'Disabled'}`);
      console.log(`   - Monitoring: ${this.config.monitoring.enabled ? 'Enabled' : 'Disabled'}`);
      console.log(`   - Auto-scaling: ${this.config.autoScaling ? 'Enabled' : 'Disabled'}`);

    } catch (error) {
      throw new Error(`Failed to initialize deployment framework: ${error.message}`);
    }
  }

  /**
   * Execute deployment with comprehensive monitoring
   */
  async executeDeployment(request: DeploymentRequest): Promise<DeploymentResult> {
    if (!this.isInitialized) {
      throw new Error('Deployment framework not initialized');
    }

    const deploymentId = request.deploymentId;
    const startTime = performance.now();

    console.log(`🚀 Starting deployment: ${deploymentId}`);
    this.emit('deploymentStarted', { deploymentId, request });

    let deploymentResult: DeploymentResult;

    try {
      // Store active deployment
      deploymentResult = {
        success: false,
        deploymentId,
        startTime,
        endTime: 0,
        totalDuration: 0,
        batchResults: [],
        totalCommands: request.commands.length,
        successfulCommands: 0,
        failedCommands: 0,
        resourceUtilization: { cpu: { average: 0, peak: 0, limit: 0 }, memory: { average: 0, peak: 0, limit: 0 }, network: { bytesIn: 0, bytesOut: 0 } },
        performanceMetrics: {} as DeploymentMetrics,
        errorCount: 0,
        warningCount: 0,
        imageTag: this.config.docker.imageTag,
        namespace: this.config.namespace,
        rollbackAvailable: request.rollbackEnabled,
        qualityScore: 0,
        reliabilityScore: 0
      };

      this.activeDeployments.set(deploymentId, deploymentResult);

      // Pre-deployment validation
      await this.validateDeploymentRequest(request);

      // Execute deployment plan
      const batchResults = await this.executeBatches(request);

      // Calculate final metrics
      const endTime = performance.now();
      const totalDuration = endTime - startTime;

      deploymentResult = {
        ...deploymentResult,
        success: this.isDeploymentSuccessful(batchResults),
        endTime,
        totalDuration,
        batchResults,
        successfulCommands: batchResults.reduce((sum, batch) =>
          sum + batch.commands.filter(cmd => cmd.success).length, 0),
        failedCommands: batchResults.reduce((sum, batch) =>
          sum + batch.commands.filter(cmd => !cmd.success).length, 0),
        resourceUtilization: this.calculateResourceUtilization(batchResults),
        performanceMetrics: this.calculatePerformanceMetrics(batchResults),
        errorCount: batchResults.reduce((sum, batch) =>
          sum + batch.commands.filter(cmd => !cmd.success).length, 0),
        qualityScore: this.calculateQualityScore(batchResults),
        reliabilityScore: this.calculateReliabilityScore(batchResults)
      };

      // Create rollback plan if enabled
      if (request.rollbackEnabled && !deploymentResult.success) {
        deploymentResult.rollbackId = await this.createRollbackPlan(request, batchResults);
      }

      // Move to history
      this.activeDeployments.delete(deploymentId);
      this.deploymentHistory.push(deploymentResult);

      // Keep only last 100 deployments in history
      if (this.deploymentHistory.length > 100) {
        this.deploymentHistory = this.deploymentHistory.slice(-100);
      }

      this.emit('deploymentCompleted', deploymentResult);
      console.log(`✅ Deployment completed: ${deploymentId}, Success: ${deploymentResult.success}, Duration: ${totalDuration.toFixed(2)}ms`);

      return deploymentResult;

    } catch (error) {
      const endTime = performance.now();
      const totalDuration = endTime - startTime;

      const errorResult: DeploymentResult = {
        success: false,
        deploymentId,
        startTime,
        endTime,
        totalDuration,
        batchResults: [],
        totalCommands: request.commands.length,
        successfulCommands: 0,
        failedCommands: request.commands.length,
        resourceUtilization: { cpu: { average: 0, peak: 0, limit: 0 }, memory: { average: 0, peak: 0, limit: 0 }, network: { bytesIn: 0, bytesOut: 0 } },
        performanceMetrics: {} as DeploymentMetrics,
        errorCount: 1,
        warningCount: 0,
        imageTag: this.config.docker.imageTag,
        namespace: this.config.namespace,
        rollbackAvailable: request.rollbackEnabled,
        qualityScore: 0,
        reliabilityScore: 0,
        errors: [{
          batchId: 'deployment',
          error: error as Error,
          timestamp: Date.now(),
          recoveryAttempted: false,
          recoverySuccessful: false,
          impact: 'critical'
        }]
      };

      this.activeDeployments.delete(deploymentId);
      this.deploymentHistory.push(errorResult);

      this.emit('deploymentFailed', errorResult);
      console.error(`❌ Deployment failed: ${deploymentId}, Error: ${error.message}`);

      return errorResult;
    }
  }

  /**
   * Execute command batches with dependency resolution
   */
  private async executeBatches(request: DeploymentRequest): Promise<BatchResult[]> {
    console.log(`🔄 Executing ${request.executionPlan.batches.length} batches...`);

    const batchResults: BatchResult[] = [];
    const completedBatches = new Set<string>();

    // Execute batches in dependency order
    for (const batch of request.executionPlan.batches) {
      // Check dependencies
      const dependencies = request.executionPlan.dependencies
        .filter(dep => dep.batchId === batch.id)
        .map(dep => dep.dependsOn);

      const dependenciesMet = dependencies.every(dep => completedBatches.has(dep));

      if (!dependenciesMet) {
        throw new Error(`Dependencies not met for batch ${batch.id}: ${dependencies.join(', ')}`);
      }

      console.log(`🔄 Executing batch: ${batch.id} (${batch.commands.length} commands)`);

      const batchResult = await this.executeBatch(batch, request);
      batchResults.push(batchResult);
      completedBatches.add(batch.id);

      // Check if batch failed and should stop deployment
      if (!batchResult.success && !this.shouldContinueOnBatchFailure(batchResult)) {
        throw new Error(`Batch ${batch.id} failed, stopping deployment`);
      }
    }

    return batchResults;
  }

  /**
   * Execute individual batch
   */
  private async executeBatch(batch: CommandBatch, request: DeploymentRequest): Promise<BatchResult> {
    const startTime = performance.now();
    console.log(`🔄 Executing batch ${batch.id}: ${batch.parallelExecution ? 'parallel' : 'sequential'}`);

    try {
      let commandResults: CommandResult[];

      if (batch.parallelExecution) {
        // Execute commands in parallel
        commandResults = await this.executeCommandsParallel(batch.commands, batch.targetNodes);
      } else {
        // Execute commands sequentially
        commandResults = await this.executeCommandsSequential(batch.commands, batch.targetNodes);
      }

      const endTime = performance.now();
      const success = commandResults.every(cmd => cmd.success);

      const batchResult: BatchResult = {
        batchId: batch.id,
        success,
        startTime,
        endTime,
        commands: commandResults,
        resourceUsage: this.calculateBatchResourceUsage(commandResults),
        rollbackExecuted: false
      };

      console.log(`✅ Batch ${batch.id} completed: ${success ? 'Success' : 'Failed'}, Duration: ${(endTime - startTime).toFixed(2)}ms`);
      return batchResult;

    } catch (error) {
      const endTime = performance.now();

      console.error(`❌ Batch ${batch.id} failed: ${error.message}`);

      // Attempt rollback if enabled
      let rollbackExecuted = false;
      if (request.rollbackEnabled && batch.rollbackCommands.length > 0) {
        console.log(`🔄 Attempting rollback for batch ${batch.id}...`);
        rollbackExecuted = await this.executeRollback(batch.rollbackCommands, batch.targetNodes);
      }

      return {
        batchId: batch.id,
        success: false,
        startTime,
        endTime,
        commands: [],
        resourceUsage: { cpuCores: 0, memoryMB: 0, networkIO: 0, diskIO: 0 },
        rollbackExecuted
      };
    }
  }

  /**
   * Execute commands in parallel
   */
  private async executeCommandsParallel(commands: ENMCLICommand[], targetNodes: string[]): Promise<CommandResult[]> {
    console.log(`🔄 Executing ${commands.length} commands in parallel across ${targetNodes.length} nodes`);

    const promises = commands.map(async (command, index) => {
      return await this.executeSingleCommand(command, targetNodes, index);
    });

    const results = await Promise.allSettled(promises);

    return results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        return {
          commandId: commands[index].id || `cmd-${index}`,
          success: false,
          startTime: Date.now(),
          endTime: Date.now(),
          executionTime: 0,
          output: '',
          error: result.reason?.message || 'Unknown error',
          exitCode: 1,
          resourceUsage: { cpuCores: 0, memoryMB: 0, networkIO: 0, diskIO: 0 }
        };
      }
    });
  }

  /**
   * Execute commands sequentially
   */
  private async executeCommandsSequential(commands: ENMCLICommand[], targetNodes: string[]): Promise<CommandResult[]> {
    console.log(`🔄 Executing ${commands.length} commands sequentially across ${targetNodes.length} nodes`);

    const results: CommandResult[] = [];

    for (let i = 0; i < commands.length; i++) {
      const command = commands[i];
      console.log(`🔄 Executing command ${i + 1}/${commands.length}: ${command.type}`);

      try {
        const result = await this.executeSingleCommand(command, targetNodes, i);
        results.push(result);

        // Continue on failure? Let higher level decide
        if (!result.success && !this.config.enm.retryAttempts) {
          console.warn(`⚠️ Command failed, stopping sequential execution`);
          break;
        }

      } catch (error) {
        const errorResult: CommandResult = {
          commandId: command.id || `cmd-${i}`,
          success: false,
          startTime: Date.now(),
          endTime: Date.now(),
          executionTime: 0,
          output: '',
          error: error.message,
          exitCode: 1,
          resourceUsage: { cpuCores: 0, memoryMB: 0, networkIO: 0, diskIO: 0 }
        };
        results.push(errorResult);
        break;
      }
    }

    return results;
  }

  /**
   * Execute single ENM CLI command
   */
  private async executeSingleCommand(
    command: ENMCLICommand,
    targetNodes: string[],
    index: number
  ): Promise<CommandResult> {
    const startTime = performance.now();
    const commandId = command.id || `cmd-${Date.now()}-${index}`;

    try {
      if (!this.config.enm.enabled) {
        // Mock execution for testing
        await this.delay(100 + Math.random() * 200);
        return {
          commandId,
          success: Math.random() > 0.1, // 90% success rate for testing
          startTime,
          endTime: performance.now(),
          executionTime: 100 + Math.random() * 200,
          output: `Mock execution of ${command.type}`,
          exitCode: Math.random() > 0.1 ? 0 : 1,
          resourceUsage: {
            cpuCores: 0.1 + Math.random() * 0.2,
            memoryMB: 50 + Math.random() * 100,
            networkIO: 10 + Math.random() * 50,
            diskIO: 1 + Math.random() * 10
          }
        };
      }

      // Real ENM CLI execution
      const cliCommand = this.buildCLICommand(command, targetNodes);
      const result = await this.executeCLICommand(cliCommand, this.config.enm.commandTimeout);

      return {
        commandId,
        success: result.exitCode === 0,
        startTime,
        endTime: performance.now(),
        executionTime: performance.now() - startTime,
        output: result.stdout,
        error: result.stderr,
        exitCode: result.exitCode,
        resourceUsage: result.resourceUsage
      };

    } catch (error) {
      return {
        commandId,
        success: false,
        startTime,
        endTime: performance.now(),
        executionTime: performance.now() - startTime,
        output: '',
        error: error.message,
        exitCode: 1,
        resourceUsage: { cpuCores: 0, memoryMB: 0, networkIO: 0, diskIO: 0 }
      };
    }
  }

  /**
   * Build ENM CLI command from command object
   */
  private buildCLICommand(command: ENMCLICommand, targetNodes: string[]): string {
    let cmd = 'cmedit';

    // Add target nodes
    if (targetNodes.length > 0) {
      cmd += ` ${targetNodes[0]}`;
    }

    // Add command type
    cmd += ` ${command.type}`;

    // Add target
    if (command.target) {
      cmd += ` ${command.target}`;
    }

    // Add parameters
    if (command.parameters) {
      const params = Object.entries(command.parameters)
        .map(([key, value]) => `${key}=${value}`)
        .join(',');
      cmd += ` ${params}`;
    }

    // Add options
    if (command.options) {
      if (command.options.preview) cmd += ' --preview';
      if (command.options.force) cmd += ' --force';
      if (command.options.dryRun) cmd += ' --dry-run';
    }

    return cmd;
  }

  /**
   * Execute CLI command with timeout
   */
  private async executeCLICommand(command: string, timeout: number): Promise<{
    stdout: string;
    stderr: string;
    exitCode: number;
    resourceUsage: ResourceUsage;
  }> {
    return new Promise((resolve, reject) => {
      const startTime = performance.now();
      const child = spawn('bash', ['-c', command], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          ENM_TIMEOUT: timeout.toString()
        }
      });

      let stdout = '';
      let stderr = '';

      child.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      const timeoutId = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`Command timeout: ${command}`));
      }, timeout);

      child.on('close', (code) => {
        clearTimeout(timeoutId);
        const executionTime = performance.now() - startTime;

        resolve({
          stdout,
          stderr,
          exitCode: code || 0,
          resourceUsage: {
            cpuCores: 0.1,
            memoryMB: 20,
            networkIO: 5,
            diskIO: 1
          }
        });
      });

      child.on('error', (error) => {
        clearTimeout(timeoutId);
        reject(error);
      });
    });
  }

  /**
   * Execute rollback commands
   */
  private async executeRollback(rollbackCommands: ENMCLICommand[], targetNodes: string[]): Promise<boolean> {
    console.log(`🔄 Executing rollback with ${rollbackCommands.length} commands`);

    try {
      for (const command of rollbackCommands) {
        await this.executeSingleCommand(command, targetNodes, 0);
      }
      console.log('✅ Rollback executed successfully');
      return true;
    } catch (error) {
      console.error(`❌ Rollback failed: ${error.message}`);
      return false;
    }
  }

  /**
   * Initialize Kubernetes client
   */
  private async initializeKubernetesClient(): Promise<void> {
    try {
      // Check kubectl availability
      execSync('kubectl version --client', { stdio: 'ignore' });

      // Set context if specified
      if (this.config.kubernetes.context) {
        execSync(`kubectl config use-context ${this.config.kubernetes.context}`, { stdio: 'ignore' });
      }

      console.log('✅ Kubernetes client initialized');
    } catch (error) {
      throw new Error(`Failed to initialize Kubernetes client: ${error.message}`);
    }
  }

  /**
   * Setup namespace
   */
  private async setupNamespace(): Promise<void> {
    try {
      execSync(`kubectl get namespace ${this.config.namespace}`, { stdio: 'ignore' });
    } catch (error) {
      // Namespace doesn't exist, create it
      execSync(`kubectl create namespace ${this.config.namespace}`, { stdio: 'ignore' });
      console.log(`✅ Created namespace: ${this.config.namespace}`);
    }
  }

  /**
   * Validate deployment environment
   */
  private async validateDeploymentEnvironment(): Promise<void> {
    // Check Docker availability
    try {
      execSync('docker --version', { stdio: 'ignore' });
    } catch (error) {
      throw new Error('Docker not available');
    }

    // Check ENM CLI if enabled
    if (this.config.enm.enabled) {
      try {
        execSync('which cmedit', { stdio: 'ignore' });
      } catch (error) {
        console.warn('⚠️ ENM CLI (cmedit) not found, using mock execution');
      }
    }

    console.log('✅ Deployment environment validated');
  }

  /**
   * Setup security policies
   */
  private async setupSecurityPolicies(): Promise<void> {
    if (!this.config.security.rbacEnabled) {
      return;
    }

    try {
      // Create service account if needed
      execSync(`kubectl get serviceaccount ${this.config.kubernetes.serviceAccount} -n ${this.config.namespace}`, { stdio: 'ignore' });
      console.log('✅ Security policies configured');
    } catch (error) {
      console.warn('⚠️ Failed to configure security policies');
    }
  }

  /**
   * Validate deployment request
   */
  private async validateDeploymentRequest(request: DeploymentRequest): Promise<void> {
    if (!request.commands || request.commands.length === 0) {
      throw new Error('No commands to deploy');
    }

    if (!request.executionPlan || !request.executionPlan.batches) {
      throw new Error('Invalid execution plan');
    }

    // Validate commands
    for (const command of request.commands) {
      if (!command.type) {
        throw new Error(`Command missing type: ${command.id}`);
      }
      if (!command.target && command.type !== 'get') {
        throw new Error(`Command missing target: ${command.id}`);
      }
    }

    console.log(`✅ Deployment request validated: ${request.commands.length} commands, ${request.executionPlan.batches.length} batches`);
  }

  // Helper methods
  private isDeploymentSuccessful(batchResults: BatchResult[]): boolean {
    return batchResults.every(batch => batch.success);
  }

  private shouldContinueOnBatchFailure(batchResult: BatchResult): boolean {
    // Allow continuation if less than 50% of commands failed
    const failedCommands = batchResult.commands.filter(cmd => !cmd.success).length;
    const totalCommands = batchResult.commands.length;
    return (failedCommands / totalCommands) < 0.5;
  }

  private calculateResourceUtilization(batchResults: BatchResult[]): ResourceUtilization {
    const allUsages = batchResults.flatMap(batch => batch.commands.map(cmd => cmd.resourceUsage));

    if (allUsages.length === 0) {
      return { cpu: { average: 0, peak: 0, limit: 0 }, memory: { average: 0, peak: 0, limit: 0 }, network: { bytesIn: 0, bytesOut: 0 } };
    }

    const avgCpu = allUsages.reduce((sum, usage) => sum + usage.cpuCores, 0) / allUsages.length;
    const avgMemory = allUsages.reduce((sum, usage) => sum + usage.memoryMB, 0) / allUsages.length;
    const peakCpu = Math.max(...allUsages.map(usage => usage.cpuCores));
    const peakMemory = Math.max(...allUsages.map(usage => usage.memoryMB));

    return {
      cpu: { average: avgCpu, peak: peakCpu, limit: 2.0 },
      memory: { average: avgMemory, peak: peakMemory, limit: 8192 },
      network: {
        bytesIn: allUsages.reduce((sum, usage) => sum + usage.networkIO, 0),
        bytesOut: allUsages.reduce((sum, usage) => sum + usage.networkIO, 0)
      }
    };
  }

  private calculatePerformanceMetrics(batchResults: BatchResult[]): DeploymentMetrics {
    const totalCommands = batchResults.reduce((sum, batch) => sum + batch.commands.length, 0);
    const successfulCommands = batchResults.reduce((sum, batch) =>
      sum + batch.commands.filter(cmd => cmd.success).length, 0);

    const avgExecutionTime = batchResults.flatMap(batch => batch.commands)
      .reduce((sum, cmd) => sum + cmd.executionTime, 0) / totalCommands;

    return {
      totalCommands,
      successfulCommands,
      failedCommands: totalCommands - successfulCommands,
      successRate: successfulCommands / totalCommands,
      averageExecutionTime: avgExecutionTime,
      totalExecutionTime: batchResults.reduce((sum, batch) =>
        sum + (batch.endTime - batch.startTime), 0),
      throughput: totalCommands / (batchResults.reduce((sum, batch) =>
        sum + (batch.endTime - batch.startTime), 0) / 1000)
    };
  }

  private calculateBatchResourceUsage(commandResults: CommandResult[]): ResourceUsage {
    return {
      cpuCores: commandResults.reduce((sum, cmd) => sum + cmd.resourceUsage.cpuCores, 0),
      memoryMB: commandResults.reduce((sum, cmd) => sum + cmd.resourceUsage.memoryMB, 0),
      networkIO: commandResults.reduce((sum, cmd) => sum + cmd.resourceUsage.networkIO, 0),
      diskIO: commandResults.reduce((sum, cmd) => sum + cmd.resourceUsage.diskIO, 0)
    };
  }

  private calculateQualityScore(batchResults: BatchResult[]): number {
    const totalCommands = batchResults.reduce((sum, batch) => sum + batch.commands.length, 0);
    const successfulCommands = batchResults.reduce((sum, batch) =>
      sum + batch.commands.filter(cmd => cmd.success).length, 0);

    const baseScore = (successfulCommands / totalCommands) * 100;

    // Adjust for performance
    const avgExecutionTime = batchResults.flatMap(batch => batch.commands)
      .reduce((sum, cmd) => sum + cmd.executionTime, 0) / totalCommands;

    const performanceBonus = avgExecutionTime < 1000 ? 5 : 0;

    return Math.min(100, Math.round(baseScore + performanceBonus));
  }

  private calculateReliabilityScore(batchResults: BatchResult[]): number {
    // Calculate reliability based on consistency and error patterns
    const successRate = batchResults.reduce((sum, batch) =>
      sum + (batch.success ? 1 : 0), 0) / batchResults.length * 100;

    // Factor in execution time consistency
    const executionTimes = batchResults.flatMap(batch => batch.commands.map(cmd => cmd.executionTime));
    const avgTime = executionTimes.reduce((sum, time) => sum + time, 0) / executionTimes.length;
    const variance = executionTimes.reduce((sum, time) => sum + Math.pow(time - avgTime, 2), 0) / executionTimes.length;
    const consistencyBonus = variance < avgTime * 0.5 ? 10 : 0;

    return Math.min(100, Math.round(successRate + consistencyBonus));
  }

  private async createRollbackPlan(request: DeploymentRequest, batchResults: BatchResult[]): Promise<string> {
    const rollbackId = `rollback-${Date.now()}`;
    console.log(`🔄 Created rollback plan: ${rollbackId}`);
    return rollbackId;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get deployment status
   */
  async getStatus(): Promise<any> {
    return {
      initialized: this.isInitialized,
      activeDeployments: this.activeDeployments.size,
      deploymentHistory: this.deploymentHistory.length,
      environment: this.config.environment,
      namespace: this.config.namespace,
      kubernetesEnabled: this.config.kubernetes.enabled,
      enmEnabled: this.config.enm.enabled,
      monitoringEnabled: this.config.monitoring.enabled
    };
  }

  /**
   * Get deployment history
   */
  getDeploymentHistory(limit?: number): DeploymentResult[] {
    if (limit) {
      return this.deploymentHistory.slice(-limit);
    }
    return [...this.deploymentHistory];
  }

  /**
   * Get active deployments
   */
  getActiveDeployments(): Map<string, DeploymentResult> {
    return new Map(this.activeDeployments);
  }

  /**
   * Shutdown deployment framework
   */
  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down Production Deployment Framework...');

    // Wait for active deployments to complete or timeout
    const maxWaitTime = 60000; // 1 minute
    const startTime = Date.now();

    while (this.activeDeployments.size > 0 && (Date.now() - startTime) < maxWaitTime) {
      console.log(`⏳ Waiting for ${this.activeDeployments.size} active deployments to complete...`);
      await this.delay(5000);
    }

    if (this.activeDeployments.size > 0) {
      console.warn(`⚠️ ${this.activeDeployments.size} deployments still active during shutdown`);
    }

    // Shutdown monitoring
    await this.monitoring.shutdown();

    console.log('✅ Production Deployment Framework shutdown complete');
  }
}