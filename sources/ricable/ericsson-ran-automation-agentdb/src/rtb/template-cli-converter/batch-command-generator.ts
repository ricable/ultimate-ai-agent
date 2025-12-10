/**
 * Batch Command Generator
 *
 * Optimizes commands for batch execution with parallel processing,
 * dependency analysis, and intelligent batching strategies.
 */

import {
  GeneratedCliCommand,
  CliCommandType,
  TemplateToCliContext,
  CliExecutionResult,
  ExecutionStatus
} from './types';

/**
 * Batch Command Generator Configuration
 */
export interface BatchGeneratorConfig {
  /** Maximum commands per batch */
  maxCommandsPerBatch: number;
  /** Enable parallel execution */
  enableParallelExecution: boolean;
  /** Maximum parallel commands */
  maxConcurrency: number;
  /** Enable intelligent batching */
  enableIntelligentBatching: boolean;
  /** Batch timeout in seconds */
  batchTimeout: number;
  /** Enable batch optimization */
  enableBatchOptimization: boolean;
}

/**
 * Command batch
 */
export interface CommandBatch {
  /** Batch identifier */
  id: string;
  /** Commands in this batch */
  commands: GeneratedCliCommand[];
  /** Execution order within batch */
  executionOrder: string[];
  /** Can execute in parallel */
  parallel: boolean;
  /** Estimated duration */
  estimatedDuration: number;
  /** Risk level */
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  /** Batch metadata */
  metadata: {
    category: string;
    priority: number;
    dependencies: string[];
    dependents: string[];
  };
}

/**
 * Batch execution result
 */
export interface BatchExecutionResult {
  /** Batch identifier */
  batchId: string;
  /** Execution status */
  status: ExecutionStatus;
  /** Command results */
  commandResults: CliExecutionResult[];
  /** Overall duration */
  duration: number;
  /** Success rate */
  successRate: number;
  /** Errors encountered */
  errors: string[];
  /** Performance metrics */
  metrics: {
    parallelEfficiency?: number;
    resourceUtilization?: number;
    throughput?: number;
  };
}

/**
 * Batch optimization result
 */
export interface BatchOptimizationResult {
  /** Original command count */
  originalCommandCount: number;
  /** Optimized command count */
  optimizedCommandCount: number;
  /** Number of batches created */
  batchCount: number;
  /** Reduction percentage */
  reductionPercentage: number;
  /** Optimization applied */
  optimizations: string[];
  /** Expected performance improvement */
  performanceImprovement: {
    timeReduction?: number;
    resourceReduction?: number;
    throughputIncrease?: number;
  };
}

/**
 * Batch Command Generator Class
 */
export class BatchCommandGenerator {
  private config: BatchGeneratorConfig;
  private batchHistory: Map<string, BatchExecutionResult> = new Map();
  private optimizationRules: BatchOptimizationRule[] = [];

  constructor(config: BatchGeneratorConfig) {
    this.config = {
      maxCommandsPerBatch: 50,
      enableParallelExecution: true,
      maxConcurrency: 10,
      enableIntelligentBatching: true,
      batchTimeout: 300,
      enableBatchOptimization: true,
      ...config
    };

    this.initializeOptimizationRules();
  }

  /**
   * Optimize commands into batches
   */
  public async optimizeBatches(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<GeneratedCliCommand[]> {
    console.log(`Optimizing ${commands.length} commands for batch execution...`);

    const startTime = Date.now();

    try {
      // Phase 1: Analyze command characteristics
      const analysisResult = this.analyzeCommands(commands);

      // Phase 2: Group commands by category
      const groupedCommands = this.groupCommandsByCategory(commands, analysisResult);

      // Phase 3: Create initial batches
      const initialBatches = this.createInitialBatches(groupedCommands, context);

      // Phase 4: Apply batch optimization
      const optimizedBatches = this.config.enableBatchOptimization
        ? await this.optimizeBatches(initialBatches, context)
        : initialBatches;

      // Phase 5: Generate batch commands
      const batchCommands = this.generateBatchCommands(optimizedBatches, context);

      // Phase 6: Create final command list
      const finalCommands = this.createFinalCommandList(batchCommands, optimizedBatches);

      const duration = Date.now() - startTime;
      console.log(`Batch optimization completed in ${duration}ms: ${optimizedBatches.length} batches created`);

      return finalCommands;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`Batch optimization failed: ${errorMessage}`);
      return commands; // Return original commands on failure
    }
  }

  /**
   * Execute command batches
   */
  public async executeBatches(
    batches: CommandBatch[],
    context: TemplateToCliContext
  ): Promise<BatchExecutionResult[]> {
    console.log(`Executing ${batches.length} command batches...`);

    const results: BatchExecutionResult[] = [];

    for (const batch of batches) {
      const result = await this.executeBatch(batch, context);
      results.push(result);
      this.batchHistory.set(batch.id, result);
    }

    return results;
  }

  /**
   * Analyze command characteristics
   */
  private analyzeCommands(commands: GeneratedCliCommand[]): CommandAnalysisResult {
    const analysis: CommandAnalysisResult = {
      totalCommands: commands.length,
      commandsByType: this.groupCommandsByType(commands),
      commandsByRisk: this.groupCommandsByRisk(commands),
      commandsByComplexity: this.groupCommandsByComplexity(commands),
      parallelizableCommands: this.identifyParallelizableCommands(commands),
      criticalCommands: commands.filter(cmd => cmd.critical),
      estimatedTotalDuration: commands.reduce((sum, cmd) => sum + cmd.metadata.estimatedDuration, 0),
      averageComplexity: this.calculateAverageComplexity(commands),
      overallRiskLevel: this.calculateOverallRiskLevel(commands)
    };

    return analysis;
  }

  /**
   * Group commands by category
   */
  private groupCommandsByCategory(
    commands: GeneratedCliCommand[],
    analysis: CommandAnalysisResult
  ): Record<string, GeneratedCliCommand[]> {
    const groups: Record<string, GeneratedCliCommand[]> = {};

    for (const command of commands) {
      const category = command.metadata.category;
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(command);
    }

    return groups;
  }

  /**
   * Create initial batches
   */
  private createInitialBatches(
    groupedCommands: Record<string, GeneratedCliCommand[]>,
    context: TemplateToCliContext
  ): CommandBatch[] {
    const batches: CommandBatch[] = [];
    let batchIndex = 0;

    for (const [category, commands] of Object.entries(groupedCommands)) {
      // Sort commands by priority and risk
      const sortedCommands = this.sortCommandsForBatching(commands);

      // Create batches based on configuration
      if (this.config.enableIntelligentBatching) {
        const intelligentBatches = this.createIntelligentBatches(sortedCommands, category, batchIndex);
        batches.push(...intelligentBatches);
        batchIndex += intelligentBatches.length;
      } else {
        const simpleBatches = this.createSimpleBatches(sortedCommands, category, batchIndex);
        batches.push(...simpleBatches);
        batchIndex += simpleBatches.length;
      }
    }

    return batches;
  }

  /**
   * Create intelligent batches
   */
  private createIntelligentBatches(
    commands: GeneratedCliCommand[],
    category: string,
    startIndex: number
  ): CommandBatch[] {
    const batches: CommandBatch[] = [];
    let currentBatch: GeneratedCliCommand[] = [];
    let currentBatchDuration = 0;
    let currentBatchRisk = 0;

    for (const command of commands) {
      const commandRisk = this.getCommandRiskValue(command);
      const canAddToBatch = this.canAddToBatch(
        command,
        currentBatch,
        currentBatchDuration,
        currentBatchRisk
      );

      if (canAddToBatch && currentBatch.length < this.config.maxCommandsPerBatch) {
        currentBatch.push(command);
        currentBatchDuration += command.metadata.estimatedDuration;
        currentBatchRisk = Math.max(currentBatchRisk, commandRisk);
      } else {
        // Finalize current batch
        if (currentBatch.length > 0) {
          batches.push(this.createBatch(currentBatch, category, startIndex + batches.length));
          currentBatch = [];
          currentBatchDuration = 0;
          currentBatchRisk = 0;
        }

        // Start new batch with current command
        currentBatch.push(command);
        currentBatchDuration = command.metadata.estimatedDuration;
        currentBatchRisk = commandRisk;
      }
    }

    // Add final batch
    if (currentBatch.length > 0) {
      batches.push(this.createBatch(currentBatch, category, startIndex + batches.length));
    }

    return batches;
  }

  /**
   * Create simple batches
   */
  private createSimpleBatches(
    commands: GeneratedCliCommand[],
    category: string,
    startIndex: number
  ): CommandBatch[] {
    const batches: CommandBatch[] = [];

    for (let i = 0; i < commands.length; i += this.config.maxCommandsPerBatch) {
      const batchCommands = commands.slice(i, i + this.config.maxCommandsPerBatch);
      const batch = this.createBatch(batchCommands, category, startIndex + Math.floor(i / this.config.maxCommandsPerBatch));
      batches.push(batch);
    }

    return batches;
  }

  /**
   * Create batch object
   */
  private createBatch(
    commands: GeneratedCliCommand[],
    category: string,
    batchIndex: number
  ): CommandBatch {
    const executionOrder = commands.map(cmd => cmd.id);
    const estimatedDuration = commands.reduce((sum, cmd) => sum + cmd.metadata.estimatedDuration, 0);
    const riskLevel = this.calculateBatchRiskLevel(commands);
    const canExecuteInParallel = this.canExecuteBatchInParallel(commands);

    return {
      id: `batch_${category}_${batchIndex}`,
      commands,
      executionOrder,
      parallel: canExecuteInParallel && this.config.enableParallelExecution,
      estimatedDuration,
      riskLevel,
      metadata: {
        category,
        priority: this.calculateBatchPriority(commands),
        dependencies: [], // Will be filled by dependency analysis
        dependents: [] // Will be filled by dependency analysis
      }
    };
  }

  /**
   * Optimize batches
   */
  private async optimizeBatches(
    batches: CommandBatch[],
    context: TemplateToCliContext
  ): Promise<CommandBatch[]> {
    let optimizedBatches = [...batches];

    for (const rule of this.optimizationRules) {
      if (rule.applicable(optimizedBatches, context)) {
        const result = rule.apply(optimizedBatches, context);
        if (result.optimized) {
          optimizedBatches = result.batches;
          console.log(`Applied batch optimization: ${rule.name}`);
        }
      }
    }

    return optimizedBatches;
  }

  /**
   * Generate batch commands
   */
  private generateBatchCommands(
    batches: CommandBatch[],
    context: TemplateToCliContext
  ): GeneratedCliCommand[] {
    const batchCommands: GeneratedCliCommand[] = [];

    for (const batch of batches) {
      if (batch.commands.length === 1) {
        // Single command - no wrapper needed
        batchCommands.push(batch.commands[0]);
      } else {
        // Multiple commands - create batch wrapper
        const batchCommand = this.createBatchWrapperCommand(batch, context);
        batchCommands.push(batchCommand);
      }
    }

    return batchCommands;
  }

  /**
   * Create batch wrapper command
   */
  private createBatchWrapperCommand(
    batch: CommandBatch,
    context: TemplateToCliContext
  ): GeneratedCliCommand {
    const commandList = batch.commands
      .map(cmd => cmd.command)
      .join('\n');

    const batchScript = this.generateBatchScript(batch, commandList);

    return {
      id: batch.id,
      type: 'SCRIPT',
      command: batchScript,
      description: `Execute batch: ${batch.commands.length} commands`,
      timeout: Math.max(this.config.batchTimeout, batch.estimatedDuration / 1000),
      critical: batch.commands.some(cmd => cmd.critical),
      metadata: {
        category: 'batch',
        complexity: 'moderate',
        riskLevel: batch.riskLevel,
        estimatedDuration: batch.estimatedDuration
      }
    };
  }

  /**
   * Generate batch script
   */
  private generateBatchScript(batch: CommandBatch, commandList: string): string {
    const script = [
      '#!/bin/bash',
      '# Batch command execution script',
      `# Batch ID: ${batch.id}`,
      `# Commands: ${batch.commands.length}`,
      `# Parallel: ${batch.parallel}`,
      '',
      'set -e  # Exit on any error',
      '',
      '# Batch start time',
      'BATCH_START=$(date +%s)',
      '',
      'if [ "' + batch.parallel + '" = "true" ]; then',
      '    # Execute commands in parallel',
      '    ' + this.generateParallelExecutionBlock(batch.commands),
      'else',
      '    # Execute commands sequentially',
      '    ' + this.generateSequentialExecutionBlock(batch.commands),
      'fi',
      '',
      '# Calculate batch duration',
      'BATCH_END=$(date +%s)',
      'BATCH_DURATION=$((BATCH_END - BATCH_START))',
      'echo "Batch completed in ${BATCH_DURATION} seconds"',
      ''
    ];

    return script.join('\n');
  }

  /**
   * Generate parallel execution block
   */
  private generateParallelExecutionBlock(commands: GeneratedCliCommand[]): string {
    const maxParallel = Math.min(commands.length, this.config.maxConcurrency);
    const script: string[] = [];

    script.push('# Parallel execution with process control');
    script.push('PIDS=()');
    script.push('');

    for (let i = 0; i < commands.length; i += maxParallel) {
      const batchCommands = commands.slice(i, i + maxParallel);

      script.push(`# Commands ${i + 1}-${Math.min(i + maxParallel, commands.length)}`);

      for (const command of batchCommands) {
        const escapedCommand = command.command.replace(/"/g, '\\"');
        script.push(`echo "Executing: ${escapedCommand}"`);
        script.push(`(${escapedCommand} & )`);
        script.push('PIDS+=($!)');
        script.push('');
      }

      script.push('# Wait for current batch to complete');
      script.push('for PID in "${PIDS[@]}"; do');
      script.push('    if wait "$PID"; then');
      script.push('        echo "Command $PID completed successfully"');
      script.push('    else');
      script.push('        echo "Command $PID failed"');
      script.push('        exit 1');
      script.push('    fi');
      script.push('done');
      script.push('PIDS=()');
      script.push('');
    }

    return script.join('\n    ');
  }

  /**
   * Generate sequential execution block
   */
  private generateSequentialExecutionBlock(commands: GeneratedCliCommand[]): string {
    const script: string[] = [];

    script.push('# Sequential execution');

    for (const command of commands) {
      const escapedCommand = command.command.replace(/"/g, '\\"');
      script.push(`echo "Executing: ${escapedCommand}"`);
      script.push(escapedCommand);
      script.push('');
    }

    return script.join('\n    ');
  }

  /**
   * Create final command list
   */
  private createFinalCommandList(
    batchCommands: GeneratedCliCommand[],
    batches: CommandBatch[]
  ): GeneratedCliCommand[] {
    // Add batch coordination commands if needed
    const finalCommands = [...batchCommands];

    // Add setup commands
    finalCommands.unshift(this.createBatchSetupCommand(batches));

    // Add cleanup commands
    finalCommands.push(this.createBatchCleanupCommand(batches));

    return finalCommands;
  }

  /**
   * Create batch setup command
   */
  private createBatchSetupCommand(batches: CommandBatch[]): GeneratedCliCommand {
    return {
      id: 'batch_setup',
      type: 'SCRIPT',
      command: this.generateBatchSetupScript(batches),
      description: 'Setup batch execution environment',
      timeout: 30,
      critical: false,
      metadata: {
        category: 'setup',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    };
  }

  /**
   * Create batch cleanup command
   */
  private createBatchCleanupCommand(batches: CommandBatch[]): GeneratedCliCommand {
    return {
      id: 'batch_cleanup',
      type: 'SCRIPT',
      command: this.generateBatchCleanupScript(batches),
      description: 'Cleanup batch execution environment',
      timeout: 30,
      critical: false,
      metadata: {
        category: 'cleanup',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    };
  }

  /**
   * Generate batch setup script
   */
  private generateBatchSetupScript(batches: CommandBatch[]): string {
    const script = [
      '#!/bin/bash',
      '# Batch execution setup',
      '',
      'echo "Setting up batch execution environment..."',
      'echo "Total batches: ' + batches.length + '"',
      'echo "Estimated duration: ' + batches.reduce((sum, batch) => sum + batch.estimatedDuration, 0) + 'ms"',
      '',
      '# Create working directories',
      'mkdir -p /tmp/batch_execution',
      'mkdir -p /tmp/batch_logs',
      '',
      '# Set environment variables',
      'export BATCH_MODE=true',
      'export BATCH_START_TIME=$(date +%s)',
      '',
      'echo "Batch setup completed"'
    ];

    return script.join('\n');
  }

  /**
   * Generate batch cleanup script
   */
  private generateBatchCleanupScript(batches: CommandBatch[]): string {
    const script = [
      '#!/bin/bash',
      '# Batch execution cleanup',
      '',
      'echo "Cleaning up batch execution environment..."',
      '',
      '# Calculate total batch time',
      'BATCH_END_TIME=$(date +%s)',
      'TOTAL_BATCH_TIME=$((BATCH_END_TIME - BATCH_START_TIME))',
      'echo "Total batch execution time: ${TOTAL_BATCH_TIME} seconds"',
      '',
      '# Cleanup temporary files',
      'rm -rf /tmp/batch_execution',
      'rm -rf /tmp/batch_logs',
      '',
      'echo "Batch cleanup completed"'
    ];

    return script.join('\n');
  }

  /**
   * Execute single batch
   */
  private async executeBatch(
    batch: CommandBatch,
    context: TemplateToCliContext
  ): Promise<BatchExecutionResult> {
    const startTime = Date.now();
    const commandResults: CliExecutionResult[] = [];
    const errors: string[] = [];

    console.log(`Executing batch ${batch.id} with ${batch.commands.length} commands...`);

    try {
      if (batch.parallel && this.config.enableParallelExecution) {
        // Execute in parallel
        const results = await this.executeBatchInParallel(batch, context);
        commandResults.push(...results);
      } else {
        // Execute sequentially
        for (const command of batch.commands) {
          const result = await this.executeCommand(command, context);
          commandResults.push(result);

          if (result.status === 'FAILED' && command.critical) {
            errors.push(`Critical command failed: ${command.id}`);
            break; // Stop batch execution on critical failure
          }
        }
      }

      const duration = Date.now() - startTime;
      const successCount = commandResults.filter(r => r.status === 'SUCCESS').length;
      const successRate = (successCount / commandResults.length) * 100;

      return {
        batchId: batch.id,
        status: errors.length > 0 ? 'FAILED' : 'SUCCESS',
        commandResults,
        duration,
        successRate,
        errors,
        metrics: {
          parallelEfficiency: batch.parallel ? successRate : undefined,
          throughput: commandResults.length / (duration / 1000),
          resourceUtilization: this.calculateResourceUtilization(batch, commandResults)
        }
      };

    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        batchId: batch.id,
        status: 'FAILED',
        commandResults,
        duration,
        successRate: 0,
        errors: [errorMessage],
        metrics: {
          throughput: commandResults.length / (duration / 1000)
        }
      };
    }
  }

  /**
   * Execute batch in parallel
   */
  private async executeBatchInParallel(
    batch: CommandBatch,
    context: TemplateToCliContext
  ): Promise<CliExecutionResult[]> {
    const maxConcurrency = Math.min(batch.commands.length, this.config.maxConcurrency);
    const results: CliExecutionResult[] = [];

    for (let i = 0; i < batch.commands.length; i += maxConcurrency) {
      const batchSlice = batch.commands.slice(i, i + maxConcurrency);
      const slicePromises = batchSlice.map(command => this.executeCommand(command, context));
      const sliceResults = await Promise.all(slicePromises);
      results.push(...sliceResults);
    }

    return results;
  }

  /**
   * Execute single command
   */
  private async executeCommand(
    command: GeneratedCliCommand,
    context: TemplateToCliContext
  ): Promise<CliExecutionResult> {
    const startTime = Date.now();

    try {
      // Simulate command execution
      await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));

      const output = `Command executed: ${command.command}`;
      const duration = Date.now() - startTime;

      return {
        commandId: command.id,
        status: 'SUCCESS',
        output,
        duration,
        timestamp: new Date()
      };

    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        commandId: command.id,
        status: 'FAILED',
        error: errorMessage,
        duration,
        timestamp: new Date()
      };
    }
  }

  /**
   * Helper methods for batch analysis and optimization
   */
  private groupCommandsByType(commands: GeneratedCliCommand[]): Record<CliCommandType, GeneratedCliCommand[]> {
    const groups: Record<string, GeneratedCliCommand[]> = {};
    for (const command of commands) {
      const type = command.type;
      if (!groups[type]) groups[type] = [];
      groups[type].push(command);
    }
    return groups as Record<CliCommandType, GeneratedCliCommand[]>;
  }

  private groupCommandsByRisk(commands: GeneratedCliCommand[]): Record<string, GeneratedCliCommand[]> {
    const groups: Record<string, GeneratedCliCommand[]> = { low: [], medium: [], high: [] };
    for (const command of commands) {
      const risk = command.metadata.riskLevel;
      groups[risk].push(command);
    }
    return groups;
  }

  private groupCommandsByComplexity(commands: GeneratedCliCommand[]): Record<string, GeneratedCliCommand[]> {
    const groups: Record<string, GeneratedCliCommand[]> = { simple: [], moderate: [], complex: [] };
    for (const command of commands) {
      const complexity = command.metadata.complexity;
      groups[complexity].push(command);
    }
    return groups;
  }

  private identifyParallelizableCommands(commands: GeneratedCliCommand[]): GeneratedCliCommand[] {
    return commands.filter(cmd =>
      !cmd.critical &&
      cmd.metadata.riskLevel !== 'high' &&
      cmd.type !== 'CREATE'
    );
  }

  private calculateAverageComplexity(commands: GeneratedCliCommand[]): number {
    const complexityMap = { simple: 1, moderate: 2, complex: 3 };
    const total = commands.reduce((sum, cmd) =>
      sum + (complexityMap[cmd.metadata.complexity] || 1), 0);
    return total / commands.length;
  }

  private calculateOverallRiskLevel(commands: GeneratedCliCommand[]): 'low' | 'medium' | 'high' | 'critical' {
    const riskCounts = { low: 0, medium: 0, high: 0 };
    for (const command of commands) {
      const risk = command.metadata.riskLevel;
      if (risk !== 'critical') riskCounts[risk]++;
    }

    if (commands.some(cmd => cmd.critical)) return 'critical';
    if (riskCounts.high > 0) return 'high';
    if (riskCounts.medium > riskCounts.low) return 'medium';
    return 'low';
  }

  private sortCommandsForBatching(commands: GeneratedCliCommand[]): GeneratedCliCommand[] {
    return [...commands].sort((a, b) => {
      // Sort by criticality (non-critical first)
      if (a.critical !== b.critical) {
        return a.critical ? 1 : -1;
      }

      // Sort by risk level (low to high)
      const riskOrder = { low: 1, medium: 2, high: 3 };
      const riskA = riskOrder[a.metadata.riskLevel] || 1;
      const riskB = riskOrder[b.metadata.riskLevel] || 1;
      if (riskA !== riskB) return riskA - riskB;

      // Sort by complexity (simple to complex)
      const complexityOrder = { simple: 1, moderate: 2, complex: 3 };
      const complexityA = complexityOrder[a.metadata.complexity] || 1;
      const complexityB = complexityOrder[b.metadata.complexity] || 1;
      return complexityA - complexityB;
    });
  }

  private getCommandRiskValue(command: GeneratedCliCommand): number {
    const riskMap = { low: 1, medium: 2, high: 3, critical: 4 };
    return riskMap[command.metadata.riskLevel] || 1;
  }

  private canAddToBatch(
    command: GeneratedCliCommand,
    currentBatch: GeneratedCliCommand[],
    currentDuration: number,
    currentRisk: number
  ): boolean {
    const commandRisk = this.getCommandRiskValue(command);
    const maxBatchDuration = 30000; // 30 seconds
    const maxBatchRisk = 6; // Allow medium risk in batch

    return (currentDuration + command.metadata.estimatedDuration) <= maxBatchDuration &&
           Math.max(currentRisk, commandRisk) <= maxBatchRisk;
  }

  private calculateBatchRiskLevel(commands: GeneratedCliCommand[]): 'low' | 'medium' | 'high' | 'critical' {
    if (commands.some(cmd => cmd.critical)) return 'critical';
    if (commands.some(cmd => cmd.metadata.riskLevel === 'high')) return 'high';
    if (commands.some(cmd => cmd.metadata.riskLevel === 'medium')) return 'medium';
    return 'low';
  }

  private canExecuteBatchInParallel(commands: GeneratedCliCommand[]): boolean {
    // Cannot execute in parallel if any command is critical or high risk
    return !commands.some(cmd => cmd.critical || cmd.metadata.riskLevel === 'high');
  }

  private calculateBatchPriority(commands: GeneratedCliCommand[]): number {
    // Higher priority for batches with more commands or critical commands
    return commands.length + (commands.some(cmd => cmd.critical) ? 10 : 0);
  }

  private calculateResourceUtilization(
    batch: CommandBatch,
    results: CliExecutionResult[]
  ): number {
    // Simple resource utilization calculation
    const avgDuration = results.reduce((sum, r) => sum + r.duration, 0) / results.length;
    const theoreticalMinDuration = batch.estimatedDuration / batch.commands.length;
    return Math.min(100, (theoreticalMinDuration / avgDuration) * 100);
  }

  /**
   * Initialize optimization rules
   */
  private initializeOptimizationRules(): void {
    this.optimizationRules = [
      {
        name: 'Merge low-risk operations',
        applicable: (batches, context) => batches.length > 1,
        apply: (batches, context) => {
          // Implementation for merging low-risk operations
          return { optimized: false, batches };
        }
      },
      {
        name: 'Separate critical operations',
        applicable: (batches, context) => batches.some(b => b.riskLevel === 'critical'),
        apply: (batches, context) => {
          // Implementation for separating critical operations
          return { optimized: false, batches };
        }
      }
    ];
  }

  /**
   * Get batch execution history
   */
  public getBatchHistory(): Map<string, BatchExecutionResult> {
    return new Map(this.batchHistory);
  }

  /**
   * Clear batch history
   */
  public clearBatchHistory(): void {
    this.batchHistory.clear();
  }
}

/**
 * Command analysis result
 */
interface CommandAnalysisResult {
  totalCommands: number;
  commandsByType: Record<CliCommandType, GeneratedCliCommand[]>;
  commandsByRisk: Record<string, GeneratedCliCommand[]>;
  commandsByComplexity: Record<string, GeneratedCliCommand[]>;
  parallelizableCommands: GeneratedCliCommand[];
  criticalCommands: GeneratedCliCommand[];
  estimatedTotalDuration: number;
  averageComplexity: number;
  overallRiskLevel: 'low' | 'medium' | 'high' | 'critical';
}

/**
 * Batch optimization rule
 */
interface BatchOptimizationRule {
  name: string;
  applicable: (batches: CommandBatch[], context: TemplateToCliContext) => boolean;
  apply: (batches: CommandBatch[], context: TemplateToCliContext) => {
    optimized: boolean;
    batches: CommandBatch[];
  };
}