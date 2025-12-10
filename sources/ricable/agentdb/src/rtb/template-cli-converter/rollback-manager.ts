/**
 * Rollback Manager
 *
 * Manages rollback capabilities for safe command execution with
 * automatic rollback generation, validation, and selective recovery.
 */

import {
  GeneratedCliCommand,
  CliCommandType,
  RollbackStrategy,
  TemplateToCliContext,
  CliExecutionResult,
  ExecutionStatus
} from './types';

/**
 * Rollback Manager Configuration
 */
export interface RollbackManagerConfig {
  /** Rollback strategy */
  strategy: 'FULL' | 'PARTIAL' | 'SELECTIVE' | 'INCREMENTAL';
  /** Enable validation after rollback */
  enableValidation: boolean;
  /** Enable selective rollback */
  enableSelectiveRollback: boolean;
  /** Maximum rollback depth */
  maxRollbackDepth: number;
  /** Rollback timeout in seconds */
  rollbackTimeout: number;
}

/**
 * Rollback plan
 */
export interface RollbackPlan {
  /** Plan identifier */
  id: string;
  /** Original command set ID */
  originalCommandSetId: string;
  /** Rollback strategy used */
  strategy: RollbackStrategy;
  /** Rollback commands */
  rollbackCommands: GeneratedCliCommand[];
  /** Execution order */
  executionOrder: string[];
  /** Validation commands */
  validationCommands: GeneratedCliCommand[];
  /** Risk assessment */
  riskAssessment: {
    rollbackRisk: 'low' | 'medium' | 'high' | 'critical';
    dataLossRisk: 'none' | 'low' | 'medium' | 'high';
    serviceImpactRisk: 'none' | 'low' | 'medium' | 'high';
  };
  /** Estimated rollback duration */
  estimatedDuration: number;
  /** Rollback metadata */
  metadata: {
    createdAt: Date;
    createdBy: string;
    description: string;
    tags: string[];
  };
}

/**
 * Rollback execution result
 */
export interface RollbackExecutionResult {
  /** Rollback plan ID */
  rollbackPlanId: string;
  /** Execution status */
  status: ExecutionStatus;
  /** Command results */
  commandResults: CliExecutionResult[];
  /** Overall duration */
  duration: number;
  /** Success rate */
  successRate: number;
  /** Validation results */
  validationResults?: CliExecutionResult[];
  /** Errors encountered */
  errors: string[];
  /** Rollback effectiveness */
  effectiveness: {
    commandsRolledBack: number;
    totalCommands: number;
    servicesRestored: string[];
    dataConsistency: 'maintained' | 'partially_maintained' | 'lost';
  };
}

/**
 * Rollback Manager Class
 */
export class RollbackManager {
  private config: RollbackManagerConfig;
  private rollbackPlans: Map<string, RollbackPlan> = new Map();
  private executionHistory: Map<string, RollbackExecutionResult> = new Map();
  private backupManager: BackupManager;

  constructor(config: RollbackManagerConfig) {
    this.config = {
      strategy: 'FULL',
      enableValidation: true,
      enableSelectiveRollback: true,
      maxRollbackDepth: 50,
      rollbackTimeout: 300,
      ...config
    };

    this.backupManager = new BackupManager();
  }

  /**
   * Generate rollback commands
   */
  public async generateRollbackCommands(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<GeneratedCliCommand[]> {
    console.log(`Generating rollback commands for ${commands.length} commands...`);

    const rollbackCommands: GeneratedCliCommand[] = [];
    const executedCommands: GeneratedCliCommand[] = [];

    // Create backups before generating rollback commands
    await this.createPreExecutionBackups(commands, context);

    // Process commands in reverse order (LIFO for rollback)
    for (let i = commands.length - 1; i >= 0; i--) {
      const command = commands[i];
      const rollbackCommand = await this.generateRollbackCommand(command, context, executedCommands);

      if (rollbackCommand) {
        rollbackCommands.push(rollbackCommand);
        executedCommands.push(command);
      }
    }

    console.log(`Generated ${rollbackCommands.length} rollback commands`);
    return rollbackCommands;
  }

  /**
   * Create rollback plan
   */
  public async createRollbackPlan(
    originalCommandSetId: string,
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext,
    description?: string
  ): Promise<RollbackPlan> {
    const planId = `rollback_${originalCommandSetId}_${Date.now()}`;
    const startTime = Date.now();

    console.log(`Creating rollback plan ${planId}...`);

    try {
      // Generate rollback commands
      const rollbackCommands = await this.generateRollbackCommands(commands, context);

      // Create validation commands
      const validationCommands = this.config.enableValidation
        ? await this.generateRollbackValidationCommands(rollbackCommands, context)
        : [];

      // Assess rollback risks
      const riskAssessment = await this.assessRollbackRisk(commands, rollbackCommands, context);

      // Calculate estimated duration
      const estimatedDuration = rollbackCommands.reduce((sum, cmd) => sum + cmd.metadata.estimatedDuration, 0);

      // Create rollback plan
      const plan: RollbackPlan = {
        id: planId,
        originalCommandSetId,
        strategy: {
          type: this.config.strategy,
          scope: this.determineRollbackScope(commands),
          order: 'REVERSE',
          preserveOnFailure: true,
          validateAfterRollback: this.config.enableValidation
        },
        rollbackCommands,
        executionOrder: rollbackCommands.map(cmd => cmd.id),
        validationCommands,
        riskAssessment,
        estimatedDuration,
        metadata: {
          createdAt: new Date(),
          createdBy: 'TemplateToCliConverter',
          description: description || `Rollback for command set ${originalCommandSetId}`,
          tags: this.generateRollbackTags(commands)
        }
      };

      // Store plan
      this.rollbackPlans.set(planId, plan);

      const duration = Date.now() - startTime;
      console.log(`Rollback plan created in ${duration}ms: ${rollbackCommands.length} commands`);

      return plan;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`Rollback plan creation failed: ${errorMessage}`);
      throw new Error(`Rollback plan creation failed: ${errorMessage}`);
    }
  }

  /**
   * Execute rollback plan
   */
  public async executeRollback(
    rollbackPlanId: string,
    context: TemplateToCliContext,
    options?: {
      dryRun?: boolean;
      selectiveCommands?: string[];
      continueOnError?: boolean;
    }
  ): Promise<RollbackExecutionResult> {
    const plan = this.rollbackPlans.get(rollbackPlanId);
    if (!plan) {
      throw new Error(`Rollback plan not found: ${rollbackPlanId}`);
    }

    console.log(`Executing rollback plan ${rollbackPlanId}...`);

    const startTime = Date.now();
    const commandResults: CliExecutionResult[] = [];
    const errors: string[] = [];

    try {
      let commandsToExecute = plan.rollbackCommands;

      // Apply selective rollback if specified
      if (options?.selectiveCommands) {
        commandsToExecute = plan.rollbackCommands.filter(cmd =>
          options.selectiveCommands!.includes(cmd.id)
        );
      }

      // Execute rollback commands
      for (const command of commandsToExecute) {
        const result = await this.executeRollbackCommand(command, context, options?.dryRun);
        commandResults.push(result);

        if (result.status === 'FAILED' && !options?.continueOnError) {
          errors.push(`Rollback command failed: ${command.id} - ${result.error}`);
          break;
        }
      }

      // Execute validation commands if enabled
      let validationResults: CliExecutionResult[] | undefined;
      if (this.config.enableValidation && plan.validationCommands.length > 0) {
        validationResults = [];
        for (const validationCommand of plan.validationCommands) {
          const result = await this.executeRollbackCommand(validationCommand, context, options?.dryRun);
          validationResults.push(result);
        }
      }

      // Calculate effectiveness
      const effectiveness = this.calculateRollbackEffectiveness(
        commandsToExecute,
        commandResults,
        validationResults
      );

      const duration = Date.now() - startTime;
      const successCount = commandResults.filter(r => r.status === 'SUCCESS').length;
      const successRate = (successCount / commandResults.length) * 100;

      const result: RollbackExecutionResult = {
        rollbackPlanId,
        status: errors.length > 0 ? 'PARTIAL' : 'SUCCESS',
        commandResults,
        duration,
        successRate,
        validationResults,
        errors,
        effectiveness
      };

      // Store execution history
      this.executionHistory.set(rollbackPlanId, result);

      console.log(`Rollback completed in ${duration}ms: ${successRate.toFixed(1)}% success rate`);
      return result;

    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      return {
        rollbackPlanId,
        status: 'FAILED',
        commandResults,
        duration,
        successRate: 0,
        errors: [errorMessage],
        effectiveness: {
          commandsRolledBack: 0,
          totalCommands: plan.rollbackCommands.length,
          servicesRestored: [],
          dataConsistency: 'lost'
        }
      };
    }
  }

  /**
   * Generate rollback command for a single command
   */
  private async generateRollbackCommand(
    command: GeneratedCliCommand,
    context: TemplateToCliContext,
    executedCommands: GeneratedCliCommand[]
  ): Promise<GeneratedCliCommand | null> {
    switch (command.type) {
      case 'CREATE':
        return this.generateCreateRollback(command, context);
      case 'SET':
        return this.generateSetRollback(command, context, executedCommands);
      case 'DELETE':
        return this.generateDeleteRollback(command, context);
      default:
        console.warn(`No rollback strategy for command type: ${command.type}`);
        return null;
    }
  }

  /**
   * Generate rollback for CREATE command
   */
  private generateCreateRollback(
    command: GeneratedCliCommand,
    context: TemplateToCliContext
  ): GeneratedCliCommand {
    // For CREATE commands, rollback is DELETE
    const rollbackCommand = command.command.replace('cmedit create', 'cmedit delete');

    return {
      id: `rollback_${command.id}`,
      type: 'DELETE',
      command: rollbackCommand,
      description: `Rollback: Delete ${command.description}`,
      targetFdn: command.targetFdn,
      timeout: this.config.rollbackTimeout,
      critical: false,
      metadata: {
        category: 'rollback',
        complexity: 'simple',
        riskLevel: 'medium',
        estimatedDuration: 2000
      }
    };
  }

  /**
   * Generate rollback for SET command
   */
  private generateSetRollback(
    command: GeneratedCliCommand,
    context: TemplateToCliContext,
    executedCommands: GeneratedCliCommand[]
  ): GeneratedCliCommand | null {
    // For SET commands, we need to restore previous values
    // This would require having stored the original values

    if (!command.parameters) {
      return null;
    }

    // Create rollback command to restore original values
    const originalValues = this.getOriginalParameterValues(command, executedCommands);
    if (!originalValues) {
      console.warn(`Cannot generate rollback for SET command ${command.id}: original values not available`);
      return null;
    }

    const paramList = Object.entries(originalValues)
      .map(([key, value]) => `${key}=${value}`)
      .join(',');

    const rollbackCommand = `cmedit set ${context.target.nodeId} ${command.targetFdn || ''} ${paramList}`;

    return {
      id: `rollback_${command.id}`,
      type: 'SET',
      command: rollbackCommand,
      description: `Rollback: Restore original values for ${command.description}`,
      targetFdn: command.targetFdn,
      parameters: originalValues,
      timeout: this.config.rollbackTimeout,
      critical: false,
      metadata: {
        category: 'rollback',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1500
      }
    };
  }

  /**
   * Generate rollback for DELETE command
   */
  private generateDeleteRollback(
    command: GeneratedCliCommand,
    context: TemplateToCliContext
  ): GeneratedCliCommand {
    // For DELETE commands, rollback is CREATE with backup data
    // This would require having stored the object data before deletion

    const backupData = this.backupManager.getBackup(command.targetFdn || '');
    if (!backupData) {
      console.warn(`Cannot generate rollback for DELETE command ${command.id}: backup data not available`);
      // Return a placeholder command
      return {
        id: `rollback_${command.id}`,
        type: 'CREATE',
        command: `# Rollback placeholder for ${command.description}`,
        description: `Rollback: Recreate ${command.description} (manual intervention required)`,
        targetFdn: command.targetFdn,
        timeout: this.config.rollbackTimeout,
        critical: false,
        metadata: {
          category: 'rollback',
          complexity: 'complex',
          riskLevel: 'high',
          estimatedDuration: 5000
        }
      };
    }

    // Create recreation command from backup data
    const createCommand = this.generateCreateCommandFromBackup(backupData, context);

    return {
      id: `rollback_${command.id}`,
      type: 'CREATE',
      command: createCommand,
      description: `Rollback: Recreate ${command.description} from backup`,
      targetFdn: command.targetFdn,
      timeout: this.config.rollbackTimeout,
      critical: false,
      metadata: {
        category: 'rollback',
        complexity: 'complex',
        riskLevel: 'high',
        estimatedDuration: 5000
      }
    };
  }

  /**
   * Generate rollback validation commands
   */
  private async generateRollbackValidationCommands(
    rollbackCommands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<GeneratedCliCommand[]> {
    const validationCommands: GeneratedCliCommand[] = [];

    for (const rollbackCommand of rollbackCommands) {
      if (rollbackCommand.targetFdn) {
        const validationCommand = {
          id: `validate_rollback_${rollbackCommand.id}`,
          type: 'GET' as CliCommandType,
          command: `cmedit get ${context.target.nodeId} ${rollbackCommand.targetFdn} -s`,
          description: `Validate rollback: ${rollbackCommand.description}`,
          targetFdn: rollbackCommand.targetFdn,
          expectedOutput: ['syncStatus=SYNCHRONIZED'],
          timeout: 30,
          critical: false,
          metadata: {
            category: 'validation',
            complexity: 'simple',
            riskLevel: 'low',
            estimatedDuration: 1000
          }
        };
        validationCommands.push(validationCommand);
      }
    }

    return validationCommands;
  }

  /**
   * Assess rollback risk
   */
  private async assessRollbackRisk(
    originalCommands: GeneratedCliCommand[],
    rollbackCommands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<{
    rollbackRisk: 'low' | 'medium' | 'high' | 'critical';
    dataLossRisk: 'none' | 'low' | 'medium' | 'high';
    serviceImpactRisk: 'none' | 'low' | 'medium' | 'high';
  }> {
    let rollbackRiskScore = 0;
    let dataLossRiskScore = 0;
    let serviceImpactRiskScore = 0;

    // Assess rollback risk based on command types
    for (const command of rollbackCommands) {
      if (command.type === 'DELETE') {
        rollbackRiskScore += 3;
        dataLossRiskScore += 2;
      } else if (command.type === 'CREATE') {
        rollbackRiskScore += 2;
        serviceImpactRiskScore += 1;
      }
    }

    // Assess based on criticality
    const criticalCommands = originalCommands.filter(cmd => cmd.critical);
    rollbackRiskScore += criticalCommands.length * 2;
    serviceImpactRiskScore += criticalCommands.length;

    // Assess based on target objects
    const criticalTargets = ['ManagedElement', 'ENBFunction', 'NRCellCU'];
    for (const command of originalCommands) {
      if (command.targetFdn && criticalTargets.some(target =>
        command.targetFdn!.includes(target))) {
        rollbackRiskScore += 2;
        serviceImpactRiskScore += 3;
      }
    }

    // Convert scores to risk levels
    const getRiskLevel = (score: number): 'none' | 'low' | 'medium' | 'high' => {
      if (score === 0) return 'none';
      if (score <= 3) return 'low';
      if (score <= 7) return 'medium';
      return 'high';
    };

    return {
      rollbackRisk: rollbackRiskScore > 5 ? 'critical' : getRiskLevel(rollbackRiskScore),
      dataLossRisk: getRiskLevel(dataLossRiskScore),
      serviceImpactRisk: getRiskLevel(serviceImpactRiskScore)
    };
  }

  /**
   * Create pre-execution backups
   */
  private async createPreExecutionBackups(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<void> {
    for (const command of commands) {
      if (command.type === 'DELETE' || (command.type === 'SET' && command.targetFdn)) {
        await this.backupManager.createBackup(command.targetFdn || '', context);
      }
    }
  }

  /**
   * Execute rollback command
   */
  private async executeRollbackCommand(
    command: GeneratedCliCommand,
    context: TemplateToCliContext,
    dryRun: boolean = false
  ): Promise<CliExecutionResult> {
    const startTime = Date.now();

    try {
      let finalCommand = command.command;

      if (dryRun) {
        finalCommand = `# DRY RUN: ${finalCommand}`;
      }

      // Simulate command execution
      await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));

      const output = dryRun
        ? `DRY RUN: ${finalCommand}`
        : `Rollback command executed: ${finalCommand}`;

      const duration = Date.now() - startTime;

      return {
        commandId: command.id,
        status: dryRun ? 'SUCCESS' : 'SUCCESS',
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
   * Helper methods
   */
  private getOriginalParameterValues(
    command: GeneratedCliCommand,
    executedCommands: GeneratedCliCommand[]
  ): Record<string, any> | null {
    // This would typically retrieve values from a backup or snapshot
    // For now, return a placeholder
    const placeholderValues: Record<string, any> = {};

    if (command.parameters) {
      for (const paramName of Object.keys(command.parameters)) {
        // Placeholder - would retrieve actual original values
        placeholderValues[paramName] = 'ORIGINAL_VALUE';
      }
    }

    return Object.keys(placeholderValues).length > 0 ? placeholderValues : null;
  }

  private generateCreateCommandFromBackup(backupData: any, context: TemplateToCliContext): string {
    // This would generate a CREATE command from backup data
    // For now, return a placeholder
    return `cmedit create ${context.target.nodeId} # CREATE command from backup data`;
  }

  private determineRollbackScope(commands: GeneratedCliCommand[]): 'ALL' | 'FAILED' | 'CRITICAL' | 'CUSTOM' {
    const hasCriticalCommands = commands.some(cmd => cmd.critical);
    return hasCriticalCommands ? 'CRITICAL' : 'ALL';
  }

  private generateRollbackTags(commands: GeneratedCliCommand[]): string[] {
    const tags = new Set<string>();

    for (const command of commands) {
      tags.add(command.type);
      tags.add(command.metadata.category);
    }

    return Array.from(tags);
  }

  private calculateRollbackEffectiveness(
    commandsExecuted: GeneratedCliCommand[],
    commandResults: CliExecutionResult[],
    validationResults?: CliExecutionResult[]
  ): {
    commandsRolledBack: number;
    totalCommands: number;
    servicesRestored: string[];
    dataConsistency: 'maintained' | 'partially_maintained' | 'lost';
  } {
    const successCount = commandResults.filter(r => r.status === 'SUCCESS').length;
    const validationSuccessCount = validationResults
      ? validationResults.filter(r => r.status === 'SUCCESS').length
      : 0;

    let dataConsistency: 'maintained' | 'partially_maintained' | 'lost' = 'maintained';

    if (successCount < commandsExecuted.length) {
      dataConsistency = 'partially_maintained';
    }

    if (validationResults && validationSuccessCount < validationResults.length) {
      dataConsistency = 'lost';
    }

    return {
      commandsRolledBack: successCount,
      totalCommands: commandsExecuted.length,
      servicesRestored: [], // Would determine from actual services affected
      dataConsistency
    };
  }

  /**
   * Get rollback plan
   */
  public getRollbackPlan(planId: string): RollbackPlan | undefined {
    return this.rollbackPlans.get(planId);
  }

  /**
   * Get all rollback plans
   */
  public getAllRollbackPlans(): Map<string, RollbackPlan> {
    return new Map(this.rollbackPlans);
  }

  /**
   * Get execution history
   */
  public getExecutionHistory(planId?: string): Map<string, RollbackExecutionResult> {
    if (planId) {
      const history = this.executionHistory.get(planId);
      return history ? new Map([[planId, history]]) : new Map();
    }
    return new Map(this.executionHistory);
  }

  /**
   * Delete rollback plan
   */
  public deleteRollbackPlan(planId: string): boolean {
    return this.rollbackPlans.delete(planId);
  }

  /**
   * Clear all rollback plans
   */
  public clearAllRollbackPlans(): void {
    this.rollbackPlans.clear();
    this.executionHistory.clear();
    this.backupManager.clearAllBackups();
  }
}

/**
 * Backup Manager
 *
 * Manages backup creation and restoration for rollback operations.
 */
class BackupManager {
  private backups: Map<string, BackupData> = new Map();

  public async createBackup(targetFdn: string, context: TemplateToCliContext): Promise<void> {
    console.log(`Creating backup for ${targetFdn}...`);

    const backupData: BackupData = {
      targetFdn,
      createdAt: new Date(),
      data: await this.captureCurrentState(targetFdn, context),
      checksum: this.generateChecksum(targetFdn)
    };

    this.backups.set(targetFdn, backupData);
  }

  public getBackup(targetFdn: string): BackupData | undefined {
    return this.backups.get(targetFdn);
  }

  public clearAllBackups(): void {
    this.backups.clear();
  }

  private async captureCurrentState(targetFdn: string, context: TemplateToCliContext): Promise<any> {
    // This would execute a GET command to capture current state
    // For now, return placeholder data
    return {
      targetFdn,
      parameters: {},
      timestamp: new Date()
    };
  }

  private generateChecksum(targetFdn: string): string {
    // Generate a simple checksum
    return Buffer.from(targetFdn + Date.now()).toString('base64').substring(0, 16);
  }
}

/**
 * Backup data structure
 */
interface BackupData {
  targetFdn: string;
  createdAt: Date;
  data: any;
  checksum: string;
}