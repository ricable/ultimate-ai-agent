/**
 * Action Executor for Closed-Loop Optimization
 * Handles safe execution and rollback of optimization actions
 */

import { EventEmitter } from 'events';
import { OptimizationAction } from '../types/optimization';

export interface ActionExecutorConfig {
  maxConcurrentActions: number; // Maximum concurrent actions
  timeout: number; // Timeout per action in milliseconds
  rollbackEnabled: boolean; // Enable rollback on failure
  retryPolicy?: {
    maxRetries: number;
    delayMs: number;
    backoffMultiplier: number;
  };
}

export interface ActionResult {
  success: boolean;
  actionId: string;
  executionTime: number;
  result: any;
  error?: string;
  rollbackAttempted: boolean;
  rollbackSuccessful?: boolean;
  resourceUtilization: {
    cpu: number;
    memory: number;
    network: number;
  };
}

export interface ExecutionResult {
  successful: number;
  failed: number;
  totalExecutionTime: number;
  resourceUtilization: {
    cpu: number;
    memory: number;
    network: number;
  };
  results: ActionResult[];
}

export interface ActionContext {
  timestamp: number;
  executionId: string;
  agentId: string;
  priority: number;
  metadata: Record<string, any>;
}

export class ActionExecutor extends EventEmitter {
  private config: ActionExecutorConfig;
  private activeExecutions: Map<string, Promise<ActionResult>> = new Map();
  private executionHistory: ActionResult[] = [];
  private actionLog: Map<string, ActionContext> = new Map();

  constructor(config: ActionExecutorConfig) {
    super();
    this.config = {
      retryPolicy: {
        maxRetries: 3,
        delayMs: 1000,
        backoffMultiplier: 2
      },
      ...config
    };
  }

  /**
   * Execute optimization actions with concurrency control
   */
  async executeActions(actions: OptimizationAction[]): Promise<ExecutionResult> {
    if (actions.length === 0) {
      throw new Error('No actions to execute');
    }

    if (actions.length > this.config.maxConcurrentActions) {
      throw new Error(`Too many concurrent actions: ${actions.length} > ${this.config.maxConcurrentActions}`);
    }

    const startTime = Date.now();
    const executionPromises: Promise<ActionResult>[] = [];
    const results: ActionResult[] = [];

    // Create execution context for each action
    actions.forEach(action => {
      const context: ActionContext = {
        timestamp: Date.now(),
        executionId: `exec-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        agentId: 'optimization-engine',
        priority: action.type === 'emergency' ? 1 : action.type === 'optimize-power' ? 2 : 3,
        metadata: {
          actionType: action.type,
          target: action.target,
          parameters: action.parameters
        }
      };

      this.actionLog.set(action.id, context);
      executionPromises.push(this.executeSingleAction(action, context));
    });

    // Execute actions concurrently
    const executionResults = await Promise.allSettled(executionPromises);

    // Process results
    let successful = 0;
    let failed = 0;
    let totalResourceUtilization = { cpu: 0, memory: 0, network: 0 };

    for (let i = 0; i < executionResults.length; i++) {
      const result = executionResults[i];
      const action = actions[i];

      if (result.status === 'fulfilled') {
        successful++;
        totalResourceUtilization = this.addResourceUtilization(
          totalResourceUtilization,
          result.value.resourceUtilization
        );
        results.push(result.value);
      } else {
        failed++;
        const errorResult: ActionResult = {
          success: false,
          actionId: action.id,
          executionTime: 0,
          result: null,
          error: result.reason.message || 'Unknown error',
          rollbackAttempted: false,
          resourceUtilization: { cpu: 0, memory: 0, network: 0 }
        };
        results.push(errorResult);
        totalResourceUtilization = this.addResourceUtilization(
          totalResourceUtilization,
          errorResult.resourceUtilization
        );
      }
    }

    const totalExecutionTime = Date.now() - startTime;

    // Store in history
    results.forEach(result => {
      this.executionHistory.push(result);
    });

    // Keep only last 1000 results
    if (this.executionHistory.length > 1000) {
      this.executionHistory = this.executionHistory.slice(-1000);
    }

    this.emit('executionCompleted', {
      totalActions: actions.length,
      successful,
      failed,
      totalExecutionTime,
      successRate: successful / actions.length
    });

    return {
      successful,
      failed,
      totalExecutionTime,
      resourceUtilization: totalResourceUtilization,
      results
    };
  }

  /**
   * Execute a single action with retry logic
   */
  private async executeSingleAction(
    action: OptimizationAction,
    context: ActionContext
  ): Promise<ActionResult> {
    const startTime = Date.now();
    let attempt = 0;

    while (attempt <= this.config.retryPolicy.maxRetries) {
      try {
        // Check if execution should be cancelled
        if (this.shouldCancelExecution(action.id)) {
          throw new Error('Execution cancelled');
        }

        this.emit('actionStarted', { actionId: action.id, attempt: attempt + 1 });

        // Execute the action
        const result = await this.performActionExecution(action, context);

        // Validate result
        if (!this.validateActionResult(action, result)) {
          throw new Error(`Action validation failed: ${action.id}`);
        }

        const executionTime = Date.now() - startTime;
        const resourceUtilization = this.calculateResourceUtilization(action);

        this.emit('actionCompleted', { actionId: action.id, success: true });

        return {
          success: true,
          actionId: action.id,
          executionTime,
          result,
          rollbackAttempted: false,
          resourceUtilization
        };

      } catch (error) {
        attempt++;

        if (attempt <= this.config.retryPolicy.maxRetries) {
          const delay = this.config.retryPolicy.delayMs * Math.pow(
            this.config.retryPolicy.backoffMultiplier,
            attempt - 1
          );

          this.emit('actionFailed', {
            actionId: action.id,
            attempt,
            error: error.message,
            nextRetry: delay
          });

          await this.delay(delay);
          continue;
        }

        // All retries exhausted
        const executionTime = Date.now() - startTime;
        const resourceUtilization = this.calculateResourceUtilization(action);

        this.emit('actionFailed', {
          actionId: action.id,
          attempt,
          error: error.message,
          final: true
        });

        // Attempt rollback if enabled
        let rollbackSuccessful = false;
        if (this.config.rollbackEnabled && action.rollbackSupported) {
          rollbackSuccessful = await this.attemptRollback(action, error as Error);
        }

        return {
          success: false,
          actionId: action.id,
          executionTime,
          result: null,
          error: error.message,
          rollbackAttempted: true,
          rollbackSuccessful,
          resourceUtilization
        };
      }
    }

    // This should never be reached due to the while loop, but TypeScript requires it
    throw new Error('Max retries exceeded');
  }

  /**
   * Perform actual action execution
   */
  private async performActionExecution(
    action: OptimizationAction,
    context: ActionContext
  ): Promise<any> {
    // Simulate action execution based on action type
    const actionHandlers = {
      'parameter-update': this.handleParameterUpdate,
      'feature-activation': this.handleFeatureActivation,
      'power-adjustment': this.handlePowerAdjustment,
      'handover-optimization': this.handleHandoverOptimization,
      'coverage-optimization': this.handleCoverageOptimization,
      'capacity-adjustment': this.handleCapacityAdjustment
    };

    const handler = actionHandlers[action.type as keyof typeof actionHandlers] || this.handleGenericAction;

    return await handler(action, context);
  }

  /**
   * Handle parameter update actions
   */
  private async handleParameterUpdate(
    action: OptimizationAction,
    context: ActionContext
  ): Promise<any> {
    // Simulate parameter update
    await this.delay(100 + Math.random() * 200); // 100-300ms simulation

    return {
      updatedParameters: action.parameters,
      target: action.target,
      timestamp: Date.now(),
      confidence: 0.95,
      validation: 'passed'
    };
  }

  /**
   * Handle feature activation actions
   */
  private async handleFeatureActivation(
    action: OptimizationAction,
    context: ActionContext
  ): Promise<any> {
    await this.delay(200 + Math.random() * 300); // 200-500ms simulation

    return {
      feature: action.parameters.featureId,
      state: 'activated',
      timestamp: Date.now(),
      impact: {
        performance: action.parameters.expectedImprovement || 0.1,
        energy: action.parameters.energyImpact || 0.05
      }
    };
  }

  /**
   * Handle power adjustment actions
   */
  private async handlePowerAdjustment(
    action: OptimizationAction,
    context: ActionContext
  ): Promise<any> {
    await this.delay(50 + Math.random() * 150); // 50-200ms simulation

    return {
      powerLevel: action.parameters.powerLevel,
      energySaved: action.parameters.energyReduction,
      coverageImpact: Math.random() * 0.1 - 0.05, // -5% to +5%
      timestamp: Date.now()
    };
  }

  /**
   * Handle handover optimization actions
   */
  private async handleHandoverOptimization(
    action: OptimizationAction,
    context: ActionContext
  ): Promise<any> {
    await this.delay(300 + Math.random() * 400); // 300-700ms simulation

    return {
      optimizedHandovers: action.parameters.targetCells || [],
      latencyImprovement: action.parameters.expectedLatencyReduction || 0.15,
      successRate: 0.98,
      timestamp: Date.now()
    };
  }

  /**
   * Handle coverage optimization actions
   */
  private async handleCoverageOptimization(
    action: OptimizationAction,
    context: ActionContext
  ): Promise<any> {
    await this.delay(400 + Math.random() * 500); // 400-900ms simulation

    return {
      coverageArea: action.parameters.area,
      signalStrength: action.parameters.targetSignal,
      interferenceReduction: action.parameters.interferenceReduction || 0.2,
      timestamp: Date.now()
    };
  }

  /**
   * Handle capacity adjustment actions
   */
  private async handleCapacityAdjustment(
    action: OptimizationAction,
    context: ActionContext
  ): Promise<any> {
    await this.delay(250 + Math.random() * 350); // 250-600ms simulation

    return {
      capacityAdded: action.parameters.capacityUnits,
      utilization: action.parameters.targetUtilization,
      throughputImprovement: action.parameters.throughputImprovement || 0.12,
      timestamp: Date.now()
    };
  }

  /**
   * Handle generic actions
   */
  private async handleGenericAction(
    action: OptimizationAction,
    context: ActionContext
  ): Promise<any> {
    await this.delay(100 + Math.random() * 400); // 100-500ms simulation

    return {
      action: action.type,
      target: action.target,
      completed: true,
      timestamp: Date.now(),
      result: 'success'
    };
  }

  /**
   * Attempt to rollback a failed action
   */
  private async attemptRollback(
    action: OptimizationAction,
    error: Error
  ): Promise<boolean> {
    try {
      this.emit('rollbackStarted', { actionId: action.id, error: error.message });

      // Simulate rollback based on action type
      await this.delay(100 + Math.random() * 200);

      // Simulate successful rollback
      this.emit('rollbackCompleted', { actionId: action.id, success: true });
      return true;

    } catch (rollbackError) {
      this.emit('rollbackFailed', {
        actionId: action.id,
        error: rollbackError.message
      });
      return false;
    }
  }

  /**
   * Validate action result
   */
  private validateActionResult(
    action: OptimizationAction,
    result: any
  ): boolean {
    // Basic validation - check if result exists and has required fields
    if (!result || typeof result !== 'object') {
      return false;
    }

    // Check for success indicator
    if (result.success !== undefined && !result.success) {
      return false;
    }

    // Check for timestamp
    if (!result.timestamp) {
      return false;
    }

    return true;
  }

  /**
   * Calculate resource utilization for action
   */
  private calculateResourceUtilization(action: OptimizationAction): {
    cpu: number;
    memory: number;
    network: number;
  } {
    const baseResources = {
      cpu: 0.1,
      memory: 0.05,
      network: 0.02
    };

    // Scale based on action complexity
    const complexityMultiplier = {
      'parameter-update': 1.0,
      'feature-activation': 1.5,
      'power-adjustment': 0.8,
      'handover-optimization': 2.0,
      'coverage-optimization': 2.5,
      'capacity-adjustment': 1.8
    };

    const multiplier = complexityMultiplier[action.type as keyof typeof complexityMultiplier] || 1.0;

    return {
      cpu: Math.min(1.0, baseResources.cpu * multiplier),
      memory: Math.min(1.0, baseResources.memory * multiplier),
      network: Math.min(1.0, baseResources.network * multiplier)
    };
  }

  /**
   * Add resource utilization values
   */
  private addResourceUtilization(
    a: { cpu: number; memory: number; network: number },
    b: { cpu: number; memory: number; network: number }
  ): { cpu: number; memory: number; network: number } {
    return {
      cpu: Math.min(1.0, a.cpu + b.cpu),
      memory: Math.min(1.0, a.memory + b.memory),
      network: Math.min(1.0, a.network + b.network)
    };
  }

  /**
   * Check if execution should be cancelled
   */
  private shouldCancelExecution(actionId: string): boolean {
    // In a real implementation, this would check for cancellation signals
    return false;
  }

  /**
   * Helper function for delay
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get execution history
   */
  getExecutionHistory(): ActionResult[] {
    return [...this.executionHistory];
  }

  /**
   * Get current active executions
   */
  getActiveExecutions(): string[] {
    return Array.from(this.activeExecutions.keys());
  }

  /**
   * Shutdown action executor
   */
  shutdown(): void {
    // Cancel all active executions
    for (const executionId of this.activeExecutions.keys()) {
      // In a real implementation, this would properly cancel the execution
    }

    this.activeExecutions.clear();
    this.actionLog.clear();
    this.emit('shutdown');
  }
}