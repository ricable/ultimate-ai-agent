/**
 * Safe Executor - Transactional configuration management with rollback
 * Implements agentic-jujutsu patterns for distributed VCS
 */

import { v4 as uuidv4 } from 'uuid';
import {
  ConfigurationChange,
  ManagedObjectType,
  LTLFormula,
  SafetyVerification,
  SafetyViolation,
  GOAPPlan,
  GOAPAction,
} from '../core/types.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('SafeExecutor');

/**
 * Configuration commit for version control
 */
export interface ConfigCommit {
  id: string;
  parentId?: string;
  timestamp: number;
  changes: ConfigurationChange[];
  message: string;
  author: string;
  verified: boolean;
  applied: boolean;
  rolledBack: boolean;
}

/**
 * LTL Safety formula registry
 */
export interface SafetyPolicy {
  formulas: LTLFormula[];
  criticalCells: Set<string>;
  maintenanceWindows: TimeWindow[];
}

interface TimeWindow {
  start: number; // Hour of day (0-23)
  end: number;
  daysOfWeek: number[]; // 0-6, Sunday-Saturday
}

/**
 * Execution context for safety verification
 */
export interface ExecutionContext {
  cellId: string;
  emergencyCallsActive: boolean;
  currentLoad: number;
  isMaintenanceWindow: boolean;
  neighborStates: Map<string, { healthy: boolean; load: number }>;
}

/**
 * Safe Executor - Manages configuration changes with safety guarantees
 */
export class SafeExecutor {
  private commits: ConfigCommit[];
  private headCommitId?: string;
  private safetyPolicy: SafetyPolicy;
  private executionHistory: ExecutionRecord[];
  private pendingChanges: ConfigurationChange[];

  constructor() {
    this.commits = [];
    this.safetyPolicy = this.initializeSafetyPolicy();
    this.executionHistory = [];
    this.pendingChanges = [];

    logger.info('Safe executor initialized', {
      formulaCount: this.safetyPolicy.formulas.length,
    });
  }

  /**
   * Execute a GOAP plan with safety verification
   */
  async executePlan(
    plan: GOAPPlan,
    context: ExecutionContext
  ): Promise<ExecutionResult> {
    logger.info('Executing plan', {
      goalId: plan.goalId,
      actionCount: plan.actions.length,
    });

    const changes: ConfigurationChange[] = [];
    const results: ActionExecutionResult[] = [];

    for (const action of plan.actions) {
      // Verify safety before each action
      const verification = await this.verifySafety(action, context);

      if (!verification.passed) {
        logger.warn('Safety verification failed', {
          action: action.name,
          violations: verification.violations,
        });

        // Rollback any changes made so far
        if (changes.length > 0) {
          await this.rollback(changes);
        }

        return {
          success: false,
          executedActions: results,
          error: `Safety violation: ${verification.violations[0]?.violation}`,
          rollbackPerformed: changes.length > 0,
        };
      }

      // Execute the action
      const change = await this.executeAction(action, context);
      changes.push(change);
      results.push({
        action: action.name,
        change,
        success: true,
        verificationTimeMs: verification.verificationTimeMs,
      });
    }

    // Create commit for successful execution
    const commit = this.createCommit(changes, `Execute plan: ${plan.goalId}`);
    this.commits.push(commit);
    this.headCommitId = commit.id;

    logger.info('Plan executed successfully', {
      commitId: commit.id,
      changeCount: changes.length,
    });

    return {
      success: true,
      executedActions: results,
      commitId: commit.id,
      rollbackPerformed: false,
    };
  }

  /**
   * Verify action against safety policies using LTL
   */
  async verifySafety(
    action: GOAPAction,
    context: ExecutionContext
  ): Promise<SafetyVerification> {
    const startTime = performance.now();
    const violations: SafetyViolation[] = [];
    const formulasChecked: string[] = [];

    for (const formula of this.safetyPolicy.formulas) {
      formulasChecked.push(formula.id);

      const result = this.evaluateLTLFormula(formula, action, context);
      if (!result.satisfied) {
        violations.push({
          formulaId: formula.id,
          violation: result.reason,
          severity: this.getSeverityForFormula(formula),
        });
      }
    }

    const verificationTime = performance.now() - startTime;

    // Target ~420ms verification as per spec
    logger.debug('Safety verification complete', {
      action: action.name,
      passed: violations.length === 0,
      verificationTimeMs: verificationTime.toFixed(2),
    });

    return {
      actionId: uuidv4(),
      formulasChecked,
      passed: violations.length === 0,
      violations,
      verificationTimeMs: verificationTime,
    };
  }

  /**
   * Evaluate an LTL formula against action and context
   */
  private evaluateLTLFormula(
    formula: LTLFormula,
    action: GOAPAction,
    context: ExecutionContext
  ): { satisfied: boolean; reason: string } {
    // Parse and evaluate LTL formula
    // Simplified implementation - production would use proper LTL model checker

    switch (formula.id) {
      case 'NO_REBOOT_DURING_EMERGENCY':
        if (
          action.name.toLowerCase().includes('restart') &&
          context.emergencyCallsActive
        ) {
          return {
            satisfied: false,
            reason: 'Cannot restart cell during emergency calls',
          };
        }
        break;

      case 'NO_POWER_CHANGE_HIGH_LOAD':
        if (
          action.name.toLowerCase().includes('power') &&
          context.currentLoad > 0.8
        ) {
          return {
            satisfied: false,
            reason: 'Cannot change power during high load (>80%)',
          };
        }
        break;

      case 'CRITICAL_CELL_PROTECTION':
        if (
          this.safetyPolicy.criticalCells.has(context.cellId) &&
          action.risk > 0.5
        ) {
          return {
            satisfied: false,
            reason: 'High-risk action blocked on critical cell',
          };
        }
        break;

      case 'MAINTENANCE_WINDOW_REQUIRED':
        if (action.risk > 0.7 && !context.isMaintenanceWindow) {
          return {
            satisfied: false,
            reason: 'High-risk action requires maintenance window',
          };
        }
        break;

      case 'NEIGHBOR_STABILITY':
        // Check that at least one neighbor is healthy
        let healthyNeighbors = 0;
        for (const [, state] of context.neighborStates) {
          if (state.healthy) healthyNeighbors++;
        }
        if (
          action.name.toLowerCase().includes('restart') &&
          healthyNeighbors === 0
        ) {
          return {
            satisfied: false,
            reason: 'Cannot restart when no healthy neighbors available',
          };
        }
        break;
    }

    return { satisfied: true, reason: '' };
  }

  /**
   * Execute a single action
   */
  private async executeAction(
    action: GOAPAction,
    context: ExecutionContext
  ): Promise<ConfigurationChange> {
    const change: ConfigurationChange = {
      id: uuidv4(),
      cellId: context.cellId,
      managedObject: this.inferManagedObject(action.name),
      attribute: this.inferAttribute(action.name),
      oldValue: null, // Would be fetched from current config
      newValue: action.effects,
      reason: `GOAP action: ${action.name}`,
      appliedAt: Date.now(),
      status: 'applied',
    };

    logger.debug('Action executed', {
      actionName: action.name,
      changeId: change.id,
    });

    return change;
  }

  /**
   * Infer managed object type from action name
   */
  private inferManagedObject(actionName: string): ManagedObjectType {
    if (actionName.toLowerCase().includes('tilt')) return 'RetDevice';
    if (actionName.toLowerCase().includes('power')) return 'EUtranCellFDD';
    if (actionName.toLowerCase().includes('pci')) return 'NRCellDU';
    if (actionName.toLowerCase().includes('handover')) return 'ReportConfigEUtra';
    return 'EUtranCellFDD';
  }

  /**
   * Infer attribute from action name
   */
  private inferAttribute(actionName: string): string {
    if (actionName.toLowerCase().includes('tilt')) return 'electricalTilt';
    if (actionName.toLowerCase().includes('power')) return 'transmitPower';
    if (actionName.toLowerCase().includes('pci')) return 'nCI';
    if (actionName.toLowerCase().includes('handover')) return 'a3Offset';
    return 'unknown';
  }

  /**
   * Rollback changes
   */
  async rollback(changes: ConfigurationChange[]): Promise<void> {
    logger.info('Initiating rollback', { changeCount: changes.length });

    for (const change of changes.reverse()) {
      change.status = 'rolled_back';
      change.rolledBackAt = Date.now();

      // In production, this would execute actual rollback commands
      logger.debug('Change rolled back', { changeId: change.id });
    }

    this.executionHistory.push({
      timestamp: Date.now(),
      type: 'rollback',
      changeIds: changes.map((c) => c.id),
    });
  }

  /**
   * Rollback to a specific commit
   */
  async rollbackToCommit(commitId: string): Promise<boolean> {
    const targetCommit = this.commits.find((c) => c.id === commitId);
    if (!targetCommit) {
      logger.error('Commit not found', { commitId });
      return false;
    }

    // Find all commits after target
    const targetIndex = this.commits.indexOf(targetCommit);
    const commitsToRollback = this.commits.slice(targetIndex + 1);

    for (const commit of commitsToRollback.reverse()) {
      await this.rollback(commit.changes);
      commit.rolledBack = true;
    }

    this.headCommitId = commitId;
    logger.info('Rolled back to commit', { commitId });

    return true;
  }

  /**
   * Rollback to HEAD~n
   */
  async rollbackHead(n: number = 1): Promise<boolean> {
    const currentIndex = this.headCommitId
      ? this.commits.findIndex((c) => c.id === this.headCommitId)
      : this.commits.length - 1;

    const targetIndex = Math.max(0, currentIndex - n);
    const targetCommit = this.commits[targetIndex];

    if (!targetCommit) {
      logger.error('Cannot rollback, insufficient history');
      return false;
    }

    return this.rollbackToCommit(targetCommit.id);
  }

  /**
   * Create a commit for a set of changes
   */
  private createCommit(
    changes: ConfigurationChange[],
    message: string
  ): ConfigCommit {
    return {
      id: uuidv4(),
      parentId: this.headCommitId,
      timestamp: Date.now(),
      changes,
      message,
      author: 'neuro-federated-swarm',
      verified: true,
      applied: true,
      rolledBack: false,
    };
  }

  /**
   * Stage a change for later commit
   */
  stageChange(change: ConfigurationChange): void {
    this.pendingChanges.push(change);
    logger.debug('Change staged', { changeId: change.id });
  }

  /**
   * Commit all staged changes
   */
  commitStaged(message: string): ConfigCommit | null {
    if (this.pendingChanges.length === 0) {
      logger.warn('No changes to commit');
      return null;
    }

    const commit = this.createCommit([...this.pendingChanges], message);
    this.commits.push(commit);
    this.headCommitId = commit.id;
    this.pendingChanges = [];

    logger.info('Staged changes committed', { commitId: commit.id });
    return commit;
  }

  /**
   * Get commit history
   */
  getHistory(): ConfigCommit[] {
    return [...this.commits];
  }

  /**
   * Get current HEAD commit
   */
  getHead(): ConfigCommit | undefined {
    return this.commits.find((c) => c.id === this.headCommitId);
  }

  /**
   * Initialize default safety policy
   */
  private initializeSafetyPolicy(): SafetyPolicy {
    return {
      formulas: [
        {
          id: 'NO_REBOOT_DURING_EMERGENCY',
          name: 'No Reboot During Emergency',
          formula: 'G(emergency_call -> !restart)',
          description: 'A cell cannot be rebooted if it carries emergency calls',
        },
        {
          id: 'NO_POWER_CHANGE_HIGH_LOAD',
          name: 'No Power Change During High Load',
          formula: 'G(high_load -> !power_change)',
          description: 'Power changes blocked during high load conditions',
        },
        {
          id: 'CRITICAL_CELL_PROTECTION',
          name: 'Critical Cell Protection',
          formula: 'G(critical_cell -> low_risk_only)',
          description: 'Only low-risk actions allowed on critical cells',
        },
        {
          id: 'MAINTENANCE_WINDOW_REQUIRED',
          name: 'Maintenance Window Required',
          formula: 'G(high_risk -> maintenance_window)',
          description: 'High-risk actions require maintenance window',
        },
        {
          id: 'NEIGHBOR_STABILITY',
          name: 'Neighbor Stability Check',
          formula: 'G(restart -> E(healthy_neighbor))',
          description: 'At least one neighbor must be healthy for restart',
        },
      ],
      criticalCells: new Set(['CELL_HOSPITAL_001', 'CELL_AIRPORT_001']),
      maintenanceWindows: [
        { start: 2, end: 5, daysOfWeek: [0, 1, 2, 3, 4, 5, 6] }, // 2-5 AM daily
      ],
    };
  }

  /**
   * Get severity for a formula violation
   */
  private getSeverityForFormula(formula: LTLFormula): 'warning' | 'error' | 'critical' {
    const criticalFormulas = ['NO_REBOOT_DURING_EMERGENCY', 'CRITICAL_CELL_PROTECTION'];
    const errorFormulas = ['MAINTENANCE_WINDOW_REQUIRED', 'NEIGHBOR_STABILITY'];

    if (criticalFormulas.includes(formula.id)) return 'critical';
    if (errorFormulas.includes(formula.id)) return 'error';
    return 'warning';
  }

  /**
   * Add a safety formula
   */
  addSafetyFormula(formula: LTLFormula): void {
    this.safetyPolicy.formulas.push(formula);
    logger.info('Safety formula added', { formulaId: formula.id });
  }

  /**
   * Mark a cell as critical
   */
  markCellCritical(cellId: string): void {
    this.safetyPolicy.criticalCells.add(cellId);
    logger.info('Cell marked as critical', { cellId });
  }

  /**
   * Check if currently in maintenance window
   */
  isMaintenanceWindow(): boolean {
    const now = new Date();
    const hour = now.getHours();
    const day = now.getDay();

    for (const window of this.safetyPolicy.maintenanceWindows) {
      if (window.daysOfWeek.includes(day)) {
        if (window.start <= window.end) {
          if (hour >= window.start && hour < window.end) return true;
        } else {
          // Handles windows that cross midnight
          if (hour >= window.start || hour < window.end) return true;
        }
      }
    }

    return false;
  }
}

/**
 * Execution result
 */
export interface ExecutionResult {
  success: boolean;
  executedActions: ActionExecutionResult[];
  error?: string;
  commitId?: string;
  rollbackPerformed: boolean;
}

interface ActionExecutionResult {
  action: string;
  change: ConfigurationChange;
  success: boolean;
  verificationTimeMs: number;
}

interface ExecutionRecord {
  timestamp: number;
  type: 'execute' | 'rollback';
  changeIds: string[];
}

/**
 * Create a configured safe executor instance
 */
export function createSafeExecutor(): SafeExecutor {
  return new SafeExecutor();
}
