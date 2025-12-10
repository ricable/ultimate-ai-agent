/**
 * SPARC Safety Hooks - Track D: Governance
 * Physics-Aware Safety Layer with 3GPP Compliance
 *
 * Implements pre/post tool use hooks to prevent dangerous parameter changes
 * before they reach the network.
 *
 * Architecture:
 * - Pre-Tool-Use: Symbolic verification (3GPP) + Physics verification (Chaos/Interference)
 * - Post-Tool-Use: Audit logging and reflexion
 *
 * @module hooks/safety
 */

import { EventEmitter } from 'events';

/**
 * Hook action result
 */
export interface HookResult {
  action: 'allow' | 'deny';
  reason: string;
  violations?: Violation[];
  metadata?: Record<string, any>;
}

/**
 * Violation record
 */
export interface Violation {
  type: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  message: string;
  constraint?: string;
  value?: any;
  limit?: any;
}

/**
 * Tool execution context
 */
export interface ToolContext {
  cellId?: string;
  hardware?: string;
  timestamp?: string;
  agentId?: string;
  sessionId?: string;
  tools?: any;
}

/**
 * Audit trail entry
 */
export interface AuditEntry {
  timestamp: string;
  toolName: string;
  args: any;
  context: ToolContext;
  result: HookResult;
  executionTime?: number;
}

/**
 * Physics state for interference analysis
 */
export interface PhysicsState {
  interference_level: number; // dBm
  rsrp?: number;
  rsrq?: number;
  sinr?: number;
  lyapunov_exponent?: number;
}

/**
 * HookManager - Central registry for pre/post hooks
 * Mimics @anthropic-ai/claude-agent-sdk HookManager interface
 */
export class HookManager extends EventEmitter {
  private static hooks: Map<string, Array<(...args: any[]) => Promise<any>>> = new Map();

  /**
   * Register a hook for a specific event
   */
  static register(event: string, handler: (...args: any[]) => Promise<any>): void {
    if (!this.hooks.has(event)) {
      this.hooks.set(event, []);
    }
    this.hooks.get(event)!.push(handler);
    console.log(`[HOOK-MANAGER] Registered handler for event: ${event}`);
  }

  /**
   * Execute all hooks for an event
   */
  static async execute(event: string, ...args: any[]): Promise<any> {
    const handlers = this.hooks.get(event) || [];

    for (const handler of handlers) {
      const result = await handler(...args);
      // If any handler denies, stop execution
      if (result?.action === 'deny') {
        return result;
      }
    }

    return { action: 'allow' };
  }

  /**
   * Clear all hooks (useful for testing)
   */
  static clear(): void {
    this.hooks.clear();
  }
}

/**
 * PsychoSymbolicGuard Interface
 * Provides symbolic verification against 3GPP standards
 */
export class PsychoSymbolicGuardInterface {
  private constraints: any;

  constructor(constraints?: any) {
    this.constraints = constraints || this.getDefaultConstraints();
  }

  private getDefaultConstraints() {
    return {
      '3gpp_constraints': {
        bler_max: 0.1,
        power_max_dbm: 46,
        rsrp_min_dbm: -140,
        cio_max_db: 24
      },
      'hardware_limits': {
        radio_6630: {
          tx_power_max_dbm: 46,
          model: 'Ericsson Radio 6630'
        }
      },
      'physics_thresholds': {
        interference_critical_dbm: -90,
        power_boost_max_dbm: 40,
        chaos_lyapunov_threshold: 1.0
      }
    };
  }

  /**
   * Verify parameters against 3GPP standards
   */
  verify3GPP(params: any): { valid: boolean; violation?: string; violations?: Violation[] } {
    const violations: Violation[] = [];
    const constraints = this.constraints['3gpp_constraints'];

    // Check power limits (3GPP TS 38.104)
    if (params.power !== undefined || params.tx_power !== undefined) {
      const power = params.power ?? params.tx_power;
      const maxPower = constraints.power_max_dbm;

      if (power > maxPower) {
        violations.push({
          type: 'POWER_LIMIT',
          severity: 'CRITICAL',
          message: `Power ${power} dBm exceeds 3GPP max ${maxPower} dBm`,
          constraint: '3GPP TS 38.104',
          value: power,
          limit: maxPower
        });
      }
    }

    // Check BLER limits (3GPP TS 38.331)
    if (params.targetBler !== undefined) {
      const maxBler = constraints.bler_max;
      if (params.targetBler > maxBler) {
        violations.push({
          type: 'BLER_LIMIT',
          severity: 'HIGH',
          message: `Target BLER ${params.targetBler} exceeds max ${maxBler}`,
          constraint: '3GPP TS 38.331',
          value: params.targetBler,
          limit: maxBler
        });
      }
    }

    // Check CIO limits (3GPP TS 38.331)
    if (params.cio !== undefined) {
      const maxCIO = constraints.cio_max_db;
      if (Math.abs(params.cio) > maxCIO) {
        violations.push({
          type: 'CIO_LIMIT',
          severity: 'MEDIUM',
          message: `CIO ${params.cio} dB exceeds range +/- ${maxCIO} dB`,
          constraint: '3GPP TS 38.331',
          value: params.cio,
          limit: maxCIO
        });
      }
    }

    if (violations.length > 0) {
      return {
        valid: false,
        violation: violations[0].message,
        violations
      };
    }

    return { valid: true };
  }

  /**
   * Check hardware-specific limits
   */
  verifyHardwareLimits(params: any, hardware: string = 'radio_6630'): Violation[] {
    const violations: Violation[] = [];
    const hwLimits = this.constraints.hardware_limits[hardware];

    if (!hwLimits) {
      return violations;
    }

    // Check TX power against hardware max
    if (params.txPower !== undefined || params.tx_power !== undefined) {
      const txPower = params.txPower ?? params.tx_power;
      const maxTxPower = hwLimits.tx_power_max_dbm;

      if (txPower > maxTxPower) {
        violations.push({
          type: 'HW_TX_POWER',
          severity: 'CRITICAL',
          message: `TX Power ${txPower} dBm exceeds ${hwLimits.model} max ${maxTxPower} dBm`,
          value: txPower,
          limit: maxTxPower
        });
      }
    }

    return violations;
  }

  /**
   * Physics verification: Prevent dangerous operations during network instability
   */
  verifyPhysics(params: any, physicsState: PhysicsState): Violation[] {
    const violations: Violation[] = [];
    const thresholds = this.constraints.physics_thresholds;

    // Critical Rule: Block power boost during interference storm
    // Rationale: Increasing power during high interference creates a positive feedback loop
    const txPower = params.tx_power ?? params.txPower ?? params.power;

    if (txPower !== undefined && physicsState.interference_level !== undefined) {
      const interferenceHigh = physicsState.interference_level > thresholds.interference_critical_dbm;
      const powerBoostRequested = txPower > thresholds.power_boost_max_dbm;

      if (interferenceHigh && powerBoostRequested) {
        violations.push({
          type: 'INTERFERENCE_STORM',
          severity: 'CRITICAL',
          message: `Power boost prohibited during interference storm (interference: ${physicsState.interference_level} dBm > ${thresholds.interference_critical_dbm} dBm, requested power: ${txPower} dBm > ${thresholds.power_boost_max_dbm} dBm)`,
          constraint: 'Physics: Prevent positive feedback loop',
          value: {
            interference: physicsState.interference_level,
            requestedPower: txPower
          },
          limit: {
            maxInterference: thresholds.interference_critical_dbm,
            maxPowerDuringStorm: thresholds.power_boost_max_dbm
          }
        });
      }
    }

    // Chaos detection: Block changes if Lyapunov exponent indicates chaos
    if (physicsState.lyapunov_exponent !== undefined) {
      if (physicsState.lyapunov_exponent > thresholds.chaos_lyapunov_threshold) {
        violations.push({
          type: 'CHAOS_DETECTED',
          severity: 'HIGH',
          message: `System chaos detected (Lyapunov: ${physicsState.lyapunov_exponent} > ${thresholds.chaos_lyapunov_threshold}). Parameter changes blocked.`,
          constraint: 'Physics: Chaos prevention',
          value: physicsState.lyapunov_exponent,
          limit: thresholds.chaos_lyapunov_threshold
        });
      }
    }

    return violations;
  }
}

/**
 * Safety Hook System - Main Implementation
 */
export class SafetyHooks {
  private guard: PsychoSymbolicGuardInterface;
  private auditLog: AuditEntry[] = [];
  private maxAuditEntries: number = 10000;

  constructor(guard?: PsychoSymbolicGuardInterface) {
    this.guard = guard || new PsychoSymbolicGuardInterface();
  }

  /**
   * Initialize safety hooks
   */
  initialize(): void {
    console.log('[SAFETY-HOOKS] Initializing SPARC Safety Hooks...');

    // Register pre-tool-use hook
    HookManager.register('pre_tool_use', this.preToolUseHook.bind(this));

    // Register post-tool-use hook
    HookManager.register('post_tool_use', this.postToolUseHook.bind(this));

    console.log('[SAFETY-HOOKS] Safety hooks registered successfully');
  }

  /**
   * Pre-Tool-Use Hook: The Physics & Compliance Firewall
   *
   * This is the critical safety layer that prevents dangerous parameter changes
   * before they reach the network.
   */
  private async preToolUseHook(
    toolName: string,
    args: any,
    context: ToolContext
  ): Promise<HookResult> {
    const startTime = Date.now();

    console.log(`[SAFETY-HOOKS] Pre-check: ${toolName}`);

    // Only validate parameter change operations
    if (toolName !== 'execute_parameter_change' && toolName !== 'modify_cell_parameters') {
      return { action: 'allow', reason: 'Non-critical tool' };
    }

    const violations: Violation[] = [];
    const { cellId, params, hardware = 'radio_6630' } = args;

    // 1. Symbolic Verification: 3GPP Compliance Check
    console.log('[SAFETY-HOOKS] Running 3GPP compliance check...');
    const gppCheck = this.guard.verify3GPP(params);

    if (!gppCheck.valid) {
      violations.push(...(gppCheck.violations || []));
    }

    // 2. Hardware Limits Verification
    console.log('[SAFETY-HOOKS] Checking hardware limits...');
    const hwViolations = this.guard.verifyHardwareLimits(params, hardware);
    violations.push(...hwViolations);

    // 3. Physics Verification: Real-time network state analysis
    console.log('[SAFETY-HOOKS] Running physics verification...');

    // Query neighbor state for interference analysis
    let physicsState: PhysicsState = { interference_level: -120 }; // Default: No interference

    if (context.tools && cellId) {
      try {
        // In production, this would call ruvector to get actual neighbor state
        // For now, we use context data if available
        const neighborState = await this.queryNeighborState(cellId, context);
        physicsState = neighborState;
      } catch (error) {
        console.warn('[SAFETY-HOOKS] Failed to query neighbor state:', error);
        // Fail-safe: If we can't verify safety, deny the change
        return {
          action: 'deny',
          reason: 'Physics verification failed: Unable to query network state',
          metadata: { error: String(error) }
        };
      }
    }

    const physicsViolations = this.guard.verifyPhysics(params, physicsState);
    violations.push(...physicsViolations);

    // 4. Determine final action
    const hasCritical = violations.some(v => v.severity === 'CRITICAL');
    const hasHigh = violations.some(v => v.severity === 'HIGH');

    const result: HookResult = {
      action: (hasCritical || hasHigh) ? 'deny' : 'allow',
      reason: violations.length > 0
        ? `Safety violations detected: ${violations.map(v => v.type).join(', ')}`
        : 'All safety checks passed',
      violations,
      metadata: {
        toolName,
        cellId,
        hardware,
        physicsState,
        checkDuration: Date.now() - startTime
      }
    };

    // Log the decision
    this.logAuditEntry({
      timestamp: new Date().toISOString(),
      toolName,
      args,
      context,
      result,
      executionTime: Date.now() - startTime
    });

    if (result.action === 'deny') {
      console.error(`[SAFETY-HOOKS] ❌ BLOCKED: ${result.reason}`);
      violations.forEach(v => {
        console.error(`  - [${v.severity}] ${v.type}: ${v.message}`);
      });
    } else {
      console.log(`[SAFETY-HOOKS] ✅ ALLOWED: ${result.reason}`);
    }

    return result;
  }

  /**
   * Post-Tool-Use Hook: Audit & Reflexion
   *
   * Logs all tool executions for compliance and learning.
   */
  private async postToolUseHook(
    toolName: string,
    args: any,
    result: any,
    context: ToolContext
  ): Promise<void> {
    console.log(`[SAFETY-HOOKS] Post-execution logging: ${toolName}`);

    const entry: AuditEntry = {
      timestamp: new Date().toISOString(),
      toolName,
      args,
      context,
      result: {
        action: 'allow', // If we got here, it was allowed
        reason: 'Executed successfully',
        metadata: { executionResult: result }
      }
    };

    this.logAuditEntry(entry);

    // Emit event for AgentDB reflexion storage
    if (context.agentId) {
      console.log(`[SAFETY-HOOKS] Logging reflexion for agent ${context.agentId}`);
    }
  }

  /**
   * Query neighbor cell state for physics verification
   * In production, this queries ruvector for spatial topology
   */
  private async queryNeighborState(cellId: string, context: ToolContext): Promise<PhysicsState> {
    // Mock implementation - in production, this would call:
    // await context.tools.ruvector_query_neighbors(cellId)

    // For now, use context data if available or return safe defaults
    if (context.tools?.ruvector_query_neighbors) {
      return await context.tools.ruvector_query_neighbors(cellId);
    }

    // Default safe state (low interference)
    return {
      interference_level: -110, // Low interference
      rsrp: -90,
      rsrq: -10,
      sinr: 15,
      lyapunov_exponent: 0.2 // Stable
    };
  }

  /**
   * Log entry to audit trail
   */
  private logAuditEntry(entry: AuditEntry): void {
    this.auditLog.push(entry);

    // Maintain max size
    if (this.auditLog.length > this.maxAuditEntries) {
      this.auditLog.shift();
    }

    // In production, this would also write to persistent storage
    // e.g., agentdb for long-term compliance tracking
  }

  /**
   * Get audit log
   */
  getAuditLog(limit: number = 100): AuditEntry[] {
    return this.auditLog.slice(-limit);
  }

  /**
   * Get audit statistics
   */
  getAuditStats(): {
    total: number;
    allowed: number;
    denied: number;
    criticalViolations: number;
    recentEntries: AuditEntry[];
  } {
    const denied = this.auditLog.filter(e => e.result.action === 'deny').length;
    const criticalViolations = this.auditLog.filter(e =>
      e.result.violations?.some(v => v.severity === 'CRITICAL')
    ).length;

    return {
      total: this.auditLog.length,
      allowed: this.auditLog.length - denied,
      denied,
      criticalViolations,
      recentEntries: this.getAuditLog(10)
    };
  }

  /**
   * Export audit trail for compliance reporting
   */
  exportAuditTrail(): string {
    return JSON.stringify({
      generated: new Date().toISOString(),
      stats: this.getAuditStats(),
      entries: this.auditLog
    }, null, 2);
  }
}

// Singleton instance for global use
export const safetyHooks = new SafetyHooks();

// Auto-initialize on import
safetyHooks.initialize();
