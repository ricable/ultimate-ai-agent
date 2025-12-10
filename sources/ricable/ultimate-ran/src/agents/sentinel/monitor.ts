/**
 * Sentinel Agent - System Guardian
 *
 * Real-time chaos detection and emergency intervention coordinator.
 * Implements circuit breaker pattern with Lyapunov monitoring.
 */

import { CircuitBreaker, CircuitState } from './circuit-breaker.js';
import { ChaosDetector, ChaosEvent, ChaosLevel } from './chaos-detector.js';
import { InterventionManager, InterventionType } from './intervention-manager.js';

export interface SentinelConfig {
  circuitBreaker?: CircuitBreaker;
  chaosDetector?: ChaosDetector;
  interventionManager?: InterventionManager;
  thresholds?: {
    lyapunovCritical: number;
    systemStability: number;
    iotMaxDbm: number;
  };
}

export interface SentinelStatus {
  circuitState: CircuitState;
  chaosLevel: ChaosLevel;
  interventionCount: number;
  lastIntervention: number;
}

export interface Agent {
  id: string;
  halt: () => Promise<void> | void;
}

export class SentinelAgent {
  private circuitBreaker: CircuitBreaker;
  private chaosDetector: ChaosDetector;
  private interventionManager: InterventionManager;
  private thresholds: {
    lyapunovCritical: number;
    systemStability: number;
    iotMaxDbm: number;
  };
  private registeredAgents: Agent[] = [];
  private interventionCount: number = 0;
  private lastInterventionTime: number = 0;

  constructor(config: SentinelConfig) {
    this.circuitBreaker = config.circuitBreaker || new CircuitBreaker({
      failureThreshold: 5,
      successThreshold: 2,
      timeout: 60000
    });

    this.thresholds = config.thresholds || {
      lyapunovCritical: 0.1,
      systemStability: 0.95,
      iotMaxDbm: -105
    };

    this.chaosDetector = config.chaosDetector || new ChaosDetector({
      lyapunovCritical: this.thresholds.lyapunovCritical,
      systemStability: this.thresholds.systemStability,
      iotMaxDbm: this.thresholds.iotMaxDbm,
      monitoringInterval: 1000
    });

    this.interventionManager = config.interventionManager || new InterventionManager();
  }

  async start(): Promise<void> {
    // Start monitoring
    this.chaosDetector.startMonitoring();

    // Register chaos event handler
    this.chaosDetector.on('chaos', this.handleChaosEvent.bind(this));
  }

  async stop(): Promise<void> {
    // Stop monitoring
    this.chaosDetector.stopMonitoring();

    // Clear event listeners
    this.chaosDetector.removeAllListeners();

    // Reset circuit breaker
    this.circuitBreaker.reset();
  }

  async handleChaosEvent(event: ChaosEvent): Promise<void> {
    // System halt on critical chaos - highest priority
    if (event.level === ChaosLevel.CRITICAL) {
      this.circuitBreaker.open();

      // Halt all registered agents
      await this.haltAllAgents();

      // Broadcast emergency halt
      await this.interventionManager.broadcastEmergencyHalt();
      await this.interventionManager.broadcastToSwarm({
        type: 'EMERGENCY_HALT',
        reason: 'CRITICAL_CHAOS_DETECTED',
        chaosLevel: event.level,
        timestamp: Date.now()
      });

      // Trigger system halt intervention
      await this.interventionManager.triggerIntervention(InterventionType.SYSTEM_HALT);
      this.interventionCount++;
      this.lastInterventionTime = Date.now();

      // Log to AgentDB
      await this.interventionManager.logToAgentDB({
        event: InterventionType.SYSTEM_HALT,
        lyapunovExponent: event.lyapunovExponent,
        systemStability: event.systemStability,
        timestamp: event.timestamp
      });

      return;
    }

    // Determine specific intervention type based on threshold violations
    let interventionType: InterventionType | null = null;

    // Check for critical Lyapunov
    if (event.lyapunovExponent > this.thresholds.lyapunovCritical) {
      interventionType = InterventionType.LYAPUNOV_CRITICAL;
    }
    // Check for low stability
    else if (event.systemStability < this.thresholds.systemStability) {
      interventionType = InterventionType.STABILITY_LOW;
    }
    // Check for high interference
    else if (event.interferenceLevel && event.interferenceLevel > this.thresholds.iotMaxDbm) {
      interventionType = InterventionType.INTERFERENCE_HIGH;
    }

    // Trigger intervention if needed
    if (interventionType) {
      await this.interventionManager.triggerIntervention(interventionType);
      this.interventionCount++;
      this.lastInterventionTime = Date.now();

      // Log to AgentDB
      await this.interventionManager.logToAgentDB({
        event: interventionType,
        lyapunovExponent: event.lyapunovExponent,
        systemStability: event.systemStability,
        timestamp: event.timestamp
      });
    }
  }

  async checkRecovery(): Promise<void> {
    if (this.circuitBreaker.shouldAttemptReset()) {
      this.circuitBreaker.halfOpen();
    }
  }

  async testRecovery(): Promise<void> {
    const isStable = await this.chaosDetector.testStability();

    if (isStable) {
      this.circuitBreaker.close();
    } else {
      this.circuitBreaker.open();
    }
  }

  async canProceedWithOptimization(): Promise<boolean> {
    const canProceed = this.circuitBreaker.canProceed();

    if (canProceed && this.circuitBreaker.getState() === CircuitState.HALF_OPEN) {
      this.circuitBreaker.recordAttempt();
    }

    return canProceed;
  }

  registerAgents(agents: Agent[]): void {
    this.registeredAgents = agents;
  }

  private async haltAllAgents(): Promise<void> {
    await Promise.all(
      this.registeredAgents.map(agent => agent.halt())
    );
  }

  getThresholds() {
    return { ...this.thresholds };
  }

  getStatus(): SentinelStatus {
    return {
      circuitState: this.circuitBreaker.getState(),
      chaosLevel: this.chaosDetector.getCurrentLevel(),
      interventionCount: this.interventionCount,
      lastIntervention: this.lastInterventionTime
    };
  }
}
