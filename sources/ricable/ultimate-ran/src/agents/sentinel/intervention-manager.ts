/**
 * Intervention Manager - Emergency Response
 *
 * Manages emergency interventions and coordinates system-wide responses
 * to detected chaos or instability.
 */

export enum InterventionType {
  SYSTEM_HALT = 'SYSTEM_HALT',
  LYAPUNOV_CRITICAL = 'LYAPUNOV_CRITICAL',
  STABILITY_LOW = 'STABILITY_LOW',
  INTERFERENCE_HIGH = 'INTERFERENCE_HIGH'
}

export interface InterventionResult {
  type: InterventionType;
  executed: boolean;
  actions: string[];
  timestamp: number;
}

export interface BroadcastMessage {
  type: string;
  priority?: string;
  reason?: string;
  chaosLevel?: any;
  timestamp: number;
}

export interface LogEntry {
  event: string;
  lyapunovExponent?: number;
  systemStability?: number;
  timestamp: number;
}

type BroadcastFunction = (message: BroadcastMessage) => Promise<void>;
type LogFunction = (entry: LogEntry) => Promise<void>;

export class InterventionManager {
  private broadcaster?: BroadcastFunction;
  private swarmBroadcaster?: BroadcastFunction;
  private logger?: LogFunction;
  private interventionHistory: InterventionResult[] = [];

  setBroadcaster(broadcaster: BroadcastFunction): void {
    this.broadcaster = broadcaster;
  }

  setSwarmBroadcaster(broadcaster: BroadcastFunction): void {
    this.swarmBroadcaster = broadcaster;
  }

  setLogger(logger: LogFunction): void {
    this.logger = logger;
  }

  async triggerIntervention(type: InterventionType): Promise<InterventionResult> {
    const actions = this.determineActions(type);

    const result: InterventionResult = {
      type,
      executed: true,
      actions,
      timestamp: Date.now()
    };

    this.interventionHistory.push(result);

    // Execute intervention actions
    await this.executeActions(type, actions);

    return result;
  }

  private determineActions(type: InterventionType): string[] {
    switch (type) {
      case InterventionType.SYSTEM_HALT:
        return [
          'FREEZE_ALL_OPTIMIZATIONS',
          'HALT_AGENT_OPERATIONS',
          'NOTIFY_OPERATORS',
          'ACTIVATE_SAFE_MODE'
        ];

      case InterventionType.LYAPUNOV_CRITICAL:
        return [
          'FREEZE_OPTIMIZATIONS',
          'ACTIVATE_DIGITAL_TWIN',
          'ROLLBACK_RECENT_CHANGES',
          'NOTIFY_GUARDIAN'
        ];

      case InterventionType.STABILITY_LOW:
        return [
          'ACTIVATE_GUARDIAN',
          'REDUCE_OPTIMIZATION_RATE',
          'INCREASE_MONITORING',
          'VALIDATE_PARAMETERS'
        ];

      case InterventionType.INTERFERENCE_HIGH:
        return [
          'REDUCE_POWER',
          'ADJUST_FREQUENCY_ALLOCATION',
          'ACTIVATE_INTERFERENCE_MITIGATION',
          'NOTIFY_CLUSTER_COORDINATOR'
        ];

      default:
        return ['LOG_EVENT'];
    }
  }

  private async executeActions(type: InterventionType, actions: string[]): Promise<void> {
    // Execute based on intervention type
    if (type === InterventionType.SYSTEM_HALT) {
      await this.broadcastEmergencyHalt();
    }

    // Log all interventions
    await this.logToAgentDB({
      event: type,
      timestamp: Date.now()
    });

    // Additional action execution logic would go here
    // For now, we're focusing on the coordination aspects
  }

  async broadcastEmergencyHalt(): Promise<void> {
    if (this.broadcaster) {
      await this.broadcaster({
        type: 'EMERGENCY_HALT',
        priority: 'CRITICAL',
        timestamp: Date.now()
      });
    }
  }

  async broadcastToSwarm(message: BroadcastMessage): Promise<void> {
    if (this.swarmBroadcaster) {
      await this.swarmBroadcaster(message);
    }
  }

  async logToAgentDB(entry: LogEntry): Promise<void> {
    if (this.logger) {
      await this.logger(entry);
    }
  }

  shouldIntervene(): boolean {
    // Default implementation - can be overridden with specific logic
    return false;
  }

  getInterventionHistory(): InterventionResult[] {
    return [...this.interventionHistory];
  }

  getInterventionCount(): number {
    return this.interventionHistory.length;
  }

  getLastIntervention(): InterventionResult | undefined {
    return this.interventionHistory[this.interventionHistory.length - 1];
  }
}
