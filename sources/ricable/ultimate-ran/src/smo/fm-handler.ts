/**
 * FM Alarm Handler for Ericsson SMO Integration
 * 3GPP TS 28.532 - Fault Management (FM)
 *
 * Real-time alarm collection and correlation from Ericsson ENM/OSS
 * Server-Sent Events (SSE) for live alarm streaming
 * Root cause analysis via alarm correlation
 * Integration with self-healing agents
 *
 * @module smo/fm-handler
 * @track FM Data Pipeline
 * @agent agent-03
 */

import { EventEmitter } from 'events';
import { MidstreamProcessor, RANDataPoint } from '../learning/self-learner';

// ============================================================
// 3GPP TS 28.532 FM Alarm Interfaces
// ============================================================

/**
 * Alarm severity levels (3GPP TS 28.532)
 */
export type AlarmSeverity = 'critical' | 'major' | 'minor' | 'warning' | 'cleared' | 'indeterminate';

/**
 * Alarm perceived severity (ITU-T X.733)
 */
export type PerceivedSeverity = 'CRITICAL' | 'MAJOR' | 'MINOR' | 'WARNING' | 'CLEARED' | 'INDETERMINATE';

/**
 * FM Alarm structure (3GPP TS 28.532)
 */
export interface FMAlarm {
  alarmId: string;                 // Unique alarm identifier
  alarmType: string;               // Alarm type (e.g., "communicationsAlarm", "equipmentAlarm")
  probableCause: string;           // Probable cause (e.g., "thresholdCrossed", "powerProblem")
  specificProblem: string;         // Specific problem description
  perceivedSeverity: PerceivedSeverity;  // ITU-T X.733 severity
  severity: AlarmSeverity;         // Simplified severity
  managedObject: string;           // DN of affected object (cell, node, etc.)
  managedObjectInstance: string;   // Full DN path
  eventTime: Date;                 // When alarm was raised
  alarmRaisedTime?: Date;          // Original raise time
  alarmChangedTime?: Date;         // Last change time
  alarmClearedTime?: Date;         // When alarm was cleared
  ackTime?: Date;                  // Acknowledgment time
  ackUserId?: string;              // User who acknowledged
  ackState: 'ACKNOWLEDGED' | 'UNACKNOWLEDGED';
  additionalText?: string;         // Additional information
  additionalInformation?: Record<string, any>;  // Extra attributes
  proposedRepairActions?: string;  // Suggested repair actions
  correlatedAlarms?: string[];     // Related alarm IDs
  rootCauseIndicator?: boolean;    // Is this a root cause?
  trendIndication?: 'MORE_SEVERE' | 'NO_CHANGE' | 'LESS_SEVERE';
}

/**
 * Alarm correlation result
 */
export interface AlarmCorrelation {
  correlationId: string;
  rootCause: FMAlarm;              // Root cause alarm
  symptoms: FMAlarm[];             // Symptom alarms
  affectedCells: string[];         // Affected cell list
  correlationScore: number;        // 0-1 confidence score
  correlationType: 'CASCADE' | 'COMMON_CAUSE' | 'DUPLICATE' | 'TEMPORAL';
  timeWindow: number;              // Correlation time window in ms
  timestamp: Date;
}

/**
 * Alarm event for SSE streaming
 */
export interface AlarmEvent {
  eventType: 'NEW_ALARM' | 'ALARM_CHANGED' | 'ALARM_CLEARED' | 'ALARM_ACK';
  alarm: FMAlarm;
  timestamp: number;
  correlation?: AlarmCorrelation;
}

/**
 * Self-healing action triggered by alarms
 */
export interface SelfHealingAction {
  actionId: string;
  alarmId: string;
  actionType: 'AUTO_RECOVERY' | 'PARAMETER_TUNE' | 'CELL_RESTART' | 'ESCALATE';
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED';
  triggeredAt: Date;
  completedAt?: Date;
  result?: {
    success: boolean;
    details: string;
    pmDelta?: Record<string, number>;
  };
}

/**
 * FM Handler Configuration
 */
export interface FMHandlerConfig {
  enmEndpoint?: string;            // Ericsson ENM endpoint
  pollingInterval: number;         // Alarm polling interval in ms
  enableSSE: boolean;              // Enable Server-Sent Events
  ssePort: number;                 // SSE server port
  correlationWindow: number;       // Time window for correlation in ms
  enableAutoHealing: boolean;      // Enable self-healing actions
  severityFilter?: AlarmSeverity[]; // Filter alarms by severity
  managedObjects?: string[];       // Filter by managed objects
  storageEnabled: boolean;         // Store to agentdb
}

// ============================================================
// FM Handler Implementation
// ============================================================

/**
 * FMHandler - Real-time fault management and alarm correlation
 *
 * Features:
 * - Real-time alarm collection from Ericsson ENM
 * - Server-Sent Events (SSE) for live streaming
 * - Alarm correlation for root cause analysis
 * - Self-healing agent integration
 * - agentdb storage for historical analysis
 */
export class FMHandler extends EventEmitter {
  private config: FMHandlerConfig;
  private midstream: MidstreamProcessor;
  private activeAlarms: Map<string, FMAlarm>;
  private alarmHistory: FMAlarm[];
  private correlations: Map<string, AlarmCorrelation>;
  private sseClients: Set<any>;
  private pollingInterval?: NodeJS.Timeout;
  private isRunning: boolean;
  private totalAlarms: number;
  private healingActions: Map<string, SelfHealingAction>;

  constructor(config: Partial<FMHandlerConfig> = {}) {
    super();

    this.config = {
      enmEndpoint: config.enmEndpoint || 'http://enm.ericsson.local/fm/v1',
      pollingInterval: config.pollingInterval || 30000,  // 30 seconds
      enableSSE: config.enableSSE !== false,
      ssePort: config.ssePort || 3001,
      correlationWindow: config.correlationWindow || 300000,  // 5 minutes
      enableAutoHealing: config.enableAutoHealing !== false,
      severityFilter: config.severityFilter || ['critical', 'major'],
      managedObjects: config.managedObjects || [],
      storageEnabled: config.storageEnabled !== false
    };

    this.midstream = new MidstreamProcessor({
      bufferSize: 500,
      flushInterval: 60000  // 1 minute flush for alarms
    });

    this.activeAlarms = new Map();
    this.alarmHistory = [];
    this.correlations = new Map();
    this.sseClients = new Set();
    this.isRunning = false;
    this.totalAlarms = 0;
    this.healingActions = new Map();

    this.setupMidstreamHandlers();
  }

  /**
   * Setup midstream event handlers
   */
  private setupMidstreamHandlers(): void {
    this.midstream.on('alarm', (dataPoint: RANDataPoint) => {
      console.log(`[FMHandler] Alarm received: ${dataPoint.cellId}`);
    });

    this.midstream.on('flush', (batch: RANDataPoint[]) => {
      const alarmBatch = batch.filter(dp => dp.dataType === 'FM');
      console.log(`[FMHandler] Flushed ${alarmBatch.length} alarms`);
    });
  }

  /**
   * Start FM handler
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      console.warn('[FMHandler] Already running');
      return;
    }

    console.log('[FMHandler] Starting FM alarm handler...');
    console.log(`[FMHandler] Polling interval: ${this.config.pollingInterval / 1000}s`);
    console.log(`[FMHandler] Correlation window: ${this.config.correlationWindow / 1000}s`);
    console.log(`[FMHandler] SSE enabled: ${this.config.enableSSE}`);
    console.log(`[FMHandler] Auto-healing: ${this.config.enableAutoHealing}`);

    this.isRunning = true;

    // Start midstream
    this.midstream.start();

    // Start SSE server if enabled
    if (this.config.enableSSE) {
      await this.startSSEServer();
    }

    // Start alarm polling
    this.pollingInterval = setInterval(() => {
      this.pollAlarms();
    }, this.config.pollingInterval);

    // Immediate first poll
    await this.pollAlarms();

    this.emit('started');
  }

  /**
   * Stop FM handler
   */
  stop(): void {
    if (!this.isRunning) return;

    console.log('[FMHandler] Stopping FM alarm handler...');

    this.isRunning = false;

    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = undefined;
    }

    this.midstream.stop();

    // Close all SSE connections
    for (const client of this.sseClients) {
      try {
        client.end();
      } catch (error) {
        // Ignore errors during shutdown
      }
    }
    this.sseClients.clear();

    this.emit('stopped');
  }

  /**
   * Poll alarms from Ericsson ENM
   */
  private async pollAlarms(): Promise<void> {
    const startTime = Date.now();

    try {
      console.log(`[FMHandler] Polling alarms from ENM at ${new Date().toISOString()}`);

      // In production, this would call ENM REST API
      // GET /fm/v1/alarms?severity=CRITICAL,MAJOR
      const newAlarms = await this.fetchAlarmsFromENM();

      console.log(`[FMHandler] Fetched ${newAlarms.length} alarms from ENM`);

      // Process each alarm
      for (const alarm of newAlarms) {
        await this.processAlarm(alarm);
      }

      // Perform correlation analysis
      await this.performCorrelation();

      const duration = Date.now() - startTime;
      console.log(`[FMHandler] Alarm poll completed in ${duration}ms`);

      this.emit('poll_complete', {
        alarmCount: newAlarms.length,
        duration,
        timestamp: Date.now()
      });

    } catch (error) {
      console.error('[FMHandler] Alarm poll failed:', error);
      this.emit('poll_error', error);
    }
  }

  /**
   * Fetch alarms from Ericsson ENM
   * In production, uses ENM REST API
   */
  private async fetchAlarmsFromENM(): Promise<FMAlarm[]> {
    // Simulate ENM alarm feed
    const mockAlarms: FMAlarm[] = [];

    // Generate 0-3 random alarms per poll
    const alarmCount = Math.floor(Math.random() * 4);

    for (let i = 0; i < alarmCount; i++) {
      const alarm = this.generateMockAlarm();
      mockAlarms.push(alarm);
    }

    return mockAlarms;
  }

  /**
   * Generate mock alarm for testing
   */
  private generateMockAlarm(): FMAlarm {
    const alarmTypes = [
      { type: 'communicationsAlarm', cause: 'signalQualityEvaluationFailure', problem: 'High UL SINR Degradation' },
      { type: 'equipmentAlarm', cause: 'powerProblem', problem: 'RRU Power Supply Failure' },
      { type: 'processingErrorAlarm', cause: 'softwareError', problem: 'Cell Configuration Inconsistency' },
      { type: 'qualityOfServiceAlarm', cause: 'thresholdCrossed', problem: 'High Call Drop Rate' },
      { type: 'communicationsAlarm', cause: 'lossOfSignal', problem: 'Transport Link Down' }
    ];

    const severities: PerceivedSeverity[] = ['CRITICAL', 'MAJOR', 'MINOR', 'WARNING'];
    const alarmTemplate = alarmTypes[Math.floor(Math.random() * alarmTypes.length)];
    const severity = severities[Math.floor(Math.random() * severities.length)];

    const cellId = `CELL-${Math.floor(Math.random() * 1000)}`;
    const alarmId = `ALM-${Date.now()}-${Math.floor(Math.random() * 10000)}`;

    return {
      alarmId,
      alarmType: alarmTemplate.type,
      probableCause: alarmTemplate.cause,
      specificProblem: alarmTemplate.problem,
      perceivedSeverity: severity,
      severity: severity.toLowerCase() as AlarmSeverity,
      managedObject: cellId,
      managedObjectInstance: `SubNetwork=ONRM_ROOT_MO,SubNetwork=RAN,MeContext=${cellId}`,
      eventTime: new Date(),
      alarmRaisedTime: new Date(),
      ackState: 'UNACKNOWLEDGED',
      additionalText: `Alarm detected on ${cellId}`,
      additionalInformation: {
        threshold: Math.random() * 100,
        currentValue: Math.random() * 100
      }
    };
  }

  /**
   * Process incoming alarm
   */
  private async processAlarm(alarm: FMAlarm): Promise<void> {
    console.log(`[FMHandler] Processing alarm: ${alarm.alarmId} [${alarm.severity}] ${alarm.specificProblem}`);

    // Check if alarm already exists
    const existing = this.activeAlarms.get(alarm.alarmId);

    if (existing) {
      // Alarm changed
      alarm.alarmChangedTime = new Date();
      this.activeAlarms.set(alarm.alarmId, alarm);
      await this.streamAlarmEvent('ALARM_CHANGED', alarm);
    } else {
      // New alarm
      this.activeAlarms.set(alarm.alarmId, alarm);
      this.alarmHistory.push(alarm);
      this.totalAlarms++;
      await this.streamAlarmEvent('NEW_ALARM', alarm);
    }

    // Store to midstream
    const ranDataPoint: RANDataPoint = {
      timestamp: alarm.eventTime.getTime(),
      cellId: alarm.managedObject,
      dataType: 'FM',
      metrics: {
        severity: this.severityToNumber(alarm.severity),
        alarmCount: 1
      },
      context: alarm
    };

    this.midstream.ingest(ranDataPoint);

    // Store to agentdb
    if (this.config.storageEnabled) {
      await this.storeAlarm(alarm);
    }

    // Trigger self-healing if enabled
    if (this.config.enableAutoHealing && this.shouldTriggerHealing(alarm)) {
      await this.triggerSelfHealing(alarm);
    }

    this.emit('alarm_processed', alarm);
  }

  /**
   * Perform alarm correlation for root cause analysis
   */
  private async performCorrelation(): Promise<void> {
    if (this.activeAlarms.size < 2) return;

    const alarms = Array.from(this.activeAlarms.values());
    const now = Date.now();

    // Temporal correlation: find alarms raised within correlation window
    const recentAlarms = alarms.filter(alarm => {
      const alarmTime = alarm.eventTime.getTime();
      return (now - alarmTime) <= this.config.correlationWindow;
    });

    if (recentAlarms.length < 2) return;

    // Group alarms by managed object (cell/node)
    const alarmsByObject = new Map<string, FMAlarm[]>();

    for (const alarm of recentAlarms) {
      const objectKey = this.extractObjectKey(alarm.managedObjectInstance);
      if (!alarmsByObject.has(objectKey)) {
        alarmsByObject.set(objectKey, []);
      }
      alarmsByObject.get(objectKey)!.push(alarm);
    }

    // Find correlations
    for (const [objectKey, objectAlarms] of alarmsByObject) {
      if (objectAlarms.length < 2) continue;

      // Sort by severity (critical first)
      objectAlarms.sort((a, b) => {
        return this.severityToNumber(b.severity) - this.severityToNumber(a.severity);
      });

      // Root cause is typically the most severe or earliest alarm
      const rootCause = objectAlarms[0];
      const symptoms = objectAlarms.slice(1);

      // Calculate correlation score based on time proximity and severity
      const correlationScore = this.calculateCorrelationScore(rootCause, symptoms);

      if (correlationScore > 0.6) {  // Threshold for correlation
        const correlation: AlarmCorrelation = {
          correlationId: `CORR-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
          rootCause,
          symptoms,
          affectedCells: [rootCause.managedObject],
          correlationScore,
          correlationType: this.determineCorrelationType(rootCause, symptoms),
          timeWindow: this.config.correlationWindow,
          timestamp: new Date()
        };

        this.correlations.set(correlation.correlationId, correlation);

        console.log(`[FMHandler] Correlation detected: ${correlation.correlationId}`);
        console.log(`[FMHandler] Root cause: ${rootCause.specificProblem}`);
        console.log(`[FMHandler] Symptoms: ${symptoms.length} related alarms`);

        // Update root cause indicator
        rootCause.rootCauseIndicator = true;
        rootCause.correlatedAlarms = symptoms.map(s => s.alarmId);

        this.emit('correlation_detected', correlation);

        // Stream correlation via SSE
        await this.streamAlarmEvent('NEW_ALARM', rootCause, correlation);
      }
    }
  }

  /**
   * Calculate correlation score between root cause and symptoms
   */
  private calculateCorrelationScore(rootCause: FMAlarm, symptoms: FMAlarm[]): number {
    if (symptoms.length === 0) return 0;

    let score = 0;
    const rootTime = rootCause.eventTime.getTime();

    for (const symptom of symptoms) {
      const symptomTime = symptom.eventTime.getTime();
      const timeDiff = Math.abs(symptomTime - rootTime);

      // Time proximity score (closer in time = higher score)
      const timeScore = Math.max(0, 1 - (timeDiff / this.config.correlationWindow));

      // Severity correlation (similar severity or cascade)
      const severityScore = this.correlateSevertiy(rootCause.severity, symptom.severity);

      // Cause correlation (related probable causes)
      const causeScore = this.correlateCause(rootCause.probableCause, symptom.probableCause);

      // Weighted average
      score += (timeScore * 0.4 + severityScore * 0.3 + causeScore * 0.3);
    }

    return score / symptoms.length;
  }

  /**
   * Determine correlation type
   */
  private determineCorrelationType(
    rootCause: FMAlarm,
    symptoms: FMAlarm[]
  ): AlarmCorrelation['correlationType'] {
    // Check for duplicate alarms
    const duplicates = symptoms.filter(s => s.specificProblem === rootCause.specificProblem);
    if (duplicates.length === symptoms.length) {
      return 'DUPLICATE';
    }

    // Check for cascade (severity decreases)
    const severities = [rootCause, ...symptoms].map(a => this.severityToNumber(a.severity));
    const isDescending = severities.every((val, idx, arr) => idx === 0 || val <= arr[idx - 1]);
    if (isDescending) {
      return 'CASCADE';
    }

    // Check for common cause (same probable cause)
    const sameCause = symptoms.filter(s => s.probableCause === rootCause.probableCause);
    if (sameCause.length > symptoms.length / 2) {
      return 'COMMON_CAUSE';
    }

    return 'TEMPORAL';
  }

  /**
   * Correlate severity levels
   */
  private correlateSevertiy(sev1: AlarmSeverity, sev2: AlarmSeverity): number {
    const num1 = this.severityToNumber(sev1);
    const num2 = this.severityToNumber(sev2);

    // Same severity = high correlation
    if (num1 === num2) return 1.0;

    // Adjacent severity = medium correlation
    if (Math.abs(num1 - num2) === 1) return 0.6;

    // Distant severity = low correlation
    return 0.3;
  }

  /**
   * Correlate probable causes
   */
  private correlateCause(cause1: string, cause2: string): number {
    if (cause1 === cause2) return 1.0;

    // Group related causes
    const causeGroups = [
      ['powerProblem', 'equipmentFailure', 'hardwareFailure'],
      ['signalQualityEvaluationFailure', 'lossOfSignal', 'degradedSignal'],
      ['thresholdCrossed', 'performanceDegraded'],
      ['softwareError', 'configurationError']
    ];

    for (const group of causeGroups) {
      if (group.includes(cause1) && group.includes(cause2)) {
        return 0.7;
      }
    }

    return 0.2;
  }

  /**
   * Convert severity to numeric value for comparison
   */
  private severityToNumber(severity: AlarmSeverity): number {
    const map: Record<AlarmSeverity, number> = {
      'critical': 5,
      'major': 4,
      'minor': 3,
      'warning': 2,
      'cleared': 1,
      'indeterminate': 0
    };
    return map[severity] || 0;
  }

  /**
   * Extract object key for grouping (e.g., SubNetwork, MeContext)
   */
  private extractObjectKey(dn: string): string {
    // Extract MeContext from DN
    const match = dn.match(/MeContext=([^,]+)/);
    return match ? match[1] : dn;
  }

  /**
   * Determine if alarm should trigger self-healing
   */
  private shouldTriggerHealing(alarm: FMAlarm): boolean {
    // Only trigger for critical/major alarms
    if (!['critical', 'major'].includes(alarm.severity)) {
      return false;
    }

    // Only trigger for specific problem types
    const healableProblems = [
      'High Call Drop Rate',
      'High UL SINR Degradation',
      'Cell Configuration Inconsistency'
    ];

    return healableProblems.includes(alarm.specificProblem);
  }

  /**
   * Trigger self-healing action
   */
  private async triggerSelfHealing(alarm: FMAlarm): Promise<void> {
    const actionId = `HEAL-${Date.now()}-${Math.floor(Math.random() * 1000)}`;

    const action: SelfHealingAction = {
      actionId,
      alarmId: alarm.alarmId,
      actionType: this.determineHealingAction(alarm),
      status: 'PENDING',
      triggeredAt: new Date()
    };

    this.healingActions.set(actionId, action);

    console.log(`[FMHandler] Triggering self-healing: ${action.actionType} for ${alarm.specificProblem}`);

    this.emit('self_healing_triggered', action);

    // Execute healing action (async)
    this.executeSelfHealing(action);
  }

  /**
   * Determine appropriate healing action
   */
  private determineHealingAction(alarm: FMAlarm): SelfHealingAction['actionType'] {
    if (alarm.specificProblem.includes('Drop Rate')) {
      return 'PARAMETER_TUNE';
    } else if (alarm.specificProblem.includes('Configuration')) {
      return 'AUTO_RECOVERY';
    } else if (alarm.specificProblem.includes('SINR')) {
      return 'PARAMETER_TUNE';
    }
    return 'ESCALATE';
  }

  /**
   * Execute self-healing action
   */
  private async executeSelfHealing(action: SelfHealingAction): Promise<void> {
    action.status = 'IN_PROGRESS';

    console.log(`[FMHandler] Executing self-healing action: ${action.actionId}`);

    // Simulate healing action (in production, this would call actual healing logic)
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Simulate success/failure
    const success = Math.random() > 0.2;  // 80% success rate

    action.status = success ? 'COMPLETED' : 'FAILED';
    action.completedAt = new Date();
    action.result = {
      success,
      details: success ? 'Parameters optimized successfully' : 'Action failed, escalating',
      pmDelta: success ? {
        pmCallDropRate: -0.015,  // Improved by 1.5%
        pmCssr: 0.02             // Improved by 2%
      } : undefined
    };

    console.log(`[FMHandler] Self-healing ${action.status}: ${action.actionId}`);

    this.emit('self_healing_completed', action);
  }

  /**
   * Start SSE server for real-time alarm streaming
   */
  private async startSSEServer(): Promise<void> {
    console.log(`[FMHandler] SSE server would start on port ${this.config.ssePort}`);
    console.log(`[FMHandler] Connect to: http://localhost:${this.config.ssePort}/alarms/stream`);

    // In production, this would create an HTTP server with SSE endpoints
    // For now, we just emit events that can be consumed by other components

    this.emit('sse_server_started', { port: this.config.ssePort });
  }

  /**
   * Stream alarm event to SSE clients
   */
  private async streamAlarmEvent(
    eventType: AlarmEvent['eventType'],
    alarm: FMAlarm,
    correlation?: AlarmCorrelation
  ): Promise<void> {
    const event: AlarmEvent = {
      eventType,
      alarm,
      timestamp: Date.now(),
      correlation
    };

    // Emit as internal event
    this.emit('alarm_event', event);

    // In production, this would send SSE message to all connected clients:
    // for (const client of this.sseClients) {
    //   client.write(`data: ${JSON.stringify(event)}\n\n`);
    // }

    console.log(`[FMHandler] SSE Event: ${eventType} - ${alarm.alarmId}`);
  }

  /**
   * Store alarm to agentdb
   */
  private async storeAlarm(alarm: FMAlarm): Promise<void> {
    try {
      // In production, this would use agentdb API
      console.log(`[FMHandler] Storing alarm ${alarm.alarmId} to agentdb`);

      this.emit('alarm_stored', { alarmId: alarm.alarmId });

    } catch (error) {
      console.error('[FMHandler] Alarm storage failed:', error);
      this.emit('storage_error', error);
    }
  }

  /**
   * Clear an alarm
   */
  clearAlarm(alarmId: string): void {
    const alarm = this.activeAlarms.get(alarmId);

    if (!alarm) {
      console.warn(`[FMHandler] Alarm ${alarmId} not found`);
      return;
    }

    alarm.severity = 'cleared';
    alarm.perceivedSeverity = 'CLEARED';
    alarm.alarmClearedTime = new Date();

    this.activeAlarms.delete(alarmId);

    console.log(`[FMHandler] Alarm cleared: ${alarmId}`);

    this.streamAlarmEvent('ALARM_CLEARED', alarm);
    this.emit('alarm_cleared', alarm);
  }

  /**
   * Acknowledge an alarm
   */
  acknowledgeAlarm(alarmId: string, userId: string): void {
    const alarm = this.activeAlarms.get(alarmId);

    if (!alarm) {
      console.warn(`[FMHandler] Alarm ${alarmId} not found`);
      return;
    }

    alarm.ackState = 'ACKNOWLEDGED';
    alarm.ackTime = new Date();
    alarm.ackUserId = userId;

    console.log(`[FMHandler] Alarm acknowledged: ${alarmId} by ${userId}`);

    this.streamAlarmEvent('ALARM_ACK', alarm);
    this.emit('alarm_acknowledged', alarm);
  }

  /**
   * Get active alarms
   */
  getActiveAlarms(severityFilter?: AlarmSeverity[]): FMAlarm[] {
    const alarms = Array.from(this.activeAlarms.values());

    if (severityFilter && severityFilter.length > 0) {
      return alarms.filter(a => severityFilter.includes(a.severity));
    }

    return alarms;
  }

  /**
   * Get alarm history
   */
  getAlarmHistory(limit: number = 100): FMAlarm[] {
    return this.alarmHistory.slice(-limit);
  }

  /**
   * Get correlations
   */
  getCorrelations(): AlarmCorrelation[] {
    return Array.from(this.correlations.values());
  }

  /**
   * Get self-healing actions
   */
  getSelfHealingActions(): SelfHealingAction[] {
    return Array.from(this.healingActions.values());
  }

  /**
   * Get FM handler statistics
   */
  getStats() {
    const activeCount = this.activeAlarms.size;
    const criticalCount = this.getActiveAlarms(['critical']).length;
    const majorCount = this.getActiveAlarms(['major']).length;
    const minorCount = this.getActiveAlarms(['minor']).length;

    return {
      isRunning: this.isRunning,
      totalAlarms: this.totalAlarms,
      activeAlarms: activeCount,
      alarmsBySeverity: {
        critical: criticalCount,
        major: majorCount,
        minor: minorCount
      },
      correlations: this.correlations.size,
      healingActions: this.healingActions.size,
      sseClients: this.sseClients.size,
      config: {
        pollingInterval: this.config.pollingInterval,
        correlationWindow: this.config.correlationWindow,
        enableSSE: this.config.enableSSE,
        enableAutoHealing: this.config.enableAutoHealing
      }
    };
  }
}

// ============================================================
// Exports
// ============================================================

// export { FMHandler };
// export type {
//   FMHandlerConfig,
//   FMAlarm,
//   AlarmCorrelation,
//   AlarmEvent,
//   SelfHealingAction,
//   AlarmSeverity,
//   PerceivedSeverity
// };
