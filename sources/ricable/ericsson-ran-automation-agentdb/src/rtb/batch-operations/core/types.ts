/**
 * ENM CLI Batch Operations Framework - Core Type Definitions
 *
 * Comprehensive type system for batch operations with cognitive optimization,
 * error handling, and monitoring capabilities for Ericsson RAN configurations.
 */

import { RTBTemplate } from '../../types/rtb-types';

/**
 * Batch operation execution context
 */
export interface BatchExecutionContext {
  /** Unique batch operation identifier */
  batchId: string;
  /** Execution timestamp */
  timestamp: Date;
  /** User requesting the operation */
  userId: string;
  /** Session identifier for tracking */
  sessionId?: string;
  /** Cognitive consciousness level */
  consciousnessLevel: 'basic' | 'enhanced' | 'maximum';
  /** Execution environment */
  environment: 'development' | 'staging' | 'production';
  /** Regional context for optimization */
  region?: string;
  /** Network slice context (if applicable) */
  networkSlice?: string;
}

/**
 * Batch operation configuration
 */
export interface BatchOperationConfig {
  /** Operation name and description */
  name: string;
  description: string;
  /** Target node collection */
  collection: NodeCollection;
  /** Scope filters for node selection */
  scopeFilters: ScopeFilter[];
  /** Template to apply */
  template: RTBTemplate;
  /** Execution options */
  options: BatchExecutionOptions;
  /** Cognitive optimization settings */
  cognitiveSettings: CognitiveOptimizationSettings;
  /** Error handling strategy */
  errorHandling: ErrorHandlingStrategy;
  /** Monitoring configuration */
  monitoring: MonitoringConfig;
}

/**
 * Node collection definition
 */
export interface NodeCollection {
  /** Collection identifier */
  id: string;
  /** Collection name */
  name: string;
  /** Node selection patterns */
  nodePatterns: NodePattern[];
  /** Collection metadata */
  metadata: Record<string, any>;
  /** Total node count (resolved) */
  nodeCount?: number;
  /** Collection type */
  type: 'static' | 'dynamic' | 'computed';
}

/**
 * Node selection pattern
 */
export interface NodePattern {
  /** Pattern identifier */
  id: string;
  /** Pattern type */
  type: 'wildcard' | 'regex' | 'list' | 'query' | 'cognitive';
  /** Pattern value */
  pattern: string;
  /** Pattern priority */
  priority: number;
  /** Exclusion patterns */
  exclusions?: string[];
  /** Inclusion patterns (additional filters) */
  inclusions?: string[];
}

/**
 * Scope filter for intelligent node filtering
 */
export interface ScopeFilter {
  /** Filter identifier */
  id: string;
  /** Filter type */
  type: 'sync_status' | 'ne_type' | 'vendor' | 'version' | 'location' | 'performance' | 'custom';
  /** Filter condition */
  condition: FilterCondition;
  /** Filter action */
  action: 'include' | 'exclude' | 'prioritize';
  /** Filter priority */
  priority: number;
}

/**
 * Filter condition definition
 */
export interface FilterCondition {
  /** Attribute to filter on */
  attribute: string;
  /** Comparison operator */
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'not_in' | 'contains' | 'regex';
  /** Expected value(s) */
  value: any;
  /** Logical operators with other conditions */
  logicalOperator?: 'and' | 'or' | 'not';
  /** Nested conditions */
  conditions?: FilterCondition[];
}

/**
 * Batch execution options
 */
export interface BatchExecutionOptions {
  /** Execution mode */
  mode: 'sequential' | 'parallel' | 'adaptive' | 'cognitive';
  /** Maximum parallel executions */
  maxConcurrency: number;
  /** Command timeout in seconds */
  commandTimeout: number;
  /** Batch timeout in seconds */
  batchTimeout: number;
  /** Enable preview mode */
  preview: boolean;
  /** Enable force execution */
  force: boolean;
  /** Enable dry run mode */
  dryRun: boolean;
  /** Continue on error */
  continueOnError: boolean;
  /** Enable verbose output */
  verbose: boolean;
  /** Save rollback configuration */
  saveRollback: boolean;
}

/**
 * Cognitive optimization settings
 */
export interface CognitiveOptimizationSettings {
  /** Enable cognitive optimization */
  enabled: boolean;
  /** Temporal reasoning depth (1-1000x) */
  temporalDepth: number;
  /** Strange-loop optimization level */
  strangeLoopLevel: number;
  /** Learning from previous executions */
  enableLearning: boolean;
  /** Pattern recognition level */
  patternRecognitionLevel: number;
  /** Autonomous optimization threshold */
  autonomousThreshold: number;
  /** Cognitive session persistence */
  persistCognitiveState: boolean;
}

/**
 * Error handling strategy
 */
export interface ErrorHandlingStrategy {
  /** Retry configuration */
  retry: RetryConfiguration;
  /** Fallback strategies */
  fallback: FallbackStrategy[];
  /** Error classification */
  classification: ErrorClassification;
  /** Recovery actions */
  recovery: RecoveryAction[];
  /** Notification settings */
  notifications: NotificationSettings;
}

/**
 * Retry configuration
 */
export interface RetryConfiguration {
  /** Maximum retry attempts */
  maxAttempts: number;
  /** Base delay in milliseconds */
  baseDelay: number;
  /** Backoff multiplier */
  backoffMultiplier: number;
  /** Maximum delay in milliseconds */
  maxDelay: number;
  /** Jitter for randomized delays */
  jitter: boolean;
  /** Retryable error patterns */
  retryablePatterns: string[];
  /** Non-retryable error patterns */
  nonRetryablePatterns: string[];
}

/**
 * Fallback strategy
 */
export interface FallbackStrategy {
  /** Strategy identifier */
  id: string;
  /** Strategy type */
  type: 'alternative_command' | 'different_template' | 'manual_intervention' | 'skip' | 'rollback';
  /** Trigger conditions */
  triggerConditions: string[];
  /** Strategy configuration */
  config: Record<string, any>;
  /** Strategy priority */
  priority: number;
}

/**
 * Error classification
 */
export interface ErrorClassification {
  /** Critical errors (stop execution) */
  critical: string[];
  /** Warning errors (continue with notification) */
  warning: string[];
  /** Informational errors (log only) */
  informational: string[];
  /** Temporary errors (retry) */
  temporary: string[];
  /** Permanent errors (no retry) */
  permanent: string[];
}

/**
 * Recovery action
 */
export interface RecoveryAction {
  /** Action identifier */
  id: string;
  /** Action type */
  type: 'restart' | 'rollback' | 'alternative' | 'compensate' | 'escalate';
  /** Trigger conditions */
  triggerConditions: string[];
  /** Action configuration */
  config: Record<string, any>;
  /** Estimated recovery time */
  estimatedTime: number;
}

/**
 * Notification settings
 */
export interface NotificationSettings {
  /** Enable notifications */
  enabled: boolean;
  /** Notification channels */
  channels: NotificationChannel[];
  /** Notification thresholds */
  thresholds: NotificationThreshold[];
}

/**
 * Notification channel
 */
export interface NotificationChannel {
  /** Channel type */
  type: 'email' | 'slack' | 'webhook' | 'sms' | 'push';
  /** Channel configuration */
  config: Record<string, any>;
  /** Notification levels */
  levels: ('critical' | 'warning' | 'info')[];
}

/**
 * Notification threshold
 */
export interface NotificationThreshold {
  /** Metric to monitor */
  metric: string;
  /** Threshold value */
  threshold: number;
  /** Comparison operator */
  operator: 'gt' | 'gte' | 'lt' | 'lte' | 'eq';
  /** Notification level */
  level: 'critical' | 'warning' | 'info';
}

/**
 * Monitoring configuration
 */
export interface MonitoringConfig {
  /** Enable monitoring */
  enabled: boolean;
  /** Metrics to collect */
  metrics: MonitoringMetric[];
  /** Real-time alerts */
  alerts: MonitoringAlert[];
  /** Performance thresholds */
  thresholds: PerformanceThreshold[];
  /** Data retention settings */
  retention: RetentionSettings;
}

/**
 * Monitoring metric
 */
export interface MonitoringMetric {
  /** Metric identifier */
  id: string;
  /** Metric name */
  name: string;
  /** Metric type */
  type: 'counter' | 'gauge' | 'histogram' | 'timer';
  /** Metric unit */
  unit: string;
  /** Collection interval */
  interval: number;
  /** Metric tags */
  tags: Record<string, string>;
}

/**
 * Monitoring alert
 */
export interface MonitoringAlert {
  /** Alert identifier */
  id: string;
  /** Alert name */
  name: string;
  /** Alert condition */
  condition: AlertCondition;
  /** Alert actions */
  actions: AlertAction[];
  /** Alert severity */
  severity: 'critical' | 'high' | 'medium' | 'low';
}

/**
 * Alert condition
 */
export interface AlertCondition {
  /** Metric to monitor */
  metric: string;
  /** Comparison operator */
  operator: 'gt' | 'gte' | 'lt' | 'lte' | 'eq';
  /** Threshold value */
  threshold: number;
  /** Evaluation window */
  window: number;
  /** Minimum number of data points */
  minDataPoints: number;
}

/**
 * Alert action
 */
export interface AlertAction {
  /** Action type */
  type: 'notification' | 'webhook' | 'script' | 'escalation';
  /** Action configuration */
  config: Record<string, any>;
  /** Action delay */
  delay: number;
}

/**
 * Performance threshold
 */
export interface PerformanceThreshold {
  /** Metric name */
  metric: string;
  /** Warning threshold */
  warning: number;
  /** Critical threshold */
  critical: number;
  /** Threshold type */
  type: 'duration' | 'rate' | 'count' | 'percentage';
}

/**
 * Data retention settings
 */
export interface RetentionSettings {
  /** Raw data retention in days */
  rawData: number;
  /** Aggregated data retention in days */
  aggregatedData: number;
  /** Alert data retention in days */
  alertData: number;
  /** Audit log retention in days */
  auditLog: number;
}

/**
 * Batch operation execution result
 */
export interface BatchExecutionResult {
  /** Batch operation identifier */
  batchId: string;
  /** Execution status */
  status: ExecutionStatus;
  /** Node execution results */
  nodeResults: NodeExecutionResult[];
  /** Execution statistics */
  statistics: ExecutionStatistics;
  /** Performance metrics */
  metrics: PerformanceMetrics;
  /** Error summary */
  errorSummary: ErrorSummary;
  /** Audit information */
  audit: AuditInformation;
  /** Rollback information */
  rollback?: RollbackInformation;
}

/**
 * Execution status
 */
export type ExecutionStatus =
  | 'pending'
  | 'validating'
  | 'preparing'
  | 'executing'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'rolling_back'
  | 'rolled_back';

/**
 * Node execution result
 */
export interface NodeExecutionResult {
  /** Node identifier */
  nodeId: string;
  /** Execution status */
  status: NodeExecutionStatus;
  /** Command execution results */
  commandResults: CommandExecutionResult[];
  /** Node-specific metrics */
  metrics: NodeMetrics;
  /** Errors encountered */
  errors: NodeError[];
  /** Execution duration */
  duration: number;
  /** Execution timestamp */
  timestamp: Date;
}

/**
 * Node execution status
 */
export type NodeExecutionStatus =
  | 'pending'
  | 'skipped'
  | 'executing'
  | 'completed'
  | 'failed'
  | 'retrying'
  | 'rolled_back';

/**
 * Command execution result
 */
export interface CommandExecutionResult {
  /** Command identifier */
  commandId: string;
  /** Command type */
  type: 'CREATE' | 'SET' | 'DELETE' | 'GET' | 'MONITOR';
  /** Command string */
  command: string;
  /** Execution status */
  status: CommandExecutionStatus;
  /** Command output */
  output?: string;
  /** Error message */
  error?: string;
  /** Execution duration */
  duration: number;
  /** Retry attempts */
  retryAttempts: number;
  /** Timestamp */
  timestamp: Date;
  /** Cognitive optimizations applied */
  cognitiveOptimizations?: string[];
}

/**
 * Command execution status
 */
export type CommandExecutionStatus =
  | 'pending'
  | 'executing'
  | 'success'
  | 'failed'
  | 'timeout'
  | 'skipped'
  | 'retrying';

/**
 * Execution statistics
 */
export interface ExecutionStatistics {
  /** Total nodes targeted */
  totalNodes: number;
  /** Successful nodes */
  successfulNodes: number;
  /** Failed nodes */
  failedNodes: number;
  /** Skipped nodes */
  skippedNodes: number;
  /** Total commands executed */
  totalCommands: number;
  /** Successful commands */
  successfulCommands: number;
  /** Failed commands */
  failedCommands: number;
  /** Total execution duration */
  totalDuration: number;
  /** Average node duration */
  averageNodeDuration: number;
  /** Success rate */
  successRate: number;
  /** Parallelism efficiency */
  parallelismEfficiency: number;
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  /** Throughput metrics */
  throughput: ThroughputMetrics;
  /** Latency metrics */
  latency: LatencyMetrics;
  /** Resource utilization */
  resources: ResourceMetrics;
  /** Cognitive performance */
  cognitive: CognitiveMetrics;
}

/**
 * Throughput metrics
 */
export interface ThroughputMetrics {
  /** Commands per second */
  commandsPerSecond: number;
  /** Nodes per second */
  nodesPerSecond: number;
  /** Average throughput */
  averageThroughput: number;
  /** Peak throughput */
  peakThroughput: number;
}

/**
 * Latency metrics
 */
export interface LatencyMetrics {
  /** Average command latency */
  averageCommandLatency: number;
  /** P95 command latency */
  p95CommandLatency: number;
  /** P99 command latency */
  p99CommandLatency: number;
  /** Average node latency */
  averageNodeLatency: number;
  /** Maximum node latency */
  maxNodeLatency: number;
}

/**
 * Resource metrics
 */
export interface ResourceMetrics {
  /** CPU usage percentage */
  cpuUsage: number;
  /** Memory usage in MB */
  memoryUsage: number;
  /** Network I/O in bytes */
  networkIO: number;
  /** Disk I/O in bytes */
  diskIO: number;
  /** Concurrent executions */
  concurrentExecutions: number;
}

/**
 * Cognitive metrics
 */
export interface CognitiveMetrics {
  /** Temporal reasoning depth used */
  temporalDepthUsed: number;
  /** Strange-loop optimizations applied */
  strangeLoopOptimizations: number;
  /** Pattern recognition accuracy */
  patternRecognitionAccuracy: number;
  /** Autonomous decisions made */
  autonomousDecisions: number;
  /** Learning improvements gained */
  learningImprovements: number;
}

/**
 * Error summary
 */
export interface ErrorSummary {
  /** Total errors encountered */
  totalErrors: number;
  /** Errors by type */
  errorsByType: Record<string, number>;
  /** Errors by severity */
  errorsBySeverity: Record<string, number>;
  /** Most frequent errors */
  mostFrequentErrors: ErrorFrequency[];
  /** Error trends */
  errorTrends: ErrorTrend[];
}

/**
 * Error frequency
 */
export interface ErrorFrequency {
  /** Error message */
  message: string;
  /** Error type */
  type: string;
  /** Occurrence count */
  count: number;
  /** First occurrence */
  firstOccurrence: Date;
  /** Last occurrence */
  lastOccurrence: Date;
}

/**
 * Error trend
 */
export interface ErrorTrend {
  /** Time window */
  timeWindow: string;
  /** Error count */
  errorCount: number;
  /** Success rate */
  successRate: number;
  /** Trend direction */
  trend: 'improving' | 'degrading' | 'stable';
}

/**
 * Audit information
 */
export interface AuditInformation {
  /** Audit trail entries */
  entries: AuditEntry[];
  /** Compliance status */
  compliance: ComplianceStatus;
  /** Security events */
  securityEvents: SecurityEvent[];
}

/**
 * Audit entry
 */
export interface AuditEntry {
  /** Entry identifier */
  id: string;
  /** Timestamp */
  timestamp: Date;
  /** Event type */
  eventType: string;
  /** Event description */
  description: string;
  /** User who performed the action */
  userId: string;
  /** Node affected */
  nodeId?: string;
  /** Command executed */
  command?: string;
  /** Previous state */
  previousState?: string;
  /** New state */
  newState?: string;
  /** Additional metadata */
  metadata: Record<string, any>;
}

/**
 * Compliance status
 */
export interface ComplianceStatus {
  /** Overall compliance status */
  status: 'compliant' | 'non_compliant' | 'pending_review';
  /** Compliance rules evaluated */
  rulesEvaluated: number;
  /** Rules passed */
  rulesPassed: number;
  /** Rules failed */
  rulesFailed: number;
  /** Violation details */
  violations: ComplianceViolation[];
}

/**
 * Compliance violation
 */
export interface ComplianceViolation {
  /** Rule identifier */
  ruleId: string;
  /** Rule description */
  description: string;
  /** Severity level */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Violation details */
  details: string;
  /** Recommended action */
  recommendedAction: string;
}

/**
 * Security event
 */
export interface SecurityEvent {
  /** Event identifier */
  id: string;
  /** Event type */
  type: string;
  /** Severity level */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Description */
  description: string;
  /** Timestamp */
  timestamp: number;
  /** Source IP address */
  sourceIp?: string;
  /** User involved */
  userId?: string;
  /** Action taken */
  actionTaken: string;
}

/**
 * Rollback information
 */
export interface RollbackInformation {
  /** Rollback identifier */
  rollbackId: string;
  /** Rollback status */
  status: 'available' | 'initiated' | 'in_progress' | 'completed' | 'failed';
  /** Rollback commands */
  rollbackCommands: RollbackCommand[];
  /** Nodes that can be rolled back */
  rollbackNodes: string[];
  /** Rollback deadline */
  rollbackDeadline: Date;
  /** Rollback reason */
  reason?: string;
}

/**
 * Rollback command
 */
export interface RollbackCommand {
  /** Command identifier */
  commandId: string;
  /** Node identifier */
  nodeId: string;
  /** Rollback command */
  command: string;
  /** Original command */
  originalCommand: string;
  /** Rollback type */
  type: 'reverse' | 'reset' | 'delete' | 'restore';
  /** Execution status */
  status: 'pending' | 'executed' | 'failed' | 'skipped';
}

/**
 * Node metrics
 */
export interface NodeMetrics {
  /** Node-specific execution time */
  executionTime: number;
  /** Commands executed */
  commandsExecuted: number;
  /** Commands successful */
  commandsSuccessful: number;
  /** Commands failed */
  commandsFailed: number;
  /** Retry attempts */
  retryAttempts: number;
  /** Memory usage */
  memoryUsage: number;
  /** Network latency */
  networkLatency: number;
}

/**
 * Node error
 */
export interface NodeError {
  /** Error identifier */
  id: string;
  /** Error type */
  type: string;
  /** Error message */
  message: string;
  /** Error severity */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Command that caused the error */
  commandId?: string;
  /** Timestamp */
  timestamp: Date;
  /** Retry attempts */
  retryAttempts: number;
  /** Recovery actions taken */
  recoveryActions?: string[];
}

/**
 * Batch operation progress
 */
export interface BatchOperationProgress {
  /** Batch identifier */
  batchId: string;
  /** Overall progress percentage */
  overallProgress: number;
  /** Current phase */
  currentPhase: ExecutionStatus;
  /** Nodes completed */
  nodesCompleted: number;
  /** Total nodes */
  totalNodes: number;
  /** Estimated time remaining */
  estimatedTimeRemaining: number;
  /** Current activity */
  currentActivity: string;
  /** Recent errors */
  recentErrors: string[];
}