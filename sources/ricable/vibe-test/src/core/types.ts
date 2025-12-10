/**
 * Neuro-Federated Swarm Intelligence for Ericsson RAN Optimization
 * Core Type Definitions
 */

// ============================================================================
// Agent Types
// ============================================================================

export type AgentRole =
  | 'queen'           // Coordinator agent at CU/MEC level
  | 'worker'          // Specialized worker at DU/RU level
  | 'optimizer'       // Performance optimization agent
  | 'healer'          // Fault detection and healing agent
  | 'configurator';   // Configuration management agent

export type AgentState =
  | 'idle'
  | 'active'
  | 'processing'
  | 'waiting'
  | 'terminated';

export interface AgentIdentity {
  id: string;
  role: AgentRole;
  nodeId: string;
  createdAt: number;
  version: string;
}

export interface Agent {
  identity: AgentIdentity;
  state: AgentState;
  capabilities: string[];
  memory: AgentMemory;
  execute(task: Task): Promise<TaskResult>;
  terminate(): Promise<void>;
}

export interface AgentMemory {
  shortTerm: Map<string, unknown>;
  experiences: Experience[];
  skills: SkillVector[];
}

export interface Experience {
  timestamp: number;
  action: string;
  context: Record<string, unknown>;
  outcome: 'success' | 'failure' | 'partial';
  kpiDelta: number;
}

export interface SkillVector {
  id: string;
  embedding: Float32Array;
  pattern: string;
  successRate: number;
}

// ============================================================================
// Task Types
// ============================================================================

export type TaskPriority = 'critical' | 'high' | 'medium' | 'low';
export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'rolled_back';

export interface Task {
  id: string;
  type: string;
  priority: TaskPriority;
  status: TaskStatus;
  payload: Record<string, unknown>;
  constraints: TaskConstraints;
  createdAt: number;
  deadline?: number;
}

export interface TaskConstraints {
  maxLatencyMs: number;
  requiredCapabilities: string[];
  safetyLevel: 'low' | 'medium' | 'high' | 'critical';
  rollbackEnabled: boolean;
}

export interface TaskResult {
  taskId: string;
  success: boolean;
  data?: Record<string, unknown>;
  error?: string;
  executionTimeMs: number;
  kpiImpact?: KPIImpact;
}

export interface KPIImpact {
  throughputDelta: number;
  latencyDelta: number;
  interferenceDelta: number;
  energyDelta: number;
}

// ============================================================================
// Network Topology Types
// ============================================================================

export type CellTechnology = '4G' | '5G' | '4G5G';

export interface CellIdentity {
  cellId: string;
  gNodeBId?: string;
  eNodeBId?: string;
  technology: CellTechnology;
  sectorId: number;
}

export interface CellState {
  identity: CellIdentity;
  location: GeoLocation;
  configuration: CellConfiguration;
  metrics: CellMetrics;
  neighbors: NeighborRelation[];
  timestamp: number;
}

export interface GeoLocation {
  latitude: number;
  longitude: number;
  altitude: number;
  azimuth: number;
}

export interface CellConfiguration {
  // 4G Parameters
  p0NominalPUSCH?: number;
  qRxLevMin?: number;
  a3Offset?: number;
  timeToTrigger?: number;

  // 5G Parameters
  ssbSubcarrierSpacing?: number;
  bwpId?: number;
  nCI?: number;

  // Common Parameters
  electricalTilt: number;
  mechanicalTilt: number;
  transmitPower: number;
  pci: number;
  bandwidth: number;
}

export interface CellMetrics {
  // Traffic Metrics
  prbUtilizationDl: number;
  prbUtilizationUl: number;
  activeUesDl: number;
  activeUesUl: number;
  throughputDl: number;
  throughputUl: number;

  // Quality Metrics
  rsrp: number;
  rsrq: number;
  sinr: number;
  bler: number;

  // Interference Metrics
  rssiUl: number;
  interferenceLevel: number;

  // Energy Metrics
  powerConsumption: number;
  sleepRatio: number;

  timestamp: number;
}

export interface NeighborRelation {
  targetCellId: string;
  noRemove: boolean;
  noHo: boolean;
  isAnr: boolean;
  handoverAttempts: number;
  handoverSuccesses: number;
  interferenceLevel: number;
}

// ============================================================================
// Performance Management Types
// ============================================================================

export interface TimeSeries {
  metricName: string;
  cellId: string;
  values: number[];
  timestamps: number[];
  resolution: number; // milliseconds between samples
}

export interface ChaosAnalysis {
  isChaoatic: boolean;
  lyapunovExponent: number;
  correlationDimension: number;
  entropy: number;
  predictability: number;
  recommendedStrategy: 'predictive' | 'damping' | 'hybrid';
}

export interface TrafficForecast {
  cellId: string;
  predictions: PredictionPoint[];
  confidence: number;
  model: 'lstm' | 'nbeats' | 'ensemble';
}

export interface PredictionPoint {
  timestamp: number;
  value: number;
  lowerBound: number;
  upperBound: number;
}

// ============================================================================
// Fault Management Types
// ============================================================================

export type AlarmSeverity = 'critical' | 'major' | 'minor' | 'warning' | 'cleared';

export interface Alarm {
  alarmId: string;
  alarmCode: string;
  severity: AlarmSeverity;
  cellId: string;
  description: string;
  raisedAt: number;
  clearedAt?: number;
  acknowledgedAt?: number;
  additionalInfo: Record<string, string>;
}

export interface Anomaly {
  id: string;
  type: AnomalyType;
  cellId: string;
  detectedAt: number;
  severity: number; // 0-1 scale
  metrics: AnomalyMetrics;
  resolved: boolean;
}

export type AnomalyType =
  | 'rssi_drop'
  | 'jamming'
  | 'sleeping_cell'
  | 'handover_storm'
  | 'traffic_spike'
  | 'interference_spike'
  | 'vswr_high'
  | 'coverage_hole';

export interface AnomalyMetrics {
  baseline: number;
  observed: number;
  deviation: number;
  trend: 'increasing' | 'decreasing' | 'stable';
}

export interface RootCauseAnalysis {
  anomalyId: string;
  probableCauses: ProbableCause[];
  recommendedActions: RecommendedAction[];
  reasoningChain: ReasoningStep[];
  confidence: number;
}

export interface ProbableCause {
  cause: string;
  probability: number;
  evidence: string[];
}

export interface RecommendedAction {
  action: string;
  priority: TaskPriority;
  risk: 'low' | 'medium' | 'high';
  expectedImpact: string;
}

export interface ReasoningStep {
  type: 'symbolic' | 'neural' | 'hybrid';
  premise: string;
  conclusion: string;
  confidence: number;
}

// ============================================================================
// Configuration Management Types
// ============================================================================

export interface ConfigurationChange {
  id: string;
  cellId: string;
  managedObject: ManagedObjectType;
  attribute: string;
  oldValue: unknown;
  newValue: unknown;
  reason: string;
  appliedAt?: number;
  rolledBackAt?: number;
  status: 'pending' | 'applied' | 'verified' | 'rolled_back' | 'failed';
}

export type ManagedObjectType =
  | 'EUtranCellFDD'
  | 'NRCellDU'
  | 'NRCellCU'
  | 'RetDevice'
  | 'ReportConfigEUtra'
  | 'Beamforming';

export interface GOAPGoal {
  id: string;
  name: string;
  targetState: Record<string, unknown>;
  priority: number;
  deadline?: number;
}

export interface GOAPAction {
  name: string;
  preconditions: Record<string, unknown>;
  effects: Record<string, unknown>;
  cost: number;
  risk: number;
}

export interface GOAPPlan {
  goalId: string;
  actions: GOAPAction[];
  totalCost: number;
  totalRisk: number;
  estimatedDuration: number;
}

// ============================================================================
// Graph Neural Network Types
// ============================================================================

export interface CellEmbedding {
  cellId: string;
  vector: Float32Array;
  staticFeatures: StaticFeatures;
  dynamicFeatures: DynamicFeatures;
  lastUpdated: number;
}

export interface StaticFeatures {
  azimuth: number;
  height: number;
  beamwidth: number;
  technology: number; // encoded
  sectorCount: number;
}

export interface DynamicFeatures {
  load: number;
  interferenceLevel: number;
  throughput: number;
  userCount: number;
}

export interface GraphEdge {
  sourceId: string;
  targetId: string;
  edgeType: 'neighbor' | 'interferer' | 'handover';
  weight: number;
  attributes: Record<string, number>;
}

export interface NetworkGraph {
  nodes: Map<string, CellEmbedding>;
  edges: GraphEdge[];
  adjacencyMatrix?: Float32Array;
}

// ============================================================================
// Safety Types
// ============================================================================

export interface LTLFormula {
  id: string;
  name: string;
  formula: string;
  description: string;
}

export interface SafetyVerification {
  actionId: string;
  formulasChecked: string[];
  passed: boolean;
  violations: SafetyViolation[];
  verificationTimeMs: number;
}

export interface SafetyViolation {
  formulaId: string;
  violation: string;
  severity: 'warning' | 'error' | 'critical';
}

// ============================================================================
// Swarm Types
// ============================================================================

export type SwarmTopology = 'mesh' | 'ring' | 'star' | 'hierarchical';

export interface SwarmConfiguration {
  topology: SwarmTopology;
  maxAgents: number;
  communicationProtocol: 'quic' | 'grpc' | 'websocket';
  federationEnabled: boolean;
  sandboxed: boolean;
}

export interface SwarmState {
  id: string;
  configuration: SwarmConfiguration;
  agents: Map<string, Agent>;
  queenId?: string;
  status: 'initializing' | 'running' | 'degraded' | 'stopped';
  metrics: SwarmMetrics;
}

export interface SwarmMetrics {
  activeAgents: number;
  tasksCompleted: number;
  tasksFailed: number;
  averageLatencyMs: number;
  successRate: number;
  uptime: number;
}

// ============================================================================
// Event Types
// ============================================================================

export type EventType =
  | 'alarm_raised'
  | 'alarm_cleared'
  | 'anomaly_detected'
  | 'config_changed'
  | 'agent_spawned'
  | 'agent_terminated'
  | 'task_completed'
  | 'rollback_triggered';

export interface SystemEvent {
  id: string;
  type: EventType;
  timestamp: number;
  source: string;
  payload: Record<string, unknown>;
}

// ============================================================================
// Federation Types
// ============================================================================

export interface FederatedModel {
  id: string;
  version: number;
  weights: Float32Array;
  aggregatedFrom: string[];
  timestamp: number;
}

export interface GradientUpdate {
  agentId: string;
  modelId: string;
  gradients: Float32Array;
  sampleCount: number;
  timestamp: number;
}
