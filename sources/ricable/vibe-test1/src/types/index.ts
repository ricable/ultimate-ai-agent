/**
 * Ericsson RAN Autonomous Optimization Platform - Type Definitions
 * Core types for the Federated Swarm Architecture
 */

// ============================================================================
// CELL & NETWORK IDENTITY TYPES
// ============================================================================

export interface CellGlobalIdentity {
  mcc: string;      // Mobile Country Code (3 digits)
  mnc: string;      // Mobile Network Code (2-3 digits)
  gnbId: number;    // gNodeB ID
  cellId: number;   // Cell ID (NR Cell Identity)
  sectorId: number; // Sector (0-2 typically)
}

export interface SliceIdentity {
  sst: number;      // Slice/Service Type (1=eMBB, 2=URLLC, 3=mMTC)
  sd?: string;      // Slice Differentiator (optional)
}

export type NanoAgentId = string; // Format: CGI or S-NSSAI based

// ============================================================================
// RAN PARAMETER TYPES
// ============================================================================

export interface UplinkPowerControlParams {
  p0NominalPusch: number;      // Target received power (dBm) -126 to -60
  alpha: number;                // Pathloss compensation factor 0.4-1.0
  p0NominalPucch: number;      // PUCCH target power
  deltaMcs: boolean;           // MCS-based power adjustment
  accumulationEnabled: boolean;
  msg3DeltaPreamble: number;
  deltaF_PUCCH_Format: {
    format1: number;
    format1a: number;
    format1b: number;
    format2: number;
    format2a: number;
    format2b: number;
    format3: number;
    format4: number;
  };
}

export interface MobilityParams {
  a3Offset: number;           // Event A3 offset (dB) -15 to 15
  hysteresis: number;         // Hysteresis (dB) 0-15
  timeToTrigger: number;      // Time to trigger (ms) 0-5120
  filterCoefficient: number;  // L3 filter coefficient
  qRxLevMin: number;          // Minimum RSRP for cell selection
  qQualMin: number;           // Minimum RSRQ for cell selection
  cellIndividualOffset: Map<string, number>; // CIO per neighbor
}

export interface AntennaParams {
  electricalTilt: number;     // RET degrees 0-15
  mechanicalTilt: number;     // Physical tilt
  azimuth: number;            // Direction 0-360
  beamwidth: number;          // Horizontal beamwidth
  height: number;             // Antenna height (meters)
  maxPower: number;           // Max TX power (dBm)
  referenceSignalPower: number; // RS power
}

export interface SchedulerParams {
  prbAllocationStrategy: 'roundRobin' | 'proportionalFair' | 'maxThroughput';
  sliceWeights: Map<string, number>;  // Per-slice scheduling weights
  qosFlowPriority: Map<number, number>; // 5QI to priority mapping
  maxRbPerUe: number;
  cqiOffset: number;
  targetBler: number;
}

// ============================================================================
// PERFORMANCE METRICS TYPES
// ============================================================================

export interface CellKPIs {
  timestamp: Date;
  cgi: CellGlobalIdentity;

  // Accessibility KPIs
  rrcSetupSuccessRate: number;
  erabSetupSuccessRate: number;
  ngSetupSuccessRate: number;

  // Retainability KPIs
  callDropRate: number;
  rrcAbnormalRelease: number;
  erabAbnormalRelease: number;

  // Mobility KPIs
  hoSuccessRate: number;
  hoAttempts: number;
  hoFailures: number;
  pingPongRate: number;

  // Integrity/Throughput KPIs
  dlThroughput: number;        // Mbps
  ulThroughput: number;        // Mbps
  dlUserThroughput5Pct: number; // 5th percentile (edge)
  ulUserThroughput5Pct: number;

  // Resource Utilization
  prbUtilizationDl: number;    // 0-100%
  prbUtilizationUl: number;
  activeUsers: number;
  rrcConnectedUsers: number;

  // RF Quality
  avgRsrp: number;             // dBm
  avgRsrq: number;             // dB
  avgSinrDl: number;           // dB
  avgSinrUl: number;
  iotUl: number;               // Interference over Thermal (dB)
  rtwp: number;                // Received Total Wideband Power

  // Error Rates
  blerDl: number;              // 0-1
  blerUl: number;
  cqiAverage: number;          // 0-15

  // Traffic Volume
  dataVolumeUl: number;        // GB
  dataVolumeDl: number;

  // Slice-specific (if applicable)
  sliceMetrics?: Map<string, SliceKPIs>;
}

export interface SliceKPIs {
  sliceId: SliceIdentity;
  throughput: number;
  latency: number;             // ms
  packetLossRate: number;
  userCount: number;
  slaCompliance: number;       // 0-100%
}

export interface TimeSeries {
  timestamps: Date[];
  values: number[];
  granularity: '15min' | '1hour' | '1day' | '1week';
  metric: string;
}

// ============================================================================
// GRAPH/SPATIAL TYPES
// ============================================================================

export interface RANGraphNode {
  id: string;                  // CGI string
  cgi: CellGlobalIdentity;

  // Static features
  staticFeatures: {
    azimuth: number;
    tilt: number;
    height: number;
    beamwidth: number;
    frequency: number;         // MHz
    bandwidth: number;         // MHz
    technology: 'LTE' | 'NR';
    latitude: number;
    longitude: number;
  };

  // Dynamic features (updated per ROP)
  dynamicFeatures: {
    txPower: number;
    load: number;
    rtwp: number;
    activeUsers: number;
    throughput: number;
  };

  // Embedding from GNN
  embedding?: number[];
}

export interface RANGraphEdge {
  sourceId: string;
  targetId: string;
  edgeType: 'geographic' | 'rf' | 'mobility' | 'interference';
  weight: number;
  features: {
    distance?: number;          // meters
    rsrp?: number;              // dBm (RF measurement)
    hoVolume?: number;          // Handover count
    interferenceLevel?: number; // dB
  };
}

export interface RANHyperedge {
  id: string;
  nodeIds: string[];           // Cells in interference cluster
  clusterType: 'interference' | 'coverage' | 'capacity';
  weight: number;
  features: {
    totalInterference: number;
    dominantInterferor?: string;
  };
}

export interface RANGraph {
  nodes: Map<string, RANGraphNode>;
  edges: RANGraphEdge[];
  hyperedges: RANHyperedge[];
  lastUpdated: Date;
}

// ============================================================================
// AGENT & REASONING TYPES
// ============================================================================

export interface ThoughtTrajectory {
  id: string;
  timestamp: Date;
  agentId: NanoAgentId;

  symptom: string;             // What triggered this reasoning
  context: {
    cellState: Partial<CellKPIs>;
    neighborStates: Map<string, Partial<CellKPIs>>;
    recentActions: ActionRecord[];
    environmentFactors: Record<string, any>;
  };

  actionSequence: Action[];
  outcome: {
    success: boolean;
    deltaKPIs: Record<string, number>;
    verdict: string;
  };

  reflexion?: string;          // Self-critique if failed
}

export interface CausalEdge {
  cause: string;
  effect: string;
  probability: number;         // P(effect | do(cause))
  confidence: number;          // Based on sample count
  observationCount: number;
}

export interface CausalGraph {
  nodes: Set<string>;          // Event types
  edges: CausalEdge[];
  lastUpdated: Date;
}

export interface ReasoningBankQuery {
  symptom: string;
  contextEmbedding: number[];
  maxResults: number;
  minSimilarity: number;
}

// ============================================================================
// ACTION & OPTIMIZATION TYPES
// ============================================================================

export type ActionType =
  | 'ADJUST_P0'
  | 'ADJUST_ALPHA'
  | 'ADJUST_TILT'
  | 'ADJUST_POWER'
  | 'ADJUST_HO_PARAMS'
  | 'ADJUST_CIO'
  | 'ADJUST_SCHEDULER'
  | 'ACTIVATE_CARRIER'
  | 'DEACTIVATE_CARRIER'
  | 'RESTART_CELL'
  | 'CHANGE_PCI'
  | 'UPDATE_ANR';

export interface Action {
  type: ActionType;
  targetCgi: string;
  parameters: Record<string, number | string | boolean>;
  timestamp: Date;
  source: 'rl' | 'rule' | 'federated' | 'manual';
  confidence: number;
}

export interface ActionRecord extends Action {
  id: string;
  executed: boolean;
  blocked: boolean;
  blockReason?: string;
  outcome?: {
    preState: Partial<CellKPIs>;
    postState: Partial<CellKPIs>;
    deltaMetrics: Record<string, number>;
  };
}

export interface OptimizationState {
  cellState: number[];         // Flattened KPI vector
  neighborStates: number[][];  // Per-neighbor state vectors
  graphEmbedding: number[];    // From GNN
  currentParams: number[];     // Current P0, Alpha, etc.
}

export interface OptimizationAction {
  p0Delta: number;             // -2, -1, 0, 1, 2 dB
  alphaDelta: number;          // -0.1, 0, 0.1
  tiltDelta?: number;          // -1, 0, 1 degree
  powerDelta?: number;         // -1, 0, 1 dB
}

export interface OptimizationReward {
  throughputAvg: number;
  throughputEdge: number;      // 5th percentile
  neighborInterference: number;
  blerPenalty: number;
  totalReward: number;
}

// ============================================================================
// ANOMALY & FAULT TYPES
// ============================================================================

export type AnomalyType =
  | 'THRESHOLD_BREACH'
  | 'PATTERN_DEVIATION'
  | 'ATTRACTOR_DEPARTURE'
  | 'CORRELATION_BREAK'
  | 'CHAOS_ONSET';

export type ProblemCategory =
  | 'UPLINK_INTERFERENCE'
  | 'PILOT_POLLUTION'
  | 'COVERAGE_HOLE'
  | 'OVERSHOOTING'
  | 'BACKHAUL_CONGESTION'
  | 'SLEEPING_CELL'
  | 'PCI_CONFLICT'
  | 'ANR_ISSUE'
  | 'CAPACITY_SATURATION';

export interface Anomaly {
  id: string;
  timestamp: Date;
  cgi: string;
  type: AnomalyType;
  severity: 'low' | 'medium' | 'high' | 'critical';

  anomalyVector: number[];     // Deviation per metric
  affectedMetrics: string[];
  confidence: number;

  context: {
    timeOfDay: number;
    dayOfWeek: number;
    isWeekend: boolean;
    weatherConditions?: string;
    nearbyEvents?: string[];
  };
}

export interface Problem {
  id: string;
  timestamp: Date;
  affectedCells: string[];
  category: ProblemCategory;
  rootCause?: string;
  causalChain?: CausalEdge[];
  confidence: number;

  symptoms: Anomaly[];

  suggestedActions: Action[];
  automatedRecovery: boolean;
  recoveryStatus?: 'pending' | 'executing' | 'completed' | 'failed';
}

// ============================================================================
// SWARM & FEDERATION TYPES
// ============================================================================

export type SwarmMessageType =
  | 'HEARTBEAT'
  | 'STATE_SYNC'
  | 'MODEL_UPDATE'
  | 'PATTERN_SHARE'
  | 'CONFLICT_RESOLUTION'
  | 'LEADER_ELECTION'
  | 'TASK_DELEGATION';

export interface SwarmMessage {
  id: string;
  type: SwarmMessageType;
  sourceAgent: NanoAgentId;
  targetAgent?: NanoAgentId;   // Broadcast if undefined
  timestamp: Date;
  ttl: number;                 // Hops remaining
  signature: string;           // Ed25519 signature
  payload: any;
}

export interface FederatedModelUpdate {
  modelId: string;
  modelType: 'uplinkOptimizer' | 'faultDetector' | 'trafficPredictor';
  sourceAgent: NanoAgentId;
  round: number;
  gradients?: number[];
  weights?: number[];
  sampleCount: number;
  metrics: {
    loss: number;
    accuracy?: number;
  };
}

export interface ClusterInfo {
  clusterId: string;
  members: NanoAgentId[];
  leader?: NanoAgentId;
  lastElection: Date;
  aggregatedModel?: FederatedModelUpdate;
}

export interface AgentState {
  id: NanoAgentId;
  cgi: CellGlobalIdentity;
  status: 'active' | 'learning' | 'optimizing' | 'recovering' | 'offline';

  cluster?: ClusterInfo;
  neighbors: NanoAgentId[];

  localModel: {
    version: number;
    lastTrained: Date;
    metrics: Record<string, number>;
  };

  reasoningStats: {
    trajectoriesStored: number;
    queriesServed: number;
    patternsLearned: number;
  };

  optimizationStats: {
    actionsExecuted: number;
    actionsBlocked: number;
    avgReward: number;
  };

  lastHeartbeat: Date;
}

// ============================================================================
// SECURITY TYPES
// ============================================================================

export interface AdversarialDetectionResult {
  isAdversarial: boolean;
  confidence: number;
  attackType?: 'poisoning' | 'evasion' | 'inference' | 'spoofing';
  anomalyScore: number;
  physicalConsistency: boolean;
  recommendation: 'accept' | 'reject' | 'quarantine';
}

export interface GuardrailViolation {
  actionId: string;
  violation: string;
  parameter: string;
  attemptedValue: number;
  allowedRange: [number, number];
  timestamp: Date;
}

export interface ModelProvenanceRecord {
  modelId: string;
  version: number;
  contributors: NanoAgentId[];
  trainingRound: number;
  hash: string;                // SHA-256 of weights
  signature: string;           // Aggregate signature
  timestamp: Date;
}

// ============================================================================
// SIMULATION TYPES
// ============================================================================

export interface SimulationConfig {
  numCells: number;
  numUsers: number;
  areaKm2: number;
  duration: number;            // Minutes
  scenario: 'urban' | 'suburban' | 'rural' | 'highway';
  trafficProfile: 'uniform' | 'hotspot' | 'event' | 'commute';
  faultInjection: {
    enabled: boolean;
    faultTypes: ProblemCategory[];
    probability: number;
  };
}

export interface SimulationState {
  tick: number;
  timestamp: Date;
  cells: Map<string, CellKPIs>;
  users: UserState[];
  activeProblems: Problem[];
  agentDecisions: ActionRecord[];
}

export interface UserState {
  imsi: string;
  servingCell: string;
  position: { lat: number; lng: number };
  velocity: { speed: number; direction: number };
  rsrp: number;
  sinr: number;
  throughput: number;
  slice?: SliceIdentity;
}
