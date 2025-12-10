// TypeScript type definitions for RAN models

export type NodeType = 'gNB' | 'eNB' | '5G-SA' | '4G-LTE';

export interface GeoLocation {
  latitude: number;
  longitude: number;
  altitude?: number;
}

export interface Cell {
  cellId: string;
  sectorId: string;
  pci: number;
  frequencyBand: string;
  bandwidthMhz: number;
  maxPowerDbm: number;
  azimuth: number;
  tilt: number;
}

export interface RanNode {
  nodeId: string;
  nodeType: NodeType;
  location?: GeoLocation;
  cells: Cell[];
  parameters: Record<string, ParameterValue>;
}

export type ParameterValue = number | string | boolean;

export type AlarmSeverity = 'Critical' | 'Major' | 'Minor' | 'Warning' | 'Cleared';

export interface Alarm {
  alarmId: string;
  timestamp: number;
  severity: AlarmSeverity;
  nodeId: string;
  cellId?: string;
  alarmType: string;
  description: string;
  additionalInfo: Record<string, string>;
}

export type Granularity = 'Hourly' | 'Daily' | 'Weekly';

export interface KpiMeasurement {
  timestamp: number;
  nodeId: string;
  cellId?: string;
  kpiName: string;
  value: number;
  unit: string;
  granularity: Granularity;
}

export interface Counter {
  counterName: string;
  value: number;
  timestamp: number;
  nodeId: string;
  cellId?: string;
}

export type RiskLevel = 'Low' | 'Medium' | 'High' | 'Critical';

export interface ParameterChangeProposal {
  proposalId: string;
  timestamp: number;
  nodeId: string;
  cellId?: string;
  parameterName: string;
  currentValue: ParameterValue;
  proposedValue: ParameterValue;
  confidenceScore: number;
  rationale: string;
  expectedImpact: Record<string, number>;
  riskAssessment: RiskLevel;
}

export type FaultType =
  | 'HardwareFailure'
  | 'SoftwareError'
  | 'ConfigurationIssue'
  | 'CapacityExceeded'
  | 'InterferenceDetected'
  | 'BackhaulIssue'
  | 'PowerOutage';

export interface FaultEvent {
  eventId: string;
  timestamp: number;
  nodeId: string;
  faultType: FaultType;
  affectedCells: string[];
  metricsAtFault: Record<string, number>;
  recoveryTimestamp?: number;
}

export interface TimeSeriesPoint {
  timestamp: number;
  value: number;
  metadata: Record<string, string>;
}

export interface MultiVariatePoint {
  timestamp: number;
  features: Record<string, number>;
  labels: Record<string, string>;
}

export interface PredictionResult {
  timestamp: number;
  predictedValue: number;
  confidenceInterval: [number, number];
  anomalyScore: number;
  featureImportance: Record<string, number>;
}

export interface RanContext {
  nodeType: string;
  cellId: string;
  clusterId?: string;
  regionId?: string;
}

export interface AggregationConfig {
  granularities: Granularity[];
  aggregationMethods: ('mean' | 'sum' | 'max' | 'min' | 'count')[];
  timeRangeHours: number;
}

export interface AgentTask {
  taskId: string;
  agentType: string;
  priority: number;
  payload: unknown;
  timestamp: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: unknown;
  error?: string;
}
