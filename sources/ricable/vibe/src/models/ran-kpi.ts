/**
 * RAN KPI Data Models and Schemas
 * Covers: Accessibility, Retainability, Radio Quality, Mobility, Uplink Interference
 */

import { z } from 'zod';

// ============================================================================
// CELL IDENTIFICATION
// ============================================================================

export const CellIdSchema = z.object({
  cellId: z.string(),
  enodebId: z.string(),
  sectorId: z.number().min(0).max(2),
  frequency: z.number(), // MHz
  band: z.string(), // e.g., "n78", "B3", "B7"
  technology: z.enum(['LTE', 'NR', '5G-NSA', '5G-SA']),
  pci: z.number().min(0).max(503), // Physical Cell ID
  tac: z.number(), // Tracking Area Code
  latitude: z.number().optional(),
  longitude: z.number().optional(),
});

export type CellId = z.infer<typeof CellIdSchema>;

// ============================================================================
// ACCESSIBILITY KPIs
// ============================================================================

export const AccessibilityKPISchema = z.object({
  timestamp: z.date(),
  cellId: z.string(),

  // RRC Setup Success Rate
  rrcSetupAttempts: z.number().int().min(0),
  rrcSetupSuccess: z.number().int().min(0),
  rrcSetupFailure: z.number().int().min(0),
  rrcSetupSuccessRate: z.number().min(0).max(100),

  // E-RAB Setup Success Rate
  erabSetupAttempts: z.number().int().min(0),
  erabSetupSuccess: z.number().int().min(0),
  erabSetupFailure: z.number().int().min(0),
  erabSetupSuccessRate: z.number().min(0).max(100),

  // S1 Signaling Connection Setup
  s1SigConnEstabAttempts: z.number().int().min(0),
  s1SigConnEstabSuccess: z.number().int().min(0),
  s1SigConnEstabSuccessRate: z.number().min(0).max(100),

  // Initial Context Setup (combined accessibility)
  initialContextSetupAttempts: z.number().int().min(0),
  initialContextSetupSuccess: z.number().int().min(0),
  initialContextSetupSuccessRate: z.number().min(0).max(100),

  // Failure causes breakdown
  rrcFailureCauses: z.object({
    congestion: z.number().int().min(0),
    unspecified: z.number().int().min(0),
    timer: z.number().int().min(0),
    radioResourceNotAvailable: z.number().int().min(0),
  }).optional(),
});

export type AccessibilityKPI = z.infer<typeof AccessibilityKPISchema>;

// ============================================================================
// RETAINABILITY KPIs
// ============================================================================

export const RetainabilityKPISchema = z.object({
  timestamp: z.date(),
  cellId: z.string(),

  // E-RAB Drop Rate
  erabNormalRelease: z.number().int().min(0),
  erabAbnormalRelease: z.number().int().min(0),
  erabDropRate: z.number().min(0).max(100),

  // Call Drop Rate
  voiceCallAttempts: z.number().int().min(0),
  voiceCallDrops: z.number().int().min(0),
  voiceCallDropRate: z.number().min(0).max(100),

  // Data Session Retainability
  dataSessionAttempts: z.number().int().min(0),
  dataSessionDrops: z.number().int().min(0),
  dataSessionRetainability: z.number().min(0).max(100),

  // UE Context Release causes
  contextReleaseCauses: z.object({
    radioConnectionWithUeLost: z.number().int().min(0),
    userInactivity: z.number().int().min(0),
    s1UPathSwitch: z.number().int().min(0),
    interRatRedirection: z.number().int().min(0),
    intraLteRedirection: z.number().int().min(0),
    x2Handover: z.number().int().min(0),
    s1Handover: z.number().int().min(0),
    other: z.number().int().min(0),
  }).optional(),
});

export type RetainabilityKPI = z.infer<typeof RetainabilityKPISchema>;

// ============================================================================
// RADIO QUALITY KPIs
// ============================================================================

export const RadioQualityKPISchema = z.object({
  timestamp: z.date(),
  cellId: z.string(),

  // Downlink Quality
  dlAvgCqi: z.number().min(0).max(15),
  dlCqiDistribution: z.array(z.number()).length(16).optional(),
  dlRi1Ratio: z.number().min(0).max(100), // Rank Indicator 1 ratio
  dlRi2Ratio: z.number().min(0).max(100), // Rank Indicator 2 ratio
  dlBlerPercent: z.number().min(0).max(100), // Block Error Rate

  // Uplink Quality
  ulSinrAvg: z.number(), // dB
  ulSinrP10: z.number(), // 10th percentile
  ulSinrP50: z.number(), // 50th percentile
  ulSinrP90: z.number(), // 90th percentile
  ulBlerPercent: z.number().min(0).max(100),

  // RSRP/RSRQ Distribution (from UE measurements)
  rsrpAvg: z.number(), // dBm
  rsrpP10: z.number(),
  rsrpP50: z.number(),
  rsrpP90: z.number(),
  rsrqAvg: z.number(), // dB
  rsrqP10: z.number(),
  rsrqP50: z.number(),
  rsrqP90: z.number(),

  // Spectral Efficiency
  dlSpectralEfficiency: z.number().min(0), // bits/s/Hz
  ulSpectralEfficiency: z.number().min(0),
});

export type RadioQualityKPI = z.infer<typeof RadioQualityKPISchema>;

// ============================================================================
// MOBILITY / HANDOVER KPIs
// ============================================================================

export const MobilityKPISchema = z.object({
  timestamp: z.date(),
  cellId: z.string(),

  // Intra-Frequency Handovers
  intraFreqHoAttempts: z.number().int().min(0),
  intraFreqHoSuccess: z.number().int().min(0),
  intraFreqHoFailure: z.number().int().min(0),
  intraFreqHoSuccessRate: z.number().min(0).max(100),

  // Inter-Frequency Handovers
  interFreqHoAttempts: z.number().int().min(0),
  interFreqHoSuccess: z.number().int().min(0),
  interFreqHoFailure: z.number().int().min(0),
  interFreqHoSuccessRate: z.number().min(0).max(100),

  // Inter-RAT Handovers (LTE <-> NR, LTE <-> 3G)
  interRatHoAttempts: z.number().int().min(0),
  interRatHoSuccess: z.number().int().min(0),
  interRatHoFailure: z.number().int().min(0),
  interRatHoSuccessRate: z.number().min(0).max(100),

  // X2-based Handovers
  x2HoAttempts: z.number().int().min(0),
  x2HoSuccess: z.number().int().min(0),
  x2HoSuccessRate: z.number().min(0).max(100),

  // S1-based Handovers
  s1HoAttempts: z.number().int().min(0),
  s1HoSuccess: z.number().int().min(0),
  s1HoSuccessRate: z.number().min(0).max(100),

  // Too Early/Late/Wrong Cell HO
  tooEarlyHo: z.number().int().min(0),
  tooLateHo: z.number().int().min(0),
  wrongCellHo: z.number().int().min(0),
  pingPongHo: z.number().int().min(0),

  // Incoming vs Outgoing
  incomingHoTotal: z.number().int().min(0),
  outgoingHoTotal: z.number().int().min(0),
});

export type MobilityKPI = z.infer<typeof MobilityKPISchema>;

// ============================================================================
// UPLINK INTERFERENCE KPIs
// ============================================================================

export const UplinkInterferenceKPISchema = z.object({
  timestamp: z.date(),
  cellId: z.string(),

  // PRB Interference
  prbUlInterferenceAvg: z.number(), // dBm
  prbUlInterferenceP10: z.number(),
  prbUlInterferenceP50: z.number(),
  prbUlInterferenceP90: z.number(),
  prbUlInterferenceP99: z.number(),

  // Per-PRB interference distribution (100 PRBs for 20MHz)
  prbInterferenceDistribution: z.array(z.number()).optional(),

  // Interference over Thermal (IoT)
  iotAvg: z.number(), // dB
  iotP95: z.number(),

  // Received Interference Power
  rip: z.number(), // dBm - Received Interference Power

  // External Interference Indicators
  externalInterferenceDetected: z.boolean(),
  externalInterferenceLevel: z.enum(['none', 'low', 'medium', 'high']),

  // PUSCH SINR Degradation
  puschSinrDegradation: z.number(), // dB from baseline

  // Affected PRB ratio
  highInterferencePrbRatio: z.number().min(0).max(100),
});

export type UplinkInterferenceKPI = z.infer<typeof UplinkInterferenceKPISchema>;

// ============================================================================
// UPLINK POWER CONTROL KPIs
// ============================================================================

export const UplinkPowerControlKPISchema = z.object({
  timestamp: z.date(),
  cellId: z.string(),

  // Current Configuration
  p0NominalPusch: z.number(), // dBm (-126 to 24)
  p0NominalPucch: z.number(), // dBm
  alpha: z.number().min(0).max(1), // Path loss compensation factor (0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

  // UE Transmit Power Distribution
  ueTxPowerAvg: z.number(), // dBm
  ueTxPowerP10: z.number(),
  ueTxPowerP50: z.number(),
  ueTxPowerP90: z.number(),
  ueTxPowerMax: z.number(), // Pcmax typically 23 dBm

  // Power Headroom Distribution
  powerHeadroomAvg: z.number(), // dB
  powerHeadroomP10: z.number(),
  powerHeadroomP50: z.number(),
  powerHeadroomP90: z.number(),
  negativePowerHeadroomRatio: z.number().min(0).max(100), // % of UEs with PHR < 0

  // Fractional Path Loss Metrics
  pathLossAvg: z.number(), // dB
  pathLossP10: z.number(),
  pathLossP50: z.number(),
  pathLossP90: z.number(),

  // TPC Commands
  tpcUpCommands: z.number().int().min(0),
  tpcDownCommands: z.number().int().min(0),
  tpcAccumulatedOffset: z.number(),

  // Power Limited UEs
  powerLimitedUeRatio: z.number().min(0).max(100),
});

export type UplinkPowerControlKPI = z.infer<typeof UplinkPowerControlKPISchema>;

// ============================================================================
// NEIGHBOR RELATION DATA
// ============================================================================

export const NeighborRelationSchema = z.object({
  sourceCellId: z.string(),
  targetCellId: z.string(),
  relationshipType: z.enum(['intra-freq', 'inter-freq', 'inter-rat']),

  // Source Cell Measurements
  sourcePci: z.number().min(0).max(503),
  sourceFrequency: z.number(),
  sourceRsrp: z.number(), // Avg RSRP at source
  sourceSinr: z.number(), // Avg SINR at source

  // Target Cell Measurements (from UE reports)
  targetPci: z.number().min(0).max(503),
  targetFrequency: z.number(),
  targetRsrp: z.number(), // Avg RSRP to target
  targetSinr: z.number(), // Avg SINR to target

  // Handover Statistics
  hoAttempts: z.number().int().min(0),
  hoSuccess: z.number().int().min(0),
  hoFailure: z.number().int().min(0),
  hoSuccessRate: z.number().min(0).max(100),

  // Timing and offset
  a3Offset: z.number(), // dB
  hysteresis: z.number(), // dB
  timeToTrigger: z.number(), // ms

  // Neighbor quality indicators
  neighborQuality: z.enum(['excellent', 'good', 'fair', 'poor', 'missing']),

  // Geographical distance (if available)
  distance: z.number().optional(), // meters
  azimuthDifference: z.number().optional(), // degrees
});

export type NeighborRelation = z.infer<typeof NeighborRelationSchema>;

// ============================================================================
// COMBINED CELL KPI SNAPSHOT
// ============================================================================

export const CellKPISnapshotSchema = z.object({
  timestamp: z.date(),
  cell: CellIdSchema,
  accessibility: AccessibilityKPISchema.omit({ timestamp: true, cellId: true }),
  retainability: RetainabilityKPISchema.omit({ timestamp: true, cellId: true }),
  radioQuality: RadioQualityKPISchema.omit({ timestamp: true, cellId: true }),
  mobility: MobilityKPISchema.omit({ timestamp: true, cellId: true }),
  uplinkInterference: UplinkInterferenceKPISchema.omit({ timestamp: true, cellId: true }),
  uplinkPowerControl: UplinkPowerControlKPISchema.omit({ timestamp: true, cellId: true }),
});

export type CellKPISnapshot = z.infer<typeof CellKPISnapshotSchema>;

// ============================================================================
// TIME SERIES DATA STRUCTURES
// ============================================================================

export interface TimeSeriesPoint<T> {
  timestamp: Date;
  value: T;
  metadata?: Record<string, unknown>;
}

export interface KPITimeSeries {
  cellId: string;
  kpiName: string;
  domain: 'accessibility' | 'retainability' | 'radioQuality' | 'mobility' | 'uplinkInterference' | 'uplinkPowerControl';
  granularity: '1min' | '5min' | '15min' | '1hour' | '1day';
  startTime: Date;
  endTime: Date;
  dataPoints: TimeSeriesPoint<number>[];
}

// ============================================================================
// ANOMALY DETECTION TYPES
// ============================================================================

export const AnomalyTypeSchema = z.enum([
  'spike',           // Sudden increase
  'dip',             // Sudden decrease
  'trend_shift',     // Change in trend
  'level_shift',     // Change in baseline level
  'variance_change', // Change in volatility
  'seasonality_break', // Break from expected seasonal pattern
  'outlier',         // Single point outlier
  'collective',      // Group of anomalous points
]);

export type AnomalyType = z.infer<typeof AnomalyTypeSchema>;

export const AnomalySeveritySchema = z.enum(['low', 'medium', 'high', 'critical']);
export type AnomalySeverity = z.infer<typeof AnomalySeveritySchema>;

export const DetectedAnomalySchema = z.object({
  id: z.string().uuid(),
  cellId: z.string(),
  kpiName: z.string(),
  domain: z.string(),
  timestamp: z.date(),
  anomalyType: AnomalyTypeSchema,
  severity: AnomalySeveritySchema,

  // Statistical details
  observedValue: z.number(),
  expectedValue: z.number(),
  deviation: z.number(), // Standard deviations from expected
  confidence: z.number().min(0).max(1),

  // Context
  duration: z.number().optional(), // Duration in minutes if persistent
  affectedKpis: z.array(z.string()).optional(),
  possibleCauses: z.array(z.string()).optional(),

  // Root cause analysis reference
  rootCauseAnalysisId: z.string().uuid().optional(),
});

export type DetectedAnomaly = z.infer<typeof DetectedAnomalySchema>;

// ============================================================================
// ROOT CAUSE ANALYSIS TYPES
// ============================================================================

export const RootCauseCategorySchema = z.enum([
  'hardware_failure',
  'software_issue',
  'configuration_error',
  'capacity_exhaustion',
  'interference',
  'coverage_issue',
  'backhaul_issue',
  'core_network_issue',
  'parameter_drift',
  'neighbor_relation_issue',
  'mobility_issue',
  'power_control_issue',
  'external_factor',
  'unknown',
]);

export type RootCauseCategory = z.infer<typeof RootCauseCategorySchema>;

export const RootCauseAnalysisSchema = z.object({
  id: z.string().uuid(),
  anomalyIds: z.array(z.string().uuid()),
  analysisTimestamp: z.date(),

  // Primary findings
  primaryCause: RootCauseCategorySchema,
  primaryCauseConfidence: z.number().min(0).max(1),

  // Contributing factors
  contributingFactors: z.array(z.object({
    category: RootCauseCategorySchema,
    description: z.string(),
    confidence: z.number().min(0).max(1),
    evidence: z.array(z.string()),
  })),

  // Affected cells (for propagation analysis)
  affectedCells: z.array(z.string()),
  propagationPattern: z.enum(['isolated', 'cluster', 'cascade', 'regional']).optional(),

  // Recommendations
  recommendations: z.array(z.object({
    action: z.string(),
    priority: z.enum(['immediate', 'high', 'medium', 'low']),
    expectedImpact: z.string(),
    parameters: z.record(z.unknown()).optional(),
  })),

  // For power control issues
  suggestedP0Adjustment: z.number().optional(),
  suggestedAlphaAdjustment: z.number().optional(),
});

export type RootCauseAnalysis = z.infer<typeof RootCauseAnalysisSchema>;

// ============================================================================
// GNN GRAPH STRUCTURES
// ============================================================================

export interface CellGraphNode {
  id: string;
  cellId: string;
  features: number[]; // Feature vector for GNN
  position?: [number, number]; // lat, lon
}

export interface CellGraphEdge {
  source: string;
  target: string;
  relationshipType: 'intra-freq' | 'inter-freq' | 'inter-rat';
  features: number[]; // Edge features (SINR delta, HO success rate, etc.)
  weight: number;
}

export interface CellGraph {
  nodes: CellGraphNode[];
  edges: CellGraphEdge[];
  adjacencyMatrix?: number[][];
  metadata: {
    timestamp: Date;
    numCells: number;
    numRelations: number;
  };
}

export default {
  CellIdSchema,
  AccessibilityKPISchema,
  RetainabilityKPISchema,
  RadioQualityKPISchema,
  MobilityKPISchema,
  UplinkInterferenceKPISchema,
  UplinkPowerControlKPISchema,
  NeighborRelationSchema,
  CellKPISnapshotSchema,
  AnomalyTypeSchema,
  AnomalySeveritySchema,
  DetectedAnomalySchema,
  RootCauseCategorySchema,
  RootCauseAnalysisSchema,
};
