/**
 * TITAN Dashboard Type Definitions
 * Type interfaces for RAN optimization dashboard components
 *
 * @module ui/types
 * @version 7.0.0-alpha.1
 */

// ============================================================================
// Cell and Network Types
// ============================================================================

export interface CellStatus {
  cell_id: string;
  pci: number; // Physical Cell ID
  earfcn: number; // E-UTRA Absolute Radio Frequency Channel Number
  power_dbm: number; // Transmit power in dBm (-130 to 46)
  tilt_deg: number; // Antenna tilt in degrees (0-15)
  azimuth_deg: number; // Azimuth angle
  latitude: number;
  longitude: number;
  status: 'active' | 'degraded' | 'outage' | 'optimizing';
  kpi: CellKPI;
  last_optimized?: string;
}

export interface CellKPI {
  rsrp_avg: number; // Reference Signal Received Power (dBm)
  rsrq_avg: number; // Reference Signal Received Quality (dB)
  sinr_avg: number; // Signal-to-Interference-plus-Noise Ratio (dB)
  throughput_mbps: number;
  rrc_connections: number;
  prb_utilization: number; // Physical Resource Block utilization (0-100%)
  ho_success_rate: number; // Handover success rate (0-100%)
  drop_rate: number; // Call drop rate (0-100%)
}

export interface InterferenceMatrix {
  cells: string[];
  matrix: number[][]; // NxN interference matrix
  threshold: number;
  timestamp: string;
}

export interface FMAlarm {
  id: string;
  severity: 'critical' | 'major' | 'minor' | 'warning';
  mo_class: string; // Managed Object class (e.g., "EUtranCellFDD")
  mo_id: string;
  alarm_text: string;
  additional_info?: Record<string, any>;
  raised_at: string;
  cleared_at?: string;
}

export interface PMCounter {
  counter_name: string;
  value: number;
  cell_id: string;
  timestamp: string;
}

// ============================================================================
// GNN Optimization Types
// ============================================================================

export interface OptimizationEvent {
  id: string;
  timestamp: string;
  event_type: 'gnn_decision' | 'council_debate' | 'hitl_approval' | 'execution' | 'rollback';
  cell_ids: string[];
  parameters_changed: ParameterChange[];
  reasoning: string;
  agent_id?: string;
  confidence: number;
  status: 'pending' | 'approved' | 'rejected' | 'executed' | 'rolled_back';
  kpi_impact?: KPIImpact;
}

export interface ParameterChange {
  parameter: string;
  old_value: number;
  new_value: number;
  cell_id: string;
  bounds: [number, number]; // [min, max] for 3GPP compliance
}

export interface KPIImpact {
  before: CellKPI;
  after: CellKPI;
  delta_percent: Record<string, number>;
}

// ============================================================================
// HITL Approval Types
// ============================================================================

export interface ApprovalRequest {
  id: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  action: string;
  target: string[];
  justification: string;
  proposed_changes: ParameterChange[];
  predicted_impact: string;
  safety_checks: SafetyCheck[];
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  expires_at: string;
}

export interface SafetyCheck {
  check_name: string;
  passed: boolean;
  message: string;
  severity: 'info' | 'warning' | 'error';
}

export interface ApprovalResponse {
  approval_id: string;
  authorized: boolean;
  signature: string;
  operator_id: string;
  notes?: string;
  timestamp: string;
}

// ============================================================================
// Dashboard State Types
// ============================================================================

export interface DashboardState {
  cells: Map<string, CellStatus>;
  alarms: FMAlarm[];
  optimizationTimeline: OptimizationEvent[];
  pendingApprovals: ApprovalRequest[];
  interferenceMatrix?: InterferenceMatrix;
  selectedCells: string[];
  viewMode: 'overview' | 'detailed' | 'optimization' | 'approval';
}

// ============================================================================
// AG-UI Protocol Types
// ============================================================================

export interface AGUIEvent {
  type: string;
  timestamp: string;
  payload: any;
}

export interface GenerativeUIComponent {
  component: string;
  props: Record<string, any>;
  interactive: boolean;
}

// ============================================================================
// OpenTUI Component Types
// ============================================================================

export interface TableColumn<T> {
  key: keyof T;
  header: string;
  width?: number;
  formatter?: (value: any) => string;
}

export interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface CommandPaletteAction {
  id: string;
  label: string;
  description: string;
  shortcut?: string;
  handler: () => void | Promise<void>;
}
