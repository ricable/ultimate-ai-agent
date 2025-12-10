// RAN-specific data models for Ericsson infrastructure

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RanNode {
    pub node_id: String,
    pub node_type: NodeType,
    pub location: Option<GeoLocation>,
    pub cells: Vec<Cell>,
    pub parameters: HashMap<String, ParameterValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    #[serde(rename = "gNB")]
    GNodeB,      // 5G base station
    #[serde(rename = "eNB")]
    ENodeB,      // 4G base station
    #[serde(rename = "5G-SA")]
    FiveGSA,     // 5G Standalone
    #[serde(rename = "4G-LTE")]
    FourGLTE,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub cell_id: String,
    pub sector_id: String,
    pub pci: u32,  // Physical Cell ID
    pub frequency_band: String,
    pub bandwidth_mhz: u32,
    pub max_power_dbm: f64,
    pub azimuth: f64,
    pub tilt: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParameterValue {
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alarm {
    pub alarm_id: String,
    pub timestamp: i64,
    pub severity: AlarmSeverity,
    pub node_id: String,
    pub cell_id: Option<String>,
    pub alarm_type: String,
    pub description: String,
    pub additional_info: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlarmSeverity {
    Critical,
    Major,
    Minor,
    Warning,
    Cleared,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpiMeasurement {
    pub timestamp: i64,
    pub node_id: String,
    pub cell_id: Option<String>,
    pub kpi_name: String,
    pub value: f64,
    pub unit: String,
    pub granularity: Granularity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Granularity {
    Hourly,
    Daily,
    Weekly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counter {
    pub counter_name: String,
    pub value: u64,
    pub timestamp: i64,
    pub node_id: String,
    pub cell_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterChangeProposal {
    pub proposal_id: String,
    pub timestamp: i64,
    pub node_id: String,
    pub cell_id: Option<String>,
    pub parameter_name: String,
    pub current_value: ParameterValue,
    pub proposed_value: ParameterValue,
    pub confidence_score: f64,
    pub rationale: String,
    pub expected_impact: HashMap<String, f64>,
    pub risk_assessment: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultEvent {
    pub event_id: String,
    pub timestamp: i64,
    pub node_id: String,
    pub fault_type: FaultType,
    pub affected_cells: Vec<String>,
    pub metrics_at_fault: HashMap<String, f64>,
    pub recovery_timestamp: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaultType {
    HardwareFailure,
    SoftwareError,
    ConfigurationIssue,
    CapacityExceeded,
    InterferenceDetected,
    BackhaulIssue,
    PowerOutage,
}
