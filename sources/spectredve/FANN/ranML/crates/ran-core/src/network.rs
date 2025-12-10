//! Network element definitions and topology management

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    error::{RanError, RanResult},
    GeoCoordinate, TimeSeries, NetworkElementId, Measurable, Optimizable, Monitorable,
};

/// Unique identifier for a cell
pub type CellId = Uuid;

/// Unique identifier for a gNodeB
pub type GNodeBId = Uuid;

/// Unique identifier for User Equipment
pub type UEId = Uuid;

/// Cell operating state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellState {
    /// Cell is active and serving traffic
    Active,
    /// Cell is temporarily inactive
    Inactive,
    /// Cell is in sleep mode for energy saving
    Sleep,
    /// Cell is under maintenance
    Maintenance,
    /// Cell has failed
    Failed,
}

/// Cell configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellConfig {
    /// Transmission power in dBm
    pub tx_power: f64,
    /// Antenna tilt in degrees
    pub antenna_tilt: f64,
    /// Azimuth angle in degrees
    pub azimuth: f64,
    /// Frequency band
    pub frequency_band: String,
    /// Channel bandwidth in MHz
    pub bandwidth: f64,
    /// Maximum number of connected UEs
    pub max_ues: u32,
}

impl Default for CellConfig {
    fn default() -> Self {
        Self {
            tx_power: 46.0,        // 40W
            antenna_tilt: 5.0,     // 5 degrees down-tilt
            azimuth: 0.0,          // North-facing
            frequency_band: "n78".to_string(), // 3.5 GHz
            bandwidth: 100.0,      // 100 MHz
            max_ues: 1000,         // 1000 UEs
        }
    }
}

/// Individual cell representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    /// Unique cell identifier
    pub id: CellId,
    /// Human-readable cell name
    pub name: String,
    /// Geographic location
    pub location: GeoCoordinate,
    /// Current operational state
    pub state: CellState,
    /// Cell configuration parameters
    pub config: CellConfig,
    /// Parent gNodeB identifier
    pub gnodeb_id: GNodeBId,
    /// Physical Cell ID (PCI)
    pub pci: u16,
    /// Cell Global Identity
    pub cgi: String,
    /// Currently connected UEs
    pub connected_ues: HashSet<UEId>,
    /// Cell measurement history
    pub measurements: TimeSeries<CellMeasurement>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Monitoring configuration
    pub monitoring_enabled: bool,
    pub monitoring_interval: u64, // seconds
}

impl Cell {
    /// Create a new cell
    pub fn new(
        name: String,
        location: GeoCoordinate,
        gnodeb_id: GNodeBId,
        pci: u16,
        cgi: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            location,
            state: CellState::Active,
            config: CellConfig::default(),
            gnodeb_id,
            pci,
            cgi,
            connected_ues: HashSet::new(),
            measurements: TimeSeries::new(format!("cell_{}_measurements", pci)),
            last_updated: Utc::now(),
            monitoring_enabled: true,
            monitoring_interval: 60, // 1 minute
        }
    }

    /// Add a UE to the cell
    pub fn add_ue(&mut self, ue_id: UEId) -> RanResult<()> {
        if self.connected_ues.len() >= self.config.max_ues as usize {
            return Err(RanError::resource_allocation(
                "UE".to_string(),
                format!("Cell {} at maximum capacity", self.id),
            ));
        }
        
        self.connected_ues.insert(ue_id);
        self.last_updated = Utc::now();
        Ok(())
    }

    /// Remove a UE from the cell
    pub fn remove_ue(&mut self, ue_id: UEId) -> bool {
        let removed = self.connected_ues.remove(&ue_id);
        if removed {
            self.last_updated = Utc::now();
        }
        removed
    }

    /// Get current load as percentage
    pub fn load_percentage(&self) -> f64 {
        (self.connected_ues.len() as f64 / self.config.max_ues as f64) * 100.0
    }

    /// Check if cell is overloaded
    pub fn is_overloaded(&self, threshold: f64) -> bool {
        self.load_percentage() > threshold
    }

    /// Update cell configuration
    pub fn update_config(&mut self, config: CellConfig) -> RanResult<()> {
        // Validate configuration parameters
        if config.tx_power < 0.0 || config.tx_power > 50.0 {
            return Err(RanError::validation(
                "tx_power",
                "Transmission power must be between 0 and 50 dBm",
            ));
        }
        
        if config.bandwidth <= 0.0 {
            return Err(RanError::validation(
                "bandwidth",
                "Bandwidth must be positive",
            ));
        }

        self.config = config;
        self.last_updated = Utc::now();
        Ok(())
    }
}

impl NetworkElementId for Cell {
    fn id(&self) -> Uuid {
        self.id
    }

    fn element_type(&self) -> &'static str {
        "Cell"
    }

    fn display_name(&self) -> String {
        format!("{} (PCI: {})", self.name, self.pci)
    }
}

impl Monitorable for Cell {
    fn is_monitored(&self) -> bool {
        self.monitoring_enabled
    }

    fn enable_monitoring(&mut self) -> RanResult<()> {
        self.monitoring_enabled = true;
        Ok(())
    }

    fn disable_monitoring(&mut self) -> RanResult<()> {
        self.monitoring_enabled = false;
        Ok(())
    }

    fn monitoring_interval(&self) -> u64 {
        self.monitoring_interval
    }

    fn set_monitoring_interval(&mut self, interval: u64) -> RanResult<()> {
        if interval == 0 {
            return Err(RanError::validation(
                "monitoring_interval",
                "Monitoring interval must be greater than 0",
            ));
        }
        self.monitoring_interval = interval;
        Ok(())
    }
}

/// Cell measurement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellMeasurement {
    /// Downlink throughput in Mbps
    pub dl_throughput: f64,
    /// Uplink throughput in Mbps
    pub ul_throughput: f64,
    /// Number of active UEs
    pub active_ues: u32,
    /// Average RSRP in dBm
    pub avg_rsrp: f64,
    /// Average SINR in dB
    pub avg_sinr: f64,
    /// Block Error Rate percentage
    pub bler: f64,
    /// Handover success rate percentage
    pub handover_success_rate: f64,
    /// Resource block utilization percentage
    pub rb_utilization: f64,
}

impl Measurable for Cell {
    type Measurement = CellMeasurement;

    fn measure(&self) -> RanResult<Self::Measurement> {
        // In a real implementation, this would interface with actual network equipment
        // For now, we return a placeholder measurement
        Ok(CellMeasurement {
            dl_throughput: 100.0 * (1.0 - self.load_percentage() / 200.0), // Decreases with load
            ul_throughput: 50.0 * (1.0 - self.load_percentage() / 200.0),
            active_ues: self.connected_ues.len() as u32,
            avg_rsrp: -80.0 + (rand::random::<f64>() - 0.5) * 10.0, // Simulated
            avg_sinr: 15.0 + (rand::random::<f64>() - 0.5) * 10.0,
            bler: self.load_percentage() / 10.0, // Increases with load
            handover_success_rate: 95.0 - self.load_percentage() / 10.0,
            rb_utilization: self.load_percentage(),
        })
    }

    fn measurement_history(&self) -> &TimeSeries<Self::Measurement> {
        &self.measurements
    }
}

impl Optimizable for Cell {
    type Parameters = CellConfig;

    fn get_parameters(&self) -> &Self::Parameters {
        &self.config
    }

    fn set_parameters(&mut self, params: Self::Parameters) -> RanResult<()> {
        self.update_config(params)
    }

    fn get_constraints(&self) -> Vec<crate::optimization::OptimizationConstraint> {
        vec![
            crate::optimization::OptimizationConstraint::new(
                "tx_power_range".to_string(),
                "Transmission power must be between 0 and 50 dBm".to_string(),
                0.0,
                50.0,
            ),
            crate::optimization::OptimizationConstraint::new(
                "bandwidth_positive".to_string(),
                "Bandwidth must be positive".to_string(),
                0.001,
                1000.0,
            ),
        ]
    }
}

/// gNodeB (5G base station) representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNodeB {
    /// Unique gNodeB identifier
    pub id: GNodeBId,
    /// Human-readable name
    pub name: String,
    /// Geographic location
    pub location: GeoCoordinate,
    /// Cells managed by this gNodeB
    pub cells: HashMap<CellId, Cell>,
    /// Connected UEs
    pub connected_ues: HashMap<UEId, UE>,
    /// gNodeB operational state
    pub state: GNodeBState,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// gNodeB operational state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GNodeBState {
    /// Operational and serving traffic
    Active,
    /// Temporarily inactive
    Inactive,
    /// Under maintenance
    Maintenance,
    /// Failed state
    Failed,
}

impl GNodeB {
    /// Create a new gNodeB
    pub fn new(name: String, location: GeoCoordinate) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            location,
            cells: HashMap::new(),
            connected_ues: HashMap::new(),
            state: GNodeBState::Active,
            last_updated: Utc::now(),
        }
    }

    /// Add a cell to the gNodeB
    pub fn add_cell(&mut self, mut cell: Cell) -> RanResult<()> {
        cell.gnodeb_id = self.id;
        self.cells.insert(cell.id, cell);
        self.last_updated = Utc::now();
        Ok(())
    }

    /// Remove a cell from the gNodeB
    pub fn remove_cell(&mut self, cell_id: CellId) -> Option<Cell> {
        let removed = self.cells.remove(&cell_id);
        if removed.is_some() {
            self.last_updated = Utc::now();
        }
        removed
    }

    /// Get total number of connected UEs across all cells
    pub fn total_connected_ues(&self) -> usize {
        self.connected_ues.len()
    }

    /// Get average load across all cells
    pub fn average_load(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        
        let total_load: f64 = self.cells.values().map(|cell| cell.load_percentage()).sum();
        total_load / self.cells.len() as f64
    }
}

impl NetworkElementId for GNodeB {
    fn id(&self) -> Uuid {
        self.id
    }

    fn element_type(&self) -> &'static str {
        "GNodeB"
    }

    fn display_name(&self) -> String {
        format!("{} ({} cells)", self.name, self.cells.len())
    }
}

/// User Equipment representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UE {
    /// Unique UE identifier
    pub id: UEId,
    /// UE category (device type)
    pub category: UECategory,
    /// Current serving cell
    pub serving_cell: Option<CellId>,
    /// Geographic location (if available)
    pub location: Option<GeoCoordinate>,
    /// UE capabilities
    pub capabilities: UECapabilities,
    /// Connection state
    pub state: UEState,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// UE categories based on 3GPP specifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UECategory {
    /// Category 1: Basic IoT devices
    Cat1,
    /// Category 4: Standard smartphones
    Cat4,
    /// Category 6: Advanced smartphones
    Cat6,
    /// Category 9: High-end devices
    Cat9,
    /// Category 12: Premium devices
    Cat12,
    /// Category 15: Advanced industrial IoT
    Cat15,
    /// Category 18: Ultra-high throughput
    Cat18,
    /// Category 20: Maximum capability
    Cat20,
}

/// UE capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UECapabilities {
    /// Maximum downlink data rate in Mbps
    pub max_dl_rate: f64,
    /// Maximum uplink data rate in Mbps
    pub max_ul_rate: f64,
    /// Supported frequency bands
    pub supported_bands: Vec<String>,
    /// MIMO capabilities
    pub mimo_layers: u8,
    /// Carrier aggregation support
    pub carrier_aggregation: bool,
}

/// UE connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UEState {
    /// Idle state
    Idle,
    /// Connected and active
    Connected,
    /// In handover process
    Handover,
    /// Disconnected
    Disconnected,
}

impl UE {
    /// Create a new UE
    pub fn new(category: UECategory) -> Self {
        let capabilities = match category {
            UECategory::Cat1 => UECapabilities {
                max_dl_rate: 10.0,
                max_ul_rate: 5.0,
                supported_bands: vec!["n1".to_string(), "n3".to_string()],
                mimo_layers: 1,
                carrier_aggregation: false,
            },
            UECategory::Cat4 => UECapabilities {
                max_dl_rate: 150.0,
                max_ul_rate: 50.0,
                supported_bands: vec!["n1".to_string(), "n3".to_string(), "n7".to_string()],
                mimo_layers: 2,
                carrier_aggregation: true,
            },
            UECategory::Cat20 => UECapabilities {
                max_dl_rate: 2000.0,
                max_ul_rate: 200.0,
                supported_bands: vec![
                    "n1".to_string(), "n3".to_string(), "n7".to_string(), 
                    "n28".to_string(), "n78".to_string()
                ],
                mimo_layers: 8,
                carrier_aggregation: true,
            },
            _ => UECapabilities {
                max_dl_rate: 300.0,
                max_ul_rate: 75.0,
                supported_bands: vec!["n1".to_string(), "n3".to_string(), "n7".to_string()],
                mimo_layers: 4,
                carrier_aggregation: true,
            },
        };

        Self {
            id: Uuid::new_v4(),
            category,
            serving_cell: None,
            location: None,
            capabilities,
            state: UEState::Idle,
            last_updated: Utc::now(),
        }
    }

    /// Connect UE to a cell
    pub fn connect_to_cell(&mut self, cell_id: CellId) {
        self.serving_cell = Some(cell_id);
        self.state = UEState::Connected;
        self.last_updated = Utc::now();
    }

    /// Disconnect UE
    pub fn disconnect(&mut self) {
        self.serving_cell = None;
        self.state = UEState::Disconnected;
        self.last_updated = Utc::now();
    }
}

impl NetworkElementId for UE {
    fn id(&self) -> Uuid {
        self.id
    }

    fn element_type(&self) -> &'static str {
        "UE"
    }

    fn display_name(&self) -> String {
        format!("UE-{:?}", self.category)
    }
}

/// Network topology representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    /// All gNodeBs in the network
    pub gnodebs: HashMap<GNodeBId, GNodeB>,
    /// Network-wide UE registry
    pub ues: HashMap<UEId, UE>,
    /// Network name
    pub name: String,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl NetworkTopology {
    /// Create a new network topology
    pub fn new(name: String) -> Self {
        Self {
            gnodebs: HashMap::new(),
            ues: HashMap::new(),
            name,
            last_updated: Utc::now(),
        }
    }

    /// Add a gNodeB to the network
    pub fn add_gnodeb(&mut self, gnodeb: GNodeB) {
        self.gnodebs.insert(gnodeb.id, gnodeb);
        self.last_updated = Utc::now();
    }

    /// Add a UE to the network
    pub fn add_ue(&mut self, ue: UE) {
        self.ues.insert(ue.id, ue);
        self.last_updated = Utc::now();
    }

    /// Get all cells in the network
    pub fn get_all_cells(&self) -> Vec<&Cell> {
        self.gnodebs.values()
            .flat_map(|gnodeb| gnodeb.cells.values())
            .collect()
    }

    /// Find the best serving cell for a UE based on location
    pub fn find_best_cell(&self, ue_location: GeoCoordinate) -> Option<CellId> {
        let mut best_cell = None;
        let mut best_distance = f64::INFINITY;

        for gnodeb in self.gnodebs.values() {
            for cell in gnodeb.cells.values() {
                if cell.state == CellState::Active {
                    let distance = ue_location.distance_to(&cell.location);
                    if distance < best_distance {
                        best_distance = distance;
                        best_cell = Some(cell.id);
                    }
                }
            }
        }

        best_cell
    }

    /// Get network statistics
    pub fn get_statistics(&self) -> NetworkStatistics {
        let total_cells = self.get_all_cells().len();
        let active_cells = self.get_all_cells()
            .iter()
            .filter(|cell| cell.state == CellState::Active)
            .count();
        
        let total_ues = self.ues.len();
        let connected_ues = self.ues.values()
            .filter(|ue| ue.state == UEState::Connected)
            .count();

        let average_load = if !self.gnodebs.is_empty() {
            self.gnodebs.values()
                .map(|gnodeb| gnodeb.average_load())
                .sum::<f64>() / self.gnodebs.len() as f64
        } else {
            0.0
        };

        NetworkStatistics {
            total_gnodebs: self.gnodebs.len(),
            total_cells,
            active_cells,
            total_ues,
            connected_ues,
            average_load,
        }
    }
}

/// Network statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatistics {
    /// Total number of gNodeBs
    pub total_gnodebs: usize,
    /// Total number of cells
    pub total_cells: usize,
    /// Number of active cells
    pub active_cells: usize,
    /// Total number of UEs
    pub total_ues: usize,
    /// Number of connected UEs
    pub connected_ues: usize,
    /// Average load across all cells
    pub average_load: f64,
}

/// Generic network element trait
pub trait NetworkElement: NetworkElementId + Send + Sync {
    /// Get the element's location
    fn location(&self) -> Option<GeoCoordinate>;
    
    /// Get the last update timestamp
    fn last_updated(&self) -> DateTime<Utc>;
    
    /// Update the element's timestamp
    fn touch(&mut self);
}

impl NetworkElement for Cell {
    fn location(&self) -> Option<GeoCoordinate> {
        Some(self.location)
    }

    fn last_updated(&self) -> DateTime<Utc> {
        self.last_updated
    }

    fn touch(&mut self) {
        self.last_updated = Utc::now();
    }
}

impl NetworkElement for GNodeB {
    fn location(&self) -> Option<GeoCoordinate> {
        Some(self.location)
    }

    fn last_updated(&self) -> DateTime<Utc> {
        self.last_updated
    }

    fn touch(&mut self) {
        self.last_updated = Utc::now();
    }
}

impl NetworkElement for UE {
    fn location(&self) -> Option<GeoCoordinate> {
        self.location
    }

    fn last_updated(&self) -> DateTime<Utc> {
        self.last_updated
    }

    fn touch(&mut self) {
        self.last_updated = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cell_creation() {
        let location = GeoCoordinate::new(52.520008, 13.404954, None);
        let gnodeb_id = Uuid::new_v4();
        let cell = Cell::new(
            "Test Cell".to_string(),
            location,
            gnodeb_id,
            123,
            "12345".to_string(),
        );

        assert_eq!(cell.name, "Test Cell");
        assert_eq!(cell.pci, 123);
        assert_eq!(cell.gnodeb_id, gnodeb_id);
        assert_eq!(cell.state, CellState::Active);
    }

    #[test]
    fn test_cell_load_management() {
        let location = GeoCoordinate::new(52.520008, 13.404954, None);
        let gnodeb_id = Uuid::new_v4();
        let mut cell = Cell::new(
            "Test Cell".to_string(),
            location,
            gnodeb_id,
            123,
            "12345".to_string(),
        );

        // Set low capacity for testing
        cell.config.max_ues = 2;

        let ue1 = Uuid::new_v4();
        let ue2 = Uuid::new_v4();
        let ue3 = Uuid::new_v4();

        // Add UEs up to capacity
        assert!(cell.add_ue(ue1).is_ok());
        assert!(cell.add_ue(ue2).is_ok());
        
        // Should fail when exceeding capacity
        assert!(cell.add_ue(ue3).is_err());
        
        assert_eq!(cell.load_percentage(), 100.0);
        assert!(cell.is_overloaded(90.0));
    }

    #[test]
    fn test_gnodeb_operations() {
        let location = GeoCoordinate::new(52.520008, 13.404954, None);
        let mut gnodeb = GNodeB::new("Test gNodeB".to_string(), location);

        let cell1 = Cell::new(
            "Cell 1".to_string(),
            location,
            gnodeb.id,
            123,
            "12345".to_string(),
        );
        
        let cell2 = Cell::new(
            "Cell 2".to_string(),
            location,
            gnodeb.id,
            124,
            "12346".to_string(),
        );

        gnodeb.add_cell(cell1).unwrap();
        gnodeb.add_cell(cell2).unwrap();

        assert_eq!(gnodeb.cells.len(), 2);
        assert_eq!(gnodeb.average_load(), 0.0); // No UEs connected
    }

    #[test]
    fn test_network_topology() {
        let mut topology = NetworkTopology::new("Test Network".to_string());
        
        let location = GeoCoordinate::new(52.520008, 13.404954, None);
        let gnodeb = GNodeB::new("Test gNodeB".to_string(), location);
        let ue = UE::new(UECategory::Cat4);

        topology.add_gnodeb(gnodeb);
        topology.add_ue(ue);

        let stats = topology.get_statistics();
        assert_eq!(stats.total_gnodebs, 1);
        assert_eq!(stats.total_ues, 1);
        assert_eq!(stats.connected_ues, 0);
    }

    #[test]
    fn test_ue_capabilities() {
        let ue_cat1 = UE::new(UECategory::Cat1);
        let ue_cat20 = UE::new(UECategory::Cat20);

        assert!(ue_cat1.capabilities.max_dl_rate < ue_cat20.capabilities.max_dl_rate);
        assert!(ue_cat1.capabilities.mimo_layers < ue_cat20.capabilities.mimo_layers);
        assert!(!ue_cat1.capabilities.carrier_aggregation);
        assert!(ue_cat20.capabilities.carrier_aggregation);
    }
}