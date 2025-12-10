//! Resource management abstractions for RAN optimization

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    error::{RanError, RanResult},
    TimeSeries, TimeSeriesPoint,
};

/// Types of resources in a RAN system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Radio spectrum resource
    Spectrum,
    /// Transmission power
    Power,
    /// Computational resources
    Compute,
    /// Memory resources
    Memory,
    /// Network bandwidth
    Bandwidth,
    /// Physical Resource Blocks (PRBs)
    PhysicalResourceBlocks,
    /// Antenna resources
    Antenna,
    /// Processing capacity
    Processing,
}

impl ResourceType {
    /// Get the unit of measurement for this resource type
    pub fn unit(&self) -> &'static str {
        match self {
            ResourceType::Spectrum => "MHz",
            ResourceType::Power => "dBm",
            ResourceType::Compute => "MIPS",
            ResourceType::Memory => "MB",
            ResourceType::Bandwidth => "Mbps",
            ResourceType::PhysicalResourceBlocks => "PRBs",
            ResourceType::Antenna => "ports",
            ResourceType::Processing => "ops/sec",
        }
    }

    /// Check if this resource type is shareable
    pub fn is_shareable(&self) -> bool {
        match self {
            ResourceType::Spectrum => false, // Exclusive allocation in frequency/time
            ResourceType::Power => false,    // Power budget is finite
            ResourceType::Compute => true,   // Can be time-multiplexed
            ResourceType::Memory => true,    // Can be allocated in chunks
            ResourceType::Bandwidth => true, // Can be shared/partitioned
            ResourceType::PhysicalResourceBlocks => false, // Exclusive in time/frequency
            ResourceType::Antenna => false, // Physical limitation
            ResourceType::Processing => true, // Can be scheduled
        }
    }
}

/// Resource allocation state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationState {
    /// Resource is available for allocation
    Available,
    /// Resource is allocated and in use
    Allocated,
    /// Resource is reserved for future use
    Reserved,
    /// Resource is unavailable (maintenance, failure, etc.)
    Unavailable,
    /// Resource allocation is pending
    Pending,
}

/// Resource constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceConstraint {
    /// Maximum available amount
    MaximumCapacity { value: f64 },
    /// Minimum required amount
    MinimumRequired { value: f64 },
    /// Interference constraint
    InterferenceLimit { threshold: f64 },
    /// Quality constraint
    QualityRequirement { min_quality: f64 },
    /// Time-based constraint
    TimeWindow { start: DateTime<Utc>, end: DateTime<Utc> },
    /// Geographic constraint
    GeographicBounds { latitude_range: (f64, f64), longitude_range: (f64, f64) },
    /// Regulatory constraint
    Regulatory { regulation: String, limit: f64 },
}

/// Individual resource representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Unique resource identifier
    pub id: Uuid,
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource name/description
    pub name: String,
    /// Total capacity
    pub total_capacity: f64,
    /// Currently allocated amount
    pub allocated_amount: f64,
    /// Currently available amount
    pub available_amount: f64,
    /// Current allocation state
    pub state: AllocationState,
    /// Resource constraints
    pub constraints: Vec<ResourceConstraint>,
    /// Utilization history
    pub utilization_history: TimeSeries<f64>,
    /// Owner/controller identifier
    pub owner_id: Option<Uuid>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Resource {
    /// Create a new resource
    pub fn new(
        resource_type: ResourceType,
        name: String,
        total_capacity: f64,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            resource_type,
            name: name.clone(),
            total_capacity,
            allocated_amount: 0.0,
            available_amount: total_capacity,
            state: AllocationState::Available,
            constraints: Vec::new(),
            utilization_history: TimeSeries::new(format!("{}_utilization", name)),
            owner_id: None,
            created_at: now,
            last_updated: now,
            metadata: HashMap::new(),
        }
    }

    /// Allocate a portion of this resource
    pub fn allocate(&mut self, amount: f64) -> RanResult<()> {
        if amount <= 0.0 {
            return Err(RanError::validation(
                "amount",
                "Allocation amount must be positive",
            ));
        }

        if amount > self.available_amount {
            return Err(RanError::resource_allocation(
                self.resource_type.unit().to_string(),
                format!(
                    "Insufficient resource: requested {}, available {}",
                    amount, self.available_amount
                ),
            ));
        }

        // Check constraints
        self.check_allocation_constraints(amount)?;

        self.allocated_amount += amount;
        self.available_amount -= amount;
        self.last_updated = Utc::now();

        // Update utilization history
        let utilization = self.utilization_percentage();
        self.utilization_history.add_point(TimeSeriesPoint::new(
            self.last_updated,
            utilization,
        ));

        // Update state
        if self.available_amount == 0.0 {
            self.state = AllocationState::Allocated;
        }

        Ok(())
    }

    /// Deallocate a portion of this resource
    pub fn deallocate(&mut self, amount: f64) -> RanResult<()> {
        if amount <= 0.0 {
            return Err(RanError::validation(
                "amount",
                "Deallocation amount must be positive",
            ));
        }

        if amount > self.allocated_amount {
            return Err(RanError::resource_allocation(
                self.resource_type.unit(),
                &format!(
                    "Cannot deallocate more than allocated: requested {}, allocated {}",
                    amount, self.allocated_amount
                ),
            ));
        }

        self.allocated_amount -= amount;
        self.available_amount += amount;
        self.last_updated = Utc::now();

        // Update utilization history
        let utilization = self.utilization_percentage();
        self.utilization_history.add_point(TimeSeriesPoint::new(
            self.last_updated,
            utilization,
        ));

        // Update state
        if self.allocated_amount == 0.0 {
            self.state = AllocationState::Available;
        }

        Ok(())
    }

    /// Get current utilization percentage
    pub fn utilization_percentage(&self) -> f64 {
        if self.total_capacity == 0.0 {
            return 0.0;
        }
        (self.allocated_amount / self.total_capacity) * 100.0
    }

    /// Check if resource is fully utilized
    pub fn is_fully_utilized(&self) -> bool {
        self.available_amount == 0.0
    }

    /// Check if resource is overutilized (over 100%)
    pub fn is_overutilized(&self) -> bool {
        self.allocated_amount > self.total_capacity
    }

    /// Add a constraint to this resource
    pub fn add_constraint(&mut self, constraint: ResourceConstraint) {
        self.constraints.push(constraint);
        self.last_updated = Utc::now();
    }

    /// Check if allocation would violate constraints
    fn check_allocation_constraints(&self, amount: f64) -> RanResult<()> {
        let new_allocated = self.allocated_amount + amount;
        
        for constraint in &self.constraints {
            match constraint {
                ResourceConstraint::MaximumCapacity { value } => {
                    if new_allocated > *value {
                        return Err(RanError::resource_allocation(
                            self.resource_type.unit(),
                            &format!("Would exceed maximum capacity: {} > {}", new_allocated, value),
                        ));
                    }
                }
                ResourceConstraint::MinimumRequired { value } => {
                    if self.total_capacity - new_allocated < *value {
                        return Err(RanError::resource_allocation(
                            self.resource_type.unit(),
                            &format!("Would leave insufficient minimum: {} < {}", 
                                    self.total_capacity - new_allocated, value),
                        ));
                    }
                }
                ResourceConstraint::TimeWindow { start, end } => {
                    let now = Utc::now();
                    if now < *start || now > *end {
                        return Err(RanError::resource_allocation(
                            self.resource_type.unit(),
                            "Allocation outside permitted time window",
                        ));
                    }
                }
                _ => {} // Other constraints would require additional context
            }
        }
        
        Ok(())
    }

    /// Reserve resource for future allocation
    pub fn reserve(&mut self, amount: f64) -> RanResult<Uuid> {
        if amount > self.available_amount {
            return Err(RanError::resource_allocation(
                self.resource_type.unit(),
                "Insufficient resource for reservation",
            ));
        }

        self.available_amount -= amount;
        self.state = AllocationState::Reserved;
        self.last_updated = Utc::now();
        
        // Return reservation ID
        Ok(Uuid::new_v4())
    }

    /// Get efficiency metric (utilization vs. capacity)
    pub fn efficiency_score(&self) -> f64 {
        let utilization = self.utilization_percentage();
        
        // Efficiency peaks around 80-90% utilization
        // Too low = wasted resources, too high = potential bottleneck
        if utilization <= 80.0 {
            utilization / 80.0 * 100.0
        } else if utilization <= 90.0 {
            100.0
        } else {
            let penalty = (utilization - 90.0) * 2.0; // 2% penalty per 1% over 90%
            (100.0 - penalty).max(0.0)
        }
    }
}

/// Spectrum-specific resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumResource {
    /// Base resource
    pub base: Resource,
    /// Center frequency in MHz
    pub center_frequency: f64,
    /// Bandwidth in MHz
    pub bandwidth: f64,
    /// Power spectral density limit in dBm/Hz
    pub psd_limit: f64,
    /// Interference level in dBm
    pub interference_level: f64,
    /// Band designation (e.g., "n78", "n77")
    pub band: String,
    /// Regulatory domain
    pub regulatory_domain: String,
}

impl SpectrumResource {
    /// Create a new spectrum resource
    pub fn new(
        name: String,
        center_frequency: f64,
        bandwidth: f64,
        band: String,
    ) -> Self {
        Self {
            base: Resource::new(ResourceType::Spectrum, name, bandwidth),
            center_frequency,
            bandwidth,
            psd_limit: -41.3, // Typical limit in dBm/Hz
            interference_level: -120.0, // Typical noise floor
            band,
            regulatory_domain: "FCC".to_string(), // Default
        }
    }

    /// Calculate Signal-to-Interference-plus-Noise Ratio (SINR)
    pub fn calculate_sinr(&self, signal_power: f64) -> f64 {
        let noise_power = self.interference_level;
        signal_power - noise_power
    }

    /// Check if frequency allocation would cause interference
    pub fn check_interference(&self, other_frequency: f64, other_power: f64) -> bool {
        let frequency_separation = (self.center_frequency - other_frequency).abs();
        let required_separation = self.bandwidth / 2.0 + 10.0; // Guard band
        
        if frequency_separation < required_separation {
            // Calculate potential interference
            let path_loss = 120.0 + 40.0 * (frequency_separation / 1000.0).log10();
            let received_power = other_power - path_loss;
            received_power > self.interference_level + 10.0 // 10 dB protection
        } else {
            false
        }
    }
}

/// Power-specific resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerResource {
    /// Base resource
    pub base: Resource,
    /// Maximum transmit power in dBm
    pub max_tx_power: f64,
    /// Current transmit power in dBm
    pub current_tx_power: f64,
    /// Power amplifier efficiency percentage
    pub pa_efficiency: f64,
    /// Thermal limit in degrees Celsius
    pub thermal_limit: f64,
    /// Current temperature in degrees Celsius
    pub current_temperature: f64,
}

impl PowerResource {
    /// Create a new power resource
    pub fn new(name: String, max_tx_power: f64) -> Self {
        Self {
            base: Resource::new(ResourceType::Power, name, max_tx_power),
            max_tx_power,
            current_tx_power: 0.0,
            pa_efficiency: 35.0, // Typical PA efficiency
            thermal_limit: 85.0, // Typical thermal limit
            current_temperature: 25.0, // Room temperature
        }
    }

    /// Calculate power consumption
    pub fn calculate_power_consumption(&self) -> f64 {
        if self.current_tx_power <= 0.0 {
            return 0.0;
        }
        
        let linear_power = 10f64.powf(self.current_tx_power / 10.0); // Convert dBm to mW
        linear_power * (100.0 / self.pa_efficiency) // Account for PA efficiency
    }

    /// Check thermal constraints
    pub fn check_thermal_limits(&self, additional_power: f64) -> RanResult<()> {
        let total_power = self.current_tx_power + additional_power;
        let power_consumption = 10f64.powf(total_power / 10.0) * (100.0 / self.pa_efficiency);
        
        // Simplified thermal model: 1W = 10°C increase
        let temp_increase = power_consumption / 1000.0 * 10.0;
        let projected_temp = self.current_temperature + temp_increase;
        
        if projected_temp > self.thermal_limit {
            return Err(RanError::resource_allocation(
                "Power",
                &format!("Would exceed thermal limit: {}°C > {}°C", 
                        projected_temp, self.thermal_limit),
            ));
        }
        
        Ok(())
    }

    /// Optimize power allocation for efficiency
    pub fn optimize_for_efficiency(&mut self, required_coverage: f64) -> RanResult<()> {
        // Simplified power optimization
        // In practice, this would consider propagation models, interference, etc.
        
        let optimal_power = (required_coverage / 100.0) * self.max_tx_power;
        let clamped_power = optimal_power.min(self.max_tx_power).max(0.0);
        
        self.check_thermal_limits(clamped_power - self.current_tx_power)?;
        
        self.current_tx_power = clamped_power;
        self.base.allocated_amount = clamped_power;
        self.base.available_amount = self.max_tx_power - clamped_power;
        self.base.last_updated = Utc::now();
        
        Ok(())
    }
}

/// Resource pool for managing multiple resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    /// Pool identifier
    pub id: Uuid,
    /// Pool name
    pub name: String,
    /// Resources in the pool
    pub resources: HashMap<Uuid, Resource>,
    /// Pool policies
    pub policies: Vec<AllocationPolicy>,
    /// Pool statistics
    pub statistics: PoolStatistics,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl ResourcePool {
    /// Create a new resource pool
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            resources: HashMap::new(),
            policies: Vec::new(),
            statistics: PoolStatistics::default(),
            last_updated: Utc::now(),
        }
    }

    /// Add a resource to the pool
    pub fn add_resource(&mut self, resource: Resource) {
        self.resources.insert(resource.id, resource);
        self.update_statistics();
        self.last_updated = Utc::now();
    }

    /// Remove a resource from the pool
    pub fn remove_resource(&mut self, resource_id: Uuid) -> Option<Resource> {
        let removed = self.resources.remove(&resource_id);
        if removed.is_some() {
            self.update_statistics();
            self.last_updated = Utc::now();
        }
        removed
    }

    /// Find resources by type
    pub fn find_resources_by_type(&self, resource_type: ResourceType) -> Vec<&Resource> {
        self.resources
            .values()
            .filter(|r| r.resource_type == resource_type)
            .collect()
    }

    /// Get total capacity for a resource type
    pub fn total_capacity(&self, resource_type: ResourceType) -> f64 {
        self.find_resources_by_type(resource_type)
            .iter()
            .map(|r| r.total_capacity)
            .sum()
    }

    /// Get available capacity for a resource type
    pub fn available_capacity(&self, resource_type: ResourceType) -> f64 {
        self.find_resources_by_type(resource_type)
            .iter()
            .map(|r| r.available_amount)
            .sum()
    }

    /// Get overall utilization percentage
    pub fn overall_utilization(&self) -> f64 {
        if self.resources.is_empty() {
            return 0.0;
        }
        
        let total_util: f64 = self.resources.values()
            .map(|r| r.utilization_percentage())
            .sum();
        total_util / self.resources.len() as f64
    }

    /// Allocate resources optimally
    pub fn allocate_optimally(
        &mut self,
        resource_type: ResourceType,
        amount: f64,
        strategy: AllocationStrategy,
    ) -> RanResult<Vec<(Uuid, f64)>> {
        let mut candidates: Vec<&mut Resource> = self.resources
            .values_mut()
            .filter(|r| r.resource_type == resource_type && r.available_amount > 0.0)
            .collect();

        if candidates.is_empty() {
            return Err(RanError::resource_allocation(
                resource_type.unit(),
                "No available resources of requested type",
            ));
        }

        // Sort candidates based on strategy
        match strategy {
            AllocationStrategy::FirstFit => {
                // Use first available resource
            }
            AllocationStrategy::BestFit => {
                candidates.sort_by(|a, b| {
                    a.available_amount.partial_cmp(&b.available_amount).unwrap()
                });
            }
            AllocationStrategy::WorstFit => {
                candidates.sort_by(|a, b| {
                    b.available_amount.partial_cmp(&a.available_amount).unwrap()
                });
            }
            AllocationStrategy::LeastUtilized => {
                candidates.sort_by(|a, b| {
                    a.utilization_percentage().partial_cmp(&b.utilization_percentage()).unwrap()
                });
            }
            AllocationStrategy::MostEfficient => {
                candidates.sort_by(|a, b| {
                    b.efficiency_score().partial_cmp(&a.efficiency_score()).unwrap()
                });
            }
        }

        let mut allocations = Vec::new();
        let mut remaining = amount;

        for resource in candidates {
            if remaining <= 0.0 {
                break;
            }

            let to_allocate = remaining.min(resource.available_amount);
            resource.allocate(to_allocate)?;
            allocations.push((resource.id, to_allocate));
            remaining -= to_allocate;
        }

        if remaining > 0.0 {
            // Rollback allocations
            for (resource_id, amount) in &allocations {
                if let Some(resource) = self.resources.get_mut(resource_id) {
                    let _ = resource.deallocate(*amount);
                }
            }
            return Err(RanError::resource_allocation(
                resource_type.unit(),
                &format!("Insufficient total capacity: {} remaining", remaining),
            ));
        }

        self.update_statistics();
        Ok(allocations)
    }

    /// Update pool statistics
    fn update_statistics(&mut self) {
        let total_resources = self.resources.len();
        let active_resources = self.resources.values()
            .filter(|r| r.state == AllocationState::Allocated)
            .count();
        
        let avg_utilization = self.overall_utilization();
        
        let total_capacity: f64 = self.resources.values()
            .map(|r| r.total_capacity)
            .sum();
        
        let total_allocated: f64 = self.resources.values()
            .map(|r| r.allocated_amount)
            .sum();

        self.statistics = PoolStatistics {
            total_resources,
            active_resources,
            average_utilization: avg_utilization,
            total_capacity,
            total_allocated,
            efficiency_score: self.calculate_efficiency_score(),
        };
    }

    /// Calculate pool efficiency score
    fn calculate_efficiency_score(&self) -> f64 {
        if self.resources.is_empty() {
            return 0.0;
        }
        
        let scores: f64 = self.resources.values()
            .map(|r| r.efficiency_score())
            .sum();
        scores / self.resources.len() as f64
    }
}

/// Pool statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoolStatistics {
    /// Total number of resources
    pub total_resources: usize,
    /// Number of active resources
    pub active_resources: usize,
    /// Average utilization percentage
    pub average_utilization: f64,
    /// Total capacity across all resources
    pub total_capacity: f64,
    /// Total allocated amount
    pub total_allocated: f64,
    /// Overall efficiency score
    pub efficiency_score: f64,
}

/// Resource allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Allocate to first available resource
    FirstFit,
    /// Allocate to resource with smallest sufficient capacity
    BestFit,
    /// Allocate to resource with largest available capacity
    WorstFit,
    /// Allocate to least utilized resource
    LeastUtilized,
    /// Allocate to most efficient resource
    MostEfficient,
}

/// Resource allocation policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationPolicy {
    /// Maximum utilization threshold
    MaxUtilization { threshold: f64 },
    /// Minimum reservation requirement
    MinReservation { percentage: f64 },
    /// Fair sharing policy
    FairShare { equal_allocation: bool },
    /// Priority-based allocation
    PriorityBased { priority_levels: Vec<u8> },
    /// Time-based policies
    TimeBased { schedule: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_allocation() {
        let mut resource = Resource::new(
            ResourceType::Spectrum,
            "Test Spectrum".to_string(),
            100.0,
        );

        // Test successful allocation
        assert!(resource.allocate(30.0).is_ok());
        assert_eq!(resource.allocated_amount, 30.0);
        assert_eq!(resource.available_amount, 70.0);
        assert_eq!(resource.utilization_percentage(), 30.0);

        // Test over-allocation
        assert!(resource.allocate(80.0).is_err());

        // Test deallocation
        assert!(resource.deallocate(10.0).is_ok());
        assert_eq!(resource.allocated_amount, 20.0);
        assert_eq!(resource.available_amount, 80.0);
    }

    #[test]
    fn test_spectrum_resource() {
        let spectrum = SpectrumResource::new(
            "5G NR Band".to_string(),
            3500.0, // 3.5 GHz
            100.0,  // 100 MHz
            "n78".to_string(),
        );

        assert_eq!(spectrum.center_frequency, 3500.0);
        assert_eq!(spectrum.bandwidth, 100.0);
        assert!(!spectrum.check_interference(3520.0, 30.0)); // Nearby frequency, low power
    }

    #[test]
    fn test_power_resource() {
        let mut power = PowerResource::new("Cell Power".to_string(), 46.0);
        
        assert!(power.optimize_for_efficiency(80.0).is_ok());
        assert!(power.current_tx_power > 0.0);
        assert!(power.calculate_power_consumption() > 0.0);
    }

    #[test]
    fn test_resource_pool() {
        let mut pool = ResourcePool::new("Test Pool".to_string());
        
        let resource1 = Resource::new(ResourceType::Spectrum, "Spectrum 1".to_string(), 50.0);
        let resource2 = Resource::new(ResourceType::Spectrum, "Spectrum 2".to_string(), 75.0);
        
        pool.add_resource(resource1);
        pool.add_resource(resource2);

        assert_eq!(pool.total_capacity(ResourceType::Spectrum), 125.0);
        assert_eq!(pool.available_capacity(ResourceType::Spectrum), 125.0);

        // Test optimal allocation
        let allocations = pool.allocate_optimally(
            ResourceType::Spectrum,
            60.0,
            AllocationStrategy::BestFit,
        );
        
        assert!(allocations.is_ok());
        let allocs = allocations.unwrap();
        assert!(!allocs.is_empty());
    }

    #[test]
    fn test_efficiency_score() {
        let mut resource = Resource::new(
            ResourceType::Compute,
            "CPU".to_string(),
            100.0,
        );

        // Test efficiency at different utilization levels
        resource.allocate(80.0).unwrap(); // 80% utilization
        let score_80 = resource.efficiency_score();
        
        resource.allocate(15.0).unwrap(); // 95% utilization
        let score_95 = resource.efficiency_score();
        
        // Efficiency should decrease when overutilized
        assert!(score_80 > score_95);
    }
}