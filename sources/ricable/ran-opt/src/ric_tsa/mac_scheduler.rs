//! MAC Scheduler for QoE-aware Resource Allocation
//! 
//! This module implements intelligent MAC scheduling algorithms that optimize
//! resource allocation based on QoE predictions and user classifications.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::pfs_core::{NeuralNetwork, Layer, Activation, Tensor, TensorOps, DenseLayer};
use super::{UEContext, UserGroup, ServiceType, SteeringFeedback, QoEMetrics, ResourceAllocation, qoe_prediction::QoEPredictionResult};

/// MAC scheduling decision with resource allocation details
#[derive(Debug, Clone)]
pub struct MacSchedulingDecision {
    pub ue_id: u64,
    pub resource_allocation: ResourceAllocation,
    pub scheduling_priority: u8,
    pub qos_parameters: QoSParameters,
    pub power_control: PowerControlParams,
    pub beam_forming: BeamFormingParams,
    pub carrier_aggregation: CarrierAggregationParams,
    pub predicted_performance: PredictedPerformance,
}

/// QoS parameters for MAC scheduling
#[derive(Debug, Clone)]
pub struct QoSParameters {
    pub guaranteed_bit_rate: f32,    // Mbps
    pub maximum_bit_rate: f32,       // Mbps
    pub packet_delay_budget: f32,    // ms
    pub packet_error_rate: f32,      // %
    pub priority_level: u8,          // 1-15
    pub resource_type: ResourceType,
}

/// Power control parameters
#[derive(Debug, Clone)]
pub struct PowerControlParams {
    pub transmit_power: f32,         // dBm
    pub power_control_mode: PowerControlMode,
    pub path_loss_compensation: f32, // dB
    pub interference_mitigation: f32, // dB
}

/// Beam forming parameters
#[derive(Debug, Clone)]
pub struct BeamFormingParams {
    pub beam_direction: (f32, f32),  // azimuth, elevation
    pub beam_width: f32,             // degrees
    pub beam_gain: f32,              // dB
    pub mimo_mode: MimoMode,
}

/// Carrier aggregation parameters
#[derive(Debug, Clone)]
pub struct CarrierAggregationParams {
    pub primary_carrier: u32,
    pub secondary_carriers: Vec<u32>,
    pub aggregation_type: AggregationType,
    pub load_balancing: LoadBalancingStrategy,
}

/// Predicted performance metrics
#[derive(Debug, Clone)]
pub struct PredictedPerformance {
    pub expected_throughput: f32,    // Mbps
    pub expected_latency: f32,       // ms
    pub expected_reliability: f32,   // %
    pub resource_efficiency: f32,    // %
    pub energy_efficiency: f32,      // bits/J
}

/// Resource allocation types
#[derive(Debug, Clone)]
pub enum ResourceType {
    GuaranteedBitRate,
    NonGuaranteedBitRate,
    DelayTolerant,
    LowLatency,
    HighReliability,
    MassiveMTC,
}

/// Power control modes
#[derive(Debug, Clone)]
pub enum PowerControlMode {
    OpenLoop,
    ClosedLoop,
    Adaptive,
    EnergyOptimized,
}

/// MIMO operation modes
#[derive(Debug, Clone)]
pub enum MimoMode {
    SingleStream,
    SpatialMultiplexing,
    TransmitDiversity,
    BeamForming,
    MassiveMimo,
}

/// Carrier aggregation types
#[derive(Debug, Clone)]
pub enum AggregationType {
    IntraBAND,
    InterBAND,
    InterBANDNonContiguous,
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    ProportionalFair,
    QoEAware,
    EnergyEfficient,
}

/// MAC scheduler with neural network-based optimization
pub struct MacScheduler {
    // Resource allocation network
    allocation_network: Arc<RwLock<NeuralNetwork>>,
    
    // QoS parameter prediction network
    qos_network: Arc<RwLock<NeuralNetwork>>,
    
    // Power control optimization network
    power_network: Arc<RwLock<NeuralNetwork>>,
    
    // Beam forming optimization network
    beam_network: Arc<RwLock<NeuralNetwork>>,
    
    // Carrier aggregation network
    ca_network: Arc<RwLock<NeuralNetwork>>,
    
    // Interference prediction network
    interference_network: Arc<RwLock<NeuralNetwork>>,
    
    // Resource utilization tracker
    resource_tracker: Arc<RwLock<ResourceTracker>>,
    
    // Configuration
    config: MacSchedulerConfig,
}

/// Configuration for MAC scheduler
#[derive(Debug, Clone)]
pub struct MacSchedulerConfig {
    pub max_prbs: u16,              // Maximum PRBs per cell
    pub max_users_per_tti: u8,      // Maximum users per TTI
    pub scheduling_algorithm: SchedulingAlgorithm,
    pub power_control_enabled: bool,
    pub beam_forming_enabled: bool,
    pub carrier_aggregation_enabled: bool,
    pub interference_coordination: bool,
    pub qoe_weight: f32,
    pub fairness_weight: f32,
    pub efficiency_weight: f32,
}

/// Scheduling algorithms
#[derive(Debug, Clone)]
pub enum SchedulingAlgorithm {
    RoundRobin,
    ProportionalFair,
    MaximumThroughput,
    QoEAware,
    MLOptimized,
}

/// Resource utilization tracker
#[derive(Debug, Clone)]
pub struct ResourceTracker {
    pub prb_utilization: HashMap<u32, Vec<bool>>, // cell_id -> PRB usage
    pub power_utilization: HashMap<u32, f32>,     // cell_id -> power usage
    pub user_allocations: HashMap<u64, ResourceAllocation>, // ue_id -> allocation
    pub interference_matrix: HashMap<(u32, u32), f32>, // (cell1, cell2) -> interference
}

impl Default for MacSchedulerConfig {
    fn default() -> Self {
        Self {
            max_prbs: 100,
            max_users_per_tti: 10,
            scheduling_algorithm: SchedulingAlgorithm::QoEAware,
            power_control_enabled: true,
            beam_forming_enabled: true,
            carrier_aggregation_enabled: true,
            interference_coordination: true,
            qoe_weight: 0.4,
            fairness_weight: 0.3,
            efficiency_weight: 0.3,
        }
    }
}

impl MacScheduler {
    /// Create a new MAC scheduler
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = MacSchedulerConfig::default();
        Self::new_with_config(config)
    }

    /// Create MAC scheduler with custom configuration
    pub fn new_with_config(config: MacSchedulerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Create neural networks
        let allocation_network = Arc::new(RwLock::new(Self::build_allocation_network(&config)?));
        let qos_network = Arc::new(RwLock::new(Self::build_qos_network(&config)?));
        let power_network = Arc::new(RwLock::new(Self::build_power_network(&config)?));
        let beam_network = Arc::new(RwLock::new(Self::build_beam_network(&config)?));
        let ca_network = Arc::new(RwLock::new(Self::build_ca_network(&config)?));
        let interference_network = Arc::new(RwLock::new(Self::build_interference_network(&config)?));
        
        // Initialize resource tracker
        let resource_tracker = Arc::new(RwLock::new(ResourceTracker {
            prb_utilization: HashMap::new(),
            power_utilization: HashMap::new(),
            user_allocations: HashMap::new(),
            interference_matrix: HashMap::new(),
        }));

        Ok(Self {
            allocation_network,
            qos_network,
            power_network,
            beam_network,
            ca_network,
            interference_network,
            resource_tracker,
            config,
        })
    }

    /// Build resource allocation network
    fn build_allocation_network(config: &MacSchedulerConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        // Input: UE context + QoE prediction + cell state
        let input_size = 64; // Comprehensive feature set
        
        // Multi-layer network for resource allocation
        network.add_layer(Box::new(DenseLayer::new(input_size, 256)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(256, 128)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(128, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        
        // Output: PRB allocation + MCS + MIMO layers + power
        network.add_layer(Box::new(DenseLayer::new(64, 32)));
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Build QoS parameter network
    fn build_qos_network(config: &MacSchedulerConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        network.add_layer(Box::new(DenseLayer::new(32, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(64, 32)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(32, 16))); // QoS parameters
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Build power control network
    fn build_power_network(config: &MacSchedulerConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        network.add_layer(Box::new(DenseLayer::new(32, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(64, 32)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(32, 8))); // Power control parameters
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Build beam forming network
    fn build_beam_network(config: &MacSchedulerConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        network.add_layer(Box::new(DenseLayer::new(32, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(64, 32)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(32, 12))); // Beam forming parameters
        network.add_layer(Box::new(Activation::Tanh)); // For directional parameters
        
        Ok(network)
    }

    /// Build carrier aggregation network
    fn build_ca_network(config: &MacSchedulerConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        network.add_layer(Box::new(DenseLayer::new(32, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(64, 32)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(32, 16))); // CA parameters
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Build interference prediction network
    fn build_interference_network(config: &MacSchedulerConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        network.add_layer(Box::new(DenseLayer::new(32, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(64, 32)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(32, 8))); // Interference parameters
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Generate resource allocation for a user
    pub async fn generate_allocation(
        &self,
        ue_context: &UEContext,
        user_group: &UserGroup,
        qoe_prediction: &QoEPredictionResult,
    ) -> Result<ResourceAllocation, Box<dyn std::error::Error>> {
        // Extract scheduling features
        let features = self.extract_scheduling_features(ue_context, user_group, qoe_prediction).await?;
        
        // Get resource allocation from neural network
        let allocation_params = self.predict_allocation(&features).await?;
        
        // Generate QoS parameters
        let qos_params = self.predict_qos_parameters(&features).await?;
        
        // Generate power control parameters
        let power_params = self.predict_power_control(&features).await?;
        
        // Generate beam forming parameters
        let beam_params = self.predict_beam_forming(&features).await?;
        
        // Generate carrier aggregation parameters
        let ca_params = self.predict_carrier_aggregation(&features).await?;
        
        // Create resource allocation
        let allocation = self.create_resource_allocation(
            ue_context,
            &allocation_params,
            &qos_params,
            &power_params,
            &beam_params,
            &ca_params,
        ).await?;
        
        // Update resource tracker
        self.update_resource_tracker(ue_context.ue_id, &allocation).await?;
        
        Ok(allocation)
    }

    /// Extract features for scheduling decision
    async fn extract_scheduling_features(
        &self,
        ue_context: &UEContext,
        user_group: &UserGroup,
        qoe_prediction: &QoEPredictionResult,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut features = Vec::new();
        
        // UE context features
        features.extend_from_slice(&[
            ue_context.current_qoe.throughput / 100.0,      // Normalize to 0-1
            ue_context.current_qoe.latency / 200.0,         // Normalize to 0-1
            ue_context.current_qoe.jitter / 50.0,           // Normalize to 0-1
            ue_context.current_qoe.packet_loss / 10.0,      // Normalize to 0-1
            ue_context.current_qoe.reliability / 100.0,     // Normalize to 0-1
            ue_context.current_qoe.availability / 100.0,    // Normalize to 0-1
        ]);
        
        // Service type encoding
        let service_encoding = match ue_context.service_type {
            ServiceType::VideoStreaming => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ServiceType::VoiceCall => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ServiceType::Gaming => [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ServiceType::FileTransfer => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ServiceType::WebBrowsing => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ServiceType::IoTSensor => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ServiceType::Emergency => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ServiceType::AR_VR => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        };
        features.extend_from_slice(&service_encoding);
        
        // User group encoding
        let group_encoding = match user_group {
            UserGroup::Premium => [1.0, 0.0, 0.0, 0.0, 0.0],
            UserGroup::Standard => [0.0, 1.0, 0.0, 0.0, 0.0],
            UserGroup::Basic => [0.0, 0.0, 1.0, 0.0, 0.0],
            UserGroup::IoT => [0.0, 0.0, 0.0, 1.0, 0.0],
            UserGroup::Emergency => [0.0, 0.0, 0.0, 0.0, 1.0],
        };
        features.extend_from_slice(&group_encoding);
        
        // QoE prediction features
        features.extend_from_slice(&[
            qoe_prediction.predicted_qoe.throughput / 100.0,
            qoe_prediction.predicted_qoe.latency / 200.0,
            qoe_prediction.predicted_qoe.jitter / 50.0,
            qoe_prediction.predicted_qoe.packet_loss / 10.0,
            qoe_prediction.confidence,
            qoe_prediction.uncertainty,
        ]);
        
        // Device capabilities
        features.extend_from_slice(&[
            ue_context.device_capabilities.max_mimo_layers as f32 / 8.0,
            if ue_context.device_capabilities.ca_support { 1.0 } else { 0.0 },
            if ue_context.device_capabilities.dual_connectivity { 1.0 } else { 0.0 },
        ]);
        
        // Service requirements
        features.extend_from_slice(&[
            ue_context.service_requirements.min_throughput / 100.0,
            ue_context.service_requirements.max_latency / 200.0,
            ue_context.service_requirements.max_jitter / 50.0,
            ue_context.service_requirements.max_packet_loss / 10.0,
            ue_context.service_requirements.priority as f32 / 255.0,
        ]);
        
        // Mobility pattern
        let mobility_encoding = match ue_context.mobility_pattern {
            super::MobilityPattern::Stationary => [1.0, 0.0, 0.0, 0.0],
            super::MobilityPattern::Pedestrian => [0.0, 1.0, 0.0, 0.0],
            super::MobilityPattern::Vehicular => [0.0, 0.0, 1.0, 0.0],
            super::MobilityPattern::HighSpeed => [0.0, 0.0, 0.0, 1.0],
        };
        features.extend_from_slice(&mobility_encoding);
        
        // Current resource utilization
        let tracker = self.resource_tracker.read().await;
        let avg_utilization = tracker.prb_utilization.values()
            .map(|prbs| prbs.iter().filter(|&&used| used).count() as f32 / prbs.len() as f32)
            .sum::<f32>() / tracker.prb_utilization.len().max(1) as f32;
        features.push(avg_utilization);
        
        // Interference level (simplified)
        let avg_interference = tracker.interference_matrix.values()
            .sum::<f32>() / tracker.interference_matrix.len().max(1) as f32;
        features.push(avg_interference);
        
        // Pad to feature size
        features.resize(64, 0.0);
        
        Ok(features)
    }

    /// Predict resource allocation parameters
    async fn predict_allocation(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.allocation_network.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Predict QoS parameters
    async fn predict_qos_parameters(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.qos_network.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Predict power control parameters
    async fn predict_power_control(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.power_network.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Predict beam forming parameters
    async fn predict_beam_forming(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.beam_network.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Predict carrier aggregation parameters
    async fn predict_carrier_aggregation(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.ca_network.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Create resource allocation from neural network outputs
    async fn create_resource_allocation(
        &self,
        ue_context: &UEContext,
        allocation_params: &[f32],
        qos_params: &[f32],
        power_params: &[f32],
        beam_params: &[f32],
        ca_params: &[f32],
    ) -> Result<ResourceAllocation, Box<dyn std::error::Error>> {
        // Extract PRB allocation
        let num_prbs = (allocation_params.get(0).unwrap_or(&0.2) * self.config.max_prbs as f32) as usize;
        let prb_allocation = self.allocate_prbs(num_prbs, ue_context.ue_id).await?;
        
        // Extract MCS index
        let mcs_index = (allocation_params.get(1).unwrap_or(&0.5) * 31.0) as u8; // MCS 0-31
        
        // Extract MIMO layers
        let mimo_layers = ((allocation_params.get(2).unwrap_or(&0.5) * 
            ue_context.device_capabilities.max_mimo_layers as f32) as u8)
            .min(ue_context.device_capabilities.max_mimo_layers);
        
        // Extract power level
        let power_level = power_params.get(0).unwrap_or(&0.5) * 46.0 - 20.0; // -20 to 26 dBm
        
        // Extract scheduling priority
        let priority = (qos_params.get(0).unwrap_or(&0.5) * 255.0) as u8;
        
        Ok(ResourceAllocation {
            prb_allocation,
            mcs_index,
            mimo_layers,
            power_level,
            scheduling_priority: priority,
        })
    }

    /// Allocate PRBs for a user
    async fn allocate_prbs(&self, num_prbs: usize, ue_id: u64) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
        let mut tracker = self.resource_tracker.write().await;
        let mut allocated_prbs = Vec::new();
        
        // Simple allocation strategy - find available PRBs
        for cell_id in 1..=3 { // Assume 3 cells for now
            let cell_prbs = tracker.prb_utilization.entry(cell_id).or_insert_with(|| {
                vec![false; self.config.max_prbs as usize]
            });
            
            let mut allocated_in_cell = 0;
            for (prb_idx, &used) in cell_prbs.iter().enumerate() {
                if !used && allocated_in_cell < num_prbs {
                    allocated_prbs.push(prb_idx as u16);
                    cell_prbs[prb_idx] = true;
                    allocated_in_cell += 1;
                }
            }
            
            if allocated_prbs.len() >= num_prbs {
                break;
            }
        }
        
        Ok(allocated_prbs)
    }

    /// Update resource tracker with new allocation
    async fn update_resource_tracker(&self, ue_id: u64, allocation: &ResourceAllocation) -> Result<(), Box<dyn std::error::Error>> {
        let mut tracker = self.resource_tracker.write().await;
        tracker.user_allocations.insert(ue_id, allocation.clone());
        Ok(())
    }

    /// Schedule resources for multiple users
    pub async fn schedule_batch_users(
        &self,
        users: &[(UEContext, UserGroup, QoEPredictionResult)],
    ) -> Result<Vec<MacSchedulingDecision>, Box<dyn std::error::Error>> {
        let mut decisions = Vec::new();
        
        // Sort users by priority
        let mut sorted_users = users.to_vec();
        sorted_users.sort_by(|a, b| {
            let priority_a = self.get_user_priority(&a.0, &a.1);
            let priority_b = self.get_user_priority(&b.0, &b.1);
            priority_b.partial_cmp(&priority_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Schedule each user
        for (ue_context, user_group, qoe_prediction) in sorted_users {
            let allocation = self.generate_allocation(&ue_context, &user_group, &qoe_prediction).await?;
            
            let decision = MacSchedulingDecision {
                ue_id: ue_context.ue_id,
                resource_allocation: allocation,
                scheduling_priority: self.get_scheduling_priority(&ue_context, &user_group),
                qos_parameters: self.get_qos_parameters(&ue_context, &user_group),
                power_control: self.get_power_control_params(&ue_context),
                beam_forming: self.get_beam_forming_params(&ue_context),
                carrier_aggregation: self.get_ca_params(&ue_context),
                predicted_performance: self.predict_performance(&ue_context, &qoe_prediction),
            };
            
            decisions.push(decision);
        }
        
        Ok(decisions)
    }

    /// Get user priority for scheduling
    fn get_user_priority(&self, ue_context: &UEContext, user_group: &UserGroup) -> f32 {
        let base_priority = match user_group {
            UserGroup::Emergency => 1.0,
            UserGroup::Premium => 0.9,
            UserGroup::Standard => 0.7,
            UserGroup::Basic => 0.5,
            UserGroup::IoT => 0.3,
        };
        
        let service_priority = match ue_context.service_type {
            ServiceType::Emergency => 1.0,
            ServiceType::AR_VR => 0.9,
            ServiceType::Gaming => 0.8,
            ServiceType::VoiceCall => 0.8,
            ServiceType::VideoStreaming => 0.7,
            ServiceType::WebBrowsing => 0.5,
            ServiceType::FileTransfer => 0.4,
            ServiceType::IoTSensor => 0.3,
        };
        
        (base_priority + service_priority) / 2.0
    }

    /// Get scheduling priority
    fn get_scheduling_priority(&self, ue_context: &UEContext, user_group: &UserGroup) -> u8 {
        let priority = (self.get_user_priority(ue_context, user_group) * 255.0) as u8;
        priority.max(1) // Ensure non-zero priority
    }

    /// Get QoS parameters
    fn get_qos_parameters(&self, ue_context: &UEContext, user_group: &UserGroup) -> QoSParameters {
        let (gbr, mbr) = match ue_context.service_type {
            ServiceType::VideoStreaming => (5.0, 25.0),
            ServiceType::VoiceCall => (0.064, 0.064),
            ServiceType::Gaming => (1.0, 10.0),
            ServiceType::FileTransfer => (0.0, 50.0),
            ServiceType::WebBrowsing => (0.0, 10.0),
            ServiceType::IoTSensor => (0.001, 0.1),
            ServiceType::Emergency => (1.0, 20.0),
            ServiceType::AR_VR => (20.0, 100.0),
        };
        
        let multiplier = match user_group {
            UserGroup::Premium => 1.5,
            UserGroup::Standard => 1.0,
            UserGroup::Basic => 0.7,
            UserGroup::IoT => 0.5,
            UserGroup::Emergency => 2.0,
        };
        
        QoSParameters {
            guaranteed_bit_rate: gbr * multiplier,
            maximum_bit_rate: mbr * multiplier,
            packet_delay_budget: ue_context.service_requirements.max_latency,
            packet_error_rate: ue_context.service_requirements.max_packet_loss,
            priority_level: ((self.get_user_priority(ue_context, user_group) * 15.0) as u8).max(1),
            resource_type: match ue_context.service_type {
                ServiceType::VideoStreaming | ServiceType::VoiceCall => ResourceType::GuaranteedBitRate,
                ServiceType::Gaming | ServiceType::AR_VR => ResourceType::LowLatency,
                ServiceType::Emergency => ResourceType::HighReliability,
                ServiceType::IoTSensor => ResourceType::MassiveMTC,
                _ => ResourceType::NonGuaranteedBitRate,
            },
        }
    }

    /// Get power control parameters
    fn get_power_control_params(&self, ue_context: &UEContext) -> PowerControlParams {
        PowerControlParams {
            transmit_power: 20.0, // Default 20 dBm
            power_control_mode: PowerControlMode::Adaptive,
            path_loss_compensation: 0.8,
            interference_mitigation: 0.5,
        }
    }

    /// Get beam forming parameters
    fn get_beam_forming_params(&self, ue_context: &UEContext) -> BeamFormingParams {
        BeamFormingParams {
            beam_direction: (0.0, 0.0), // Placeholder
            beam_width: 30.0,
            beam_gain: 6.0,
            mimo_mode: if ue_context.device_capabilities.max_mimo_layers >= 4 {
                MimoMode::SpatialMultiplexing
            } else {
                MimoMode::TransmitDiversity
            },
        }
    }

    /// Get carrier aggregation parameters
    fn get_ca_params(&self, ue_context: &UEContext) -> CarrierAggregationParams {
        CarrierAggregationParams {
            primary_carrier: 1,
            secondary_carriers: if ue_context.device_capabilities.ca_support {
                vec![2, 3]
            } else {
                vec![]
            },
            aggregation_type: AggregationType::IntraBAND,
            load_balancing: LoadBalancingStrategy::QoEAware,
        }
    }

    /// Predict performance metrics
    fn predict_performance(&self, ue_context: &UEContext, qoe_prediction: &QoEPredictionResult) -> PredictedPerformance {
        PredictedPerformance {
            expected_throughput: qoe_prediction.predicted_qoe.throughput,
            expected_latency: qoe_prediction.predicted_qoe.latency,
            expected_reliability: qoe_prediction.predicted_qoe.reliability,
            resource_efficiency: 0.8, // Placeholder
            energy_efficiency: 100.0, // Placeholder bits/J
        }
    }

    /// Update models with feedback
    pub async fn update_with_feedback(&self, feedback: &[SteeringFeedback]) -> Result<(), Box<dyn std::error::Error>> {
        // Update resource utilization statistics
        for fb in feedback {
            // This would typically involve retraining the neural networks
            // with the actual performance data
            println!("Updating MAC scheduler with feedback for UE {}", fb.ue_id);
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ric_tsa::{DeviceCapabilities, ServiceRequirements, FrequencyBand, MobilityPattern};

    #[tokio::test]
    async fn test_mac_scheduler_creation() {
        let scheduler = MacScheduler::new();
        assert!(scheduler.is_ok());
    }

    #[tokio::test]
    async fn test_resource_allocation() {
        let scheduler = MacScheduler::new().unwrap();
        
        let ue_context = UEContext {
            ue_id: 1,
            user_group: UserGroup::Premium,
            service_type: ServiceType::VideoStreaming,
            current_qoe: QoEMetrics {
                throughput: 15.0,
                latency: 25.0,
                jitter: 8.0,
                packet_loss: 0.2,
                video_quality: 3.8,
                audio_quality: 4.0,
                reliability: 98.5,
                availability: 99.8,
            },
            location: (40.7128, -74.0060),
            mobility_pattern: MobilityPattern::Pedestrian,
            device_capabilities: DeviceCapabilities {
                supported_bands: vec![FrequencyBand::Band1800MHz],
                max_mimo_layers: 4,
                ca_support: true,
                dual_connectivity: true,
            },
            service_requirements: ServiceRequirements {
                min_throughput: 10.0,
                max_latency: 40.0,
                max_jitter: 15.0,
                max_packet_loss: 1.0,
                priority: 200,
            },
        };

        let qoe_prediction = QoEPredictionResult {
            predicted_qoe: QoEMetrics {
                throughput: 20.0,
                latency: 20.0,
                jitter: 5.0,
                packet_loss: 0.1,
                video_quality: 4.2,
                audio_quality: 4.3,
                reliability: 99.2,
                availability: 99.9,
            },
            cell_scores: std::collections::HashMap::new(),
            confidence: 0.85,
            prediction_horizon: 30,
            uncertainty: 0.15,
        };

        let result = scheduler.generate_allocation(&ue_context, &UserGroup::Premium, &qoe_prediction).await;
        assert!(result.is_ok());
        
        let allocation = result.unwrap();
        assert!(!allocation.prb_allocation.is_empty());
        assert!(allocation.mcs_index <= 31);
        assert!(allocation.mimo_layers <= 4);
        assert!(allocation.scheduling_priority > 0);
    }
}