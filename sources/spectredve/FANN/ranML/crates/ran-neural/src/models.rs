//! Neural network models for RAN optimization tasks

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use ruv_fann::ActivationFunction;

use crate::{NetworkConfig, FeatureConfig, NeuralError, NeuralResult};

/// Types of neural network models for RAN optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    // Regression models
    /// Predict cell throughput based on network conditions
    ThroughputPredictor,
    /// Estimate end-to-end latency for different paths
    LatencyEstimator,
    /// Optimize transmission power for efficiency
    PowerOptimizer,
    /// Predict resource utilization
    ResourcePredictor,
    /// Predict QoS metrics
    QoSPredictor,
    
    // Classification models
    /// Classify cell operational states
    CellStateClassifier,
    /// Determine optimal handover targets
    HandoverDecision,
    /// Identify traffic patterns and anomalies
    TrafficPatternClassifier,
    /// Classify interference types and sources
    InterferenceClassifier,
    /// Classify network alarms by severity
    AlarmClassifier,
    
    // Load balancing and optimization
    /// Distribute traffic optimally across cells
    LoadBalancer,
    /// Optimize coverage and capacity
    CoverageOptimizer,
    /// Optimize spectrum allocation
    SpectrumOptimizer,
    /// Optimize energy consumption
    EnergyOptimizer,
    
    // Advanced models
    /// Reinforcement learning for resource allocation
    ResourceAllocator,
    /// Deep Q-Network for network control
    NetworkController,
    /// Policy gradient for adaptive scheduling
    AdaptiveScheduler,
    /// Anomaly detection in network behavior
    AnomalyDetector,
    
    // Multi-objective optimization
    /// Multi-objective optimization model
    MultiObjectiveOptimizer,
    /// Pareto frontier exploration
    ParetoOptimizer,
}

impl ModelType {
    /// Get the default network configuration for this model type
    pub fn default_config(&self) -> NetworkConfig {
        match self {
            ModelType::ThroughputPredictor => NetworkConfig {
                layers: vec![8, 16, 12, 8, 1], // More complex for throughput prediction
                activation: ActivationFunction::Tanh,
                feature_config: self.feature_config(),
                learning_rate: 0.001,
                training_algorithm: ruv_fann::TrainingAlgorithm::RProp,
                parameters: HashMap::new(),
            },
            
            ModelType::LatencyEstimator => NetworkConfig {
                layers: vec![6, 12, 8, 1],
                activation: ActivationFunction::Sigmoid,
                feature_config: self.feature_config(),
                learning_rate: 0.01,
                training_algorithm: ruv_fann::TrainingAlgorithm::QuickProp,
                parameters: HashMap::new(),
            },
            
            ModelType::PowerOptimizer => NetworkConfig {
                layers: vec![5, 10, 8, 3], // Multi-output for power settings
                activation: ActivationFunction::Linear,
                feature_config: self.feature_config(),
                learning_rate: 0.005,
                training_algorithm: ruv_fann::TrainingAlgorithm::RProp,
                parameters: HashMap::new(),
            },
            
            ModelType::CellStateClassifier => NetworkConfig {
                layers: vec![10, 20, 15, 5], // 5 cell states
                activation: ActivationFunction::Sigmoid,
                feature_config: self.feature_config(),
                learning_rate: 0.01,
                training_algorithm: ruv_fann::TrainingAlgorithm::BatchBackprop,
                parameters: HashMap::new(),
            },
            
            ModelType::HandoverDecision => NetworkConfig {
                layers: vec![12, 24, 16, 1], // Binary decision
                activation: ActivationFunction::Sigmoid,
                feature_config: self.feature_config(),
                learning_rate: 0.01,
                training_algorithm: ruv_fann::TrainingAlgorithm::RProp,
                parameters: HashMap::new(),
            },
            
            ModelType::TrafficPatternClassifier => NetworkConfig {
                layers: vec![15, 30, 20, 8], // 8 traffic patterns
                activation: ActivationFunction::Tanh,
                feature_config: self.feature_config(),
                learning_rate: 0.001,
                training_algorithm: ruv_fann::TrainingAlgorithm::RProp,
                parameters: HashMap::new(),
            },
            
            ModelType::LoadBalancer => NetworkConfig {
                layers: vec![20, 40, 30, 10], // Load distribution across cells
                activation: ActivationFunction::Sigmoid,
                feature_config: self.feature_config(),
                learning_rate: 0.001,
                training_algorithm: ruv_fann::TrainingAlgorithm::RProp,
                parameters: HashMap::new(),
            },
            
            ModelType::ResourceAllocator => NetworkConfig {
                layers: vec![25, 50, 40, 15], // Complex RL state space
                activation: ActivationFunction::Tanh,
                feature_config: self.feature_config(),
                learning_rate: 0.0001,
                training_algorithm: ruv_fann::TrainingAlgorithm::RProp,
                parameters: HashMap::new(),
            },
            
            ModelType::AnomalyDetector => NetworkConfig {
                layers: vec![30, 20, 10, 1], // Autoencoder-like structure
                activation: ActivationFunction::Tanh,
                feature_config: self.feature_config(),
                learning_rate: 0.001,
                training_algorithm: ruv_fann::TrainingAlgorithm::RProp,
                parameters: HashMap::new(),
            },
            
            _ => NetworkConfig::default(), // Default config for other types
        }
    }

    /// Get the feature configuration for this model type
    pub fn feature_config(&self) -> FeatureConfig {
        match self {
            ModelType::ThroughputPredictor => FeatureConfig {
                features: vec![
                    "cell_load".to_string(),
                    "tx_power".to_string(),
                    "sinr".to_string(),
                    "active_ues".to_string(),
                    "rb_utilization".to_string(),
                    "interference_level".to_string(),
                    "channel_quality".to_string(),
                    "time_of_day".to_string(),
                ],
                normalization: NormalizationType::StandardScore,
                missing_value_strategy: MissingValueStrategy::Mean,
                feature_scaling: true,
                feature_selection: false,
            },
            
            ModelType::LatencyEstimator => FeatureConfig {
                features: vec![
                    "distance".to_string(),
                    "load".to_string(),
                    "processing_delay".to_string(),
                    "queue_length".to_string(),
                    "network_congestion".to_string(),
                    "service_type".to_string(),
                ],
                normalization: NormalizationType::MinMax,
                missing_value_strategy: MissingValueStrategy::Zero,
                feature_scaling: true,
                feature_selection: false,
            },
            
            ModelType::PowerOptimizer => FeatureConfig {
                features: vec![
                    "coverage_target".to_string(),
                    "interference_level".to_string(),
                    "energy_efficiency".to_string(),
                    "thermal_state".to_string(),
                    "regulatory_limit".to_string(),
                ],
                normalization: NormalizationType::MinMax,
                missing_value_strategy: MissingValueStrategy::Interpolate,
                feature_scaling: true,
                feature_selection: true,
            },
            
            ModelType::CellStateClassifier => FeatureConfig {
                features: vec![
                    "throughput".to_string(),
                    "latency".to_string(),
                    "error_rate".to_string(),
                    "utilization".to_string(),
                    "temperature".to_string(),
                    "alarms_count".to_string(),
                    "handover_rate".to_string(),
                    "connectivity_issues".to_string(),
                    "power_consumption".to_string(),
                    "performance_trend".to_string(),
                ],
                normalization: NormalizationType::StandardScore,
                missing_value_strategy: MissingValueStrategy::Mean,
                feature_scaling: true,
                feature_selection: false,
            },
            
            ModelType::HandoverDecision => FeatureConfig {
                features: vec![
                    "source_rsrp".to_string(),
                    "target_rsrp".to_string(),
                    "source_sinr".to_string(),
                    "target_sinr".to_string(),
                    "source_load".to_string(),
                    "target_load".to_string(),
                    "ue_mobility".to_string(),
                    "hysteresis".to_string(),
                    "time_to_trigger".to_string(),
                    "frequency_band".to_string(),
                    "service_priority".to_string(),
                    "network_condition".to_string(),
                ],
                normalization: NormalizationType::StandardScore,
                missing_value_strategy: MissingValueStrategy::Mean,
                feature_scaling: true,
                feature_selection: false,
            },
            
            ModelType::TrafficPatternClassifier => FeatureConfig {
                features: vec![
                    "hourly_traffic".to_string(),
                    "day_of_week".to_string(),
                    "month".to_string(),
                    "location_type".to_string(),
                    "event_indicator".to_string(),
                    "weather_condition".to_string(),
                    "user_density".to_string(),
                    "service_mix".to_string(),
                    "mobility_pattern".to_string(),
                    "seasonal_factor".to_string(),
                    "holiday_indicator".to_string(),
                    "peak_hour_indicator".to_string(),
                    "weekend_indicator".to_string(),
                    "special_event".to_string(),
                    "traffic_trend".to_string(),
                ],
                normalization: NormalizationType::MinMax,
                missing_value_strategy: MissingValueStrategy::Mode,
                feature_scaling: true,
                feature_selection: true,
            },
            
            ModelType::LoadBalancer => FeatureConfig {
                features: vec![
                    "cell1_load".to_string(),
                    "cell2_load".to_string(),
                    "cell3_load".to_string(),
                    "cell4_load".to_string(),
                    "cell5_load".to_string(),
                    "total_demand".to_string(),
                    "priority_traffic".to_string(),
                    "qos_requirements".to_string(),
                    "cell_capacities".to_string(),
                    "interference_matrix".to_string(),
                    "mobility_patterns".to_string(),
                    "service_areas".to_string(),
                    "network_topology".to_string(),
                    "optimization_objective".to_string(),
                    "constraints".to_string(),
                    "current_time".to_string(),
                    "predicted_demand".to_string(),
                    "resource_availability".to_string(),
                    "energy_costs".to_string(),
                    "sla_requirements".to_string(),
                ],
                normalization: NormalizationType::StandardScore,
                missing_value_strategy: MissingValueStrategy::Mean,
                feature_scaling: true,
                feature_selection: true,
            },
            
            ModelType::AnomalyDetector => FeatureConfig {
                features: vec![
                    "throughput_deviation".to_string(),
                    "latency_spike".to_string(),
                    "error_rate_increase".to_string(),
                    "unusual_traffic_pattern".to_string(),
                    "resource_exhaustion".to_string(),
                    "performance_degradation".to_string(),
                    "connectivity_issues".to_string(),
                    "interference_anomaly".to_string(),
                    "power_consumption_anomaly".to_string(),
                    "temperature_anomaly".to_string(),
                    "alarm_frequency".to_string(),
                    "user_complaint_rate".to_string(),
                    "handover_failure_rate".to_string(),
                    "call_drop_rate".to_string(),
                    "service_unavailability".to_string(),
                    "configuration_drift".to_string(),
                    "security_indicators".to_string(),
                    "hardware_health".to_string(),
                    "software_health".to_string(),
                    "network_congestion".to_string(),
                    "seasonal_baseline_deviation".to_string(),
                    "peer_comparison_score".to_string(),
                    "historical_trend_deviation".to_string(),
                    "correlation_anomalies".to_string(),
                    "statistical_outliers".to_string(),
                    "behavioral_changes".to_string(),
                    "pattern_breaks".to_string(),
                    "cascade_indicators".to_string(),
                    "early_warning_signals".to_string(),
                    "composite_health_score".to_string(),
                ],
                normalization: NormalizationType::RobustScaling,
                missing_value_strategy: MissingValueStrategy::Median,
                feature_scaling: true,
                feature_selection: true,
            },
            
            _ => FeatureConfig::default(),
        }
    }

    /// Get the model description
    pub fn description(&self) -> &'static str {
        match self {
            ModelType::ThroughputPredictor => "Predicts cell throughput based on network conditions and load",
            ModelType::LatencyEstimator => "Estimates end-to-end latency for different network paths",
            ModelType::PowerOptimizer => "Optimizes transmission power for energy efficiency and coverage",
            ModelType::ResourcePredictor => "Predicts future resource utilization patterns",
            ModelType::QoSPredictor => "Predicts Quality of Service metrics for different scenarios",
            ModelType::CellStateClassifier => "Classifies cell operational states (active, degraded, failed, etc.)",
            ModelType::HandoverDecision => "Determines optimal handover decisions for mobile users",
            ModelType::TrafficPatternClassifier => "Identifies and classifies network traffic patterns",
            ModelType::InterferenceClassifier => "Classifies interference types and identifies sources",
            ModelType::AlarmClassifier => "Classifies network alarms by type and severity",
            ModelType::LoadBalancer => "Optimally distributes traffic load across multiple cells",
            ModelType::CoverageOptimizer => "Optimizes network coverage while minimizing interference",
            ModelType::SpectrumOptimizer => "Optimizes spectrum allocation and utilization",
            ModelType::EnergyOptimizer => "Minimizes energy consumption while maintaining QoS",
            ModelType::ResourceAllocator => "Dynamic resource allocation using reinforcement learning",
            ModelType::NetworkController => "Autonomous network optimization and control",
            ModelType::AdaptiveScheduler => "Adaptive traffic scheduling optimization",
            ModelType::AnomalyDetector => "Detects anomalies and unusual patterns in network behavior",
            ModelType::MultiObjectiveOptimizer => "Optimizes multiple conflicting objectives simultaneously",
            ModelType::ParetoOptimizer => "Explores Pareto-optimal solutions for multi-objective problems",
        }
    }

    /// Get the problem type (regression, classification, RL)
    pub fn problem_type(&self) -> ProblemType {
        match self {
            ModelType::ThroughputPredictor | ModelType::LatencyEstimator | 
            ModelType::PowerOptimizer | ModelType::ResourcePredictor | 
            ModelType::QoSPredictor => ProblemType::Regression,
            
            ModelType::CellStateClassifier | ModelType::HandoverDecision |
            ModelType::TrafficPatternClassifier | ModelType::InterferenceClassifier |
            ModelType::AlarmClassifier => ProblemType::Classification,
            
            ModelType::LoadBalancer | ModelType::CoverageOptimizer |
            ModelType::SpectrumOptimizer | ModelType::EnergyOptimizer => ProblemType::Optimization,
            
            ModelType::ResourceAllocator | ModelType::NetworkController |
            ModelType::AdaptiveScheduler => ProblemType::ReinforcementLearning,
            
            ModelType::AnomalyDetector => ProblemType::UnsupervisedLearning,
            
            ModelType::MultiObjectiveOptimizer | ModelType::ParetoOptimizer => ProblemType::MultiObjectiveOptimization,
        }
    }

    /// Get expected output range for this model type
    pub fn output_range(&self) -> (f64, f64) {
        match self {
            ModelType::ThroughputPredictor => (0.0, 1000.0), // Mbps
            ModelType::LatencyEstimator => (0.0, 1000.0), // ms
            ModelType::PowerOptimizer => (0.0, 50.0), // dBm
            ModelType::ResourcePredictor => (0.0, 100.0), // %
            ModelType::QoSPredictor => (0.0, 5.0), // MOS scale
            ModelType::CellStateClassifier => (0.0, 4.0), // 5 states (0-4)
            ModelType::HandoverDecision => (0.0, 1.0), // Binary decision
            ModelType::TrafficPatternClassifier => (0.0, 7.0), // 8 patterns (0-7)
            ModelType::InterferenceClassifier => (0.0, 3.0), // 4 types (0-3)
            ModelType::AlarmClassifier => (0.0, 4.0), // 5 severities (0-4)
            ModelType::LoadBalancer => (0.0, 1.0), // Normalized load distribution
            ModelType::AnomalyDetector => (0.0, 1.0), // Anomaly score
            _ => (0.0, 1.0), // Default normalized range
        }
    }

    /// Check if this model type supports online learning
    pub fn supports_online_learning(&self) -> bool {
        match self {
            ModelType::ResourceAllocator | ModelType::NetworkController |
            ModelType::AdaptiveScheduler | ModelType::AnomalyDetector => true,
            _ => false,
        }
    }

    /// Get recommended batch size for training
    pub fn recommended_batch_size(&self) -> usize {
        match self {
            ModelType::ThroughputPredictor | ModelType::LatencyEstimator => 64,
            ModelType::TrafficPatternClassifier | ModelType::LoadBalancer => 128,
            ModelType::AnomalyDetector => 256,
            ModelType::ResourceAllocator | ModelType::NetworkController => 32,
            _ => 64,
        }
    }
}

/// Problem types for neural network models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProblemType {
    /// Continuous value prediction
    Regression,
    /// Discrete class prediction
    Classification,
    /// Optimization problems
    Optimization,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Unsupervised learning
    UnsupervisedLearning,
    /// Multi-objective optimization
    MultiObjectiveOptimization,
}

/// Feature configuration for data preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// List of feature names
    pub features: Vec<String>,
    /// Normalization method
    pub normalization: NormalizationType,
    /// Strategy for handling missing values
    pub missing_value_strategy: MissingValueStrategy,
    /// Whether to apply feature scaling
    pub feature_scaling: bool,
    /// Whether to apply feature selection
    pub feature_selection: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            features: vec![
                "feature1".to_string(),
                "feature2".to_string(),
                "feature3".to_string(),
            ],
            normalization: NormalizationType::StandardScore,
            missing_value_strategy: MissingValueStrategy::Mean,
            feature_scaling: true,
            feature_selection: false,
        }
    }
}

/// Normalization methods for features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationType {
    /// Z-score normalization (mean=0, std=1)
    StandardScore,
    /// Min-max normalization (range 0-1)
    MinMax,
    /// Robust scaling (median and IQR)
    RobustScaling,
    /// Unit vector scaling
    UnitVector,
    /// No normalization
    None,
}

/// Strategies for handling missing values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    /// Fill with mean value
    Mean,
    /// Fill with median value
    Median,
    /// Fill with mode (most frequent)
    Mode,
    /// Fill with zero
    Zero,
    /// Forward fill (use last valid value)
    ForwardFill,
    /// Backward fill (use next valid value)
    BackwardFill,
    /// Linear interpolation
    Interpolate,
    /// Remove samples with missing values
    Remove,
}

/// RAN-specific neural network model trait
pub trait RanNeuralModel {
    /// Get model type
    fn model_type(&self) -> ModelType;
    
    /// Get model configuration
    fn config(&self) -> &NetworkConfig;
    
    /// Validate input features
    fn validate_input(&self, features: &[f64]) -> NeuralResult<()>;
    
    /// Preprocess input features
    fn preprocess(&self, features: &[f64]) -> NeuralResult<Vec<f64>>;
    
    /// Postprocess output predictions
    fn postprocess(&self, outputs: &[f64]) -> NeuralResult<Vec<f64>>;
    
    /// Get feature importance scores
    fn feature_importance(&self) -> Option<Vec<f64>>;
    
    /// Get model interpretation
    fn interpret_output(&self, outputs: &[f64]) -> String;
}

/// Model registry for managing different model types
#[derive(Debug, Default)]
pub struct ModelRegistry {
    /// Registered models
    models: HashMap<ModelType, Box<dyn RanNeuralModel + Send + Sync>>,
    /// Model metadata
    metadata: HashMap<ModelType, ModelRegistryEntry>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a model
    pub fn register_model(
        &mut self,
        model: Box<dyn RanNeuralModel + Send + Sync>,
        metadata: ModelRegistryEntry,
    ) {
        let model_type = model.model_type();
        self.models.insert(model_type, model);
        self.metadata.insert(model_type, metadata);
    }

    /// Get a model by type
    pub fn get_model(&self, model_type: ModelType) -> Option<&dyn RanNeuralModel> {
        self.models.get(&model_type).map(|m| m.as_ref())
    }

    /// List available models
    pub fn list_models(&self) -> Vec<ModelType> {
        self.models.keys().copied().collect()
    }

    /// Get model metadata
    pub fn get_metadata(&self, model_type: ModelType) -> Option<&ModelRegistryEntry> {
        self.metadata.get(&model_type)
    }

    /// Check if model is available
    pub fn has_model(&self, model_type: ModelType) -> bool {
        self.models.contains_key(&model_type)
    }
}

/// Model registry entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Author/creator
    pub author: String,
    /// Creation date
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Model file path
    pub model_path: Option<String>,
    /// Training dataset info
    pub training_data: Option<String>,
    /// Performance metrics
    pub performance: HashMap<String, f64>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_config() {
        let config = ModelType::ThroughputPredictor.default_config();
        assert_eq!(config.layers[0], 8); // Input size
        assert_eq!(config.layers.last(), &1); // Output size
        assert!(!config.features.is_empty());
    }

    #[test]
    fn test_feature_config() {
        let feature_config = ModelType::HandoverDecision.feature_config();
        assert!(feature_config.features.contains(&"source_rsrp".to_string()));
        assert!(feature_config.features.contains(&"target_rsrp".to_string()));
        assert_eq!(feature_config.normalization, NormalizationType::StandardScore);
    }

    #[test]
    fn test_problem_type() {
        assert_eq!(ModelType::ThroughputPredictor.problem_type(), ProblemType::Regression);
        assert_eq!(ModelType::CellStateClassifier.problem_type(), ProblemType::Classification);
        assert_eq!(ModelType::ResourceAllocator.problem_type(), ProblemType::ReinforcementLearning);
    }

    #[test]
    fn test_output_range() {
        let (min, max) = ModelType::ThroughputPredictor.output_range();
        assert_eq!(min, 0.0);
        assert_eq!(max, 1000.0);
        
        let (min, max) = ModelType::HandoverDecision.output_range();
        assert_eq!(min, 0.0);
        assert_eq!(max, 1.0);
    }

    #[test]
    fn test_online_learning_support() {
        assert!(ModelType::ResourceAllocator.supports_online_learning());
        assert!(!ModelType::ThroughputPredictor.supports_online_learning());
    }

    #[test]
    fn test_model_registry() {
        let mut registry = ModelRegistry::new();
        assert_eq!(registry.list_models().len(), 0);
        
        // Test registry operations
        assert!(!registry.has_model(ModelType::ThroughputPredictor));
        assert!(registry.get_model(ModelType::ThroughputPredictor).is_none());
    }

    #[test]
    fn test_normalization_types() {
        assert_eq!(
            ModelType::ThroughputPredictor.feature_config().normalization,
            NormalizationType::StandardScore
        );
        assert_eq!(
            ModelType::LatencyEstimator.feature_config().normalization,
            NormalizationType::MinMax
        );
    }

    #[test]
    fn test_missing_value_strategies() {
        assert_eq!(
            ModelType::ThroughputPredictor.feature_config().missing_value_strategy,
            MissingValueStrategy::Mean
        );
        assert_eq!(
            ModelType::LatencyEstimator.feature_config().missing_value_strategy,
            MissingValueStrategy::Zero
        );
    }
}