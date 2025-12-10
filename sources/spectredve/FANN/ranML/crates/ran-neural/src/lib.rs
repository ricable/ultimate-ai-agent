//! # RAN Neural - Neural Network Integration for Radio Access Networks
//!
//! This crate provides high-performance neural network capabilities specifically
//! designed for Radio Access Network (RAN) optimization and automation tasks.
//! Built on top of the ruv-FANN library, it offers specialized neural network
//! models and inference engines optimized for real-time RAN operations.
//!
//! ## Key Features
//!
//! - **Real-time Inference**: Sub-millisecond neural network inference for live network decisions
//! - **GPU Acceleration**: WebGPU backend for high-throughput processing
//! - **WASM Support**: Edge deployment capabilities for distributed processing
//! - **RAN-specific Models**: Pre-trained models for common RAN optimization tasks
//! - **Adaptive Learning**: Online learning capabilities for dynamic network conditions
//! - **Multi-objective Optimization**: Support for multiple conflicting objectives
//!
//! ## Neural Network Models
//!
//! ### Classification Models
//! - **Cell State Classifier**: Classify cell operational states
//! - **Handover Decision**: Determine optimal handover targets
//! - **Traffic Pattern Recognition**: Identify traffic patterns and anomalies
//! - **Interference Classification**: Classify interference types and sources
//!
//! ### Regression Models  
//! - **Throughput Predictor**: Predict cell throughput under various conditions
//! - **Latency Estimator**: Estimate end-to-end latency for different paths
//! - **Power Optimizer**: Optimize transmission power for efficiency
//! - **Load Balancer**: Distribute traffic optimally across cells
//!
//! ### Reinforcement Learning
//! - **Resource Allocator**: Dynamic resource allocation policies
//! - **Network Controller**: Autonomous network optimization
//! - **Adaptive Scheduler**: Traffic scheduling optimization
//!
//! ## Quick Start
//!
//! ```rust
//! use ran_neural::{RanNeuralNetwork, ModelType, InferenceEngine};
//! use ran_core::{Cell, PerformanceMetrics};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a throughput predictor
//! let mut predictor = RanNeuralNetwork::new(ModelType::ThroughputPredictor)?;
//! 
//! // Load pre-trained model
//! predictor.load_model("models/throughput_predictor.fann")?;
//! 
//! // Prepare input features
//! let features = vec![
//!     cell.load_percentage(),           // Cell load
//!     cell.config.tx_power,            // Transmission power
//!     metrics.get_kpi_value(KpiType::SignalQuality).unwrap_or(0.0), // SINR
//!     active_ues as f64,               // Number of active UEs
//! ];
//! 
//! // Run inference
//! let predicted_throughput = predictor.predict(&features)?;
//! println!("Predicted throughput: {:.2} Mbps", predicted_throughput[0]);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Re-export essential types
pub use ran_core::{
    RanError, RanResult, 
    Cell, GNodeB, UE, NetworkTopology,
    PerformanceMetrics, KpiType,
    OptimizationTarget, OptimizationObjective,
    TimeSeries, TimeSeriesPoint,
};

pub use ruv_fann::{
    Network, NetworkBuilder, ActivationFunction,
    TrainingAlgorithm, TrainingData, TrainingState,
};

/// Core error handling
pub mod error;

/// Neural network models for RAN optimization
pub mod models;

/// Inference engine for real-time prediction
pub mod inference;

/// Training utilities and data preparation
pub mod training;

/// Model management and persistence
pub mod model_store;

/// Feature engineering for RAN data
pub mod features;

/// Batch processing utilities
pub mod batch;

// Re-export main types
pub use error::{NeuralError, NeuralResult};
pub use models::{ModelType, RanNeuralModel};
pub use inference::{InferenceEngine, InferenceConfig, InferenceResult};
pub use training::{RanTrainer, TrainingConfig, TrainingMetrics};
pub use model_store::{ModelStore, ModelMetadata};
pub use features::{FeatureExtractor, FeatureConfig, FeatureVector};

/// Main neural network interface for RAN optimization
#[derive(Debug, Clone)]
pub struct RanNeuralNetwork {
    /// Network identifier
    pub id: Uuid,
    /// Model type
    pub model_type: ModelType,
    /// Underlying FANN network
    pub network: Network<f64>,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Feature extractor
    pub feature_extractor: FeatureExtractor,
    /// Inference configuration
    pub inference_config: InferenceConfig,
    /// Performance statistics
    pub stats: InferenceStats,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last inference timestamp
    pub last_inference: Option<DateTime<Utc>>,
}

impl RanNeuralNetwork {
    /// Create a new RAN neural network
    pub fn new(model_type: ModelType) -> NeuralResult<Self> {
        let config = model_type.default_config();
        let network = NetworkBuilder::new()
            .with_layers(&config.layers)
            .with_activation_function(config.activation)
            .build()
            .map_err(|e| NeuralError::NetworkCreation(e.to_string()))?;

        Ok(Self {
            id: Uuid::new_v4(),
            model_type,
            network,
            metadata: ModelMetadata::new(model_type),
            feature_extractor: FeatureExtractor::new(model_type.feature_config()),
            inference_config: InferenceConfig::default(),
            stats: InferenceStats::default(),
            created_at: Utc::now(),
            last_inference: None,
        })
    }

    /// Create a neural network with custom configuration
    pub fn with_config(model_type: ModelType, config: NetworkConfig) -> NeuralResult<Self> {
        let network = NetworkBuilder::new()
            .with_layers(&config.layers)
            .with_activation_function(config.activation)
            .build()
            .map_err(|e| NeuralError::NetworkCreation(e.to_string()))?;

        let mut nn = Self {
            id: Uuid::new_v4(),
            model_type,
            network,
            metadata: ModelMetadata::new(model_type),
            feature_extractor: FeatureExtractor::new(config.feature_config),
            inference_config: InferenceConfig::default(),
            stats: InferenceStats::default(),
            created_at: Utc::now(),
            last_inference: None,
        };

        nn.metadata.config = config;
        Ok(nn)
    }

    /// Load a pre-trained model from file
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> NeuralResult<()> {
        self.network = Network::load(path.as_ref())
            .map_err(|e| NeuralError::ModelLoading(e.to_string()))?;
        
        self.metadata.last_loaded = Some(Utc::now());
        tracing::info!("Loaded model from {:?}", path.as_ref());
        Ok(())
    }

    /// Save the model to file
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> NeuralResult<()> {
        self.network.save(path.as_ref())
            .map_err(|e| NeuralError::ModelSaving(e.to_string()))?;
        
        tracing::info!("Saved model to {:?}", path.as_ref());
        Ok(())
    }

    /// Run inference on input features
    pub fn predict(&mut self, features: &[f64]) -> NeuralResult<Vec<f64>> {
        let start_time = std::time::Instant::now();
        
        // Validate input features
        if features.len() != self.network.num_inputs() {
            return Err(NeuralError::InvalidInput(format!(
                "Expected {} inputs, got {}",
                self.network.num_inputs(),
                features.len()
            )));
        }

        // Run inference
        let outputs = self.network.run(features)
            .map_err(|e| NeuralError::InferenceError(e.to_string()))?;

        // Update statistics
        let inference_time = start_time.elapsed();
        self.stats.total_inferences += 1;
        self.stats.total_inference_time += inference_time;
        self.stats.last_inference_time = inference_time;
        self.last_inference = Some(Utc::now());

        // Update moving averages
        if self.stats.total_inferences == 1 {
            self.stats.avg_inference_time = inference_time;
        } else {
            let alpha = 0.1; // Exponential moving average factor
            self.stats.avg_inference_time = 
                self.stats.avg_inference_time.mul_f64(1.0 - alpha) + inference_time.mul_f64(alpha);
        }

        Ok(outputs)
    }

    /// Run batch inference on multiple inputs
    pub fn predict_batch(&mut self, inputs: &[Vec<f64>]) -> NeuralResult<Vec<Vec<f64>>> {
        let mut results = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let output = self.predict(input)?;
            results.push(output);
        }
        
        Ok(results)
    }

    /// Extract features from RAN data
    pub fn extract_features(&self, data: &RanData) -> NeuralResult<Vec<f64>> {
        self.feature_extractor.extract(data)
    }

    /// Train the network with provided data
    pub fn train(&mut self, training_data: &TrainingData<f64>) -> NeuralResult<TrainingMetrics> {
        let mut trainer = RanTrainer::new(self.model_type);
        trainer.train(&mut self.network, training_data)
    }

    /// Get network information
    pub fn info(&self) -> NetworkInfo {
        NetworkInfo {
            id: self.id,
            model_type: self.model_type,
            num_inputs: self.network.num_inputs(),
            num_outputs: self.network.num_outputs(),
            num_layers: self.network.num_layers(),
            total_neurons: self.network.num_neurons(),
            total_connections: self.network.num_connections(),
            created_at: self.created_at,
            last_inference: self.last_inference,
            stats: self.stats.clone(),
        }
    }

    /// Reset inference statistics
    pub fn reset_stats(&mut self) {
        self.stats = InferenceStats::default();
    }

    /// Check if model is ready for inference
    pub fn is_ready(&self) -> bool {
        self.network.num_inputs() > 0 && self.network.num_outputs() > 0
    }

    /// Get model performance metrics
    pub fn performance_metrics(&self) -> PerformanceReport {
        PerformanceReport {
            model_id: self.id,
            model_type: self.model_type,
            inference_stats: self.stats.clone(),
            throughput_ops_per_sec: if self.stats.avg_inference_time.as_secs_f64() > 0.0 {
                1.0 / self.stats.avg_inference_time.as_secs_f64()
            } else {
                0.0
            },
            memory_usage_mb: std::mem::size_of_val(&self.network) as f64 / 1024.0 / 1024.0,
            accuracy: self.metadata.accuracy,
            last_updated: Utc::now(),
        }
    }
}

/// Network configuration for custom models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Layer sizes (including input and output)
    pub layers: Vec<usize>,
    /// Activation function
    pub activation: ActivationFunction,
    /// Feature configuration
    pub feature_config: FeatureConfig,
    /// Learning rate for training
    pub learning_rate: f64,
    /// Training algorithm
    pub training_algorithm: TrainingAlgorithm,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            layers: vec![10, 20, 10, 1],
            activation: ActivationFunction::Sigmoid,
            feature_config: FeatureConfig::default(),
            learning_rate: 0.001,
            training_algorithm: TrainingAlgorithm::RProp,
            parameters: HashMap::new(),
        }
    }
}

/// RAN data container for feature extraction
#[derive(Debug, Clone)]
pub struct RanData {
    /// Cell information
    pub cell: Option<Cell>,
    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,
    /// Network topology
    pub topology: Option<NetworkTopology>,
    /// UE information
    pub ues: Vec<UE>,
    /// Time series data
    pub timeseries: HashMap<String, TimeSeries<f64>>,
    /// Additional context data
    pub context: HashMap<String, serde_json::Value>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl RanData {
    /// Create new RAN data container
    pub fn new() -> Self {
        Self {
            cell: None,
            metrics: None,
            topology: None,
            ues: Vec::new(),
            timeseries: HashMap::new(),
            context: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    /// Create RAN data from cell and metrics
    pub fn from_cell_metrics(cell: Cell, metrics: PerformanceMetrics) -> Self {
        Self {
            cell: Some(cell),
            metrics: Some(metrics),
            topology: None,
            ues: Vec::new(),
            timeseries: HashMap::new(),
            context: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    /// Add time series data
    pub fn add_timeseries(&mut self, name: String, series: TimeSeries<f64>) {
        self.timeseries.insert(name, series);
    }

    /// Add context data
    pub fn add_context<T: serde::Serialize>(&mut self, key: String, value: T) -> NeuralResult<()> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
        self.context.insert(key, json_value);
        Ok(())
    }
}

impl Default for RanData {
    fn default() -> Self {
        Self::new()
    }
}

/// Network information summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    /// Network identifier
    pub id: Uuid,
    /// Model type
    pub model_type: ModelType,
    /// Number of input neurons
    pub num_inputs: usize,
    /// Number of output neurons
    pub num_outputs: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Total number of neurons
    pub total_neurons: usize,
    /// Total number of connections
    pub total_connections: usize,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last inference timestamp
    pub last_inference: Option<DateTime<Utc>>,
    /// Performance statistics
    pub stats: InferenceStats,
}

/// Inference performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceStats {
    /// Total number of inferences performed
    pub total_inferences: u64,
    /// Total time spent on inference
    pub total_inference_time: std::time::Duration,
    /// Average inference time
    pub avg_inference_time: std::time::Duration,
    /// Last inference time
    pub last_inference_time: std::time::Duration,
    /// Minimum inference time recorded
    pub min_inference_time: Option<std::time::Duration>,
    /// Maximum inference time recorded
    pub max_inference_time: Option<std::time::Duration>,
}

impl InferenceStats {
    /// Calculate throughput in inferences per second
    pub fn throughput(&self) -> f64 {
        if self.avg_inference_time.as_secs_f64() > 0.0 {
            1.0 / self.avg_inference_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get statistics summary
    pub fn summary(&self) -> String {
        format!(
            "Inferences: {}, Avg Time: {:.2}ms, Throughput: {:.1} ops/sec",
            self.total_inferences,
            self.avg_inference_time.as_secs_f64() * 1000.0,
            self.throughput()
        )
    }
}

/// Performance report for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Model identifier
    pub model_id: Uuid,
    /// Model type
    pub model_type: ModelType,
    /// Inference statistics
    pub inference_stats: InferenceStats,
    /// Throughput in operations per second
    pub throughput_ops_per_sec: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Model accuracy (if available)
    pub accuracy: Option<f64>,
    /// Report timestamp
    pub last_updated: DateTime<Utc>,
}

/// Ensemble neural network for improved performance
#[derive(Debug)]
pub struct NeuralEnsemble {
    /// Ensemble identifier
    pub id: Uuid,
    /// Individual models in the ensemble
    pub models: Vec<RanNeuralNetwork>,
    /// Model weights for voting
    pub weights: Vec<f64>,
    /// Ensemble strategy
    pub strategy: EnsembleStrategy,
    /// Performance metrics
    pub ensemble_stats: InferenceStats,
}

impl NeuralEnsemble {
    /// Create a new neural ensemble
    pub fn new(strategy: EnsembleStrategy) -> Self {
        Self {
            id: Uuid::new_v4(),
            models: Vec::new(),
            weights: Vec::new(),
            strategy,
            ensemble_stats: InferenceStats::default(),
        }
    }

    /// Add a model to the ensemble
    pub fn add_model(&mut self, model: RanNeuralNetwork, weight: f64) {
        self.models.push(model);
        self.weights.push(weight);
    }

    /// Run ensemble prediction
    pub fn predict(&mut self, features: &[f64]) -> NeuralResult<Vec<f64>> {
        if self.models.is_empty() {
            return Err(NeuralError::InvalidInput("Empty ensemble".to_string()));
        }

        let mut predictions = Vec::new();
        for model in &mut self.models {
            let prediction = model.predict(features)?;
            predictions.push(prediction);
        }

        let ensemble_output = match self.strategy {
            EnsembleStrategy::Average => self.average_predictions(&predictions),
            EnsembleStrategy::WeightedAverage => self.weighted_average_predictions(&predictions),
            EnsembleStrategy::Majority => self.majority_vote(&predictions)?,
            EnsembleStrategy::BestModel => self.best_model_prediction(&predictions),
        };

        self.ensemble_stats.total_inferences += 1;
        Ok(ensemble_output)
    }

    fn average_predictions(&self, predictions: &[Vec<f64>]) -> Vec<f64> {
        let num_outputs = predictions[0].len();
        let mut result = vec![0.0; num_outputs];
        
        for prediction in predictions {
            for (i, &value) in prediction.iter().enumerate() {
                result[i] += value;
            }
        }
        
        for value in &mut result {
            *value /= predictions.len() as f64;
        }
        
        result
    }

    fn weighted_average_predictions(&self, predictions: &[Vec<f64>]) -> Vec<f64> {
        let num_outputs = predictions[0].len();
        let mut result = vec![0.0; num_outputs];
        let total_weight: f64 = self.weights.iter().sum();
        
        for (prediction, &weight) in predictions.iter().zip(self.weights.iter()) {
            for (i, &value) in prediction.iter().enumerate() {
                result[i] += value * weight;
            }
        }
        
        for value in &mut result {
            *value /= total_weight;
        }
        
        result
    }

    fn majority_vote(&self, predictions: &[Vec<f64>]) -> NeuralResult<Vec<f64>> {
        // For classification tasks, convert to class indices and vote
        let num_outputs = predictions[0].len();
        let mut votes = vec![HashMap::new(); num_outputs];
        
        for prediction in predictions {
            for (i, &value) in prediction.iter().enumerate() {
                let class = if value > 0.5 { 1 } else { 0 };
                *votes[i].entry(class).or_insert(0) += 1;
            }
        }
        
        let mut result = vec![0.0; num_outputs];
        for (i, vote_map) in votes.iter().enumerate() {
            let majority_class = vote_map.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&class, _)| class)
                .unwrap_or(0);
            result[i] = majority_class as f64;
        }
        
        Ok(result)
    }

    fn best_model_prediction(&self, predictions: &[Vec<f64>]) -> Vec<f64> {
        // Return prediction from the model with highest weight
        let best_idx = self.weights.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        predictions[best_idx].clone()
    }
}

/// Ensemble strategies for combining predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    /// Simple average of all predictions
    Average,
    /// Weighted average using model weights
    WeightedAverage,
    /// Majority voting (for classification)
    Majority,
    /// Use prediction from best performing model
    BestModel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_neural_network_creation() {
        let nn = RanNeuralNetwork::new(ModelType::ThroughputPredictor);
        assert!(nn.is_ok());
        
        let nn = nn.unwrap();
        assert_eq!(nn.model_type, ModelType::ThroughputPredictor);
        assert!(nn.is_ready());
    }

    #[test]
    fn test_custom_network_config() {
        let config = NetworkConfig {
            layers: vec![5, 10, 5, 1],
            activation: ActivationFunction::Tanh,
            feature_config: FeatureConfig::default(),
            learning_rate: 0.01,
            training_algorithm: TrainingAlgorithm::RProp,
            parameters: HashMap::new(),
        };

        let nn = RanNeuralNetwork::with_config(ModelType::HandoverDecision, config);
        assert!(nn.is_ok());
        
        let nn = nn.unwrap();
        assert_eq!(nn.network.num_inputs(), 5);
        assert_eq!(nn.network.num_outputs(), 1);
    }

    #[test]
    fn test_ran_data_creation() {
        let mut data = RanData::new();
        assert_eq!(data.ues.len(), 0);
        assert_eq!(data.timeseries.len(), 0);
        
        data.add_context("test_key".to_string(), "test_value").unwrap();
        assert_eq!(data.context.len(), 1);
    }

    #[test]
    fn test_inference_stats() {
        let mut stats = InferenceStats::default();
        assert_eq!(stats.total_inferences, 0);
        assert_eq!(stats.throughput(), 0.0);
        
        stats.avg_inference_time = std::time::Duration::from_millis(10);
        assert_abs_diff_eq!(stats.throughput(), 100.0, epsilon = 0.1);
    }

    #[test]
    fn test_neural_ensemble() {
        let mut ensemble = NeuralEnsemble::new(EnsembleStrategy::Average);
        assert_eq!(ensemble.models.len(), 0);
        
        let nn1 = RanNeuralNetwork::new(ModelType::ThroughputPredictor).unwrap();
        let nn2 = RanNeuralNetwork::new(ModelType::ThroughputPredictor).unwrap();
        
        ensemble.add_model(nn1, 1.0);
        ensemble.add_model(nn2, 1.0);
        
        assert_eq!(ensemble.models.len(), 2);
        assert_eq!(ensemble.weights.len(), 2);
    }

    #[test]
    fn test_performance_report() {
        let nn = RanNeuralNetwork::new(ModelType::LoadBalancer).unwrap();
        let report = nn.performance_metrics();
        
        assert_eq!(report.model_type, ModelType::LoadBalancer);
        assert!(report.memory_usage_mb >= 0.0);
        assert_eq!(report.inference_stats.total_inferences, 0);
    }
}