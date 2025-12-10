//! # RAN ML - Radio Access Network Machine Learning Suite
//!
//! A comprehensive machine learning platform for Radio Access Network (RAN) optimization
//! and automation. This library provides neural network training, time series forecasting,
//! and edge deployment capabilities specifically designed for telecom infrastructure.
//!
//! ## Features
//!
//! - **Neural Network Training**: Multi-architecture neural networks with swarm coordination
//! - **Time Series Forecasting**: Traffic prediction and capacity planning models
//! - **Edge Deployment**: WebAssembly compilation for edge devices
//! - **Performance Optimization**: GPU acceleration and SIMD optimization
//! - **Integration Pipeline**: Complete end-to-end ML pipeline
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use ranml::{RanMLPipeline, PipelineConfig};
//! 
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create pipeline with default configuration
//!     let config = PipelineConfig::default();
//!     let mut pipeline = RanMLPipeline::new(config);
//!     
//!     // Initialize all components
//!     pipeline.initialize().await?;
//!     
//!     // Run training pipeline
//!     let results = pipeline.run_training_pipeline("data/training.csv").await?;
//!     
//!     // Run integration tests
//!     let test_results = pipeline.run_integration_tests().await?;
//!     
//!     // Compile to WASM for edge deployment
//!     let wasm_metrics = pipeline.compile_to_wasm().await?;
//!     
//!     // Export results
//!     pipeline.export_results("output").await?;
//!     
//!     println!("Pipeline completed successfully!");
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The RAN ML suite is built with a modular architecture:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    RAN ML Pipeline                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Integration Pipeline  │  Performance Monitoring           │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Neural Training  │  Forecasting  │  Edge Deployment       │
//! ├─────────────────────────────────────────────────────────────┤
//! │  RAN Neural      │  RAN Forecasting  │  RAN Edge           │
//! ├─────────────────────────────────────────────────────────────┤
//! │             RAN Core (Domain Abstractions)                 │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ruv-FANN (Neural Network Engine)  │  WASM Runtime         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Components
//!
//! ### Neural Network Training (`neural-training`)
//! - Swarm-coordinated training with specialized agents
//! - Multiple neural network architectures
//! - Hyperparameter optimization
//! - Parallel training execution
//!
//! ### RAN Neural Networks (`ran-neural`)
//! - Real-time inference engines
//! - RAN-specific model types
//! - GPU acceleration support
//! - Performance optimization
//!
//! ### Time Series Forecasting (`ran-forecasting`)
//! - Traffic prediction models
//! - Capacity planning algorithms
//! - Multi-horizon forecasting
//! - Anomaly detection
//!
//! ### Domain Core (`ran-core`)
//! - Network element abstractions
//! - Performance metrics
//! - Geographic utilities
//! - Common data structures
//!
//! ### Edge Deployment (`ran-edge`)
//! - WebAssembly compilation
//! - Edge inference optimization
//! - Distributed deployment
//! - Resource-constrained execution
//!
//! ## Integration Pipeline
//!
//! The integration pipeline coordinates all components:
//!
//! 1. **Initialization**: Set up all components and connections
//! 2. **Training**: Execute neural network training with swarm coordination
//! 3. **Evaluation**: Run comprehensive integration tests
//! 4. **Benchmarking**: Measure performance across all components
//! 5. **Compilation**: Generate WASM binaries for edge deployment
//! 6. **Export**: Save results, metrics, and artifacts
//!
//! ## WASM Support
//!
//! All components can be compiled to WebAssembly for edge deployment:
//!
//! ```bash
//! # Build WASM packages
//! ./build-wasm.sh --webgpu --output wasm-dist
//! 
//! # Test WASM build
//! ./build-wasm.sh --test
//! ```
//!
//! ## Performance
//!
//! The system is optimized for high performance:
//!
//! - **Parallel Training**: Multi-agent swarm coordination
//! - **SIMD Optimization**: Vectorized computations
//! - **GPU Acceleration**: WebGPU and native GPU support
//! - **Memory Efficiency**: Optimized data structures
//! - **Edge Optimization**: Minimal WASM footprint

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![cfg_attr(not(feature = "std"), no_std)]

// Re-export core components
pub use ran_core::*;
pub use ran_neural::*;
pub use neural_training::*;

// Conditional exports based on features
#[cfg(feature = "forecasting")]
pub use ran_forecasting::*;

#[cfg(feature = "edge")]
pub use ran_edge::*;

// Integration pipeline
pub mod integration_pipeline;
pub use integration_pipeline::*;

// Version and build information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_HASH: &str = env!("VERGEN_GIT_SHA");
pub const BUILD_DATE: &str = env!("VERGEN_BUILD_DATE");

/// Build information for the RAN ML suite
#[derive(Debug, Clone)]
pub struct BuildInfo {
    /// Version string
    pub version: &'static str,
    /// Git commit hash
    pub git_hash: &'static str,
    /// Build date
    pub build_date: &'static str,
    /// Enabled features
    pub features: Vec<&'static str>,
    /// Target architecture
    pub target: &'static str,
}

impl BuildInfo {
    /// Get build information
    pub fn get() -> Self {
        let mut features = vec!["core"];
        
        #[cfg(feature = "neural")]
        features.push("neural");
        
        #[cfg(feature = "forecasting")]
        features.push("forecasting");
        
        #[cfg(feature = "edge")]
        features.push("edge");
        
        #[cfg(feature = "wasm")]
        features.push("wasm");
        
        #[cfg(feature = "gpu")]
        features.push("gpu");
        
        #[cfg(feature = "simd")]
        features.push("simd");

        Self {
            version: VERSION,
            git_hash: GIT_HASH,
            build_date: BUILD_DATE,
            features,
            target: std::env::consts::ARCH,
        }
    }
}

impl std::fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "RAN ML v{}", self.version)?;
        writeln!(f, "Git: {}", self.git_hash)?;
        writeln!(f, "Built: {}", self.build_date)?;
        writeln!(f, "Target: {}", self.target)?;
        writeln!(f, "Features: {}", self.features.join(", "))?;
        Ok(())
    }
}

/// Initialize the RAN ML library
/// 
/// This function performs any necessary global initialization,
/// including setting up logging, error handlers, and WASM bindings.
pub fn init() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging if not already done
    #[cfg(feature = "tracing")]
    {
        use tracing_subscriber::FmtSubscriber;
        
        let subscriber = FmtSubscriber::builder()
            .with_max_level(tracing::Level::INFO)
            .finish();
        
        let _ = tracing::subscriber::set_global_default(subscriber);
    }
    
    // Set panic hook for better error reporting
    #[cfg(feature = "wasm")]
    {
        console_error_panic_hook::set_once();
    }
    
    // Initialize performance counters
    #[cfg(feature = "monitoring")]
    {
        // TODO: Initialize performance monitoring
    }
    
    Ok(())
}

/// Create a default RAN ML pipeline
/// 
/// This is a convenience function that creates a pipeline with default
/// configuration suitable for most use cases.
pub fn create_default_pipeline() -> RanMLPipeline {
    let config = PipelineConfig::default();
    RanMLPipeline::new(config)
}

/// Create a RAN ML pipeline optimized for edge deployment
/// 
/// This configuration prioritizes small memory footprint and
/// fast inference over training capabilities.
pub fn create_edge_pipeline() -> RanMLPipeline {
    let mut config = PipelineConfig::default();
    
    // Optimize for edge deployment
    config.wasm.enabled = true;
    config.wasm.optimization = "s".to_string(); // Size optimization
    config.wasm.simd = true;
    config.wasm.threads = false; // Better compatibility
    
    // Minimal monitoring for edge
    config.monitoring.enabled = false;
    config.monitoring.profiling = false;
    
    // Reduced training capabilities
    config.training.parallel_training = false;
    config.training.save_checkpoints = false;
    
    RanMLPipeline::new(config)
}

/// Create a RAN ML pipeline optimized for training performance
/// 
/// This configuration maximizes training throughput and enables
/// all performance optimizations.
pub fn create_training_pipeline() -> RanMLPipeline {
    let mut config = PipelineConfig::default();
    
    // Maximize training performance
    config.training.parallel_training = true;
    config.training.save_checkpoints = true;
    
    // Enable comprehensive monitoring
    config.monitoring.enabled = true;
    config.monitoring.profiling = true;
    config.monitoring.interval = 5; // More frequent monitoring
    
    // Enable all integration features
    config.integration.tests_enabled = true;
    config.integration.benchmarks_enabled = true;
    config.integration.cross_validation = true;
    
    RanMLPipeline::new(config)
}

/// Utility functions for common operations
pub mod utils {
    use super::*;
    
    /// Load training data from CSV file
    pub fn load_training_data<P: AsRef<std::path::Path>>(
        path: P
    ) -> Result<neural_training::TelecomDataset, Box<dyn std::error::Error>> {
        neural_training::TelecomDataLoader::load(path)
            .map_err(|e| e.into())
    }
    
    /// Create a simple neural network for testing
    pub fn create_test_network() -> Result<RanNeuralNetwork, NeuralError> {
        RanNeuralNetwork::new(ModelType::ThroughputPredictor)
    }
    
    /// Generate synthetic training data for testing
    pub fn generate_synthetic_data(
        samples: usize, 
        features: usize
    ) -> neural_training::TelecomDataset {
        use ndarray::Array2;
        use rand::Rng;
        
        let mut rng = rand::thread_rng();
        let mut data = Vec::new();
        let mut targets = Vec::new();
        
        for _ in 0..samples {
            let mut feature_row = Vec::new();
            let mut target = 0.0;
            
            for j in 0..features {
                let feature = rng.gen::<f32>();
                feature_row.push(feature);
                target += feature * (j as f32 + 1.0) / features as f32;
            }
            
            data.extend(feature_row);
            targets.push(target);
        }
        
        let features_array = Array2::from_shape_vec((samples, features), data)
            .expect("Failed to create features array");
        
        neural_training::TelecomDataset {
            features: features_array,
            targets,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Validate pipeline configuration
    pub fn validate_config(config: &PipelineConfig) -> Result<(), String> {
        if config.training.train_test_split <= 0.0 || config.training.train_test_split >= 1.0 {
            return Err("Invalid train/test split ratio".to_string());
        }
        
        if config.monitoring.interval == 0 {
            return Err("Monitoring interval must be greater than 0".to_string());
        }
        
        if config.integration.test_iterations == 0 {
            return Err("Test iterations must be greater than 0".to_string());
        }
        
        Ok(())
    }
    
    /// Calculate memory usage estimate for configuration
    pub fn estimate_memory_usage(config: &PipelineConfig) -> f64 {
        let base_usage = 50.0; // Base memory in MB
        let training_overhead = if config.training.parallel_training { 100.0 } else { 50.0 };
        let monitoring_overhead = if config.monitoring.enabled { 20.0 } else { 0.0 };
        
        base_usage + training_overhead + monitoring_overhead
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        RanMLPipeline, PipelineConfig, PipelineState,
        BuildInfo, init, create_default_pipeline,
        create_edge_pipeline, create_training_pipeline,
    };
    
    pub use crate::utils::*;
    
    // Re-export commonly used types
    pub use ran_core::{RanError, RanResult, GeoCoordinate, TimeSeries};
    pub use ran_neural::{RanNeuralNetwork, ModelType, NetworkConfig};
    pub use neural_training::{
        NeuralTrainingSystem, TrainingResults, SwarmOrchestrator,
        SimpleTrainingConfig, NeuralTrainer
    };
    
    #[cfg(feature = "forecasting")]
    pub use ran_forecasting::{RanForecaster, ForecastHorizon, AccuracyMetric};
}

// WASM-specific exports
#[cfg(feature = "wasm")]
pub mod wasm {
    use wasm_bindgen::prelude::*;
    use super::*;
    
    /// Initialize WASM module
    #[wasm_bindgen(start)]
    pub fn wasm_init() {
        console_error_panic_hook::set_once();
        init().expect("Failed to initialize RAN ML");
    }
    
    /// Get version string for WASM
    #[wasm_bindgen]
    pub fn get_version() -> String {
        VERSION.to_string()
    }
    
    /// Get build info for WASM
    #[wasm_bindgen]
    pub fn get_build_info() -> JsValue {
        let build_info = BuildInfo::get();
        serde_wasm_bindgen::to_value(&build_info).unwrap()
    }
    
    /// Create a neural network in WASM
    #[wasm_bindgen]
    pub fn create_wasm_network(model_type: &str) -> Result<JsValue, JsError> {
        let model_type = match model_type {
            "throughput" => ModelType::ThroughputPredictor,
            "handover" => ModelType::HandoverDecision,
            "load_balancer" => ModelType::LoadBalancer,
            "interference" => ModelType::InterferenceClassifier,
            _ => return Err(JsError::new("Invalid model type")),
        };
        
        let network = RanNeuralNetwork::new(model_type)
            .map_err(|e| JsError::new(&e.to_string()))?;
        
        serde_wasm_bindgen::to_value(&network)
            .map_err(|e| JsError::new(&e.to_string()))
    }
    
    /// Run prediction in WASM
    #[wasm_bindgen]
    pub fn wasm_predict(network: JsValue, features: &[f64]) -> Result<Vec<f64>, JsError> {
        // Note: This is a simplified interface for WASM
        // In practice, we would need to properly handle the network state
        Ok(vec![features.iter().sum::<f64>() / features.len() as f64])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_build_info() {
        let info = BuildInfo::get();
        assert!(!info.version.is_empty());
        assert!(!info.features.is_empty());
        assert!(info.features.contains(&"core"));
    }
    
    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }
    
    #[test]
    fn test_pipeline_creation() {
        let default_pipeline = create_default_pipeline();
        assert_eq!(default_pipeline.get_state(), PipelineState::Uninitialized);
        
        let edge_pipeline = create_edge_pipeline();
        assert_eq!(edge_pipeline.get_state(), PipelineState::Uninitialized);
        
        let training_pipeline = create_training_pipeline();
        assert_eq!(training_pipeline.get_state(), PipelineState::Uninitialized);
    }
    
    #[test]
    fn test_utils() {
        // Test synthetic data generation
        let dataset = utils::generate_synthetic_data(100, 5);
        assert_eq!(dataset.features.nrows(), 100);
        assert_eq!(dataset.features.ncols(), 5);
        assert_eq!(dataset.targets.len(), 100);
        
        // Test configuration validation
        let valid_config = PipelineConfig::default();
        assert!(utils::validate_config(&valid_config).is_ok());
        
        // Test memory estimation
        let memory_estimate = utils::estimate_memory_usage(&valid_config);
        assert!(memory_estimate > 0.0);
    }
    
    #[tokio::test]
    async fn test_integration_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = RanMLPipeline::new(config);
        
        assert_eq!(pipeline.get_state(), PipelineState::Uninitialized);
        assert!(pipeline.networks.lock().await.is_empty());
    }
}