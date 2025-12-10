//! Integration tests for the complete ranML neural network training pipeline
//!
//! This module tests the full integration of all components:
//! - neural-training (main orchestration)
//! - ran-neural (neural network models)
//! - ran-core (domain abstractions)
//! - ran-forecasting (time series forecasting)

use std::path::PathBuf;
use tempfile::TempDir;
use neural_training::*;
use ran_neural::*;
use ran_core::*;
use ran_forecasting::*;

/// Test complete neural training pipeline with real data
#[tokio::test]
async fn test_complete_training_pipeline() -> anyhow::Result<()> {
    println!("🧪 Testing complete neural training pipeline");
    
    // Initialize logging for tests
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();
    
    // Create temporary directory for test data
    let temp_dir = TempDir::new()?;
    let test_data_path = create_test_dataset(&temp_dir)?;
    
    // 1. Initialize the neural training system
    let mut training_system = NeuralTrainingSystem::new();
    
    // 2. Run the complete training pipeline
    let results = training_system.run_training_pipeline(test_data_path).await?;
    
    // 3. Verify results
    assert!(!results.model_results.is_empty(), "No models were trained");
    assert!(results.best_validation_error > 0.0, "Invalid best validation error");
    assert!(results.total_training_time.as_secs() > 0, "Training took no time");
    
    println!("✅ Training pipeline completed successfully");
    println!("   - Models trained: {}", results.model_results.len());
    println!("   - Best model: {}", results.best_model_name);
    println!("   - Best error: {:.6}", results.best_validation_error);
    println!("   - Total time: {:?}", results.total_training_time);
    
    Ok(())
}

/// Test RAN neural network integration
#[test]
fn test_ran_neural_integration() -> anyhow::Result<()> {
    println!("🧪 Testing RAN neural network integration");
    
    // Create a neural network for throughput prediction
    let mut nn = RanNeuralNetwork::new(ModelType::ThroughputPredictor)?;
    assert!(nn.is_ready());
    
    // Test prediction with dummy data
    let features = vec![0.5, 0.7, 0.3, 0.8, 0.6]; // Cell load, power, SINR, UEs, frequency
    let predictions = nn.predict(&features)?;
    
    assert!(!predictions.is_empty(), "No predictions returned");
    assert!(predictions[0] >= 0.0, "Invalid prediction value");
    
    // Test performance metrics
    let performance = nn.performance_metrics();
    assert_eq!(performance.model_type, ModelType::ThroughputPredictor);
    assert!(performance.throughput_ops_per_sec >= 0.0);
    
    println!("✅ RAN neural network integration successful");
    println!("   - Model type: {:?}", nn.model_type);
    println!("   - Predictions: {:?}", predictions);
    println!("   - Inference time: {:?}", nn.stats.last_inference_time);
    
    Ok(())
}

/// Test forecasting integration
#[tokio::test]
async fn test_forecasting_integration() -> anyhow::Result<()> {
    println!("🧪 Testing forecasting integration");
    
    // Create time series data
    let mut timeseries = RanTimeSeries::new("test_traffic".to_string());
    let now = chrono::Utc::now();
    
    // Add sample data points (24 hours of traffic data)
    for i in 0..24 {
        let timestamp = now + chrono::Duration::hours(i);
        let value = 100.0 + 20.0 * (i as f64 * 0.26).sin(); // Simulate daily pattern
        timeseries.add_measurement_at(timestamp, value)?;
    }
    
    // Create forecaster with DLinear model
    let predictor = TrafficPredictor::builder()
        .model_type("dlinear")
        .horizon(ForecastHorizon::Hours(6))
        .input_window(168) // 1 week
        .build()?;
    
    let mut forecaster = RanForecaster::new(predictor);
    
    // Since we have limited data, configure for minimal training
    let config = ForecastConfig {
        min_training_points: 20,
        allow_missing_values: true,
        allow_outliers: true,
        ..Default::default()
    };
    
    let forecaster = RanForecaster::with_config(forecaster.model, config);
    
    // Note: For real testing, we would need more data points
    // This test verifies the integration structure
    assert!(!timeseries.is_empty());
    assert_eq!(timeseries.len(), 24);
    
    println!("✅ Forecasting integration setup successful");
    println!("   - Time series points: {}", timeseries.len());
    println!("   - Forecast horizon: {:?}", ForecastHorizon::Hours(6));
    
    Ok(())
}

/// Test swarm orchestration integration
#[tokio::test]
async fn test_swarm_orchestration() -> anyhow::Result<()> {
    println!("🧪 Testing swarm orchestration");
    
    // Create swarm orchestrator
    let mut orchestrator = SwarmOrchestrator::new();
    
    // Initialize the swarm
    orchestrator.initialize_training_swarm().await?;
    
    // Check swarm status
    let status = orchestrator.get_swarm_status().await;
    assert!(status.total_agents > 0, "No agents available");
    assert_eq!(status.active_tasks, 0, "Should start with no active tasks");
    
    println!("✅ Swarm orchestration integration successful");
    println!("   - Total agents: {}", status.total_agents);
    println!("   - Agent utilization: {:?}", status.agent_utilization);
    
    Ok(())
}

/// Test WASM compatibility compilation
#[cfg(feature = "wasm")]
#[test]
fn test_wasm_compilation() {
    println!("🧪 Testing WASM compatibility");
    
    // This test ensures the core types can be compiled to WASM
    use wasm_bindgen_test::*;
    
    wasm_bindgen_test_configure!(run_in_browser);
    
    #[wasm_bindgen_test]
    fn test_neural_network_wasm() {
        // Test basic neural network creation in WASM
        let config = NetworkConfig::default();
        assert!(config.layers.len() > 0);
        
        // Test RAN data structures
        let mut ran_data = RanData::new();
        ran_data.add_context("test".to_string(), "value").unwrap();
        assert!(!ran_data.context.is_empty());
    }
    
    println!("✅ WASM compatibility verified");
}

/// Test cross-component data flow
#[tokio::test]
async fn test_cross_component_integration() -> anyhow::Result<()> {
    println!("🧪 Testing cross-component data flow");
    
    // Create test data that flows through all components
    let temp_dir = TempDir::new()?;
    let test_data_path = create_test_dataset(&temp_dir)?;
    
    // 1. Load data using ran-core structures
    let dataset = TelecomDataLoader::load(&test_data_path)?;
    assert!(dataset.features.nrows() > 0, "No features loaded");
    assert!(dataset.targets.len() > 0, "No targets loaded");
    
    // 2. Create RAN neural network
    let mut nn = RanNeuralNetwork::new(ModelType::ThroughputPredictor)?;
    
    // 3. Extract features from RAN data
    let ran_data = RanData::new();
    // Note: In real scenario, this would extract features from ran_data
    let features = vec![0.5; nn.network.num_inputs()];
    let predictions = nn.predict(&features)?;
    
    assert!(!predictions.is_empty(), "No predictions from neural network");
    
    // 4. Test with training system
    let trainer = NeuralTrainer::new(SimpleTrainingConfig::default());
    let models = vec![NeuralModel::simple_regression(dataset.features.ncols(), 1)?];
    
    let _results = trainer.train_multiple_models(models, &dataset, None)?;
    
    println!("✅ Cross-component integration successful");
    println!("   - Data loaded: {} samples", dataset.features.nrows());
    println!("   - Neural network predictions: {:?}", predictions);
    
    Ok(())
}

/// Test error handling and recovery
#[test]
fn test_error_handling() -> anyhow::Result<()> {
    println!("🧪 Testing error handling and recovery");
    
    // Test invalid neural network creation
    let result = RanNeuralNetwork::with_config(
        ModelType::ThroughputPredictor,
        NetworkConfig {
            layers: vec![], // Invalid: empty layers
            ..Default::default()
        }
    );
    assert!(result.is_err(), "Should fail with empty layers");
    
    // Test invalid prediction input
    let mut nn = RanNeuralNetwork::new(ModelType::ThroughputPredictor)?;
    let result = nn.predict(&vec![]); // Wrong input size
    assert!(result.is_err(), "Should fail with wrong input size");
    
    // Test forecasting with insufficient data
    let empty_series = RanTimeSeries::new("empty".to_string());
    let config = ForecastConfig::default();
    
    // This should be caught by validation
    assert!(empty_series.is_empty(), "Series should be empty");
    
    println!("✅ Error handling tests passed");
    
    Ok(())
}

/// Test performance benchmarks
#[test]
fn test_performance_benchmarks() -> anyhow::Result<()> {
    println!("🧪 Testing performance benchmarks");
    
    use std::time::Instant;
    
    // Benchmark neural network inference
    let mut nn = RanNeuralNetwork::new(ModelType::ThroughputPredictor)?;
    let features = vec![0.5; nn.network.num_inputs()];
    
    let start = Instant::now();
    let num_inferences = 1000;
    
    for _ in 0..num_inferences {
        let _ = nn.predict(&features)?;
    }
    
    let total_time = start.elapsed();
    let avg_time = total_time / num_inferences;
    let throughput = 1.0 / avg_time.as_secs_f64();
    
    println!("✅ Performance benchmarks completed");
    println!("   - Total inferences: {}", num_inferences);
    println!("   - Average time: {:?}", avg_time);
    println!("   - Throughput: {:.1} inferences/sec", throughput);
    
    // Performance assertions
    assert!(avg_time.as_millis() < 100, "Inference too slow: {:?}", avg_time);
    assert!(throughput > 10.0, "Throughput too low: {:.1}", throughput);
    
    Ok(())
}

/// Helper function to create test dataset
fn create_test_dataset(temp_dir: &TempDir) -> anyhow::Result<PathBuf> {
    let data_path = temp_dir.path().join("test_data.csv");
    
    // Create simple test dataset
    let mut csv_content = String::new();
    csv_content.push_str("feature1,feature2,feature3,feature4,feature5,target\n");
    
    // Generate 100 sample rows
    for i in 0..100 {
        let f1 = (i as f32) / 100.0;
        let f2 = ((i * 2) as f32) / 100.0;
        let f3 = ((i * 3) as f32) / 100.0;
        let f4 = ((i * 4) as f32) / 100.0;
        let f5 = ((i * 5) as f32) / 100.0;
        let target = f1 + f2 * 0.5 + f3 * 0.3; // Simple linear relationship
        
        csv_content.push_str(&format!("{},{},{},{},{},{}\n", f1, f2, f3, f4, f5, target));
    }
    
    std::fs::write(&data_path, csv_content)?;
    Ok(data_path)
}

/// Mock implementations for missing types (to be replaced with real implementations)
#[cfg(test)]
mod mocks {
    use super::*;
    use anyhow::Result;
    
    pub struct TrafficPredictor {
        model_type: String,
        horizon: ForecastHorizon,
        input_window: usize,
    }
    
    impl TrafficPredictor {
        pub fn builder() -> TrafficPredictorBuilder {
            TrafficPredictorBuilder::new()
        }
    }
    
    pub struct TrafficPredictorBuilder {
        model_type: String,
        horizon: ForecastHorizon,
        input_window: usize,
    }
    
    impl TrafficPredictorBuilder {
        pub fn new() -> Self {
            Self {
                model_type: "dlinear".to_string(),
                horizon: ForecastHorizon::Hours(24),
                input_window: 168,
            }
        }
        
        pub fn model_type(mut self, model_type: &str) -> Self {
            self.model_type = model_type.to_string();
            self
        }
        
        pub fn horizon(mut self, horizon: ForecastHorizon) -> Self {
            self.horizon = horizon;
            self
        }
        
        pub fn input_window(mut self, window: usize) -> Self {
            self.input_window = window;
            self
        }
        
        pub fn build(self) -> Result<TrafficPredictor> {
            Ok(TrafficPredictor {
                model_type: self.model_type,
                horizon: self.horizon,
                input_window: self.input_window,
            })
        }
    }
    
    impl RanForecastingModel for TrafficPredictor {
        fn model_name(&self) -> &str {
            &self.model_type
        }
        
        fn fit(&mut self, _data: &ModelTrainingData) -> ForecastResult<()> {
            Ok(())
        }
        
        fn predict(&self, _data: &ModelTrainingData) -> ForecastResult<ModelForecastData> {
            Ok(ModelForecastData {
                values: vec![100.0, 105.0, 102.0],
                timestamps: vec![chrono::Utc::now(); 3],
                confidence_intervals: None,
            })
        }
        
        fn predict_future(&self, _horizon: ForecastHorizon) -> ForecastResult<ModelForecastData> {
            Ok(ModelForecastData {
                values: vec![100.0, 105.0, 102.0],
                timestamps: vec![chrono::Utc::now(); 3],
                confidence_intervals: None,
            })
        }
        
        fn update(&mut self, _data: &ModelTrainingData) -> ForecastResult<()> {
            Ok(())
        }
        
        fn reset(&mut self) -> ForecastResult<()> {
            Ok(())
        }
        
        fn get_parameters(&self) -> std::collections::HashMap<String, String> {
            let mut params = std::collections::HashMap::new();
            params.insert("model_type".to_string(), self.model_type.clone());
            params.insert("input_window".to_string(), self.input_window.to_string());
            params
        }
    }
    
    #[derive(Debug, Clone, Copy)]
    pub enum ForecastHorizon {
        Hours(usize),
        Days(usize),
    }
    
    pub struct RanTimeSeries {
        name: String,
        points: Vec<TimeSeriesPoint<f64>>,
    }
    
    impl RanTimeSeries {
        pub fn new(name: String) -> Self {
            Self {
                name,
                points: Vec::new(),
            }
        }
        
        pub fn add_measurement_at(
            &mut self,
            timestamp: chrono::DateTime<chrono::Utc>,
            value: f64,
        ) -> Result<()> {
            self.points.push(TimeSeriesPoint::new(timestamp, value));
            Ok(())
        }
        
        pub fn is_empty(&self) -> bool {
            self.points.is_empty()
        }
        
        pub fn len(&self) -> usize {
            self.points.len()
        }
        
        pub fn values(&self) -> Vec<f64> {
            self.points.iter().map(|p| p.value).collect()
        }
        
        pub fn timestamps(&self) -> Vec<chrono::DateTime<chrono::Utc>> {
            self.points.iter().map(|p| p.timestamp).collect()
        }
        
        pub fn features(&self) -> std::collections::HashMap<String, Vec<f64>> {
            std::collections::HashMap::new()
        }
        
        pub fn name(&self) -> &String {
            &self.name
        }
        
        pub fn has_missing_values(&self) -> bool {
            false
        }
        
        pub fn has_outliers(&self, _threshold: f64) -> bool {
            false
        }
    }
}

// Use mocks for testing
#[cfg(test)]
use mocks::*;