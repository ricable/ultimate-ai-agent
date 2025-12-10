//! Inference engine for real-time neural network prediction

use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{NeuralError, NeuralResult, ModelType, RanNeuralNetwork};

/// Inference engine for running neural network predictions
#[derive(Debug)]
pub struct InferenceEngine {
    /// Engine identifier
    pub id: Uuid,
    /// Configuration
    pub config: InferenceConfig,
    /// Performance statistics
    pub stats: InferenceEngineStats,
    /// GPU acceleration enabled
    pub gpu_enabled: bool,
    /// Batch processing enabled
    pub batch_enabled: bool,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            stats: InferenceEngineStats::default(),
            gpu_enabled: false,
            batch_enabled: true,
        }
    }

    /// Run inference on a single input
    pub fn predict(
        &mut self,
        network: &mut RanNeuralNetwork,
        input: &[f64],
    ) -> NeuralResult<InferenceResult> {
        let start_time = Instant::now();
        
        // Validate input
        self.validate_input(network, input)?;
        
        // Apply preprocessing if configured
        let preprocessed_input = if self.config.preprocessing_enabled {
            self.preprocess(input)?
        } else {
            input.to_vec()
        };

        // Run neural network inference
        let raw_output = network.predict(&preprocessed_input)?;
        
        // Apply postprocessing if configured
        let final_output = if self.config.postprocessing_enabled {
            self.postprocess(&raw_output, network.model_type)?
        } else {
            raw_output
        };

        let inference_time = start_time.elapsed();
        
        // Update statistics
        self.update_stats(inference_time, input.len(), final_output.len());
        
        Ok(InferenceResult {
            id: Uuid::new_v4(),
            model_type: network.model_type,
            input: input.to_vec(),
            output: final_output,
            confidence: self.calculate_confidence(&input, &network)?,
            inference_time,
            timestamp: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Run batch inference on multiple inputs
    pub fn predict_batch(
        &mut self,
        network: &mut RanNeuralNetwork,
        inputs: &[Vec<f64>],
    ) -> NeuralResult<Vec<InferenceResult>> {
        if !self.batch_enabled {
            return Err(NeuralError::Configuration("Batch processing not enabled".to_string()));
        }

        let start_time = Instant::now();
        let mut results = Vec::with_capacity(inputs.len());

        for input in inputs {
            let result = self.predict(network, input)?;
            results.push(result);
        }

        let total_time = start_time.elapsed();
        self.stats.total_batch_inferences += 1;
        self.stats.total_batch_time += total_time;

        Ok(results)
    }

    /// Run streaming inference (for real-time applications)
    pub fn predict_streaming(
        &mut self,
        network: &mut RanNeuralNetwork,
        input_stream: &mut dyn Iterator<Item = Vec<f64>>,
        callback: impl Fn(InferenceResult) -> NeuralResult<()>,
    ) -> NeuralResult<()> {
        for input in input_stream {
            let result = self.predict(network, &input)?;
            callback(result)?;
        }
        Ok(())
    }

    /// Validate input dimensions and values
    fn validate_input(&self, network: &RanNeuralNetwork, input: &[f64]) -> NeuralResult<()> {
        // Check input size
        if input.len() != network.network.num_inputs() {
            return Err(NeuralError::InvalidInput(format!(
                "Expected {} inputs, got {}",
                network.network.num_inputs(),
                input.len()
            )));
        }

        // Check for invalid values
        for (i, &value) in input.iter().enumerate() {
            if value.is_nan() {
                return Err(NeuralError::InvalidInput(format!(
                    "NaN value at input index {}",
                    i
                )));
            }
            if value.is_infinite() {
                return Err(NeuralError::InvalidInput(format!(
                    "Infinite value at input index {}",
                    i
                )));
            }
        }

        // Check value ranges if configured
        if let Some(ref ranges) = self.config.input_ranges {
            for (i, &value) in input.iter().enumerate() {
                if let Some((min, max)) = ranges.get(i) {
                    if value < *min || value > *max {
                        return Err(NeuralError::InvalidInput(format!(
                            "Input {} value {} outside valid range [{}, {}]",
                            i, value, min, max
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Preprocess input data
    fn preprocess(&self, input: &[f64]) -> NeuralResult<Vec<f64>> {
        let mut processed = input.to_vec();

        // Apply normalization if configured
        if let Some(ref normalization) = self.config.normalization {
            match normalization {
                NormalizationConfig::StandardScore { mean, std } => {
                    for (i, value) in processed.iter_mut().enumerate() {
                        if let (Some(m), Some(s)) = (mean.get(i), std.get(i)) {
                            *value = (*value - m) / s;
                        }
                    }
                }
                NormalizationConfig::MinMax { min, max } => {
                    for (i, value) in processed.iter_mut().enumerate() {
                        if let (Some(min_val), Some(max_val)) = (min.get(i), max.get(i)) {
                            *value = (*value - min_val) / (max_val - min_val);
                        }
                    }
                }
            }
        }

        // Apply feature scaling if configured
        if let Some(scale_factor) = self.config.scale_factor {
            for value in &mut processed {
                *value *= scale_factor;
            }
        }

        Ok(processed)
    }

    /// Postprocess output data
    fn postprocess(&self, output: &[f64], model_type: ModelType) -> NeuralResult<Vec<f64>> {
        let mut processed = output.to_vec();

        // Apply model-specific postprocessing
        match model_type {
            ModelType::ThroughputPredictor => {
                // Ensure non-negative throughput values
                for value in &mut processed {
                    *value = value.max(0.0);
                }
            }
            ModelType::HandoverDecision => {
                // Apply sigmoid for binary classification
                for value in &mut processed {
                    *value = 1.0 / (1.0 + (-*value).exp());
                }
            }
            ModelType::CellStateClassifier => {
                // Apply softmax for multi-class classification
                let max_val = processed.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let mut exp_sum = 0.0;
                for value in &mut processed {
                    *value = (*value - max_val).exp();
                    exp_sum += *value;
                }
                for value in &mut processed {
                    *value /= exp_sum;
                }
            }
            _ => {
                // Default: clamp to output range
                let (min_val, max_val) = model_type.output_range();
                for value in &mut processed {
                    *value = value.clamp(min_val, max_val);
                }
            }
        }

        Ok(processed)
    }

    /// Calculate prediction confidence
    fn calculate_confidence(&self, _input: &[f64], _network: &RanNeuralNetwork) -> NeuralResult<f64> {
        // Simplified confidence calculation
        // In practice, this would use more sophisticated methods like:
        // - Ensemble variance
        // - Dropout-based uncertainty estimation
        // - Bayesian neural networks
        // - Model calibration
        Ok(0.85) // Default confidence
    }

    /// Update engine statistics
    fn update_stats(&mut self, inference_time: Duration, input_size: usize, output_size: usize) {
        self.stats.total_inferences += 1;
        self.stats.total_inference_time += inference_time;
        self.stats.last_inference_time = inference_time;
        
        // Update running averages
        if self.stats.total_inferences == 1 {
            self.stats.avg_inference_time = inference_time;
            self.stats.avg_input_size = input_size as f64;
            self.stats.avg_output_size = output_size as f64;
        } else {
            let alpha = 0.1; // Exponential moving average factor
            let new_avg_time = self.stats.avg_inference_time.mul_f64(1.0 - alpha) + inference_time.mul_f64(alpha);
            self.stats.avg_inference_time = new_avg_time;
            self.stats.avg_input_size = self.stats.avg_input_size * (1.0 - alpha) + input_size as f64 * alpha;
            self.stats.avg_output_size = self.stats.avg_output_size * (1.0 - alpha) + output_size as f64 * alpha;
        }

        // Update min/max
        if self.stats.min_inference_time.map_or(true, |min| inference_time < min) {
            self.stats.min_inference_time = Some(inference_time);
        }
        if self.stats.max_inference_time.map_or(true, |max| inference_time > max) {
            self.stats.max_inference_time = Some(inference_time);
        }
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> &InferenceEngineStats {
        &self.stats
    }

    /// Reset engine statistics
    pub fn reset_stats(&mut self) {
        self.stats = InferenceEngineStats::default();
    }

    /// Enable GPU acceleration
    pub fn enable_gpu(&mut self) -> NeuralResult<()> {
        // In practice, this would initialize GPU resources
        self.gpu_enabled = true;
        tracing::info!("GPU acceleration enabled for inference engine {}", self.id);
        Ok(())
    }

    /// Disable GPU acceleration
    pub fn disable_gpu(&mut self) {
        self.gpu_enabled = false;
        tracing::info!("GPU acceleration disabled for inference engine {}", self.id);
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        // In practice, this would check actual GPU availability
        cfg!(feature = "gpu")
    }

    /// Get throughput in predictions per second
    pub fn throughput(&self) -> f64 {
        if self.stats.avg_inference_time.as_secs_f64() > 0.0 {
            1.0 / self.stats.avg_inference_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Configuration for inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Enable preprocessing
    pub preprocessing_enabled: bool,
    /// Enable postprocessing
    pub postprocessing_enabled: bool,
    /// Normalization configuration
    pub normalization: Option<NormalizationConfig>,
    /// Scale factor for inputs
    pub scale_factor: Option<f64>,
    /// Valid input ranges
    pub input_ranges: Option<Vec<(f64, f64)>>,
    /// Inference timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Enable confidence estimation
    pub confidence_enabled: bool,
    /// Batch size for batch processing
    pub batch_size: usize,
    /// Enable GPU acceleration
    pub gpu_enabled: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            preprocessing_enabled: true,
            postprocessing_enabled: true,
            normalization: None,
            scale_factor: None,
            input_ranges: None,
            timeout_ms: Some(1000), // 1 second timeout
            confidence_enabled: true,
            batch_size: 64,
            gpu_enabled: false,
        }
    }
}

/// Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationConfig {
    /// Z-score normalization
    StandardScore {
        mean: Vec<f64>,
        std: Vec<f64>,
    },
    /// Min-max normalization
    MinMax {
        min: Vec<f64>,
        max: Vec<f64>,
    },
}

/// Result of an inference operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Result identifier
    pub id: Uuid,
    /// Model type used
    pub model_type: ModelType,
    /// Input features
    pub input: Vec<f64>,
    /// Predicted output
    pub output: Vec<f64>,
    /// Prediction confidence score
    pub confidence: f64,
    /// Time taken for inference
    pub inference_time: Duration,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl InferenceResult {
    /// Get the primary prediction value
    pub fn primary_prediction(&self) -> f64 {
        self.output.first().copied().unwrap_or(0.0)
    }

    /// Get prediction as classification result
    pub fn as_classification(&self) -> usize {
        self.output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Check if prediction is above threshold
    pub fn above_threshold(&self, threshold: f64) -> bool {
        self.primary_prediction() > threshold
    }

    /// Get confidence level
    pub fn confidence_level(&self) -> ConfidenceLevel {
        match self.confidence {
            c if c >= 0.9 => ConfidenceLevel::VeryHigh,
            c if c >= 0.8 => ConfidenceLevel::High,
            c if c >= 0.6 => ConfidenceLevel::Medium,
            c if c >= 0.4 => ConfidenceLevel::Low,
            _ => ConfidenceLevel::VeryLow,
        }
    }

    /// Add metadata
    pub fn add_metadata<T: serde::Serialize>(&mut self, key: String, value: T) -> NeuralResult<()> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
        self.metadata.insert(key, json_value);
        Ok(())
    }
}

/// Confidence levels for predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Inference engine performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceEngineStats {
    /// Total number of inferences
    pub total_inferences: u64,
    /// Total inference time
    pub total_inference_time: Duration,
    /// Average inference time
    pub avg_inference_time: Duration,
    /// Last inference time
    pub last_inference_time: Duration,
    /// Minimum inference time
    pub min_inference_time: Option<Duration>,
    /// Maximum inference time
    pub max_inference_time: Option<Duration>,
    /// Average input size
    pub avg_input_size: f64,
    /// Average output size
    pub avg_output_size: f64,
    /// Total batch inferences
    pub total_batch_inferences: u64,
    /// Total batch processing time
    pub total_batch_time: Duration,
}

impl InferenceEngineStats {
    /// Get throughput in inferences per second
    pub fn throughput(&self) -> f64 {
        if self.avg_inference_time.as_secs_f64() > 0.0 {
            1.0 / self.avg_inference_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get latency percentiles (simplified)
    pub fn latency_percentiles(&self) -> (Duration, Duration, Duration) {
        // Simplified - in practice would track histogram
        let p50 = self.avg_inference_time;
        let p95 = self.max_inference_time.unwrap_or(self.avg_inference_time);
        let p99 = self.max_inference_time.unwrap_or(self.avg_inference_time);
        (p50, p95, p99)
    }

    /// Get statistics summary
    pub fn summary(&self) -> String {
        format!(
            "Inferences: {}, Avg Time: {:.2}ms, Throughput: {:.1} ops/sec, Min: {:.2}ms, Max: {:.2}ms",
            self.total_inferences,
            self.avg_inference_time.as_secs_f64() * 1000.0,
            self.throughput(),
            self.min_inference_time.map_or(0.0, |d| d.as_secs_f64() * 1000.0),
            self.max_inference_time.map_or(0.0, |d| d.as_secs_f64() * 1000.0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RanNeuralNetwork, ModelType};

    #[test]
    fn test_inference_engine_creation() {
        let config = InferenceConfig::default();
        let engine = InferenceEngine::new(config);
        
        assert!(engine.batch_enabled);
        assert!(!engine.gpu_enabled);
        assert_eq!(engine.stats.total_inferences, 0);
    }

    #[test]
    fn test_inference_config() {
        let config = InferenceConfig {
            preprocessing_enabled: false,
            postprocessing_enabled: false,
            timeout_ms: Some(500),
            batch_size: 32,
            ..Default::default()
        };
        
        assert!(!config.preprocessing_enabled);
        assert!(!config.postprocessing_enabled);
        assert_eq!(config.timeout_ms, Some(500));
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_inference_result() {
        let result = InferenceResult {
            id: Uuid::new_v4(),
            model_type: ModelType::ThroughputPredictor,
            input: vec![1.0, 2.0, 3.0],
            output: vec![150.0],
            confidence: 0.85,
            inference_time: Duration::from_millis(10),
            timestamp: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        
        assert_eq!(result.primary_prediction(), 150.0);
        assert_eq!(result.confidence_level(), ConfidenceLevel::High);
        assert!(result.above_threshold(100.0));
    }

    #[test]
    fn test_confidence_levels() {
        assert_eq!(ConfidenceLevel::VeryHigh, ConfidenceLevel::VeryHigh);
        assert_ne!(ConfidenceLevel::High, ConfidenceLevel::Medium);
    }

    #[test]
    fn test_stats_throughput() {
        let mut stats = InferenceEngineStats::default();
        stats.avg_inference_time = Duration::from_millis(10);
        
        assert!((stats.throughput() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_normalization_config() {
        let config = NormalizationConfig::StandardScore {
            mean: vec![0.0, 1.0, 2.0],
            std: vec![1.0, 1.0, 1.0],
        };
        
        match config {
            NormalizationConfig::StandardScore { mean, std } => {
                assert_eq!(mean.len(), 3);
                assert_eq!(std.len(), 3);
            }
            _ => panic!("Wrong normalization type"),
        }
    }
}