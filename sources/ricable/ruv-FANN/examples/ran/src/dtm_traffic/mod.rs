pub mod models;
pub mod cuda_kernels;
pub mod amos_generator;

use std::collections::HashMap;
use ndarray::{Array2, Array3, Array4};

/// 5QI Quality Indicators for QoS-aware predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QoSIndicator {
    QI1,   // Conversational Voice
    QI2,   // Conversational Video (Live Streaming)
    QI3,   // Real Time Gaming
    QI4,   // Non-Conversational Video (Buffered Streaming)
    QI5,   // IMS Signaling
    QI65,  // Mission Critical user plane Push To Talk voice
    QI66,  // Non-Mission-Critical user plane Push To Talk voice
    QI67,  // Mission Critical Video user plane
    QI69,  // Non-Mission Critical Video user plane
    QI70,  // Mission Critical Data user plane
    QI79,  // V2X messages
    QI80,  // Low Latency eMBB applications
}

/// Service types for traffic classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ServiceType {
    EMBB,   // Enhanced Mobile Broadband
    VoNR,   // Voice over New Radio
    URLLC,  // Ultra-Reliable Low-Latency Communications
    MIoT,   // Massive IoT
}

/// Network layers for layer-specific predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetworkLayer {
    L2100,  // 2100 MHz LTE layer
    N78,    // 3.5 GHz 5G NR layer
    N258,   // 26 GHz mmWave layer
}

/// Traffic pattern data structure
#[derive(Debug, Clone)]
pub struct TrafficPattern {
    pub timestamp: i64,
    pub prb_utilization: f32,
    pub layer: NetworkLayer,
    pub service_type: ServiceType,
    pub qos_indicators: HashMap<QoSIndicator, f32>,
    pub user_count: u32,
    pub throughput_mbps: f32,
}

/// Multi-horizon forecast result
#[derive(Debug, Clone)]
pub struct ForecastResult {
    pub horizons: Vec<i64>,  // Timestamps for each horizon
    pub prb_predictions: Array2<f32>,  // [horizon, layer]
    pub service_predictions: Array3<f32>,  // [horizon, layer, service_type]
    pub qos_predictions: Array3<f32>,  // [horizon, layer, qos_indicator]
    pub confidence_intervals: Array3<f32>,  // [horizon, layer, (lower, upper)]
}

/// Traffic predictor configuration
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    pub sequence_length: usize,
    pub forecast_horizons: Vec<usize>,  // e.g., [15, 30, 60] minutes
    pub lstm_hidden_size: usize,
    pub gru_hidden_size: usize,
    pub tcn_filters: usize,
    pub tcn_kernel_size: usize,
    pub tcn_dilations: Vec<usize>,
    pub dropout_rate: f32,
    pub learning_rate: f32,
    pub batch_size: usize,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            sequence_length: 96,  // 24 hours with 15-min intervals
            forecast_horizons: vec![4, 8, 16, 24],  // 1h, 2h, 4h, 6h
            lstm_hidden_size: 256,
            gru_hidden_size: 256,
            tcn_filters: 64,
            tcn_kernel_size: 3,
            tcn_dilations: vec![1, 2, 4, 8, 16, 32],
            dropout_rate: 0.2,
            learning_rate: 0.001,
            batch_size: 32,
        }
    }
}

/// Main traffic predictor combining LSTM, GRU, and TCN
pub struct TrafficPredictor {
    config: PredictorConfig,
    lstm_model: models::LSTMPredictor,
    gru_model: models::GRUPredictor,
    tcn_model: models::TCNPredictor,
    ensemble_weights: Array2<f32>,
}

impl TrafficPredictor {
    pub fn new(config: PredictorConfig) -> Self {
        let lstm_model = models::LSTMPredictor::new(&config);
        let gru_model = models::GRUPredictor::new(&config);
        let tcn_model = models::TCNPredictor::new(&config);
        
        // Initialize ensemble weights [model, horizon]
        let n_models = 3;
        let n_horizons = config.forecast_horizons.len();
        let ensemble_weights = Array2::from_elem((n_models, n_horizons), 1.0 / n_models as f32);
        
        Self {
            config,
            lstm_model,
            gru_model,
            tcn_model,
            ensemble_weights,
        }
    }
    
    /// Train the predictor on historical traffic patterns
    pub fn train(&mut self, patterns: &[TrafficPattern], epochs: usize) -> Result<(), String> {
        // Prepare training data
        let (features, targets) = self.prepare_training_data(patterns)?;
        
        // Train individual models
        self.lstm_model.train(&features, &targets, epochs)?;
        self.gru_model.train(&features, &targets, epochs)?;
        self.tcn_model.train(&features, &targets, epochs)?;
        
        // Optimize ensemble weights using validation data
        self.optimize_ensemble_weights(&features, &targets)?;
        
        Ok(())
    }
    
    /// Predict traffic patterns for multiple horizons
    pub fn predict(&self, recent_patterns: &[TrafficPattern]) -> Result<ForecastResult, String> {
        if recent_patterns.len() < self.config.sequence_length {
            return Err("Insufficient historical data for prediction".to_string());
        }
        
        // Extract features from recent patterns
        let features = self.extract_features(recent_patterns)?;
        
        // Get predictions from each model
        let lstm_pred = self.lstm_model.predict(&features)?;
        let gru_pred = self.gru_model.predict(&features)?;
        let tcn_pred = self.tcn_model.predict(&features)?;
        
        // Ensemble predictions
        let ensemble_pred = self.ensemble_predictions(&lstm_pred, &gru_pred, &tcn_pred);
        
        // Calculate confidence intervals
        let confidence_intervals = self.calculate_confidence_intervals(&lstm_pred, &gru_pred, &tcn_pred);
        
        // Parse predictions into structured result
        self.parse_predictions(ensemble_pred, confidence_intervals)
    }
    
    /// Classify service types based on traffic patterns
    pub fn classify_service_type(&self, pattern: &TrafficPattern) -> ServiceType {
        // Rule-based classification enhanced by ML predictions
        match pattern.layer {
            NetworkLayer::L2100 => {
                if pattern.qos_indicators.get(&QoSIndicator::QI1).unwrap_or(&0.0) > &0.5 {
                    ServiceType::VoNR
                } else if pattern.throughput_mbps > 100.0 {
                    ServiceType::EMBB
                } else {
                    ServiceType::MIoT
                }
            }
            NetworkLayer::N78 => {
                if pattern.qos_indicators.get(&QoSIndicator::QI80).unwrap_or(&0.0) > &0.5 {
                    ServiceType::URLLC
                } else {
                    ServiceType::EMBB
                }
            }
            NetworkLayer::N258 => ServiceType::EMBB,  // mmWave primarily for eMBB
        }
    }
    
    /// Generate AMOS scripts for load balancing based on predictions
    pub fn generate_amos_scripts(&self, predictions: &ForecastResult) -> Vec<String> {
        amos_generator::generate_load_balancing_scripts(predictions, &self.config)
    }
    
    // Helper methods
    fn prepare_training_data(&self, _patterns: &[TrafficPattern]) -> Result<(Array4<f32>, Array4<f32>), String> {
        // Implementation for data preparation
        // Returns (features, targets) arrays
        todo!("Implement training data preparation")
    }
    
    fn extract_features(&self, patterns: &[TrafficPattern]) -> Result<Array3<f32>, String> {
        // Extract features: [batch, sequence, features]
        let n_features = 10 + QoSIndicator::QI80 as usize + 1;  // Basic + QoS indicators
        let mut features = Array3::zeros((1, self.config.sequence_length, n_features));
        
        for (i, pattern) in patterns.iter().rev().take(self.config.sequence_length).enumerate() {
            let idx = self.config.sequence_length - i - 1;
            
            // Basic features
            features[[0, idx, 0]] = pattern.prb_utilization;
            features[[0, idx, 1]] = pattern.user_count as f32;
            features[[0, idx, 2]] = pattern.throughput_mbps;
            features[[0, idx, 3]] = pattern.timestamp as f32 / 1e9;  // Normalized timestamp
            
            // Layer encoding (one-hot)
            match pattern.layer {
                NetworkLayer::L2100 => features[[0, idx, 4]] = 1.0,
                NetworkLayer::N78 => features[[0, idx, 5]] = 1.0,
                NetworkLayer::N258 => features[[0, idx, 6]] = 1.0,
            }
            
            // Service type encoding
            match pattern.service_type {
                ServiceType::EMBB => features[[0, idx, 7]] = 1.0,
                ServiceType::VoNR => features[[0, idx, 8]] = 1.0,
                ServiceType::URLLC => features[[0, idx, 9]] = 1.0,
                ServiceType::MIoT => features[[0, idx, 10]] = 1.0,
            }
            
            // QoS indicators
            let qos_offset = 11;
            for (qos, value) in &pattern.qos_indicators {
                let qos_idx = *qos as usize;
                if qos_idx + qos_offset < n_features {
                    features[[0, idx, qos_idx + qos_offset]] = *value;
                }
            }
        }
        
        Ok(features)
    }
    
    fn ensemble_predictions(&self, lstm: &Array3<f32>, gru: &Array3<f32>, tcn: &Array3<f32>) -> Array3<f32> {
        let mut ensemble = Array3::zeros(lstm.dim());
        
        for h in 0..self.config.forecast_horizons.len() {
            let lstm_weight = self.ensemble_weights[[0, h]];
            let gru_weight = self.ensemble_weights[[1, h]];
            let tcn_weight = self.ensemble_weights[[2, h]];
            
            for i in 0..ensemble.shape()[0] {
                for j in 0..ensemble.shape()[2] {
                    ensemble[[i, h, j]] = lstm[[i, h, j]] * lstm_weight
                        + gru[[i, h, j]] * gru_weight
                        + tcn[[i, h, j]] * tcn_weight;
                }
            }
        }
        
        ensemble
    }
    
    fn calculate_confidence_intervals(&self, lstm: &Array3<f32>, gru: &Array3<f32>, tcn: &Array3<f32>) -> Array3<f32> {
        // Calculate confidence intervals based on model variance
        let mut intervals = Array3::zeros((lstm.shape()[0], lstm.shape()[1], 2));
        
        for i in 0..lstm.shape()[0] {
            for h in 0..lstm.shape()[1] {
                for j in 0..lstm.shape()[2] {
                    let predictions = vec![lstm[[i, h, j]], gru[[i, h, j]], tcn[[i, h, j]]];
                    let mean: f32 = predictions.iter().sum::<f32>() / predictions.len() as f32;
                    let variance: f32 = predictions.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f32>() / predictions.len() as f32;
                    let std_dev = variance.sqrt();
                    
                    // 95% confidence interval
                    intervals[[i, h, 0]] = mean - 1.96 * std_dev;
                    intervals[[i, h, 1]] = mean + 1.96 * std_dev;
                }
            }
        }
        
        intervals
    }
    
    fn optimize_ensemble_weights(&mut self, _features: &Array4<f32>, _targets: &Array4<f32>) -> Result<(), String> {
        // Optimize ensemble weights using validation performance
        // This is a simplified version - in practice, use gradient-based optimization
        todo!("Implement ensemble weight optimization")
    }
    
    fn parse_predictions(&self, predictions: Array3<f32>, confidence_intervals: Array3<f32>) -> Result<ForecastResult, String> {
        // Parse raw predictions into structured forecast result
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        let horizons: Vec<i64> = self.config.forecast_horizons.iter()
            .map(|&h| current_time + (h as i64 * 15 * 60))  // 15-minute intervals
            .collect();
        
        // Extract PRB predictions for each layer
        let prb_predictions = predictions.slice(ndarray::s![0, .., 0..3]).to_owned();
        
        // Extract service type predictions
        let service_predictions = predictions.slice(ndarray::s![0, .., 3..7]).to_owned()
            .into_shape((horizons.len(), 3, 4)).unwrap();
        
        // Extract QoS predictions
        let qos_predictions = predictions.slice(ndarray::s![0, .., 7..]).to_owned()
            .into_shape((horizons.len(), 3, 11)).unwrap();
        
        Ok(ForecastResult {
            horizons,
            prb_predictions,
            service_predictions,
            qos_predictions,
            confidence_intervals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_traffic_predictor_creation() {
        let config = PredictorConfig::default();
        let predictor = TrafficPredictor::new(config);
        assert_eq!(predictor.config.sequence_length, 96);
    }
    
    #[test]
    fn test_service_classification() {
        let config = PredictorConfig::default();
        let predictor = TrafficPredictor::new(config);
        
        let mut qos_indicators = HashMap::new();
        qos_indicators.insert(QoSIndicator::QI1, 0.8);
        
        let pattern = TrafficPattern {
            timestamp: 1234567890,
            prb_utilization: 0.75,
            layer: NetworkLayer::L2100,
            service_type: ServiceType::VoNR,
            qos_indicators,
            user_count: 100,
            throughput_mbps: 50.0,
        };
        
        assert_eq!(predictor.classify_service_type(&pattern), ServiceType::VoNR);
    }
}