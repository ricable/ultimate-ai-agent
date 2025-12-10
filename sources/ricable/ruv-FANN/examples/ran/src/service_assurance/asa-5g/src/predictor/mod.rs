use crate::{Config, Result, Error, EndcFailurePredictor, EndcFeatures, FailurePredictionResult, EndcMetrics};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

pub struct EndcPredictionEngine {
    predictor: EndcFailurePredictor,
    config: Config,
    historical_data: Vec<HistoricalRecord>,
}

#[derive(Debug, Clone)]
struct HistoricalRecord {
    timestamp: DateTime<Utc>,
    ue_id: String,
    metrics: EndcMetrics,
    actual_outcome: Option<String>,
}

impl EndcPredictionEngine {
    pub fn new(config: Config) -> Result<Self> {
        let predictor = EndcFailurePredictor::new(&config)?;
        
        Ok(Self {
            predictor,
            config,
            historical_data: Vec::new(),
        })
    }
    
    pub fn predict_failure(&self, metrics: &EndcMetrics) -> Result<FailurePredictionResult> {
        let features = self.extract_features(metrics)?;
        self.predictor.predict(&features)
    }
    
    fn extract_features(&self, metrics: &EndcMetrics) -> Result<EndcFeatures> {
        // Signal quality analysis
        let signal_quality_score = self.calculate_signal_quality_score(
            metrics.lte_rsrp, 
            metrics.lte_sinr, 
            metrics.nr_ssb_rsrp, 
            metrics.nr_ssb_sinr
        );
        
        // Network congestion assessment
        let network_congestion_score = self.calculate_congestion_score(metrics);
        
        // UE capability assessment
        let ue_capability_score = self.calculate_ue_capability_score(metrics);
        
        // Bearer configuration assessment
        let bearer_config_score = self.calculate_bearer_config_score(metrics);
        
        // Historical success rate
        let historical_success_rate = self.get_historical_success_rate(&metrics.ue_id);
        
        // Load indicators
        let load_indicators = vec![
            metrics.data_volume_mbps / 100.0, // Normalize to 0-1 range
            metrics.endc_setup_success_rate_cell,
        ];
        
        // Interference level (simplified)
        let interference_level = 1.0 - signal_quality_score;
        
        // Handover success rate (from historical data)
        let handover_success_rate = self.get_handover_success_rate(&metrics.ue_id);
        
        Ok(EndcFeatures {
            signal_quality_score,
            network_congestion_score,
            ue_capability_score,
            bearer_config_score,
            historical_success_rate,
            load_indicators,
            interference_level,
            handover_success_rate,
        })
    }
    
    fn calculate_signal_quality_score(&self, lte_rsrp: f64, lte_sinr: f64, nr_rsrp: f64, nr_sinr: f64) -> f64 {
        // Normalize signal values to 0-1 score
        let lte_rsrp_norm = ((lte_rsrp + 140.0) / 90.0).clamp(0.0, 1.0); // -140 to -50 dBm range
        let lte_sinr_norm = (lte_sinr / 30.0).clamp(0.0, 1.0); // 0 to 30 dB range
        let nr_rsrp_norm = ((nr_rsrp + 140.0) / 90.0).clamp(0.0, 1.0);
        let nr_sinr_norm = (nr_sinr / 40.0).clamp(0.0, 1.0); // 0 to 40 dB range
        
        // Weighted average
        (lte_rsrp_norm * 0.25 + lte_sinr_norm * 0.25 + nr_rsrp_norm * 0.25 + nr_sinr_norm * 0.25)
    }
    
    fn calculate_congestion_score(&self, metrics: &EndcMetrics) -> f64 {
        // Simple congestion estimation based on data volume and success rate
        let volume_factor = (metrics.data_volume_mbps / 100.0).clamp(0.0, 1.0);
        let success_factor = 1.0 - metrics.endc_setup_success_rate_cell;
        
        (volume_factor * 0.6 + success_factor * 0.4).clamp(0.0, 1.0)
    }
    
    fn calculate_ue_capability_score(&self, metrics: &EndcMetrics) -> f64 {
        // Estimate UE capability based on signal handling and configuration
        let signal_handling = self.calculate_signal_quality_score(
            metrics.lte_rsrp, 
            metrics.lte_sinr, 
            metrics.nr_ssb_rsrp, 
            metrics.nr_ssb_sinr
        );
        
        // Consider bearer configuration capability
        let bearer_complexity = if metrics.current_bearer_config.contains("SPLIT") {
            0.9 // High capability device
        } else if metrics.current_bearer_config.contains("SCG") {
            0.7 // Medium capability
        } else {
            0.5 // Basic capability
        };
        
        (signal_handling * 0.6 + bearer_complexity * 0.4).clamp(0.0, 1.0)
    }
    
    fn calculate_bearer_config_score(&self, metrics: &EndcMetrics) -> f64 {
        // Score bearer configuration optimality
        let config = &metrics.current_bearer_config;
        
        if config.contains("SPLIT_DRB") {
            0.95 // Optimal for high throughput
        } else if config.contains("SCG_DRB") {
            0.85 // Good for NR offload
        } else if config.contains("MCG_DRB") {
            0.70 // Basic configuration
        } else {
            0.50 // Suboptimal
        }
    }
    
    fn get_historical_success_rate(&self, ue_id: &str) -> f64 {
        let ue_records: Vec<_> = self.historical_data.iter()
            .filter(|record| record.ue_id == ue_id)
            .collect();
        
        if ue_records.is_empty() {
            return 0.95; // Default assumption
        }
        
        let successful = ue_records.iter()
            .filter(|record| {
                record.actual_outcome.as_ref()
                    .map(|outcome| outcome != "FAILURE")
                    .unwrap_or(true)
            })
            .count();
        
        successful as f64 / ue_records.len() as f64
    }
    
    fn get_handover_success_rate(&self, ue_id: &str) -> f64 {
        // Simplified handover success rate calculation
        let ue_records: Vec<_> = self.historical_data.iter()
            .filter(|record| record.ue_id == ue_id)
            .collect();
        
        if ue_records.is_empty() {
            return 0.90; // Default assumption
        }
        
        // In a real implementation, this would track actual handover events
        0.90 // Placeholder
    }
    
    pub fn update_historical_data(&mut self, ue_id: String, metrics: EndcMetrics, outcome: Option<String>) {
        let record = HistoricalRecord {
            timestamp: Utc::now(),
            ue_id,
            metrics,
            actual_outcome: outcome,
        };
        
        self.historical_data.push(record);
        
        // Keep only recent history (last 24 hours)
        let cutoff_time = Utc::now() - chrono::Duration::hours(24);
        self.historical_data.retain(|record| record.timestamp > cutoff_time);
    }
    
    pub fn train_model(&mut self, training_data: Vec<(EndcMetrics, String)>) -> Result<()> {
        if training_data.is_empty() {
            return Err(Error::InsufficientData("No training data provided".to_string()));
        }
        
        // Extract features and labels
        let mut features_list = Vec::new();
        let mut labels = Vec::new();
        
        for (metrics, label) in training_data {
            let features = self.extract_features(&metrics)?;
            let feature_vector = self.predictor.extract_feature_vector(&features);
            features_list.push(feature_vector);
            labels.push(label);
        }
        
        // Convert to ndarray format
        let n_samples = features_list.len();
        let n_features = features_list[0].len();
        let features_flat: Vec<f64> = features_list.into_iter().flat_map(|f| f.to_vec()).collect();
        
        let features_matrix = ndarray::Array2::from_shape_vec((n_samples, n_features), features_flat)
            .map_err(|e| Error::ModelTraining(format!("Failed to create feature matrix: {}", e)))?;
        
        self.predictor.train(&features_matrix, &labels)?;
        
        Ok(())
    }
    
    pub fn get_model_metrics(&self) -> &crate::models::PredictionMetrics {
        self.predictor.get_metrics()
    }
}