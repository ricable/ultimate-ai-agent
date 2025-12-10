//! Data Processing Utilities
//! 
//! This module provides data processing capabilities for the swarm system.

use crate::models::{RANConfiguration, RANMetrics};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessor {
    pub normalization_params: HashMap<String, NormalizationParams>,
    pub feature_extractors: Vec<FeatureExtractor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub min_value: f64,
    pub max_value: f64,
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractor {
    pub name: String,
    pub weight: f64,
    pub transform: String, // "linear", "log", "sqrt", "square"
}

impl DataProcessor {
    pub fn new() -> Self {
        Self {
            normalization_params: HashMap::new(),
            feature_extractors: Vec::new(),
        }
    }
    
    pub fn extract_features(&self, config: &RANConfiguration) -> Vec<f64> {
        vec![
            config.power_level as f64,
            config.antenna_tilt as f64,
            config.bandwidth as f64,
            config.frequency_band as f64,
            if config.beamforming_enabled { 1.0 } else { 0.0 },
            // Add more features as needed
        ]
    }
    
    pub fn normalize_features(&self, features: &[f64], feature_names: &[String]) -> Vec<f64> {
        features.iter()
            .zip(feature_names.iter())
            .map(|(&value, name)| {
                if let Some(params) = self.normalization_params.get(name) {
                    self.normalize_value(value, params)
                } else {
                    value
                }
            })
            .collect()
    }
    
    fn normalize_value(&self, value: f64, params: &NormalizationParams) -> f64 {
        if params.std_dev > 0.0 {
            // Z-score normalization
            (value - params.mean) / params.std_dev
        } else if params.max_value > params.min_value {
            // Min-max normalization
            (value - params.min_value) / (params.max_value - params.min_value)
        } else {
            value
        }
    }
    
    pub fn calculate_normalization_params(&mut self, data: &[(String, Vec<f64>)]) {
        for (name, values) in data {
            if values.is_empty() {
                continue;
            }
            
            let min_value = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_value = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            
            let variance = values.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();
            
            let params = NormalizationParams {
                min_value,
                max_value,
                mean,
                std_dev,
            };
            
            self.normalization_params.insert(name.clone(), params);
        }
    }
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}