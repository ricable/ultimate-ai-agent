//! Enhanced Machine Learning Model Implementation
//! 
//! This module provides ML models for RAN optimization tasks with CSV data integration.

use serde::{Deserialize, Serialize};
use crate::neural::kpi_predictor::CsvKpiFeatures;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModel {
    pub model_type: String,
    pub accuracy: f32,
    pub training_data_size: usize,
    pub feature_count: usize,
    pub is_trained: bool,
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl MLModel {
    pub fn new(model_type: String) -> Self {
        Self {
            model_type,
            accuracy: 0.0,
            training_data_size: 0,
            feature_count: 0,
            is_trained: false,
            weights: Vec::new(),
            bias: 0.0,
        }
    }
    
    pub fn train(&mut self, features: &[Vec<f32>], labels: &[f32]) -> Result<(), String> {
        if features.is_empty() || labels.is_empty() {
            return Err("Training data cannot be empty".to_string());
        }
        
        if features.len() != labels.len() {
            return Err("Features and labels must have the same length".to_string());
        }
        
        self.training_data_size = features.len();
        self.feature_count = features[0].len();
        
        // Initialize weights randomly
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        self.weights = (0..self.feature_count)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        self.bias = rng.gen_range(-1.0..1.0);
        
        // Simple gradient descent training simulation
        let learning_rate = 0.01;
        let epochs = 100;
        
        for _epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for (feature_vec, &target) in features.iter().zip(labels.iter()) {
                let prediction = self.predict_raw(feature_vec)?;
                let error = target - prediction;
                
                // Update weights
                for (i, &feature_val) in feature_vec.iter().enumerate() {
                    self.weights[i] += learning_rate * error * feature_val;
                }
                self.bias += learning_rate * error;
                
                total_loss += error * error;
            }
            
            // Calculate accuracy (simplified)
            let mse = total_loss / features.len() as f32;
            self.accuracy = (1.0 - mse.sqrt().min(1.0)).max(0.0);
        }
        
        self.is_trained = true;
        Ok(())
    }
    
    pub fn predict(&self, features: &[f32]) -> Result<f32, String> {
        if !self.is_trained {
            return Err("Model must be trained before prediction".to_string());
        }
        
        self.predict_raw(features)
    }
    
    fn predict_raw(&self, features: &[f32]) -> Result<f32, String> {
        if features.len() != self.feature_count {
            return Err(format!(
                "Feature count mismatch: expected {}, got {}",
                self.feature_count,
                features.len()
            ));
        }
        
        let weighted_sum: f32 = features.iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum::<f32>() + self.bias;
        
        // Apply sigmoid activation for bounded output
        Ok(1.0 / (1.0 + (-weighted_sum).exp()))
    }
    
    pub fn evaluate(&self, test_features: &[Vec<f32>], test_labels: &[f32]) -> Result<f32, String> {
        if !self.is_trained {
            return Err("Model must be trained before evaluation".to_string());
        }
        
        if test_features.len() != test_labels.len() {
            return Err("Test features and labels must have the same length".to_string());
        }
        
        let mut total_error = 0.0;
        
        for (features, &label) in test_features.iter().zip(test_labels.iter()) {
            let prediction = self.predict(features)?;
            let error = (prediction - label).abs();
            total_error += error;
        }
        
        let mean_absolute_error = total_error / test_features.len() as f32;
        Ok(1.0 - mean_absolute_error.min(1.0)) // Convert to accuracy
    }
    
    pub fn get_feature_importance(&self) -> Vec<(usize, f32)> {
        if !self.is_trained {
            return Vec::new();
        }
        
        let mut importance: Vec<(usize, f32)> = self.weights.iter()
            .enumerate()
            .map(|(i, &weight)| (i, weight.abs()))
            .collect();
        
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        importance
    }
}

impl Default for MLModel {
    fn default() -> Self {
        Self::new("linear_regression".to_string())
    }
}

impl MLModel {
    /// Extract features from CSV KPI data for training
    pub fn extract_csv_features(&self, csv_data: &CsvKpiFeatures) -> Vec<f32> {
        vec![
            csv_data.sinr_pusch_avg,
            csv_data.sinr_pucch_avg,
            csv_data.mac_dl_bler,
            csv_data.mac_ul_bler,
            csv_data.rrc_connected_users_avg,
            csv_data.ave_4g_lte_dl_user_thrput,
            csv_data.ave_4g_lte_ul_user_thrput,
            csv_data.dl_latency_avg,
            csv_data.erab_drop_rate_qci_5,
            csv_data.lte_intra_freq_ho_sr,
            csv_data.lte_inter_freq_ho_sr,
            csv_data.endc_setup_sr,
            csv_data.band_count,
            csv_data.active_ues_dl,
            csv_data.active_ues_ul,
        ]
    }
    
    /// Train model with CSV KPI data
    pub fn train_with_csv_data(&mut self, csv_data: &[(CsvKpiFeatures, f32)]) -> Result<(), String> {
        if csv_data.is_empty() {
            return Err("CSV training data cannot be empty".to_string());
        }
        
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        for (csv_features, label) in csv_data {
            let feature_vector = self.extract_csv_features(csv_features);
            features.push(feature_vector);
            labels.push(*label);
        }
        
        self.train(&features, &labels)
    }
    
    /// Predict using CSV KPI features
    pub fn predict_from_csv(&self, csv_data: &CsvKpiFeatures) -> Result<f32, String> {
        if !self.is_trained {
            return Err("Model must be trained before prediction".to_string());
        }
        
        let features = self.extract_csv_features(csv_data);
        self.predict(&features)
    }
    
    /// Get feature names for CSV data
    pub fn get_csv_feature_names(&self) -> Vec<String> {
        vec![
            "sinr_pusch_avg".to_string(),
            "sinr_pucch_avg".to_string(),
            "mac_dl_bler".to_string(),
            "mac_ul_bler".to_string(),
            "rrc_connected_users_avg".to_string(),
            "ave_4g_lte_dl_user_thrput".to_string(),
            "ave_4g_lte_ul_user_thrput".to_string(),
            "dl_latency_avg".to_string(),
            "erab_drop_rate_qci_5".to_string(),
            "lte_intra_freq_ho_sr".to_string(),
            "lte_inter_freq_ho_sr".to_string(),
            "endc_setup_sr".to_string(),
            "band_count".to_string(),
            "active_ues_dl".to_string(),
            "active_ues_ul".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_training() {
        let mut model = MLModel::new("test_model".to_string());
        
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        let labels = vec![0.5, 0.7, 0.9];
        
        assert!(model.train(&features, &labels).is_ok());
        assert!(model.is_trained);
        assert_eq!(model.feature_count, 3);
        assert_eq!(model.training_data_size, 3);
    }
    
    #[test]
    fn test_model_prediction() {
        let mut model = MLModel::new("test_model".to_string());
        
        let features = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
        ];
        let labels = vec![0.5, 0.8];
        
        model.train(&features, &labels).unwrap();
        
        let prediction = model.predict(&[1.5, 2.5]).unwrap();
        assert!(prediction >= 0.0 && prediction <= 1.0);
    }
}