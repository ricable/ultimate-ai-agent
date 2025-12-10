//! KPI Prediction Model
//! 
//! Comprehensive KPI prediction system that integrates with real CSV data
//! from fanndata.csv to predict network performance indicators.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Key Performance Indicator types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum KpiType {
    Availability,
    Throughput,
    Latency,
    SignalQuality,
    ErrorRate,
    HandoverSuccess,
    EndcEstablishment,
    UserExperience,
}

/// KPI prediction model specialized for network metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpiPredictor {
    pub model_id: String,
    pub kpi_type: KpiType,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub feature_names: Vec<String>,
    pub normalization_params: NormalizationParams,
    pub is_trained: bool,
    pub training_history: TrainingHistory,
    pub prediction_accuracy: f32,
    pub feature_importance: HashMap<String, f32>,
}

/// Normalization parameters for feature scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub means: Vec<f32>,
    pub std_devs: Vec<f32>,
    pub min_vals: Vec<f32>,
    pub max_vals: Vec<f32>,
}

/// Training history for performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    pub epochs: Vec<u32>,
    pub losses: Vec<f32>,
    pub accuracies: Vec<f32>,
    pub validation_losses: Vec<f32>,
    pub training_time_ms: u64,
    pub convergence_epoch: Option<u32>,
}

/// KPI prediction result with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpiPrediction {
    pub kpi_type: KpiType,
    pub predicted_value: f32,
    pub confidence: f32,
    pub prediction_interval: (f32, f32),
    pub contributing_features: Vec<FeatureContribution>,
    pub risk_factors: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Feature contribution to prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureContribution {
    pub feature_name: String,
    pub value: f32,
    pub contribution: f32,
    pub importance: f32,
}

/// CSV-based KPI features extracted from real data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvKpiFeatures {
    // Signal quality features
    pub sinr_pusch_avg: f32,
    pub sinr_pucch_avg: f32,
    pub ul_rssi_total: f32,
    pub mac_dl_bler: f32,
    pub mac_ul_bler: f32,
    
    // Traffic features
    pub rrc_connected_users_avg: f32,
    pub ul_volume_pdcp_gbytes: f32,
    pub dl_volume_pdcp_gbytes: f32,
    pub volte_traffic_erl: f32,
    pub eric_traff_erab_erl: f32,
    
    // Performance features
    pub ave_4g_lte_dl_user_thrput: f32,
    pub ave_4g_lte_ul_user_thrput: f32,
    pub dl_latency_avg: f32,
    pub erab_drop_rate_qci_5: f32,
    pub lte_dcr_volte: f32,
    
    // Handover features
    pub lte_intra_freq_ho_sr: f32,
    pub lte_inter_freq_ho_sr: f32,
    pub inter_freq_ho_attempts: f32,
    pub intra_freq_ho_attempts: f32,
    
    // ENDC features
    pub endc_establishment_att: f32,
    pub endc_establishment_succ: f32,
    pub endc_setup_sr: f32,
    pub endc_scg_failure_ratio: f32,
    
    // Context features
    pub frequency_band: String,
    pub band_count: f32,
    pub active_ues_dl: f32,
    pub active_ues_ul: f32,
}

impl KpiPredictor {
    /// Create a new KPI predictor for specific metric
    pub fn new(kpi_type: KpiType, model_id: String) -> Self {
        let feature_names = Self::get_feature_names_for_kpi(&kpi_type);
        let feature_count = feature_names.len();
        
        Self {
            model_id,
            kpi_type,
            weights: vec![0.0; feature_count],
            bias: 0.0,
            feature_names,
            normalization_params: NormalizationParams {
                means: vec![0.0; feature_count],
                std_devs: vec![1.0; feature_count],
                min_vals: vec![0.0; feature_count],
                max_vals: vec![1.0; feature_count],
            },
            is_trained: false,
            training_history: TrainingHistory {
                epochs: Vec::new(),
                losses: Vec::new(),
                accuracies: Vec::new(),
                validation_losses: Vec::new(),
                training_time_ms: 0,
                convergence_epoch: None,
            },
            prediction_accuracy: 0.0,
            feature_importance: HashMap::new(),
        }
    }
    
    /// Get relevant feature names for each KPI type
    fn get_feature_names_for_kpi(kpi_type: &KpiType) -> Vec<String> {
        match kpi_type {
            KpiType::Availability => vec![
                "cell_availability_pct".to_string(),
                "eric_traff_erab_erl".to_string(),
                "rrc_connected_users_avg".to_string(),
                "sinr_pusch_avg".to_string(),
                "mac_dl_bler".to_string(),
                "erab_drop_rate_qci_5".to_string(),
                "lte_intra_freq_ho_sr".to_string(),
            ],
            KpiType::Throughput => vec![
                "ave_4g_lte_dl_user_thrput".to_string(),
                "ave_4g_lte_ul_user_thrput".to_string(),
                "sinr_pusch_avg".to_string(),
                "sinr_pucch_avg".to_string(),
                "mac_dl_bler".to_string(),
                "rrc_connected_users_avg".to_string(),
                "ul_volume_pdcp_gbytes".to_string(),
                "dl_volume_pdcp_gbytes".to_string(),
                "band_count".to_string(),
            ],
            KpiType::Latency => vec![
                "dl_latency_avg".to_string(),
                "sinr_pusch_avg".to_string(),
                "rrc_connected_users_avg".to_string(),
                "eric_traff_erab_erl".to_string(),
                "lte_intra_freq_ho_sr".to_string(),
                "lte_inter_freq_ho_sr".to_string(),
                "mac_dl_bler".to_string(),
            ],
            KpiType::SignalQuality => vec![
                "sinr_pusch_avg".to_string(),
                "sinr_pucch_avg".to_string(),
                "ul_rssi_total".to_string(),
                "mac_dl_bler".to_string(),
                "mac_ul_bler".to_string(),
                "band_count".to_string(),
            ],
            KpiType::ErrorRate => vec![
                "mac_dl_bler".to_string(),
                "mac_ul_bler".to_string(),
                "erab_drop_rate_qci_5".to_string(),
                "lte_dcr_volte".to_string(),
                "sinr_pusch_avg".to_string(),
                "rrc_connected_users_avg".to_string(),
            ],
            KpiType::HandoverSuccess => vec![
                "lte_intra_freq_ho_sr".to_string(),
                "lte_inter_freq_ho_sr".to_string(),
                "inter_freq_ho_attempts".to_string(),
                "intra_freq_ho_attempts".to_string(),
                "sinr_pusch_avg".to_string(),
                "mac_dl_bler".to_string(),
            ],
            KpiType::EndcEstablishment => vec![
                "endc_establishment_att".to_string(),
                "endc_establishment_succ".to_string(),
                "endc_setup_sr".to_string(),
                "endc_scg_failure_ratio".to_string(),
                "sinr_pusch_avg".to_string(),
                "rrc_connected_users_avg".to_string(),
                "band_count".to_string(),
            ],
            KpiType::UserExperience => vec![
                "ave_4g_lte_dl_user_thrput".to_string(),
                "dl_latency_avg".to_string(),
                "sinr_pusch_avg".to_string(),
                "mac_dl_bler".to_string(),
                "lte_intra_freq_ho_sr".to_string(),
                "volte_traffic_erl".to_string(),
                "rrc_connected_users_avg".to_string(),
            ],
        }
    }
    
    /// Extract features from CSV data for prediction
    pub fn extract_features(&self, csv_features: &CsvKpiFeatures) -> Vec<f32> {
        let mut features = Vec::new();
        
        for feature_name in &self.feature_names {
            let value = match feature_name.as_str() {
                "sinr_pusch_avg" => csv_features.sinr_pusch_avg,
                "sinr_pucch_avg" => csv_features.sinr_pucch_avg,
                "ul_rssi_total" => csv_features.ul_rssi_total,
                "mac_dl_bler" => csv_features.mac_dl_bler,
                "mac_ul_bler" => csv_features.mac_ul_bler,
                "rrc_connected_users_avg" => csv_features.rrc_connected_users_avg,
                "ul_volume_pdcp_gbytes" => csv_features.ul_volume_pdcp_gbytes,
                "dl_volume_pdcp_gbytes" => csv_features.dl_volume_pdcp_gbytes,
                "volte_traffic_erl" => csv_features.volte_traffic_erl,
                "eric_traff_erab_erl" => csv_features.eric_traff_erab_erl,
                "ave_4g_lte_dl_user_thrput" => csv_features.ave_4g_lte_dl_user_thrput,
                "ave_4g_lte_ul_user_thrput" => csv_features.ave_4g_lte_ul_user_thrput,
                "dl_latency_avg" => csv_features.dl_latency_avg,
                "erab_drop_rate_qci_5" => csv_features.erab_drop_rate_qci_5,
                "lte_dcr_volte" => csv_features.lte_dcr_volte,
                "lte_intra_freq_ho_sr" => csv_features.lte_intra_freq_ho_sr,
                "lte_inter_freq_ho_sr" => csv_features.lte_inter_freq_ho_sr,
                "inter_freq_ho_attempts" => csv_features.inter_freq_ho_attempts,
                "intra_freq_ho_attempts" => csv_features.intra_freq_ho_attempts,
                "endc_establishment_att" => csv_features.endc_establishment_att,
                "endc_establishment_succ" => csv_features.endc_establishment_succ,
                "endc_setup_sr" => csv_features.endc_setup_sr,
                "endc_scg_failure_ratio" => csv_features.endc_scg_failure_ratio,
                "band_count" => csv_features.band_count,
                "active_ues_dl" => csv_features.active_ues_dl,
                "active_ues_ul" => csv_features.active_ues_ul,
                _ => 0.0, // Default for unknown features
            };
            features.push(value);
        }
        
        features
    }
    
    /// Train the model with CSV data
    pub fn train(&mut self, training_data: &[(CsvKpiFeatures, f32)]) -> Result<(), String> {
        if training_data.is_empty() {
            return Err("Training data cannot be empty".to_string());
        }
        
        let start_time = std::time::Instant::now();
        
        // Extract features and targets
        let mut feature_matrix = Vec::new();
        let mut targets = Vec::new();
        
        for (csv_features, target) in training_data {
            let features = self.extract_features(csv_features);
            feature_matrix.push(features);
            targets.push(*target);
        }
        
        // Calculate normalization parameters
        self.calculate_normalization_params(&feature_matrix)?;
        
        // Normalize features
        let normalized_features = self.normalize_features(&feature_matrix)?;
        
        // Train using gradient descent
        self.train_gradient_descent(&normalized_features, &targets)?;
        
        // Calculate feature importance
        self.calculate_feature_importance(&normalized_features, &targets)?;
        
        self.training_history.training_time_ms = start_time.elapsed().as_millis() as u64;
        self.is_trained = true;
        
        Ok(())
    }
    
    /// Calculate normalization parameters from training data
    fn calculate_normalization_params(&mut self, feature_matrix: &[Vec<f32>]) -> Result<(), String> {
        if feature_matrix.is_empty() {
            return Err("Feature matrix is empty".to_string());
        }
        
        let num_features = feature_matrix[0].len();
        let num_samples = feature_matrix.len() as f32;
        
        // Calculate means
        self.normalization_params.means = vec![0.0; num_features];
        for features in feature_matrix {
            for (i, &value) in features.iter().enumerate() {
                self.normalization_params.means[i] += value;
            }
        }
        for mean in &mut self.normalization_params.means {
            *mean /= num_samples;
        }
        
        // Calculate standard deviations
        self.normalization_params.std_devs = vec![0.0; num_features];
        for features in feature_matrix {
            for (i, &value) in features.iter().enumerate() {
                let diff = value - self.normalization_params.means[i];
                self.normalization_params.std_devs[i] += diff * diff;
            }
        }
        for std_dev in &mut self.normalization_params.std_devs {
            *std_dev = (*std_dev / num_samples).sqrt();
            if *std_dev == 0.0 {
                *std_dev = 1.0; // Prevent division by zero
            }
        }
        
        // Calculate min/max values
        self.normalization_params.min_vals = feature_matrix[0].clone();
        self.normalization_params.max_vals = feature_matrix[0].clone();
        
        for features in feature_matrix.iter().skip(1) {
            for (i, &value) in features.iter().enumerate() {
                self.normalization_params.min_vals[i] = self.normalization_params.min_vals[i].min(value);
                self.normalization_params.max_vals[i] = self.normalization_params.max_vals[i].max(value);
            }
        }
        
        Ok(())
    }
    
    /// Normalize features using calculated parameters
    fn normalize_features(&self, feature_matrix: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let mut normalized = Vec::new();
        
        for features in feature_matrix {
            let mut normalized_row = Vec::new();
            for (i, &value) in features.iter().enumerate() {
                let normalized_value = (value - self.normalization_params.means[i]) 
                    / self.normalization_params.std_devs[i];
                normalized_row.push(normalized_value);
            }
            normalized.push(normalized_row);
        }
        
        Ok(normalized)
    }
    
    /// Train using gradient descent
    fn train_gradient_descent(&mut self, features: &[Vec<f32>], targets: &[f32]) -> Result<(), String> {
        let learning_rate = 0.01;
        let epochs = 1000;
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let patience = 50;
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;
            
            // Forward pass and loss calculation
            for (features_row, &target) in features.iter().zip(targets.iter()) {
                let prediction = self.predict_normalized(features_row);
                let loss = (prediction - target).powi(2);
                total_loss += loss;
                
                // Update weights using gradient descent
                let error = target - prediction;
                for (i, &feature_val) in features_row.iter().enumerate() {
                    self.weights[i] += learning_rate * error * feature_val;
                }
                self.bias += learning_rate * error;
                
                // Check accuracy (within 10% for regression)
                if target != 0.0 && (prediction - target).abs() / target.abs() < 0.1 {
                    correct_predictions += 1;
                } else if target == 0.0 && prediction.abs() < 0.1 {
                    correct_predictions += 1;
                }
            }
            
            let avg_loss = total_loss / features.len() as f32;
            let accuracy = correct_predictions as f32 / features.len() as f32;
            
            self.training_history.epochs.push(epoch);
            self.training_history.losses.push(avg_loss);
            self.training_history.accuracies.push(accuracy);
            
            // Early stopping
            if avg_loss < best_loss {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= patience {
                    self.training_history.convergence_epoch = Some(epoch);
                    break;
                }
            }
            
            // Learning rate decay
            if epoch % 100 == 0 && epoch > 0 {
                // learning_rate *= 0.95;
            }
        }
        
        self.prediction_accuracy = self.training_history.accuracies.last().copied().unwrap_or(0.0);
        Ok(())
    }
    
    /// Calculate feature importance based on weight magnitudes
    fn calculate_feature_importance(&mut self, _features: &[Vec<f32>], _targets: &[f32]) -> Result<(), String> {
        let total_weight: f32 = self.weights.iter().map(|w| w.abs()).sum();
        
        if total_weight > 0.0 {
            for (i, feature_name) in self.feature_names.iter().enumerate() {
                let importance = self.weights[i].abs() / total_weight;
                self.feature_importance.insert(feature_name.clone(), importance);
            }
        }
        
        Ok(())
    }
    
    /// Make prediction with normalized features
    fn predict_normalized(&self, features: &[f32]) -> f32 {
        let weighted_sum: f32 = features.iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum::<f32>() + self.bias;
        
        // Apply appropriate activation based on KPI type
        match self.kpi_type {
            KpiType::Availability => (1.0 / (1.0 + (-weighted_sum).exp())) * 100.0, // Sigmoid scaled to percentage
            KpiType::ErrorRate => (1.0 / (1.0 + (-weighted_sum).exp())) * 20.0, // Error rates typically 0-20%
            KpiType::HandoverSuccess => (1.0 / (1.0 + (-weighted_sum).exp())) * 100.0, // Success rate percentage
            KpiType::EndcEstablishment => (1.0 / (1.0 + (-weighted_sum).exp())) * 100.0, // Success rate percentage
            _ => weighted_sum.max(0.0), // ReLU for throughput, latency, etc.
        }
    }
    
    /// Make prediction with comprehensive output
    pub fn predict(&self, csv_features: &CsvKpiFeatures) -> Result<KpiPrediction, String> {
        if !self.is_trained {
            return Err("Model must be trained before making predictions".to_string());
        }
        
        let features = self.extract_features(csv_features);
        let normalized_features = self.normalize_single_sample(&features)?;
        let predicted_value = self.predict_normalized(&normalized_features);
        
        // Calculate confidence based on feature values and training history
        let confidence = self.calculate_confidence(&normalized_features);
        
        // Calculate prediction interval (simplified)
        let uncertainty = (1.0 - confidence) * predicted_value * 0.2;
        let prediction_interval = (
            (predicted_value - uncertainty).max(0.0),
            predicted_value + uncertainty
        );
        
        // Calculate feature contributions
        let contributing_features = self.calculate_feature_contributions(&features, &normalized_features);
        
        // Generate risk factors and recommendations
        let risk_factors = self.identify_risk_factors(csv_features);
        let recommendations = self.generate_recommendations(csv_features, predicted_value);
        
        Ok(KpiPrediction {
            kpi_type: self.kpi_type.clone(),
            predicted_value,
            confidence,
            prediction_interval,
            contributing_features,
            risk_factors,
            recommendations,
        })
    }
    
    /// Normalize a single sample for prediction
    fn normalize_single_sample(&self, features: &[f32]) -> Result<Vec<f32>, String> {
        if features.len() != self.normalization_params.means.len() {
            return Err("Feature count mismatch".to_string());
        }
        
        let mut normalized = Vec::new();
        for (i, &value) in features.iter().enumerate() {
            let normalized_value = (value - self.normalization_params.means[i]) 
                / self.normalization_params.std_devs[i];
            normalized.push(normalized_value);
        }
        
        Ok(normalized)
    }
    
    /// Calculate prediction confidence
    fn calculate_confidence(&self, _normalized_features: &[f32]) -> f32 {
        // Simplified confidence calculation based on training accuracy
        self.prediction_accuracy * 0.8 + 0.2 // Base confidence of 20%
    }
    
    /// Calculate feature contributions to prediction
    fn calculate_feature_contributions(&self, features: &[f32], normalized_features: &[f32]) -> Vec<FeatureContribution> {
        let mut contributions = Vec::new();
        
        for (i, feature_name) in self.feature_names.iter().enumerate() {
            let contribution = normalized_features[i] * self.weights[i];
            let importance = self.feature_importance.get(feature_name).copied().unwrap_or(0.0);
            
            contributions.push(FeatureContribution {
                feature_name: feature_name.clone(),
                value: features[i],
                contribution,
                importance,
            });
        }
        
        // Sort by absolute contribution
        contributions.sort_by(|a, b| b.contribution.abs().partial_cmp(&a.contribution.abs()).unwrap());
        contributions
    }
    
    /// Identify risk factors based on feature values
    fn identify_risk_factors(&self, csv_features: &CsvKpiFeatures) -> Vec<String> {
        let mut risks = Vec::new();
        
        match self.kpi_type {
            KpiType::Availability => {
                if csv_features.mac_dl_bler > 10.0 {
                    risks.push("High downlink error rate detected".to_string());
                }
                if csv_features.sinr_pusch_avg < 5.0 {
                    risks.push("Poor uplink signal quality".to_string());
                }
                if csv_features.erab_drop_rate_qci_5 > 3.0 {
                    risks.push("High bearer drop rate".to_string());
                }
            },
            KpiType::Throughput => {
                if csv_features.sinr_pusch_avg < 10.0 {
                    risks.push("Low SINR may limit throughput".to_string());
                }
                if csv_features.rrc_connected_users_avg > 200.0 {
                    risks.push("High user load may impact throughput".to_string());
                }
                if csv_features.mac_dl_bler > 5.0 {
                    risks.push("Error rate affecting throughput".to_string());
                }
            },
            KpiType::HandoverSuccess => {
                if csv_features.sinr_pusch_avg < 8.0 {
                    risks.push("Poor signal quality may cause handover failures".to_string());
                }
                if csv_features.inter_freq_ho_attempts > 1000.0 {
                    risks.push("High handover attempt rate".to_string());
                }
            },
            _ => {}
        }
        
        risks
    }
    
    /// Generate recommendations based on prediction and features
    fn generate_recommendations(&self, csv_features: &CsvKpiFeatures, predicted_value: f32) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match self.kpi_type {
            KpiType::Availability => {
                if predicted_value < 95.0 {
                    recommendations.push("Consider increasing power levels to improve coverage".to_string());
                    recommendations.push("Check for hardware issues or interference".to_string());
                }
                if csv_features.erab_drop_rate_qci_5 > 2.0 {
                    recommendations.push("Optimize handover parameters".to_string());
                }
            },
            KpiType::Throughput => {
                if predicted_value < 20.0 {
                    recommendations.push("Consider load balancing to adjacent cells".to_string());
                    recommendations.push("Optimize MIMO configuration".to_string());
                }
                if csv_features.mac_dl_bler > 3.0 {
                    recommendations.push("Adjust modulation and coding scheme".to_string());
                }
            },
            KpiType::Latency => {
                if predicted_value > 30.0 {
                    recommendations.push("Optimize transport network configuration".to_string());
                    recommendations.push("Check for processing delays".to_string());
                }
            },
            _ => {}
        }
        
        recommendations
    }
    
    /// Get model performance metrics
    pub fn get_performance_metrics(&self) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), self.prediction_accuracy);
        metrics.insert("training_time_ms".to_string(), self.training_history.training_time_ms as f32);
        
        if let Some(last_loss) = self.training_history.losses.last() {
            metrics.insert("final_loss".to_string(), *last_loss);
        }
        
        if let Some(convergence_epoch) = self.training_history.convergence_epoch {
            metrics.insert("convergence_epoch".to_string(), convergence_epoch as f32);
        }
        
        metrics
    }
    
    /// Save model to file
    pub fn save_model(&self, file_path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Serialization error: {}", e))?;
        
        std::fs::write(file_path, json)
            .map_err(|e| format!("File write error: {}", e))?;
        
        Ok(())
    }
    
    /// Load model from file
    pub fn load_model(file_path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(file_path)
            .map_err(|e| format!("File read error: {}", e))?;
        
        let model: Self = serde_json::from_str(&json)
            .map_err(|e| format!("Deserialization error: {}", e))?;
        
        Ok(model)
    }
}

impl Default for KpiPredictor {
    fn default() -> Self {
        Self::new(KpiType::Availability, "default_kpi_predictor".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kpi_predictor_creation() {
        let predictor = KpiPredictor::new(KpiType::Throughput, "test_model".to_string());
        assert_eq!(predictor.kpi_type, KpiType::Throughput);
        assert_eq!(predictor.model_id, "test_model");
        assert!(!predictor.is_trained);
    }
    
    #[test]
    fn test_feature_extraction() {
        let predictor = KpiPredictor::new(KpiType::Availability, "test".to_string());
        let csv_features = create_test_csv_features();
        
        let features = predictor.extract_features(&csv_features);
        assert!(!features.is_empty());
        assert_eq!(features.len(), predictor.feature_names.len());
    }
    
    #[test]
    fn test_normalization() {
        let mut predictor = KpiPredictor::new(KpiType::Throughput, "test".to_string());
        let feature_matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        assert!(predictor.calculate_normalization_params(&feature_matrix).is_ok());
        assert!(predictor.normalize_features(&feature_matrix).is_ok());
    }
    
    fn create_test_csv_features() -> CsvKpiFeatures {
        CsvKpiFeatures {
            sinr_pusch_avg: 15.0,
            sinr_pucch_avg: 14.0,
            ul_rssi_total: -105.0,
            mac_dl_bler: 2.0,
            mac_ul_bler: 1.5,
            rrc_connected_users_avg: 50.0,
            ul_volume_pdcp_gbytes: 1.0,
            dl_volume_pdcp_gbytes: 5.0,
            volte_traffic_erl: 1.0,
            eric_traff_erab_erl: 10.0,
            ave_4g_lte_dl_user_thrput: 50.0,
            ave_4g_lte_ul_user_thrput: 25.0,
            dl_latency_avg: 15.0,
            erab_drop_rate_qci_5: 1.0,
            lte_dcr_volte: 0.5,
            lte_intra_freq_ho_sr: 95.0,
            lte_inter_freq_ho_sr: 93.0,
            inter_freq_ho_attempts: 50.0,
            intra_freq_ho_attempts: 100.0,
            endc_establishment_att: 100.0,
            endc_establishment_succ: 95.0,
            endc_setup_sr: 95.0,
            endc_scg_failure_ratio: 2.0,
            frequency_band: "LTE1800".to_string(),
            band_count: 4.0,
            active_ues_dl: 20.0,
            active_ues_ul: 15.0,
        }
    }
}