use crate::{Config, Result, Error};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndcFailurePredictor {
    model_type: String,
    weights: Option<Array2<f64>>,
    bias: Option<Array1<f64>>,
    feature_importance: Option<Array1<f64>>,
    classes: Vec<String>,
    metrics: PredictionMetrics,
    training_config: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_score: f64,
    pub confusion_matrix: Array2<i32>,
    pub training_time_seconds: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub max_iterations: u32,
    pub regularization: f64,
    pub validation_split: f64,
    pub early_stopping_patience: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePredictionResult {
    pub failure_probability: f64,
    pub predicted_failure_type: String,
    pub confidence: f64,
    pub contributing_factors: Vec<String>,
    pub risk_level: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndcFeatures {
    pub signal_quality_score: f64,
    pub network_congestion_score: f64,
    pub ue_capability_score: f64,
    pub bearer_config_score: f64,
    pub historical_success_rate: f64,
    pub load_indicators: Vec<f64>,
    pub interference_level: f64,
    pub handover_success_rate: f64,
}

impl EndcFailurePredictor {
    pub fn new(config: &Config) -> Result<Self> {
        let classes = config.endc.supported_failure_types.clone();
        
        Ok(Self {
            model_type: config.prediction.model_type.clone(),
            weights: None,
            bias: None,
            feature_importance: None,
            classes,
            metrics: PredictionMetrics {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                auc_score: 0.0,
                confusion_matrix: Array2::zeros((4, 4)), // 4 failure types
                training_time_seconds: 0.0,
                last_updated: Utc::now(),
            },
            training_config: TrainingConfig {
                learning_rate: 0.01,
                max_iterations: 1000,
                regularization: 0.01,
                validation_split: 0.2,
                early_stopping_patience: 10,
            },
        })
    }
    
    pub fn train(&mut self, features: &Array2<f64>, labels: &[String]) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        if features.nrows() != labels.len() {
            return Err(Error::InvalidInput("Feature matrix and labels length mismatch".to_string()));
        }
        
        // Simplified gradient boosting implementation
        // In production, use proper libraries like LightGBM or XGBoost
        
        let n_features = features.ncols();
        let n_classes = self.classes.len();
        
        // Initialize weights and bias
        self.weights = Some(Array2::zeros((n_classes, n_features)));
        self.bias = Some(Array1::zeros(n_classes));
        
        // Simple training loop (logistic regression as approximation)
        let mut weights = self.weights.as_mut().unwrap();
        let mut bias = self.bias.as_mut().unwrap();
        
        for iteration in 0..self.training_config.max_iterations {
            let mut total_loss = 0.0;
            
            for (i, (feature_row, label)) in features.rows().into_iter().zip(labels.iter()).enumerate() {
                let class_idx = self.classes.iter().position(|c| c == label).unwrap_or(0);
                
                // Forward pass
                let logits = weights.dot(&feature_row.to_owned()) + &*bias;
                let probabilities = self.softmax(&logits);
                
                // Compute loss (cross-entropy)
                let loss = -probabilities[class_idx].ln();
                total_loss += loss;
                
                // Backward pass (simplified gradient computation)
                for (j, &prob) in probabilities.iter().enumerate() {
                    let target = if j == class_idx { 1.0 } else { 0.0 };
                    let gradient = prob - target;
                    
                    // Update weights
                    for (k, &feature_val) in feature_row.iter().enumerate() {
                        weights[[j, k]] -= self.training_config.learning_rate * gradient * feature_val;
                    }
                    
                    // Update bias
                    bias[j] -= self.training_config.learning_rate * gradient;
                }
            }
            
            // Check convergence
            let avg_loss = total_loss / features.nrows() as f64;
            if avg_loss < 0.01 {
                break;
            }
        }
        
        let training_time = start_time.elapsed().as_secs_f64();
        
        // Evaluate model
        let mut correct_predictions = 0;
        let mut predictions = Vec::new();
        
        for (feature_row, true_label) in features.rows().into_iter().zip(labels.iter()) {
            let result = self.predict_internal(&feature_row.to_owned())?;
            predictions.push(result.predicted_failure_type.clone());
            
            if result.predicted_failure_type == *true_label {
                correct_predictions += 1;
            }
        }
        
        let accuracy = correct_predictions as f64 / labels.len() as f64;
        
        // Validate accuracy requirement
        if accuracy < 0.80 {
            return Err(Error::PerformanceThreshold {
                expected: 0.80,
                actual: accuracy,
            });
        }
        
        // Update metrics
        self.metrics.accuracy = accuracy;
        self.metrics.precision = accuracy; // Simplified
        self.metrics.recall = accuracy; // Simplified
        self.metrics.f1_score = accuracy; // Simplified
        self.metrics.auc_score = accuracy; // Simplified
        self.metrics.training_time_seconds = training_time;
        self.metrics.last_updated = Utc::now();
        
        // Calculate feature importance (simplified)
        let importance = Array1::from_vec(
            (0..n_features).map(|i| {
                weights.column(i).iter().map(|&w| w.abs()).sum::<f64>()
            }).collect()
        );
        let importance_sum = importance.sum();
        if importance_sum > 0.0 {
            self.feature_importance = Some(importance / importance_sum);
        }
        
        Ok(())
    }
    
    pub fn predict(&self, features: &EndcFeatures) -> Result<FailurePredictionResult> {
        let feature_vector = self.extract_feature_vector(features);
        self.predict_internal(&feature_vector)
    }
    
    fn predict_internal(&self, features: &Array1<f64>) -> Result<FailurePredictionResult> {
        let weights = self.weights.as_ref()
            .ok_or_else(|| Error::ModelTraining("Model not trained yet".to_string()))?;
        let bias = self.bias.as_ref()
            .ok_or_else(|| Error::ModelTraining("Model not trained yet".to_string()))?;
        
        // Forward pass
        let logits = weights.dot(features) + bias;
        let probabilities = self.softmax(&logits);
        
        // Find predicted class
        let (max_idx, max_prob) = probabilities.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let predicted_failure_type = self.classes[max_idx].clone();
        let failure_probability = *max_prob;
        let confidence = *max_prob;
        
        // Determine risk level
        let risk_level = if failure_probability > 0.8 {
            "CRITICAL"
        } else if failure_probability > 0.6 {
            "HIGH"
        } else if failure_probability > 0.4 {
            "MEDIUM"
        } else {
            "LOW"
        }.to_string();
        
        // Generate contributing factors based on feature importance
        let contributing_factors = self.analyze_contributing_factors(features);
        
        Ok(FailurePredictionResult {
            failure_probability,
            predicted_failure_type,
            confidence,
            contributing_factors,
            risk_level,
            timestamp: Utc::now(),
        })
    }
    
    fn extract_feature_vector(&self, features: &EndcFeatures) -> Array1<f64> {
        let mut vector = Vec::new();
        
        // Signal quality features
        vector.push(features.signal_quality_score);
        vector.push(features.network_congestion_score);
        vector.push(features.ue_capability_score);
        vector.push(features.bearer_config_score);
        vector.push(features.historical_success_rate);
        vector.push(features.interference_level);
        vector.push(features.handover_success_rate);
        
        // Load indicators
        vector.extend(&features.load_indicators);
        
        // Additional derived features
        vector.push(features.signal_quality_score * features.ue_capability_score); // Interaction
        vector.push(1.0 - features.network_congestion_score); // Inverted congestion
        vector.push(features.historical_success_rate.powi(2)); // Squared success rate
        
        Array1::from_vec(vector)
    }
    
    fn softmax(&self, logits: &Array1<f64>) -> Array1<f64> {
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits = logits.map(|&x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        exp_logits.map(|&x| x / sum_exp)
    }
    
    fn analyze_contributing_factors(&self, features: &Array1<f64>) -> Vec<String> {
        let mut factors = Vec::new();
        
        if let Some(importance) = &self.feature_importance {
            let feature_names = vec![
                "Signal Quality",
                "Network Congestion",
                "UE Capability",
                "Bearer Configuration",
                "Historical Success Rate",
                "Interference Level",
                "Handover Success Rate",
            ];
            
            // Find top contributing factors
            let mut indexed_importance: Vec<(usize, f64)> = importance.iter()
                .enumerate()
                .take(feature_names.len())
                .map(|(i, &imp)| (i, imp * features[i].abs()))
                .collect();
            
            indexed_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            for (i, _) in indexed_importance.iter().take(3) {
                if *i < feature_names.len() {
                    factors.push(feature_names[*i].to_string());
                }
            }
        }
        
        if factors.is_empty() {
            factors.push("Insufficient feature analysis data".to_string());
        }
        
        factors
    }
    
    pub fn get_metrics(&self) -> &PredictionMetrics {
        &self.metrics
    }
    
    pub fn get_feature_importance(&self) -> Option<&Array1<f64>> {
        self.feature_importance.as_ref()
    }
    
    pub fn validate_performance(&self, config: &Config) -> Result<bool> {
        Ok(self.metrics.accuracy >= config.performance.min_prediction_accuracy)
    }
    
    pub fn save_model(&self, path: &str) -> Result<()> {
        let model_data = serde_json::to_string_pretty(self)?;
        std::fs::write(format!("{}/endc_predictor.json", path), model_data)?;
        Ok(())
    }
    
    pub fn load_model(&mut self, path: &str) -> Result<()> {
        let model_data = std::fs::read_to_string(format!("{}/endc_predictor.json", path))?;
        let model: EndcFailurePredictor = serde_json::from_str(&model_data)?;
        *self = model;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    
    #[test]
    fn test_endc_predictor_creation() {
        let config = Config::default();
        let predictor = EndcFailurePredictor::new(&config).unwrap();
        
        assert_eq!(predictor.model_type, "gradient_boosting");
        assert_eq!(predictor.classes.len(), 4);
    }
    
    #[test]
    fn test_feature_extraction() {
        let config = Config::default();
        let predictor = EndcFailurePredictor::new(&config).unwrap();
        
        let features = EndcFeatures {
            signal_quality_score: 0.8,
            network_congestion_score: 0.3,
            ue_capability_score: 0.9,
            bearer_config_score: 0.85,
            historical_success_rate: 0.95,
            load_indicators: vec![0.6, 0.7, 0.5],
            interference_level: 0.2,
            handover_success_rate: 0.92,
        };
        
        let feature_vector = predictor.extract_feature_vector(&features);
        assert!(feature_vector.len() > 8); // Should include derived features
    }
    
    #[test]
    fn test_softmax() {
        let config = Config::default();
        let predictor = EndcFailurePredictor::new(&config).unwrap();
        
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probabilities = predictor.softmax(&logits);
        
        // Probabilities should sum to 1
        let sum: f64 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Highest logit should have highest probability
        assert!(probabilities[2] > probabilities[1]);
        assert!(probabilities[1] > probabilities[0]);
    }
}