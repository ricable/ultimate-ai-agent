use super::*;
use crate::{Config, Result, Error};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvmClassifier {
    classes: Vec<String>,
    metrics: ModelMetrics,
    support_vectors: Option<Array2<f64>>,
    alphas: Option<Array1<f64>>,
    bias: f64,
    kernel: String,
    c_parameter: f64,
    gamma: f64,
}

impl SvmClassifier {
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            classes: config.classification.interference_classes.clone(),
            metrics: ModelMetrics {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                confusion_matrix: Array2::zeros((config.classification.interference_classes.len(), config.classification.interference_classes.len())),
                class_metrics: HashMap::new(),
                training_time_seconds: 0.0,
                prediction_time_ms: 0.0,
                last_updated: Utc::now(),
            },
            support_vectors: None,
            alphas: None,
            bias: 0.0,
            kernel: "rbf".to_string(),
            c_parameter: 1.0,
            gamma: 0.1,
        })
    }
    
    fn kernel_function(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        match self.kernel.as_str() {
            "linear" => x1.dot(x2),
            "rbf" => {
                let diff = x1 - x2;
                let squared_distance = diff.dot(&diff);
                (-self.gamma * squared_distance).exp()
            }
            "polynomial" => {
                let dot_product = x1.dot(x2);
                (self.gamma * dot_product + 1.0).powi(3)
            }
            _ => x1.dot(x2), // Default to linear
        }
    }
}

impl InterferenceClassifier for SvmClassifier {
    fn train(&mut self, features: &Array2<f64>, labels: &[String]) -> Result<TrainingResult> {
        let start_time = std::time::Instant::now();
        
        if features.nrows() != labels.len() {
            return Err(Error::InvalidInput("Feature matrix and labels length mismatch".to_string()));
        }
        
        // This is a simplified SVM implementation
        // In a real implementation, you would use SMO algorithm or existing library
        
        // For now, create a simple placeholder model
        self.support_vectors = Some(features.clone());
        self.alphas = Some(Array1::ones(features.nrows()));
        self.bias = 0.0;
        
        let training_time = start_time.elapsed().as_secs_f64();
        
        // Evaluate on training data
        let mut predictions = Vec::new();
        for row in features.rows() {
            let result = self.predict(&row.to_owned())?;
            predictions.push(result.predicted_class);
        }
        
        let metrics = compute_metrics(labels, &predictions, &self.classes, training_time, 0.0)?;
        
        // Validate accuracy requirement
        validate_accuracy_requirement(&metrics)?;
        
        self.metrics = metrics.clone();
        
        Ok(TrainingResult {
            model_id: format!("svm_{}", Utc::now().timestamp()),
            metrics: metrics.clone(),
            epochs_trained: 1,
            convergence_achieved: true,
            final_loss: 0.0,
            validation_metrics: metrics,
        })
    }
    
    fn predict(&self, features: &Array1<f64>) -> Result<ClassificationResult> {
        let start_time = std::time::Instant::now();
        
        if self.support_vectors.is_none() || self.alphas.is_none() {
            return Err(Error::ModelTraining("Model not trained yet".to_string()));
        }
        
        let support_vectors = self.support_vectors.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();
        
        // Compute decision function
        let mut decision_values = Vec::new();
        for class_idx in 0..self.classes.len() {
            let mut decision_value = self.bias;
            for (i, sv) in support_vectors.rows().into_iter().enumerate() {
                let kernel_value = self.kernel_function(&sv.to_owned(), features);
                decision_value += alphas[i] * kernel_value;
            }
            decision_values.push(decision_value);
        }
        
        // Find predicted class
        let (max_idx, max_value) = decision_values.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let predicted_class = self.classes[max_idx].clone();
        
        // Convert decision values to probabilities (simplified)
        let exp_values: Vec<f64> = decision_values.iter().map(|&x| x.exp()).collect();
        let sum_exp: f64 = exp_values.iter().sum();
        let probabilities: Vec<f64> = exp_values.iter().map(|&x| x / sum_exp).collect();
        
        let confidence = probabilities[max_idx];
        
        // Build class probabilities
        let mut class_probabilities = HashMap::new();
        for (i, class) in self.classes.iter().enumerate() {
            class_probabilities.insert(class.clone(), probabilities[i]);
        }
        
        let prediction_time = start_time.elapsed().as_millis() as f64;
        
        Ok(ClassificationResult {
            predicted_class,
            confidence,
            class_probabilities,
            feature_importance: None,
            timestamp: Utc::now(),
        })
    }
    
    fn predict_proba(&self, features: &Array1<f64>) -> Result<HashMap<String, f64>> {
        let result = self.predict(features)?;
        Ok(result.class_probabilities)
    }
    
    fn get_feature_importance(&self) -> Option<Array1<f64>> {
        // SVM doesn't provide direct feature importance
        None
    }
    
    fn get_metrics(&self) -> &ModelMetrics {
        &self.metrics
    }
    
    fn validate_performance(&self, config: &Config) -> Result<bool> {
        let metrics = &self.metrics;
        
        if metrics.accuracy < config.performance.min_accuracy {
            return Ok(false);
        }
        
        if metrics.precision < config.performance.min_precision {
            return Ok(false);
        }
        
        if metrics.recall < config.performance.min_recall {
            return Ok(false);
        }
        
        if metrics.f1_score < config.performance.min_f1_score {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    fn save_model(&self, path: &str) -> Result<()> {
        let model_data = serde_json::to_string_pretty(self)?;
        std::fs::write(format!("{}/svm.json", path), model_data)?;
        Ok(())
    }
    
    fn load_model(&mut self, path: &str) -> Result<()> {
        let model_data = std::fs::read_to_string(format!("{}/svm.json", path))?;
        let model: SvmClassifier = serde_json::from_str(&model_data)?;
        *self = model;
        Ok(())
    }
}