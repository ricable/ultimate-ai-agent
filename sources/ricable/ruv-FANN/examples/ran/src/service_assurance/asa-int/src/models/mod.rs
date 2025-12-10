use crate::{Config, Result, Error};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

pub mod ensemble;
pub mod neural_network;
pub mod svm;
pub mod random_forest;

pub use ensemble::EnsembleClassifier;
pub use neural_network::NeuralNetworkClassifier;
pub use svm::SvmClassifier;
pub use random_forest::RandomForestClassifier;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: Array2<i32>,
    pub class_metrics: HashMap<String, ClassMetrics>,
    pub training_time_seconds: f64,
    pub prediction_time_ms: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub predicted_class: String,
    pub confidence: f64,
    pub class_probabilities: HashMap<String, f64>,
    pub feature_importance: Option<Array1<f64>>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub model_id: String,
    pub metrics: ModelMetrics,
    pub epochs_trained: u32,
    pub convergence_achieved: bool,
    pub final_loss: f64,
    pub validation_metrics: ModelMetrics,
}

pub trait InterferenceClassifier: Send + Sync {
    fn train(&mut self, features: &Array2<f64>, labels: &[String]) -> Result<TrainingResult>;
    fn predict(&self, features: &Array1<f64>) -> Result<ClassificationResult>;
    fn predict_proba(&self, features: &Array1<f64>) -> Result<HashMap<String, f64>>;
    fn get_feature_importance(&self) -> Option<Array1<f64>>;
    fn get_metrics(&self) -> &ModelMetrics;
    fn validate_performance(&self, config: &Config) -> Result<bool>;
    fn save_model(&self, path: &str) -> Result<()>;
    fn load_model(&mut self, path: &str) -> Result<()>;
}

pub struct InterferenceClassifierFactory;

impl InterferenceClassifierFactory {
    pub fn create_classifier(config: &Config) -> Result<Box<dyn InterferenceClassifier>> {
        match config.model.architecture.as_str() {
            "ensemble" => Ok(Box::new(EnsembleClassifier::new(config)?)),
            "neural_network" => Ok(Box::new(NeuralNetworkClassifier::new(config)?)),
            "svm" => Ok(Box::new(SvmClassifier::new(config)?)),
            "random_forest" => Ok(Box::new(RandomForestClassifier::new(config)?)),
            _ => Err(Error::InvalidInput(format!(
                "Unknown model architecture: {}",
                config.model.architecture
            ))),
        }
    }
}

pub fn validate_accuracy_requirement(metrics: &ModelMetrics) -> Result<()> {
    const REQUIRED_ACCURACY: f64 = 0.95;
    
    if metrics.accuracy < REQUIRED_ACCURACY {
        return Err(Error::PerformanceThreshold {
            expected: REQUIRED_ACCURACY,
            actual: metrics.accuracy,
        });
    }
    
    Ok(())
}

pub fn compute_metrics(
    y_true: &[String],
    y_pred: &[String],
    classes: &[String],
    training_time: f64,
    prediction_time: f64,
) -> Result<ModelMetrics> {
    if y_true.len() != y_pred.len() {
        return Err(Error::InvalidInput("Length mismatch between true and predicted labels".to_string()));
    }
    
    let n_classes = classes.len();
    let mut confusion_matrix = Array2::zeros((n_classes, n_classes));
    
    // Build confusion matrix
    for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
        let true_idx = classes.iter().position(|x| x == true_label).unwrap_or(0);
        let pred_idx = classes.iter().position(|x| x == pred_label).unwrap_or(0);
        confusion_matrix[[true_idx, pred_idx]] += 1;
    }
    
    // Compute overall metrics
    let mut correct = 0;
    let total = y_true.len();
    
    for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
        if true_label == pred_label {
            correct += 1;
        }
    }
    
    let accuracy = correct as f64 / total as f64;
    
    // Compute per-class metrics
    let mut class_metrics = HashMap::new();
    let mut macro_precision = 0.0;
    let mut macro_recall = 0.0;
    let mut macro_f1 = 0.0;
    
    for (i, class) in classes.iter().enumerate() {
        let tp = confusion_matrix[[i, i]];
        let fp: i32 = confusion_matrix.column(i).sum() - tp;
        let fn_: i32 = confusion_matrix.row(i).sum() - tp;
        
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        
        class_metrics.insert(class.clone(), ClassMetrics {
            precision,
            recall,
            f1_score,
            support: tp + fn_,
        });
        
        macro_precision += precision;
        macro_recall += recall;
        macro_f1 += f1_score;
    }
    
    macro_precision /= n_classes as f64;
    macro_recall /= n_classes as f64;
    macro_f1 /= n_classes as f64;
    
    Ok(ModelMetrics {
        accuracy,
        precision: macro_precision,
        recall: macro_recall,
        f1_score: macro_f1,
        confusion_matrix,
        class_metrics,
        training_time_seconds: training_time,
        prediction_time_ms: prediction_time,
        last_updated: Utc::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_compute_metrics() {
        let y_true = vec!["A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()];
        let y_pred = vec!["A".to_string(), "B".to_string(), "A".to_string(), "A".to_string()];
        let classes = vec!["A".to_string(), "B".to_string()];
        
        let metrics = compute_metrics(&y_true, &y_pred, &classes, 10.0, 1.0).unwrap();
        
        assert_eq!(metrics.accuracy, 0.75);
        assert!(metrics.precision > 0.0);
        assert!(metrics.recall > 0.0);
        assert!(metrics.f1_score > 0.0);
    }
    
    #[test]
    fn test_accuracy_requirement() {
        let mut metrics = ModelMetrics {
            accuracy: 0.96,
            precision: 0.95,
            recall: 0.95,
            f1_score: 0.95,
            confusion_matrix: Array2::zeros((2, 2)),
            class_metrics: HashMap::new(),
            training_time_seconds: 10.0,
            prediction_time_ms: 1.0,
            last_updated: Utc::now(),
        };
        
        assert!(validate_accuracy_requirement(&metrics).is_ok());
        
        metrics.accuracy = 0.94;
        assert!(validate_accuracy_requirement(&metrics).is_err());
    }
}