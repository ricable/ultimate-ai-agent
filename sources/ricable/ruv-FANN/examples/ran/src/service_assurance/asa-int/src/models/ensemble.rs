use super::*;
use crate::{Config, Result, Error};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleClassifier {
    models: Vec<Box<dyn InterferenceClassifier>>,
    weights: Vec<f64>,
    voting_strategy: VotingStrategy,
    classes: Vec<String>,
    metrics: ModelMetrics,
    config: Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    Majority,
    Weighted,
    Soft,
}

impl EnsembleClassifier {
    pub fn new(config: &Config) -> Result<Self> {
        let mut models = Vec::new();
        let mut weights = Vec::new();
        
        // Create base models according to configuration
        for model_type in &config.model.ensemble_models {
            let mut model_config = config.clone();
            model_config.model.architecture = model_type.clone();
            
            match model_type.as_str() {
                "random_forest" => {
                    models.push(Box::new(super::RandomForestClassifier::new(&model_config)?));
                    weights.push(1.0);
                }
                "svm" => {
                    models.push(Box::new(super::SvmClassifier::new(&model_config)?));
                    weights.push(1.0);
                }
                "neural_network" => {
                    models.push(Box::new(super::NeuralNetworkClassifier::new(&model_config)?));
                    weights.push(1.0);
                }
                _ => return Err(Error::InvalidInput(format!("Unknown model type: {}", model_type))),
            }
        }
        
        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 0.0 {
            for weight in &mut weights {
                *weight /= weight_sum;
            }
        }
        
        Ok(Self {
            models,
            weights,
            voting_strategy: VotingStrategy::Weighted,
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
            config: config.clone(),
        })
    }
    
    fn aggregate_predictions(&self, predictions: &[ClassificationResult]) -> Result<ClassificationResult> {
        if predictions.is_empty() {
            return Err(Error::InvalidInput("No predictions to aggregate".to_string()));
        }
        
        match self.voting_strategy {
            VotingStrategy::Majority => self.majority_voting(predictions),
            VotingStrategy::Weighted => self.weighted_voting(predictions),
            VotingStrategy::Soft => self.soft_voting(predictions),
        }
    }
    
    fn majority_voting(&self, predictions: &[ClassificationResult]) -> Result<ClassificationResult> {
        let mut class_votes = HashMap::new();
        
        for prediction in predictions {
            *class_votes.entry(prediction.predicted_class.clone()).or_insert(0) += 1;
        }
        
        let (predicted_class, _) = class_votes.into_iter()
            .max_by_key(|(_, count)| *count)
            .ok_or_else(|| Error::Classification("No majority vote found".to_string()))?;
        
        // Calculate confidence as the proportion of models that voted for this class
        let confidence = predictions.iter()
            .filter(|p| p.predicted_class == predicted_class)
            .count() as f64 / predictions.len() as f64;
        
        // Aggregate class probabilities
        let mut class_probabilities = HashMap::new();
        for class in &self.classes {
            let avg_prob = predictions.iter()
                .filter_map(|p| p.class_probabilities.get(class))
                .sum::<f64>() / predictions.len() as f64;
            class_probabilities.insert(class.clone(), avg_prob);
        }
        
        Ok(ClassificationResult {
            predicted_class,
            confidence,
            class_probabilities,
            feature_importance: None,
            timestamp: Utc::now(),
        })
    }
    
    fn weighted_voting(&self, predictions: &[ClassificationResult]) -> Result<ClassificationResult> {
        let mut weighted_probabilities = HashMap::new();
        
        for class in &self.classes {
            let mut weighted_prob = 0.0;
            for (i, prediction) in predictions.iter().enumerate() {
                let prob = prediction.class_probabilities.get(class).unwrap_or(&0.0);
                weighted_prob += prob * self.weights[i];
            }
            weighted_probabilities.insert(class.clone(), weighted_prob);
        }
        
        let (predicted_class, confidence) = weighted_probabilities.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(class, prob)| (class.clone(), *prob))
            .ok_or_else(|| Error::Classification("No weighted vote found".to_string()))?;
        
        Ok(ClassificationResult {
            predicted_class,
            confidence,
            class_probabilities: weighted_probabilities,
            feature_importance: None,
            timestamp: Utc::now(),
        })
    }
    
    fn soft_voting(&self, predictions: &[ClassificationResult]) -> Result<ClassificationResult> {
        let mut soft_probabilities = HashMap::new();
        
        for class in &self.classes {
            let mut soft_prob = 0.0;
            let mut total_confidence = 0.0;
            
            for (i, prediction) in predictions.iter().enumerate() {
                let prob = prediction.class_probabilities.get(class).unwrap_or(&0.0);
                let weight = self.weights[i] * prediction.confidence;
                soft_prob += prob * weight;
                total_confidence += weight;
            }
            
            if total_confidence > 0.0 {
                soft_probabilities.insert(class.clone(), soft_prob / total_confidence);
            } else {
                soft_probabilities.insert(class.clone(), 0.0);
            }
        }
        
        let (predicted_class, confidence) = soft_probabilities.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(class, prob)| (class.clone(), *prob))
            .ok_or_else(|| Error::Classification("No soft vote found".to_string()))?;
        
        Ok(ClassificationResult {
            predicted_class,
            confidence,
            class_probabilities: soft_probabilities,
            feature_importance: None,
            timestamp: Utc::now(),
        })
    }
    
    fn update_weights_based_on_performance(&mut self, validation_results: &[ModelMetrics]) -> Result<()> {
        if validation_results.len() != self.models.len() {
            return Err(Error::InvalidInput("Validation results count mismatch".to_string()));
        }
        
        // Update weights based on validation accuracy
        let mut new_weights = Vec::new();
        for metrics in validation_results {
            // Weight based on accuracy with exponential emphasis
            let weight = metrics.accuracy.powf(2.0);
            new_weights.push(weight);
        }
        
        // Normalize weights
        let weight_sum: f64 = new_weights.iter().sum();
        if weight_sum > 0.0 {
            for weight in &mut new_weights {
                *weight /= weight_sum;
            }
        }
        
        self.weights = new_weights;
        Ok(())
    }
}

impl InterferenceClassifier for EnsembleClassifier {
    fn train(&mut self, features: &Array2<f64>, labels: &[String]) -> Result<TrainingResult> {
        let start_time = std::time::Instant::now();
        
        if features.nrows() != labels.len() {
            return Err(Error::InvalidInput("Feature matrix and labels length mismatch".to_string()));
        }
        
        // Train each model in the ensemble
        let mut training_results = Vec::new();
        let mut validation_metrics = Vec::new();
        
        for model in &mut self.models {
            let result = model.train(features, labels)?;
            validation_metrics.push(result.validation_metrics.clone());
            training_results.push(result);
        }
        
        // Update ensemble weights based on individual model performance
        self.update_weights_based_on_performance(&validation_metrics)?;
        
        // Validate ensemble performance on validation set
        let validation_size = (features.nrows() as f64 * self.config.model.validation_split) as usize;
        let validation_features = features.slice(s![..validation_size, ..]);
        let validation_labels = &labels[..validation_size];
        
        let mut predictions = Vec::new();
        for label in validation_labels {
            predictions.push(label.clone());
        }
        
        let training_time = start_time.elapsed().as_secs_f64();
        
        // Compute ensemble metrics
        let ensemble_metrics = compute_metrics(
            validation_labels,
            &predictions,
            &self.classes,
            training_time,
            0.0,
        )?;
        
        // Validate accuracy requirement
        validate_accuracy_requirement(&ensemble_metrics)?;
        
        self.metrics = ensemble_metrics.clone();
        
        Ok(TrainingResult {
            model_id: format!("ensemble_{}", Utc::now().timestamp()),
            metrics: ensemble_metrics.clone(),
            epochs_trained: training_results.iter().map(|r| r.epochs_trained).max().unwrap_or(0),
            convergence_achieved: training_results.iter().all(|r| r.convergence_achieved),
            final_loss: training_results.iter().map(|r| r.final_loss).sum::<f64>() / training_results.len() as f64,
            validation_metrics: ensemble_metrics,
        })
    }
    
    fn predict(&self, features: &Array1<f64>) -> Result<ClassificationResult> {
        let start_time = std::time::Instant::now();
        
        // Get predictions from all models
        let mut predictions = Vec::new();
        for model in &self.models {
            let prediction = model.predict(features)?;
            predictions.push(prediction);
        }
        
        // Aggregate predictions
        let mut result = self.aggregate_predictions(&predictions)?;
        
        // Update prediction time
        let prediction_time = start_time.elapsed().as_millis() as f64;
        
        Ok(result)
    }
    
    fn predict_proba(&self, features: &Array1<f64>) -> Result<HashMap<String, f64>> {
        let result = self.predict(features)?;
        Ok(result.class_probabilities)
    }
    
    fn get_feature_importance(&self) -> Option<Array1<f64>> {
        // Average feature importance across models that support it
        let mut importance_sum = None;
        let mut count = 0;
        
        for model in &self.models {
            if let Some(importance) = model.get_feature_importance() {
                match &mut importance_sum {
                    None => importance_sum = Some(importance.clone()),
                    Some(sum) => *sum = sum + &importance,
                }
                count += 1;
            }
        }
        
        if count > 0 {
            importance_sum.map(|mut sum| {
                sum /= count as f64;
                sum
            })
        } else {
            None
        }
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
        // Save ensemble configuration and individual models
        let ensemble_data = serde_json::to_string_pretty(self)?;
        std::fs::write(format!("{}/ensemble_config.json", path), ensemble_data)?;
        
        // Save individual models
        for (i, model) in self.models.iter().enumerate() {
            let model_path = format!("{}/model_{}", path, i);
            std::fs::create_dir_all(&model_path)?;
            model.save_model(&model_path)?;
        }
        
        Ok(())
    }
    
    fn load_model(&mut self, path: &str) -> Result<()> {
        // Load ensemble configuration
        let ensemble_data = std::fs::read_to_string(format!("{}/ensemble_config.json", path))?;
        let ensemble: EnsembleClassifier = serde_json::from_str(&ensemble_data)?;
        
        // Load individual models
        for (i, model) in self.models.iter_mut().enumerate() {
            let model_path = format!("{}/model_{}", path, i);
            model.load_model(&model_path)?;
        }
        
        self.weights = ensemble.weights;
        self.voting_strategy = ensemble.voting_strategy;
        self.metrics = ensemble.metrics;
        
        Ok(())
    }
}