//! Performance optimization using ReasoningBank learning

use crate::{ClimateInput, ClimatePrediction, ClimateData, ModelType, Result, ClimateError};
use super::{ReasoningBankLearner, PatternStorage};
use std::collections::HashMap;
use std::path::PathBuf;
use reasoningbank_storage::{SqliteStorage, StorageConfig};

/// ReasoningBank optimizer for climate predictions
pub struct ReasoningBankOptimizer {
    learner: ReasoningBankLearner,
    patterns: PatternStorage,
    namespace: String,
    optimization_cache: HashMap<String, ModelType>,
}

impl ReasoningBankOptimizer {
    /// Create a new optimizer
    pub fn new(db_path: PathBuf, namespace: &str) -> Result<Self> {
        let config = StorageConfig {
            database_path: db_path.clone(),
            max_connections: 10,
            enable_wal: true,
            cache_size_kb: 8192,
        };

        let storage = SqliteStorage::new(config)
            .map_err(|e| ClimateError::Storage(e.to_string()))?;

        let learner = ReasoningBankLearner::new(db_path, namespace)?;
        let patterns = PatternStorage::new(storage, namespace);

        Ok(Self {
            learner,
            patterns,
            namespace: namespace.to_string(),
            optimization_cache: HashMap::new(),
        })
    }

    /// Optimize model selection for a location
    pub fn optimize_model_selection(
        &mut self,
        location: (f64, f64),
    ) -> Result<OptimizationResult> {
        let cache_key = format!("{:.2}_{:.2}", location.0, location.1);

        // Check cache first
        if let Some(cached_model) = self.optimization_cache.get(&cache_key) {
            tracing::debug!("Using cached model selection: {:?}", cached_model);
            return Ok(OptimizationResult {
                recommended_model: *cached_model,
                confidence: 0.9,
                reason: "Cached from previous optimization".to_string(),
                patterns_analyzed: 0,
            });
        }

        // Query historical patterns
        let patterns = self.patterns.query_location_patterns(location, 1.0)?;
        let patterns_count = patterns.len();

        if patterns.is_empty() {
            // No historical data - use default
            return Ok(OptimizationResult {
                recommended_model: ModelType::Ensemble,
                confidence: 0.5,
                reason: "No historical data - using default ensemble model".to_string(),
                patterns_analyzed: 0,
            });
        }

        // Determine best model from patterns
        let best_model = self.patterns
            .get_best_model_for_location(location)?
            .unwrap_or(ModelType::Ensemble);

        // Calculate confidence based on pattern count and accuracy
        let avg_confidence = patterns.iter()
            .map(|p| p.confidence)
            .sum::<f64>() / patterns.len() as f64;

        let confidence = (avg_confidence * 0.7) + (0.3 * (patterns.len().min(10) as f64 / 10.0));

        // Cache the result
        self.optimization_cache.insert(cache_key, best_model);

        Ok(OptimizationResult {
            recommended_model: best_model,
            confidence,
            reason: format!(
                "Based on {} historical patterns with {:.2}% avg accuracy",
                patterns_count,
                avg_confidence * 100.0
            ),
            patterns_analyzed: patterns_count,
        })
    }

    /// Record a prediction result for learning
    pub fn record_prediction_result(
        &mut self,
        input: &ClimateInput,
        prediction: &ClimatePrediction,
        actual: Option<&ClimateData>,
    ) -> Result<()> {
        // Store the prediction in learning system
        self.learner.store_successful_prediction(input, prediction, actual)?;

        // If we have actual data, learn from the outcome
        if let Some(actual_data) = actual {
            let _insight = self.learner.learn_from_outcome(prediction, actual_data)?;

            // Update location pattern
            let location = (prediction.location.latitude, prediction.location.longitude);
            let accuracy = calculate_accuracy(prediction, actual_data);

            self.patterns.store_location_pattern(
                location,
                prediction.model_used,
                accuracy,
                1,
            )?;

            tracing::info!(
                "Recorded prediction result: accuracy={:.2}%, model={:?}",
                accuracy * 100.0,
                prediction.model_used
            );
        }

        Ok(())
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> Result<OptimizationStats> {
        let learning_stats = self.learner.get_learning_stats()?;

        Ok(OptimizationStats {
            total_predictions: learning_stats.total_patterns,
            avg_confidence: learning_stats.avg_confidence,
            cache_size: self.optimization_cache.len(),
            patterns_by_model: learning_stats.patterns_by_model,
        })
    }

    /// Clear optimization cache
    pub fn clear_cache(&mut self) {
        self.optimization_cache.clear();
        tracing::info!("Optimization cache cleared");
    }

    /// Query similar successful predictions
    pub fn query_similar_predictions(
        &self,
        location: (f64, f64),
        model: ModelType,
    ) -> Result<Vec<reasoningbank_core::Pattern>> {
        self.learner.query_similar_patterns(location, model)
    }
}

/// Result of optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub recommended_model: ModelType,
    pub confidence: f64,
    pub reason: String,
    pub patterns_analyzed: usize,
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub total_predictions: usize,
    pub avg_confidence: f64,
    pub cache_size: usize,
    pub patterns_by_model: HashMap<String, usize>,
}

/// Calculate prediction accuracy
fn calculate_accuracy(prediction: &ClimatePrediction, actual: &ClimateData) -> f64 {
    let temp_diff = (prediction.temperature - actual.temperature).abs();
    let temp_accuracy = 1.0 - (temp_diff / 10.0).min(1.0);

    let humidity_diff = (prediction.humidity - actual.humidity).abs();
    let humidity_accuracy = 1.0 - (humidity_diff / 20.0).min(1.0);

    (temp_accuracy + humidity_accuracy) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Location;
    use tempfile::tempdir;
    use chrono::Utc;

    #[test]
    fn test_optimizer_creation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let optimizer = ReasoningBankOptimizer::new(db_path, "test").unwrap();
        assert_eq!(optimizer.namespace, "test");
    }

    #[test]
    fn test_model_optimization() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let mut optimizer = ReasoningBankOptimizer::new(db_path, "test").unwrap();

        let location = (40.7128, -74.0060);
        let result = optimizer.optimize_model_selection(location).unwrap();

        assert!(result.confidence > 0.0);
        assert!(!result.reason.is_empty());
    }

    #[test]
    fn test_calculate_accuracy() {
        let prediction = ClimatePrediction {
            location: Location {
                latitude: 40.7128,
                longitude: -74.0060,
                name: Some("New York".to_string()),
            },
            predicted_at: Utc::now(),
            prediction_time: Utc::now() + chrono::Duration::hours(24),
            temperature: 25.0,
            humidity: 60.0,
            precipitation_probability: 0.3,
            confidence: 0.85,
            model_used: ModelType::Ensemble,
        };

        let actual = ClimateData {
            location: prediction.location.clone(),
            timestamp: Utc::now(),
            temperature: 24.0,
            humidity: 58.0,
            pressure: 1013.25,
            wind_speed: 5.0,
            precipitation: 0.0,
            cloud_cover: 30.0,
        };

        let accuracy = calculate_accuracy(&prediction, &actual);
        assert!(accuracy > 0.8);
        assert!(accuracy <= 1.0);
    }
}
