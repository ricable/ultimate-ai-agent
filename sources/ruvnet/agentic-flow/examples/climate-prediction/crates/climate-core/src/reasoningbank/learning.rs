//! Learning integration for climate predictions

use crate::{ClimateData, ClimatePrediction, ClimateInput, ModelType, Result, ClimateError};
use reasoningbank_core::Pattern;
use reasoningbank_storage::{SqliteStorage, ConnectionPool, StorageConfig};
use reasoningbank_learning::{AdaptiveLearner, LearningConfig, LearningInsight};
use serde_json::json;
use chrono::Utc;
use std::path::PathBuf;
use uuid::Uuid;

/// ReasoningBank learner for climate predictions
pub struct ReasoningBankLearner {
    storage: SqliteStorage,
    learner: AdaptiveLearner,
    namespace: String,
}

impl ReasoningBankLearner {
    /// Create a new learner
    pub fn new(db_path: PathBuf, namespace: &str) -> Result<Self> {
        let config = StorageConfig {
            database_path: db_path,
            max_connections: 10,
            enable_wal: true,
            cache_size_kb: 8192,
        };

        let storage = SqliteStorage::new(config)
            .map_err(|e| ClimateError::Storage(e.to_string()))?;

        let learning_config = LearningConfig::default();
        let learner = AdaptiveLearner::new(learning_config);

        Ok(Self {
            storage,
            learner,
            namespace: namespace.to_string(),
        })
    }

    /// Store a successful prediction for learning
    pub fn store_successful_prediction(
        &self,
        input: &ClimateInput,
        prediction: &ClimatePrediction,
        actual: Option<&ClimateData>,
    ) -> Result<String> {
        let pattern_data = json!({
            "type": "prediction_success",
            "location": {
                "lat": input.location.latitude,
                "lon": input.location.longitude,
            },
            "model": prediction.model_used.as_str(),
            "confidence": prediction.confidence,
            "predicted_temp": prediction.temperature,
            "predicted_humidity": prediction.humidity,
            "actual_temp": actual.map(|a| a.temperature),
            "actual_humidity": actual.map(|a| a.humidity),
            "forecast_hours": input.forecast_hours,
            "timestamp": Utc::now().to_rfc3339(),
        });

        let pattern = Pattern {
            id: Uuid::new_v4().to_string(),
            title: format!(
                "Climate prediction for ({}, {}) using {}",
                input.location.latitude,
                input.location.longitude,
                prediction.model_used.as_str()
            ),
            content: format!(
                "Successful prediction with {:.2}% confidence. Predicted: {:.1}°C, Actual: {}",
                prediction.confidence * 100.0,
                actual.map(|a| format!("{:.1}°C", a.temperature))
                    .unwrap_or_else(|| "N/A".to_string())
            ),
            domain: self.namespace.clone(),
            agent: Some("climate-predictor".to_string()),
            task_type: Some("weather_forecasting".to_string()),
            confidence: prediction.confidence,
            usage_count: 1,
            created_at: Utc::now(),
            last_used: Some(Utc::now()),
            metadata: Some(pattern_data),
        };

        self.storage
            .store_pattern(&pattern)
            .map_err(|e| ClimateError::Storage(e.to_string()))?;

        tracing::info!("Stored successful prediction pattern: {}", pattern.id);
        Ok(pattern.id)
    }

    /// Learn from a prediction outcome
    pub fn learn_from_outcome(
        &mut,,
        prediction: &ClimatePrediction,
        actual: &ClimateData,
    ) -> Result<LearningInsight> {
        // Calculate accuracy
        let temp_diff = (prediction.temperature - actual.temperature).abs();
        let temp_accuracy = 1.0 - (temp_diff / 10.0).min(1.0);

        let humidity_diff = (prediction.humidity - actual.humidity).abs();
        let humidity_accuracy = 1.0 - (humidity_diff / 20.0).min(1.0);

        let overall_accuracy = (temp_accuracy + humidity_accuracy) / 2.0;

        // Create task trajectory
        let task_data = json!({
            "model": prediction.model_used.as_str(),
            "location": {
                "lat": prediction.location.latitude,
                "lon": prediction.location.longitude,
            },
            "accuracy": overall_accuracy,
            "temp_accuracy": temp_accuracy,
            "humidity_accuracy": humidity_accuracy,
        });

        let outcome = json!({
            "success": overall_accuracy > 0.7,
            "accuracy": overall_accuracy,
            "predicted": {
                "temperature": prediction.temperature,
                "humidity": prediction.humidity,
            },
            "actual": {
                "temperature": actual.temperature,
                "humidity": actual.humidity,
            },
        });

        // Learn from this trajectory
        let insight = self.learner
            .learn_from_trajectory(task_data, outcome, vec![])
            .map_err(|e| ClimateError::ReasoningBank(e.to_string()))?;

        tracing::info!(
            "Learned from prediction: accuracy={:.2}%, model={}",
            overall_accuracy * 100.0,
            prediction.model_used.as_str()
        );

        Ok(insight)
    }

    /// Query similar patterns for a location
    pub fn query_similar_patterns(
        &self,
        location: (f64, f64),
        model: ModelType,
    ) -> Result<Vec<Pattern>> {
        let query = format!(
            "lat={:.2},lon={:.2},model={}",
            location.0, location.1, model.as_str()
        );

        let patterns = self.storage
            .query_patterns(&query, &self.namespace, 10)
            .map_err(|e| ClimateError::Storage(e.to_string()))?;

        tracing::debug!(
            "Found {} similar patterns for query: {}",
            patterns.len(),
            query
        );

        Ok(patterns)
    }

    /// Get statistics about learned patterns
    pub fn get_learning_stats(&self) -> Result<LearningStats> {
        let patterns = self.storage
            .list_patterns(&self.namespace, None)
            .map_err(|e| ClimateError::Storage(e.to_string()))?;

        let total_patterns = patterns.len();
        let avg_confidence = if total_patterns > 0 {
            patterns.iter().map(|p| p.confidence).sum::<f64>() / total_patterns as f64
        } else {
            0.0
        };

        let by_model = patterns.iter().fold(
            std::collections::HashMap::new(),
            |mut acc, p| {
                if let Some(metadata) = &p.metadata {
                    if let Some(model) = metadata.get("model").and_then(|v| v.as_str()) {
                        *acc.entry(model.to_string()).or_insert(0) += 1;
                    }
                }
                acc
            }
        );

        Ok(LearningStats {
            total_patterns,
            avg_confidence,
            patterns_by_model: by_model,
        })
    }
}

/// Learning statistics
#[derive(Debug, Clone)]
pub struct LearningStats {
    pub total_patterns: usize,
    pub avg_confidence: f64,
    pub patterns_by_model: std::collections::HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Location;
    use tempfile::tempdir;

    #[test]
    fn test_store_pattern() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let learner = ReasoningBankLearner::new(db_path, "test").unwrap();

        let input = ClimateInput {
            location: Location {
                latitude: 40.7128,
                longitude: -74.0060,
                name: Some("New York".to_string()),
            },
            forecast_hours: 24,
            model_preference: Some(ModelType::Ensemble),
        };

        let prediction = ClimatePrediction {
            location: input.location.clone(),
            predicted_at: Utc::now(),
            prediction_time: Utc::now() + chrono::Duration::hours(24),
            temperature: 25.0,
            humidity: 60.0,
            precipitation_probability: 0.3,
            confidence: 0.85,
            model_used: ModelType::Ensemble,
        };

        let pattern_id = learner
            .store_successful_prediction(&input, &prediction, None)
            .unwrap();

        assert!(!pattern_id.is_empty());
    }
}
