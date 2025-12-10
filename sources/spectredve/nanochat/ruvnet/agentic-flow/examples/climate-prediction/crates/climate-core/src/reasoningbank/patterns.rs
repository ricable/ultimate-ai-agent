//! Pattern storage and retrieval for climate predictions

use crate::{ClimateError, ModelType, Result};
use reasoningbank_core::Pattern;
use reasoningbank_storage::SqliteStorage;
use serde_json::json;
use std::collections::HashMap;

/// Pattern storage for climate predictions
pub struct PatternStorage {
    storage: SqliteStorage,
    namespace: String,
}

impl PatternStorage {
    /// Create a new pattern storage
    pub fn new(storage: SqliteStorage, namespace: &str) -> Self {
        Self {
            storage,
            namespace: namespace.to_string(),
        }
    }

    /// Store a location-specific pattern
    pub fn store_location_pattern(
        &self,
        location: (f64, f64),
        model: ModelType,
        accuracy: f64,
        samples: usize,
    ) -> Result<String> {
        let location_hash = Self::location_hash(location);

        let pattern = Pattern {
            id: format!("{}_{}", location_hash, model.as_str()),
            title: format!(
                "Pattern for location ({:.2}, {:.2}) with {}",
                location.0, location.1, model.as_str()
            ),
            content: format!(
                "Historical accuracy: {:.2}% based on {} samples",
                accuracy * 100.0,
                samples
            ),
            domain: self.namespace.clone(),
            agent: Some("climate-predictor".to_string()),
            task_type: Some("location_pattern".to_string()),
            confidence: accuracy,
            usage_count: samples,
            created_at: chrono::Utc::now(),
            last_used: Some(chrono::Utc::now()),
            metadata: Some(json!({
                "location": {
                    "lat": location.0,
                    "lon": location.1,
                    "hash": location_hash,
                },
                "model": model.as_str(),
                "accuracy": accuracy,
                "samples": samples,
            })),
        };

        self.storage
            .store_pattern(&pattern)
            .map_err(|e| ClimateError::Storage(e.to_string()))?;

        tracing::debug!("Stored location pattern: {}", pattern.id);
        Ok(pattern.id)
    }

    /// Store a seasonal pattern
    pub fn store_seasonal_pattern(
        &self,
        season: &str,
        model: ModelType,
        accuracy: f64,
    ) -> Result<String> {
        let pattern = Pattern {
            id: format!("season_{}_{}", season, model.as_str()),
            title: format!("Seasonal pattern for {} using {}", season, model.as_str()),
            content: format!(
                "Model {} performs at {:.2}% accuracy during {}",
                model.as_str(),
                accuracy * 100.0,
                season
            ),
            domain: self.namespace.clone(),
            agent: Some("climate-predictor".to_string()),
            task_type: Some("seasonal_pattern".to_string()),
            confidence: accuracy,
            usage_count: 1,
            created_at: chrono::Utc::now(),
            last_used: Some(chrono::Utc::now()),
            metadata: Some(json!({
                "season": season,
                "model": model.as_str(),
                "accuracy": accuracy,
            })),
        };

        self.storage
            .store_pattern(&pattern)
            .map_err(|e| ClimateError::Storage(e.to_string()))?;

        tracing::debug!("Stored seasonal pattern: {}", pattern.id);
        Ok(pattern.id)
    }

    /// Query patterns for a location
    pub fn query_location_patterns(
        &self,
        location: (f64, f64),
        radius_degrees: f64,
    ) -> Result<Vec<Pattern>> {
        let patterns = self.storage
            .list_patterns(&self.namespace, Some("location_pattern"))
            .map_err(|e| ClimateError::Storage(e.to_string()))?;

        // Filter patterns within radius
        let nearby: Vec<Pattern> = patterns
            .into_iter()
            .filter(|p| {
                if let Some(metadata) = &p.metadata {
                    if let (Some(lat), Some(lon)) = (
                        metadata.get("location").and_then(|l| l.get("lat")).and_then(|v| v.as_f64()),
                        metadata.get("location").and_then(|l| l.get("lon")).and_then(|v| v.as_f64()),
                    ) {
                        let distance = Self::haversine_distance(location, (lat, lon));
                        return distance <= radius_degrees;
                    }
                }
                false
            })
            .collect();

        tracing::debug!(
            "Found {} patterns within {:.2}° of ({:.2}, {:.2})",
            nearby.len(),
            radius_degrees,
            location.0,
            location.1
        );

        Ok(nearby)
    }

    /// Get best model for a location based on historical patterns
    pub fn get_best_model_for_location(
        &self,
        location: (f64, f64),
    ) -> Result<Option<ModelType>> {
        let patterns = self.query_location_patterns(location, 1.0)?;

        if patterns.is_empty() {
            return Ok(None);
        }

        // Group by model and calculate average confidence
        let mut model_scores: HashMap<String, (f64, usize)> = HashMap::new();

        for pattern in patterns {
            if let Some(metadata) = &pattern.metadata {
                if let Some(model) = metadata.get("model").and_then(|v| v.as_str()) {
                    let entry = model_scores.entry(model.to_string()).or_insert((0.0, 0));
                    entry.0 += pattern.confidence;
                    entry.1 += 1;
                }
            }
        }

        // Find model with highest average confidence
        let best = model_scores
            .iter()
            .max_by(|a, b| {
                let avg_a = a.1.0 / a.1.1 as f64;
                let avg_b = b.1.0 / b.1.1 as f64;
                avg_a.partial_cmp(&avg_b).unwrap()
            })
            .map(|(model, _)| model.as_str());

        let best_model = best.and_then(|m| match m {
            "linear" => Some(ModelType::Linear),
            "arima" => Some(ModelType::Arima),
            "neural" => Some(ModelType::Neural),
            "ensemble" => Some(ModelType::Ensemble),
            _ => None,
        });

        tracing::info!(
            "Best model for location ({:.2}, {:.2}): {:?}",
            location.0,
            location.1,
            best_model
        );

        Ok(best_model)
    }

    /// Generate location hash
    fn location_hash(location: (f64, f64)) -> String {
        format!("{:.2}_{:.2}", location.0, location.1)
    }

    /// Calculate Haversine distance between two points (in degrees)
    fn haversine_distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
        let dlat = (p2.0 - p1.0).to_radians();
        let dlon = (p2.1 - p1.1).to_radians();

        let a = (dlat / 2.0).sin().powi(2)
            + p1.0.to_radians().cos()
                * p2.0.to_radians().cos()
                * (dlon / 2.0).sin().powi(2);

        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        c.to_degrees()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reasoningbank_storage::StorageConfig;
    use tempfile::tempdir;

    #[test]
    fn test_location_pattern() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            database_path: dir.path().join("test.db"),
            max_connections: 10,
            enable_wal: true,
            cache_size_kb: 8192,
        };

        let storage = SqliteStorage::new(config).unwrap();
        let patterns = PatternStorage::new(storage, "test");

        let id = patterns
            .store_location_pattern((40.7128, -74.0060), ModelType::Ensemble, 0.85, 100)
            .unwrap();

        assert!(!id.is_empty());
    }

    #[test]
    fn test_haversine_distance() {
        let p1 = (40.7128, -74.0060); // New York
        let p2 = (34.0522, -118.2437); // Los Angeles

        let distance = PatternStorage::haversine_distance(p1, p2);
        assert!(distance > 30.0); // Should be ~40 degrees
    }
}
