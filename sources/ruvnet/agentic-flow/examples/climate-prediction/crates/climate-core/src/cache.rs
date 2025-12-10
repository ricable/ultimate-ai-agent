//! Prediction caching with ReasoningBank integration

use crate::{ClimateData, ClimatePrediction, Location, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Duration, Utc};

/// Cache entry with TTL
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    data: T,
    expires_at: DateTime<Utc>,
}

/// Prediction cache with intelligent TTL
pub struct PredictionCache {
    predictions: Arc<RwLock<HashMap<String, CacheEntry<ClimatePrediction>>>>,
    climate_data: Arc<RwLock<HashMap<String, CacheEntry<ClimateData>>>>,
    default_ttl: Duration,
}

impl PredictionCache {
    /// Create a new cache with default TTL of 5 minutes
    pub fn new() -> Self {
        Self::with_ttl(Duration::minutes(5))
    }

    /// Create a new cache with custom TTL
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            predictions: Arc::new(RwLock::new(HashMap::new())),
            climate_data: Arc::new(RwLock::new(HashMap::new())),
            default_ttl: ttl,
        }
    }

    /// Generate cache key for a location
    fn location_key(location: &Location) -> String {
        format!("{}:{}", location.latitude, location.longitude)
    }

    /// Store a prediction in cache
    pub fn store_prediction(&self, prediction: ClimatePrediction) -> Result<()> {
        let key = Self::location_key(&prediction.location);
        let entry = CacheEntry {
            data: prediction,
            expires_at: Utc::now() + self.default_ttl,
        };

        let mut cache = self.predictions.write().unwrap();
        cache.insert(key, entry);

        tracing::debug!("Stored prediction in cache");
        Ok(())
    }

    /// Retrieve a prediction from cache
    pub fn get_prediction(&self, location: &Location) -> Option<ClimatePrediction> {
        let key = Self::location_key(location);
        let cache = self.predictions.read().unwrap();

        if let Some(entry) = cache.get(&key) {
            if entry.expires_at > Utc::now() {
                tracing::debug!("Cache hit for prediction");
                return Some(entry.data.clone());
            }
            tracing::debug!("Cache entry expired");
        }

        tracing::debug!("Cache miss for prediction");
        None
    }

    /// Store climate data in cache
    pub fn store_climate_data(&self, data: ClimateData) -> Result<()> {
        let key = Self::location_key(&data.location);
        let entry = CacheEntry {
            data,
            expires_at: Utc::now() + self.default_ttl,
        };

        let mut cache = self.climate_data.write().unwrap();
        cache.insert(key, entry);

        tracing::debug!("Stored climate data in cache");
        Ok(())
    }

    /// Retrieve climate data from cache
    pub fn get_climate_data(&self, location: &Location) -> Option<ClimateData> {
        let key = Self::location_key(location);
        let cache = self.climate_data.read().unwrap();

        if let Some(entry) = cache.get(&key) {
            if entry.expires_at > Utc::now() {
                tracing::debug!("Cache hit for climate data");
                return Some(entry.data.clone());
            }
        }

        tracing::debug!("Cache miss for climate data");
        None
    }

    /// Clean expired entries
    pub fn cleanup(&self) {
        let now = Utc::now();

        let mut predictions = self.predictions.write().unwrap();
        predictions.retain(|_, entry| entry.expires_at > now);

        let mut climate = self.climate_data.write().unwrap();
        climate.retain(|_, entry| entry.expires_at > now);

        tracing::debug!("Cache cleanup completed");
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let predictions = self.predictions.read().unwrap();
        let climate = self.climate_data.read().unwrap();

        CacheStats {
            prediction_count: predictions.len(),
            climate_data_count: climate.len(),
            total_entries: predictions.len() + climate.len(),
        }
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        self.predictions.write().unwrap().clear();
        self.climate_data.write().unwrap().clear();
        tracing::info!("Cache cleared");
    }
}

impl Default for PredictionCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub prediction_count: usize,
    pub climate_data_count: usize,
    pub total_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelType;

    #[test]
    fn test_cache_prediction() {
        let cache = PredictionCache::new();
        let location = Location {
            latitude: 40.7128,
            longitude: -74.0060,
            name: Some("New York".to_string()),
        };

        let prediction = ClimatePrediction {
            location: location.clone(),
            predicted_at: Utc::now(),
            prediction_time: Utc::now() + Duration::hours(1),
            temperature: 25.0,
            humidity: 60.0,
            precipitation_probability: 0.3,
            confidence: 0.85,
            model_used: ModelType::Ensemble,
        };

        cache.store_prediction(prediction.clone()).unwrap();
        let cached = cache.get_prediction(&location).unwrap();
        assert_eq!(cached.temperature, 25.0);
    }

    #[test]
    fn test_cache_expiry() {
        let cache = PredictionCache::with_ttl(Duration::milliseconds(100));
        let location = Location {
            latitude: 40.7128,
            longitude: -74.0060,
            name: Some("New York".to_string()),
        };

        let prediction = ClimatePrediction {
            location: location.clone(),
            predicted_at: Utc::now(),
            prediction_time: Utc::now() + Duration::hours(1),
            temperature: 25.0,
            humidity: 60.0,
            precipitation_probability: 0.3,
            confidence: 0.85,
            model_used: ModelType::Ensemble,
        };

        cache.store_prediction(prediction).unwrap();

        // Wait for expiry
        std::thread::sleep(std::time::Duration::from_millis(150));

        let cached = cache.get_prediction(&location);
        assert!(cached.is_none());
    }
}
