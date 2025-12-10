use crate::{error::Result, types::WeatherData, ClimateDataError};
use moka::future::Cache;
use std::sync::Arc;
use std::time::Duration;

/// Cache key for weather data
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct CacheKey {
    pub lat: i32,  // Rounded to 2 decimal places
    pub lon: i32,  // Rounded to 2 decimal places
    pub timestamp: i64,
}

impl CacheKey {
    pub fn from_coords(lat: f64, lon: f64, timestamp: i64) -> Self {
        Self {
            lat: (lat * 100.0) as i32,
            lon: (lon * 100.0) as i32,
            timestamp,
        }
    }
}

/// Caching layer for weather data
pub struct DataCache {
    cache: Arc<Cache<CacheKey, WeatherData>>,
}

impl DataCache {
    /// Create a new cache with specified capacity and TTL
    pub fn new(max_capacity: u64, ttl: Duration) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(ttl)
            .build();

        Self {
            cache: Arc::new(cache),
        }
    }

    /// Create a default cache (1000 entries, 15 minute TTL)
    pub fn default() -> Self {
        Self::new(1000, Duration::from_secs(15 * 60))
    }

    /// Get data from cache
    pub async fn get(&self, key: &CacheKey) -> Option<WeatherData> {
        self.cache.get(key).await
    }

    /// Store data in cache
    pub async fn set(&self, key: CacheKey, data: WeatherData) -> Result<()> {
        self.cache.insert(key, data).await;
        Ok(())
    }

    /// Check if cache contains key
    pub async fn contains(&self, key: &CacheKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Clear all cached data
    pub async fn clear(&self) {
        self.cache.invalidate_all();
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.cache.entry_count(),
            weighted_size: self.cache.weighted_size(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entry_count: u64,
    pub weighted_size: u64,
}

impl Clone for DataCache {
    fn clone(&self) -> Self {
        Self {
            cache: Arc::clone(&self.cache),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::types::Coordinates;

    fn create_test_data() -> WeatherData {
        WeatherData {
            timestamp: Utc::now(),
            location: Coordinates::new(40.7128, -74.0060),
            temperature: 20.0,
            humidity: 60.0,
            pressure: 1013.25,
            wind_speed: 5.0,
            wind_direction: 180.0,
            precipitation: 0.0,
            cloud_cover: 50.0,
            visibility: Some(10000.0),
            uv_index: Some(5.0),
            condition_code: Some(800),
            description: Some("Clear sky".to_string()),
        }
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let cache = DataCache::new(10, Duration::from_secs(60));
        let key = CacheKey::from_coords(40.71, -74.01, Utc::now().timestamp());
        let data = create_test_data();

        // Store and retrieve
        cache.set(key.clone(), data.clone()).await.unwrap();
        let retrieved = cache.get(&key).await;
        assert!(retrieved.is_some());

        // Check contains
        assert!(cache.contains(&key).await);
    }

    #[tokio::test]
    async fn test_cache_expiry() {
        let cache = DataCache::new(10, Duration::from_millis(100));
        let key = CacheKey::from_coords(40.71, -74.01, Utc::now().timestamp());
        let data = create_test_data();

        cache.set(key.clone(), data).await.unwrap();
        assert!(cache.contains(&key).await);

        // Wait for expiry
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Entry should be expired
        assert!(!cache.contains(&key).await);
    }
}
