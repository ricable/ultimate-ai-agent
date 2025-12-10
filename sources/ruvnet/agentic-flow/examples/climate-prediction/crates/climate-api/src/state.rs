use std::collections::HashMap;
use std::sync::RwLock;
use anyhow::Result;
use uuid::Uuid;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    /// API key store (in production, use a database)
    pub api_keys: Arc<RwLock<HashMap<String, ApiKeyInfo>>>,
    /// Prediction cache
    pub prediction_cache: Arc<RwLock<HashMap<Uuid, CachedPrediction>>>,
}

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    pub key: String,
    pub name: String,
    pub tier: ApiTier,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiTier {
    Free,
    Premium,
    Enterprise,
}

impl ApiTier {
    pub fn rate_limit(&self) -> u32 {
        match self {
            ApiTier::Free => 10,      // 10 requests per minute
            ApiTier::Premium => 100,  // 100 requests per minute
            ApiTier::Enterprise => 1000, // 1000 requests per minute
        }
    }
}

#[derive(Debug, Clone)]
pub struct CachedPrediction {
    pub id: Uuid,
    pub data: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

impl AppState {
    pub async fn new() -> Result<Self> {
        let mut api_keys = HashMap::new();

        // Add default API key for development
        api_keys.insert(
            "dev-key-12345".to_string(),
            ApiKeyInfo {
                key: "dev-key-12345".to_string(),
                name: "Development Key".to_string(),
                tier: ApiTier::Premium,
                created_at: chrono::Utc::now(),
                last_used: None,
            },
        );

        Ok(Self {
            api_keys: Arc::new(RwLock::new(api_keys)),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn validate_api_key(&self, key: &str) -> Option<ApiKeyInfo> {
        let keys = self.api_keys.read().ok()?;
        keys.get(key).cloned()
    }

    pub fn update_api_key_usage(&self, key: &str) {
        if let Ok(mut keys) = self.api_keys.write() {
            if let Some(key_info) = keys.get_mut(key) {
                key_info.last_used = Some(chrono::Utc::now());
            }
        }
    }

    pub fn cache_prediction(&self, id: Uuid, data: serde_json::Value, ttl_seconds: i64) {
        let now = chrono::Utc::now();
        let prediction = CachedPrediction {
            id,
            data,
            created_at: now,
            expires_at: now + chrono::Duration::seconds(ttl_seconds),
        };

        if let Ok(mut cache) = self.prediction_cache.write() {
            cache.insert(id, prediction);
        }
    }

    pub fn get_cached_prediction(&self, id: &Uuid) -> Option<serde_json::Value> {
        let cache = self.prediction_cache.read().ok()?;
        let prediction = cache.get(id)?;

        if prediction.expires_at > chrono::Utc::now() {
            Some(prediction.data.clone())
        } else {
            None
        }
    }

    pub fn cleanup_expired_cache(&self) {
        if let Ok(mut cache) = self.prediction_cache.write() {
            let now = chrono::Utc::now();
            cache.retain(|_, pred| pred.expires_at > now);
        }
    }
}
