//! Storage adapters for different environments
//!
//! This module provides storage adapters for:
//! - Native: rusqlite (Node.js, desktop)
//! - WASM Browser: sql.js or IndexedDB
//! - Remote: HTTP API fallback

use crate::{Pattern, StorageError, StorageStats};
use std::sync::Arc;
use uuid::Uuid;

/// Storage backend trait - unified interface across all backends
#[async_trait::async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store a pattern
    async fn store_pattern(&self, pattern: &Pattern) -> Result<(), StorageError>;

    /// Retrieve a pattern by ID
    async fn get_pattern(&self, id: &Uuid) -> Result<Option<Pattern>, StorageError>;

    /// Get patterns by category
    async fn get_patterns_by_category(&self, category: &str, limit: usize) -> Result<Vec<Pattern>, StorageError>;

    /// Get storage statistics
    async fn get_stats(&self) -> Result<StorageStats, StorageError>;

    /// Close the storage backend
    async fn close(&self) -> Result<(), StorageError>;
}

// Native storage (rusqlite)
#[cfg(not(target_family = "wasm"))]
pub mod native;

#[cfg(not(target_family = "wasm"))]
pub use native::NativeStorage;

// WASM storage adapters
#[cfg(target_family = "wasm")]
pub mod wasm;

#[cfg(target_family = "wasm")]
pub use wasm::{SqlJsStorage, IndexedDbStorage};

/// Auto-detect best storage backend for current environment
pub async fn auto_detect_storage(config: crate::StorageConfig) -> Result<Arc<dyn StorageBackend>, StorageError> {
    #[cfg(not(target_family = "wasm"))]
    {
        // Native environment: Use rusqlite
        let storage = native::NativeStorage::new(config).await?;
        Ok(Arc::new(storage))
    }

    #[cfg(target_family = "wasm")]
    {
        // WASM environment: Try IndexedDB first, fallback to sql.js
        if wasm::has_indexed_db() {
            let storage = wasm::IndexedDbStorage::new(config).await?;
            Ok(Arc::new(storage))
        } else {
            let storage = wasm::SqlJsStorage::new(config).await?;
            Ok(Arc::new(storage))
        }
    }
}
