//! Async wrapper for SqliteStorage to enable Send + Sync async operations

use crate::{SqliteStorage, Result};
use crate::sqlite::{StorageConfig, StorageStats};
use reasoningbank_core::Pattern;
use std::path::PathBuf;
use uuid::Uuid;

/// Async-safe wrapper around SqliteStorage
///
/// This wrapper uses tokio::task::spawn_blocking to run synchronous SQLite
/// operations on a blocking thread pool, allowing async functions to be Send.
#[derive(Clone)]
pub struct AsyncStorage {
    /// Path to database (None for in-memory)
    db_path: Option<PathBuf>,
    /// Config for creating new connections
    config: StorageConfig,
}

impl AsyncStorage {
    /// Create a new async storage wrapper
    pub fn new(config: StorageConfig) -> Result<Self> {
        let db_path = if config.database_path.to_str() == Some(":memory:") {
            None
        } else {
            Some(config.database_path.clone())
        };

        Ok(Self {
            db_path,
            config,
        })
    }

    /// Create in-memory storage
    pub fn in_memory() -> Result<Self> {
        Self::new(StorageConfig {
            database_path: PathBuf::from(":memory:"),
            max_connections: 1,
            enable_wal: false,
            cache_size_kb: 4096,
        })
    }

    /// Execute a blocking storage operation
    async fn execute_blocking<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut SqliteStorage) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let config = self.config.clone();
        tokio::task::spawn_blocking(move || {
            let mut storage = SqliteStorage::new(config)?;
            f(&mut storage)
        })
        .await
        .map_err(|e| crate::StorageError::Internal(format!("Task join error: {}", e)))?
    }

    /// Store a pattern
    pub async fn store_pattern(&self, pattern: &Pattern) -> Result<()> {
        let pattern = pattern.clone();
        self.execute_blocking(move |storage| storage.store_pattern(&pattern)).await
    }

    /// Get a pattern by ID
    pub async fn get_pattern(&self, id: &Uuid) -> Result<Option<Pattern>> {
        let id = *id;
        self.execute_blocking(move |storage| storage.get_pattern(&id)).await
    }

    /// Get all patterns with optional limit
    pub async fn get_all_patterns(&self, limit: Option<usize>) -> Result<Vec<Pattern>> {
        self.execute_blocking(move |storage| storage.get_all_patterns(limit)).await
    }

    /// Get patterns by category
    pub async fn get_patterns_by_category(&self, category: &str, limit: usize) -> Result<Vec<Pattern>> {
        let category = category.to_string();
        self.execute_blocking(move |storage| storage.get_patterns_by_category(&category, limit)).await
    }

    /// Search patterns by query
    pub async fn search_patterns(&self, query: &str, limit: usize) -> Result<Vec<Pattern>> {
        let query = query.to_string();
        self.execute_blocking(move |storage| storage.search_patterns(&query, limit)).await
    }

    /// Delete a pattern
    pub async fn delete_pattern(&self, id: &Uuid) -> Result<()> {
        let id = *id;
        self.execute_blocking(move |storage| storage.delete_pattern(&id)).await
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> Result<StorageStats> {
        self.execute_blocking(|storage| storage.get_stats()).await
    }

    /// Delete patterns older than specified days
    pub async fn delete_old_patterns(&self, days: i64) -> Result<usize> {
        self.execute_blocking(move |storage| storage.delete_old_patterns(days)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_storage() {
        // Use temp file instead of in-memory since each execute_blocking creates new connection
        let temp_db = std::env::temp_dir().join(format!("test_{}.db", uuid::Uuid::new_v4()));
        let config = StorageConfig {
            database_path: temp_db.clone(),
            max_connections: 1,
            enable_wal: false,
            cache_size_kb: 4096,
        };
        let storage = AsyncStorage::new(config).unwrap();

        let pattern = Pattern::new(
            "Test task".to_string(),
            "testing".to_string(),
            "strategy".to_string(),
        );

        storage.store_pattern(&pattern).await.unwrap();
        let retrieved = storage.get_pattern(&pattern.id).await.unwrap();
        assert!(retrieved.is_some());

        // Cleanup
        let _ = std::fs::remove_file(&temp_db);
    }

    #[tokio::test]
    async fn test_async_parallel_operations() {
        // Use temp file for shared storage
        let temp_db = std::env::temp_dir().join(format!("test_{}.db", uuid::Uuid::new_v4()));
        let config = StorageConfig {
            database_path: temp_db.clone(),
            max_connections: 5,
            enable_wal: true,
            cache_size_kb: 4096,
        };
        let storage = AsyncStorage::new(config).unwrap();

        // Store multiple patterns in parallel
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let storage = storage.clone();
                tokio::spawn(async move {
                    let pattern = Pattern::new(
                        format!("Task {}", i),
                        "test".to_string(),
                        "strategy".to_string(),
                    );
                    storage.store_pattern(&pattern).await
                })
            })
            .collect();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_patterns, 5);

        // Cleanup
        let _ = std::fs::remove_file(&temp_db);
    }
}
