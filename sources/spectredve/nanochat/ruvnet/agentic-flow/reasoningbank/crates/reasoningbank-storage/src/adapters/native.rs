//! Native storage backend using rusqlite
//!
//! High-performance SQLite storage with connection pooling,
//! WAL mode, and optimized queries.

use crate::{Pattern, StorageConfig, StorageError, StorageStats};
use super::StorageBackend;
use rusqlite::{Connection, OptionalExtension};
use parking_lot::Mutex;
use std::sync::Arc;
use uuid::Uuid;

/// Native storage backend using rusqlite
pub struct NativeStorage {
    conn: Arc<Mutex<Connection>>,
    config: StorageConfig,
}

impl NativeStorage {
    /// Create a new native storage backend
    pub async fn new(config: StorageConfig) -> Result<Self, StorageError> {
        let conn = Connection::open(&config.database_path)
            .map_err(|e| StorageError::Connection(e.to_string()))?;

        // Enable WAL mode for concurrent reads
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| StorageError::Query(e.to_string()))?;

        // Optimize pragmas
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| StorageError::Query(e.to_string()))?;
        conn.pragma_update(None, "cache_size", -(config.cache_size_kb as i32))
            .map_err(|e| StorageError::Query(e.to_string()))?;
        conn.pragma_update(None, "temp_store", "MEMORY")
            .map_err(|e| StorageError::Query(e.to_string()))?;

        // Create tables if they don't exist
        conn.execute_batch(include_str!("../../migrations/001_initial.sql"))
            .map_err(|e| StorageError::Migration(e.to_string()))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            config,
        })
    }
}

#[async_trait::async_trait]
impl StorageBackend for NativeStorage {
    async fn store_pattern(&self, pattern: &Pattern) -> Result<(), StorageError> {
        let conn = self.conn.lock();

        let json_data = serde_json::to_string(pattern)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        conn.execute(
            "INSERT OR REPLACE INTO patterns (id, task_category, task_description, strategy, success_score, data, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, CURRENT_TIMESTAMP)",
            rusqlite::params![
                pattern.id.to_string(),
                pattern.task_category,
                pattern.task_description,
                pattern.strategy,
                pattern.outcome.success_score,
                json_data,
            ],
        ).map_err(|e| StorageError::Query(e.to_string()))?;

        Ok(())
    }

    async fn get_pattern(&self, id: &Uuid) -> Result<Option<Pattern>, StorageError> {
        let conn = self.conn.lock();

        let result: Option<String> = conn
            .query_row(
                "SELECT data FROM patterns WHERE id = ?1",
                [id.to_string()],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| StorageError::Query(e.to_string()))?;

        match result {
            Some(json_data) => {
                let pattern = serde_json::from_str(&json_data)
                    .map_err(|e| StorageError::Deserialization(e.to_string()))?;
                Ok(Some(pattern))
            }
            None => Ok(None),
        }
    }

    async fn get_patterns_by_category(&self, category: &str, limit: usize) -> Result<Vec<Pattern>, StorageError> {
        let conn = self.conn.lock();

        let mut stmt = conn.prepare(
            "SELECT data FROM patterns WHERE task_category = ?1 ORDER BY created_at DESC LIMIT ?2"
        ).map_err(|e| StorageError::Query(e.to_string()))?;

        let rows = stmt.query_map([category, &limit.to_string()], |row| {
            row.get::<_, String>(0)
        }).map_err(|e| StorageError::Query(e.to_string()))?;

        let mut patterns = Vec::new();
        for row in rows {
            let json_data = row.map_err(|e| StorageError::Query(e.to_string()))?;
            let pattern: Pattern = serde_json::from_str(&json_data)
                .map_err(|e| StorageError::Deserialization(e.to_string()))?;
            patterns.push(pattern);
        }

        Ok(patterns)
    }

    async fn get_stats(&self) -> Result<StorageStats, StorageError> {
        let conn = self.conn.lock();

        let total_patterns: i64 = conn.query_row(
            "SELECT COUNT(*) FROM patterns",
            [],
            |row| row.get(0),
        ).map_err(|e| StorageError::Query(e.to_string()))?;

        let total_categories: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT task_category) FROM patterns",
            [],
            |row| row.get(0),
        ).map_err(|e| StorageError::Query(e.to_string()))?;

        Ok(StorageStats {
            total_patterns: total_patterns as usize,
            total_categories: total_categories as usize,
            backend_type: "native-rusqlite".to_string(),
        })
    }

    async fn close(&self) -> Result<(), StorageError> {
        // rusqlite connections are automatically closed on drop
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reasoningbank_core::TaskOutcome;

    #[tokio::test]
    async fn test_native_storage() {
        use std::path::PathBuf;

        let config = StorageConfig {
            database_path: PathBuf::from(":memory:"),
            max_connections: 1,
            enable_wal: false,
            cache_size_kb: 2048,
        };

        let storage = NativeStorage::new(config).await.unwrap();

        let pattern = Pattern::new(
            "test task".to_string(),
            "test_category".to_string(),
            "test_strategy".to_string(),
        ).with_outcome(TaskOutcome::success(0.95));

        // Store
        storage.store_pattern(&pattern).await.unwrap();

        // Retrieve
        let retrieved = storage.get_pattern(&pattern.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, pattern.id);

        // Stats
        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_patterns, 1);
    }
}
