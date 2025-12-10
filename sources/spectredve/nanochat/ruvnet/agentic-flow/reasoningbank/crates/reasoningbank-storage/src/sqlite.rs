//! SQLite storage implementation with connection pooling

use parking_lot::RwLock;
use reasoningbank_core::Pattern;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::{migrations, Result};

/// Configuration for SQLite storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Path to the database file (or ":memory:" for in-memory)
    pub database_path: PathBuf,

    /// Maximum number of connections in pool
    pub max_connections: usize,

    /// Enable WAL mode for better concurrency
    pub enable_wal: bool,

    /// Page cache size in KB
    pub cache_size_kb: i32,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("reasoningbank.db"),
            max_connections: 10,
            enable_wal: true,
            cache_size_kb: 8192, // 8MB
        }
    }
}

impl StorageConfig {
    /// Create an in-memory configuration
    pub fn in_memory() -> Self {
        Self {
            database_path: PathBuf::from(":memory:"),
            max_connections: 1,
            enable_wal: false,
            cache_size_kb: 4096,
        }
    }
}

/// Connection pool for SQLite
pub struct ConnectionPool {
    config: Arc<StorageConfig>,
    connections: Arc<RwLock<Vec<Connection>>>,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(config: StorageConfig) -> Result<Self> {
        let mut connections = Vec::new();

        // Create initial connections
        for _ in 0..config.max_connections.min(3) {
            let conn = create_connection(&config)?;
            connections.push(conn);
        }

        Ok(Self {
            config: Arc::new(config),
            connections: Arc::new(RwLock::new(connections)),
        })
    }

    /// Get a connection from the pool
    pub fn get(&self) -> Result<PooledConnection> {
        let mut pool = self.connections.write();

        if let Some(conn) = pool.pop() {
            Ok(PooledConnection {
                conn: Some(conn),
                pool: Arc::clone(&self.connections),
            })
        } else {
            // Create a new connection if pool is empty
            let conn = create_connection(&self.config)?;
            Ok(PooledConnection {
                conn: Some(conn),
                pool: Arc::clone(&self.connections),
            })
        }
    }
}

/// A pooled connection that returns to the pool on drop
pub struct PooledConnection {
    conn: Option<Connection>,
    pool: Arc<RwLock<Vec<Connection>>>,
}

impl std::ops::Deref for PooledConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        self.conn.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for PooledConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.conn.as_mut().unwrap()
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            let mut pool = self.pool.write();
            pool.push(conn);
        }
    }
}

/// Create and configure a connection
fn create_connection(config: &StorageConfig) -> Result<Connection> {
    let conn = Connection::open(&config.database_path)?;

    // Enable WAL mode for better concurrency
    if config.enable_wal && config.database_path.to_str() != Some(":memory:") {
        // Use pragma_update for PRAGMA statements that return values
        conn.pragma_update(None, "journal_mode", "WAL")?;
    }

    // Set cache size (use pragma_update to avoid return values)
    conn.pragma_update(None, "cache_size", &format!("-{}", config.cache_size_kb))?;

    // Other optimizations
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    conn.pragma_update(None, "mmap_size", "30000000000")?;

    // Run migrations
    migrations::migrate(&conn)?;

    Ok(conn)
}

/// SQLite storage implementation
pub struct SqliteStorage {
    pool: ConnectionPool,
}

impl SqliteStorage {
    /// Create a new SQLite storage
    pub fn new(config: StorageConfig) -> Result<Self> {
        info!("Initializing SQLite storage at: {:?}", config.database_path);
        let pool = ConnectionPool::new(config)?;
        Ok(Self { pool })
    }

    /// Create with default configuration
    pub fn default() -> Result<Self> {
        Self::new(StorageConfig::default())
    }

    /// Create an in-memory database (for testing)
    pub fn in_memory() -> Result<Self> {
        let config = StorageConfig {
            database_path: PathBuf::from(":memory:"),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Store a pattern
    pub fn store_pattern(&self, pattern: &Pattern) -> Result<()> {
        debug!("Storing pattern: {}", pattern.id);

        let conn = self.pool.get()?;

        // Serialize complex fields
        let context_json = serde_json::to_string(&pattern.context)?;
        let outcome_json = serde_json::to_string(&pattern.outcome)?;
        let metadata_json = serde_json::to_string(&pattern.metadata)?;

        // Serialize embedding if present
        let embedding_blob = pattern.embedding.as_ref().map(|emb| {
            let bytes: Vec<u8> = emb.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();
            bytes
        });

        conn.execute(
            "INSERT OR REPLACE INTO patterns
             (id, task_description, task_category, strategy, context, outcome,
              metadata, embedding, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                pattern.id.to_string(),
                pattern.task_description,
                pattern.task_category,
                pattern.strategy,
                context_json,
                outcome_json,
                metadata_json,
                embedding_blob,
                pattern.metadata.created_at,
                pattern.metadata.updated_at,
            ],
        )?;

        Ok(())
    }

    /// Retrieve a pattern by ID
    pub fn get_pattern(&self, id: &Uuid) -> Result<Option<Pattern>> {
        debug!("Retrieving pattern: {}", id);

        let conn = self.pool.get()?;

        let pattern = conn.query_row(
            "SELECT id, task_description, task_category, strategy, context,
                    outcome, metadata, embedding
             FROM patterns WHERE id = ?1",
            params![id.to_string()],
            |row| {
                let context_json: String = row.get(4)?;
                let outcome_json: String = row.get(5)?;
                let metadata_json: String = row.get(6)?;
                let embedding_blob: Option<Vec<u8>> = row.get(7)?;

                let context = serde_json::from_str(&context_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                let outcome = serde_json::from_str(&outcome_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                let metadata = serde_json::from_str(&metadata_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                let embedding = embedding_blob.map(|bytes| {
                    bytes.chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect()
                });

                Ok(Pattern {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                    task_description: row.get(1)?,
                    task_category: row.get(2)?,
                    strategy: row.get(3)?,
                    context,
                    outcome,
                    metadata,
                    embedding,
                })
            },
        ).optional()?;

        Ok(pattern)
    }

    /// Get patterns by category
    pub fn get_patterns_by_category(&self, category: &str, limit: usize) -> Result<Vec<Pattern>> {
        debug!("Retrieving patterns for category: {}", category);

        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT id, task_description, task_category, strategy, context,
                    outcome, metadata, embedding
             FROM patterns
             WHERE task_category = ?1
             ORDER BY created_at DESC
             LIMIT ?2"
        )?;

        let patterns = stmt.query_map(params![category, limit as i64], row_to_pattern)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(patterns)
    }

    /// Search patterns by text
    pub fn search_patterns(&self, query: &str, limit: usize) -> Result<Vec<Pattern>> {
        debug!("Searching patterns: {}", query);

        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT p.id, p.task_description, p.task_category, p.strategy,
                    p.context, p.outcome, p.metadata, p.embedding
             FROM patterns p
             INNER JOIN patterns_fts fts ON p.id = fts.id
             WHERE patterns_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2"
        )?;

        let patterns = stmt.query_map(params![query, limit as i64], row_to_pattern)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(patterns)
    }

    /// Get all patterns (for similarity search)
    pub fn get_all_patterns(&self, limit: Option<usize>) -> Result<Vec<Pattern>> {
        let conn = self.pool.get()?;

        let query = if let Some(lim) = limit {
            format!(
                "SELECT id, task_description, task_category, strategy, context,
                        outcome, metadata, embedding
                 FROM patterns
                 ORDER BY created_at DESC
                 LIMIT {}",
                lim
            )
        } else {
            "SELECT id, task_description, task_category, strategy, context,
                    outcome, metadata, embedding
             FROM patterns
             ORDER BY created_at DESC".to_string()
        };

        let mut stmt = conn.prepare(&query)?;
        let patterns = stmt.query_map([], row_to_pattern)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(patterns)
    }

    /// Delete a pattern
    pub fn delete_pattern(&self, id: &Uuid) -> Result<()> {
        debug!("Deleting pattern: {}", id);

        let conn = self.pool.get()?;
        conn.execute(
            "DELETE FROM patterns WHERE id = ?1",
            params![id.to_string()],
        )?;

        Ok(())
    }

    /// Delete patterns older than specified days
    pub fn delete_old_patterns(&self, days: i64) -> Result<usize> {
        let conn = self.pool.get()?;
        let rows = conn.execute(
            "DELETE FROM patterns WHERE created_at < datetime('now', '-' || ?1 || ' days')",
            params![days],
        )?;
        Ok(rows)
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> Result<StorageStats> {
        let conn = self.pool.get()?;

        let total_patterns: i64 = conn.query_row(
            "SELECT COUNT(*) FROM patterns",
            [],
            |row| row.get(0),
        )?;

        let total_categories: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT task_category) FROM patterns",
            [],
            |row| row.get(0),
        )?;

        let db_size_bytes = if let Ok(metadata) = std::fs::metadata(&conn.path().unwrap()) {
            metadata.len()
        } else {
            0
        };

        Ok(StorageStats {
            total_patterns: total_patterns as usize,
            total_categories: total_categories as usize,
            database_size_bytes: db_size_bytes,
        })
    }
}

/// Helper function to convert a row to a Pattern
fn row_to_pattern(row: &rusqlite::Row) -> rusqlite::Result<Pattern> {
    let context_json: String = row.get(4)?;
    let outcome_json: String = row.get(5)?;
    let metadata_json: String = row.get(6)?;
    let embedding_blob: Option<Vec<u8>> = row.get(7)?;

    let context = serde_json::from_str(&context_json)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let outcome = serde_json::from_str(&outcome_json)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    let metadata = serde_json::from_str(&metadata_json)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

    let embedding = embedding_blob.map(|bytes| {
        bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    });

    Ok(Pattern {
        id: Uuid::parse_str(&row.get::<_, String>(0)?)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
        task_description: row.get(1)?,
        task_category: row.get(2)?,
        strategy: row.get(3)?,
        context,
        outcome,
        metadata,
        embedding,
    })
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_patterns: usize,
    pub total_categories: usize,
    pub database_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use reasoningbank_core::TaskOutcome;

    #[test]
    fn test_storage_create() {
        let storage = SqliteStorage::in_memory().unwrap();
        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.total_patterns, 0);
    }

    #[test]
    fn test_store_and_retrieve() {
        let storage = SqliteStorage::in_memory().unwrap();

        let pattern = Pattern::new(
            "Test task".to_string(),
            "testing".to_string(),
            "test-strategy".to_string(),
        ).with_outcome(TaskOutcome::success(1.0));

        let id = pattern.id;

        storage.store_pattern(&pattern).unwrap();

        let retrieved = storage.get_pattern(&id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().task_description, "Test task");
    }

    #[test]
    fn test_search_by_category() {
        let storage = SqliteStorage::in_memory().unwrap();

        let pattern = Pattern::new(
            "Test task".to_string(),
            "testing".to_string(),
            "test-strategy".to_string(),
        );

        storage.store_pattern(&pattern).unwrap();

        let results = storage.get_patterns_by_category("testing", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_delete_pattern() {
        let storage = SqliteStorage::in_memory().unwrap();

        let pattern = Pattern::new(
            "Test task".to_string(),
            "testing".to_string(),
            "test-strategy".to_string(),
        );

        let id = pattern.id;
        storage.store_pattern(&pattern).unwrap();

        storage.delete_pattern(&id).unwrap();

        let retrieved = storage.get_pattern(&id).unwrap();
        assert!(retrieved.is_none());
    }
}
