//! # ReasoningBank Storage
//!
//! Multi-backend storage supporting native and WASM environments.
//!
//! This crate provides persistent storage with indexing, connection pooling,
//! and efficient query capabilities across platforms.

use thiserror::Error;

// Re-export core types
pub use reasoningbank_core::Pattern;

#[cfg(not(target_family = "wasm"))]
pub mod sqlite;
#[cfg(not(target_family = "wasm"))]
pub mod migrations;
#[cfg(not(target_family = "wasm"))]
pub mod async_wrapper;

// Storage adapters (platform-independent)
pub mod adapters;

#[cfg(not(target_family = "wasm"))]
pub use sqlite::{SqliteStorage, ConnectionPool};
#[cfg(not(target_family = "wasm"))]
pub use async_wrapper::AsyncStorage;

pub use adapters::{StorageBackend, auto_detect_storage};

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub database_path: std::path::PathBuf,
    pub max_connections: usize,
    pub enable_wal: bool,
    pub cache_size_kb: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_path: std::path::PathBuf::from(":memory:"),
            max_connections: 10,
            enable_wal: true,
            cache_size_kb: 8192,
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StorageStats {
    pub total_patterns: usize,
    pub total_categories: usize,
    pub backend_type: String,
}

/// Result type for storage operations
pub type Result<T> = std::result::Result<T, StorageError>;

/// Errors that can occur in storage operations
#[derive(Error, Debug)]
pub enum StorageError {
    #[cfg(not(target_family = "wasm"))]
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Pattern not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Pool error: {0}")]
    Pool(String),

    #[error("Migration error: {0}")]
    Migration(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("JSON error: {0}")]
    Json(String),
}

// Manual From implementations for JSON errors
impl From<serde_json::Error> for StorageError {
    fn from(err: serde_json::Error) -> Self {
        StorageError::Json(err.to_string())
    }
}
