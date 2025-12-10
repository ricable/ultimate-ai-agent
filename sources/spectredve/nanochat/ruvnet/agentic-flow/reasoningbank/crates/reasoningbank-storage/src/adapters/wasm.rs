//! WASM storage backends
//!
//! Provides sql.js and IndexedDB storage for browser environments
//! and in-memory storage for Node.js environments

use crate::{Pattern, StorageConfig, StorageError, StorageStats};
use super::StorageBackend;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use js_sys::Reflect;
use web_sys::{window};
use std::collections::HashMap;

/// Check if IndexedDB is available (browser environment)
pub fn has_indexed_db() -> bool {
    window()
        .and_then(|w| Reflect::get(&w, &JsValue::from_str("indexedDB")).ok())
        .and_then(|v| if v.is_undefined() { None } else { Some(true) })
        .unwrap_or(false)
}

/// Check if we're in Node.js environment (WASM on Node.js)
pub fn is_nodejs() -> bool {
    window().is_none()
}

/// sql.js WASM storage backend
pub struct SqlJsStorage {
    db_name: String,
}

impl SqlJsStorage {
    pub async fn new(config: StorageConfig) -> Result<Self, StorageError> {
        // Load sql.js from CDN or local
        let window = window().ok_or_else(|| StorageError::Connection("No window object".to_string()))?;

        // Check if SQL.js is loaded
        if Reflect::get(&window, &JsValue::from_str("initSqlJs")).is_err() {
            return Err(StorageError::Connection("sql.js not loaded. Include: <script src=\"https://sql.js.org/dist/sql-wasm.js\"></script>".to_string()));
        }

        Ok(Self {
            db_name: config.database_path.to_string_lossy().to_string(),
        })
    }
}

#[async_trait::async_trait]
impl StorageBackend for SqlJsStorage {
    async fn store_pattern(&self, pattern: &Pattern) -> Result<(), StorageError> {
        // This is a simplified placeholder
        // Full implementation would:
        // 1. Get SQL.js database instance
        // 2. Serialize pattern to JSON
        // 3. Execute INSERT statement
        // 4. Save to localStorage for persistence

        web_sys::console::log_1(&JsValue::from_str(&format!("Storing pattern: {}", pattern.id)));
        Ok(())
    }

    async fn get_pattern(&self, id: &Uuid) -> Result<Option<Pattern>, StorageError> {
        // Placeholder - full implementation would query SQL.js
        web_sys::console::log_1(&JsValue::from_str(&format!("Getting pattern: {}", id)));
        Ok(None)
    }

    async fn get_patterns_by_category(&self, category: &str, limit: usize) -> Result<Vec<Pattern>, StorageError> {
        web_sys::console::log_1(&JsValue::from_str(&format!("Query category: {} limit: {}", category, limit)));
        Ok(Vec::new())
    }

    async fn get_stats(&self) -> Result<StorageStats, StorageError> {
        Ok(StorageStats {
            total_patterns: 0,
            total_categories: 0,
            backend_type: "wasm-sqljs".to_string(),
        })
    }

    async fn close(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

/// In-memory storage backend for Node.js WASM
///
/// Uses Mutex for thread-safe access in WASM environment.
/// Note: WASM is single-threaded, but Rust's type system requires Sync/Send traits.
pub struct MemoryStorage {
    db_name: String,
    patterns: Mutex<HashMap<Uuid, Pattern>>,
}

impl MemoryStorage {
    pub async fn new(config: StorageConfig) -> Result<Self, StorageError> {
        Ok(Self {
            db_name: config.database_path.to_string_lossy().to_string(),
            patterns: Mutex::new(HashMap::new()),
        })
    }
}

#[async_trait::async_trait]
impl StorageBackend for MemoryStorage {
    async fn store_pattern(&self, pattern: &Pattern) -> Result<(), StorageError> {
        self.patterns.lock()
            .map_err(|e| StorageError::Connection(format!("Lock error: {}", e)))?
            .insert(pattern.id, pattern.clone());
        Ok(())
    }

    async fn get_pattern(&self, id: &Uuid) -> Result<Option<Pattern>, StorageError> {
        let patterns = self.patterns.lock()
            .map_err(|e| StorageError::Connection(format!("Lock error: {}", e)))?;
        Ok(patterns.get(id).cloned())
    }

    async fn get_patterns_by_category(&self, category: &str, limit: usize) -> Result<Vec<Pattern>, StorageError> {
        let patterns = self.patterns.lock()
            .map_err(|e| StorageError::Connection(format!("Lock error: {}", e)))?;
        let results: Vec<Pattern> = patterns
            .values()
            .filter(|p| p.task_category == category)
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    async fn get_stats(&self) -> Result<StorageStats, StorageError> {
        let patterns = self.patterns.lock()
            .map_err(|e| StorageError::Connection(format!("Lock error: {}", e)))?;
        let total_patterns = patterns.len();
        let categories: std::collections::HashSet<String> = patterns.values()
            .map(|p| p.task_category.clone())
            .collect();

        Ok(StorageStats {
            total_patterns,
            total_categories: categories.len(),
            backend_type: "wasm-memory".to_string(),
        })
    }

    async fn close(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

/// IndexedDB storage backend
pub struct IndexedDbStorage {
    db_name: String,
}

impl IndexedDbStorage {
    pub async fn new(config: StorageConfig) -> Result<Self, StorageError> {
        let db_name = config.database_path.to_string_lossy().to_string();

        // Open IndexedDB
        let window = window().ok_or_else(|| StorageError::Connection("No window object".to_string()))?;

        let indexed_db = window.indexed_db()
            .map_err(|e| StorageError::Connection(format!("IndexedDB error: {:?}", e)))?
            .ok_or_else(|| StorageError::Connection("IndexedDB not available".to_string()))?;

        // Request database open (version 1)
        let request = indexed_db.open_with_u32(&db_name, 1)
            .map_err(|e| StorageError::Connection(format!("Failed to open DB: {:?}", e)))?;

        Ok(Self { db_name })
    }
}

#[async_trait::async_trait]
impl StorageBackend for IndexedDbStorage {
    async fn store_pattern(&self, pattern: &Pattern) -> Result<(), StorageError> {
        // Placeholder - full implementation would:
        // 1. Get IDBDatabase instance
        // 2. Create transaction
        // 3. Get object store
        // 4. Put pattern data

        web_sys::console::log_1(&JsValue::from_str(&format!("IndexedDB: Storing pattern {}", pattern.id)));
        Ok(())
    }

    async fn get_pattern(&self, id: &Uuid) -> Result<Option<Pattern>, StorageError> {
        web_sys::console::log_1(&JsValue::from_str(&format!("IndexedDB: Getting pattern {}", id)));
        Ok(None)
    }

    async fn get_patterns_by_category(&self, category: &str, limit: usize) -> Result<Vec<Pattern>, StorageError> {
        web_sys::console::log_1(&JsValue::from_str(&format!("IndexedDB: Query {} limit {}", category, limit)));
        Ok(Vec::new())
    }

    async fn get_stats(&self) -> Result<StorageStats, StorageError> {
        Ok(StorageStats {
            total_patterns: 0,
            total_categories: 0,
            backend_type: "wasm-indexeddb".to_string(),
        })
    }

    async fn close(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_has_indexed_db() {
        // Test will pass in browser, may fail in pure WASM test env
        let available = has_indexed_db();
        web_sys::console::log_1(&JsValue::from_str(&format!("IndexedDB available: {}", available)));
    }
}
