//! # ReasoningBank WASM
//!
//! WebAssembly bindings for browser and Node.js environments.
//!
//! This crate provides WASM bindings for the ReasoningBank, enabling
//! reasoning capabilities in web browsers and Node.js applications.

use reasoningbank_core::{Pattern, TaskOutcome, ReasoningEngine, EngineConfig};
use reasoningbank_storage::{StorageBackend, StorageConfig};
use reasoningbank_storage::adapters::wasm::{IndexedDbStorage, SqlJsStorage, MemoryStorage};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use web_sys::console;

/// Initialize logging for WASM
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// WASM wrapper for ReasoningBank
#[wasm_bindgen]
pub struct ReasoningBankWasm {
    engine: ReasoningEngine,
    storage: Arc<dyn StorageBackend>,
}

#[wasm_bindgen]
impl ReasoningBankWasm {
    /// Create a new ReasoningBank instance
    #[wasm_bindgen(constructor)]
    pub async fn new(db_name: Option<String>) -> Result<ReasoningBankWasm, JsValue> {
        let name = db_name.unwrap_or_else(|| "reasoningbank".to_string());

        let config = StorageConfig {
            database_path: std::path::PathBuf::from(name),
            max_connections: 1,
            enable_wal: false,
            cache_size_kb: 2048,
        };

        // Auto-detect environment and select appropriate storage backend:
        // 1. Node.js (no window object) -> MemoryStorage
        // 2. Browser with IndexedDB -> IndexedDbStorage
        // 3. Browser without IndexedDB -> SqlJsStorage
        let storage: Arc<dyn StorageBackend> = if reasoningbank_storage::adapters::wasm::is_nodejs() {
            // Node.js environment - use in-memory storage
            let db = MemoryStorage::new(config).await
                .map_err(|e| JsValue::from_str(&format!("Memory storage error: {}", e)))?;
            Arc::new(db)
        } else if reasoningbank_storage::adapters::wasm::has_indexed_db() {
            // Browser with IndexedDB support
            let db = IndexedDbStorage::new(config).await
                .map_err(|e| JsValue::from_str(&format!("IndexedDB error: {}", e)))?;
            Arc::new(db)
        } else {
            // Browser fallback to sql.js
            let db = SqlJsStorage::new(config).await
                .map_err(|e| JsValue::from_str(&format!("sql.js error: {}", e)))?;
            Arc::new(db)
        };

        let engine = ReasoningEngine::default();

        Ok(Self { engine, storage })
    }

    /// Store a reasoning pattern
    #[wasm_bindgen(js_name = storePattern)]
    pub async fn store_pattern(&self, pattern_json: &str) -> Result<String, JsValue> {
        #[derive(Deserialize)]
        struct PatternInput {
            task_description: String,
            task_category: String,
            strategy: String,
            success_score: f64,
            duration_seconds: Option<f64>,
        }

        let input: PatternInput = serde_json::from_str(pattern_json)
            .map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;

        let outcome = TaskOutcome::partial(
            input.success_score,
            input.duration_seconds.unwrap_or(0.0),
        );

        let pattern = Pattern::new(
            input.task_description,
            input.task_category,
            input.strategy,
        ).with_outcome(outcome);

        let prepared = self.engine.prepare_pattern(pattern)
            .map_err(|e| JsValue::from_str(&format!("Engine error: {}", e)))?;

        let id = prepared.id;

        self.storage.store_pattern(&prepared).await
            .map_err(|e| JsValue::from_str(&format!("Storage error: {}", e)))?;

        Ok(id.to_string())
    }

    /// Retrieve a pattern by ID
    #[wasm_bindgen(js_name = getPattern)]
    pub async fn get_pattern(&self, id: &str) -> Result<String, JsValue> {
        let uuid = uuid::Uuid::parse_str(id)
            .map_err(|e| JsValue::from_str(&format!("Invalid ID: {}", e)))?;

        let pattern = self.storage.get_pattern(&uuid).await
            .map_err(|e| JsValue::from_str(&format!("Storage error: {}", e)))?
            .ok_or_else(|| JsValue::from_str("Pattern not found"))?;

        let json = serde_json::to_string(&pattern)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))?;

        Ok(json)
    }

    /// Search patterns by category
    #[wasm_bindgen(js_name = searchByCategory)]
    pub async fn search_by_category(&self, category: &str, limit: usize) -> Result<String, JsValue> {
        let patterns = self.storage.get_patterns_by_category(category, limit).await
            .map_err(|e| JsValue::from_str(&format!("Storage error: {}", e)))?;

        let json = serde_json::to_string(&patterns)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))?;

        Ok(json)
    }

    /// Find similar patterns
    #[wasm_bindgen(js_name = findSimilar)]
    pub async fn find_similar(&self, task_description: &str, task_category: &str, top_k: usize) -> Result<String, JsValue> {
        let query = Pattern::new(
            task_description.to_string(),
            task_category.to_string(),
            String::new(),
        );

        let prepared = self.engine.prepare_pattern(query)
            .map_err(|e| JsValue::from_str(&format!("Engine error: {}", e)))?;

        let candidates = self.storage.get_patterns_by_category(task_category, 1000).await
            .map_err(|e| JsValue::from_str(&format!("Storage error: {}", e)))?;

        let similar = self.engine.find_similar(&prepared, &candidates);

        #[derive(Serialize)]
        struct SimilarPattern {
            pattern: Pattern,
            similarity_score: f64,
        }

        let results: Vec<SimilarPattern> = similar.into_iter()
            .take(top_k)
            .map(|(pattern, metrics)| SimilarPattern {
                pattern,
                similarity_score: metrics.overall_score,
            })
            .collect();

        let json = serde_json::to_string(&results)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))?;

        Ok(json)
    }

    /// Get storage statistics
    #[wasm_bindgen(js_name = getStats)]
    pub async fn get_stats(&self) -> Result<String, JsValue> {
        let stats = self.storage.get_stats().await
            .map_err(|e| JsValue::from_str(&format!("Storage error: {}", e)))?;

        let json = serde_json::to_string(&stats)
            .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))?;

        Ok(json)
    }
}

/// Log a message to the browser console
#[wasm_bindgen]
pub fn log(message: &str) {
    console::log_1(&JsValue::from_str(message));
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_reasoningbank_creation() {
        let rb = ReasoningBankWasm::new(None).unwrap();
        let stats = rb.get_stats().unwrap();
        assert!(stats.contains("total_patterns"));
    }
}
