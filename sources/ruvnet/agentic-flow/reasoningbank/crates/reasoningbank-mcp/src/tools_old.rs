//! MCP tools and resources implementation

use async_trait::async_trait;
use reasoningbank_core::{Pattern, TaskOutcome, EngineConfig, ReasoningEngine};
use reasoningbank_learning::{AdaptiveLearner, LearningConfig, StrategyOptimizer};
use reasoningbank_storage::SqliteStorage;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

use crate::{McpError, Result};

/// MCP tool interface
#[async_trait]
pub trait McpTool: Send + Sync {
    /// Tool name
    fn name(&self) -> &str;

    /// Tool description
    fn description(&self) -> &str;

    /// Input schema
    fn input_schema(&self) -> JsonValue;

    /// Execute the tool
    async fn execute(&self, params: JsonValue) -> Result<JsonValue>;
}

/// MCP resource interface
#[async_trait]
pub trait McpResource: Send + Sync {
    /// Resource URI pattern
    fn uri_pattern(&self) -> &str;

    /// Resource description
    fn description(&self) -> &str;

    /// Get resource content
    async fn get(&self, uri: &str) -> Result<JsonValue>;
}

/// Tool for storing reasoning patterns
pub struct StorePatternTool {
    learner: Arc<RwLock<AdaptiveLearner>>,
}

impl StorePatternTool {
    pub fn new(learner: Arc<RwLock<AdaptiveLearner>>) -> Self {
        Self { learner }
    }
}

#[async_trait]
impl McpTool for StorePatternTool {
    fn name(&self) -> &str {
        "reasoning_store"
    }

    fn description(&self) -> &str {
        "Store a reasoning pattern with task description, strategy, and outcome"
    }

    fn input_schema(&self) -> JsonValue {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the task"
                },
                "task_category": {
                    "type": "string",
                    "description": "Category of the task"
                },
                "strategy": {
                    "type": "string",
                    "description": "Strategy used for the task"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the task succeeded"
                },
                "success_score": {
                    "type": "number",
                    "description": "Success score from 0.0 to 1.0"
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "Duration in seconds"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context information"
                }
            },
            "required": ["task_description", "task_category", "strategy", "success_score"]
        })
    }

    async fn execute(&self, params: JsonValue) -> Result<JsonValue> {
        #[derive(Deserialize)]
        struct Input {
            task_description: String,
            task_category: String,
            strategy: String,
            success: Option<bool>,
            success_score: f64,
            duration_seconds: Option<f64>,
            context: Option<HashMap<String, JsonValue>>,
        }

        let input: Input = serde_json::from_value(params)
            .map_err(|e| McpError::InvalidParameters(e.to_string()))?;

        let mut pattern = Pattern::new(
            input.task_description,
            input.task_category,
            input.strategy,
        );

        if let Some(ctx) = input.context {
            for (k, v) in ctx {
                pattern = pattern.with_context(k, v);
            }
        }

        let outcome = TaskOutcome {
            success: input.success.unwrap_or(input.success_score >= 0.5),
            success_score: input.success_score,
            duration_seconds: input.duration_seconds.unwrap_or(0.0),
            ..Default::default()
        };

        pattern = pattern.with_outcome(outcome);

        let mut learner = self.learner.write().await;
        let insight = learner.learn_from_task(&pattern)?;

        Ok(serde_json::json!({
            "pattern_id": pattern.id.to_string(),
            "stored": true,
            "insight": {
                "similar_patterns_found": insight.similar_patterns_found,
                "avg_similarity": insight.avg_similarity,
                "improvement_potential": insight.improvement_potential,
                "suggestions": insight.suggested_optimizations
            }
        }))
    }
}

/// Tool for retrieving similar patterns
pub struct RetrievePatternsTool {
    learner: Arc<RwLock<AdaptiveLearner>>,
}

impl RetrievePatternsTool {
    pub fn new(learner: Arc<RwLock<AdaptiveLearner>>) -> Self {
        Self { learner }
    }
}

#[async_trait]
impl McpTool for RetrievePatternsTool {
    fn name(&self) -> &str {
        "reasoning_retrieve"
    }

    fn description(&self) -> &str {
        "Retrieve similar reasoning patterns for a given task"
    }

    fn input_schema(&self) -> JsonValue {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the task"
                },
                "task_category": {
                    "type": "string",
                    "description": "Category of the task"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of similar patterns to retrieve",
                    "default": 5
                }
            },
            "required": ["task_description", "task_category"]
        })
    }

    async fn execute(&self, params: JsonValue) -> Result<JsonValue> {
        #[derive(Deserialize)]
        struct Input {
            task_description: String,
            task_category: String,
            top_k: Option<usize>,
        }

        let input: Input = serde_json::from_value(params)
            .map_err(|e| McpError::InvalidParameters(e.to_string()))?;

        let learner = self.learner.read().await;
        let applied = learner.apply_learning(&input.task_description, &input.task_category)?;

        Ok(serde_json::json!({
            "recommended_strategy": applied.recommended_strategy,
            "similar_patterns_count": applied.similar_patterns_count,
            "confidence": applied.confidence,
            "expected_success": applied.expected_success
        }))
    }
}

/// Tool for analyzing patterns
pub struct AnalyzePatternsTool {
    optimizer: Arc<RwLock<StrategyOptimizer>>,
}

impl AnalyzePatternsTool {
    pub fn new(optimizer: Arc<RwLock<StrategyOptimizer>>) -> Self {
        Self { optimizer }
    }
}

#[async_trait]
impl McpTool for AnalyzePatternsTool {
    fn name(&self) -> &str {
        "reasoning_analyze"
    }

    fn description(&self) -> &str {
        "Analyze patterns and get strategy recommendations for a category"
    }

    fn input_schema(&self) -> JsonValue {
        serde_json::json!({
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Task category to analyze"
                }
            },
            "required": ["category"]
        })
    }

    async fn execute(&self, params: JsonValue) -> Result<JsonValue> {
        #[derive(Deserialize)]
        struct Input {
            category: String,
        }

        let input: Input = serde_json::from_value(params)
            .map_err(|e| McpError::InvalidParameters(e.to_string()))?;

        let optimizer = self.optimizer.read().await;
        let result = optimizer.optimize_for_category(&input.category)?;

        Ok(serde_json::json!({
            "category": result.category,
            "total_patterns": result.total_patterns,
            "recommended_strategy": result.recommended_strategy,
            "strategy_rankings": result.strategy_rankings
        }))
    }
}

/// Tool for optimizing strategies
pub struct OptimizeStrategyTool {
    optimizer: Arc<RwLock<StrategyOptimizer>>,
}

impl OptimizeStrategyTool {
    pub fn new(optimizer: Arc<RwLock<StrategyOptimizer>>) -> Self {
        Self { optimizer }
    }
}

#[async_trait]
impl McpTool for OptimizeStrategyTool {
    fn name(&self) -> &str {
        "reasoning_optimize"
    }

    fn description(&self) -> &str {
        "Get global optimization suggestions across all categories"
    }

    fn input_schema(&self) -> JsonValue {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _params: JsonValue) -> Result<JsonValue> {
        let optimizer = self.optimizer.read().await;
        let results = optimizer.optimize_global()?;

        Ok(serde_json::json!({
            "optimizations": results
        }))
    }
}

/// Resource for accessing stored patterns
pub struct PatternsResource {
    storage: Arc<RwLock<SqliteStorage>>,
}

impl PatternsResource {
    pub fn new(storage: Arc<RwLock<SqliteStorage>>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl McpResource for PatternsResource {
    fn uri_pattern(&self) -> &str {
        "reasoning://patterns/{id}"
    }

    fn description(&self) -> &str {
        "Access stored reasoning patterns by ID"
    }

    async fn get(&self, uri: &str) -> Result<JsonValue> {
        // Extract ID from URI
        let id_str = uri.strip_prefix("reasoning://patterns/")
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))?;

        let id = Uuid::parse_str(id_str)
            .map_err(|e| McpError::InvalidParameters(e.to_string()))?;

        let storage = self.storage.read().await;
        let pattern = storage.get_pattern(&id)?
            .ok_or_else(|| McpError::ResourceNotFound(id_str.to_string()))?;

        Ok(serde_json::to_value(&pattern)?)
    }
}

/// Resource for performance metrics
pub struct MetricsResource {
    learner: Arc<RwLock<AdaptiveLearner>>,
}

impl MetricsResource {
    pub fn new(learner: Arc<RwLock<AdaptiveLearner>>) -> Self {
        Self { learner }
    }
}

#[async_trait]
impl McpResource for MetricsResource {
    fn uri_pattern(&self) -> &str {
        "reasoning://metrics/performance"
    }

    fn description(&self) -> &str {
        "Get performance metrics and learning statistics"
    }

    async fn get(&self, _uri: &str) -> Result<JsonValue> {
        let learner = self.learner.read().await;
        let stats = learner.get_learning_stats()?;

        Ok(serde_json::json!({
            "total_patterns": stats.total_patterns,
            "total_categories": stats.total_categories,
            "total_successes": stats.total_successes,
            "avg_success_score": stats.avg_success_score,
            "success_rate": if stats.total_patterns > 0 {
                stats.total_successes as f64 / stats.total_patterns as f64
            } else {
                0.0
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reasoningbank_core::ReasoningEngine;

    #[tokio::test]
    async fn test_store_pattern_tool() {
        let engine = ReasoningEngine::default();
        let storage = SqliteStorage::in_memory().unwrap();
        let learner = AdaptiveLearner::new(engine, storage, LearningConfig::default());

        let tool = StorePatternTool::new(Arc::new(RwLock::new(learner)));

        assert_eq!(tool.name(), "reasoning_store");
        assert!(!tool.description().is_empty());
    }

    #[tokio::test]
    async fn test_retrieve_patterns_tool() {
        let engine = ReasoningEngine::default();
        let storage = SqliteStorage::in_memory().unwrap();
        let learner = AdaptiveLearner::new(engine, storage, LearningConfig::default());

        let tool = RetrievePatternsTool::new(Arc::new(RwLock::new(learner)));

        assert_eq!(tool.name(), "reasoning_retrieve");
    }
}
