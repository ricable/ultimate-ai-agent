//! MCP tools and resources implementation with async wrappers

use async_trait::async_trait;
use reasoningbank_core::{Pattern, TaskOutcome};
use reasoningbank_learning::{AsyncLearnerV2, LearningInsight, AppliedLearning};
use reasoningbank_storage::AsyncStorage;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

use crate::{McpError, Result};

/// MCP tool interface
#[async_trait]
pub trait McpTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> JsonValue;
    async fn execute(&self, params: JsonValue) -> Result<JsonValue>;
}

/// MCP resource interface
#[async_trait]
pub trait McpResource: Send + Sync {
    fn uri_pattern(&self) -> &str;
    fn description(&self) -> &str;
    async fn get(&self, uri: &str) -> Result<JsonValue>;
}

/// Tool for storing reasoning patterns
pub struct StorePatternTool {
    learner: AsyncLearnerV2,
}

impl StorePatternTool {
    pub fn new(learner: AsyncLearnerV2) -> Self {
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
                "task_description": {"type": "string"},
                "task_category": {"type": "string"},
                "strategy": {"type": "string"},
                "success": {"type": "boolean"},
                "success_score": {"type": "number"},
                "duration_seconds": {"type": "number"},
                "context": {"type": "object"}
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

        let input: Input = serde_json::from_value(params)?;

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

        let insight = self.learner.learn_from_task(&pattern).await?;

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
    learner: AsyncLearnerV2,
}

impl RetrievePatternsTool {
    pub fn new(learner: AsyncLearnerV2) -> Self {
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
                "task_description": {"type": "string"},
                "task_category": {"type": "string"},
                "top_k": {"type": "integer", "default": 5}
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

        let input: Input = serde_json::from_value(params)?;

        let applied = self.learner.apply_learning(&input.task_description, &input.task_category).await?;

        Ok(serde_json::json!({
            "recommended_strategy": applied.recommended_strategy,
            "similar_patterns_count": applied.similar_patterns_count,
            "confidence": applied.confidence,
            "expected_success": applied.expected_success
        }))
    }
}

/// Tool for analyzing patterns (simplified - uses AsyncStorage directly)
pub struct AnalyzePatternsTool {
    storage: AsyncStorage,
}

impl AnalyzePatternsTool {
    pub fn new(storage: AsyncStorage) -> Self {
        Self { storage }
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
                "category": {"type": "string"}
            },
            "required": ["category"]
        })
    }

    async fn execute(&self, params: JsonValue) -> Result<JsonValue> {
        #[derive(Deserialize)]
        struct Input {
            category: String,
        }

        let input: Input = serde_json::from_value(params)?;

        let patterns = self.storage.get_patterns_by_category(&input.category, 1000).await?;

        Ok(serde_json::json!({
            "category": input.category,
            "total_patterns": patterns.len(),
            "recommended_strategy": "analysis-pending",
            "strategy_rankings": []
        }))
    }
}

/// Tool for optimizing strategies (simplified)
pub struct OptimizeStrategyTool {
    storage: AsyncStorage,
}

impl OptimizeStrategyTool {
    pub fn new(storage: AsyncStorage) -> Self {
        Self { storage }
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
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn execute(&self, _params: JsonValue) -> Result<JsonValue> {
        let stats = self.storage.get_stats().await?;

        Ok(serde_json::json!({
            "optimizations": [{
                "category": "global",
                "total_patterns": stats.total_patterns,
                "recommendation": "Continue collecting patterns"
            }]
        }))
    }
}

/// Resource for accessing stored patterns
pub struct PatternsResource {
    storage: AsyncStorage,
}

impl PatternsResource {
    pub fn new(storage: AsyncStorage) -> Self {
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
        let id_str = uri.strip_prefix("reasoning://patterns/")
            .ok_or_else(|| McpError::ResourceNotFound(uri.to_string()))?;

        let id = Uuid::parse_str(id_str)
            .map_err(|e| McpError::InvalidParameters(e.to_string()))?;

        let pattern = self.storage.get_pattern(&id).await?
            .ok_or_else(|| McpError::ResourceNotFound(id_str.to_string()))?;

        Ok(serde_json::to_value(&pattern)?)
    }
}

/// Resource for performance metrics
pub struct MetricsResource {
    learner: AsyncLearnerV2,
}

impl MetricsResource {
    pub fn new(learner: AsyncLearnerV2) -> Self {
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
        let stats = self.learner.get_learning_stats().await?;

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
