//! Pattern types and metadata for reasoning operations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A reasoning pattern with context and outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Unique identifier for the pattern
    pub id: Uuid,

    /// Task description that generated this pattern
    pub task_description: String,

    /// Category of the task (e.g., "api-development", "algorithm-design")
    pub task_category: String,

    /// Strategy used for this task
    pub strategy: String,

    /// Context information at the time of task execution
    pub context: HashMap<String, serde_json::Value>,

    /// Outcome of the task execution
    pub outcome: TaskOutcome,

    /// Metadata about the pattern
    pub metadata: PatternMetadata,

    /// Vector embedding for similarity matching
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl Pattern {
    /// Create a new pattern
    pub fn new(
        task_description: String,
        task_category: String,
        strategy: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            task_description,
            task_category,
            strategy,
            context: HashMap::new(),
            outcome: TaskOutcome::default(),
            metadata: PatternMetadata::default(),
            embedding: None,
        }
    }

    /// Add context information
    pub fn with_context(mut self, key: String, value: serde_json::Value) -> Self {
        self.context.insert(key, value);
        self
    }

    /// Set the outcome
    pub fn with_outcome(mut self, outcome: TaskOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Set the embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Get the success score (0.0 to 1.0)
    pub fn success_score(&self) -> f64 {
        self.outcome.success_score
    }
}

/// Outcome of a task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutcome {
    /// Whether the task succeeded
    pub success: bool,

    /// Success score from 0.0 (complete failure) to 1.0 (perfect success)
    pub success_score: f64,

    /// Duration in seconds
    pub duration_seconds: f64,

    /// Number of tokens used (if applicable)
    pub tokens_used: Option<u32>,

    /// Error message if task failed
    pub error_message: Option<String>,

    /// Additional outcome metrics
    pub metrics: HashMap<String, f64>,
}

impl Default for TaskOutcome {
    fn default() -> Self {
        Self {
            success: false,
            success_score: 0.0,
            duration_seconds: 0.0,
            tokens_used: None,
            error_message: None,
            metrics: HashMap::new(),
        }
    }
}

impl TaskOutcome {
    /// Create a successful outcome
    pub fn success(duration_seconds: f64) -> Self {
        Self {
            success: true,
            success_score: 1.0,
            duration_seconds,
            ..Default::default()
        }
    }

    /// Create a failed outcome
    pub fn failure(error_message: String, duration_seconds: f64) -> Self {
        Self {
            success: false,
            success_score: 0.0,
            duration_seconds,
            error_message: Some(error_message),
            ..Default::default()
        }
    }

    /// Create a partial success outcome
    pub fn partial(success_score: f64, duration_seconds: f64) -> Self {
        Self {
            success: success_score >= 0.5,
            success_score: success_score.clamp(0.0, 1.0),
            duration_seconds,
            ..Default::default()
        }
    }
}

/// Metadata about a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMetadata {
    /// When the pattern was created
    pub created_at: i64,

    /// When the pattern was last updated
    pub updated_at: i64,

    /// Number of times this pattern has been used
    pub usage_count: u32,

    /// Average success score across all uses
    pub avg_success_score: f64,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Custom metadata
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for PatternMetadata {
    fn default() -> Self {
        // WASM doesn't support SystemTime, use a static timestamp
        #[cfg(not(target_family = "wasm"))]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        #[cfg(target_family = "wasm")]
        let now = 0i64; // WASM: timestamps not supported, use 0 as placeholder

        Self {
            created_at: now,
            updated_at: now,
            usage_count: 0,
            avg_success_score: 0.0,
            tags: Vec::new(),
            custom: HashMap::new(),
        }
    }
}

impl PatternMetadata {
    /// Update the usage statistics
    pub fn record_usage(&mut self, success_score: f64) {
        self.usage_count += 1;

        // Update average success score using incremental average
        let n = self.usage_count as f64;
        self.avg_success_score = ((n - 1.0) * self.avg_success_score + success_score) / n;

        // Update timestamp (WASM doesn't support SystemTime)
        #[cfg(not(target_family = "wasm"))]
        {
            self.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64;
        }

        #[cfg(target_family = "wasm")]
        {
            self.updated_at = 0; // WASM: timestamps not supported
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_creation() {
        let pattern = Pattern::new(
            "Test task".to_string(),
            "testing".to_string(),
            "test-strategy".to_string(),
        );

        assert!(!pattern.id.is_nil());
        assert_eq!(pattern.task_category, "testing");
    }

    #[test]
    fn test_outcome_success() {
        let outcome = TaskOutcome::success(1.5);
        assert!(outcome.success);
        assert_eq!(outcome.success_score, 1.0);
    }

    #[test]
    fn test_metadata_usage_tracking() {
        let mut metadata = PatternMetadata::default();

        metadata.record_usage(0.8);
        assert_eq!(metadata.usage_count, 1);
        assert_eq!(metadata.avg_success_score, 0.8);

        metadata.record_usage(0.6);
        assert_eq!(metadata.usage_count, 2);
        assert_eq!(metadata.avg_success_score, 0.7);
    }
}
