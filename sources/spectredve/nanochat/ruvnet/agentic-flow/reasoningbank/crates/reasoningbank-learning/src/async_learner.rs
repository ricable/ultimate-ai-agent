//! Async wrapper for AdaptiveLearner to enable Send + Sync async operations
//!
//! NOTE: This is a legacy wrapper. Use AsyncLearnerV2 for new code.

use crate::{LearningConfig, LearningInsight, AppliedLearning, LearningStats, Result, AsyncLearnerV2};
use reasoningbank_core::{ Pattern, EngineConfig};
use reasoningbank_storage::sqlite::StorageConfig;

/// Async-safe wrapper around AdaptiveLearner (legacy)
///
/// This type is deprecated. Use AsyncLearnerV2 instead.
#[derive(Clone)]
#[deprecated(note = "Use AsyncLearnerV2 instead")]
pub struct AsyncLearner {
    inner: AsyncLearnerV2,
}

impl AsyncLearner {
    /// Create a new async learner
    pub async fn new(
        engine_config: EngineConfig,
        storage_config: StorageConfig,
        learning_config: LearningConfig,
    ) -> Result<Self> {
        let inner = AsyncLearnerV2::new(engine_config, storage_config, learning_config).await?;
        Ok(Self { inner })
    }

    /// Create with default configuration
    pub async fn default() -> Result<Self> {
        let inner = AsyncLearnerV2::default().await?;
        Ok(Self { inner })
    }

    /// Learn from a new task execution
    pub async fn learn_from_task(&self, pattern: &Pattern) -> Result<LearningInsight> {
        self.inner.learn_from_task(pattern).await
    }

    /// Retrieve and apply learned patterns to a new task
    pub async fn apply_learning(&self, task_description: &str, task_category: &str) -> Result<AppliedLearning> {
        self.inner.apply_learning(task_description, task_category).await
    }

    /// Get learning statistics
    pub async fn get_learning_stats(&self) -> Result<LearningStats> {
        self.inner.get_learning_stats().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reasoningbank_core::TaskOutcome;

    #[tokio::test]
    async fn test_async_learner() {
        let learner = AsyncLearner::default().await.unwrap();

        let pattern = Pattern::new(
            "Test task".to_string(),
            "testing".to_string(),
            "test-strategy".to_string(),
        ).with_outcome(TaskOutcome::success(1.0));

        let insight = learner.learn_from_task(&pattern).await.unwrap();
        assert!(insight.improvement_potential >= 0.0);
    }
}
