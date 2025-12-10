//! Async wrapper using channel-based message passing for Send + Sync

use crate::{AdaptiveLearner, LearningConfig, LearningInsight, AppliedLearning, LearningStats, Result};
use reasoningbank_core::{Pattern, ReasoningEngine, EngineConfig};
use reasoningbank_storage::sqlite::StorageConfig;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

/// Commands that can be sent to the learner actor
enum LearnerCommand {
    LearnFromTask {
        pattern: Pattern,
        respond_to: oneshot::Sender<Result<LearningInsight>>,
    },
    ApplyLearning {
        task_description: String,
        task_category: String,
        respond_to: oneshot::Sender<Result<AppliedLearning>>,
    },
    GetStats {
        respond_to: oneshot::Sender<Result<LearningStats>>,
    },
}

/// Async-safe learner using actor pattern
#[derive(Clone)]
pub struct AsyncLearnerV2 {
    sender: mpsc::Sender<LearnerCommand>,
}

impl AsyncLearnerV2 {
    /// Create a new async learner with actor pattern
    pub async fn new(
        engine_config: EngineConfig,
        storage_config: StorageConfig,
        learning_config: LearningConfig,
    ) -> Result<Self> {
        let (tx, mut rx) = mpsc::channel::<LearnerCommand>(100);
        let (init_tx, init_rx) = oneshot::channel::<std::result::Result<(), String>>();

        // Spawn dedicated thread for sync SQLite operations
        std::thread::spawn(move || {
            // Initialize learner in this thread (avoids Send requirement)
            let engine = ReasoningEngine::new(engine_config);
            let storage = match reasoningbank_storage::SqliteStorage::new(storage_config) {
                Ok(s) => s,
                Err(e) => {
                    let _ = init_tx.send(Err(format!("Storage init failed: {}", e)));
                    return;
                }
            };
            let mut learner = AdaptiveLearner::new(engine, storage, learning_config);

            // Signal successful initialization
            let _ = init_tx.send(Ok(()));

            // Process commands in blocking mode
            while let Some(cmd) = rx.blocking_recv() {
                match cmd {
                    LearnerCommand::LearnFromTask { pattern, respond_to } => {
                        let result = learner.learn_from_task(&pattern);
                        let _ = respond_to.send(result);
                    }
                    LearnerCommand::ApplyLearning { task_description, task_category, respond_to } => {
                        let result = learner.apply_learning(&task_description, &task_category);
                        let _ = respond_to.send(result);
                    }
                    LearnerCommand::GetStats { respond_to } => {
                        let result = learner.get_learning_stats();
                        let _ = respond_to.send(result);
                    }
                }
            }
        });

        // Wait for initialization
        init_rx.await
            .map_err(|_| crate::LearningError::Internal("Init channel closed".to_string()))?
            .map_err(|e| crate::LearningError::Internal(e))?;

        Ok(Self { sender: tx })
    }

    /// Create with default configuration
    pub async fn default() -> Result<Self> {
        Self::new(
            EngineConfig::default(),
            StorageConfig::default(),
            LearningConfig::default(),
        ).await
    }

    /// Learn from a new task execution
    pub async fn learn_from_task(&self, pattern: &Pattern) -> Result<LearningInsight> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(LearnerCommand::LearnFromTask {
                pattern: pattern.clone(),
                respond_to: tx,
            })
            .await
            .map_err(|_| crate::LearningError::Internal("Learner actor died".to_string()))?;

        rx.await
            .map_err(|_| crate::LearningError::Internal("Response channel closed".to_string()))?
    }

    /// Apply learning to a new task
    pub async fn apply_learning(&self, task_description: &str, task_category: &str) -> Result<AppliedLearning> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(LearnerCommand::ApplyLearning {
                task_description: task_description.to_string(),
                task_category: task_category.to_string(),
                respond_to: tx,
            })
            .await
            .map_err(|_| crate::LearningError::Internal("Learner actor died".to_string()))?;

        rx.await
            .map_err(|_| crate::LearningError::Internal("Response channel closed".to_string()))?
    }

    /// Get learning statistics
    pub async fn get_learning_stats(&self) -> Result<LearningStats> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(LearnerCommand::GetStats { respond_to: tx })
            .await
            .map_err(|_| crate::LearningError::Internal("Learner actor died".to_string()))?;

        rx.await
            .map_err(|_| crate::LearningError::Internal("Response channel closed".to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reasoningbank_core::TaskOutcome;

    #[tokio::test]
    async fn test_async_learner_v2() {
        let learner = AsyncLearnerV2::default().await.unwrap();

        let pattern = Pattern::new(
            "Test task".to_string(),
            "testing".to_string(),
            "test-strategy".to_string(),
        ).with_outcome(TaskOutcome::success(1.0));

        let insight = learner.learn_from_task(&pattern).await.unwrap();
        assert!(insight.improvement_potential >= 0.0);
    }
}
