//! # ReasoningBank Learning
//!
//! Adaptive self-learning algorithms with pattern optimization.
//!
//! This crate implements algorithms for learning from task execution history,
//! optimizing strategy selection, and improving performance over time.

pub mod adaptive;
pub mod optimizer;
pub mod async_learner;
pub mod async_learner_v2;

pub use adaptive::{AdaptiveLearner, LearningConfig, LearningInsight, AppliedLearning, LearningStats};
pub use optimizer::{StrategyOptimizer, OptimizationResult, StrategyRanking};
pub use async_learner::AsyncLearner;
pub use async_learner_v2::AsyncLearnerV2;

use thiserror::Error;

/// Result type for learning operations
pub type Result<T> = std::result::Result<T, LearningError>;

/// Errors that can occur in learning operations
#[derive(Error, Debug)]
pub enum LearningError {
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("Storage error: {0}")]
    Storage(#[from] reasoningbank_storage::StorageError),

    #[error("Core error: {0}")]
    Core(#[from] reasoningbank_core::ReasoningError),

    #[error("Internal error: {0}")]
    Internal(String),
}
