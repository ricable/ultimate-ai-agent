//! Adaptive learning from task execution history

use reasoningbank_core::{Pattern, ReasoningEngine};
use reasoningbank_storage::SqliteStorage;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{Result, LearningError};

/// Configuration for adaptive learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Minimum patterns required for learning
    pub min_patterns: usize,

    /// Weight for recent patterns (0.0 to 1.0)
    pub recency_weight: f64,

    /// Weight for success score (0.0 to 1.0)
    pub success_weight: f64,

    /// Weight for usage frequency (0.0 to 1.0)
    pub frequency_weight: f64,

    /// Enable continuous learning
    pub enable_continuous: bool,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            min_patterns: 5,
            recency_weight: 0.3,
            success_weight: 0.5,
            frequency_weight: 0.2,
            enable_continuous: true,
        }
    }
}

/// Adaptive learner that improves from experience
pub struct AdaptiveLearner {
    engine: ReasoningEngine,
    storage: SqliteStorage,
    config: LearningConfig,
}

impl AdaptiveLearner {
    /// Create a new adaptive learner
    pub fn new(
        engine: ReasoningEngine,
        storage: SqliteStorage,
        config: LearningConfig,
    ) -> Self {
        info!("Initializing adaptive learner");
        Self {
            engine,
            storage,
            config,
        }
    }

    /// Learn from a new task execution
    pub fn learn_from_task(&mut self, pattern: &Pattern) -> Result<LearningInsight> {
        debug!("Learning from task: {}", pattern.id);

        // Prepare pattern with embeddings
        let prepared = self.engine.prepare_pattern(pattern.clone())?;

        // Store the pattern
        self.storage.store_pattern(&prepared)?;

        // Find similar historical patterns
        let all_patterns = self.storage.get_all_patterns(Some(1000))?;
        let similar = self.engine.find_similar(&prepared, &all_patterns);

        // Analyze learning insights
        let insight = self.analyze_learning(&prepared, &similar);

        info!("Learning insight generated: improvement potential = {:.2}%",
              insight.improvement_potential * 100.0);

        Ok(insight)
    }

    /// Retrieve and apply learned patterns to a new task
    pub fn apply_learning(&self, task_description: &str, task_category: &str) -> Result<AppliedLearning> {
        debug!("Applying learning for task: {}", task_description);

        // Create a query pattern
        let query = Pattern::new(
            task_description.to_string(),
            task_category.to_string(),
            String::new(), // Strategy to be determined
        );

        let prepared = self.engine.prepare_pattern(query)?;

        // Get patterns from the same category
        let category_patterns = self.storage.get_patterns_by_category(task_category, 100)?;

        if category_patterns.len() < self.config.min_patterns {
            return Err(LearningError::InsufficientData(
                format!("Only {} patterns available, need at least {}",
                        category_patterns.len(), self.config.min_patterns)
            ));
        }

        // Find similar patterns
        let similar = self.engine.find_similar(&prepared, &category_patterns);
        let filtered = self.engine.filter_by_threshold(similar);

        if filtered.is_empty() {
            return Ok(AppliedLearning {
                recommended_strategy: None,
                similar_patterns_count: 0,
                confidence: 0.0,
                expected_success: 0.0,
            });
        }

        // Get strategy recommendation
        let recommendation = self.engine.recommend_strategy(&filtered)
            .ok_or_else(|| LearningError::Internal("Failed to generate recommendation".to_string()))?;

        // Calculate expected success based on historical data
        let expected_success = self.calculate_expected_success(&filtered);

        Ok(AppliedLearning {
            recommended_strategy: Some(recommendation.strategy),
            similar_patterns_count: filtered.len(),
            confidence: recommendation.confidence,
            expected_success,
        })
    }

    /// Analyze learning insights from similar patterns
    fn analyze_learning(
        &self,
        current: &Pattern,
        similar: &[(Pattern, reasoningbank_core::SimilarityMetrics)],
    ) -> LearningInsight {
        if similar.is_empty() {
            return LearningInsight {
                similar_patterns_found: 0,
                avg_similarity: 0.0,
                improvement_potential: 1.0,
                suggested_optimizations: vec![
                    "This is a novel task type - no similar patterns found".to_string(),
                ],
            };
        }

        let avg_similarity = similar.iter()
            .map(|(_, metrics)| metrics.overall_score)
            .sum::<f64>() / similar.len() as f64;

        let avg_historical_success = similar.iter()
            .map(|(pattern, _)| pattern.outcome.success_score)
            .sum::<f64>() / similar.len() as f64;

        let improvement_potential = (1.0 - current.outcome.success_score)
            .max(0.0)
            .min(avg_historical_success);

        let mut optimizations = Vec::new();

        // Suggest optimizations based on analysis
        if current.outcome.success_score < avg_historical_success {
            optimizations.push(format!(
                "Similar tasks achieved {:.1}% success - consider using proven strategies",
                avg_historical_success * 100.0
            ));
        }

        if avg_similarity > 0.8 {
            optimizations.push(
                "High similarity to past tasks - reuse successful patterns".to_string()
            );
        }

        if current.outcome.duration_seconds > 0.0 {
            let avg_duration = similar.iter()
                .map(|(p, _)| p.outcome.duration_seconds)
                .sum::<f64>() / similar.len() as f64;

            if current.outcome.duration_seconds > avg_duration * 1.5 {
                optimizations.push(format!(
                    "Task took {:.1}s vs average {:.1}s - investigate performance",
                    current.outcome.duration_seconds, avg_duration
                ));
            }
        }

        LearningInsight {
            similar_patterns_found: similar.len(),
            avg_similarity,
            improvement_potential,
            suggested_optimizations: optimizations,
        }
    }

    /// Calculate expected success rate based on historical patterns
    fn calculate_expected_success(
        &self,
        similar: &[(Pattern, reasoningbank_core::SimilarityMetrics)],
    ) -> f64 {
        if similar.is_empty() {
            return 0.5; // Default uncertainty
        }

        // Weighted average based on similarity
        let total_weight: f64 = similar.iter()
            .map(|(_, metrics)| metrics.overall_score)
            .sum();

        let weighted_success: f64 = similar.iter()
            .map(|(pattern, metrics)| {
                pattern.outcome.success_score * metrics.overall_score
            })
            .sum();

        if total_weight > 0.0 {
            weighted_success / total_weight
        } else {
            0.5
        }
    }

    /// Get learning statistics
    pub fn get_learning_stats(&self) -> Result<LearningStats> {
        let storage_stats = self.storage.get_stats()?;

        let all_patterns = self.storage.get_all_patterns(None)?;
        let total_successes = all_patterns.iter()
            .filter(|p| p.outcome.success)
            .count();

        let avg_success_score = if !all_patterns.is_empty() {
            all_patterns.iter()
                .map(|p| p.outcome.success_score)
                .sum::<f64>() / all_patterns.len() as f64
        } else {
            0.0
        };

        Ok(LearningStats {
            total_patterns: storage_stats.total_patterns,
            total_categories: storage_stats.total_categories,
            total_successes,
            avg_success_score,
        })
    }
}

/// Insight gained from learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningInsight {
    pub similar_patterns_found: usize,
    pub avg_similarity: f64,
    pub improvement_potential: f64,
    pub suggested_optimizations: Vec<String>,
}

/// Applied learning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedLearning {
    pub recommended_strategy: Option<String>,
    pub similar_patterns_count: usize,
    pub confidence: f64,
    pub expected_success: f64,
}

/// Learning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    pub total_patterns: usize,
    pub total_categories: usize,
    pub total_successes: usize,
    pub avg_success_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use reasoningbank_core::TaskOutcome;

    #[test]
    fn test_learner_creation() {
        let engine = ReasoningEngine::default();
        let storage = SqliteStorage::in_memory().unwrap();
        let config = LearningConfig::default();

        let learner = AdaptiveLearner::new(engine, storage, config);
        let stats = learner.get_learning_stats().unwrap();

        assert_eq!(stats.total_patterns, 0);
    }

    #[test]
    fn test_learn_from_task() {
        let engine = ReasoningEngine::default();
        let storage = SqliteStorage::in_memory().unwrap();
        let config = LearningConfig::default();

        let mut learner = AdaptiveLearner::new(engine, storage, config);

        let pattern = Pattern::new(
            "Test task".to_string(),
            "testing".to_string(),
            "test-strategy".to_string(),
        ).with_outcome(TaskOutcome::success(1.0));

        let insight = learner.learn_from_task(&pattern).unwrap();
        assert!(insight.improvement_potential >= 0.0);
    }
}
