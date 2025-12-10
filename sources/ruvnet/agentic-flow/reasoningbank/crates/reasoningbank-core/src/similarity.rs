//! Similarity metrics and vector embedding operations

use ndarray::ArrayView1;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::Pattern;

/// Vector embedding for similarity calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEmbedding {
    /// Dimension of the embedding vector
    pub dimension: usize,

    /// The embedding values
    pub values: Vec<f32>,
}

impl VectorEmbedding {
    /// Create a new embedding
    pub fn new(values: Vec<f32>) -> Self {
        let dimension = values.len();
        Self { dimension, values }
    }

    /// Create a simple embedding from text (basic implementation)
    /// In production, this should use a proper embedding model
    pub fn from_text(text: &str) -> Self {
        // Simple character-based embedding for demonstration
        // In production, use sentence-transformers or similar
        let mut values = vec![0.0; 384]; // Common embedding dimension

        for (i, byte) in text.bytes().enumerate() {
            if i >= values.len() {
                break;
            }
            values[i] = (byte as f32) / 255.0;
        }

        // Normalize
        let norm: f32 = values.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut values {
                *v /= norm;
            }
        }

        Self::new(values)
    }

    /// Calculate cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &VectorEmbedding) -> f32 {
        if self.dimension != other.dimension {
            return 0.0;
        }

        let a = ArrayView1::from(&self.values);
        let b = ArrayView1::from(&other.values);

        let dot_product = a.dot(&b);
        let norm_a = a.dot(&a).sqrt();
        let norm_b = b.dot(&b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Calculate Euclidean distance to another embedding
    pub fn euclidean_distance(&self, other: &VectorEmbedding) -> f32 {
        if self.dimension != other.dimension {
            return f32::INFINITY;
        }

        self.values.iter()
            .zip(&other.values)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Metrics for pattern similarity
#[derive(Debug, Clone)]
pub struct SimilarityMetrics {
    /// Overall similarity score (0.0 to 1.0)
    pub overall_score: f64,

    /// Vector embedding similarity
    pub embedding_similarity: f64,

    /// Category match score
    pub category_match: f64,

    /// Context overlap score
    pub context_overlap: f64,

    /// Success score difference (lower is better)
    pub success_delta: f64,
}

impl SimilarityMetrics {
    /// Calculate comprehensive similarity between two patterns
    pub fn calculate(pattern_a: &Pattern, pattern_b: &Pattern) -> Self {
        // Embedding similarity
        let embedding_similarity = if let (Some(emb_a), Some(emb_b)) =
            (&pattern_a.embedding, &pattern_b.embedding) {
            let vec_a = VectorEmbedding::new(emb_a.clone());
            let vec_b = VectorEmbedding::new(emb_b.clone());
            vec_a.cosine_similarity(&vec_b) as f64
        } else {
            0.5 // Default if no embeddings
        };

        // Category match
        let category_match = if pattern_a.task_category == pattern_b.task_category {
            1.0
        } else {
            0.0
        };

        // Context overlap (Jaccard similarity of keys)
        let keys_a: std::collections::HashSet<_> = pattern_a.context.keys().collect();
        let keys_b: std::collections::HashSet<_> = pattern_b.context.keys().collect();
        let intersection = keys_a.intersection(&keys_b).count();
        let union = keys_a.union(&keys_b).count();
        let context_overlap = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };

        // Success score difference
        let success_delta = (pattern_a.outcome.success_score - pattern_b.outcome.success_score).abs();

        // Weighted overall score
        let overall_score =
            embedding_similarity * 0.5 +
            category_match * 0.2 +
            context_overlap * 0.2 +
            (1.0 - success_delta) * 0.1;

        Self {
            overall_score,
            embedding_similarity,
            category_match,
            context_overlap,
            success_delta,
        }
    }

    /// Check if patterns are similar enough (threshold 0.7)
    pub fn is_similar(&self) -> bool {
        self.overall_score >= 0.7
    }
}

/// Find top-k most similar patterns
pub fn find_similar_patterns(
    query: &Pattern,
    candidates: &[Pattern],
    k: usize,
) -> Vec<(Pattern, SimilarityMetrics)> {
    let mut scored: Vec<_> = candidates.iter()
        .map(|candidate| {
            let metrics = SimilarityMetrics::calculate(query, candidate);
            (candidate.clone(), metrics)
        })
        .collect();

    // Sort by overall score descending
    scored.sort_by_key(|(_, metrics)| OrderedFloat(-metrics.overall_score));

    // Take top k
    scored.into_iter().take(k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TaskOutcome;

    #[test]
    fn test_embedding_creation() {
        let values = vec![1.0, 2.0, 3.0];
        let emb = VectorEmbedding::new(values.clone());
        assert_eq!(emb.dimension, 3);
        assert_eq!(emb.values, values);
    }

    #[test]
    fn test_cosine_similarity() {
        let emb1 = VectorEmbedding::new(vec![1.0, 0.0, 0.0]);
        let emb2 = VectorEmbedding::new(vec![1.0, 0.0, 0.0]);
        let emb3 = VectorEmbedding::new(vec![0.0, 1.0, 0.0]);

        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < 0.001);
        assert!((emb1.cosine_similarity(&emb3)).abs() < 0.001);
    }

    #[test]
    fn test_similarity_metrics() {
        let pattern_a = Pattern::new(
            "Task A".to_string(),
            "category-1".to_string(),
            "strategy-1".to_string(),
        ).with_outcome(TaskOutcome::success(1.0));

        let pattern_b = Pattern::new(
            "Task B".to_string(),
            "category-1".to_string(),
            "strategy-2".to_string(),
        ).with_outcome(TaskOutcome::success(1.0));

        let metrics = SimilarityMetrics::calculate(&pattern_a, &pattern_b);

        assert_eq!(metrics.category_match, 1.0);
        assert!(metrics.overall_score > 0.0);
    }

    #[test]
    fn test_find_similar_patterns() {
        let query = Pattern::new(
            "Query task".to_string(),
            "cat-1".to_string(),
            "strat-1".to_string(),
        );

        let candidates = vec![
            Pattern::new("Candidate 1".to_string(), "cat-1".to_string(), "s1".to_string()),
            Pattern::new("Candidate 2".to_string(), "cat-2".to_string(), "s2".to_string()),
            Pattern::new("Candidate 3".to_string(), "cat-1".to_string(), "s3".to_string()),
        ];

        let similar = find_similar_patterns(&query, &candidates, 2);
        assert_eq!(similar.len(), 2);
    }
}
