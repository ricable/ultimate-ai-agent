use std::collections::HashMap;
use ndarray::{Array2, Array3, Axis, s};
use rand::Rng;

use super::EvidenceItem;

/// Cross-attention mechanism for evidence correlation
pub struct CrossAttentionMechanism {
    hidden_dim: usize,
    num_heads: usize,
    dropout_rate: f32,
    // Weight matrices
    w_query: Array2<f32>,
    w_key: Array2<f32>,
    w_value: Array2<f32>,
    w_output: Array2<f32>,
}

impl CrossAttentionMechanism {
    pub fn new(hidden_dim: usize, num_heads: usize, dropout_rate: f32) -> Self {
        let head_dim = hidden_dim / num_heads;
        let mut rng = rand::thread_rng();

        // Initialize weight matrices with Xavier initialization
        let scale = (2.0 / hidden_dim as f32).sqrt();
        
        Self {
            hidden_dim,
            num_heads,
            dropout_rate,
            w_query: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            w_key: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            w_value: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            w_output: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
        }
    }

    /// Compute cross-attention between evidence items
    pub fn compute_attention(&self, evidence_items: &[EvidenceItem]) -> Vec<Vec<f32>> {
        if evidence_items.is_empty() {
            return Vec::new();
        }

        // Convert evidence features to embeddings
        let embeddings = self.create_embeddings(evidence_items);
        
        // Compute multi-head attention
        let attention_output = self.multi_head_attention(&embeddings);
        
        // Apply cross-domain attention
        let cross_domain_features = self.cross_domain_attention(
            &attention_output,
            evidence_items,
        );

        cross_domain_features
    }

    /// Create embeddings from evidence features
    fn create_embeddings(&self, evidence_items: &[EvidenceItem]) -> Array2<f32> {
        let n_items = evidence_items.len();
        let mut embeddings = Array2::zeros((n_items, self.hidden_dim));

        for (i, item) in evidence_items.iter().enumerate() {
            // Expand features to hidden dimension
            let expanded = self.expand_features(&item.features, item.severity, item.confidence);
            
            for (j, &value) in expanded.iter().enumerate() {
                if j < self.hidden_dim {
                    embeddings[[i, j]] = value;
                }
            }
        }

        embeddings
    }

    /// Expand features to hidden dimension with metadata encoding
    fn expand_features(&self, features: &[f32], severity: f32, confidence: f32) -> Vec<f32> {
        let mut expanded = vec![0.0; self.hidden_dim];
        
        // Copy original features
        for (i, &feat) in features.iter().enumerate() {
            if i < self.hidden_dim {
                expanded[i] = feat;
            }
        }

        // Add severity and confidence encoding
        if self.hidden_dim > features.len() + 2 {
            expanded[features.len()] = severity;
            expanded[features.len() + 1] = confidence;
        }

        // Apply sinusoidal position encoding for remaining dimensions
        for i in (features.len() + 2)..self.hidden_dim {
            let pos = i - features.len() - 2;
            if pos % 2 == 0 {
                expanded[i] = (pos as f32 / 10000.0).sin();
            } else {
                expanded[i] = (pos as f32 / 10000.0).cos();
            }
        }

        expanded
    }

    /// Multi-head attention computation
    fn multi_head_attention(&self, embeddings: &Array2<f32>) -> Array2<f32> {
        let n_items = embeddings.shape()[0];
        let head_dim = self.hidden_dim / self.num_heads;
        
        // Linear projections
        let queries = embeddings.dot(&self.w_query);
        let keys = embeddings.dot(&self.w_key);
        let values = embeddings.dot(&self.w_value);

        // Reshape for multi-head attention
        let queries_heads = self.reshape_for_heads(&queries, n_items, head_dim);
        let keys_heads = self.reshape_for_heads(&keys, n_items, head_dim);
        let values_heads = self.reshape_for_heads(&values, n_items, head_dim);

        // Compute attention scores
        let mut attention_output = Array2::zeros((n_items, self.hidden_dim));
        
        for head in 0..self.num_heads {
            let head_start = head * head_dim;
            let head_end = (head + 1) * head_dim;
            
            // Scaled dot-product attention
            let scores = self.scaled_dot_product_attention(
                &queries_heads.slice(s![.., head_start..head_end]),
                &keys_heads.slice(s![.., head_start..head_end]),
                &values_heads.slice(s![.., head_start..head_end]),
                head_dim,
            );
            
            // Concatenate heads
            for i in 0..n_items {
                for j in 0..head_dim {
                    attention_output[[i, head_start + j]] = scores[[i, j]];
                }
            }
        }

        // Final linear projection
        attention_output.dot(&self.w_output)
    }

    /// Reshape tensor for multi-head processing
    fn reshape_for_heads(&self, tensor: &Array2<f32>, n_items: usize, head_dim: usize) -> Array2<f32> {
        // For simplicity, we'll keep the 2D structure and process heads sequentially
        tensor.clone()
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        queries: &ndarray::ArrayView2<f32>,
        keys: &ndarray::ArrayView2<f32>,
        values: &ndarray::ArrayView2<f32>,
        head_dim: usize,
    ) -> Array2<f32> {
        let n_items = queries.shape()[0];
        let scale = (head_dim as f32).sqrt();
        
        // Compute attention scores
        let scores = queries.dot(&keys.t()) / scale;
        
        // Apply softmax
        let attention_weights = self.softmax(&scores);
        
        // Apply dropout
        let attention_weights = self.apply_dropout(&attention_weights);
        
        // Apply attention to values
        attention_weights.dot(values)
    }

    /// Softmax activation
    fn softmax(&self, scores: &Array2<f32>) -> Array2<f32> {
        let mut result = scores.clone();
        
        for mut row in result.rows_mut() {
            let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            
            for val in row.iter_mut() {
                *val = (*val - max).exp() / sum;
            }
        }
        
        result
    }

    /// Apply dropout
    fn apply_dropout(&self, weights: &Array2<f32>) -> Array2<f32> {
        if self.dropout_rate == 0.0 {
            return weights.clone();
        }
        
        let mut rng = rand::thread_rng();
        let mut result = weights.clone();
        let keep_prob = 1.0 - self.dropout_rate;
        
        for val in result.iter_mut() {
            if rng.gen::<f32>() > keep_prob {
                *val = 0.0;
            } else {
                *val /= keep_prob;
            }
        }
        
        result
    }

    /// Cross-domain attention for different evidence sources
    fn cross_domain_attention(
        &self,
        attention_output: &Array2<f32>,
        evidence_items: &[EvidenceItem],
    ) -> Vec<Vec<f32>> {
        let n_items = evidence_items.len();
        let mut cross_domain_features = Vec::new();

        // Group by evidence source
        let mut source_groups: HashMap<super::EvidenceSource, Vec<usize>> = HashMap::new();
        for (i, item) in evidence_items.iter().enumerate() {
            source_groups.entry(item.source).or_insert_with(Vec::new).push(i);
        }

        // Compute cross-domain attention between different sources
        for i in 0..n_items {
            let item_source = evidence_items[i].source;
            let mut item_features = attention_output.row(i).to_vec();
            
            // Attend to items from different sources
            for (source, indices) in &source_groups {
                if *source != item_source && !indices.is_empty() {
                    let mut source_attention = vec![0.0; self.hidden_dim];
                    
                    for &j in indices {
                        let other_features = attention_output.row(j);
                        let similarity = self.cosine_similarity(
                            &item_features,
                            &other_features.to_vec(),
                        );
                        
                        // Weight by similarity and confidence
                        let weight = similarity * evidence_items[j].confidence;
                        
                        for k in 0..self.hidden_dim {
                            source_attention[k] += other_features[k] * weight;
                        }
                    }
                    
                    // Normalize and combine
                    let norm_factor = indices.len() as f32;
                    for k in 0..self.hidden_dim {
                        item_features[k] += source_attention[k] / norm_factor * 0.5;
                    }
                }
            }
            
            cross_domain_features.push(item_features);
        }

        cross_domain_features
    }

    /// Compute cosine similarity between vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_cross_attention() {
        let mechanism = CrossAttentionMechanism::new(64, 8, 0.1);
        
        let evidence_items = vec![
            EvidenceItem {
                id: "1".to_string(),
                source: super::super::EvidenceSource::KpiDeviation,
                timestamp: Utc::now(),
                severity: 0.8,
                confidence: 0.9,
                features: vec![0.1, 0.2, 0.3],
                metadata: HashMap::new(),
            },
            EvidenceItem {
                id: "2".to_string(),
                source: super::super::EvidenceSource::AlarmSequence,
                timestamp: Utc::now(),
                severity: 0.7,
                confidence: 0.85,
                features: vec![0.2, 0.3, 0.4],
                metadata: HashMap::new(),
            },
        ];
        
        let features = mechanism.compute_attention(&evidence_items);
        assert_eq!(features.len(), evidence_items.len());
        assert_eq!(features[0].len(), 64);
    }
}