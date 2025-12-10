use std::collections::HashMap;
use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;

use super::CorrelationResult;

/// Hierarchical attention network for multi-scale correlation analysis
pub struct HierarchicalAttentionNetwork {
    hidden_dim: usize,
    scale_levels: Vec<usize>,
    // Multi-scale attention layers
    scale_attention_layers: Vec<ScaleAttentionLayer>,
    // Hierarchical fusion
    hierarchical_fusion: Array2<f32>,
    fusion_bias: Array1<f32>,
}

impl HierarchicalAttentionNetwork {
    pub fn new(hidden_dim: usize, scale_levels: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / hidden_dim as f32).sqrt();
        
        // Create attention layers for each scale
        let mut scale_attention_layers = Vec::new();
        for &scale_size in &scale_levels {
            scale_attention_layers.push(ScaleAttentionLayer::new(hidden_dim, scale_size));
        }
        
        Self {
            hidden_dim,
            scale_levels: scale_levels.clone(),
            scale_attention_layers,
            hierarchical_fusion: Array2::from_shape_fn(
                (scale_levels.len() * hidden_dim, hidden_dim),
                |_| rng.gen_range(-scale..scale)
            ),
            fusion_bias: Array1::from_shape_fn(hidden_dim, |_| rng.gen_range(-0.1..0.1)),
        }
    }

    /// Process features with hierarchical attention
    pub fn process(
        &self,
        fused_features: &[Vec<f32>],
        correlations: &[CorrelationResult],
    ) -> HashMap<String, Vec<f32>> {
        let mut hierarchical_features = HashMap::new();
        
        // Process each scale level
        for (i, layer) in self.scale_attention_layers.iter().enumerate() {
            let scale_name = match i {
                0 => "fine",
                1 => "medium", 
                2 => "coarse",
                _ => "extra",
            };
            
            let scale_features = layer.process_scale(fused_features, correlations);
            hierarchical_features.insert(scale_name.to_string(), scale_features);
        }
        
        // Cross-scale attention
        let cross_scale_features = self.cross_scale_attention(&hierarchical_features);
        hierarchical_features.insert("cross_scale".to_string(), cross_scale_features);
        
        // Hierarchical fusion
        let fused_hierarchical = self.hierarchical_fusion_process(&hierarchical_features);
        hierarchical_features.insert("fused".to_string(), fused_hierarchical);
        
        hierarchical_features
    }

    /// Cross-scale attention mechanism
    fn cross_scale_attention(&self, scale_features: &HashMap<String, Vec<f32>>) -> Vec<f32> {
        let mut cross_scale_output = vec![0.0; self.hidden_dim];
        
        if scale_features.is_empty() {
            return cross_scale_output;
        }
        
        // Convert to attention format
        let mut scale_matrices = Vec::new();
        let mut scale_names = Vec::new();
        
        for (scale_name, features) in scale_features {
            if scale_name != "cross_scale" && scale_name != "fused" {
                scale_matrices.push(self.vec_to_matrix(features));
                scale_names.push(scale_name.clone());
            }
        }
        
        if scale_matrices.is_empty() {
            return cross_scale_output;
        }
        
        // Compute cross-scale attention weights
        let attention_weights = self.compute_cross_scale_weights(&scale_matrices);
        
        // Apply attention to combine scales
        for (i, matrix) in scale_matrices.iter().enumerate() {
            let weight = attention_weights[i];
            for (j, &value) in matrix.iter().enumerate() {
                if j < cross_scale_output.len() {
                    cross_scale_output[j] += value * weight;
                }
            }
        }
        
        // Normalize
        let norm = cross_scale_output.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in cross_scale_output.iter_mut() {
                *val /= norm;
            }
        }
        
        cross_scale_output
    }

    /// Convert vector to matrix for processing
    fn vec_to_matrix(&self, features: &[f32]) -> Vec<f32> {
        let mut matrix = vec![0.0; self.hidden_dim];
        for (i, &val) in features.iter().enumerate() {
            if i < self.hidden_dim {
                matrix[i] = val;
            }
        }
        matrix
    }

    /// Compute cross-scale attention weights
    fn compute_cross_scale_weights(&self, scale_matrices: &[Vec<f32>]) -> Vec<f32> {
        if scale_matrices.is_empty() {
            return Vec::new();
        }
        
        let mut weights = Vec::new();
        
        // Compute pairwise similarities
        for i in 0..scale_matrices.len() {
            let mut total_similarity = 0.0;
            
            for j in 0..scale_matrices.len() {
                if i != j {
                    let similarity = self.cosine_similarity(&scale_matrices[i], &scale_matrices[j]);
                    total_similarity += similarity;
                }
            }
            
            // Weight by average similarity (higher similarity = higher weight)
            let avg_similarity = if scale_matrices.len() > 1 {
                total_similarity / (scale_matrices.len() - 1) as f32
            } else {
                1.0
            };
            
            weights.push(avg_similarity);
        }
        
        // Softmax normalization
        let max_weight = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp: f32 = weights.iter().map(|&w| (w - max_weight).exp()).sum();
        
        for weight in weights.iter_mut() {
            *weight = (*weight - max_weight).exp() / sum_exp;
        }
        
        weights
    }

    /// Hierarchical fusion process
    fn hierarchical_fusion_process(&self, hierarchical_features: &HashMap<String, Vec<f32>>) -> Vec<f32> {
        // Concatenate all scale features
        let mut concatenated = Vec::new();
        
        let scale_order = ["fine", "medium", "coarse", "cross_scale"];
        for scale_name in &scale_order {
            if let Some(features) = hierarchical_features.get(*scale_name) {
                concatenated.extend_from_slice(features);
            }
        }
        
        // Pad or truncate to expected size
        let expected_size = self.scale_levels.len() * self.hidden_dim;
        concatenated.resize(expected_size, 0.0);
        
        // Apply fusion transformation
        let input_array = Array1::from_vec(concatenated);
        let output = input_array.dot(&self.hierarchical_fusion) + &self.fusion_bias;
        
        output.to_vec()
    }

    /// Cosine similarity
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

/// Scale-specific attention layer
struct ScaleAttentionLayer {
    hidden_dim: usize,
    scale_size: usize,
    // Attention parameters
    query_projection: Array2<f32>,
    key_projection: Array2<f32>,
    value_projection: Array2<f32>,
    output_projection: Array2<f32>,
    // Position encodings
    position_encodings: Array2<f32>,
}

impl ScaleAttentionLayer {
    fn new(hidden_dim: usize, scale_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / hidden_dim as f32).sqrt();
        
        Self {
            hidden_dim,
            scale_size,
            query_projection: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            key_projection: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            value_projection: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            output_projection: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            position_encodings: Self::create_position_encodings(scale_size, hidden_dim),
        }
    }

    /// Create sinusoidal position encodings
    fn create_position_encodings(max_len: usize, hidden_dim: usize) -> Array2<f32> {
        let mut encodings = Array2::zeros((max_len, hidden_dim));
        
        for pos in 0..max_len {
            for i in 0..hidden_dim {
                let angle = pos as f32 / 10000.0_f32.powf(2.0 * i as f32 / hidden_dim as f32);
                if i % 2 == 0 {
                    encodings[[pos, i]] = angle.sin();
                } else {
                    encodings[[pos, i]] = angle.cos();
                }
            }
        }
        
        encodings
    }

    /// Process features at this scale
    fn process_scale(
        &self,
        fused_features: &[Vec<f32>],
        correlations: &[CorrelationResult],
    ) -> Vec<f32> {
        if fused_features.is_empty() {
            return vec![0.0; self.hidden_dim];
        }
        
        // Create patches based on scale size
        let patches = self.create_patches(fused_features);
        
        // Apply attention within each patch
        let attended_patches = self.apply_patch_attention(&patches);
        
        // Aggregate patches
        let aggregated = self.aggregate_patches(&attended_patches);
        
        // Apply correlation weighting
        self.apply_correlation_weighting(&aggregated, correlations)
    }

    /// Create patches for multi-scale processing
    fn create_patches(&self, features: &[Vec<f32>]) -> Vec<Vec<Vec<f32>>> {
        let mut patches = Vec::new();
        let patch_size = self.scale_size;
        
        for i in (0..features.len()).step_by(patch_size) {
            let mut patch = Vec::new();
            for j in 0..patch_size {
                if i + j < features.len() {
                    patch.push(features[i + j].clone());
                } else {
                    patch.push(vec![0.0; self.hidden_dim]);
                }
            }
            patches.push(patch);
        }
        
        patches
    }

    /// Apply attention within patches
    fn apply_patch_attention(&self, patches: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
        let mut attended_patches = Vec::new();
        
        for patch in patches {
            let attended_patch = self.self_attention_patch(patch);
            attended_patches.push(attended_patch);
        }
        
        attended_patches
    }

    /// Self-attention within a patch
    fn self_attention_patch(&self, patch: &[Vec<f32>]) -> Vec<f32> {
        if patch.is_empty() {
            return vec![0.0; self.hidden_dim];
        }
        
        // Convert to matrix
        let n_items = patch.len();
        let mut feature_matrix = Array2::zeros((n_items, self.hidden_dim));
        
        for (i, features) in patch.iter().enumerate() {
            for (j, &val) in features.iter().enumerate() {
                if j < self.hidden_dim {
                    feature_matrix[[i, j]] = val;
                }
            }
        }
        
        // Add position encodings
        for i in 0..n_items.min(self.position_encodings.shape()[0]) {
            for j in 0..self.hidden_dim {
                feature_matrix[[i, j]] += self.position_encodings[[i, j]];
            }
        }
        
        // Apply attention
        let queries = feature_matrix.dot(&self.query_projection);
        let keys = feature_matrix.dot(&self.key_projection);
        let values = feature_matrix.dot(&self.value_projection);
        
        // Compute attention scores
        let scores = queries.dot(&keys.t()) / (self.hidden_dim as f32).sqrt();
        let attention_weights = self.softmax(&scores);
        
        // Apply attention to values
        let attended = attention_weights.dot(&values);
        
        // Output projection
        let output = attended.dot(&self.output_projection);
        
        // Pool across items (mean pooling)
        let mut pooled = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let sum: f32 = (0..n_items).map(|j| output[[j, i]]).sum();
            pooled[i] = sum / n_items as f32;
        }
        
        pooled
    }

    /// Softmax activation
    fn softmax(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.clone();
        
        for mut row in output.rows_mut() {
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            
            for val in row.iter_mut() {
                *val = (*val - max_val).exp() / sum_exp;
            }
        }
        
        output
    }

    /// Aggregate patches
    fn aggregate_patches(&self, patches: &[Vec<f32>]) -> Vec<f32> {
        if patches.is_empty() {
            return vec![0.0; self.hidden_dim];
        }
        
        let mut aggregated = vec![0.0; self.hidden_dim];
        
        for patch in patches {
            for (i, &val) in patch.iter().enumerate() {
                if i < aggregated.len() {
                    aggregated[i] += val;
                }
            }
        }
        
        // Normalize
        let norm = aggregated.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in aggregated.iter_mut() {
                *val /= norm;
            }
        }
        
        aggregated
    }

    /// Apply correlation weighting
    fn apply_correlation_weighting(
        &self,
        features: &[f32],
        correlations: &[CorrelationResult],
    ) -> Vec<f32> {
        let mut weighted_features = features.to_vec();
        
        if correlations.is_empty() {
            return weighted_features;
        }
        
        // Compute average correlation strength
        let avg_correlation = correlations.iter()
            .map(|c| c.correlation_score)
            .sum::<f32>() / correlations.len() as f32;
        
        // Weight features by correlation strength
        let weight = (1.0 + avg_correlation) / 2.0;
        
        for val in weighted_features.iter_mut() {
            *val *= weight;
        }
        
        weighted_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use chrono::Utc;

    #[test]
    fn test_hierarchical_attention() {
        let network = HierarchicalAttentionNetwork::new(64, vec![4, 8, 16]);
        
        let fused_features = vec![
            vec![0.1; 64],
            vec![0.2; 64],
            vec![0.3; 64],
            vec![0.4; 64],
        ];
        
        let correlations = vec![
            CorrelationResult {
                correlation_id: "test".to_string(),
                evidence_items: Vec::new(),
                correlation_score: 0.8,
                confidence: 0.9,
                temporal_alignment: 0.7,
                cross_domain_score: 0.6,
                impact_assessment: super::super::ImpactAssessment {
                    severity: 0.8,
                    scope: "test".to_string(),
                    affected_components: Vec::new(),
                    propagation_risk: 0.5,
                },
            },
        ];
        
        let hierarchical_features = network.process(&fused_features, &correlations);
        assert!(!hierarchical_features.is_empty());
        assert!(hierarchical_features.contains_key("fine"));
        assert!(hierarchical_features.contains_key("medium"));
        assert!(hierarchical_features.contains_key("coarse"));
    }
}