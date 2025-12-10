use std::collections::HashMap;
use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;

use super::EvidenceSource;

/// Multi-source fusion network for evidence integration
pub struct MultiSourceFusionNetwork {
    hidden_dim: usize,
    num_sources: usize,
    // Source-specific encoders
    source_encoders: HashMap<EvidenceSource, Array2<f32>>,
    // Fusion layers
    fusion_layer1: Array2<f32>,
    fusion_layer2: Array2<f32>,
    // Attention weights for source combination
    source_attention: Array2<f32>,
    // Normalization parameters
    layer_norms: Vec<(Array1<f32>, Array1<f32>)>, // (gamma, beta)
}

impl MultiSourceFusionNetwork {
    pub fn new(hidden_dim: usize, num_sources: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / hidden_dim as f32).sqrt();
        
        // Initialize source encoders
        let mut source_encoders = HashMap::new();
        let evidence_sources = [
            EvidenceSource::KpiDeviation,
            EvidenceSource::AlarmSequence,
            EvidenceSource::ConfigurationChange,
            EvidenceSource::TopologyImpact,
            EvidenceSource::PerformanceMetric,
            EvidenceSource::LogPattern,
        ];
        
        for source in evidence_sources.iter() {
            source_encoders.insert(
                *source,
                Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                    rng.gen_range(-scale..scale)
                }),
            );
        }
        
        Self {
            hidden_dim,
            num_sources,
            source_encoders,
            fusion_layer1: Array2::from_shape_fn((hidden_dim * num_sources, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            fusion_layer2: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            source_attention: Array2::from_shape_fn((num_sources, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            layer_norms: vec![
                (Array1::ones(hidden_dim), Array1::zeros(hidden_dim)),
                (Array1::ones(hidden_dim), Array1::zeros(hidden_dim)),
            ],
        }
    }

    /// Fuse multi-source features
    pub fn fuse(
        &self,
        attention_features: &[Vec<f32>],
        source_distribution: &HashMap<EvidenceSource, f32>,
    ) -> Vec<Vec<f32>> {
        if attention_features.is_empty() {
            return Vec::new();
        }

        // Group features by source
        let source_features = self.group_features_by_source(attention_features);
        
        // Encode each source
        let encoded_sources = self.encode_sources(&source_features, source_distribution);
        
        // Apply cross-source attention
        let attended_features = self.cross_source_attention(&encoded_sources);
        
        // Fuse all sources
        let fused_features = self.fuse_sources(&attended_features);
        
        // Apply residual connections and normalization
        self.apply_residual_and_norm(&fused_features, attention_features)
    }

    /// Group features by their source type
    fn group_features_by_source(
        &self,
        attention_features: &[Vec<f32>],
    ) -> HashMap<EvidenceSource, Vec<Vec<f32>>> {
        // This would need evidence source mapping from the original call
        // For now, we'll distribute features evenly across sources
        let mut source_features = HashMap::new();
        let sources = [
            EvidenceSource::KpiDeviation,
            EvidenceSource::AlarmSequence,
            EvidenceSource::ConfigurationChange,
            EvidenceSource::TopologyImpact,
            EvidenceSource::PerformanceMetric,
            EvidenceSource::LogPattern,
        ];
        
        for (i, features) in attention_features.iter().enumerate() {
            let source = sources[i % sources.len()];
            source_features.entry(source).or_insert_with(Vec::new).push(features.clone());
        }
        
        source_features
    }

    /// Encode features for each source
    fn encode_sources(
        &self,
        source_features: &HashMap<EvidenceSource, Vec<Vec<f32>>>,
        source_distribution: &HashMap<EvidenceSource, f32>,
    ) -> HashMap<EvidenceSource, Array2<f32>> {
        let mut encoded_sources = HashMap::new();
        
        for (source, features) in source_features {
            if let Some(encoder) = self.source_encoders.get(source) {
                let n_items = features.len();
                let mut source_matrix = Array2::zeros((n_items, self.hidden_dim));
                
                // Convert features to array
                for (i, feature_vec) in features.iter().enumerate() {
                    for (j, &val) in feature_vec.iter().enumerate() {
                        if j < self.hidden_dim {
                            source_matrix[[i, j]] = val;
                        }
                    }
                }
                
                // Apply source-specific encoding
                let encoded = source_matrix.dot(encoder);
                
                // Weight by source distribution
                let weight = source_distribution.get(source).unwrap_or(&1.0);
                let weighted_encoded = &encoded * *weight;
                
                encoded_sources.insert(*source, weighted_encoded);
            }
        }
        
        encoded_sources
    }

    /// Apply cross-source attention
    fn cross_source_attention(
        &self,
        encoded_sources: &HashMap<EvidenceSource, Array2<f32>>,
    ) -> HashMap<EvidenceSource, Array2<f32>> {
        let mut attended_sources = HashMap::new();
        
        for (source, features) in encoded_sources {
            let mut attended_features = features.clone();
            
            // Attend to other sources
            for (other_source, other_features) in encoded_sources {
                if source != other_source {
                    let attention_weights = self.compute_source_attention(
                        features,
                        other_features,
                    );
                    
                    let attended = self.apply_attention_weights(
                        &attention_weights,
                        other_features,
                    );
                    
                    // Combine with original features
                    attended_features = &attended_features + &attended * 0.3;
                }
            }
            
            attended_sources.insert(*source, attended_features);
        }
        
        attended_sources
    }

    /// Compute attention weights between sources
    fn compute_source_attention(
        &self,
        query_features: &Array2<f32>,
        key_features: &Array2<f32>,
    ) -> Array2<f32> {
        let n_query = query_features.shape()[0];
        let n_key = key_features.shape()[0];
        let mut attention = Array2::zeros((n_query, n_key));
        
        for i in 0..n_query {
            for j in 0..n_key {
                let query_row = query_features.row(i);
                let key_row = key_features.row(j);
                
                // Compute dot product attention
                let score = query_row.dot(&key_row) / (self.hidden_dim as f32).sqrt();
                attention[[i, j]] = score;
            }
        }
        
        // Apply softmax
        self.softmax(&attention)
    }

    /// Apply attention weights
    fn apply_attention_weights(
        &self,
        attention_weights: &Array2<f32>,
        values: &Array2<f32>,
    ) -> Array2<f32> {
        attention_weights.dot(values)
    }

    /// Fuse features from all sources
    fn fuse_sources(
        &self,
        attended_sources: &HashMap<EvidenceSource, Array2<f32>>,
    ) -> Array2<f32> {
        let mut all_features = Vec::new();
        
        // Collect all source features
        for (_, features) in attended_sources {
            for row in features.rows() {
                all_features.push(row.to_vec());
            }
        }
        
        if all_features.is_empty() {
            return Array2::zeros((0, self.hidden_dim));
        }
        
        let n_items = all_features.len();
        let mut feature_matrix = Array2::zeros((n_items, self.hidden_dim));
        
        // Build feature matrix
        for (i, features) in all_features.iter().enumerate() {
            for (j, &val) in features.iter().enumerate() {
                if j < self.hidden_dim {
                    feature_matrix[[i, j]] = val;
                }
            }
        }
        
        // Apply fusion layers
        let hidden1 = self.relu(&feature_matrix.dot(&self.fusion_layer1));
        let hidden1_norm = self.layer_norm(&hidden1, &self.layer_norms[0]);
        
        let output = hidden1_norm.dot(&self.fusion_layer2);
        self.layer_norm(&output, &self.layer_norms[1])
    }

    /// Apply residual connections and normalization
    fn apply_residual_and_norm(
        &self,
        fused_features: &Array2<f32>,
        original_features: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let mut result = Vec::new();
        
        for i in 0..fused_features.shape()[0] {
            let mut feature_vec = fused_features.row(i).to_vec();
            
            // Add residual connection if we have original features
            if i < original_features.len() {
                for (j, &orig_val) in original_features[i].iter().enumerate() {
                    if j < feature_vec.len() {
                        feature_vec[j] += orig_val * 0.5; // Residual weight
                    }
                }
            }
            
            // Apply final normalization
            let norm = feature_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in feature_vec.iter_mut() {
                    *val /= norm;
                }
            }
            
            result.push(feature_vec);
        }
        
        result
    }

    /// ReLU activation
    fn relu(&self, input: &Array2<f32>) -> Array2<f32> {
        input.map(|&x| x.max(0.0))
    }

    /// Layer normalization
    fn layer_norm(&self, input: &Array2<f32>, norm_params: &(Array1<f32>, Array1<f32>)) -> Array2<f32> {
        let (gamma, beta) = norm_params;
        let mut output = input.clone();
        
        for mut row in output.rows_mut() {
            let mean = row.mean().unwrap_or(0.0);
            let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;
            let std_dev = (variance + 1e-8).sqrt();
            
            for (i, val) in row.iter_mut().enumerate() {
                *val = (*val - mean) / std_dev;
                if i < gamma.len() {
                    *val = *val * gamma[i] + beta[i];
                }
            }
        }
        
        output
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_network() {
        let network = MultiSourceFusionNetwork::new(64, 6);
        
        let attention_features = vec![
            vec![0.1; 64],
            vec![0.2; 64],
        ];
        
        let mut source_distribution = HashMap::new();
        source_distribution.insert(EvidenceSource::KpiDeviation, 0.5);
        source_distribution.insert(EvidenceSource::AlarmSequence, 0.5);
        
        let fused = network.fuse(&attention_features, &source_distribution);
        assert!(!fused.is_empty());
    }
}