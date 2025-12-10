use std::collections::HashMap;
use ndarray::{Array1, Array2};
use rand::Rng;
use chrono::{DateTime, Utc, Duration};

use super::{EvidenceItem, CorrelationResult, ImpactAssessment, EvidenceSource};

/// Evidence scoring network for correlation assessment
pub struct EvidenceScoringNetwork {
    hidden_dim: usize,
    // Neural network layers
    scoring_layer1: Array2<f32>,
    scoring_layer2: Array2<f32>,
    scoring_layer3: Array2<f32>,
    // Bias vectors
    bias1: Array1<f32>,
    bias2: Array1<f32>,
    bias3: Array1<f32>,
    // Normalization parameters
    feature_mean: Array1<f32>,
    feature_std: Array1<f32>,
}

impl EvidenceScoringNetwork {
    pub fn new(hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / hidden_dim as f32).sqrt();
        
        Self {
            hidden_dim,
            scoring_layer1: Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            scoring_layer2: Array2::from_shape_fn((hidden_dim, hidden_dim / 2), |_| {
                rng.gen_range(-scale..scale)
            }),
            scoring_layer3: Array2::from_shape_fn((hidden_dim / 2, 1), |_| {
                rng.gen_range(-scale..scale)
            }),
            bias1: Array1::from_shape_fn(hidden_dim, |_| rng.gen_range(-0.1..0.1)),
            bias2: Array1::from_shape_fn(hidden_dim / 2, |_| rng.gen_range(-0.1..0.1)),
            bias3: Array1::from_shape_fn(1, |_| rng.gen_range(-0.1..0.1)),
            feature_mean: Array1::zeros(hidden_dim),
            feature_std: Array1::ones(hidden_dim),
        }
    }

    /// Score correlations between evidence items
    pub fn score_correlations(
        &self,
        evidence_items: &[EvidenceItem],
        fused_features: &[Vec<f32>],
    ) -> Vec<CorrelationResult> {
        let mut correlations = Vec::new();
        
        // Generate pairwise correlations
        for i in 0..evidence_items.len() {
            for j in (i + 1)..evidence_items.len() {
                if let Some(correlation) = self.score_pair(
                    &evidence_items[i],
                    &evidence_items[j],
                    &fused_features[i],
                    &fused_features[j],
                ) {
                    correlations.push(correlation);
                }
            }
        }
        
        // Generate group correlations
        let group_correlations = self.score_groups(evidence_items, fused_features);
        correlations.extend(group_correlations);
        
        // Filter and rank correlations
        self.filter_and_rank_correlations(correlations)
    }

    /// Score correlation between two evidence items
    fn score_pair(
        &self,
        item1: &EvidenceItem,
        item2: &EvidenceItem,
        features1: &[f32],
        features2: &[f32],
    ) -> Option<CorrelationResult> {
        // Compute correlation features
        let correlation_features = self.compute_correlation_features(
            item1, item2, features1, features2,
        );
        
        // Score with neural network
        let correlation_score = self.forward_pass(&correlation_features);
        
        // Compute confidence based on multiple factors
        let confidence = self.compute_confidence(item1, item2, correlation_score);
        
        // Compute temporal alignment score
        let temporal_alignment = self.compute_temporal_alignment(item1, item2);
        
        // Compute cross-domain score
        let cross_domain_score = self.compute_cross_domain_score(item1, item2);
        
        // Assess impact
        let impact_assessment = self.assess_impact(item1, item2, correlation_score);
        
        if correlation_score > 0.5 {
            Some(CorrelationResult {
                correlation_id: self.generate_correlation_id(item1, item2),
                evidence_items: vec![item1.clone(), item2.clone()],
                correlation_score,
                confidence,
                temporal_alignment,
                cross_domain_score,
                impact_assessment,
            })
        } else {
            None
        }
    }

    /// Compute correlation features for a pair of evidence items
    fn compute_correlation_features(
        &self,
        item1: &EvidenceItem,
        item2: &EvidenceItem,
        features1: &[f32],
        features2: &[f32],
    ) -> Vec<f32> {
        let mut correlation_features = Vec::new();
        
        // Feature similarity
        let feature_similarity = self.cosine_similarity(features1, features2);
        correlation_features.push(feature_similarity);
        
        // Temporal proximity
        let time_diff = (item1.timestamp - item2.timestamp).num_seconds().abs() as f32;
        let temporal_proximity = (-time_diff / 3600.0).exp(); // Exponential decay
        correlation_features.push(temporal_proximity);
        
        // Severity correlation
        let severity_diff = (item1.severity - item2.severity).abs();
        let severity_correlation = 1.0 - severity_diff;
        correlation_features.push(severity_correlation);
        
        // Confidence correlation
        let confidence_diff = (item1.confidence - item2.confidence).abs();
        let confidence_correlation = 1.0 - confidence_diff;
        correlation_features.push(confidence_correlation);
        
        // Source relationship
        let source_relationship = self.compute_source_relationship(item1.source, item2.source);
        correlation_features.push(source_relationship);
        
        // Metadata similarity
        let metadata_similarity = self.compute_metadata_similarity(item1, item2);
        correlation_features.push(metadata_similarity);
        
        // Expand to hidden dimension
        while correlation_features.len() < self.hidden_dim {
            correlation_features.push(0.0);
        }
        
        correlation_features.truncate(self.hidden_dim);
        correlation_features
    }

    /// Compute cosine similarity between feature vectors
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

    /// Compute relationship score between evidence sources
    fn compute_source_relationship(&self, source1: EvidenceSource, source2: EvidenceSource) -> f32 {
        // Define source relationship matrix
        let relationships = [
            // KpiDeviation, AlarmSequence, ConfigurationChange, TopologyImpact, PerformanceMetric, LogPattern
            [1.0, 0.8, 0.6, 0.7, 0.9, 0.5], // KpiDeviation
            [0.8, 1.0, 0.5, 0.6, 0.7, 0.8], // AlarmSequence
            [0.6, 0.5, 1.0, 0.9, 0.4, 0.3], // ConfigurationChange
            [0.7, 0.6, 0.9, 1.0, 0.5, 0.4], // TopologyImpact
            [0.9, 0.7, 0.4, 0.5, 1.0, 0.6], // PerformanceMetric
            [0.5, 0.8, 0.3, 0.4, 0.6, 1.0], // LogPattern
        ];
        
        let source1_idx = self.source_to_index(source1);
        let source2_idx = self.source_to_index(source2);
        
        relationships[source1_idx][source2_idx]
    }

    /// Convert evidence source to index
    fn source_to_index(&self, source: EvidenceSource) -> usize {
        match source {
            EvidenceSource::KpiDeviation => 0,
            EvidenceSource::AlarmSequence => 1,
            EvidenceSource::ConfigurationChange => 2,
            EvidenceSource::TopologyImpact => 3,
            EvidenceSource::PerformanceMetric => 4,
            EvidenceSource::LogPattern => 5,
        }
    }

    /// Compute metadata similarity
    fn compute_metadata_similarity(&self, item1: &EvidenceItem, item2: &EvidenceItem) -> f32 {
        let mut common_keys = 0;
        let mut total_keys = 0;
        
        for key in item1.metadata.keys() {
            total_keys += 1;
            if let Some(value2) = item2.metadata.get(key) {
                if let Some(value1) = item1.metadata.get(key) {
                    if value1 == value2 {
                        common_keys += 1;
                    }
                }
            }
        }
        
        for key in item2.metadata.keys() {
            if !item1.metadata.contains_key(key) {
                total_keys += 1;
            }
        }
        
        if total_keys > 0 {
            common_keys as f32 / total_keys as f32
        } else {
            0.0
        }
    }

    /// Forward pass through neural network
    fn forward_pass(&self, features: &[f32]) -> f32 {
        // Normalize features
        let mut normalized_features = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            if i < features.len() {
                normalized_features[i] = (features[i] - self.feature_mean[i]) / self.feature_std[i];
            }
        }
        
        // Layer 1
        let hidden1 = normalized_features.dot(&self.scoring_layer1) + &self.bias1;
        let hidden1_activated = hidden1.map(|&x| self.relu(x));
        
        // Layer 2
        let hidden2 = hidden1_activated.dot(&self.scoring_layer2) + &self.bias2;
        let hidden2_activated = hidden2.map(|&x| self.relu(x));
        
        // Layer 3 (output)
        let output = hidden2_activated.dot(&self.scoring_layer3) + &self.bias3;
        
        // Apply sigmoid activation
        self.sigmoid(output[0])
    }

    /// ReLU activation
    fn relu(&self, x: f32) -> f32 {
        x.max(0.0)
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Compute confidence score
    fn compute_confidence(&self, item1: &EvidenceItem, item2: &EvidenceItem, correlation_score: f32) -> f32 {
        let avg_confidence = (item1.confidence + item2.confidence) / 2.0;
        let score_weight = correlation_score.powf(2.0);
        let temporal_weight = self.compute_temporal_weight(item1, item2);
        
        (avg_confidence * 0.4 + score_weight * 0.4 + temporal_weight * 0.2).min(1.0)
    }

    /// Compute temporal weight
    fn compute_temporal_weight(&self, item1: &EvidenceItem, item2: &EvidenceItem) -> f32 {
        let time_diff = (item1.timestamp - item2.timestamp).num_seconds().abs() as f32;
        let max_time = 3600.0; // 1 hour
        
        (1.0 - (time_diff / max_time).min(1.0)).max(0.0)
    }

    /// Compute temporal alignment score
    fn compute_temporal_alignment(&self, item1: &EvidenceItem, item2: &EvidenceItem) -> f32 {
        let time_diff = (item1.timestamp - item2.timestamp).num_seconds().abs() as f32;
        let alignment_window = 300.0; // 5 minutes
        
        if time_diff <= alignment_window {
            1.0 - (time_diff / alignment_window)
        } else {
            (1.0 / (1.0 + (time_diff - alignment_window) / 1800.0)).max(0.1) // Decay after 5 minutes
        }
    }

    /// Compute cross-domain score
    fn compute_cross_domain_score(&self, item1: &EvidenceItem, item2: &EvidenceItem) -> f32 {
        if item1.source != item2.source {
            self.compute_source_relationship(item1.source, item2.source)
        } else {
            0.2 // Low score for same-source correlation
        }
    }

    /// Assess impact of correlation
    fn assess_impact(&self, item1: &EvidenceItem, item2: &EvidenceItem, correlation_score: f32) -> ImpactAssessment {
        let severity = ((item1.severity + item2.severity) / 2.0 * correlation_score).min(1.0);
        
        let scope = if item1.source != item2.source {
            "cross-domain".to_string()
        } else {
            format!("{:?}", item1.source)
        };
        
        let mut affected_components = Vec::new();
        if let Some(component1) = item1.metadata.get("component") {
            affected_components.push(component1.clone());
        }
        if let Some(component2) = item2.metadata.get("component") {
            if !affected_components.contains(component2) {
                affected_components.push(component2.clone());
            }
        }
        
        let propagation_risk = self.compute_propagation_risk(item1, item2, severity);
        
        ImpactAssessment {
            severity,
            scope,
            affected_components,
            propagation_risk,
        }
    }

    /// Compute propagation risk
    fn compute_propagation_risk(&self, item1: &EvidenceItem, item2: &EvidenceItem, severity: f32) -> f32 {
        let mut risk = severity;
        
        // Increase risk for cross-domain correlations
        if item1.source != item2.source {
            risk *= 1.3;
        }
        
        // Increase risk for high-severity items
        if item1.severity > 0.8 || item2.severity > 0.8 {
            risk *= 1.2;
        }
        
        // Increase risk for configuration or topology changes
        if item1.source == EvidenceSource::ConfigurationChange || 
           item2.source == EvidenceSource::ConfigurationChange ||
           item1.source == EvidenceSource::TopologyImpact ||
           item2.source == EvidenceSource::TopologyImpact {
            risk *= 1.4;
        }
        
        risk.min(1.0)
    }

    /// Score groups of evidence items
    fn score_groups(&self, evidence_items: &[EvidenceItem], fused_features: &[Vec<f32>]) -> Vec<CorrelationResult> {
        let mut group_correlations = Vec::new();
        
        // Group by source
        let source_groups = self.group_by_source(evidence_items);
        
        // Score intra-group correlations
        for (source, indices) in source_groups {
            if indices.len() >= 3 {
                let group_correlation = self.score_group(&indices, evidence_items, fused_features, source);
                if let Some(correlation) = group_correlation {
                    group_correlations.push(correlation);
                }
            }
        }
        
        // Score temporal clusters
        let temporal_clusters = self.find_temporal_clusters(evidence_items);
        for cluster in temporal_clusters {
            if cluster.len() >= 3 {
                let cluster_correlation = self.score_cluster(&cluster, evidence_items, fused_features);
                if let Some(correlation) = cluster_correlation {
                    group_correlations.push(correlation);
                }
            }
        }
        
        group_correlations
    }

    /// Group evidence items by source
    fn group_by_source(&self, evidence_items: &[EvidenceItem]) -> HashMap<EvidenceSource, Vec<usize>> {
        let mut groups = HashMap::new();
        
        for (i, item) in evidence_items.iter().enumerate() {
            groups.entry(item.source).or_insert_with(Vec::new).push(i);
        }
        
        groups
    }

    /// Score a group of evidence items
    fn score_group(
        &self,
        indices: &[usize],
        evidence_items: &[EvidenceItem],
        fused_features: &[Vec<f32>],
        source: EvidenceSource,
    ) -> Option<CorrelationResult> {
        let group_items: Vec<EvidenceItem> = indices.iter()
            .map(|&i| evidence_items[i].clone())
            .collect();
        
        let group_features: Vec<&Vec<f32>> = indices.iter()
            .map(|&i| &fused_features[i])
            .collect();
        
        // Compute group coherence
        let coherence = self.compute_group_coherence(&group_features);
        
        if coherence > 0.6 {
            let avg_severity = group_items.iter().map(|item| item.severity).sum::<f32>() / group_items.len() as f32;
            let avg_confidence = group_items.iter().map(|item| item.confidence).sum::<f32>() / group_items.len() as f32;
            
            Some(CorrelationResult {
                correlation_id: format!("group_{:?}_{}", source, Utc::now().timestamp()),
                evidence_items: group_items.clone(),
                correlation_score: coherence,
                confidence: avg_confidence,
                temporal_alignment: self.compute_group_temporal_alignment(&group_items),
                cross_domain_score: 0.0, // Same source
                impact_assessment: self.assess_group_impact(&group_items, coherence),
            })
        } else {
            None
        }
    }

    /// Compute group coherence
    fn compute_group_coherence(&self, features: &[&Vec<f32>]) -> f32 {
        if features.len() < 2 {
            return 0.0;
        }
        
        let mut total_similarity = 0.0;
        let mut count = 0;
        
        for i in 0..features.len() {
            for j in (i + 1)..features.len() {
                total_similarity += self.cosine_similarity(features[i], features[j]);
                count += 1;
            }
        }
        
        if count > 0 {
            total_similarity / count as f32
        } else {
            0.0
        }
    }

    /// Compute group temporal alignment
    fn compute_group_temporal_alignment(&self, items: &[EvidenceItem]) -> f32 {
        if items.len() < 2 {
            return 1.0;
        }
        
        let timestamps: Vec<i64> = items.iter().map(|item| item.timestamp.timestamp()).collect();
        let min_time = *timestamps.iter().min().unwrap();
        let max_time = *timestamps.iter().max().unwrap();
        let time_span = max_time - min_time;
        
        // Good alignment if all items are within 10 minutes
        let alignment_window = 600; // 10 minutes
        
        if time_span <= alignment_window {
            1.0 - (time_span as f32 / alignment_window as f32)
        } else {
            (1.0 / (1.0 + (time_span - alignment_window) as f32 / 3600.0)).max(0.1)
        }
    }

    /// Assess group impact
    fn assess_group_impact(&self, items: &[EvidenceItem], coherence: f32) -> ImpactAssessment {
        let avg_severity = items.iter().map(|item| item.severity).sum::<f32>() / items.len() as f32;
        let severity = (avg_severity * coherence).min(1.0);
        
        let scope = format!("group_{:?}", items[0].source);
        
        let mut affected_components = Vec::new();
        for item in items {
            if let Some(component) = item.metadata.get("component") {
                if !affected_components.contains(component) {
                    affected_components.push(component.clone());
                }
            }
        }
        
        let propagation_risk = (severity * items.len() as f32 / 10.0).min(1.0);
        
        ImpactAssessment {
            severity,
            scope,
            affected_components,
            propagation_risk,
        }
    }

    /// Find temporal clusters
    fn find_temporal_clusters(&self, evidence_items: &[EvidenceItem]) -> Vec<Vec<usize>> {
        let mut clusters = Vec::new();
        let cluster_window = Duration::minutes(5);
        
        let mut sorted_indices: Vec<usize> = (0..evidence_items.len()).collect();
        sorted_indices.sort_by_key(|&i| evidence_items[i].timestamp);
        
        let mut current_cluster = Vec::new();
        let mut cluster_start = None;
        
        for &i in &sorted_indices {
            let item = &evidence_items[i];
            
            if let Some(start_time) = cluster_start {
                if item.timestamp - start_time <= cluster_window {
                    current_cluster.push(i);
                } else {
                    if current_cluster.len() >= 2 {
                        clusters.push(current_cluster.clone());
                    }
                    current_cluster.clear();
                    current_cluster.push(i);
                    cluster_start = Some(item.timestamp);
                }
            } else {
                current_cluster.push(i);
                cluster_start = Some(item.timestamp);
            }
        }
        
        if current_cluster.len() >= 2 {
            clusters.push(current_cluster);
        }
        
        clusters
    }

    /// Score a temporal cluster
    fn score_cluster(
        &self,
        cluster: &[usize],
        evidence_items: &[EvidenceItem],
        fused_features: &[Vec<f32>],
    ) -> Option<CorrelationResult> {
        let cluster_items: Vec<EvidenceItem> = cluster.iter()
            .map(|&i| evidence_items[i].clone())
            .collect();
        
        let cluster_features: Vec<&Vec<f32>> = cluster.iter()
            .map(|&i| &fused_features[i])
            .collect();
        
        let coherence = self.compute_group_coherence(&cluster_features);
        
        if coherence > 0.5 {
            let avg_confidence = cluster_items.iter().map(|item| item.confidence).sum::<f32>() / cluster_items.len() as f32;
            let cross_domain_score = self.compute_cluster_cross_domain_score(&cluster_items);
            
            Some(CorrelationResult {
                correlation_id: format!("cluster_{}", Utc::now().timestamp()),
                evidence_items: cluster_items.clone(),
                correlation_score: coherence,
                confidence: avg_confidence,
                temporal_alignment: 0.9, // High by definition
                cross_domain_score,
                impact_assessment: self.assess_cluster_impact(&cluster_items, coherence),
            })
        } else {
            None
        }
    }

    /// Compute cluster cross-domain score
    fn compute_cluster_cross_domain_score(&self, items: &[EvidenceItem]) -> f32 {
        let mut unique_sources = std::collections::HashSet::new();
        for item in items {
            unique_sources.insert(item.source);
        }
        
        (unique_sources.len() as f32 - 1.0) / 5.0 // Normalize by max possible sources - 1
    }

    /// Assess cluster impact
    fn assess_cluster_impact(&self, items: &[EvidenceItem], coherence: f32) -> ImpactAssessment {
        let avg_severity = items.iter().map(|item| item.severity).sum::<f32>() / items.len() as f32;
        let severity = (avg_severity * coherence).min(1.0);
        
        let scope = "temporal_cluster".to_string();
        
        let mut affected_components = Vec::new();
        for item in items {
            if let Some(component) = item.metadata.get("component") {
                if !affected_components.contains(component) {
                    affected_components.push(component.clone());
                }
            }
        }
        
        let propagation_risk = (severity * items.len() as f32 / 8.0).min(1.0);
        
        ImpactAssessment {
            severity,
            scope,
            affected_components,
            propagation_risk,
        }
    }

    /// Filter and rank correlations
    fn filter_and_rank_correlations(&self, mut correlations: Vec<CorrelationResult>) -> Vec<CorrelationResult> {
        // Filter by minimum correlation score
        correlations.retain(|c| c.correlation_score >= 0.3);
        
        // Sort by combined score
        correlations.sort_by(|a, b| {
            let score_a = a.correlation_score * a.confidence * a.impact_assessment.severity;
            let score_b = b.correlation_score * b.confidence * b.impact_assessment.severity;
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        // Limit number of correlations
        correlations.truncate(20);
        
        correlations
    }

    /// Generate correlation ID
    fn generate_correlation_id(&self, item1: &EvidenceItem, item2: &EvidenceItem) -> String {
        format!("corr_{}_{}", item1.id, item2.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_evidence_scoring() {
        let network = EvidenceScoringNetwork::new(64);
        
        let evidence_items = vec![
            EvidenceItem {
                id: "1".to_string(),
                source: EvidenceSource::KpiDeviation,
                timestamp: Utc::now(),
                severity: 0.8,
                confidence: 0.9,
                features: vec![0.1, 0.2, 0.3],
                metadata: HashMap::new(),
            },
            EvidenceItem {
                id: "2".to_string(),
                source: EvidenceSource::AlarmSequence,
                timestamp: Utc::now() + Duration::seconds(30),
                severity: 0.7,
                confidence: 0.85,
                features: vec![0.2, 0.3, 0.4],
                metadata: HashMap::new(),
            },
        ];
        
        let fused_features = vec![
            vec![0.1; 64],
            vec![0.2; 64],
        ];
        
        let correlations = network.score_correlations(&evidence_items, &fused_features);
        assert!(!correlations.is_empty());
    }
}