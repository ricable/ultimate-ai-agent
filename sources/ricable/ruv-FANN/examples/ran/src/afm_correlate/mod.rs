use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

pub mod cross_attention;
pub mod fusion_network;
pub mod temporal_alignment;
pub mod evidence_scoring;
pub mod hierarchical_attention;
pub mod examples;
pub mod integration_test;

use cross_attention::CrossAttentionMechanism;
use fusion_network::MultiSourceFusionNetwork;
use temporal_alignment::TemporalAlignmentAlgorithm;
use evidence_scoring::EvidenceScoringNetwork;
use hierarchical_attention::HierarchicalAttentionNetwork;

/// Evidence source types for correlation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceSource {
    KpiDeviation,
    AlarmSequence,
    ConfigurationChange,
    TopologyImpact,
    PerformanceMetric,
    LogPattern,
}

/// Evidence item with source metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub id: String,
    pub source: EvidenceSource,
    pub timestamp: DateTime<Utc>,
    pub severity: f32,
    pub confidence: f32,
    pub features: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

/// Correlation result with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    pub correlation_id: String,
    pub evidence_items: Vec<EvidenceItem>,
    pub correlation_score: f32,
    pub confidence: f32,
    pub temporal_alignment: f32,
    pub cross_domain_score: f32,
    pub impact_assessment: ImpactAssessment,
}

/// Impact assessment for correlated events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub severity: f32,
    pub scope: String,
    pub affected_components: Vec<String>,
    pub propagation_risk: f32,
}

/// Evidence bundle with hierarchical structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceBundle {
    pub bundle_id: String,
    pub timestamp: DateTime<Utc>,
    pub correlations: Vec<CorrelationResult>,
    pub hierarchical_score: f32,
    pub multi_scale_features: HashMap<String, Vec<f32>>,
    pub synthesis_report: SynthesisReport,
}

/// Synthesis report for evidence bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisReport {
    pub primary_cause: Option<String>,
    pub contributing_factors: Vec<String>,
    pub timeline: Vec<TimelineEvent>,
    pub recommendations: Vec<String>,
}

/// Timeline event for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub description: String,
    pub evidence_ids: Vec<String>,
}

/// Configuration for correlation engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    pub temporal_window: Duration,
    pub min_correlation_score: f32,
    pub max_evidence_items: usize,
    pub attention_heads: usize,
    pub hidden_dim: usize,
    pub dropout_rate: f32,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            temporal_window: Duration::minutes(15),
            min_correlation_score: 0.7,
            max_evidence_items: 100,
            attention_heads: 8,
            hidden_dim: 256,
            dropout_rate: 0.1,
        }
    }
}

/// Main correlation engine
pub struct CorrelationEngine {
    config: CorrelationConfig,
    cross_attention: CrossAttentionMechanism,
    fusion_network: MultiSourceFusionNetwork,
    temporal_alignment: TemporalAlignmentAlgorithm,
    evidence_scoring: EvidenceScoringNetwork,
    hierarchical_attention: HierarchicalAttentionNetwork,
    evidence_buffer: Arc<RwLock<BTreeMap<DateTime<Utc>, Vec<EvidenceItem>>>>,
}

impl CorrelationEngine {
    pub fn new(config: CorrelationConfig) -> Self {
        Self {
            config: config.clone(),
            cross_attention: CrossAttentionMechanism::new(
                config.hidden_dim,
                config.attention_heads,
                config.dropout_rate,
            ),
            fusion_network: MultiSourceFusionNetwork::new(
                config.hidden_dim,
                6, // Number of evidence sources
            ),
            temporal_alignment: TemporalAlignmentAlgorithm::new(
                config.temporal_window,
            ),
            evidence_scoring: EvidenceScoringNetwork::new(
                config.hidden_dim,
            ),
            hierarchical_attention: HierarchicalAttentionNetwork::new(
                config.hidden_dim,
                vec![4, 8, 16], // Multi-scale levels
            ),
            evidence_buffer: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    /// Add evidence item to buffer
    pub async fn add_evidence(&self, evidence: EvidenceItem) {
        let mut buffer = self.evidence_buffer.write().await;
        buffer.entry(evidence.timestamp)
            .or_insert_with(Vec::new)
            .push(evidence);
        
        // Clean old evidence
        let cutoff = Utc::now() - self.config.temporal_window * 2;
        buffer.retain(|timestamp, _| *timestamp > cutoff);
    }

    /// Correlate evidence across domains
    pub async fn correlate(&self) -> Vec<EvidenceBundle> {
        let buffer = self.evidence_buffer.read().await;
        let mut evidence_groups = self.group_evidence_by_window(&buffer);
        let mut bundles = Vec::new();

        for (window_start, evidence_items) in evidence_groups.drain() {
            if evidence_items.len() < 2 {
                continue;
            }

            // Perform temporal alignment
            let aligned_evidence = self.temporal_alignment.align(&evidence_items);

            // Apply cross-attention between different sources
            let attention_features = self.cross_attention.compute_attention(
                &aligned_evidence,
            );

            // Fuse multi-source features
            let fused_features = self.fusion_network.fuse(
                &attention_features,
                &self.get_source_distribution(&aligned_evidence),
            );

            // Score evidence relationships
            let correlations = self.evidence_scoring.score_correlations(
                &aligned_evidence,
                &fused_features,
            );

            // Apply hierarchical attention for multi-scale analysis
            let hierarchical_features = self.hierarchical_attention.process(
                &fused_features,
                &correlations,
            );

            // Generate evidence bundle
            if let Some(bundle) = self.create_evidence_bundle(
                window_start,
                aligned_evidence,
                correlations,
                hierarchical_features,
            ) {
                bundles.push(bundle);
            }
        }

        bundles
    }

    /// Group evidence by temporal windows
    fn group_evidence_by_window(
        &self,
        buffer: &BTreeMap<DateTime<Utc>, Vec<EvidenceItem>>,
    ) -> HashMap<DateTime<Utc>, Vec<EvidenceItem>> {
        let mut groups = HashMap::new();
        let window_duration = self.config.temporal_window;

        for (timestamp, items) in buffer.iter() {
            let window_start = self.get_window_start(*timestamp, window_duration);
            
            for item in items {
                groups.entry(window_start)
                    .or_insert_with(Vec::new)
                    .push(item.clone());
            }
        }

        // Limit evidence items per window
        for evidence_items in groups.values_mut() {
            if evidence_items.len() > self.config.max_evidence_items {
                // Sort by severity and confidence
                evidence_items.sort_by(|a, b| {
                    let score_a = a.severity * a.confidence;
                    let score_b = b.severity * b.confidence;
                    score_b.partial_cmp(&score_a).unwrap()
                });
                evidence_items.truncate(self.config.max_evidence_items);
            }
        }

        groups
    }

    /// Get window start time
    fn get_window_start(
        &self,
        timestamp: DateTime<Utc>,
        window_duration: Duration,
    ) -> DateTime<Utc> {
        let window_millis = window_duration.num_milliseconds();
        let timestamp_millis = timestamp.timestamp_millis();
        let window_start_millis = (timestamp_millis / window_millis) * window_millis;
        
        DateTime::from_timestamp_millis(window_start_millis).unwrap()
    }

    /// Get source distribution for evidence items
    fn get_source_distribution(
        &self,
        evidence_items: &[EvidenceItem],
    ) -> HashMap<EvidenceSource, f32> {
        let mut distribution = HashMap::new();
        let total = evidence_items.len() as f32;

        for item in evidence_items {
            *distribution.entry(item.source).or_insert(0.0) += 1.0 / total;
        }

        distribution
    }

    /// Create evidence bundle from correlations
    fn create_evidence_bundle(
        &self,
        window_start: DateTime<Utc>,
        evidence_items: Vec<EvidenceItem>,
        correlations: Vec<CorrelationResult>,
        hierarchical_features: HashMap<String, Vec<f32>>,
    ) -> Option<EvidenceBundle> {
        // Filter correlations by minimum score
        let significant_correlations: Vec<_> = correlations
            .into_iter()
            .filter(|c| c.correlation_score >= self.config.min_correlation_score)
            .collect();

        if significant_correlations.is_empty() {
            return None;
        }

        // Calculate hierarchical score
        let hierarchical_score = self.calculate_hierarchical_score(&hierarchical_features);

        // Generate synthesis report
        let synthesis_report = self.synthesize_evidence(
            &evidence_items,
            &significant_correlations,
        );

        Some(EvidenceBundle {
            bundle_id: self.generate_bundle_id(),
            timestamp: window_start,
            correlations: significant_correlations,
            hierarchical_score,
            multi_scale_features: hierarchical_features,
            synthesis_report,
        })
    }

    /// Calculate hierarchical score from features
    fn calculate_hierarchical_score(
        &self,
        features: &HashMap<String, Vec<f32>>,
    ) -> f32 {
        let mut total_score = 0.0;
        let mut count = 0;

        for (scale, feature_vec) in features {
            let scale_weight = match scale.as_str() {
                "fine" => 0.3,
                "medium" => 0.4,
                "coarse" => 0.3,
                _ => 0.0,
            };

            if let Some(max_feature) = feature_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                total_score += max_feature * scale_weight;
                count += 1;
            }
        }

        if count > 0 {
            total_score
        } else {
            0.0
        }
    }

    /// Synthesize evidence into report
    fn synthesize_evidence(
        &self,
        evidence_items: &[EvidenceItem],
        correlations: &[CorrelationResult],
    ) -> SynthesisReport {
        // Build timeline
        let mut timeline = Vec::new();
        for item in evidence_items {
            timeline.push(TimelineEvent {
                timestamp: item.timestamp,
                event_type: format!("{:?}", item.source),
                description: item.metadata.get("description")
                    .cloned()
                    .unwrap_or_else(|| "No description".to_string()),
                evidence_ids: vec![item.id.clone()],
            });
        }
        timeline.sort_by_key(|e| e.timestamp);

        // Identify primary cause (highest severity correlation)
        let primary_cause = correlations
            .iter()
            .max_by(|a, b| {
                a.impact_assessment.severity
                    .partial_cmp(&b.impact_assessment.severity)
                    .unwrap()
            })
            .map(|c| format!(
                "Correlation {} with severity {:.2}",
                c.correlation_id,
                c.impact_assessment.severity
            ));

        // Extract contributing factors
        let contributing_factors: Vec<String> = correlations
            .iter()
            .filter(|c| c.confidence > 0.8)
            .map(|c| format!(
                "{:?} correlation (confidence: {:.2})",
                c.evidence_items.first().map(|e| e.source).unwrap_or(EvidenceSource::KpiDeviation),
                c.confidence
            ))
            .collect();

        // Generate recommendations
        let recommendations = self.generate_recommendations(correlations);

        SynthesisReport {
            primary_cause,
            contributing_factors,
            timeline,
            recommendations,
        }
    }

    /// Generate recommendations based on correlations
    fn generate_recommendations(
        &self,
        correlations: &[CorrelationResult],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        for correlation in correlations {
            if correlation.impact_assessment.propagation_risk > 0.7 {
                recommendations.push(format!(
                    "High propagation risk detected. Isolate affected components: {:?}",
                    correlation.impact_assessment.affected_components
                ));
            }

            if correlation.cross_domain_score > 0.8 {
                recommendations.push(
                    "Strong cross-domain correlation detected. Investigate system-wide impacts."
                        .to_string()
                );
            }

            if correlation.temporal_alignment < 0.5 {
                recommendations.push(
                    "Poor temporal alignment. Consider delayed effects or cascade failures."
                        .to_string()
                );
            }
        }

        recommendations
    }

    /// Generate unique bundle ID
    fn generate_bundle_id(&self) -> String {
        format!("bundle_{}", Utc::now().timestamp_nanos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_correlation_engine() {
        let config = CorrelationConfig::default();
        let engine = CorrelationEngine::new(config);

        // Add test evidence
        let evidence1 = EvidenceItem {
            id: "ev1".to_string(),
            source: EvidenceSource::KpiDeviation,
            timestamp: Utc::now(),
            severity: 0.8,
            confidence: 0.9,
            features: vec![0.1, 0.2, 0.3],
            metadata: HashMap::new(),
        };

        let evidence2 = EvidenceItem {
            id: "ev2".to_string(),
            source: EvidenceSource::AlarmSequence,
            timestamp: Utc::now() + Duration::seconds(30),
            severity: 0.7,
            confidence: 0.85,
            features: vec![0.2, 0.3, 0.4],
            metadata: HashMap::new(),
        };

        engine.add_evidence(evidence1).await;
        engine.add_evidence(evidence2).await;

        // Perform correlation
        let bundles = engine.correlate().await;
        assert!(!bundles.is_empty());
    }
}