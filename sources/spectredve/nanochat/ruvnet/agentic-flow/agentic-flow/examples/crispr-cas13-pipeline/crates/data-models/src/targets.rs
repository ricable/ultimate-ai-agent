//! CRISPR target and off-target prediction models

use crate::error::{DataModelError, Result};
use crate::sequencing::GenomicCoordinate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a CRISPR-Cas13 guide RNA target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisprTarget {
    /// Unique identifier
    pub id: Uuid,
    /// Guide RNA sequence
    pub guide_rna: String,
    /// Target RNA sequence
    pub target_sequence: String,
    /// Genomic location
    pub location: GenomicCoordinate,
    /// Target gene name
    pub gene_name: Option<String>,
    /// Target gene ID
    pub gene_id: Option<String>,
    /// Transcript ID
    pub transcript_id: Option<String>,
    /// On-target score
    pub on_target_score: f64,
}

impl CrisprTarget {
    /// Create a new CRISPR target
    pub fn new(
        guide_rna: String,
        target_sequence: String,
        location: GenomicCoordinate,
    ) -> Result<Self> {
        if guide_rna.is_empty() {
            return Err(DataModelError::MissingField("guide_rna".to_string()));
        }

        if target_sequence.is_empty() {
            return Err(DataModelError::MissingField("target_sequence".to_string()));
        }

        Ok(Self {
            id: Uuid::new_v4(),
            guide_rna,
            target_sequence,
            location,
            gene_name: None,
            gene_id: None,
            transcript_id: None,
            on_target_score: 0.0,
        })
    }

    /// Calculate guide RNA melting temperature (Tm)
    pub fn calculate_tm(&self) -> f64 {
        let gc_count = self
            .guide_rna
            .chars()
            .filter(|&c| matches!(c.to_ascii_uppercase(), 'G' | 'C'))
            .count() as f64;
        let at_count = self.guide_rna.len() as f64 - gc_count;

        // Simple Tm calculation: Tm = 4(G+C) + 2(A+T)
        4.0 * gc_count + 2.0 * at_count
    }
}

/// Represents a predicted off-target site
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffTargetSite {
    /// Unique identifier
    pub id: Uuid,
    /// Reference to the original target
    pub target_id: Uuid,
    /// Off-target sequence
    pub sequence: String,
    /// Genomic location
    pub location: GenomicCoordinate,
    /// Number of mismatches with guide RNA
    pub mismatches: u8,
    /// Off-target score (0-1, higher = more likely to be cleaved)
    pub score: f64,
    /// Gene context if within a gene
    pub gene_context: Option<GeneContext>,
    /// Features used for prediction
    pub features: OffTargetFeatures,
}

impl OffTargetSite {
    /// Check if this off-target is high-risk (score > threshold)
    pub fn is_high_risk(&self, threshold: f64) -> bool {
        self.score > threshold
    }

    /// Get severity level based on score and mismatches
    pub fn severity(&self) -> OffTargetSeverity {
        match (self.score, self.mismatches) {
            (s, _) if s > 0.8 => OffTargetSeverity::Critical,
            (s, m) if s > 0.5 && m <= 2 => OffTargetSeverity::High,
            (s, _) if s > 0.3 => OffTargetSeverity::Medium,
            _ => OffTargetSeverity::Low,
        }
    }
}

/// Severity classification for off-target effects
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OffTargetSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Gene context for off-target sites
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneContext {
    pub gene_name: String,
    pub gene_id: String,
    pub transcript_id: Option<String>,
    pub region_type: GeneRegionType,
}

/// Type of gene region
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GeneRegionType {
    Exon,
    Intron,
    UTR5,
    UTR3,
    Promoter,
    Intergenic,
}

/// Features used for off-target prediction
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OffTargetFeatures {
    /// Position-specific mismatch penalties
    pub mismatch_positions: Vec<usize>,
    /// GC content difference from target
    pub gc_content_diff: f64,
    /// Minimum free energy for RNA secondary structure
    pub mfe: Option<f64>,
    /// Seed region mismatches (first 8-10 nt)
    pub seed_mismatches: u8,
    /// Read coverage at this location
    pub coverage: u32,
    /// Additional custom features
    pub custom: HashMap<String, f64>,
}

impl OffTargetFeatures {
    /// Create features from mismatch analysis
    pub fn from_mismatches(mismatch_positions: Vec<usize>, guide_gc: f64, target_gc: f64) -> Self {
        let seed_mismatches = mismatch_positions.iter().filter(|&&pos| pos < 10).count() as u8;

        Self {
            mismatch_positions,
            gc_content_diff: (guide_gc - target_gc).abs(),
            mfe: None,
            seed_mismatches,
            coverage: 0,
            custom: HashMap::new(),
        }
    }
}

/// Results from off-target prediction analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffTargetPrediction {
    /// Target being analyzed
    pub target: CrisprTarget,
    /// Predicted off-target sites
    pub off_targets: Vec<OffTargetSite>,
    /// Prediction metadata
    pub metadata: PredictionMetadata,
}

impl OffTargetPrediction {
    /// Filter off-targets by minimum score threshold
    pub fn filter_by_score(&self, min_score: f64) -> Vec<&OffTargetSite> {
        self.off_targets
            .iter()
            .filter(|ot| ot.score >= min_score)
            .collect()
    }

    /// Get high-risk off-targets
    pub fn high_risk_sites(&self, threshold: f64) -> Vec<&OffTargetSite> {
        self.off_targets
            .iter()
            .filter(|ot| ot.is_high_risk(threshold))
            .collect()
    }

    /// Calculate specificity score (ratio of on-target to sum of off-target scores)
    pub fn specificity_score(&self) -> f64 {
        let off_target_sum: f64 = self.off_targets.iter().map(|ot| ot.score).sum();
        if off_target_sum == 0.0 {
            return 1.0;
        }
        self.target.on_target_score / (self.target.on_target_score + off_target_sum)
    }
}

/// Metadata for prediction runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetadata {
    /// When prediction was performed
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Model version used
    pub model_version: String,
    /// Reference genome used
    pub reference_genome: String,
    /// Maximum mismatches allowed
    pub max_mismatches: u8,
    /// Processing time in seconds
    pub processing_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crispr_target_creation() {
        let location = GenomicCoordinate::new(0, 1000, 1023, true).unwrap();
        let target = CrisprTarget::new(
            "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
            "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
            location,
        );
        assert!(target.is_ok());
    }

    #[test]
    fn test_off_target_severity() {
        let location = GenomicCoordinate::new(0, 2000, 2023, true).unwrap();
        let site = OffTargetSite {
            id: Uuid::new_v4(),
            target_id: Uuid::new_v4(),
            sequence: "GUUUUAGAGCUAUGCUGUUUUA".to_string(),
            location,
            mismatches: 1,
            score: 0.85,
            gene_context: None,
            features: OffTargetFeatures::default(),
        };

        assert_eq!(site.severity(), OffTargetSeverity::Critical);
    }

    #[test]
    fn test_specificity_score() {
        let location = GenomicCoordinate::new(0, 1000, 1023, true).unwrap();
        let mut target = CrisprTarget::new(
            "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
            "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
            location,
        )
        .unwrap();
        target.on_target_score = 0.9;

        let prediction = OffTargetPrediction {
            target,
            off_targets: vec![],
            metadata: PredictionMetadata {
                timestamp: chrono::Utc::now(),
                model_version: "1.0.0".to_string(),
                reference_genome: "GRCh38".to_string(),
                max_mismatches: 3,
                processing_time: 1.5,
            },
        };

        assert_eq!(prediction.specificity_score(), 1.0);
    }
}
