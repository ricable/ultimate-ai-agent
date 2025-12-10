//! Scoring algorithms for off-target sites

use crate::error::Result;
use crate::features::FeatureExtractor;
use crate::ml_model::{LinearModel, MlModel};
use crate::{OffTargetPredictor, PredictionConfig};
use async_trait::async_trait;
use data_models::sequencing::GenomicCoordinate;
use data_models::targets::{
    CrisprTarget, OffTargetFeatures, OffTargetPrediction, OffTargetSite, PredictionMetadata,
};
use rayon::prelude::*;
use std::sync::Arc;
use uuid::Uuid;

/// Default off-target predictor implementation
pub struct DefaultPredictor {
    config: PredictionConfig,
    feature_extractor: Arc<FeatureExtractor>,
    ml_model: Arc<Box<dyn MlModel>>,
}

impl DefaultPredictor {
    /// Create a new predictor with configuration
    pub fn new(config: PredictionConfig) -> Result<Self> {
        let feature_extractor = Arc::new(FeatureExtractor::new());

        let ml_model: Box<dyn MlModel> = if config.use_ml_model {
            if let Some(ref path) = config.model_path {
                Box::new(LinearModel::load(path)?)
            } else {
                Box::new(LinearModel::default_model())
            }
        } else {
            Box::new(LinearModel::default_model())
        };

        Ok(Self {
            config,
            feature_extractor,
            ml_model: Arc::new(ml_model),
        })
    }

    /// Search for potential off-target sites
    fn search_off_targets(&self, target: &CrisprTarget) -> Vec<String> {
        // In production, this would:
        // 1. Use a k-mer index of the reference genome
        // 2. Find all sequences within max_mismatches of the guide RNA
        // 3. Filter by genomic context

        // Mock implementation: generate some test candidates
        let mut candidates = Vec::new();

        // Perfect match
        candidates.push(target.guide_rna.clone());

        // Single mismatches at different positions
        for i in 0..target.guide_rna.len().min(5) {
            let mut mutated = target.guide_rna.clone();
            if let Some(ch) = mutated.chars().nth(i) {
                let new_char = match ch.to_ascii_uppercase() {
                    'A' => 'G',
                    'G' => 'A',
                    'C' => 'U',
                    'U' => 'C',
                    _ => 'N',
                };
                mutated.replace_range(i..=i, &new_char.to_string());
                candidates.push(mutated);
            }
        }

        candidates
    }

    /// Score off-target candidates
    fn score_candidates(
        &self,
        target: &CrisprTarget,
        candidates: Vec<String>,
    ) -> Result<Vec<(String, f64, OffTargetFeatures)>> {
        let scored: Vec<_> = candidates
            .par_iter()
            .filter_map(|candidate| {
                // Extract features
                let features = self.feature_extractor.extract(target, candidate).ok()?;

                // Predict score
                let score = self.ml_model.predict_single(&features).ok()?;

                // Create OffTargetFeatures
                let mismatch_positions: Vec<usize> = target
                    .guide_rna
                    .chars()
                    .zip(candidate.chars())
                    .enumerate()
                    .filter_map(|(i, (a, b))| {
                        if a.to_ascii_uppercase() != b.to_ascii_uppercase() {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();

                let target_gc = self
                    .feature_extractor
                    .calculate_gc_content(&target.guide_rna);
                let candidate_gc = self.feature_extractor.calculate_gc_content(candidate);

                let off_target_features =
                    OffTargetFeatures::from_mismatches(mismatch_positions, target_gc, candidate_gc);

                Some((candidate.clone(), score, off_target_features))
            })
            .collect();

        Ok(scored)
    }

    /// Convert scored candidates to OffTargetSite objects
    fn create_off_target_sites(
        &self,
        target: &CrisprTarget,
        scored: Vec<(String, f64, OffTargetFeatures)>,
    ) -> Vec<OffTargetSite> {
        scored
            .into_iter()
            .enumerate()
            .filter_map(|(i, (sequence, score, features))| {
                // Filter by minimum score threshold
                if score < self.config.min_score {
                    return None;
                }

                // Create mock genomic location
                // In production, this would come from genome alignment
                let location = GenomicCoordinate::new(
                    0,
                    2000 + (i as u64 * 100),
                    2000 + (i as u64 * 100) + sequence.len() as u64,
                    true,
                )
                .ok()?;

                let mismatches = features.mismatch_positions.len() as u8;

                Some(OffTargetSite {
                    id: Uuid::new_v4(),
                    target_id: target.id,
                    sequence,
                    location,
                    mismatches,
                    score,
                    gene_context: None,
                    features,
                })
            })
            .collect()
    }
}

#[async_trait]
impl OffTargetPredictor for DefaultPredictor {
    async fn predict(&self, target: &CrisprTarget) -> Result<OffTargetPrediction> {
        let start = std::time::Instant::now();

        // Search for potential off-targets
        let candidates = self.search_off_targets(target);

        // Score candidates
        let scored = self.score_candidates(target, candidates)?;

        // Create OffTargetSite objects
        let off_targets = self.create_off_target_sites(target, scored);

        let processing_time = start.elapsed().as_secs_f64();

        let metadata = PredictionMetadata {
            timestamp: chrono::Utc::now(),
            model_version: self.ml_model.metadata().version,
            reference_genome: self.config.reference_genome.clone(),
            max_mismatches: self.config.max_mismatches,
            processing_time,
        };

        Ok(OffTargetPrediction {
            target: target.clone(),
            off_targets,
            metadata,
        })
    }

    async fn predict_batch(&self, targets: Vec<CrisprTarget>) -> Result<Vec<OffTargetPrediction>> {
        let mut predictions = Vec::new();

        for target in targets {
            let prediction = self.predict(&target).await?;
            predictions.push(prediction);
        }

        Ok(predictions)
    }

    fn name(&self) -> &str {
        "DefaultPredictor"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }
}

/// CFD (Cutting Frequency Determination) scoring algorithm
/// Based on Doench et al. 2016
pub struct CfdScorer;

impl CfdScorer {
    /// Calculate CFD score for a sequence pair
    pub fn score(guide: &str, target: &str) -> f64 {
        if guide.len() != target.len() {
            return 0.0;
        }

        let mut score = 1.0;

        for (pos, (g, t)) in guide.chars().zip(target.chars()).enumerate() {
            if g.to_ascii_uppercase() != t.to_ascii_uppercase() {
                // Apply position-specific mismatch penalty
                let penalty = Self::mismatch_penalty(pos, g, t);
                score *= penalty;
            }
        }

        score
    }

    /// Get mismatch penalty for specific position and mismatch type
    fn mismatch_penalty(position: usize, guide_base: char, target_base: char) -> f64 {
        // Simplified CFD penalties
        // Real implementation would use experimentally-derived matrix
        let position_weight = 1.0 - (position as f64 / 23.0);

        // Higher penalty for PAM-proximal mismatches
        let base_penalty = match (
            guide_base.to_ascii_uppercase(),
            target_base.to_ascii_uppercase(),
        ) {
            ('A', 'G') | ('G', 'A') => 0.6, // Purine-purine transition
            ('C', 'U') | ('U', 'C') => 0.6, // Pyrimidine-pyrimidine transition
            _ => 0.3,                       // Transversion
        };

        base_penalty * position_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_models::sequencing::GenomicCoordinate;

    fn create_test_target() -> CrisprTarget {
        let location = GenomicCoordinate::new(0, 1000, 1023, true).unwrap();
        CrisprTarget::new(
            "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
            "GUUUUAGAGCUAUGCUGUUUUG".to_string(),
            location,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_default_predictor() {
        let config = PredictionConfig::default();
        let predictor = DefaultPredictor::new(config).unwrap();
        let target = create_test_target();

        let prediction = predictor.predict(&target).await.unwrap();
        assert!(!prediction.off_targets.is_empty());
    }

    #[test]
    fn test_cfd_scoring() {
        let score = CfdScorer::score("ACGT", "ACGT");
        assert_eq!(score, 1.0);

        let score_mismatch = CfdScorer::score("ACGT", "ACCT");
        assert!(score_mismatch < 1.0);
    }
}
