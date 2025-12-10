//! Feature extraction for off-target prediction

use crate::error::{PredictionError, Result};
use data_models::sequencing::AlignedRead;
use data_models::targets::{CrisprTarget, OffTargetFeatures};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Feature extractor for machine learning models
pub struct FeatureExtractor {
    /// Include position-weighted features
    pub position_weighted: bool,
    /// Include thermodynamic features
    pub thermodynamic: bool,
    /// Include sequence context features
    pub sequence_context: bool,
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            position_weighted: true,
            thermodynamic: true,
            sequence_context: true,
        }
    }

    /// Extract features from a potential off-target site
    pub fn extract(&self, target: &CrisprTarget, candidate_sequence: &str) -> Result<Array1<f64>> {
        let mut features = Vec::new();

        // Basic mismatch features
        let mismatches = self.count_mismatches(&target.guide_rna, candidate_sequence);
        features.push(mismatches as f64);

        // Position-weighted mismatch score
        if self.position_weighted {
            let position_score =
                self.position_weighted_score(&target.guide_rna, candidate_sequence);
            features.push(position_score);
        }

        // GC content features
        let target_gc = self.calculate_gc_content(&target.guide_rna);
        let candidate_gc = self.calculate_gc_content(candidate_sequence);
        features.push(target_gc);
        features.push(candidate_gc);
        features.push((target_gc - candidate_gc).abs());

        // Seed region mismatches (first 10 nucleotides)
        let seed_mismatches = self.count_seed_mismatches(&target.guide_rna, candidate_sequence, 10);
        features.push(seed_mismatches as f64);

        // Thermodynamic features
        if self.thermodynamic {
            let tm_diff = (target.calculate_tm() - self.calculate_tm(candidate_sequence)).abs();
            features.push(tm_diff);
        }

        // Sequence context features
        if self.sequence_context {
            let context = self.extract_context_features(candidate_sequence);
            features.extend(context);
        }

        Ok(Array1::from(features))
    }

    /// Extract features for batch prediction
    pub fn extract_batch(
        &self,
        target: &CrisprTarget,
        candidates: &[String],
    ) -> Result<Array2<f64>> {
        let num_features = self.count_features();
        let mut feature_matrix = Array2::zeros((candidates.len(), num_features));

        for (i, candidate) in candidates.iter().enumerate() {
            let features = self.extract(target, candidate)?;
            feature_matrix.row_mut(i).assign(&features);
        }

        Ok(feature_matrix)
    }

    /// Count number of features extracted
    fn count_features(&self) -> usize {
        let mut count = 6; // Base features
        if self.position_weighted {
            count += 1;
        }
        if self.thermodynamic {
            count += 1;
        }
        if self.sequence_context {
            count += 4; // 4 context features
        }
        count
    }

    /// Count mismatches between two sequences
    fn count_mismatches(&self, seq1: &str, seq2: &str) -> usize {
        seq1.chars()
            .zip(seq2.chars())
            .filter(|(a, b)| a.to_ascii_uppercase() != b.to_ascii_uppercase())
            .count()
    }

    /// Count mismatches in seed region
    fn count_seed_mismatches(&self, seq1: &str, seq2: &str, seed_length: usize) -> usize {
        let len = seed_length.min(seq1.len()).min(seq2.len());
        seq1.chars()
            .take(len)
            .zip(seq2.chars().take(len))
            .filter(|(a, b)| a.to_ascii_uppercase() != b.to_ascii_uppercase())
            .count()
    }

    /// Calculate position-weighted mismatch score
    /// Higher penalties for mismatches closer to 5' end (PAM-proximal)
    fn position_weighted_score(&self, seq1: &str, seq2: &str) -> f64 {
        let mut score = 0.0;
        let length = seq1.len().min(seq2.len());

        for (pos, (a, b)) in seq1.chars().zip(seq2.chars()).enumerate() {
            if a.to_ascii_uppercase() != b.to_ascii_uppercase() {
                // Weight decreases linearly from 5' to 3'
                let weight = 1.0 - (pos as f64 / length as f64);
                score += weight;
            }
        }

        score
    }

    /// Calculate GC content percentage
    pub fn calculate_gc_content(&self, sequence: &str) -> f64 {
        let gc_count = sequence
            .chars()
            .filter(|&c| matches!(c.to_ascii_uppercase(), 'G' | 'C'))
            .count();
        (gc_count as f64 / sequence.len() as f64) * 100.0
    }

    /// Calculate melting temperature
    fn calculate_tm(&self, sequence: &str) -> f64 {
        let gc_count = sequence
            .chars()
            .filter(|&c| matches!(c.to_ascii_uppercase(), 'G' | 'C'))
            .count() as f64;
        let at_count = sequence.len() as f64 - gc_count;
        4.0 * gc_count + 2.0 * at_count
    }

    /// Extract sequence context features
    fn extract_context_features(&self, sequence: &str) -> Vec<f64> {
        vec![
            self.count_homopolymers(sequence) as f64,
            self.calculate_complexity(sequence),
            self.count_dinucleotide_repeats(sequence) as f64,
            self.calculate_purine_content(sequence),
        ]
    }

    /// Count homopolymer runs (e.g., AAAA, GGGG)
    fn count_homopolymers(&self, sequence: &str) -> usize {
        let mut count = 0;
        let mut last_char = '\0';
        let mut run_length = 0;

        for ch in sequence.chars() {
            if ch == last_char {
                run_length += 1;
                if run_length >= 4 {
                    count += 1;
                }
            } else {
                last_char = ch;
                run_length = 1;
            }
        }

        count
    }

    /// Calculate sequence complexity (Shannon entropy)
    fn calculate_complexity(&self, sequence: &str) -> f64 {
        let mut counts: HashMap<char, usize> = HashMap::new();
        for ch in sequence.chars() {
            *counts.entry(ch.to_ascii_uppercase()).or_insert(0) += 1;
        }

        let len = sequence.len() as f64;
        let mut entropy = 0.0;

        for count in counts.values() {
            let freq = *count as f64 / len;
            entropy -= freq * freq.log2();
        }

        entropy
    }

    /// Count dinucleotide repeats
    fn count_dinucleotide_repeats(&self, sequence: &str) -> usize {
        let mut count = 0;
        let chars: Vec<char> = sequence.chars().collect();

        for i in 0..chars.len().saturating_sub(3) {
            if chars[i] == chars[i + 2] && chars[i + 1] == chars[i + 3] {
                count += 1;
            }
        }

        count
    }

    /// Calculate purine content (A, G)
    fn calculate_purine_content(&self, sequence: &str) -> f64 {
        let purine_count = sequence
            .chars()
            .filter(|&c| matches!(c.to_ascii_uppercase(), 'A' | 'G'))
            .count();
        (purine_count as f64 / sequence.len() as f64) * 100.0
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn test_feature_extraction() {
        let extractor = FeatureExtractor::new();
        let target = create_test_target();
        let candidate = "GUUUUAGAGCUAUGCUGUUUUA";

        let features = extractor.extract(&target, candidate).unwrap();
        assert!(features.len() > 0);
    }

    #[test]
    fn test_mismatch_counting() {
        let extractor = FeatureExtractor::new();
        let mismatches = extractor.count_mismatches("ACGT", "ACCT");
        assert_eq!(mismatches, 1);
    }

    #[test]
    fn test_gc_content() {
        let extractor = FeatureExtractor::new();
        let gc = extractor.calculate_gc_content("GCGCGCGC");
        assert_eq!(gc, 100.0);
    }
}
