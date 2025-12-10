//! Count normalization methods

use crate::error::Result;
use data_models::expression::ExpressionSample;
use std::collections::HashMap;

/// Count normalization methods
pub trait Normalizer: Send + Sync {
    /// Normalize count data
    fn normalize(&self, samples: &mut [ExpressionSample]) -> Result<()>;

    /// Get normalizer name
    fn name(&self) -> &str;
}

/// TPM (Transcripts Per Million) normalizer
pub struct TpmNormalizer;

impl TpmNormalizer {
    pub fn new() -> Self {
        Self
    }
}

impl Normalizer for TpmNormalizer {
    fn normalize(&self, samples: &mut [ExpressionSample]) -> Result<()> {
        for sample in samples.iter_mut() {
            let total: u64 = sample.counts.values().sum();
            let scaling_factor = 1_000_000.0 / total as f64;

            sample.normalized = sample
                .counts
                .iter()
                .map(|(gene, &count)| (gene.clone(), count as f64 * scaling_factor))
                .collect();
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "TPM"
    }
}

impl Default for TpmNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// DESeq2-style size factor normalization
pub struct SizeFactorNormalizer;

impl SizeFactorNormalizer {
    pub fn new() -> Self {
        Self
    }

    /// Calculate size factors for samples
    fn calculate_size_factors(&self, samples: &[ExpressionSample]) -> Vec<f64> {
        // Get all genes
        let mut all_genes = std::collections::HashSet::new();
        for sample in samples {
            all_genes.extend(sample.counts.keys().cloned());
        }

        // Calculate geometric means across samples for each gene
        let mut geo_means: HashMap<String, f64> = HashMap::new();
        for gene in &all_genes {
            let values: Vec<u64> = samples
                .iter()
                .map(|s| *s.counts.get(gene).unwrap_or(&0))
                .filter(|&v| v > 0)
                .collect();

            if !values.is_empty() {
                let product: f64 = values.iter().map(|&v| v as f64).product();
                let geo_mean = product.powf(1.0 / values.len() as f64);
                geo_means.insert(gene.clone(), geo_mean);
            }
        }

        // Calculate size factor for each sample
        samples
            .iter()
            .map(|sample| {
                let ratios: Vec<f64> = geo_means
                    .iter()
                    .filter_map(|(gene, &geo_mean)| {
                        let count = *sample.counts.get(gene).unwrap_or(&0) as f64;
                        if count > 0.0 && geo_mean > 0.0 {
                            Some(count / geo_mean)
                        } else {
                            None
                        }
                    })
                    .collect();

                Self::median(&ratios)
            })
            .collect()
    }

    /// Calculate median of values
    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted.len();
        if len == 0 {
            return 1.0;
        }

        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }
}

impl Normalizer for SizeFactorNormalizer {
    fn normalize(&self, samples: &mut [ExpressionSample]) -> Result<()> {
        let size_factors = self.calculate_size_factors(samples);

        for (sample, &size_factor) in samples.iter_mut().zip(size_factors.iter()) {
            sample.normalized = sample
                .counts
                .iter()
                .map(|(gene, &count)| (gene.clone(), count as f64 / size_factor))
                .collect();
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "SizeFactor"
    }
}

impl Default for SizeFactorNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sample(counts: Vec<(String, u64)>) -> ExpressionSample {
        let mut sample = ExpressionSample::new("test".to_string(), "test".to_string(), 1);
        for (gene, count) in counts {
            sample.add_count(gene, count);
        }
        sample
    }

    #[test]
    fn test_tpm_normalization() {
        let normalizer = TpmNormalizer::new();
        let mut samples = vec![create_test_sample(vec![
            ("GENE1".to_string(), 100),
            ("GENE2".to_string(), 200),
        ])];

        normalizer.normalize(&mut samples).unwrap();
        let total: f64 = samples[0].normalized.values().sum();
        assert!((total - 1_000_000.0).abs() < 1.0);
    }
}
