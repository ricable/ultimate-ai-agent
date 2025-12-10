//! DESeq2-style differential expression analysis

use crate::error::{AnalysisError, Result};
use data_models::expression::{
    AnalysisParameters, DifferentialExpression, DifferentialExpressionAnalysis, ExpressionSample,
};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;
use uuid::Uuid;

/// DESeq2 analyzer for differential expression
pub struct DeseqAnalyzer {
    params: AnalysisParameters,
}

impl DeseqAnalyzer {
    pub fn new(params: AnalysisParameters) -> Self {
        Self { params }
    }

    /// Perform differential expression analysis
    pub fn analyze(
        &self,
        control_samples: Vec<ExpressionSample>,
        treatment_samples: Vec<ExpressionSample>,
    ) -> Result<DifferentialExpressionAnalysis> {
        if control_samples.is_empty() || treatment_samples.is_empty() {
            return Err(AnalysisError::InsufficientData(
                "Need at least one sample per condition".to_string(),
            ));
        }

        // Get all genes
        let genes = self.collect_genes(&control_samples, &treatment_samples);

        // Perform analysis for each gene
        let results: Vec<DifferentialExpression> = genes
            .par_iter()
            .filter_map(|gene_id| {
                self.analyze_gene(gene_id, &control_samples, &treatment_samples)
                    .ok()
            })
            .collect();

        Ok(DifferentialExpressionAnalysis {
            id: Uuid::new_v4(),
            control_condition: control_samples[0].condition.clone(),
            treatment_condition: treatment_samples[0].condition.clone(),
            results,
            parameters: self.params.clone(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Collect all genes present in samples
    fn collect_genes(
        &self,
        control_samples: &[ExpressionSample],
        treatment_samples: &[ExpressionSample],
    ) -> Vec<String> {
        let mut genes = std::collections::HashSet::new();

        for sample in control_samples.iter().chain(treatment_samples.iter()) {
            for gene_id in sample.counts.keys() {
                genes.insert(gene_id.clone());
            }
        }

        genes.into_iter().collect()
    }

    /// Analyze a single gene
    fn analyze_gene(
        &self,
        gene_id: &str,
        control_samples: &[ExpressionSample],
        treatment_samples: &[ExpressionSample],
    ) -> Result<DifferentialExpression> {
        // Get counts for this gene
        let control_counts: Vec<u64> = control_samples
            .iter()
            .map(|s| *s.counts.get(gene_id).unwrap_or(&0))
            .collect();

        let treatment_counts: Vec<u64> = treatment_samples
            .iter()
            .map(|s| *s.counts.get(gene_id).unwrap_or(&0))
            .collect();

        // Filter low-count genes
        let total_count: u64 = control_counts.iter().chain(treatment_counts.iter()).sum();
        if total_count < self.params.min_counts {
            return Err(AnalysisError::InsufficientData("Low counts".to_string()));
        }

        // Calculate base mean
        let base_mean = total_count as f64 / (control_counts.len() + treatment_counts.len()) as f64;

        // Calculate log2 fold change
        let control_mean = Self::geometric_mean(&control_counts);
        let treatment_mean = Self::geometric_mean(&treatment_counts);
        let log2_fold_change = (treatment_mean / control_mean).log2();

        // Calculate standard error (simplified)
        let log2_fold_change_se = self.estimate_standard_error(&control_counts, &treatment_counts);

        // Calculate Wald statistic
        let wald_statistic = log2_fold_change / log2_fold_change_se;

        // Calculate p-value from Wald statistic
        let pvalue = self.wald_test_pvalue(wald_statistic);

        // Adjusted p-value (will be calculated later with all genes)
        let padj = pvalue; // Placeholder

        Ok(DifferentialExpression {
            gene_id: gene_id.to_string(),
            gene_name: None,
            log2_fold_change,
            log2_fold_change_se,
            base_mean,
            wald_statistic,
            pvalue,
            padj,
        })
    }

    /// Calculate geometric mean
    fn geometric_mean(values: &[u64]) -> f64 {
        let product: f64 = values.iter().map(|&v| (v + 1) as f64).product();
        product.powf(1.0 / values.len() as f64)
    }

    /// Estimate standard error (simplified)
    fn estimate_standard_error(&self, control: &[u64], treatment: &[u64]) -> f64 {
        let control_var = Self::variance(control);
        let treatment_var = Self::variance(treatment);

        ((control_var / control.len() as f64) + (treatment_var / treatment.len() as f64)).sqrt()
    }

    /// Calculate variance
    fn variance(values: &[u64]) -> f64 {
        let mean = values.iter().sum::<u64>() as f64 / values.len() as f64;
        values
            .iter()
            .map(|&v| {
                let diff = v as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64
    }

    /// Calculate p-value from Wald statistic (using normal approximation)
    fn wald_test_pvalue(&self, wald_stat: f64) -> f64 {
        // Two-tailed test
        let z = wald_stat.abs();
        2.0 * (1.0 - Self::normal_cdf(z))
    }

    /// Standard normal CDF approximation
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.5 * x.abs());
        let tau = t
            * (-x * x - 1.26551223
                + t * (1.00002368
                    + t * (0.37409196
                        + t * (0.09678418
                            + t * (-0.18628806
                                + t * (0.27886807
                                    + t * (-1.13520398
                                        + t * (1.48851587
                                            + t * (-0.82215223 + t * 0.17087277)))))))))
                .exp();

        if x >= 0.0 {
            1.0 - tau
        } else {
            tau - 1.0
        }
    }

    /// Apply FDR correction (Benjamini-Hochberg)
    pub fn fdr_correction(pvalues: &mut [DifferentialExpression]) {
        let n = pvalues.len();
        if n == 0 {
            return;
        }

        // Sort by p-value
        pvalues.sort_by(|a, b| a.pvalue.partial_cmp(&b.pvalue).unwrap());

        // Apply BH correction
        for (i, de) in pvalues.iter_mut().enumerate() {
            de.padj = de.pvalue * (n as f64) / ((i + 1) as f64);
            de.padj = de.padj.min(1.0);
        }

        // Enforce monotonicity
        for i in (0..n - 1).rev() {
            if pvalues[i].padj > pvalues[i + 1].padj {
                pvalues[i].padj = pvalues[i + 1].padj;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sample(
        name: &str,
        condition: &str,
        counts: Vec<(String, u64)>,
    ) -> ExpressionSample {
        let mut sample = ExpressionSample::new(name.to_string(), condition.to_string(), 1);
        for (gene, count) in counts {
            sample.add_count(gene, count);
        }
        sample
    }

    #[test]
    fn test_deseq_analysis() {
        let params = AnalysisParameters::default();
        let analyzer = DeseqAnalyzer::new(params);

        let control = vec![
            create_test_sample(
                "C1",
                "control",
                vec![("GENE1".to_string(), 100), ("GENE2".to_string(), 50)],
            ),
            create_test_sample(
                "C2",
                "control",
                vec![("GENE1".to_string(), 110), ("GENE2".to_string(), 45)],
            ),
        ];

        let treatment = vec![
            create_test_sample(
                "T1",
                "treatment",
                vec![("GENE1".to_string(), 200), ("GENE2".to_string(), 30)],
            ),
            create_test_sample(
                "T2",
                "treatment",
                vec![("GENE1".to_string(), 220), ("GENE2".to_string(), 35)],
            ),
        ];

        let result = analyzer.analyze(control, treatment).unwrap();
        assert!(!result.results.is_empty());
    }
}
