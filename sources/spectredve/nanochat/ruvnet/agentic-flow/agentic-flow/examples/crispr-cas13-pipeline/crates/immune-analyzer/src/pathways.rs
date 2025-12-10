//! Immune pathway enrichment analysis

use crate::error::Result;
use data_models::expression::{DifferentialExpression, PathwayAnalysis, PathwayEnrichment};
use std::collections::HashMap;
use uuid::Uuid;

/// Pathway enrichment analyzer
pub struct PathwayAnalyzer {
    /// Pathway database
    pathways: HashMap<String, Pathway>,
}

/// Pathway definition
#[derive(Debug, Clone)]
pub struct Pathway {
    pub id: String,
    pub name: String,
    pub database: String,
    pub genes: Vec<String>,
}

impl PathwayAnalyzer {
    /// Create analyzer with default immune pathways
    pub fn new() -> Self {
        let mut pathways = HashMap::new();

        // Add some example immune pathways
        pathways.insert(
            "PATH:004612".to_string(),
            Pathway {
                id: "PATH:004612".to_string(),
                name: "Antigen processing and presentation".to_string(),
                database: "KEGG".to_string(),
                genes: vec![
                    "HLA-A".to_string(),
                    "HLA-B".to_string(),
                    "TAP1".to_string(),
                    "TAP2".to_string(),
                ],
            },
        );

        pathways.insert(
            "PATH:004620".to_string(),
            Pathway {
                id: "PATH:004620".to_string(),
                name: "Toll-like receptor signaling pathway".to_string(),
                database: "KEGG".to_string(),
                genes: vec!["TLR4".to_string(), "MYD88".to_string(), "NFKB1".to_string()],
            },
        );

        Self { pathways }
    }

    /// Perform pathway enrichment analysis
    pub fn analyze(
        &self,
        de_analysis_id: Uuid,
        de_genes: &[DifferentialExpression],
        padj_threshold: f64,
        lfc_threshold: f64,
    ) -> Result<PathwayAnalysis> {
        // Get significantly DE genes
        let sig_genes: Vec<String> = de_genes
            .iter()
            .filter(|de| de.is_significant(padj_threshold, lfc_threshold))
            .map(|de| de.gene_id.clone())
            .collect();

        let total_de_genes = sig_genes.len();

        // Test each pathway for enrichment
        let pathway_results: Vec<PathwayEnrichment> = self
            .pathways
            .values()
            .filter_map(|pathway| {
                let overlap: Vec<String> = pathway
                    .genes
                    .iter()
                    .filter(|gene| sig_genes.contains(gene))
                    .cloned()
                    .collect();

                if overlap.is_empty() {
                    return None;
                }

                let de_genes_in_pathway = overlap.len();

                // Calculate enrichment (Fisher's exact test approximation)
                let pvalue = self.fisher_test_pvalue(
                    de_genes_in_pathway,
                    total_de_genes,
                    pathway.genes.len(),
                    20000, // Approximate total genes
                );

                let enrichment_score = (de_genes_in_pathway as f64 / pathway.genes.len() as f64)
                    / (total_de_genes as f64 / 20000.0);

                Some(PathwayEnrichment {
                    pathway_id: pathway.id.clone(),
                    pathway_name: pathway.name.clone(),
                    database: pathway.database.clone(),
                    pathway_size: pathway.genes.len(),
                    de_genes_in_pathway,
                    gene_ids: overlap,
                    pvalue,
                    padj: pvalue, // Will be corrected later
                    enrichment_score,
                })
            })
            .collect();

        Ok(PathwayAnalysis {
            id: Uuid::new_v4(),
            de_analysis_id,
            pathways: pathway_results,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Fisher's exact test p-value approximation
    fn fisher_test_pvalue(
        &self,
        overlap: usize,
        total_de: usize,
        pathway_size: usize,
        total_genes: usize,
    ) -> f64 {
        // Hypergeometric test approximation
        let k = overlap as f64;
        let n = total_de as f64;
        let m = pathway_size as f64;
        let t = total_genes as f64;

        let expected = (n * m) / t;
        let variance = (n * m * (t - n) * (t - m)) / (t * t * (t - 1.0));

        if variance == 0.0 {
            return 1.0;
        }

        let z = (k - expected) / variance.sqrt();
        let pvalue = 2.0 * (1.0 - Self::normal_cdf(z.abs()));

        pvalue.max(1e-10).min(1.0)
    }

    /// Standard normal CDF
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
}

impl Default for PathwayAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathway_analyzer() {
        let analyzer = PathwayAnalyzer::new();
        assert!(!analyzer.pathways.is_empty());
    }
}
