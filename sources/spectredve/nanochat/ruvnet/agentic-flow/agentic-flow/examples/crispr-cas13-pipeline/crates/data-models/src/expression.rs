//! Gene expression analysis data models

use crate::error::DataModelError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents gene expression data for a single sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionSample {
    /// Unique sample identifier
    pub id: Uuid,
    /// Sample name
    pub name: String,
    /// Experimental condition
    pub condition: String,
    /// Replicate number
    pub replicate: u32,
    /// Raw read counts per gene
    pub counts: HashMap<String, u64>,
    /// Normalized expression values (e.g., TPM, FPKM)
    pub normalized: HashMap<String, f64>,
    /// Sample metadata
    pub metadata: SampleMetadata,
}

impl ExpressionSample {
    /// Create a new expression sample
    pub fn new(name: String, condition: String, replicate: u32) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            condition,
            replicate,
            counts: HashMap::new(),
            normalized: HashMap::new(),
            metadata: SampleMetadata::default(),
        }
    }

    /// Get total read count
    pub fn total_counts(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Get number of detected genes (count > 0)
    pub fn detected_genes(&self) -> usize {
        self.counts.values().filter(|&&c| c > 0).count()
    }

    /// Add raw count for a gene
    pub fn add_count(&mut self, gene_id: String, count: u64) {
        self.counts.insert(gene_id, count);
    }

    /// Add normalized expression for a gene
    pub fn add_normalized(&mut self, gene_id: String, value: f64) {
        self.normalized.insert(gene_id, value);
    }
}

/// Sample metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SampleMetadata {
    /// Sequencing depth
    pub sequencing_depth: u64,
    /// Library preparation protocol
    pub library_protocol: Option<String>,
    /// Sample collection date
    pub collection_date: Option<chrono::DateTime<chrono::Utc>>,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

/// Differential expression analysis result for a single gene
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialExpression {
    /// Gene identifier
    pub gene_id: String,
    /// Gene symbol/name
    pub gene_name: Option<String>,
    /// Log2 fold change
    pub log2_fold_change: f64,
    /// Standard error of log2 fold change
    pub log2_fold_change_se: f64,
    /// Base mean expression across all samples
    pub base_mean: f64,
    /// Wald test statistic
    pub wald_statistic: f64,
    /// P-value
    pub pvalue: f64,
    /// Adjusted p-value (FDR)
    pub padj: f64,
}

impl DifferentialExpression {
    /// Check if gene is significantly differentially expressed
    pub fn is_significant(&self, padj_threshold: f64, lfc_threshold: f64) -> bool {
        self.padj < padj_threshold && self.log2_fold_change.abs() > lfc_threshold
    }

    /// Check if gene is up-regulated
    pub fn is_upregulated(&self, padj_threshold: f64, lfc_threshold: f64) -> bool {
        self.is_significant(padj_threshold, lfc_threshold) && self.log2_fold_change > 0.0
    }

    /// Check if gene is down-regulated
    pub fn is_downregulated(&self, padj_threshold: f64, lfc_threshold: f64) -> bool {
        self.is_significant(padj_threshold, lfc_threshold) && self.log2_fold_change < 0.0
    }

    /// Get regulation direction as string
    pub fn regulation_direction(&self, padj_threshold: f64, lfc_threshold: f64) -> String {
        if self.is_upregulated(padj_threshold, lfc_threshold) {
            "up".to_string()
        } else if self.is_downregulated(padj_threshold, lfc_threshold) {
            "down".to_string()
        } else {
            "not_significant".to_string()
        }
    }
}

/// Complete differential expression analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialExpressionAnalysis {
    /// Analysis identifier
    pub id: Uuid,
    /// Control condition name
    pub control_condition: String,
    /// Treatment condition name
    pub treatment_condition: String,
    /// Results for each gene
    pub results: Vec<DifferentialExpression>,
    /// Analysis parameters
    pub parameters: AnalysisParameters,
    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl DifferentialExpressionAnalysis {
    /// Get significantly differentially expressed genes
    pub fn significant_genes(
        &self,
        padj_threshold: f64,
        lfc_threshold: f64,
    ) -> Vec<&DifferentialExpression> {
        self.results
            .iter()
            .filter(|de| de.is_significant(padj_threshold, lfc_threshold))
            .collect()
    }

    /// Count up-regulated genes
    pub fn upregulated_count(&self, padj_threshold: f64, lfc_threshold: f64) -> usize {
        self.results
            .iter()
            .filter(|de| de.is_upregulated(padj_threshold, lfc_threshold))
            .count()
    }

    /// Count down-regulated genes
    pub fn downregulated_count(&self, padj_threshold: f64, lfc_threshold: f64) -> usize {
        self.results
            .iter()
            .filter(|de| de.is_downregulated(padj_threshold, lfc_threshold))
            .count()
    }
}

/// Parameters for differential expression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisParameters {
    /// Statistical test used
    pub test_type: String,
    /// FDR method
    pub fdr_method: String,
    /// Minimum counts threshold
    pub min_counts: u64,
    /// Alpha level for hypothesis testing
    pub alpha: f64,
}

impl Default for AnalysisParameters {
    fn default() -> Self {
        Self {
            test_type: "Wald".to_string(),
            fdr_method: "BH".to_string(),
            min_counts: 10,
            alpha: 0.05,
        }
    }
}

/// Immune pathway enrichment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayEnrichment {
    /// Pathway identifier
    pub pathway_id: String,
    /// Pathway name
    pub pathway_name: String,
    /// Pathway database (e.g., KEGG, Reactome, GO)
    pub database: String,
    /// Number of genes in pathway
    pub pathway_size: usize,
    /// Number of DE genes in pathway
    pub de_genes_in_pathway: usize,
    /// List of DE gene IDs in pathway
    pub gene_ids: Vec<String>,
    /// Enrichment p-value
    pub pvalue: f64,
    /// Adjusted p-value
    pub padj: f64,
    /// Enrichment score
    pub enrichment_score: f64,
}

impl PathwayEnrichment {
    /// Check if pathway is significantly enriched
    pub fn is_significant(&self, padj_threshold: f64) -> bool {
        self.padj < padj_threshold
    }

    /// Calculate gene ratio
    pub fn gene_ratio(&self) -> f64 {
        self.de_genes_in_pathway as f64 / self.pathway_size as f64
    }
}

/// Complete pathway enrichment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayAnalysis {
    /// Analysis identifier
    pub id: Uuid,
    /// Differential expression analysis reference
    pub de_analysis_id: Uuid,
    /// Enrichment results
    pub pathways: Vec<PathwayEnrichment>,
    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl PathwayAnalysis {
    /// Get significantly enriched pathways
    pub fn significant_pathways(&self, padj_threshold: f64) -> Vec<&PathwayEnrichment> {
        self.pathways
            .iter()
            .filter(|p| p.is_significant(padj_threshold))
            .collect()
    }

    /// Get top N enriched pathways by score
    pub fn top_pathways(&self, n: usize) -> Vec<&PathwayEnrichment> {
        let mut sorted = self.pathways.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| {
            b.enrichment_score
                .partial_cmp(&a.enrichment_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_sample() {
        let mut sample = ExpressionSample::new("sample1".to_string(), "control".to_string(), 1);
        sample.add_count("GENE1".to_string(), 100);
        sample.add_count("GENE2".to_string(), 200);

        assert_eq!(sample.total_counts(), 300);
        assert_eq!(sample.detected_genes(), 2);
    }

    #[test]
    fn test_differential_expression_significance() {
        let de = DifferentialExpression {
            gene_id: "GENE1".to_string(),
            gene_name: Some("TestGene".to_string()),
            log2_fold_change: 2.5,
            log2_fold_change_se: 0.3,
            base_mean: 100.0,
            wald_statistic: 8.33,
            pvalue: 1e-10,
            padj: 1e-8,
        };

        assert!(de.is_significant(0.05, 1.0));
        assert!(de.is_upregulated(0.05, 1.0));
        assert!(!de.is_downregulated(0.05, 1.0));
    }

    #[test]
    fn test_pathway_enrichment() {
        let pathway = PathwayEnrichment {
            pathway_id: "PATH:001".to_string(),
            pathway_name: "Immune Response".to_string(),
            database: "KEGG".to_string(),
            pathway_size: 100,
            de_genes_in_pathway: 15,
            gene_ids: vec!["GENE1".to_string(), "GENE2".to_string()],
            pvalue: 0.001,
            padj: 0.01,
            enrichment_score: 3.5,
        };

        assert!(pathway.is_significant(0.05));
        assert_eq!(pathway.gene_ratio(), 0.15);
    }
}
