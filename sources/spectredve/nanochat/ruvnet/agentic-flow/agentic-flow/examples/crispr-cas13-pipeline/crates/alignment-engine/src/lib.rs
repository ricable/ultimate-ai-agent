//! # Alignment Engine
//!
//! Read alignment module for the CRISPR-Cas13 pipeline.
//! Provides interfaces for aligning sequencing reads to reference genomes.

pub mod bwa;
pub mod error;
pub mod quality;

use async_trait::async_trait;
use data_models::sequencing::{AlignedRead, FastqRecord};
pub use error::{AlignmentError, Result};

/// Main alignment interface
#[async_trait]
pub trait Aligner: Send + Sync {
    /// Align a single read to the reference
    async fn align_read(&self, read: &FastqRecord) -> Result<Option<AlignedRead>>;

    /// Align multiple reads in batch
    async fn align_batch(&self, reads: Vec<FastqRecord>) -> Result<Vec<AlignedRead>>;

    /// Get aligner name
    fn name(&self) -> &str;

    /// Get aligner version
    fn version(&self) -> &str;
}

/// Alignment configuration
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Reference genome path
    pub reference_path: String,
    /// Minimum mapping quality
    pub min_mapq: u8,
    /// Number of threads
    pub threads: usize,
    /// Maximum number of mismatches allowed
    pub max_mismatches: u32,
    /// Seed length for alignment
    pub seed_length: usize,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            reference_path: String::new(),
            min_mapq: 20,
            threads: 4,
            max_mismatches: 3,
            seed_length: 19,
        }
    }
}

/// Alignment statistics collector
pub struct AlignmentStatsCollector {
    pub total_reads: u64,
    pub mapped_reads: u64,
    pub unmapped_reads: u64,
    pub low_quality_reads: u64,
    pub duplicates: u64,
    sum_mapq: u64,
}

impl AlignmentStatsCollector {
    pub fn new() -> Self {
        Self {
            total_reads: 0,
            mapped_reads: 0,
            unmapped_reads: 0,
            low_quality_reads: 0,
            duplicates: 0,
            sum_mapq: 0,
        }
    }

    pub fn record_alignment(&mut self, read: &AlignedRead, min_mapq: u8) {
        self.total_reads += 1;

        if read.is_mapped() {
            self.mapped_reads += 1;
            self.sum_mapq += read.mapq as u64;

            if read.mapq < min_mapq {
                self.low_quality_reads += 1;
            }

            if read.is_duplicate() {
                self.duplicates += 1;
            }
        } else {
            self.unmapped_reads += 1;
        }
    }

    pub fn average_mapq(&self) -> f64 {
        if self.mapped_reads == 0 {
            return 0.0;
        }
        self.sum_mapq as f64 / self.mapped_reads as f64
    }

    pub fn mapping_rate(&self) -> f64 {
        if self.total_reads == 0 {
            return 0.0;
        }
        (self.mapped_reads as f64 / self.total_reads as f64) * 100.0
    }
}

impl Default for AlignmentStatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_config_default() {
        let config = AlignmentConfig::default();
        assert_eq!(config.min_mapq, 20);
        assert_eq!(config.threads, 4);
    }

    #[test]
    fn test_stats_collector() {
        let mut collector = AlignmentStatsCollector::new();
        assert_eq!(collector.total_reads, 0);
        assert_eq!(collector.mapping_rate(), 0.0);
    }
}
