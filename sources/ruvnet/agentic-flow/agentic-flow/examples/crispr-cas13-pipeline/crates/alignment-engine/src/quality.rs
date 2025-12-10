//! Quality scoring and filtering for aligned reads

use crate::error::Result;
use data_models::sequencing::AlignedRead;

/// Quality metrics for aligned reads
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Mapping quality score
    pub mapq: u8,
    /// Average base quality
    pub avg_base_quality: f64,
    /// Number of mismatches
    pub mismatches: u32,
    /// Alignment score
    pub alignment_score: i32,
    /// Percent identity
    pub percent_identity: f64,
}

impl QualityMetrics {
    /// Calculate quality metrics for an aligned read
    pub fn from_aligned_read(read: &AlignedRead) -> Self {
        let avg_base_quality = Self::calculate_average_quality(&read.quality);
        let mismatches = read.nm.unwrap_or(0);
        let percent_identity = Self::calculate_percent_identity(read, mismatches);

        Self {
            mapq: read.mapq,
            avg_base_quality,
            mismatches,
            alignment_score: (read.mapq as i32) * 2,
            percent_identity,
        }
    }

    /// Calculate average base quality from Phred scores
    fn calculate_average_quality(quality: &[u8]) -> f64 {
        if quality.is_empty() {
            return 0.0;
        }
        let sum: u32 = quality.iter().map(|&q| (q - 33) as u32).sum();
        sum as f64 / quality.len() as f64
    }

    /// Calculate percent identity for alignment
    fn calculate_percent_identity(read: &AlignedRead, mismatches: u32) -> f64 {
        let read_length = read.sequence.len() as f64;
        if read_length == 0.0 {
            return 0.0;
        }
        ((read_length - mismatches as f64) / read_length) * 100.0
    }

    /// Check if read passes quality thresholds
    pub fn passes_thresholds(&self, min_mapq: u8, min_identity: f64) -> bool {
        self.mapq >= min_mapq && self.percent_identity >= min_identity
    }
}

/// Quality filter for aligned reads
pub struct QualityFilter {
    /// Minimum mapping quality
    pub min_mapq: u8,
    /// Minimum base quality
    pub min_base_quality: f64,
    /// Maximum mismatches allowed
    pub max_mismatches: u32,
    /// Minimum percent identity
    pub min_percent_identity: f64,
    /// Filter duplicates
    pub filter_duplicates: bool,
    /// Filter secondary alignments
    pub filter_secondary: bool,
}

impl QualityFilter {
    /// Create a new quality filter with default settings
    pub fn new() -> Self {
        Self {
            min_mapq: 20,
            min_base_quality: 25.0,
            max_mismatches: 3,
            min_percent_identity: 95.0,
            filter_duplicates: true,
            filter_secondary: true,
        }
    }

    /// Create a strict quality filter
    pub fn strict() -> Self {
        Self {
            min_mapq: 30,
            min_base_quality: 30.0,
            max_mismatches: 2,
            min_percent_identity: 98.0,
            filter_duplicates: true,
            filter_secondary: true,
        }
    }

    /// Create a lenient quality filter
    pub fn lenient() -> Self {
        Self {
            min_mapq: 10,
            min_base_quality: 20.0,
            max_mismatches: 5,
            min_percent_identity: 90.0,
            filter_duplicates: false,
            filter_secondary: false,
        }
    }

    /// Check if a read passes all filters
    pub fn passes(&self, read: &AlignedRead) -> bool {
        // Check if read is mapped
        if !read.is_mapped() {
            return false;
        }

        // Filter duplicates
        if self.filter_duplicates && read.is_duplicate() {
            return false;
        }

        // Filter secondary alignments
        if self.filter_secondary && !read.is_primary() {
            return false;
        }

        // Check mapping quality
        if read.mapq < self.min_mapq {
            return false;
        }

        // Calculate and check quality metrics
        let metrics = QualityMetrics::from_aligned_read(read);

        if metrics.avg_base_quality < self.min_base_quality {
            return false;
        }

        if metrics.mismatches > self.max_mismatches {
            return false;
        }

        if metrics.percent_identity < self.min_percent_identity {
            return false;
        }

        true
    }

    /// Filter a collection of reads
    pub fn filter_reads(&self, reads: Vec<AlignedRead>) -> Vec<AlignedRead> {
        reads.into_iter().filter(|read| self.passes(read)).collect()
    }

    /// Count how many reads pass the filter
    pub fn count_passing(&self, reads: &[AlignedRead]) -> usize {
        reads.iter().filter(|read| self.passes(read)).count()
    }
}

impl Default for QualityFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality statistics for a batch of reads
#[derive(Debug, Default)]
pub struct BatchQualityStats {
    pub total_reads: usize,
    pub passed_reads: usize,
    pub failed_mapq: usize,
    pub failed_base_quality: usize,
    pub failed_mismatches: usize,
    pub failed_identity: usize,
    pub duplicates: usize,
    pub secondary: usize,
}

impl BatchQualityStats {
    /// Calculate passing rate
    pub fn passing_rate(&self) -> f64 {
        if self.total_reads == 0 {
            return 0.0;
        }
        (self.passed_reads as f64 / self.total_reads as f64) * 100.0
    }

    /// Generate detailed statistics for a batch of reads
    pub fn analyze(reads: &[AlignedRead], filter: &QualityFilter) -> Self {
        let mut stats = Self {
            total_reads: reads.len(),
            ..Default::default()
        };

        for read in reads {
            if filter.passes(read) {
                stats.passed_reads += 1;
            } else {
                // Categorize failures
                if read.mapq < filter.min_mapq {
                    stats.failed_mapq += 1;
                }

                let metrics = QualityMetrics::from_aligned_read(read);
                if metrics.avg_base_quality < filter.min_base_quality {
                    stats.failed_base_quality += 1;
                }
                if metrics.mismatches > filter.max_mismatches {
                    stats.failed_mismatches += 1;
                }
                if metrics.percent_identity < filter.min_percent_identity {
                    stats.failed_identity += 1;
                }
            }

            if read.is_duplicate() {
                stats.duplicates += 1;
            }
            if !read.is_primary() {
                stats.secondary += 1;
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_models::sequencing::GenomicCoordinate;
    use uuid::Uuid;

    fn create_test_read(mapq: u8, quality: Vec<u8>, nm: u32) -> AlignedRead {
        let position = GenomicCoordinate::new(0, 1000, 1100, true).unwrap();
        AlignedRead {
            id: Uuid::new_v4(),
            qname: "test_read".to_string(),
            position,
            mapq,
            cigar: "100M".to_string(),
            sequence: "A".repeat(100),
            quality,
            flags: 0,
            nm: Some(nm),
        }
    }

    #[test]
    fn test_quality_metrics() {
        let read = create_test_read(30, vec![40; 100], 2);
        let metrics = QualityMetrics::from_aligned_read(&read);

        assert_eq!(metrics.mapq, 30);
        assert_eq!(metrics.avg_base_quality, 7.0); // 40 - 33 = 7
        assert_eq!(metrics.mismatches, 2);
    }

    #[test]
    fn test_quality_filter_default() {
        let filter = QualityFilter::new();
        let good_read = create_test_read(25, vec![58; 100], 1); // Quality 25
        let bad_read = create_test_read(15, vec![53; 100], 1); // Quality 20

        assert!(filter.passes(&good_read));
        assert!(!filter.passes(&bad_read));
    }

    #[test]
    fn test_batch_quality_stats() {
        let filter = QualityFilter::new();
        let reads = vec![
            create_test_read(30, vec![58; 100], 1),
            create_test_read(15, vec![58; 100], 1),
            create_test_read(25, vec![53; 100], 1),
        ];

        let stats = BatchQualityStats::analyze(&reads, &filter);
        assert_eq!(stats.total_reads, 3);
        assert!(stats.passing_rate() > 0.0);
    }
}
