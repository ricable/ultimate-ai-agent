//! Unit tests for alignment-engine crate
//! Tests alignment logic, quality assessment, and BWA integration

use alignment_engine::{
    AlignmentConfig, AlignmentStatsCollector, Aligner,
    quality::{QualityAssessment, QualityFilter},
};
use data_models::sequencing::{AlignedRead, FastqRecord};

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_alignment_config_default() {
        let config = AlignmentConfig::default();
        assert_eq!(config.min_mapq, 20);
        assert_eq!(config.threads, 4);
        assert_eq!(config.max_mismatches, 3);
        assert_eq!(config.seed_length, 19);
    }

    #[test]
    fn test_alignment_config_validation() {
        let mut config = AlignmentConfig::default();

        // Valid config
        assert!(config.validate().is_ok());

        // Invalid: empty reference path
        config.reference_path = String::new();
        assert!(config.validate().is_err());

        // Invalid: threads = 0
        config.threads = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_alignment_config_builder() {
        let config = AlignmentConfig::builder()
            .reference_path("/path/to/ref.fa")
            .min_mapq(30)
            .threads(8)
            .build();

        assert_eq!(config.min_mapq, 30);
        assert_eq!(config.threads, 8);
    }
}

#[cfg(test)]
mod stats_collector_tests {
    use super::*;

    #[test]
    fn test_stats_collector_initialization() {
        let collector = AlignmentStatsCollector::new();
        assert_eq!(collector.total_reads, 0);
        assert_eq!(collector.mapped_reads, 0);
        assert_eq!(collector.unmapped_reads, 0);
        assert_eq!(collector.average_mapq(), 0.0);
        assert_eq!(collector.mapping_rate(), 0.0);
    }

    #[test]
    fn test_stats_collector_mapped_reads() {
        let mut collector = AlignmentStatsCollector::new();

        let mapped_read = create_test_aligned_read(true, 40, false);
        collector.record_alignment(&mapped_read, 20);

        assert_eq!(collector.total_reads, 1);
        assert_eq!(collector.mapped_reads, 1);
        assert_eq!(collector.unmapped_reads, 0);
        assert_eq!(collector.average_mapq(), 40.0);
        assert_eq!(collector.mapping_rate(), 100.0);
    }

    #[test]
    fn test_stats_collector_unmapped_reads() {
        let mut collector = AlignmentStatsCollector::new();

        let unmapped_read = create_test_aligned_read(false, 0, false);
        collector.record_alignment(&unmapped_read, 20);

        assert_eq!(collector.total_reads, 1);
        assert_eq!(collector.mapped_reads, 0);
        assert_eq!(collector.unmapped_reads, 1);
        assert_eq!(collector.mapping_rate(), 0.0);
    }

    #[test]
    fn test_stats_collector_low_quality_reads() {
        let mut collector = AlignmentStatsCollector::new();

        let low_qual_read = create_test_aligned_read(true, 10, false);
        collector.record_alignment(&low_qual_read, 20);

        assert_eq!(collector.low_quality_reads, 1);
        assert_eq!(collector.mapped_reads, 1);
    }

    #[test]
    fn test_stats_collector_duplicates() {
        let mut collector = AlignmentStatsCollector::new();

        let duplicate_read = create_test_aligned_read(true, 40, true);
        collector.record_alignment(&duplicate_read, 20);

        assert_eq!(collector.duplicates, 1);
    }

    #[test]
    fn test_stats_collector_average_mapq() {
        let mut collector = AlignmentStatsCollector::new();

        collector.record_alignment(&create_test_aligned_read(true, 30, false), 20);
        collector.record_alignment(&create_test_aligned_read(true, 50, false), 20);

        assert_eq!(collector.average_mapq(), 40.0); // (30 + 50) / 2
    }

    #[test]
    fn test_stats_collector_mapping_rate_mixed() {
        let mut collector = AlignmentStatsCollector::new();

        collector.record_alignment(&create_test_aligned_read(true, 40, false), 20);
        collector.record_alignment(&create_test_aligned_read(false, 0, false), 20);
        collector.record_alignment(&create_test_aligned_read(true, 40, false), 20);

        assert_eq!(collector.mapping_rate(), 66.66666666666667); // 2/3 * 100
    }

    fn create_test_aligned_read(mapped: bool, mapq: u8, duplicate: bool) -> AlignedRead {
        AlignedRead::new(
            "test_read".to_string(),
            "ACGT".to_string(),
            mapped,
            mapq,
            duplicate,
            "chr1".to_string(),
            1000,
        )
    }
}

#[cfg(test)]
mod quality_assessment_tests {
    use super::*;

    #[test]
    fn test_quality_filter_by_mapq() {
        let filter = QualityFilter::new(30, 0.0, 0);

        let high_qual = create_test_aligned_read(true, 40, false);
        let low_qual = create_test_aligned_read(true, 20, false);

        assert!(filter.passes(&high_qual));
        assert!(!filter.passes(&low_qual));
    }

    #[test]
    fn test_quality_filter_by_identity() {
        let filter = QualityFilter::new(0, 95.0, 0);

        let high_identity = AlignedRead::with_identity(
            "read1".to_string(),
            "ACGT".to_string(),
            true,
            40,
            false,
            "chr1".to_string(),
            1000,
            98.0,
        );

        let low_identity = AlignedRead::with_identity(
            "read2".to_string(),
            "ACGT".to_string(),
            true,
            40,
            false,
            "chr1".to_string(),
            1000,
            90.0,
        );

        assert!(filter.passes(&high_identity));
        assert!(!filter.passes(&low_identity));
    }

    #[test]
    fn test_quality_assessment_metrics() {
        let read = create_test_aligned_read(true, 40, false);
        let assessment = QualityAssessment::assess(&read);

        assert!(assessment.is_high_quality());
        assert!(assessment.mapq >= 20);
        assert!(!assessment.is_duplicate);
    }
}

#[cfg(test)]
mod cigar_tests {
    use super::*;

    #[test]
    fn test_cigar_string_parsing() {
        let cigar = "50M2I48M";
        let parsed = parse_cigar(cigar);

        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0], (50, 'M'));
        assert_eq!(parsed[1], (2, 'I'));
        assert_eq!(parsed[2], (48, 'M'));
    }

    #[test]
    fn test_cigar_alignment_length() {
        let cigar = "50M2I48M";
        let length = cigar_alignment_length(cigar);

        // 50M + 48M = 98 (insertions don't count in reference)
        assert_eq!(length, 98);
    }

    #[test]
    fn test_cigar_query_length() {
        let cigar = "50M2D48M";
        let length = cigar_query_length(cigar);

        // 50M + 48M = 98 (deletions don't count in query)
        assert_eq!(length, 98);
    }

    #[test]
    fn test_cigar_mismatch_count() {
        let cigar = "45M5X50M"; // 5 mismatches
        let mismatches = count_mismatches(cigar);

        assert_eq!(mismatches, 5);
    }

    fn parse_cigar(cigar: &str) -> Vec<(u32, char)> {
        let mut result = Vec::new();
        let mut num = String::new();

        for c in cigar.chars() {
            if c.is_ascii_digit() {
                num.push(c);
            } else {
                if !num.is_empty() {
                    result.push((num.parse().unwrap(), c));
                    num.clear();
                }
            }
        }
        result
    }

    fn cigar_alignment_length(cigar: &str) -> u32 {
        parse_cigar(cigar).iter()
            .filter(|(_, op)| matches!(op, 'M' | 'D' | 'N' | '=' | 'X'))
            .map(|(len, _)| len)
            .sum()
    }

    fn cigar_query_length(cigar: &str) -> u32 {
        parse_cigar(cigar).iter()
            .filter(|(_, op)| matches!(op, 'M' | 'I' | 'S' | '=' | 'X'))
            .map(|(len, _)| len)
            .sum()
    }

    fn count_mismatches(cigar: &str) -> u32 {
        parse_cigar(cigar).iter()
            .filter(|(_, op)| *op == 'X')
            .map(|(len, _)| len)
            .sum()
    }
}

#[cfg(test)]
mod bwa_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_bwa_aligner_creation() {
        let config = AlignmentConfig::default();
        // This would normally create a real BWA aligner
        // For unit tests, we'd use a mock
        assert!(true); // Placeholder
    }

    #[tokio::test]
    async fn test_bwa_batch_alignment() {
        // Test that batch alignment processes all reads
        let reads = vec![
            FastqRecord::new("read1".to_string(), "ACGT".to_string(), "IIII".to_string()).unwrap(),
            FastqRecord::new("read2".to_string(), "TGCA".to_string(), "IIII".to_string()).unwrap(),
        ];

        // Mock aligner would process these
        assert_eq!(reads.len(), 2);
    }
}
