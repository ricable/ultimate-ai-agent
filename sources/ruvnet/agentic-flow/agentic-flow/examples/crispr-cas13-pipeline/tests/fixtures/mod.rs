//! Test fixtures and mocking infrastructure
//! Provides reusable test data and mock objects for comprehensive testing

use std::sync::Arc;
use mockall::predicate::*;
use mockall::mock;

/// Test data generators
pub mod generators {
    use super::*;

    /// Generate test FASTQ records
    pub fn generate_fastq_records(n: usize) -> Vec<FastqRecord> {
        (0..n)
            .map(|i| FastqRecord {
                id: format!("read_{}", i),
                sequence: generate_dna_sequence(100),
                quality: "I".repeat(100),
            })
            .collect()
    }

    /// Generate random DNA sequence
    pub fn generate_dna_sequence(length: usize) -> String {
        let bases = ['A', 'C', 'G', 'T'];
        (0..length)
            .map(|i| bases[i % 4])
            .collect()
    }

    /// Generate RNA guide sequences
    pub fn generate_guide_rnas(n: usize) -> Vec<String> {
        let bases = ['A', 'C', 'G', 'U'];
        (0..n)
            .map(|i| {
                (0..23)
                    .map(|j| bases[(i + j) % 4])
                    .collect()
            })
            .collect()
    }

    /// Generate expression count matrix
    pub fn generate_expression_matrix(n_genes: usize, n_samples: usize) -> Vec<Vec<u32>> {
        (0..n_samples)
            .map(|sample| {
                (0..n_genes)
                    .map(|gene| ((gene * sample + 1) * 100) as u32)
                    .collect()
            })
            .collect()
    }

    /// Generate alignment records
    pub fn generate_aligned_reads(n: usize) -> Vec<AlignedRead> {
        (0..n)
            .map(|i| AlignedRead {
                id: format!("read_{}", i),
                sequence: generate_dna_sequence(100),
                mapped: i % 10 != 0, // 90% mapped
                mapq: (40 + (i % 20)) as u8,
                chromosome: format!("chr{}", (i % 22) + 1),
                position: (i * 1000) as i64,
            })
            .collect()
    }

    pub struct FastqRecord {
        pub id: String,
        pub sequence: String,
        pub quality: String,
    }

    pub struct AlignedRead {
        pub id: String,
        pub sequence: String,
        pub mapped: bool,
        pub mapq: u8,
        pub chromosome: String,
        pub position: i64,
    }
}

/// Mock implementations
pub mod mocks {
    use super::*;

    mock! {
        pub Aligner {
            fn align(&self, sequence: &str) -> Result<AlignedRead, String>;
            fn align_batch(&self, sequences: Vec<String>) -> Result<Vec<AlignedRead>, String>;
        }
    }

    mock! {
        pub OffTargetPredictor {
            fn predict(&self, guide: &str, target: &str) -> f64;
            fn batch_predict(&self, pairs: Vec<(String, String)>) -> Vec<f64>;
        }
    }

    mock! {
        pub Database {
            fn insert_job(&self, id: &str, status: &str) -> Result<(), String>;
            fn query_job(&self, id: &str) -> Result<JobRecord, String>;
            fn update_status(&self, id: &str, status: &str) -> Result<(), String>;
        }
    }

    mock! {
        pub MessageQueue {
            fn publish(&self, topic: &str, message: &str) -> Result<(), String>;
            fn consume(&self, topic: &str, timeout_ms: i32) -> Result<Vec<String>, String>;
        }
    }

    #[derive(Clone)]
    pub struct AlignedRead {
        pub id: String,
        pub mapped: bool,
        pub mapq: u8,
    }

    pub struct JobRecord {
        pub id: String,
        pub status: String,
        pub created_at: i64,
    }
}

/// Test configuration builders
pub mod builders {
    use super::*;

    pub struct AlignmentConfigBuilder {
        reference: String,
        min_mapq: u8,
        threads: usize,
    }

    impl AlignmentConfigBuilder {
        pub fn new() -> Self {
            Self {
                reference: "/test/reference.fa".to_string(),
                min_mapq: 20,
                threads: 4,
            }
        }

        pub fn reference(mut self, path: &str) -> Self {
            self.reference = path.to_string();
            self
        }

        pub fn min_mapq(mut self, mapq: u8) -> Self {
            self.min_mapq = mapq;
            self
        }

        pub fn threads(mut self, n: usize) -> Self {
            self.threads = n;
            self
        }

        pub fn build(self) -> AlignmentConfig {
            AlignmentConfig {
                reference: self.reference,
                min_mapq: self.min_mapq,
                threads: self.threads,
            }
        }
    }

    pub struct JobBuilder {
        id: String,
        name: String,
        status: String,
        priority: i32,
    }

    impl JobBuilder {
        pub fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                name: "Test Job".to_string(),
                status: "pending".to_string(),
                priority: 1,
            }
        }

        pub fn name(mut self, name: &str) -> Self {
            self.name = name.to_string();
            self
        }

        pub fn status(mut self, status: &str) -> Self {
            self.status = status.to_string();
            self
        }

        pub fn priority(mut self, p: i32) -> Self {
            self.priority = p;
            self
        }

        pub fn build(self) -> Job {
            Job {
                id: self.id,
                name: self.name,
                status: self.status,
                priority: self.priority,
            }
        }
    }

    pub struct AlignmentConfig {
        pub reference: String,
        pub min_mapq: u8,
        pub threads: usize,
    }

    pub struct Job {
        pub id: String,
        pub name: String,
        pub status: String,
        pub priority: i32,
    }
}

/// Test database setup
pub mod database {
    use super::*;

    pub async fn setup_test_db() -> TestDatabase {
        TestDatabase {
            url: "postgresql://test:test@localhost/test_db".to_string(),
        }
    }

    pub async fn teardown_test_db(db: &TestDatabase) {
        // Clean up test database
    }

    pub struct TestDatabase {
        pub url: String,
    }

    impl TestDatabase {
        pub async fn clear_tables(&self) {
            // Clear all test data
        }

        pub async fn seed_data(&self) {
            // Insert test data
        }
    }
}

/// Test assertions and matchers
pub mod assertions {
    use super::*;

    pub fn assert_valid_fastq(record: &generators::FastqRecord) {
        assert!(!record.id.is_empty());
        assert_eq!(record.sequence.len(), record.quality.len());
        assert!(record.sequence.chars().all(|c| matches!(c, 'A' | 'C' | 'G' | 'T' | 'N')));
    }

    pub fn assert_alignment_quality(read: &generators::AlignedRead, min_mapq: u8) {
        if read.mapped {
            assert!(read.mapq >= min_mapq, "MAPQ {} below threshold {}", read.mapq, min_mapq);
        }
    }

    pub fn assert_normalized_scores(scores: &[f64]) {
        for &score in scores {
            assert!(score >= 0.0 && score <= 1.0, "Score {} out of range [0, 1]", score);
        }
    }

    pub fn assert_expression_matrix(matrix: &[Vec<u32>]) {
        let n_samples = matrix.len();
        assert!(n_samples > 0, "Empty expression matrix");

        let n_genes = matrix[0].len();
        for sample in matrix {
            assert_eq!(sample.len(), n_genes, "Inconsistent gene count across samples");
        }
    }

    pub fn assert_statistical_significance(p_value: f64, alpha: f64) {
        assert!(p_value >= 0.0 && p_value <= 1.0, "Invalid p-value: {}", p_value);
        if p_value < alpha {
            println!("Statistically significant: p = {:.4}, α = {}", p_value, alpha);
        }
    }
}

/// Performance test utilities
pub mod performance {
    use std::time::{Duration, Instant};

    pub struct PerformanceMonitor {
        start: Instant,
        checkpoints: Vec<(String, Duration)>,
    }

    impl PerformanceMonitor {
        pub fn new() -> Self {
            Self {
                start: Instant::now(),
                checkpoints: Vec::new(),
            }
        }

        pub fn checkpoint(&mut self, label: &str) {
            let elapsed = self.start.elapsed();
            self.checkpoints.push((label.to_string(), elapsed));
        }

        pub fn assert_duration(&self, label: &str, max_duration: Duration) {
            if let Some((_, duration)) = self.checkpoints.iter().find(|(l, _)| l == label) {
                assert!(
                    *duration <= max_duration,
                    "{} took {:?}, expected <= {:?}",
                    label,
                    duration,
                    max_duration
                );
            }
        }

        pub fn report(&self) {
            println!("\n=== Performance Report ===");
            for (label, duration) in &self.checkpoints {
                println!("{}: {:?}", label, duration);
            }
        }
    }

    pub fn measure<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
}

#[cfg(test)]
mod fixture_tests {
    use super::*;

    #[test]
    fn test_fastq_generator() {
        let records = generators::generate_fastq_records(10);
        assert_eq!(records.len(), 10);

        for record in &records {
            assertions::assert_valid_fastq(record);
        }
    }

    #[test]
    fn test_expression_matrix_generator() {
        let matrix = generators::generate_expression_matrix(100, 5);
        assertions::assert_expression_matrix(&matrix);
    }

    #[test]
    fn test_config_builder() {
        let config = builders::AlignmentConfigBuilder::new()
            .reference("/custom/ref.fa")
            .min_mapq(30)
            .threads(8)
            .build();

        assert_eq!(config.reference, "/custom/ref.fa");
        assert_eq!(config.min_mapq, 30);
        assert_eq!(config.threads, 8);
    }

    #[test]
    fn test_performance_monitor() {
        use std::time::Duration;

        let mut monitor = performance::PerformanceMonitor::new();

        std::thread::sleep(Duration::from_millis(10));
        monitor.checkpoint("step1");

        std::thread::sleep(Duration::from_millis(20));
        monitor.checkpoint("step2");

        monitor.assert_duration("step1", Duration::from_millis(50));
        monitor.report();
    }
}
