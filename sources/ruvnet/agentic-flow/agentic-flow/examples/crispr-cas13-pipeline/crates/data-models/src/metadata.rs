//! Experiment and pipeline metadata models

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents a complete experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Unique experiment identifier
    pub id: Uuid,
    /// Experiment name
    pub name: String,
    /// Description
    pub description: String,
    /// Principal investigator
    pub pi: String,
    /// Lab or institution
    pub institution: String,
    /// Experiment type
    pub experiment_type: ExperimentType,
    /// Creation date
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated date
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Experiment status
    pub status: ExperimentStatus,
    /// Associated samples
    pub sample_ids: Vec<Uuid>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Experiment {
    /// Create a new experiment
    pub fn new(name: String, description: String, pi: String, institution: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Uuid::new_v4(),
            name,
            description,
            pi,
            institution,
            experiment_type: ExperimentType::CrisprCas13,
            created_at: now,
            updated_at: now,
            status: ExperimentStatus::Planning,
            sample_ids: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a sample to the experiment
    pub fn add_sample(&mut self, sample_id: Uuid) {
        self.sample_ids.push(sample_id);
        self.updated_at = chrono::Utc::now();
    }

    /// Update experiment status
    pub fn update_status(&mut self, status: ExperimentStatus) {
        self.status = status;
        self.updated_at = chrono::Utc::now();
    }
}

/// Type of experiment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperimentType {
    CrisprCas13,
    RnaSeq,
    WholeGenomeSequencing,
    Immunoprofiling,
    Other(String),
}

/// Experiment status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperimentStatus {
    Planning,
    InProgress,
    Sequencing,
    Analyzing,
    Completed,
    Failed,
    Archived,
}

/// Pipeline execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRun {
    /// Unique run identifier
    pub id: Uuid,
    /// Associated experiment ID
    pub experiment_id: Uuid,
    /// Pipeline version
    pub pipeline_version: String,
    /// Start time
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// End time
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Run status
    pub status: PipelineStatus,
    /// Configuration used
    pub config: PipelineConfig,
    /// Resource usage
    pub resources: ResourceUsage,
    /// Error message if failed
    pub error: Option<String>,
}

impl PipelineRun {
    /// Create a new pipeline run
    pub fn new(experiment_id: Uuid, pipeline_version: String, config: PipelineConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            experiment_id,
            pipeline_version,
            started_at: chrono::Utc::now(),
            completed_at: None,
            status: PipelineStatus::Queued,
            config,
            resources: ResourceUsage::default(),
            error: None,
        }
    }

    /// Mark run as completed
    pub fn complete(&mut self) {
        self.status = PipelineStatus::Completed;
        self.completed_at = Some(chrono::Utc::now());
    }

    /// Mark run as failed
    pub fn fail(&mut self, error: String) {
        self.status = PipelineStatus::Failed;
        self.completed_at = Some(chrono::Utc::now());
        self.error = Some(error);
    }

    /// Get duration in seconds
    pub fn duration_seconds(&self) -> Option<f64> {
        self.completed_at
            .map(|end| (end - self.started_at).num_milliseconds() as f64 / 1000.0)
    }
}

/// Pipeline execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PipelineStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Reference genome path
    pub reference_genome: String,
    /// Annotation file path
    pub annotation_file: String,
    /// Maximum mismatches for off-target prediction
    pub max_mismatches: u8,
    /// Off-target score threshold
    pub offtarget_threshold: f64,
    /// Minimum read quality
    pub min_read_quality: u8,
    /// Differential expression padj threshold
    pub de_padj_threshold: f64,
    /// Log2 fold change threshold
    pub de_lfc_threshold: f64,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Additional parameters
    pub params: HashMap<String, String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            reference_genome: String::new(),
            annotation_file: String::new(),
            max_mismatches: 3,
            offtarget_threshold: 0.5,
            min_read_quality: 20,
            de_padj_threshold: 0.05,
            de_lfc_threshold: 1.0,
            num_workers: 4,
            params: HashMap::new(),
        }
    }
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// CPU time in seconds
    pub cpu_time_seconds: f64,
    /// Wall time in seconds
    pub wall_time_seconds: f64,
    /// Number of CPU cores used
    pub cpu_cores: usize,
    /// Disk I/O in bytes
    pub disk_io_bytes: u64,
}

/// Sample information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Unique sample identifier
    pub id: Uuid,
    /// Sample name
    pub name: String,
    /// Associated experiment ID
    pub experiment_id: Uuid,
    /// Sample type
    pub sample_type: SampleType,
    /// Organism
    pub organism: String,
    /// Tissue type
    pub tissue: Option<String>,
    /// Cell type
    pub cell_type: Option<String>,
    /// Treatment
    pub treatment: Option<String>,
    /// Timepoint
    pub timepoint: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Sample {
    /// Create a new sample
    pub fn new(name: String, experiment_id: Uuid, organism: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            experiment_id,
            sample_type: SampleType::Unknown,
            organism,
            tissue: None,
            cell_type: None,
            treatment: None,
            timepoint: None,
            metadata: HashMap::new(),
        }
    }
}

/// Sample type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SampleType {
    Control,
    Treatment,
    Baseline,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_creation() {
        let exp = Experiment::new(
            "Test Experiment".to_string(),
            "Testing CRISPR-Cas13".to_string(),
            "Dr. Smith".to_string(),
            "Test University".to_string(),
        );
        assert_eq!(exp.status, ExperimentStatus::Planning);
        assert_eq!(exp.sample_ids.len(), 0);
    }

    #[test]
    fn test_pipeline_run_duration() {
        let config = PipelineConfig::default();
        let mut run = PipelineRun::new(Uuid::new_v4(), "1.0.0".to_string(), config);

        // Simulate completion after 1 second
        std::thread::sleep(std::time::Duration::from_millis(100));
        run.complete();

        let duration = run.duration_seconds();
        assert!(duration.is_some());
        assert!(duration.unwrap() >= 0.1);
    }

    #[test]
    fn test_sample_creation() {
        let sample = Sample::new(
            "Sample1".to_string(),
            Uuid::new_v4(),
            "Homo sapiens".to_string(),
        );
        assert_eq!(sample.sample_type, SampleType::Unknown);
    }
}
