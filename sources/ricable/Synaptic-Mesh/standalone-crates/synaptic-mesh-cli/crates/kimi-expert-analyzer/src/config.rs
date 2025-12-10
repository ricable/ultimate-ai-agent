//! Configuration management for expert analysis

use crate::expert::ExpertDomain;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use anyhow::Result;

/// Main configuration for the expert analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Input/Output configuration
    pub io_config: IoConfig,
    /// Model analysis parameters
    pub analysis_params: AnalysisParameters,
    /// Distillation configuration
    pub distillation_config: DistillationConfig,
    /// Validation configuration
    pub validation_config: ValidationConfig,
    /// Performance optimization settings
    pub performance_config: PerformanceConfig,
    /// Logging and monitoring
    pub logging_config: LoggingConfig,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            io_config: IoConfig::default(),
            analysis_params: AnalysisParameters::default(),
            distillation_config: DistillationConfig::default(),
            validation_config: ValidationConfig::default(),
            performance_config: PerformanceConfig::default(),
            logging_config: LoggingConfig::default(),
        }
    }
}

impl AnalysisConfig {
    /// Load configuration from file
    pub fn load_from_file(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = if path.extension().map(|s| s.to_str()) == Some(Some("yaml")) || 
                              path.extension().map(|s| s.to_str()) == Some(Some("yml")) {
            serde_yaml::from_str(&content)?
        } else {
            serde_json::from_str(&content)?
        };
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file(&self, path: &PathBuf) -> Result<()> {
        let content = if path.extension().map(|s| s.to_str()) == Some(Some("yaml")) || 
                         path.extension().map(|s| s.to_str()) == Some(Some("yml")) {
            serde_yaml::to_string(self)?
        } else {
            serde_json::to_string_pretty(self)?
        };
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.io_config.validate()?;
        self.analysis_params.validate()?;
        self.distillation_config.validate()?;
        self.validation_config.validate()?;
        self.performance_config.validate()?;
        self.logging_config.validate()?;
        Ok(())
    }

    /// Get domain-specific configuration
    pub fn get_domain_config(&self, domain: &ExpertDomain) -> DomainSpecificConfig {
        self.analysis_params.domain_configs.get(domain)
            .cloned()
            .unwrap_or_else(|| DomainSpecificConfig::default_for_domain(domain))
    }
}

/// Input/Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoConfig {
    /// Input model path
    pub model_path: PathBuf,
    /// Output directory for analysis results
    pub output_dir: PathBuf,
    /// Temporary directory for intermediate files
    pub temp_dir: Option<PathBuf>,
    /// Cache directory for preprocessed data
    pub cache_dir: Option<PathBuf>,
    /// Maximum file size for processing (bytes)
    pub max_file_size: u64,
    /// File format preferences
    pub file_formats: FileFormatConfig,
    /// Compression settings
    pub compression: CompressionConfig,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("./kimi-k2-model"),
            output_dir: PathBuf::from("./analysis_output"),
            temp_dir: None, // Will use system temp
            cache_dir: Some(PathBuf::from("./cache")),
            max_file_size: 10 * 1024 * 1024 * 1024, // 10GB
            file_formats: FileFormatConfig::default(),
            compression: CompressionConfig::default(),
        }
    }
}

impl IoConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.model_path.exists() {
            return Err(anyhow::anyhow!("Model path does not exist: {:?}", self.model_path));
        }
        
        // Create output directory if it doesn't exist
        if !self.output_dir.exists() {
            std::fs::create_dir_all(&self.output_dir)?;
        }
        
        // Validate cache directory
        if let Some(cache_dir) = &self.cache_dir {
            if !cache_dir.exists() {
                std::fs::create_dir_all(cache_dir)?;
            }
        }
        
        Ok(())
    }
}

/// File format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileFormatConfig {
    /// Preferred format for analysis results
    pub analysis_format: FileFormat,
    /// Preferred format for training data
    pub training_data_format: FileFormat,
    /// Preferred format for model weights
    pub weights_format: WeightsFormat,
    /// Enable binary formats for performance
    pub prefer_binary: bool,
}

impl Default for FileFormatConfig {
    fn default() -> Self {
        Self {
            analysis_format: FileFormat::Json,
            training_data_format: FileFormat::Json,
            weights_format: WeightsFormat::Safetensors,
            prefer_binary: true,
        }
    }
}

/// Supported file formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileFormat {
    Json,
    Yaml,
    Toml,
    Csv,
    Parquet,
    MessagePack,
}

/// Supported weight formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightsFormat {
    Safetensors,
    Pytorch,
    Numpy,
    Onnx,
    Custom,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression for large files
    pub enable_compression: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (0-9)
    pub level: u8,
    /// Minimum file size for compression (bytes)
    pub min_size_for_compression: u64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            algorithm: CompressionAlgorithm::Zstd,
            level: 6,
            min_size_for_compression: 1024 * 1024, // 1MB
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Brotli,
}

/// Analysis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisParameters {
    /// Minimum specialization threshold for experts
    pub min_specialization_threshold: f32,
    /// Maximum micro-expert size (parameters)
    pub max_micro_expert_size: usize,
    /// Minimum performance retention target
    pub min_performance_retention: f32,
    /// Analysis depth level
    pub analysis_depth: AnalysisDepth,
    /// Enable activation pattern analysis
    pub enable_activation_analysis: bool,
    /// Enable cross-layer relationship analysis
    pub enable_relationship_analysis: bool,
    /// Enable redundancy analysis
    pub enable_redundancy_analysis: bool,
    /// Domain-specific configurations
    pub domain_configs: HashMap<ExpertDomain, DomainSpecificConfig>,
    /// Statistical analysis settings
    pub statistical_config: StatisticalConfig,
    /// Expert clustering settings
    pub clustering_config: ClusteringConfig,
}

impl Default for AnalysisParameters {
    fn default() -> Self {
        let mut domain_configs = HashMap::new();
        for domain in ExpertDomain::all_domains() {
            domain_configs.insert(domain.clone(), DomainSpecificConfig::default_for_domain(&domain));
        }

        Self {
            min_specialization_threshold: 0.6,
            max_micro_expert_size: 100_000,
            min_performance_retention: 0.8,
            analysis_depth: AnalysisDepth::Medium,
            enable_activation_analysis: true,
            enable_relationship_analysis: true,
            enable_redundancy_analysis: true,
            domain_configs,
            statistical_config: StatisticalConfig::default(),
            clustering_config: ClusteringConfig::default(),
        }
    }
}

impl AnalysisParameters {
    pub fn validate(&self) -> Result<()> {
        if self.min_specialization_threshold < 0.0 || self.min_specialization_threshold > 1.0 {
            return Err(anyhow::anyhow!("Specialization threshold must be between 0.0 and 1.0"));
        }
        
        if self.max_micro_expert_size == 0 {
            return Err(anyhow::anyhow!("Max micro-expert size must be greater than 0"));
        }
        
        if self.min_performance_retention < 0.0 || self.min_performance_retention > 1.0 {
            return Err(anyhow::anyhow!("Performance retention must be between 0.0 and 1.0"));
        }
        
        // Validate domain configs
        for (domain, config) in &self.domain_configs {
            config.validate(domain)?;
        }
        
        self.statistical_config.validate()?;
        self.clustering_config.validate()?;
        
        Ok(())
    }
}

/// Analysis depth levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDepth {
    Shallow,
    Medium,
    Deep,
    Comprehensive,
}

/// Domain-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSpecificConfig {
    /// Target parameter count for this domain
    pub target_parameters: usize,
    /// Minimum accuracy threshold
    pub min_accuracy: f32,
    /// Maximum latency tolerance (ms)
    pub max_latency_ms: f32,
    /// Domain weight in overall scoring
    pub domain_weight: f32,
    /// Specific analysis features to enable
    pub analysis_features: DomainAnalysisFeatures,
    /// Custom benchmarks for this domain
    pub custom_benchmarks: Vec<String>,
    /// Optimization priorities
    pub optimization_priorities: OptimizationPriorities,
}

impl DomainSpecificConfig {
    pub fn default_for_domain(domain: &ExpertDomain) -> Self {
        match domain {
            ExpertDomain::Reasoning => Self {
                target_parameters: 10_000,
                min_accuracy: 0.8,
                max_latency_ms: 100.0,
                domain_weight: 1.0,
                analysis_features: DomainAnalysisFeatures::reasoning_features(),
                custom_benchmarks: vec!["logical_reasoning".to_string(), "deductive_reasoning".to_string()],
                optimization_priorities: OptimizationPriorities::accuracy_first(),
            },
            ExpertDomain::Coding => Self {
                target_parameters: 50_000,
                min_accuracy: 0.7,
                max_latency_ms: 500.0,
                domain_weight: 1.2, // Higher weight due to complexity
                analysis_features: DomainAnalysisFeatures::coding_features(),
                custom_benchmarks: vec!["code_generation".to_string(), "debugging".to_string()],
                optimization_priorities: OptimizationPriorities::balanced(),
            },
            ExpertDomain::Language => Self {
                target_parameters: 25_000,
                min_accuracy: 0.85,
                max_latency_ms: 100.0,
                domain_weight: 1.0,
                analysis_features: DomainAnalysisFeatures::language_features(),
                custom_benchmarks: vec!["translation".to_string(), "summarization".to_string()],
                optimization_priorities: OptimizationPriorities::quality_first(),
            },
            ExpertDomain::ToolUse => Self {
                target_parameters: 15_000,
                min_accuracy: 0.9,
                max_latency_ms: 50.0,
                domain_weight: 1.1,
                analysis_features: DomainAnalysisFeatures::tool_features(),
                custom_benchmarks: vec!["api_calling".to_string(), "parameter_extraction".to_string()],
                optimization_priorities: OptimizationPriorities::speed_first(),
            },
            ExpertDomain::Mathematics => Self {
                target_parameters: 20_000,
                min_accuracy: 0.9,
                max_latency_ms: 100.0,
                domain_weight: 1.0,
                analysis_features: DomainAnalysisFeatures::math_features(),
                custom_benchmarks: vec!["arithmetic".to_string(), "algebra".to_string()],
                optimization_priorities: OptimizationPriorities::accuracy_first(),
            },
            ExpertDomain::Context => Self {
                target_parameters: 30_000,
                min_accuracy: 0.75,
                max_latency_ms: 1000.0,
                domain_weight: 0.9,
                analysis_features: DomainAnalysisFeatures::context_features(),
                custom_benchmarks: vec!["long_context".to_string(), "context_switching".to_string()],
                optimization_priorities: OptimizationPriorities::memory_efficient(),
            },
        }
    }

    pub fn validate(&self, domain: &ExpertDomain) -> Result<()> {
        if self.target_parameters == 0 {
            return Err(anyhow::anyhow!("Target parameters must be greater than 0 for domain {:?}", domain));
        }
        
        if self.min_accuracy < 0.0 || self.min_accuracy > 1.0 {
            return Err(anyhow::anyhow!("Min accuracy must be between 0.0 and 1.0 for domain {:?}", domain));
        }
        
        if self.max_latency_ms <= 0.0 {
            return Err(anyhow::anyhow!("Max latency must be positive for domain {:?}", domain));
        }
        
        self.analysis_features.validate()?;
        self.optimization_priorities.validate()?;
        
        Ok(())
    }
}

/// Domain-specific analysis features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAnalysisFeatures {
    pub enable_semantic_analysis: bool,
    pub enable_pattern_mining: bool,
    pub enable_complexity_analysis: bool,
    pub enable_performance_profiling: bool,
    pub enable_failure_analysis: bool,
    pub custom_features: Vec<String>,
}

impl DomainAnalysisFeatures {
    pub fn reasoning_features() -> Self {
        Self {
            enable_semantic_analysis: true,
            enable_pattern_mining: true,
            enable_complexity_analysis: true,
            enable_performance_profiling: true,
            enable_failure_analysis: true,
            custom_features: vec!["logical_consistency".to_string(), "inference_depth".to_string()],
        }
    }

    pub fn coding_features() -> Self {
        Self {
            enable_semantic_analysis: true,
            enable_pattern_mining: true,
            enable_complexity_analysis: true,
            enable_performance_profiling: true,
            enable_failure_analysis: true,
            custom_features: vec!["syntax_correctness".to_string(), "code_style".to_string()],
        }
    }

    pub fn language_features() -> Self {
        Self {
            enable_semantic_analysis: true,
            enable_pattern_mining: false,
            enable_complexity_analysis: false,
            enable_performance_profiling: true,
            enable_failure_analysis: true,
            custom_features: vec!["fluency_analysis".to_string(), "coherence_check".to_string()],
        }
    }

    pub fn tool_features() -> Self {
        Self {
            enable_semantic_analysis: false,
            enable_pattern_mining: true,
            enable_complexity_analysis: false,
            enable_performance_profiling: true,
            enable_failure_analysis: true,
            custom_features: vec!["parameter_accuracy".to_string(), "tool_selection".to_string()],
        }
    }

    pub fn math_features() -> Self {
        Self {
            enable_semantic_analysis: false,
            enable_pattern_mining: true,
            enable_complexity_analysis: true,
            enable_performance_profiling: true,
            enable_failure_analysis: true,
            custom_features: vec!["numerical_accuracy".to_string(), "formula_correctness".to_string()],
        }
    }

    pub fn context_features() -> Self {
        Self {
            enable_semantic_analysis: true,
            enable_pattern_mining: false,
            enable_complexity_analysis: true,
            enable_performance_profiling: true,
            enable_failure_analysis: false,
            custom_features: vec!["context_coherence".to_string(), "long_range_dependencies".to_string()],
        }
    }

    pub fn validate(&self) -> Result<()> {
        // All features are optional, so no specific validation needed
        Ok(())
    }
}

/// Optimization priorities for different domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPriorities {
    pub accuracy_weight: f32,
    pub speed_weight: f32,
    pub memory_weight: f32,
    pub quality_weight: f32,
}

impl OptimizationPriorities {
    pub fn accuracy_first() -> Self {
        Self {
            accuracy_weight: 0.5,
            speed_weight: 0.2,
            memory_weight: 0.1,
            quality_weight: 0.2,
        }
    }

    pub fn speed_first() -> Self {
        Self {
            accuracy_weight: 0.2,
            speed_weight: 0.5,
            memory_weight: 0.2,
            quality_weight: 0.1,
        }
    }

    pub fn memory_efficient() -> Self {
        Self {
            accuracy_weight: 0.2,
            speed_weight: 0.2,
            memory_weight: 0.5,
            quality_weight: 0.1,
        }
    }

    pub fn quality_first() -> Self {
        Self {
            accuracy_weight: 0.3,
            speed_weight: 0.1,
            memory_weight: 0.1,
            quality_weight: 0.5,
        }
    }

    pub fn balanced() -> Self {
        Self {
            accuracy_weight: 0.25,
            speed_weight: 0.25,
            memory_weight: 0.25,
            quality_weight: 0.25,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let total_weight = self.accuracy_weight + self.speed_weight + self.memory_weight + self.quality_weight;
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(anyhow::anyhow!("Optimization priority weights must sum to 1.0, got {}", total_weight));
        }
        Ok(())
    }
}

/// Statistical analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    pub confidence_level: f32,
    pub significance_threshold: f32,
    pub min_sample_size: usize,
    pub enable_hypothesis_testing: bool,
    pub enable_correlation_analysis: bool,
    pub outlier_detection_method: OutlierDetectionMethod,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            significance_threshold: 0.05,
            min_sample_size: 30,
            enable_hypothesis_testing: true,
            enable_correlation_analysis: true,
            outlier_detection_method: OutlierDetectionMethod::IQR,
        }
    }
}

impl StatisticalConfig {
    pub fn validate(&self) -> Result<()> {
        if self.confidence_level <= 0.0 || self.confidence_level >= 1.0 {
            return Err(anyhow::anyhow!("Confidence level must be between 0.0 and 1.0"));
        }
        
        if self.significance_threshold <= 0.0 || self.significance_threshold >= 1.0 {
            return Err(anyhow::anyhow!("Significance threshold must be between 0.0 and 1.0"));
        }
        
        if self.min_sample_size == 0 {
            return Err(anyhow::anyhow!("Minimum sample size must be greater than 0"));
        }
        
        Ok(())
    }
}

/// Outlier detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierDetectionMethod {
    IQR,
    ZScore,
    ModifiedZScore,
    IsolationForest,
    LOF, // Local Outlier Factor
}

/// Clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    pub algorithm: ClusteringAlgorithm,
    pub num_clusters: Option<usize>, // None for auto-detection
    pub distance_metric: DistanceMetric,
    pub enable_dimensionality_reduction: bool,
    pub reduction_method: DimensionalityReductionMethod,
    pub target_dimensions: usize,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::KMeans,
            num_clusters: None, // Auto-detect
            distance_metric: DistanceMetric::Euclidean,
            enable_dimensionality_reduction: true,
            reduction_method: DimensionalityReductionMethod::PCA,
            target_dimensions: 50,
        }
    }
}

impl ClusteringConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(num_clusters) = self.num_clusters {
            if num_clusters == 0 {
                return Err(anyhow::anyhow!("Number of clusters must be greater than 0"));
            }
        }
        
        if self.target_dimensions == 0 {
            return Err(anyhow::anyhow!("Target dimensions must be greater than 0"));
        }
        
        Ok(())
    }
}

/// Clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    GaussianMixture,
}

/// Distance metrics for clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Hamming,
    Jaccard,
}

/// Dimensionality reduction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionalityReductionMethod {
    PCA,
    TSNE,
    UMAP,
    ICA,
    FactorAnalysis,
}

/// Distillation configuration (re-exported from distillation module)
pub use crate::distillation::DistillationConfig;

/// Validation configuration (re-exported from validation module)
pub use crate::validation::ValidationConfig;

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Number of threads to use (None for auto-detection)
    pub num_threads: Option<usize>,
    /// Memory limit for operations (bytes)
    pub memory_limit: Option<u64>,
    /// Enable GPU acceleration if available
    pub enable_gpu_acceleration: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Performance optimization level
    pub optimization_level: OptimizationLevel,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_parallel_processing: true,
            num_threads: None, // Auto-detect
            memory_limit: None, // No limit
            enable_gpu_acceleration: false, // Conservative default
            batch_size: 32,
            enable_monitoring: true,
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}

impl PerformanceConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(num_threads) = self.num_threads {
            if num_threads == 0 {
                return Err(anyhow::anyhow!("Number of threads must be greater than 0"));
            }
        }
        
        if self.batch_size == 0 {
            return Err(anyhow::anyhow!("Batch size must be greater than 0"));
        }
        
        Ok(())
    }
}

/// Performance optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Conservative,
    Balanced,
    Aggressive,
    Maximum,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    /// Log to file
    pub log_to_file: bool,
    /// Log file path
    pub log_file_path: Option<PathBuf>,
    /// Maximum log file size (bytes)
    pub max_log_file_size: u64,
    /// Number of log files to keep
    pub log_file_rotation_count: usize,
    /// Enable structured logging (JSON)
    pub structured_logging: bool,
    /// Enable performance logging
    pub enable_performance_logging: bool,
    /// Enable debug information
    pub enable_debug_info: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            log_to_file: true,
            log_file_path: Some(PathBuf::from("./logs/expert_analyzer.log")),
            max_log_file_size: 100 * 1024 * 1024, // 100MB
            log_file_rotation_count: 5,
            structured_logging: false,
            enable_performance_logging: true,
            enable_debug_info: false,
        }
    }
}

impl LoggingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_log_file_size == 0 {
            return Err(anyhow::anyhow!("Max log file size must be greater than 0"));
        }
        
        if self.log_file_rotation_count == 0 {
            return Err(anyhow::anyhow!("Log file rotation count must be greater than 0"));
        }
        
        // Create log directory if logging to file
        if self.log_to_file {
            if let Some(log_path) = &self.log_file_path {
                if let Some(parent) = log_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
            }
        }
        
        Ok(())
    }
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Configuration builder for easy configuration creation
#[derive(Debug, Default)]
pub struct AnalysisConfigBuilder {
    config: AnalysisConfig,
}

impl AnalysisConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: AnalysisConfig::default(),
        }
    }

    pub fn model_path(mut self, path: PathBuf) -> Self {
        self.config.io_config.model_path = path;
        self
    }

    pub fn output_dir(mut self, path: PathBuf) -> Self {
        self.config.io_config.output_dir = path;
        self
    }

    pub fn analysis_depth(mut self, depth: AnalysisDepth) -> Self {
        self.config.analysis_params.analysis_depth = depth;
        self
    }

    pub fn min_specialization(mut self, threshold: f32) -> Self {
        self.config.analysis_params.min_specialization_threshold = threshold;
        self
    }

    pub fn max_expert_size(mut self, size: usize) -> Self {
        self.config.analysis_params.max_micro_expert_size = size;
        self
    }

    pub fn enable_gpu(mut self, enable: bool) -> Self {
        self.config.performance_config.enable_gpu_acceleration = enable;
        self
    }

    pub fn log_level(mut self, level: LogLevel) -> Self {
        self.config.logging_config.level = level;
        self
    }

    pub fn domain_config(mut self, domain: ExpertDomain, config: DomainSpecificConfig) -> Self {
        self.config.analysis_params.domain_configs.insert(domain, config);
        self
    }

    pub fn build(self) -> Result<AnalysisConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Configuration presets for common use cases
pub struct ConfigPresets;

impl ConfigPresets {
    /// Fast analysis configuration (reduced depth, optimized for speed)
    pub fn fast_analysis() -> AnalysisConfig {
        AnalysisConfigBuilder::new()
            .analysis_depth(AnalysisDepth::Shallow)
            .min_specialization(0.7)
            .max_expert_size(50_000)
            .log_level(LogLevel::Warn)
            .build()
            .unwrap_or_default()
    }

    /// Comprehensive analysis configuration (maximum depth and features)
    pub fn comprehensive_analysis() -> AnalysisConfig {
        let mut config = AnalysisConfig::default();
        config.analysis_params.analysis_depth = AnalysisDepth::Comprehensive;
        config.analysis_params.enable_activation_analysis = true;
        config.analysis_params.enable_relationship_analysis = true;
        config.analysis_params.enable_redundancy_analysis = true;
        config.logging_config.level = LogLevel::Debug;
        config.logging_config.enable_debug_info = true;
        config
    }

    /// Memory-optimized configuration (minimized memory usage)
    pub fn memory_optimized() -> AnalysisConfig {
        let mut config = AnalysisConfig::default();
        config.performance_config.memory_limit = Some(4 * 1024 * 1024 * 1024); // 4GB
        config.performance_config.batch_size = 16;
        config.io_config.compression.enable_compression = true;
        config.io_config.compression.level = 9; // Maximum compression
        config
    }

    /// GPU-accelerated configuration
    pub fn gpu_accelerated() -> AnalysisConfig {
        AnalysisConfigBuilder::new()
            .enable_gpu(true)
            .build()
            .unwrap_or_default()
    }

    /// Development configuration (debugging enabled, verbose logging)
    pub fn development() -> AnalysisConfig {
        let mut config = AnalysisConfig::default();
        config.logging_config.level = LogLevel::Debug;
        config.logging_config.enable_debug_info = true;
        config.logging_config.structured_logging = true;
        config.performance_config.enable_monitoring = true;
        config.analysis_params.analysis_depth = AnalysisDepth::Medium;
        config
    }

    /// Production configuration (optimized for production use)
    pub fn production() -> AnalysisConfig {
        let mut config = AnalysisConfig::default();
        config.logging_config.level = LogLevel::Info;
        config.logging_config.enable_debug_info = false;
        config.performance_config.optimization_level = OptimizationLevel::Aggressive;
        config.performance_config.enable_parallel_processing = true;
        config.io_config.compression.enable_compression = true;
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AnalysisConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = AnalysisConfigBuilder::new()
            .model_path(PathBuf::from("/tmp/test"))
            .analysis_depth(AnalysisDepth::Deep)
            .min_specialization(0.8);
        
        // Note: This will fail validation due to non-existent path, which is expected
        assert!(config.build().is_err());
    }

    #[test]
    fn test_domain_config_validation() {
        let config = DomainSpecificConfig::default_for_domain(&ExpertDomain::Reasoning);
        assert!(config.validate(&ExpertDomain::Reasoning).is_ok());
    }

    #[test]
    fn test_optimization_priorities() {
        let priorities = OptimizationPriorities::balanced();
        assert!(priorities.validate().is_ok());
        
        let total = priorities.accuracy_weight + priorities.speed_weight + 
                   priorities.memory_weight + priorities.quality_weight;
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_config_presets() {
        let fast_config = ConfigPresets::fast_analysis();
        assert!(fast_config.validate().is_ok());
        
        let comprehensive_config = ConfigPresets::comprehensive_analysis();
        assert!(comprehensive_config.validate().is_ok());
    }
}