//! Configuration management for the training system

use crate::error::{TrainingError, TrainingResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// System-wide configuration for the training environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Training environment settings
    pub training: TrainingEnvironmentConfig,
    /// Data processing settings
    pub data: DataConfig,
    /// Performance and optimization settings
    pub performance: PerformanceConfig,
    /// Logging and monitoring settings
    pub logging: LoggingConfig,
    /// Export and reporting settings
    pub export: ExportConfig,
}

/// Training environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEnvironmentConfig {
    /// Default random seed for reproducibility
    pub default_random_seed: Option<u64>,
    /// Maximum number of epochs for any model
    pub max_epochs_limit: u32,
    /// Default validation split ratio
    pub default_validation_split: f32,
    /// Enable early stopping by default
    pub enable_early_stopping: bool,
    /// Early stopping patience
    pub early_stopping_patience: u32,
    /// Save model checkpoints
    pub save_checkpoints: bool,
    /// Checkpoint save interval (epochs)
    pub checkpoint_interval: usize,
    /// Working directory for temporary files
    pub work_dir: String,
}

/// Data processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Default train/test split ratio
    pub train_test_split: f32,
    /// Enable data normalization by default
    pub normalize_features: bool,
    /// Handle missing values strategy
    pub missing_value_strategy: MissingValueStrategy,
    /// Maximum number of records to load (None = unlimited)
    pub max_records: Option<usize>,
    /// Data validation settings
    pub validation: DataValidationConfig,
    /// Feature engineering settings
    pub feature_engineering: FeatureEngineeringConfig,
}

/// Strategy for handling missing values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    /// Drop records with missing values
    Drop,
    /// Fill with mean value
    FillMean,
    /// Fill with median value
    FillMedian,
    /// Fill with specified value
    FillValue(f32),
    /// Forward fill (use previous value)
    ForwardFill,
}

/// Data validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationConfig {
    /// Enable data quality checks
    pub enable_quality_checks: bool,
    /// Maximum allowed missing values percentage
    pub max_missing_percentage: f32,
    /// Outlier detection threshold (Z-score)
    pub outlier_threshold: f32,
    /// Enable outlier removal
    pub remove_outliers: bool,
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Enable automatic feature scaling
    pub auto_scaling: bool,
    /// Scaling method
    pub scaling_method: ScalingMethod,
    /// Enable feature selection
    pub enable_feature_selection: bool,
    /// Feature selection method
    pub selection_method: FeatureSelectionMethod,
    /// Maximum number of features to select
    pub max_features: Option<usize>,
}

/// Feature scaling methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// Z-score normalization
    StandardScaling,
    /// Min-max scaling to [0, 1]
    MinMaxScaling,
    /// Robust scaling using quartiles
    RobustScaling,
    /// No scaling
    None,
}

/// Feature selection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    /// Correlation-based selection
    Correlation,
    /// Mutual information
    MutualInformation,
    /// Variance threshold
    VarianceThreshold,
    /// Recursive feature elimination
    RecursiveElimination,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable parallel training
    pub enable_parallel: bool,
    /// Number of CPU threads to use (None = auto-detect)
    pub num_threads: Option<usize>,
    /// Memory usage limit in MB
    pub memory_limit_mb: Option<usize>,
    /// Enable GPU acceleration if available
    pub enable_gpu: bool,
    /// Batch size for GPU training
    pub gpu_batch_size: usize,
    /// Cache settings
    pub cache: CacheConfig,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache directory
    pub cache_dir: String,
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
    /// Cache TTL in hours
    pub cache_ttl_hours: u32,
}

/// Logging and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Enable file logging
    pub log_to_file: bool,
    /// Log file path
    pub log_file: String,
    /// Enable performance metrics collection
    pub collect_metrics: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval: u32,
    /// Enable progress reporting
    pub enable_progress: bool,
}

/// Export and reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Default export format
    pub default_format: ExportFormat,
    /// Export directory
    pub export_dir: String,
    /// Include detailed metrics in exports
    pub include_detailed_metrics: bool,
    /// Generate visualizations
    pub generate_plots: bool,
    /// Plot format (png, svg, pdf)
    pub plot_format: String,
    /// Include model comparison report
    pub include_comparison: bool,
}

/// Export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Yaml,
    Csv,
    Excel,
    Html,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            training: TrainingEnvironmentConfig::default(),
            data: DataConfig::default(),
            performance: PerformanceConfig::default(),
            logging: LoggingConfig::default(),
            export: ExportConfig::default(),
        }
    }
}

impl Default for TrainingEnvironmentConfig {
    fn default() -> Self {
        Self {
            default_random_seed: Some(42),
            max_epochs_limit: 10000,
            default_validation_split: 0.2,
            enable_early_stopping: true,
            early_stopping_patience: 50,
            save_checkpoints: true,
            checkpoint_interval: 100,
            work_dir: "./training_work".to_string(),
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train_test_split: 0.8,
            normalize_features: true,
            missing_value_strategy: MissingValueStrategy::FillMean,
            max_records: None,
            validation: DataValidationConfig::default(),
            feature_engineering: FeatureEngineeringConfig::default(),
        }
    }
}

impl Default for DataValidationConfig {
    fn default() -> Self {
        Self {
            enable_quality_checks: true,
            max_missing_percentage: 10.0,
            outlier_threshold: 3.0,
            remove_outliers: false,
        }
    }
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            auto_scaling: true,
            scaling_method: ScalingMethod::StandardScaling,
            enable_feature_selection: false,
            selection_method: FeatureSelectionMethod::Correlation,
            max_features: None,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            num_threads: None,
            memory_limit_mb: None,
            enable_gpu: false,
            gpu_batch_size: 32,
            cache: CacheConfig::default(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_dir: "./cache".to_string(),
            max_cache_size_mb: 1024,
            cache_ttl_hours: 24,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            log_to_file: true,
            log_file: "./training.log".to_string(),
            collect_metrics: true,
            metrics_interval: 30,
            enable_progress: true,
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            default_format: ExportFormat::Json,
            export_dir: "./exports".to_string(),
            include_detailed_metrics: true,
            generate_plots: false,
            plot_format: "png".to_string(),
            include_comparison: true,
        }
    }
}

impl SystemConfig {
    /// Load configuration from file
    pub fn load<P: AsRef<Path>>(path: P) -> TrainingResult<Self> {
        let content = std::fs::read_to_string(path)?;
        
        // Try YAML first, then JSON
        let config = if let Ok(config) = serde_yaml::from_str::<SystemConfig>(&content) {
            config
        } else {
            serde_json::from_str::<SystemConfig>(&content)?
        };
        
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> TrainingResult<()> {
        let path = path.as_ref();
        
        let content = if path.extension().and_then(|s| s.to_str()) == Some("yaml") 
            || path.extension().and_then(|s| s.to_str()) == Some("yml") {
            serde_yaml::to_string(self)?
        } else {
            serde_json::to_string_pretty(self)?
        };
        
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration values
    pub fn validate(&self) -> TrainingResult<()> {
        // Validate training config
        if self.training.default_validation_split < 0.0 || self.training.default_validation_split > 1.0 {
            return Err(TrainingError::InvalidInput(
                "Validation split must be between 0.0 and 1.0".into()
            ));
        }
        
        if self.training.max_epochs_limit == 0 {
            return Err(TrainingError::InvalidInput(
                "Max epochs limit must be greater than 0".into()
            ));
        }
        
        // Validate data config
        if self.data.train_test_split < 0.0 || self.data.train_test_split > 1.0 {
            return Err(TrainingError::InvalidInput(
                "Train/test split must be between 0.0 and 1.0".into()
            ));
        }
        
        if self.data.validation.max_missing_percentage < 0.0 || self.data.validation.max_missing_percentage > 100.0 {
            return Err(TrainingError::InvalidInput(
                "Max missing percentage must be between 0.0 and 100.0".into()
            ));
        }
        
        // Validate performance config
        if let Some(threads) = self.performance.num_threads {
            if threads == 0 {
                return Err(TrainingError::InvalidInput(
                    "Number of threads must be greater than 0".into()
                ));
            }
        }
        
        if self.performance.gpu_batch_size == 0 {
            return Err(TrainingError::InvalidInput(
                "GPU batch size must be greater than 0".into()
            ));
        }
        
        Ok(())
    }
    
    /// Create directories specified in configuration
    pub fn create_directories(&self) -> TrainingResult<()> {
        std::fs::create_dir_all(&self.training.work_dir)?;
        std::fs::create_dir_all(&self.performance.cache.cache_dir)?;
        std::fs::create_dir_all(&self.export.export_dir)?;
        
        // Create log directory if logging to file
        if self.logging.log_to_file {
            if let Some(parent) = Path::new(&self.logging.log_file).parent() {
                std::fs::create_dir_all(parent)?;
            }
        }
        
        Ok(())
    }
    
    /// Get effective number of threads for parallel processing
    pub fn get_thread_count(&self) -> usize {
        self.performance.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }
    
    /// Check if GPU is available and enabled
    pub fn should_use_gpu(&self) -> bool {
        self.performance.enable_gpu && self.is_gpu_available()
    }
    
    /// Check if GPU is available (placeholder implementation)
    fn is_gpu_available(&self) -> bool {
        // This would need actual GPU detection logic
        false
    }
    
    /// Get log level as a proper log level
    pub fn get_log_level(&self) -> log::LevelFilter {
        match self.logging.level.to_lowercase().as_str() {
            "trace" => log::LevelFilter::Trace,
            "debug" => log::LevelFilter::Debug,
            "info" => log::LevelFilter::Info,
            "warn" => log::LevelFilter::Warn,
            "error" => log::LevelFilter::Error,
            _ => log::LevelFilter::Info,
        }
    }
    
    /// Initialize logging based on configuration
    pub fn init_logging(&self) -> TrainingResult<()> {
        let level = self.get_log_level();
        
        if self.logging.log_to_file {
            // Initialize file logging
            env_logger::Builder::from_default_env()
                .filter_level(level)
                .init();
        } else {
            // Initialize console logging
            env_logger::Builder::from_default_env()
                .filter_level(level)
                .init();
        }
        
        Ok(())
    }
    
    /// Get cache key for a given operation
    pub fn get_cache_key(&self, operation: &str, params: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        operation.hash(&mut hasher);
        params.hash(&mut hasher);
        
        format!("{}_{:x}", operation, hasher.finish())
    }
    
    /// Clean up old cache files
    pub fn cleanup_cache(&self) -> TrainingResult<()> {
        if !self.performance.cache.enable_caching {
            return Ok(());
        }
        
        let cache_dir = Path::new(&self.performance.cache.cache_dir);
        if !cache_dir.exists() {
            return Ok(());
        }
        
        let ttl_seconds = self.performance.cache.cache_ttl_hours as u64 * 3600;
        let cutoff_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() - ttl_seconds;
        
        for entry in std::fs::read_dir(cache_dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            
            if let Ok(modified) = metadata.modified() {
                if let Ok(modified_secs) = modified.duration_since(std::time::UNIX_EPOCH) {
                    if modified_secs.as_secs() < cutoff_time {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }
        
        Ok(())
    }
}

/// Configuration builder for easier setup
pub struct ConfigBuilder {
    config: SystemConfig,
}

impl ConfigBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: SystemConfig::default(),
        }
    }
    
    /// Set training configuration
    pub fn training(mut self, training: TrainingEnvironmentConfig) -> Self {
        self.config.training = training;
        self
    }
    
    /// Set data configuration
    pub fn data(mut self, data: DataConfig) -> Self {
        self.config.data = data;
        self
    }
    
    /// Set performance configuration
    pub fn performance(mut self, performance: PerformanceConfig) -> Self {
        self.config.performance = performance;
        self
    }
    
    /// Set logging configuration
    pub fn logging(mut self, logging: LoggingConfig) -> Self {
        self.config.logging = logging;
        self
    }
    
    /// Set export configuration
    pub fn export(mut self, export: ExportConfig) -> Self {
        self.config.export = export;
        self
    }
    
    /// Enable parallel processing
    pub fn enable_parallel(mut self, enable: bool) -> Self {
        self.config.performance.enable_parallel = enable;
        self
    }
    
    /// Set number of threads
    pub fn threads(mut self, threads: usize) -> Self {
        self.config.performance.num_threads = Some(threads);
        self
    }
    
    /// Enable GPU acceleration
    pub fn enable_gpu(mut self, enable: bool) -> Self {
        self.config.performance.enable_gpu = enable;
        self
    }
    
    /// Set log level
    pub fn log_level(mut self, level: &str) -> Self {
        self.config.logging.level = level.to_string();
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> TrainingResult<SystemConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_default_config() {
        let config = SystemConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.training.default_validation_split, 0.2);
        assert!(config.performance.enable_parallel);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = SystemConfig::default();
        
        // Test invalid validation split
        config.training.default_validation_split = 1.5;
        assert!(config.validate().is_err());
        
        // Reset and test invalid train/test split
        config.training.default_validation_split = 0.2;
        config.data.train_test_split = -0.1;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .enable_parallel(true)
            .threads(4)
            .log_level("debug")
            .build()
            .unwrap();
        
        assert!(config.performance.enable_parallel);
        assert_eq!(config.performance.num_threads, Some(4));
        assert_eq!(config.logging.level, "debug");
    }
    
    #[test]
    fn test_config_save_load() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("test_config.yaml");
        
        let original_config = SystemConfig::default();
        original_config.save(&config_path).unwrap();
        
        let loaded_config = SystemConfig::load(&config_path).unwrap();
        assert_eq!(original_config.training.default_validation_split, 
                   loaded_config.training.default_validation_split);
    }
}