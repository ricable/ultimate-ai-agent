//! Configuration management for neural network training system

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main configuration for the neural training system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub data: DataConfig,
    pub models: ModelsConfig,
    pub training: TrainingParameters,
    pub swarm: SwarmConfig,
    pub evaluation: EvaluationConfig,
    pub output: OutputConfig,
}

/// Data-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub input_file: String,
    pub train_test_split: f32,
    pub validation_split: f32,
    pub normalize_features: bool,
    pub target_column: String,
    pub feature_selection: FeatureSelectionConfig,
    pub preprocessing: PreprocessingConfig,
}

/// Feature selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    pub enabled: bool,
    pub method: String, // "variance", "correlation", "mutual_info"
    pub max_features: Option<usize>,
    pub threshold: f32,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub remove_outliers: bool,
    pub outlier_threshold: f32,
    pub handle_missing: String, // "mean", "median", "drop"
    pub feature_scaling: String, // "standard", "minmax", "robust"
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub architectures: Vec<ArchitectureConfig>,
    pub default_activation: String,
    pub use_bias: bool,
    pub connection_rate: f32,
    pub weight_initialization: WeightInitConfig,
}

/// Architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub name: String,
    pub layer_sizes: Vec<usize>,
    pub activations: Vec<String>,
    pub dropout_rates: Option<Vec<f32>>,
    pub enabled: bool,
}

/// Weight initialization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightInitConfig {
    pub method: String, // "xavier", "he", "random", "zeros"
    pub seed: Option<u64>,
    pub scale: f32,
}

/// Training parameters configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParameters {
    pub algorithm: String, // "backprop", "rprop", "quickprop"
    pub learning_rate: f32,
    pub momentum: f32,
    pub max_epochs: usize,
    pub target_error: f32,
    pub batch_size: Option<usize>,
    pub early_stopping: EarlyStoppingConfig,
    pub regularization: RegularizationConfig,
    pub optimization: OptimizationConfig,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub enabled: bool,
    pub patience: usize,
    pub min_delta: f32,
    pub monitor: String, // "loss", "accuracy", "val_loss"
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    pub l1_lambda: f32,
    pub l2_lambda: f32,
    pub dropout_rate: f32,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub adaptive_learning_rate: bool,
    pub learning_rate_decay: f32,
    pub decay_schedule: String, // "exponential", "step", "cosine"
    pub warmup_epochs: usize,
}

/// Swarm coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub enabled: bool,
    pub max_parallel_agents: usize,
    pub agent_coordination: AgentCoordinationConfig,
    pub load_balancing: LoadBalancingConfig,
    pub fault_tolerance: FaultToleranceConfig,
}

/// Agent coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCoordinationConfig {
    pub communication_protocol: String,
    pub task_distribution: String, // "round_robin", "capability_based", "load_aware"
    pub synchronization_points: Vec<String>,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    pub strategy: String, // "static", "dynamic", "adaptive"
    pub rebalance_interval: u64, // seconds
    pub utilization_threshold: f32,
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    pub retry_attempts: usize,
    pub timeout_seconds: u64,
    pub checkpoint_interval: usize,
    pub auto_recovery: bool,
}

/// Evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub metrics: Vec<String>,
    pub cross_validation: CrossValidationConfig,
    pub statistical_tests: StatisticalTestsConfig,
    pub visualization: VisualizationConfig,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    pub enabled: bool,
    pub k_folds: usize,
    pub stratified: bool,
    pub shuffle: bool,
    pub random_seed: Option<u64>,
}

/// Statistical tests configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestsConfig {
    pub significance_level: f32,
    pub tests: Vec<String>, // "t_test", "wilcoxon", "friedman"
    pub multiple_comparisons: String, // "bonferroni", "holm", "fdr"
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub enabled: bool,
    pub plot_types: Vec<String>,
    pub save_plots: bool,
    pub output_format: String, // "png", "svg", "pdf"
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub results_directory: String,
    pub save_models: bool,
    pub save_training_history: bool,
    pub save_predictions: bool,
    pub report_format: String, // "json", "yaml", "html"
    pub logging: LoggingConfig,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String, // "debug", "info", "warn", "error"
    pub file_output: bool,
    pub console_output: bool,
    pub log_file: String,
    pub max_file_size: String, // "10MB", "100MB"
    pub rotation: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            data: DataConfig {
                input_file: "data/pm/fanndata.csv".to_string(),
                train_test_split: 0.8,
                validation_split: 0.2,
                normalize_features: true,
                target_column: "cell_availability".to_string(),
                feature_selection: FeatureSelectionConfig {
                    enabled: false,
                    method: "correlation".to_string(),
                    max_features: None,
                    threshold: 0.1,
                },
                preprocessing: PreprocessingConfig {
                    remove_outliers: true,
                    outlier_threshold: 3.0,
                    handle_missing: "mean".to_string(),
                    feature_scaling: "standard".to_string(),
                },
            },
            models: ModelsConfig {
                architectures: vec![
                    ArchitectureConfig {
                        name: "shallow".to_string(),
                        layer_sizes: vec![22, 32, 1],
                        activations: vec!["linear".to_string(), "relu".to_string(), "linear".to_string()],
                        dropout_rates: None,
                        enabled: true,
                    },
                    ArchitectureConfig {
                        name: "deep".to_string(),
                        layer_sizes: vec![22, 64, 32, 16, 8, 1],
                        activations: vec![
                            "linear".to_string(), 
                            "relu".to_string(), 
                            "relu".to_string(), 
                            "relu".to_string(), 
                            "sigmoid".to_string(), 
                            "linear".to_string()
                        ],
                        dropout_rates: None,
                        enabled: true,
                    },
                ],
                default_activation: "relu".to_string(),
                use_bias: true,
                connection_rate: 1.0,
                weight_initialization: WeightInitConfig {
                    method: "xavier".to_string(),
                    seed: Some(42),
                    scale: 1.0,
                },
            },
            training: TrainingParameters {
                algorithm: "backprop".to_string(),
                learning_rate: 0.01,
                momentum: 0.9,
                max_epochs: 1000,
                target_error: 0.001,
                batch_size: None,
                early_stopping: EarlyStoppingConfig {
                    enabled: true,
                    patience: 50,
                    min_delta: 0.0001,
                    monitor: "val_loss".to_string(),
                },
                regularization: RegularizationConfig {
                    l1_lambda: 0.0,
                    l2_lambda: 0.0001,
                    dropout_rate: 0.0,
                },
                optimization: OptimizationConfig {
                    adaptive_learning_rate: false,
                    learning_rate_decay: 0.95,
                    decay_schedule: "exponential".to_string(),
                    warmup_epochs: 0,
                },
            },
            swarm: SwarmConfig {
                enabled: true,
                max_parallel_agents: 5,
                agent_coordination: AgentCoordinationConfig {
                    communication_protocol: "async".to_string(),
                    task_distribution: "capability_based".to_string(),
                    synchronization_points: vec![
                        "data_ready".to_string(),
                        "models_created".to_string(),
                        "training_complete".to_string(),
                    ],
                },
                load_balancing: LoadBalancingConfig {
                    strategy: "dynamic".to_string(),
                    rebalance_interval: 30,
                    utilization_threshold: 0.8,
                },
                fault_tolerance: FaultToleranceConfig {
                    retry_attempts: 3,
                    timeout_seconds: 300,
                    checkpoint_interval: 100,
                    auto_recovery: true,
                },
            },
            evaluation: EvaluationConfig {
                metrics: vec![
                    "mse".to_string(),
                    "mae".to_string(),
                    "r_squared".to_string(),
                    "rmse".to_string(),
                ],
                cross_validation: CrossValidationConfig {
                    enabled: false,
                    k_folds: 5,
                    stratified: false,
                    shuffle: true,
                    random_seed: Some(42),
                },
                statistical_tests: StatisticalTestsConfig {
                    significance_level: 0.05,
                    tests: vec!["t_test".to_string()],
                    multiple_comparisons: "bonferroni".to_string(),
                },
                visualization: VisualizationConfig {
                    enabled: false,
                    plot_types: vec!["training_curves".to_string(), "predictions".to_string()],
                    save_plots: true,
                    output_format: "png".to_string(),
                },
            },
            output: OutputConfig {
                results_directory: "results".to_string(),
                save_models: true,
                save_training_history: true,
                save_predictions: true,
                report_format: "json".to_string(),
                logging: LoggingConfig {
                    level: "info".to_string(),
                    file_output: true,
                    console_output: true,
                    log_file: "training.log".to_string(),
                    max_file_size: "10MB".to_string(),
                    rotation: true,
                },
            },
        }
    }
}

impl TrainingConfig {
    /// Load configuration from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read configuration file")?;
        
        // Try different formats
        if let Ok(config) = serde_json::from_str::<Self>(&content) {
            Ok(config)
        } else if let Ok(config) = serde_yaml::from_str::<Self>(&content) {
            Ok(config)
        } else {
            Err(anyhow::anyhow!("Unsupported configuration format"))
        }
    }
    
    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        
        let content = if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yaml::to_string(self)
                .context("Failed to serialize configuration to YAML")?
        } else {
            serde_json::to_string_pretty(self)
                .context("Failed to serialize configuration to JSON")?
        };
        
        std::fs::write(path, content)
            .context("Failed to write configuration file")?;
        
        Ok(())
    }
    
    /// Create configuration with custom data file
    pub fn with_data_file<P: AsRef<Path>>(mut self, data_file: P) -> Self {
        self.data.input_file = data_file.as_ref().to_string_lossy().to_string();
        self
    }
    
    /// Enable/disable swarm coordination
    pub fn with_swarm_enabled(mut self, enabled: bool) -> Self {
        self.swarm.enabled = enabled;
        self
    }
    
    /// Set maximum number of parallel agents
    pub fn with_max_agents(mut self, max_agents: usize) -> Self {
        self.swarm.max_parallel_agents = max_agents;
        self
    }
    
    /// Set training parameters
    pub fn with_training_params(
        mut self,
        learning_rate: f32,
        momentum: f32,
        max_epochs: usize,
    ) -> Self {
        self.training.learning_rate = learning_rate;
        self.training.momentum = momentum;
        self.training.max_epochs = max_epochs;
        self
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate data configuration
        if !(0.0..=1.0).contains(&self.data.train_test_split) {
            return Err(anyhow::anyhow!("train_test_split must be between 0 and 1"));
        }
        
        if !(0.0..=1.0).contains(&self.data.validation_split) {
            return Err(anyhow::anyhow!("validation_split must be between 0 and 1"));
        }
        
        // Validate training parameters
        if self.training.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!("learning_rate must be positive"));
        }
        
        if !(0.0..=1.0).contains(&self.training.momentum) {
            return Err(anyhow::anyhow!("momentum must be between 0 and 1"));
        }
        
        if self.training.max_epochs == 0 {
            return Err(anyhow::anyhow!("max_epochs must be greater than 0"));
        }
        
        // Validate model architectures
        for arch in &self.models.architectures {
            if arch.layer_sizes.len() < 2 {
                return Err(anyhow::anyhow!("Architecture '{}' must have at least 2 layers", arch.name));
            }
            
            if arch.activations.len() != arch.layer_sizes.len() {
                return Err(anyhow::anyhow!(
                    "Architecture '{}' must have same number of activations as layers", 
                    arch.name
                ));
            }
        }
        
        // Validate swarm configuration
        if self.swarm.enabled && self.swarm.max_parallel_agents == 0 {
            return Err(anyhow::anyhow!("max_parallel_agents must be greater than 0 when swarm is enabled"));
        }
        
        Ok(())
    }
    
    /// Get enabled architectures
    pub fn enabled_architectures(&self) -> Vec<&ArchitectureConfig> {
        self.models.architectures.iter()
            .filter(|arch| arch.enabled)
            .collect()
    }
    
    /// Initialize logging based on configuration
    pub fn init_logging(&self) -> Result<()> {
        use env_logger::Builder;
        use log::LevelFilter;
        use std::io::Write;
        
        let mut builder = Builder::new();
        
        // Set log level
        let level = match self.output.logging.level.as_str() {
            "debug" => LevelFilter::Debug,
            "info" => LevelFilter::Info,
            "warn" => LevelFilter::Warn,
            "error" => LevelFilter::Error,
            _ => LevelFilter::Info,
        };
        builder.filter_level(level);
        
        // Set format
        builder.format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                record.level(),
                record.args()
            )
        });
        
        // Initialize
        builder.init();
        
        log::info!("Logging initialized with level: {}", self.output.logging.level);
        Ok(())
    }
}