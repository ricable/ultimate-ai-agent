use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub classification: ClassificationConfig,
    pub performance: PerformanceConfig,
    pub features: FeatureConfig,
    pub mitigation: MitigationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: u32,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String, // "ensemble", "svm", "neural_network", "random_forest"
    pub ensemble_models: Vec<String>,
    pub hidden_layers: Vec<u32>,
    pub learning_rate: f64,
    pub max_epochs: u32,
    pub target_accuracy: f64, // Must be > 0.95
    pub validation_split: f64,
    pub early_stopping_patience: u32,
    pub regularization_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationConfig {
    pub interference_classes: Vec<String>,
    pub confidence_threshold: f64,
    pub min_samples_per_class: u32,
    pub feature_selection_method: String,
    pub class_weights: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub min_accuracy: f64, // Must be > 0.95
    pub min_precision: f64,
    pub min_recall: f64,
    pub min_f1_score: f64,
    pub evaluation_window_minutes: u32,
    pub performance_monitoring_interval_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub noise_floor_window_size: u32,
    pub spectral_features: bool,
    pub temporal_features: bool,
    pub statistical_features: bool,
    pub frequency_bins: u32,
    pub fft_size: u32,
    pub overlap_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationConfig {
    pub strategies: HashMap<String, MitigationStrategy>,
    pub effectiveness_threshold: f64,
    pub implementation_priority: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    pub name: String,
    pub description: String,
    pub effectiveness_score: f64,
    pub implementation_complexity: u32,
    pub resource_requirements: Vec<String>,
    pub estimated_improvement: f64,
}

impl Default for Config {
    fn default() -> Self {
        let mut interference_classes = Vec::new();
        interference_classes.push("EXTERNAL_JAMMER".to_string());
        interference_classes.push("PIM".to_string());
        interference_classes.push("ADJACENT_CHANNEL".to_string());
        interference_classes.push("THERMAL_NOISE".to_string());
        interference_classes.push("LEGITIMATE_TRAFFIC".to_string());
        
        let mut class_weights = HashMap::new();
        class_weights.insert("EXTERNAL_JAMMER".to_string(), 1.2);
        class_weights.insert("PIM".to_string(), 1.1);
        class_weights.insert("ADJACENT_CHANNEL".to_string(), 1.0);
        class_weights.insert("THERMAL_NOISE".to_string(), 0.9);
        class_weights.insert("LEGITIMATE_TRAFFIC".to_string(), 0.8);
        
        let mut mitigation_strategies = HashMap::new();
        mitigation_strategies.insert("FREQUENCY_HOPPING".to_string(), MitigationStrategy {
            name: "Frequency Hopping".to_string(),
            description: "Dynamic frequency allocation to avoid interference".to_string(),
            effectiveness_score: 0.85,
            implementation_complexity: 3,
            resource_requirements: vec!["Spectrum analyzer".to_string(), "Frequency coordinator".to_string()],
            estimated_improvement: 0.7,
        });
        mitigation_strategies.insert("POWER_CONTROL".to_string(), MitigationStrategy {
            name: "Power Control".to_string(),
            description: "Adjust transmit power to minimize interference".to_string(),
            effectiveness_score: 0.75,
            implementation_complexity: 2,
            resource_requirements: vec!["Power control unit".to_string()],
            estimated_improvement: 0.6,
        });
        mitigation_strategies.insert("BEAMFORMING".to_string(), MitigationStrategy {
            name: "Beamforming".to_string(),
            description: "Directional antenna patterns to reduce interference".to_string(),
            effectiveness_score: 0.90,
            implementation_complexity: 4,
            resource_requirements: vec!["Antenna array".to_string(), "Beamforming processor".to_string()],
            estimated_improvement: 0.8,
        });
        
        let mut implementation_priority = HashMap::new();
        implementation_priority.insert("FREQUENCY_HOPPING".to_string(), 1);
        implementation_priority.insert("POWER_CONTROL".to_string(), 2);
        implementation_priority.insert("BEAMFORMING".to_string(), 3);
        
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 50051,
                max_connections: 1000,
                timeout_seconds: 30,
            },
            model: ModelConfig {
                architecture: "ensemble".to_string(),
                ensemble_models: vec![
                    "random_forest".to_string(),
                    "svm".to_string(),
                    "neural_network".to_string(),
                ],
                hidden_layers: vec![128, 64, 32],
                learning_rate: 0.001,
                max_epochs: 200,
                target_accuracy: 0.95, // 95% accuracy requirement
                validation_split: 0.2,
                early_stopping_patience: 10,
                regularization_factor: 0.01,
            },
            classification: ClassificationConfig {
                interference_classes,
                confidence_threshold: 0.8,
                min_samples_per_class: 100,
                feature_selection_method: "recursive_feature_elimination".to_string(),
                class_weights,
            },
            performance: PerformanceConfig {
                min_accuracy: 0.95, // 95% accuracy requirement
                min_precision: 0.90,
                min_recall: 0.90,
                min_f1_score: 0.90,
                evaluation_window_minutes: 60,
                performance_monitoring_interval_seconds: 300,
            },
            features: FeatureConfig {
                noise_floor_window_size: 100,
                spectral_features: true,
                temporal_features: true,
                statistical_features: true,
                frequency_bins: 64,
                fft_size: 1024,
                overlap_factor: 0.5,
            },
            mitigation: MitigationConfig {
                strategies: mitigation_strategies,
                effectiveness_threshold: 0.7,
                implementation_priority,
            },
        }
    }
}

impl Config {
    pub fn from_file(path: &str) -> crate::Result<Self> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::with_prefix("ASA_INT"))
            .build()?;
        
        Ok(settings.try_deserialize()?)
    }
    
    pub fn validate(&self) -> crate::Result<()> {
        // Validate accuracy requirements
        if self.model.target_accuracy < 0.95 {
            return Err(crate::Error::InvalidInput(
                "Target accuracy must be >= 95%".to_string()
            ));
        }
        
        if self.performance.min_accuracy < 0.95 {
            return Err(crate::Error::InvalidInput(
                "Minimum accuracy must be >= 95%".to_string()
            ));
        }
        
        // Validate interference classes
        if self.classification.interference_classes.is_empty() {
            return Err(crate::Error::InvalidInput(
                "At least one interference class must be defined".to_string()
            ));
        }
        
        // Validate model configuration
        if self.model.hidden_layers.is_empty() && self.model.architecture == "neural_network" {
            return Err(crate::Error::InvalidInput(
                "Neural network architecture requires at least one hidden layer".to_string()
            ));
        }
        
        Ok(())
    }
}