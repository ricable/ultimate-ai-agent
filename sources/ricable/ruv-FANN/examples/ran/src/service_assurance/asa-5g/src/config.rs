use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub prediction: PredictionConfig,
    pub signal_analysis: SignalAnalysisConfig,
    pub mitigation: MitigationConfig,
    pub performance: PerformanceConfig,
    pub endc: EndcConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: u32,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    pub model_type: String, // "gradient_boosting", "neural_network", "ensemble"
    pub prediction_horizon_seconds: u32,
    pub target_accuracy: f64, // Must be > 0.80
    pub confidence_threshold: f64,
    pub feature_window_size: u32,
    pub update_interval_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalAnalysisConfig {
    pub lte_rsrp_threshold: f64,
    pub lte_sinr_threshold: f64,
    pub nr_ssb_rsrp_threshold: f64,
    pub nr_ssb_sinr_threshold: f64,
    pub signal_quality_weight: f64,
    pub signal_stability_window: u32,
    pub degradation_alert_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationConfig {
    pub strategies: HashMap<String, MitigationStrategy>,
    pub effectiveness_threshold: f64,
    pub auto_mitigation_enabled: bool,
    pub rollback_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    pub name: String,
    pub description: String,
    pub effectiveness_score: f64,
    pub implementation_complexity: u32,
    pub prerequisites: Vec<String>,
    pub estimated_improvement: f64,
    pub risk_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub min_prediction_accuracy: f64, // Must be > 0.80
    pub min_precision: f64,
    pub min_recall: f64,
    pub evaluation_window_minutes: u32,
    pub performance_monitoring_interval_seconds: u32,
    pub alert_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndcConfig {
    pub supported_failure_types: Vec<String>,
    pub bearer_types: Vec<String>,
    pub success_rate_threshold: f64,
    pub setup_timeout_ms: u32,
    pub max_retry_attempts: u32,
    pub congestion_threshold: f64,
}

impl Default for Config {
    fn default() -> Self {
        let mut mitigation_strategies = HashMap::new();
        
        mitigation_strategies.insert("BEARER_RECONFIGURATION".to_string(), MitigationStrategy {
            name: "Bearer Reconfiguration".to_string(),
            description: "Reconfigure bearer parameters to optimize ENDC setup".to_string(),
            effectiveness_score: 0.85,
            implementation_complexity: 2,
            prerequisites: vec!["Bearer configuration access".to_string()],
            estimated_improvement: 0.75,
            risk_level: "LOW".to_string(),
        });
        
        mitigation_strategies.insert("CELL_RESELECTION".to_string(), MitigationStrategy {
            name: "Cell Reselection".to_string(),
            description: "Force UE to select better serving cells".to_string(),
            effectiveness_score: 0.78,
            implementation_complexity: 3,
            prerequisites: vec!["Neighbor cell availability".to_string(), "Handover capability".to_string()],
            estimated_improvement: 0.70,
            risk_level: "MEDIUM".to_string(),
        });
        
        mitigation_strategies.insert("PARAMETER_OPTIMIZATION".to_string(), MitigationStrategy {
            name: "Parameter Optimization".to_string(),
            description: "Optimize radio parameters for better ENDC performance".to_string(),
            effectiveness_score: 0.82,
            implementation_complexity: 4,
            prerequisites: vec!["Parameter access".to_string(), "Performance monitoring".to_string()],
            estimated_improvement: 0.68,
            risk_level: "LOW".to_string(),
        });
        
        mitigation_strategies.insert("LOAD_BALANCING".to_string(), MitigationStrategy {
            name: "Load Balancing".to_string(),
            description: "Redistribute load to reduce congestion".to_string(),
            effectiveness_score: 0.75,
            implementation_complexity: 3,
            prerequisites: vec!["Load monitoring".to_string(), "Traffic steering capability".to_string()],
            estimated_improvement: 0.65,
            risk_level: "MEDIUM".to_string(),
        });
        
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 50052,
                max_connections: 1000,
                timeout_seconds: 30,
            },
            prediction: PredictionConfig {
                model_type: "gradient_boosting".to_string(),
                prediction_horizon_seconds: 60,
                target_accuracy: 0.80, // 80% accuracy requirement
                confidence_threshold: 0.75,
                feature_window_size: 10,
                update_interval_seconds: 30,
            },
            signal_analysis: SignalAnalysisConfig {
                lte_rsrp_threshold: -110.0,
                lte_sinr_threshold: 10.0,
                nr_ssb_rsrp_threshold: -115.0,
                nr_ssb_sinr_threshold: 15.0,
                signal_quality_weight: 0.4,
                signal_stability_window: 30,
                degradation_alert_threshold: 0.8,
            },
            mitigation: MitigationConfig {
                strategies: mitigation_strategies,
                effectiveness_threshold: 0.7,
                auto_mitigation_enabled: false,
                rollback_enabled: true,
            },
            performance: PerformanceConfig {
                min_prediction_accuracy: 0.80, // 80% accuracy requirement
                min_precision: 0.75,
                min_recall: 0.75,
                evaluation_window_minutes: 60,
                performance_monitoring_interval_seconds: 300,
                alert_threshold: 0.75,
            },
            endc: EndcConfig {
                supported_failure_types: vec![
                    "INITIAL_SETUP".to_string(),
                    "BEARER_SETUP".to_string(),
                    "BEARER_MODIFICATION".to_string(),
                    "RELEASE".to_string(),
                ],
                bearer_types: vec![
                    "SRB".to_string(),
                    "DRB".to_string(),
                    "MCG_DRB".to_string(),
                    "SCG_DRB".to_string(),
                    "SPLIT_DRB".to_string(),
                ],
                success_rate_threshold: 0.95,
                setup_timeout_ms: 5000,
                max_retry_attempts: 3,
                congestion_threshold: 0.8,
            },
        }
    }
}

impl Config {
    pub fn from_file(path: &str) -> crate::Result<Self> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::with_prefix("ASA_5G"))
            .build()?;
        
        Ok(settings.try_deserialize()?)
    }
    
    pub fn validate(&self) -> crate::Result<()> {
        // Validate accuracy requirements
        if self.prediction.target_accuracy < 0.80 {
            return Err(crate::Error::InvalidInput(
                "Target accuracy must be >= 80%".to_string()
            ));
        }
        
        if self.performance.min_prediction_accuracy < 0.80 {
            return Err(crate::Error::InvalidInput(
                "Minimum prediction accuracy must be >= 80%".to_string()
            ));
        }
        
        // Validate failure types
        if self.endc.supported_failure_types.is_empty() {
            return Err(crate::Error::InvalidInput(
                "At least one failure type must be supported".to_string()
            ));
        }
        
        // Validate signal thresholds
        if self.signal_analysis.lte_rsrp_threshold > -50.0 {
            return Err(crate::Error::InvalidInput(
                "LTE RSRP threshold should be reasonable (< -50 dBm)".to_string()
            ));
        }
        
        // Validate mitigation strategies
        if self.mitigation.strategies.is_empty() {
            return Err(crate::Error::InvalidInput(
                "At least one mitigation strategy must be configured".to_string()
            ));
        }
        
        Ok(())
    }
    
    pub fn get_mitigation_strategy(&self, strategy_name: &str) -> Option<&MitigationStrategy> {
        self.mitigation.strategies.get(strategy_name)
    }
    
    pub fn is_signal_quality_acceptable(&self, lte_rsrp: f64, lte_sinr: f64, nr_rsrp: f64, nr_sinr: f64) -> bool {
        lte_rsrp >= self.signal_analysis.lte_rsrp_threshold &&
        lte_sinr >= self.signal_analysis.lte_sinr_threshold &&
        nr_rsrp >= self.signal_analysis.nr_ssb_rsrp_threshold &&
        nr_sinr >= self.signal_analysis.nr_ssb_sinr_threshold
    }
}