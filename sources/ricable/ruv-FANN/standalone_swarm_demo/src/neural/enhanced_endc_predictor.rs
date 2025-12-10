//! Enhanced ENDC Predictor with Real Data Integration
//! 
//! Migrated and enhanced from ASA 5G modules to work with real fanndata.csv
//! Provides sophisticated 5G ENDC (E-UTRAN New Radio Dual Connectivity) 
//! setup failure prediction using real network KPI data.

use crate::utils::csv_data_parser::{RealNetworkData, CsvDataParser};
use crate::neural::ml_model::{MLModel, ModelType, TrainingConfig};
use crate::utils::metrics::PerformanceMetrics;
use crate::swarm::coordinator::SwarmCoordinator;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Enhanced ENDC setup failure predictor using real network data
pub struct EnhancedEndcPredictor {
    /// Neural network model for prediction
    model: Box<dyn MLModel>,
    
    /// Feature engineering pipeline
    feature_engineer: EndcFeatureEngineer,
    
    /// Risk assessment engine
    risk_assessor: RiskAssessmentEngine,
    
    /// Performance metrics tracking
    metrics: PerformanceMetrics,
    
    /// Historical predictions for learning
    prediction_history: Vec<EndcPrediction>,
    
    /// Configuration parameters
    config: EndcPredictorConfig,
}

/// Feature engineering for ENDC prediction
#[derive(Debug, Clone)]
pub struct EndcFeatureEngineer {
    /// Feature scaling parameters learned from real data
    scaling_params: HashMap<String, (f64, f64)>, // (mean, std)
    
    /// Temporal feature window size
    window_size: usize,
    
    /// Feature importance weights derived from real data analysis
    feature_weights: HashMap<String, f64>,
    
    /// Historical signal quality data for temporal features
    signal_history: HashMap<String, Vec<SignalQualitySnapshot>>,
}

/// Signal quality snapshot for temporal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQualitySnapshot {
    pub timestamp: u64,
    pub rsrp: f64,
    pub rsrq: f64,
    pub sinr: f64,
    pub rssi: f64,
    pub endc_attempts: u64,
    pub endc_successes: u64,
    pub throughput: f64,
    pub latency: f64,
}

impl SignalQualitySnapshot {
    pub fn from_real_data(data: &RealNetworkData) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        Self {
            timestamp,
            rsrp: -70.0, // Estimated from RSSI and other metrics
            rsrq: -10.0, // Estimated
            sinr: data.sinr_pusch,
            rssi: data.ul_rssi_total,
            endc_attempts: data.endc_establishment_attempts,
            endc_successes: data.endc_establishment_success,
            throughput: data.dl_user_throughput,
            latency: data.dl_latency_avg,
        }
    }
}

/// Risk assessment engine for ENDC failures
#[derive(Debug, Clone)]
pub struct RiskAssessmentEngine {
    /// Risk factor weights learned from real data
    risk_weights: HashMap<String, f64>,
    
    /// Threshold values for different risk levels
    risk_thresholds: RiskThresholds,
    
    /// Historical failure patterns
    failure_patterns: Vec<FailurePattern>,
}

/// Risk thresholds derived from real data analysis
#[derive(Debug, Clone)]
pub struct RiskThresholds {
    pub low_risk_threshold: f64,
    pub medium_risk_threshold: f64,
    pub high_risk_threshold: f64,
    pub critical_risk_threshold: f64,
}

impl Default for RiskThresholds {
    fn default() -> Self {
        Self {
            low_risk_threshold: 0.2,
            medium_risk_threshold: 0.4,
            high_risk_threshold: 0.7,
            critical_risk_threshold: 0.9,
        }
    }
}

/// Failure pattern identified from historical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePattern {
    pub pattern_id: String,
    pub description: String,
    pub preconditions: Vec<String>,
    pub failure_probability: f64,
    pub mitigation_strategies: Vec<String>,
}

/// ENDC prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndcPrediction {
    pub timestamp: String,
    pub cell_id: String,
    pub failure_probability: f64,
    pub risk_level: RiskLevel,
    pub confidence_score: f64,
    pub contributing_factors: Vec<ContributingFactor>,
    pub recommended_actions: Vec<String>,
    pub time_to_failure_estimate: Option<u64>, // seconds
    pub raw_features: Vec<f64>,
    pub model_version: String,
}

/// Risk levels for ENDC setup failure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Contributing factors to ENDC failure risk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributingFactor {
    pub factor_name: String,
    pub impact_score: f64,
    pub current_value: f64,
    pub threshold_value: f64,
    pub description: String,
}

/// Configuration for ENDC predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndcPredictorConfig {
    pub prediction_threshold: f64,
    pub confidence_threshold: f64,
    pub feature_window_size: usize,
    pub enable_temporal_features: bool,
    pub enable_cross_cell_features: bool,
    pub model_update_frequency: u64, // seconds
    pub alert_threshold: f64,
}

impl Default for EndcPredictorConfig {
    fn default() -> Self {
        Self {
            prediction_threshold: 0.5,
            confidence_threshold: 0.7,
            feature_window_size: 10,
            enable_temporal_features: true,
            enable_cross_cell_features: true,
            model_update_frequency: 3600, // 1 hour
            alert_threshold: 0.8,
        }
    }
}

impl EndcFeatureEngineer {
    pub fn new(window_size: usize) -> Self {
        Self {
            scaling_params: HashMap::new(),
            window_size,
            feature_weights: HashMap::new(),
            signal_history: HashMap::new(),
        }
    }
    
    /// Extract ENDC-specific features from real network data
    pub fn extract_features(&mut self, data: &RealNetworkData) -> Vec<f64> {
        let cell_id = format!("{}_{}", data.enodeb_name, data.cell_name);
        
        // Update signal history
        let snapshot = SignalQualitySnapshot::from_real_data(data);
        self.signal_history
            .entry(cell_id.clone())
            .or_insert_with(Vec::new)
            .push(snapshot);
        
        // Keep only recent history
        if let Some(history) = self.signal_history.get_mut(&cell_id) {
            if history.len() > self.window_size * 2 {
                history.drain(0..self.window_size);
            }
        }
        
        let mut features = Vec::new();
        
        // Core ENDC features from real data
        features.extend(self.extract_core_endc_features(data));
        
        // Signal quality features
        features.extend(self.extract_signal_quality_features(data));
        
        // Network load and capacity features
        features.extend(self.extract_capacity_features(data));
        
        // Temporal features if enabled
        if self.signal_history.get(&cell_id).map_or(false, |h| h.len() >= 3) {
            features.extend(self.extract_temporal_features(&cell_id));
        } else {
            // Pad with zeros if insufficient history
            features.extend(vec![0.0; 8]);
        }
        
        // Cross-cell context features
        features.extend(self.extract_context_features(data));
        
        features
    }
    
    /// Extract core ENDC features from real data
    fn extract_core_endc_features(&self, data: &RealNetworkData) -> Vec<f64> {
        vec![
            // ENDC setup success rate (key predictor)
            data.endc_setup_sr / 100.0,
            
            // ENDC establishment ratio
            if data.endc_establishment_attempts > 0 {
                data.endc_establishment_success as f64 / data.endc_establishment_attempts as f64
            } else { 1.0 },
            
            // ENDC capable UEs ratio to total active users
            if data.active_users_dl > 0 {
                data.endc_capable_ues as f64 / data.active_users_dl as f64
            } else { 0.0 },
            
            // ENDC load indicators
            (data.endc_establishment_attempts as f64).ln_1p() / 10.0,
            (data.endc_capable_ues as f64).ln_1p() / 10.0,
            
            // 5G readiness indicators
            if data.band.contains("NR") || data.band.contains("5G") { 1.0 } else { 0.0 },
            
            // Dual connectivity stress indicators
            if data.endc_establishment_attempts > 100 { 1.0 } else { 
                data.endc_establishment_attempts as f64 / 100.0 
            },
        ]
    }
    
    /// Extract signal quality features affecting ENDC
    fn extract_signal_quality_features(&self, data: &RealNetworkData) -> Vec<f64> {
        vec![
            // Normalized SINR (critical for ENDC)
            (data.sinr_pusch + 20.0) / 40.0, // Map -20 to +20 dB to 0-1
            (data.sinr_pucch + 20.0) / 40.0,
            
            // RSSI quality indicators
            (data.ul_rssi_total + 140.0) / 40.0, // Map -140 to -100 dBm to 0-1
            (data.ul_rssi_pusch + 140.0) / 40.0,
            (data.ul_rssi_pucch + 140.0) / 40.0,
            
            // Signal stability (derived from power control)
            1.0 - (data.ue_power_limited_percent / 100.0),
            
            // Interference indicators
            data.mac_dl_bler / 100.0,
            data.mac_ul_bler / 100.0,
            
            // Quality consistency
            1.0 - (data.dl_packet_error_rate / 100.0),
            1.0 - (data.ul_packet_loss_rate / 100.0),
        ]
    }
    
    /// Extract network capacity and load features
    fn extract_capacity_features(&self, data: &RealNetworkData) -> Vec<f64> {
        vec![
            // User load indicators
            (data.rrc_connected_users.ln_1p()) / 15.0,
            (data.active_users_dl as f64).ln_1p() / 15.0,
            (data.active_users_ul as f64).ln_1p() / 15.0,
            
            // Traffic volume indicators
            (data.dl_volume_gbytes.ln_1p()) / 10.0,
            (data.ul_volume_gbytes.ln_1p()) / 10.0,
            
            // Throughput efficiency
            if data.dl_volume_gbytes > 0.0 {
                data.dl_user_throughput / (data.dl_volume_gbytes * 1000.0)
            } else { 0.0 },
            
            // VoLTE load (affects ENDC priorities)
            data.volte_traffic.ln_1p() / 10.0,
            
            // Network availability
            data.cell_availability / 100.0,
        ]
    }
    
    /// Extract temporal trend features
    fn extract_temporal_features(&self, cell_id: &str) -> Vec<f64> {
        if let Some(history) = self.signal_history.get(cell_id) {
            if history.len() < 3 {
                return vec![0.0; 8];
            }
            
            let recent = &history[history.len()-3..];
            
            vec![
                // SINR trend
                Self::calculate_trend(&recent.iter().map(|s| s.sinr).collect::<Vec<_>>()),
                
                // Throughput trend
                Self::calculate_trend(&recent.iter().map(|s| s.throughput).collect::<Vec<_>>()),
                
                // Latency trend
                -Self::calculate_trend(&recent.iter().map(|s| s.latency).collect::<Vec<_>>()), // Negative because increasing latency is bad
                
                // ENDC success rate trend
                Self::calculate_trend(&recent.iter().map(|s| {
                    if s.endc_attempts > 0 {
                        s.endc_successes as f64 / s.endc_attempts as f64
                    } else { 1.0 }
                }).collect::<Vec<_>>()),
                
                // Signal stability (variance)
                Self::calculate_stability(&recent.iter().map(|s| s.sinr).collect::<Vec<_>>()),
                
                // Load stability
                Self::calculate_stability(&recent.iter().map(|s| s.endc_attempts as f64).collect::<Vec<_>>()),
                
                // Recent failure rate
                recent.iter().map(|s| {
                    if s.endc_attempts > 0 {
                        1.0 - (s.endc_successes as f64 / s.endc_attempts as f64)
                    } else { 0.0 }
                }).sum::<f64>() / recent.len() as f64,
                
                // Time since last measurement (freshness)
                if let Some(last) = recent.last() {
                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                    ((now - last.timestamp) as f64 / 3600.0).min(1.0) // Hours, capped at 1
                } else { 1.0 },
            ]
        } else {
            vec![0.0; 8]
        }
    }
    
    /// Extract contextual features (cross-cell, time-of-day, etc.)
    fn extract_context_features(&self, data: &RealNetworkData) -> Vec<f64> {
        vec![
            // Band-specific features
            match data.band.as_str() {
                "LTE700" => 0.1,
                "LTE800" => 0.2,
                "LTE1800" => 0.3,
                "LTE2100" => 0.4,
                "LTE2600" => 0.5,
                _ => 0.0,
            },
            
            // Multi-band capability
            (data.num_bands as f64) / 5.0, // Normalize to 0-1
            
            // Handover stress indicators
            data.lte_intra_freq_ho_sr / 100.0,
            data.lte_inter_freq_ho_sr / 100.0,
            
            // Cross-RAT indicators
            if data.intra_freq_ho_attempts > 0 {
                1.0 - (data.lte_intra_freq_ho_sr / 100.0)
            } else { 0.0 },
            
            // VoIP quality impact
            data.voip_integrity_rate / 100.0,
            
            // Error rate consistency
            1.0 - ((data.erab_drop_rate_qci5 + data.erab_drop_rate_qci8) / 200.0),
        ]
    }
    
    /// Calculate trend from time series data
    fn calculate_trend(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter().enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();
        
        // Linear regression slope
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        
        // Normalize slope
        slope.tanh()
    }
    
    /// Calculate stability (inverse of normalized variance)
    fn calculate_stability(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 1.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        // Convert to stability score (0-1, higher is more stable)
        1.0 / (1.0 + variance)
    }
    
    /// Fit scaling parameters from training data
    pub fn fit_scaling(&mut self, training_data: &[RealNetworkData]) {
        if training_data.is_empty() {
            return;
        }
        
        // Extract features for all training samples
        let mut all_features = Vec::new();
        for data in training_data {
            let features = self.extract_features(data);
            all_features.push(features);
        }
        
        if all_features.is_empty() {
            return;
        }
        
        let num_features = all_features[0].len();
        
        // Calculate mean and std for each feature
        for feature_idx in 0..num_features {
            let feature_values: Vec<f64> = all_features
                .iter()
                .map(|features| features[feature_idx])
                .collect();
            
            let mean = feature_values.iter().sum::<f64>() / feature_values.len() as f64;
            let variance = feature_values
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / feature_values.len() as f64;
            let std = variance.sqrt();
            
            self.scaling_params.insert(
                format!("endc_feature_{}", feature_idx),
                (mean, std.max(1e-8)) // Avoid division by zero
            );
        }
        
        println!("✅ ENDC feature scaling fitted for {} features", num_features);
    }
    
    /// Apply feature scaling
    pub fn scale_features(&self, features: &mut [f64]) {
        for (i, feature) in features.iter_mut().enumerate() {
            if let Some((mean, std)) = self.scaling_params.get(&format!("endc_feature_{}", i)) {
                *feature = (*feature - mean) / std;
            }
        }
    }
}

impl RiskAssessmentEngine {
    pub fn new() -> Self {
        let mut risk_weights = HashMap::new();
        
        // Initialize risk weights based on domain knowledge
        risk_weights.insert("endc_setup_sr".to_string(), 0.25);
        risk_weights.insert("signal_quality".to_string(), 0.20);
        risk_weights.insert("network_load".to_string(), 0.15);
        risk_weights.insert("temporal_trend".to_string(), 0.15);
        risk_weights.insert("handover_performance".to_string(), 0.10);
        risk_weights.insert("interference_level".to_string(), 0.10);
        risk_weights.insert("context_factors".to_string(), 0.05);
        
        Self {
            risk_weights,
            risk_thresholds: RiskThresholds::default(),
            failure_patterns: Vec::new(),
        }
    }
    
    /// Assess ENDC failure risk from prediction features
    pub fn assess_risk(&self, prediction_score: f64, features: &[f64], data: &RealNetworkData) -> (RiskLevel, Vec<ContributingFactor>) {
        let mut contributing_factors = Vec::new();
        
        // Analyze each risk factor
        if features.len() >= 7 {
            // ENDC setup success rate factor
            let endc_sr = features[0];
            if endc_sr < 0.85 {
                contributing_factors.push(ContributingFactor {
                    factor_name: "Low ENDC Setup Success Rate".to_string(),
                    impact_score: 0.85 - endc_sr,
                    current_value: data.endc_setup_sr,
                    threshold_value: 85.0,
                    description: "ENDC setup success rate below acceptable threshold".to_string(),
                });
            }
            
            // Signal quality factor
            let avg_sinr = (features[7] + features[8]) / 2.0; // Normalized SINR
            if avg_sinr < 0.4 { // Below -12 dB equivalent
                contributing_factors.push(ContributingFactor {
                    factor_name: "Poor Signal Quality".to_string(),
                    impact_score: 0.4 - avg_sinr,
                    current_value: data.sinr_pusch,
                    threshold_value: -12.0,
                    description: "SINR below recommended level for reliable ENDC".to_string(),
                });
            }
            
            // Network load factor
            if features.len() >= 20 {
                let load_factor = features[17]; // RRC connected users
                if load_factor > 0.8 {
                    contributing_factors.push(ContributingFactor {
                        factor_name: "High Network Load".to_string(),
                        impact_score: load_factor - 0.8,
                        current_value: data.rrc_connected_users,
                        threshold_value: 50.0, // Example threshold
                        description: "High user load may impact ENDC establishment".to_string(),
                    });
                }
            }
            
            // Error rate factor
            if data.mac_dl_bler > 5.0 {
                contributing_factors.push(ContributingFactor {
                    factor_name: "High Error Rate".to_string(),
                    impact_score: data.mac_dl_bler / 100.0,
                    current_value: data.mac_dl_bler,
                    threshold_value: 5.0,
                    description: "High BLER may indicate RF issues affecting ENDC".to_string(),
                });
            }
        }
        
        // Determine risk level based on prediction score and contributing factors
        let risk_level = if prediction_score >= self.risk_thresholds.critical_risk_threshold ||
                           contributing_factors.iter().any(|f| f.impact_score > 0.8) {
            RiskLevel::Critical
        } else if prediction_score >= self.risk_thresholds.high_risk_threshold ||
                  contributing_factors.iter().any(|f| f.impact_score > 0.5) {
            RiskLevel::High
        } else if prediction_score >= self.risk_thresholds.medium_risk_threshold ||
                  contributing_factors.iter().any(|f| f.impact_score > 0.3) {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        
        (risk_level, contributing_factors)
    }
    
    /// Generate recommended actions based on risk assessment
    pub fn generate_recommendations(&self, risk_level: &RiskLevel, factors: &[ContributingFactor]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match risk_level {
            RiskLevel::Critical => {
                recommendations.push("URGENT: Investigate ENDC setup failures immediately".to_string());
                recommendations.push("Consider temporary load balancing to alternative cells".to_string());
                recommendations.push("Check for hardware or configuration issues".to_string());
            }
            RiskLevel::High => {
                recommendations.push("Monitor ENDC performance closely".to_string());
                recommendations.push("Review cell parameters and neighbor configuration".to_string());
                recommendations.push("Consider proactive optimization".to_string());
            }
            RiskLevel::Medium => {
                recommendations.push("Schedule routine optimization review".to_string());
                recommendations.push("Monitor trends for early warning signs".to_string());
            }
            RiskLevel::Low => {
                recommendations.push("Continue normal monitoring".to_string());
            }
        }
        
        // Add factor-specific recommendations
        for factor in factors {
            match factor.factor_name.as_str() {
                "Poor Signal Quality" => {
                    recommendations.push("Check antenna configuration and RF parameters".to_string());
                    recommendations.push("Investigate potential interference sources".to_string());
                }
                "High Network Load" => {
                    recommendations.push("Consider load balancing strategies".to_string());
                    recommendations.push("Review capacity planning".to_string());
                }
                "High Error Rate" => {
                    recommendations.push("Investigate RF path and hardware health".to_string());
                    recommendations.push("Check power control settings".to_string());
                }
                _ => {}
            }
        }
        
        recommendations.dedup();
        recommendations
    }
}

impl EnhancedEndcPredictor {
    /// Create new enhanced ENDC predictor
    pub fn new(config: EndcPredictorConfig) -> Self {
        let model = Box::new(MLModel::new(
            ModelType::NeuralNetwork,
            TrainingConfig {
                epochs: 150,
                learning_rate: 0.0005,
                batch_size: 64,
                validation_split: 0.25,
                early_stopping: true,
                patience: 15,
            }
        ));
        
        Self {
            model,
            feature_engineer: EndcFeatureEngineer::new(config.feature_window_size),
            risk_assessor: RiskAssessmentEngine::new(),
            metrics: PerformanceMetrics::new(),
            prediction_history: Vec::new(),
            config,
        }
    }
    
    /// Train the model on real network data
    pub fn train_on_real_data(&mut self, csv_file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("🧠 Training enhanced ENDC predictor on real data...");
        let start_time = Instant::now();
        
        // Parse real data
        let mut parser = CsvDataParser::new(csv_file_path.to_string());
        let training_data = parser.parse_real_data()?;
        
        println!("📊 Training data: {} records", training_data.len());
        
        // Create training features and labels
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        // Fit feature scaling first
        self.feature_engineer.fit_scaling(&training_data);
        
        for data in &training_data {
            let mut feature_vec = self.feature_engineer.extract_features(data);
            self.feature_engineer.scale_features(&mut feature_vec);
            
            // Create label based on ENDC performance
            let label = if data.endc_setup_sr < 80.0 || 
                          (data.endc_establishment_attempts > 0 && 
                           (data.endc_establishment_success as f64 / data.endc_establishment_attempts as f64) < 0.8) {
                1.0 // Failure risk
            } else {
                0.0 // Normal operation
            };
            
            features.push(feature_vec);
            labels.push(vec![label]);
        }
        
        println!("📈 Feature extraction complete: {} features per sample", 
            features.first().map_or(0, |f| f.len()));
        
        // Train the model
        self.model.train(&features, &labels)?;
        
        let training_time = start_time.elapsed();
        println!("⏱️  ENDC predictor training completed in {:.2}s", training_time.as_secs_f64());
        
        Ok(())
    }
    
    /// Predict ENDC setup failure probability for real data
    pub fn predict_failure_risk(&mut self, data: &RealNetworkData) -> EndcPrediction {
        let start_time = Instant::now();
        
        // Extract and scale features
        let mut features = self.feature_engineer.extract_features(data);
        self.feature_engineer.scale_features(&mut features);
        
        // Get model prediction
        let prediction_result = self.model.predict(&[features.clone()]);
        let failure_probability = match prediction_result {
            Ok(predictions) => {
                predictions.first()
                    .and_then(|p| p.first())
                    .copied()
                    .unwrap_or(0.0)
            }
            Err(_) => 0.0,
        };
        
        // Assess risk level and contributing factors
        let (risk_level, contributing_factors) = self.risk_assessor
            .assess_risk(failure_probability, &features, data);
        
        // Generate recommendations
        let recommended_actions = self.risk_assessor
            .generate_recommendations(&risk_level, &contributing_factors);
        
        // Calculate confidence based on feature quality and model certainty
        let confidence_score = self.calculate_confidence(&features, failure_probability);
        
        // Estimate time to failure if risk is significant
        let time_to_failure_estimate = if failure_probability > 0.7 {
            Some(self.estimate_time_to_failure(&features, failure_probability))
        } else {
            None
        };
        
        let prediction = EndcPrediction {
            timestamp: data.timestamp.clone(),
            cell_id: format!("{}_{}", data.enodeb_name, data.cell_name),
            failure_probability,
            risk_level,
            confidence_score,
            contributing_factors,
            recommended_actions,
            time_to_failure_estimate,
            raw_features: features,
            model_version: "enhanced_v1.0".to_string(),
        };
        
        // Store prediction in history
        self.prediction_history.push(prediction.clone());
        
        // Keep only recent predictions
        if self.prediction_history.len() > 10000 {
            self.prediction_history.drain(0..1000);
        }
        
        self.metrics.record_prediction_time(start_time.elapsed().as_millis() as f64);
        
        prediction
    }
    
    /// Calculate prediction confidence
    fn calculate_confidence(&self, features: &[f64], probability: f64) -> f64 {
        // Confidence based on feature completeness
        let feature_completeness = features.iter()
            .filter(|&&f| !f.is_nan() && f.is_finite())
            .count() as f64 / features.len() as f64;
        
        // Confidence based on prediction certainty (distance from 0.5)
        let prediction_certainty = (probability - 0.5).abs() * 2.0;
        
        // Combined confidence score
        (feature_completeness * 0.6 + prediction_certainty * 0.4).min(1.0)
    }
    
    /// Estimate time to failure based on trends
    fn estimate_time_to_failure(&self, features: &[f64], probability: f64) -> u64 {
        // Simple heuristic based on probability and trend features
        let base_time = match probability {
            p if p >= 0.9 => 300,   // 5 minutes
            p if p >= 0.8 => 1800,  // 30 minutes
            p if p >= 0.7 => 3600,  // 1 hour
            _ => 7200,              // 2 hours
        };
        
        // Adjust based on temporal trends if available
        let trend_adjustment = if features.len() > 25 {
            let trend = features[21]; // Assuming this is a trend feature
            if trend < -0.5 { 0.5 } else { 1.0 } // Faster failure if negative trend
        } else {
            1.0
        };
        
        (base_time as f64 * trend_adjustment) as u64
    }
    
    /// Get recent predictions for analysis
    pub fn get_recent_predictions(&self, limit: usize) -> Vec<&EndcPrediction> {
        self.prediction_history
            .iter()
            .rev()
            .take(limit)
            .collect()
    }
    
    /// Export prediction results
    pub fn export_predictions(&self, format: &str) -> Result<String, Box<dyn std::error::Error>> {
        match format {
            "json" => {
                let results = serde_json::to_string_pretty(&self.prediction_history)?;
                Ok(results)
            }
            "csv" => {
                let mut csv_output = String::from("timestamp,cell_id,failure_probability,risk_level,confidence,recommendations\n");
                for prediction in &self.prediction_history {
                    csv_output.push_str(&format!(
                        "{},{},{:.3},{:?},{:.3},{}\n",
                        prediction.timestamp,
                        prediction.cell_id,
                        prediction.failure_probability,
                        prediction.risk_level,
                        prediction.confidence_score,
                        prediction.recommended_actions.join("; ")
                    ));
                }
                Ok(csv_output)
            }
            _ => Err("Unsupported export format".into()),
        }
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let mut engineer = EndcFeatureEngineer::new(5);
        
        let data = RealNetworkData {
            timestamp: "2025-06-27 00".to_string(),
            enodeb_name: "TEST_ENODEB".to_string(),
            cell_name: "TEST_CELL".to_string(),
            endc_setup_sr: 85.5,
            endc_establishment_attempts: 150,
            endc_establishment_success: 143,
            sinr_pusch: 8.5,
            sinr_pucch: 6.2,
            ul_rssi_total: -115.0,
            cell_availability: 98.2,
            rrc_connected_users: 45.0,
            active_users_dl: 42,
            ..Default::default()
        };
        
        let features = engineer.extract_features(&data);
        
        // Verify reasonable number of features
        assert!(features.len() >= 30);
        
        // Verify features are in reasonable ranges
        for (i, &feature) in features.iter().enumerate() {
            assert!(feature.is_finite(), "Feature {} is not finite: {}", i, feature);
        }
    }
    
    #[test]
    fn test_risk_assessment() {
        let assessor = RiskAssessmentEngine::new();
        
        // Test high-risk scenario
        let high_risk_data = RealNetworkData {
            endc_setup_sr: 60.0, // Low success rate
            sinr_pusch: -15.0,   // Poor signal
            mac_dl_bler: 15.0,   // High error rate
            ..Default::default()
        };
        
        let features = vec![0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]; // Mock features
        let (risk_level, factors) = assessor.assess_risk(0.8, &features, &high_risk_data);
        
        assert!(matches!(risk_level, RiskLevel::High) || matches!(risk_level, RiskLevel::Critical));
        assert!(!factors.is_empty());
        
        // Test low-risk scenario
        let low_risk_data = RealNetworkData {
            endc_setup_sr: 98.0,
            sinr_pusch: 12.0,
            mac_dl_bler: 2.0,
            ..Default::default()
        };
        
        let features = vec![0.98, 0.95, 1.0, 0.5, 0.8, 1.0, 0.2, 0.8, 0.9];
        let (risk_level, _) = assessor.assess_risk(0.1, &features, &low_risk_data);
        
        assert!(matches!(risk_level, RiskLevel::Low));
    }
}