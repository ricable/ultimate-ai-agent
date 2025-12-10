//! ENDC (E-UTRA-NR Dual Connectivity) Prediction Model
//! 
//! Specialized neural network for predicting 5G ENDC establishment success,
//! SCG (Secondary Cell Group) failure rates, and dual connectivity performance.

use crate::neural::kpi_predictor::{CsvKpiFeatures, KpiPredictor, KpiType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ENDC prediction model for 5G dual connectivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndcPredictor {
    pub base_predictor: KpiPredictor,
    pub establishment_model: EstablishmentModel,
    pub scg_failure_model: ScgFailureModel,
    pub capability_matcher: CapabilityMatcher,
    pub performance_optimizer: PerformanceOptimizer,
}

/// ENDC establishment success modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstablishmentModel {
    pub anchor_requirements: AnchorRequirements,
    pub ue_capability_weights: HashMap<String, f32>,
    pub network_readiness_factors: NetworkReadinessFactors,
    pub measurement_thresholds: MeasurementThresholds,
}

/// SCG failure prediction and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScgFailureModel {
    pub failure_causes: HashMap<FailureCause, f32>,
    pub recovery_mechanisms: Vec<RecoveryMechanism>,
    pub failure_prediction_weights: Vec<f32>,
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

/// UE capability matching for ENDC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityMatcher {
    pub supported_band_combinations: Vec<BandCombination>,
    pub capability_indicators: Vec<CapabilityIndicator>,
    pub compatibility_matrix: HashMap<String, f32>,
}

/// Performance optimization for dual connectivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimizer {
    pub traffic_splitting_ratios: HashMap<String, (f32, f32)>, // (LTE, NR) ratios
    pub bearer_mapping_rules: Vec<BearerMappingRule>,
    pub load_balancing_thresholds: LoadBalancingThresholds,
}

/// LTE anchor cell requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorRequirements {
    pub minimum_sinr: f32,
    pub minimum_throughput: f32,
    pub maximum_load: f32,
    pub required_features: Vec<String>,
}

/// Network readiness factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkReadinessFactors {
    pub nr_coverage_availability: f32,
    pub transport_capacity: f32,
    pub synchronization_accuracy: f32,
    pub backhaul_latency: f32,
}

/// Measurement thresholds for ENDC decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementThresholds {
    pub b1_threshold: f32,        // NR measurement threshold
    pub a2_threshold: f32,        // LTE serving cell threshold
    pub time_to_trigger: u32,     // TTT in ms
    pub hysteresis: f32,          // Hysteresis in dB
}

/// SCG failure causes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FailureCause {
    PoorNrCoverage,
    HighInterference,
    TransportIssues,
    UeCapabilityMismatch,
    ConfigurationError,
    HandoverFailure,
    RadioLinkFailure,
}

/// Recovery mechanisms for SCG failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryMechanism {
    pub mechanism_name: String,
    pub trigger_conditions: Vec<String>,
    pub success_rate: f32,
    pub recovery_time_ms: u32,
}

/// Mitigation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    pub strategy_name: String,
    pub target_failure_cause: FailureCause,
    pub effectiveness: f32,
    pub implementation_priority: Priority,
}

/// Band combination for dual connectivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandCombination {
    pub lte_band: String,
    pub nr_band: String,
    pub aggregation_type: AggregationType,
    pub performance_factor: f32,
}

/// Aggregation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    IntraFrequency,
    InterFrequency,
    InterBand,
}

/// UE capability indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityIndicator {
    pub indicator_name: String,
    pub requirement_type: RequirementType,
    pub threshold_value: f32,
}

/// Requirement types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequirementType {
    Mandatory,
    Optional,
    Conditional,
}

/// Bearer mapping rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BearerMappingRule {
    pub qci: u8,
    pub preferred_rat: RatType, // Radio Access Technology
    pub splitting_allowed: bool,
    pub priority: u8,
}

/// Radio Access Technology types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RatType {
    Lte,
    Nr,
    Both,
}

/// Load balancing thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingThresholds {
    pub lte_utilization_threshold: f32,
    pub nr_utilization_threshold: f32,
    pub traffic_offload_ratio: f32,
    pub rebalancing_hysteresis: f32,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// ENDC prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndcPrediction {
    pub establishment_success_probability: f32,
    pub expected_setup_time_ms: u32,
    pub scg_failure_probability: f32,
    pub performance_gain_estimate: PerformanceGain,
    pub capability_assessment: CapabilityAssessment,
    pub optimization_recommendations: Vec<String>,
    pub risk_factors: Vec<String>,
    pub confidence: f32,
}

/// Performance gain estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceGain {
    pub throughput_improvement: f32,
    pub latency_reduction: f32,
    pub capacity_increase: f32,
    pub user_experience_score: f32,
}

/// UE capability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityAssessment {
    pub endc_capable_percentage: f32,
    pub supported_combinations: Vec<BandCombination>,
    pub limitation_factors: Vec<String>,
    pub upgrade_recommendations: Vec<String>,
}

impl EndcPredictor {
    /// Create new ENDC predictor
    pub fn new(model_id: String) -> Self {
        let base_predictor = KpiPredictor::new(KpiType::EndcEstablishment, model_id);
        
        Self {
            base_predictor,
            establishment_model: EstablishmentModel {
                anchor_requirements: AnchorRequirements {
                    minimum_sinr: 10.0,
                    minimum_throughput: 20.0,
                    maximum_load: 80.0,
                    required_features: vec![
                        "ENDC_SUPPORT".to_string(),
                        "B1_MEASUREMENTS".to_string(),
                        "SCG_CONFIG".to_string(),
                    ],
                },
                ue_capability_weights: HashMap::from([
                    ("DC_SUPPORT".to_string(), 1.0),
                    ("NR_BANDS".to_string(), 0.8),
                    ("CA_SUPPORT".to_string(), 0.6),
                    ("MIMO_CAPABILITY".to_string(), 0.4),
                ]),
                network_readiness_factors: NetworkReadinessFactors {
                    nr_coverage_availability: 0.8,
                    transport_capacity: 0.9,
                    synchronization_accuracy: 0.95,
                    backhaul_latency: 5.0, // ms
                },
                measurement_thresholds: MeasurementThresholds {
                    b1_threshold: -110.0, // dBm
                    a2_threshold: -100.0,  // dBm
                    time_to_trigger: 320,  // ms
                    hysteresis: 2.0,       // dB
                },
            },
            scg_failure_model: ScgFailureModel {
                failure_causes: HashMap::from([
                    (FailureCause::PoorNrCoverage, 0.4),
                    (FailureCause::HighInterference, 0.2),
                    (FailureCause::TransportIssues, 0.15),
                    (FailureCause::UeCapabilityMismatch, 0.1),
                    (FailureCause::ConfigurationError, 0.08),
                    (FailureCause::HandoverFailure, 0.05),
                    (FailureCause::RadioLinkFailure, 0.02),
                ]),
                recovery_mechanisms: vec![
                    RecoveryMechanism {
                        mechanism_name: "SCG Reconfiguration".to_string(),
                        trigger_conditions: vec!["RLF_DETECTED".to_string()],
                        success_rate: 0.8,
                        recovery_time_ms: 200,
                    },
                    RecoveryMechanism {
                        mechanism_name: "Fallback to LTE".to_string(),
                        trigger_conditions: vec!["SCG_FAILURE".to_string()],
                        success_rate: 0.95,
                        recovery_time_ms: 100,
                    },
                ],
                failure_prediction_weights: vec![0.3, 0.25, 0.2, 0.15, 0.1],
                mitigation_strategies: Vec::new(),
            },
            capability_matcher: CapabilityMatcher {
                supported_band_combinations: vec![
                    BandCombination {
                        lte_band: "B3".to_string(),
                        nr_band: "n78".to_string(),
                        aggregation_type: AggregationType::InterBand,
                        performance_factor: 1.5,
                    },
                    BandCombination {
                        lte_band: "B1".to_string(),
                        nr_band: "n28".to_string(),
                        aggregation_type: AggregationType::InterBand,
                        performance_factor: 1.3,
                    },
                ],
                capability_indicators: vec![
                    CapabilityIndicator {
                        indicator_name: "ENDC_DC_SUPPORT".to_string(),
                        requirement_type: RequirementType::Mandatory,
                        threshold_value: 1.0,
                    },
                ],
                compatibility_matrix: HashMap::new(),
            },
            performance_optimizer: PerformanceOptimizer {
                traffic_splitting_ratios: HashMap::from([
                    ("VOICE".to_string(), (1.0, 0.0)),   // Voice stays on LTE
                    ("VIDEO".to_string(), (0.3, 0.7)),   // Video prefers NR
                    ("DATA".to_string(), (0.4, 0.6)),    // Data can split
                    ("BACKGROUND".to_string(), (0.6, 0.4)), // Background prefers LTE
                ]),
                bearer_mapping_rules: vec![
                    BearerMappingRule {
                        qci: 1,
                        preferred_rat: RatType::Lte, // Voice
                        splitting_allowed: false,
                        priority: 1,
                    },
                    BearerMappingRule {
                        qci: 8,
                        preferred_rat: RatType::Nr, // Best effort data
                        splitting_allowed: true,
                        priority: 3,
                    },
                ],
                load_balancing_thresholds: LoadBalancingThresholds {
                    lte_utilization_threshold: 80.0,
                    nr_utilization_threshold: 70.0,
                    traffic_offload_ratio: 0.3,
                    rebalancing_hysteresis: 5.0,
                },
            },
        }
    }
    
    /// Train ENDC prediction model
    pub fn train(&mut self, training_data: &[(CsvKpiFeatures, f32)]) -> Result<(), String> {
        if training_data.is_empty() {
            return Err("Training data cannot be empty".to_string());
        }
        
        // Train base predictor using ENDC setup success rate
        self.base_predictor.train(training_data)?;
        
        // Calibrate establishment model
        self.calibrate_establishment_model(training_data)?;
        
        // Analyze failure patterns
        self.analyze_failure_patterns(training_data)?;
        
        // Update capability assessment
        self.update_capability_assessment(training_data)?;
        
        Ok(())
    }
    
    /// Calibrate establishment model from training data
    fn calibrate_establishment_model(&mut self, training_data: &[(CsvKpiFeatures, f32)]) -> Result<(), String> {
        // Analyze anchor cell requirements based on successful ENDC sessions
        let successful_sessions: Vec<&CsvKpiFeatures> = training_data.iter()
            .filter_map(|(features, success_rate)| {
                if *success_rate > 90.0 { Some(features) } else { None }
            })
            .collect();
        
        if !successful_sessions.is_empty() {
            // Calculate minimum requirements from successful sessions
            let min_sinr = successful_sessions.iter()
                .map(|f| f.sinr_pusch_avg)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(10.0);
            
            let min_throughput = successful_sessions.iter()
                .map(|f| f.ave_4g_lte_dl_user_thrput)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(20.0);
            
            // Update anchor requirements
            self.establishment_model.anchor_requirements.minimum_sinr = min_sinr * 0.9; // 10% margin
            self.establishment_model.anchor_requirements.minimum_throughput = min_throughput * 0.8; // 20% margin
        }
        
        Ok(())
    }
    
    /// Analyze failure patterns from training data
    fn analyze_failure_patterns(&mut self, training_data: &[(CsvKpiFeatures, f32)]) -> Result<(), String> {
        let failed_sessions: Vec<&CsvKpiFeatures> = training_data.iter()
            .filter_map(|(features, success_rate)| {
                if *success_rate < 70.0 { Some(features) } else { None }
            })
            .collect();
        
        if !failed_sessions.is_empty() {
            // Analyze common failure causes
            let poor_sinr_failures = failed_sessions.iter()
                .filter(|f| f.sinr_pusch_avg < 8.0)
                .count() as f32 / failed_sessions.len() as f32;
            
            let high_scg_failure_rate = failed_sessions.iter()
                .filter(|f| f.endc_scg_failure_ratio > 10.0)
                .count() as f32 / failed_sessions.len() as f32;
            
            // Update failure cause probabilities
            if poor_sinr_failures > 0.3 {
                self.scg_failure_model.failure_causes.insert(FailureCause::PoorNrCoverage, poor_sinr_failures);
            }
            
            if high_scg_failure_rate > 0.2 {
                self.scg_failure_model.failure_causes.insert(FailureCause::ConfigurationError, high_scg_failure_rate);
            }
        }
        
        Ok(())
    }
    
    /// Update capability assessment based on training data
    fn update_capability_assessment(&mut self, training_data: &[(CsvKpiFeatures, f32)]) -> Result<(), String> {
        // Calculate ENDC capable UE percentage from data
        let total_sessions = training_data.len() as f32;
        let endc_attempts = training_data.iter()
            .map(|(features, _)| features.endc_establishment_att)
            .sum::<f32>();
        
        if total_sessions > 0.0 && endc_attempts > 0.0 {
            // Update network readiness factors
            let avg_success_rate: f32 = training_data.iter()
                .map(|(_, success_rate)| success_rate)
                .sum::<f32>() / total_sessions;
            
            // Adjust network readiness based on observed performance
            self.establishment_model.network_readiness_factors.nr_coverage_availability = 
                (avg_success_rate / 100.0).clamp(0.5, 1.0);
        }
        
        Ok(())
    }
    
    /// Predict ENDC establishment success and performance
    pub fn predict_endc_performance(&self, features: &CsvKpiFeatures) -> Result<EndcPrediction, String> {
        if !self.base_predictor.is_trained {
            return Err("Model must be trained before prediction".to_string());
        }
        
        // Get base establishment probability
        let base_prediction = self.base_predictor.predict(features)?;
        let establishment_success_probability = base_prediction.predicted_value / 100.0;
        
        // Calculate setup time
        let expected_setup_time_ms = self.calculate_setup_time(features);
        
        // Predict SCG failure probability
        let scg_failure_probability = self.predict_scg_failure_probability(features);
        
        // Estimate performance gains
        let performance_gain_estimate = self.estimate_performance_gains(features);
        
        // Assess UE capabilities
        let capability_assessment = self.assess_ue_capabilities(features);
        
        // Generate recommendations
        let optimization_recommendations = self.generate_endc_recommendations(features);
        
        // Identify risk factors
        let risk_factors = self.identify_endc_risk_factors(features);
        
        Ok(EndcPrediction {
            establishment_success_probability,
            expected_setup_time_ms,
            scg_failure_probability,
            performance_gain_estimate,
            capability_assessment,
            optimization_recommendations,
            risk_factors,
            confidence: base_prediction.confidence,
        })
    }
    
    /// Calculate expected ENDC setup time
    fn calculate_setup_time(&self, features: &CsvKpiFeatures) -> u32 {
        let mut setup_time = 500; // Base setup time in ms
        
        // Add delays based on conditions
        if features.sinr_pusch_avg < 10.0 {
            setup_time += 200; // Poor signal quality adds delay
        }
        
        if features.rrc_connected_users_avg > 100.0 {
            setup_time += 150; // High load adds delay
        }
        
        if features.lte_intra_freq_ho_sr < 95.0 {
            setup_time += 100; // Poor handover performance indicates network issues
        }
        
        setup_time
    }
    
    /// Predict SCG failure probability
    fn predict_scg_failure_probability(&self, features: &CsvKpiFeatures) -> f32 {
        let mut failure_probability = 0.0;
        
        // Analyze each failure cause
        for (cause, base_probability) in &self.scg_failure_model.failure_causes {
            let cause_probability = match cause {
                FailureCause::PoorNrCoverage => {
                    if features.sinr_pusch_avg < 8.0 {
                        base_probability * 1.5
                    } else if features.sinr_pusch_avg < 12.0 {
                        *base_probability
                    } else {
                        base_probability * 0.5
                    }
                },
                FailureCause::HighInterference => {
                    if features.mac_dl_bler > 10.0 {
                        base_probability * 1.3
                    } else {
                        base_probability * 0.7
                    }
                },
                FailureCause::UeCapabilityMismatch => {
                    // Based on ENDC setup attempts vs success
                    if features.endc_establishment_att > 0.0 && features.endc_establishment_succ == 0.0 {
                        base_probability * 2.0
                    } else {
                        base_probability * 0.5
                    }
                },
                _ => *base_probability,
            };
            
            failure_probability += cause_probability;
        }
        
        failure_probability.min(1.0)
    }
    
    /// Estimate performance gains from ENDC
    fn estimate_performance_gains(&self, features: &CsvKpiFeatures) -> PerformanceGain {
        // Conservative estimates based on dual connectivity theory
        let current_throughput = features.ave_4g_lte_dl_user_thrput + features.ave_4g_lte_ul_user_thrput;
        
        // Throughput improvement depends on NR availability and quality
        let throughput_improvement = if features.sinr_pusch_avg > 15.0 {
            current_throughput * 0.6 // 60% improvement in good conditions
        } else if features.sinr_pusch_avg > 10.0 {
            current_throughput * 0.3 // 30% improvement in fair conditions
        } else {
            current_throughput * 0.1 // 10% improvement in poor conditions
        };
        
        // Latency reduction from NR's lower latency
        let latency_reduction = if features.dl_latency_avg > 20.0 {
            features.dl_latency_avg * 0.3 // 30% latency reduction
        } else {
            features.dl_latency_avg * 0.15 // 15% reduction for already good latency
        };
        
        // Capacity increase from load distribution
        let capacity_increase = if features.rrc_connected_users_avg > 80.0 {
            25.0 // 25% capacity increase in high-load scenarios
        } else {
            15.0 // 15% capacity increase in normal load
        };
        
        // User experience score (0-100)
        let user_experience_score = ((throughput_improvement / current_throughput) * 40.0 +
                                   (latency_reduction / features.dl_latency_avg) * 30.0 +
                                   (capacity_increase / 100.0) * 30.0).min(100.0);
        
        PerformanceGain {
            throughput_improvement,
            latency_reduction,
            capacity_increase,
            user_experience_score,
        }
    }
    
    /// Assess UE capabilities for ENDC
    fn assess_ue_capabilities(&self, features: &CsvKpiFeatures) -> CapabilityAssessment {
        // Estimate ENDC capable UE percentage based on establishment attempts
        let endc_capable_percentage = if features.endc_establishment_att > 0.0 {
            (features.endc_establishment_succ / features.endc_establishment_att * 100.0).min(100.0)
        } else {
            50.0 // Default estimate
        };
        
        // Supported combinations (simplified based on frequency band)
        let supported_combinations = self.capability_matcher.supported_band_combinations.clone();
        
        // Limitation factors
        let mut limitation_factors = Vec::new();
        if endc_capable_percentage < 70.0 {
            limitation_factors.push("Low UE ENDC capability penetration".to_string());
        }
        if features.endc_scg_failure_ratio > 5.0 {
            limitation_factors.push("High SCG failure rate".to_string());
        }
        
        // Upgrade recommendations
        let mut upgrade_recommendations = Vec::new();
        if endc_capable_percentage < 80.0 {
            upgrade_recommendations.push("Encourage UE upgrades to ENDC-capable devices".to_string());
        }
        if features.sinr_pusch_avg < 12.0 {
            upgrade_recommendations.push("Improve LTE anchor cell performance".to_string());
        }
        
        CapabilityAssessment {
            endc_capable_percentage,
            supported_combinations,
            limitation_factors,
            upgrade_recommendations,
        }
    }
    
    /// Generate ENDC optimization recommendations
    fn generate_endc_recommendations(&self, features: &CsvKpiFeatures) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Anchor cell optimization
        if features.sinr_pusch_avg < self.establishment_model.anchor_requirements.minimum_sinr {
            recommendations.push("Improve LTE anchor cell SINR through power optimization".to_string());
            recommendations.push("Check for interference on LTE anchor frequency".to_string());
        }
        
        // Throughput optimization
        if features.ave_4g_lte_dl_user_thrput < self.establishment_model.anchor_requirements.minimum_throughput {
            recommendations.push("Optimize LTE cell throughput before enabling ENDC".to_string());
        }
        
        // SCG failure mitigation
        if features.endc_scg_failure_ratio > 5.0 {
            recommendations.push("Optimize B1 measurement thresholds".to_string());
            recommendations.push("Review NR cell configuration parameters".to_string());
            recommendations.push("Implement SCG failure recovery mechanisms".to_string());
        }
        
        // Load balancing
        if features.rrc_connected_users_avg > 100.0 {
            recommendations.push("Implement intelligent traffic steering between LTE and NR".to_string());
            recommendations.push("Configure QCI-based bearer mapping".to_string());
        }
        
        // UE capability
        if features.endc_establishment_att == 0.0 {
            recommendations.push("Verify ENDC feature is enabled in network".to_string());
            recommendations.push("Check UE capability signaling".to_string());
        }
        
        recommendations.sort();
        recommendations.dedup();
        recommendations
    }
    
    /// Identify ENDC risk factors
    fn identify_endc_risk_factors(&self, features: &CsvKpiFeatures) -> Vec<String> {
        let mut risk_factors = Vec::new();
        
        // Poor anchor cell performance
        if features.sinr_pusch_avg < 8.0 {
            risk_factors.push("Very poor LTE anchor cell SINR may prevent ENDC establishment".to_string());
        }
        
        // High error rates
        if features.mac_dl_bler > 15.0 {
            risk_factors.push("High LTE error rate may cause frequent SCG failures".to_string());
        }
        
        // Handover performance issues
        if features.lte_intra_freq_ho_sr < 90.0 {
            risk_factors.push("Poor handover performance indicates network instability".to_string());
        }
        
        // High failure rates
        if features.endc_scg_failure_ratio > 10.0 {
            risk_factors.push("High historical SCG failure rate".to_string());
        }
        
        // Low capability penetration
        if features.endc_establishment_att > 0.0 {
            let success_rate = features.endc_establishment_succ / features.endc_establishment_att;
            if success_rate < 0.8 {
                risk_factors.push("Low ENDC establishment success rate".to_string());
            }
        }
        
        risk_factors
    }
    
    /// Predict optimal ENDC configuration
    pub fn predict_optimal_endc_configuration(&self, features: &CsvKpiFeatures) -> Result<HashMap<String, f32>, String> {
        let mut config = HashMap::new();
        
        // B1 threshold optimization
        let optimal_b1_threshold = if features.sinr_pusch_avg > 15.0 {
            -105.0 // More aggressive threshold for good anchor cells
        } else {
            -110.0 // Conservative threshold for poor anchor cells
        };
        config.insert("b1_threshold_dbm".to_string(), optimal_b1_threshold);
        
        // Time to trigger optimization
        let optimal_ttt = if features.lte_intra_freq_ho_sr > 95.0 {
            160.0 // Shorter TTT for stable networks
        } else {
            320.0 // Longer TTT for unstable networks
        };
        config.insert("time_to_trigger_ms".to_string(), optimal_ttt);
        
        // Traffic splitting ratio
        let nr_traffic_ratio = if features.ave_4g_lte_dl_user_thrput > 50.0 {
            0.6 // More traffic to NR in good conditions
        } else {
            0.3 // Conservative traffic splitting
        };
        config.insert("nr_traffic_ratio".to_string(), nr_traffic_ratio);
        
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_endc_predictor_creation() {
        let predictor = EndcPredictor::new("test_endc_predictor".to_string());
        assert_eq!(predictor.base_predictor.model_id, "test_endc_predictor");
        assert_eq!(predictor.base_predictor.kpi_type, KpiType::EndcEstablishment);
    }
    
    #[test]
    fn test_setup_time_calculation() {
        let predictor = EndcPredictor::new("test".to_string());
        let features = create_test_features();
        
        let setup_time = predictor.calculate_setup_time(&features);
        assert!(setup_time >= 500); // Should include base time
    }
    
    #[test]
    fn test_scg_failure_prediction() {
        let predictor = EndcPredictor::new("test".to_string());
        let mut features = create_test_features();
        features.sinr_pusch_avg = 5.0; // Poor SINR
        features.mac_dl_bler = 15.0;   // High error rate
        
        let failure_prob = predictor.predict_scg_failure_probability(&features);
        assert!(failure_prob > 0.0 && failure_prob <= 1.0);
    }
    
    fn create_test_features() -> CsvKpiFeatures {
        CsvKpiFeatures {
            sinr_pusch_avg: 15.0,
            sinr_pucch_avg: 14.0,
            ul_rssi_total: -105.0,
            mac_dl_bler: 2.0,
            mac_ul_bler: 1.5,
            rrc_connected_users_avg: 50.0,
            ul_volume_pdcp_gbytes: 1.0,
            dl_volume_pdcp_gbytes: 5.0,
            volte_traffic_erl: 1.0,
            eric_traff_erab_erl: 10.0,
            ave_4g_lte_dl_user_thrput: 50.0,
            ave_4g_lte_ul_user_thrput: 25.0,
            dl_latency_avg: 15.0,
            erab_drop_rate_qci_5: 1.0,
            lte_dcr_volte: 0.5,
            lte_intra_freq_ho_sr: 95.0,
            lte_inter_freq_ho_sr: 93.0,
            inter_freq_ho_attempts: 50.0,
            intra_freq_ho_attempts: 100.0,
            endc_establishment_att: 100.0,
            endc_establishment_succ: 95.0,
            endc_setup_sr: 95.0,
            endc_scg_failure_ratio: 2.0,
            frequency_band: "LTE1800".to_string(),
            band_count: 4.0,
            active_ues_dl: 20.0,
            active_ues_ul: 15.0,
        }
    }
}