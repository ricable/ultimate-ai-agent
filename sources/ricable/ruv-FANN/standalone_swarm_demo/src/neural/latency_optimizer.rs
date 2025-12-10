//! Latency Optimization Model
//! 
//! Neural network specialized in predicting and optimizing network latency
//! based on handover patterns, processing delays, and transport network conditions.

use crate::neural::kpi_predictor::{CsvKpiFeatures, KpiPredictor, KpiType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Latency components for detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyComponents {
    pub air_interface_latency: f32,
    pub processing_delay: f32,
    pub transport_delay: f32,
    pub handover_delay: f32,
    pub queuing_delay: f32,
    pub total_latency: f32,
}

/// Latency optimization model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyOptimizer {
    pub base_predictor: KpiPredictor,
    pub processing_model: ProcessingDelayModel,
    pub handover_latency_model: HandoverLatencyModel,
    pub qci_latency_requirements: HashMap<u8, f32>,
    pub optimization_targets: LatencyTargets,
}

/// Processing delay modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingDelayModel {
    pub base_processing_delay: f32,
    pub load_factor_impact: f32,
    pub queue_depth_factor: f32,
    pub modulation_complexity_factor: HashMap<String, f32>,
    pub mimo_processing_overhead: f32,
}

/// Handover latency modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoverLatencyModel {
    pub measurement_reporting_delay: f32,
    pub decision_delay: f32,
    pub execution_delay: f32,
    pub inter_freq_penalty: f32,
    pub failure_retry_penalty: f32,
}

/// Latency optimization targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyTargets {
    pub voice_target_ms: f32,
    pub video_target_ms: f32,
    pub data_target_ms: f32,
    pub background_target_ms: f32,
    pub emergency_target_ms: f32,
}

/// Latency prediction with optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPrediction {
    pub predicted_latency: f32,
    pub latency_components: LatencyComponents,
    pub qci_specific_latencies: HashMap<u8, f32>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub performance_grade: LatencyGrade,
    pub confidence: f32,
}

/// Optimization opportunity identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub component: String,
    pub current_value: f32,
    pub target_value: f32,
    pub potential_improvement: f32,
    pub implementation_complexity: ComplexityLevel,
    pub recommendations: Vec<String>,
}

/// Implementation complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,    // Configuration changes
    Medium, // Software updates
    High,   // Hardware changes
}

/// Latency performance grading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyGrade {
    Excellent, // < 10ms
    Good,      // 10-20ms
    Acceptable, // 20-50ms
    Poor,      // 50-100ms
    Critical,  // > 100ms
}

impl LatencyOptimizer {
    /// Create new latency optimizer
    pub fn new(model_id: String) -> Self {
        let base_predictor = KpiPredictor::new(KpiType::Latency, model_id);
        
        Self {
            base_predictor,
            processing_model: ProcessingDelayModel {
                base_processing_delay: 2.0, // Base 2ms processing
                load_factor_impact: 0.02,   // 0.02ms per % load
                queue_depth_factor: 0.1,    // Queue impact
                modulation_complexity_factor: HashMap::from([
                    ("QPSK".to_string(), 1.0),
                    ("16QAM".to_string(), 1.1),
                    ("64QAM".to_string(), 1.3),
                    ("256QAM".to_string(), 1.5),
                ]),
                mimo_processing_overhead: 0.5, // Additional 0.5ms for MIMO
            },
            handover_latency_model: HandoverLatencyModel {
                measurement_reporting_delay: 5.0, // 5ms for measurements
                decision_delay: 3.0,              // 3ms for handover decision
                execution_delay: 8.0,             // 8ms for handover execution
                inter_freq_penalty: 10.0,         // 10ms additional for inter-freq
                failure_retry_penalty: 20.0,      // 20ms for failed handovers
            },
            qci_latency_requirements: HashMap::from([
                (1, 10.0),   // Voice - 10ms target
                (2, 15.0),   // Video call - 15ms
                (3, 25.0),   // Real-time gaming - 25ms
                (4, 30.0),   // Non-conversational video - 30ms
                (5, 50.0),   // IMS signaling - 50ms
                (6, 30.0),   // Voice, video, interactive gaming - 30ms
                (7, 100.0),  // Video, TCP - 100ms
                (8, 300.0),  // TCP - 300ms
                (9, 300.0),  // Background - 300ms
            ]),
            optimization_targets: LatencyTargets {
                voice_target_ms: 10.0,
                video_target_ms: 25.0,
                data_target_ms: 50.0,
                background_target_ms: 100.0,
                emergency_target_ms: 5.0,
            },
        }
    }
    
    /// Train latency model with historical data
    pub fn train(&mut self, training_data: &[(CsvKpiFeatures, f32)]) -> Result<(), String> {
        if training_data.is_empty() {
            return Err("Training data cannot be empty".to_string());
        }
        
        // Train base predictor
        self.base_predictor.train(training_data)?;
        
        // Calibrate component models based on training data
        self.calibrate_processing_model(training_data)?;
        self.calibrate_handover_model(training_data)?;
        
        Ok(())
    }
    
    /// Calibrate processing delay model
    fn calibrate_processing_model(&mut self, training_data: &[(CsvKpiFeatures, f32)]) -> Result<(), String> {
        // Analyze relationship between load and latency
        let mut load_latency_pairs = Vec::new();
        
        for (features, latency) in training_data {
            let load_ratio = features.rrc_connected_users_avg / 200.0; // Normalize
            load_latency_pairs.push((load_ratio, *latency));
        }
        
        // Simple linear regression to find load impact factor
        if !load_latency_pairs.is_empty() {
            let n = load_latency_pairs.len() as f32;
            let sum_x: f32 = load_latency_pairs.iter().map(|(x, _)| x).sum();
            let sum_y: f32 = load_latency_pairs.iter().map(|(_, y)| y).sum();
            let sum_xy: f32 = load_latency_pairs.iter().map(|(x, y)| x * y).sum();
            let sum_x2: f32 = load_latency_pairs.iter().map(|(x, _)| x * x).sum();
            
            let denominator = n * sum_x2 - sum_x * sum_x;
            if denominator != 0.0 {
                let slope = (n * sum_xy - sum_x * sum_y) / denominator;
                self.processing_model.load_factor_impact = slope.max(0.0).min(1.0);
            }
        }
        
        Ok(())
    }
    
    /// Calibrate handover latency model
    fn calibrate_handover_model(&mut self, training_data: &[(CsvKpiFeatures, f32)]) -> Result<(), String> {
        // Analyze handover success rate vs latency relationship
        let mut ho_latency_pairs = Vec::new();
        
        for (features, latency) in training_data {
            let avg_ho_sr = (features.lte_intra_freq_ho_sr + features.lte_inter_freq_ho_sr) / 2.0;
            ho_latency_pairs.push((avg_ho_sr, *latency));
        }
        
        // Higher handover failure rates should correlate with higher latency
        if !ho_latency_pairs.is_empty() {
            let avg_ho_sr: f32 = ho_latency_pairs.iter().map(|(ho, _)| ho).sum::<f32>() 
                / ho_latency_pairs.len() as f32;
            
            if avg_ho_sr < 95.0 {
                // Increase failure penalty for networks with poor handover performance
                self.handover_latency_model.failure_retry_penalty *= 1.5;
            }
        }
        
        Ok(())
    }
    
    /// Predict latency with detailed component analysis
    pub fn predict_latency(&self, features: &CsvKpiFeatures) -> Result<LatencyPrediction, String> {
        if !self.base_predictor.is_trained {
            return Err("Model must be trained before prediction".to_string());
        }
        
        // Get base prediction
        let base_prediction = self.base_predictor.predict(features)?;
        let predicted_latency = base_prediction.predicted_value;
        
        // Calculate latency components
        let latency_components = self.calculate_latency_components(features);
        
        // Calculate QCI-specific latencies
        let qci_specific_latencies = self.calculate_qci_latencies(features);
        
        // Identify optimization opportunities
        let optimization_opportunities = self.identify_optimization_opportunities(features, &latency_components);
        
        // Grade performance
        let performance_grade = self.grade_latency_performance(predicted_latency);
        
        Ok(LatencyPrediction {
            predicted_latency,
            latency_components,
            qci_specific_latencies,
            optimization_opportunities,
            performance_grade,
            confidence: base_prediction.confidence,
        })
    }
    
    /// Calculate detailed latency components
    fn calculate_latency_components(&self, features: &CsvKpiFeatures) -> LatencyComponents {
        // Air interface latency (based on SINR and modulation)
        let air_interface_latency = if features.sinr_pusch_avg > 20.0 {
            2.0 // Excellent conditions
        } else if features.sinr_pusch_avg > 15.0 {
            3.0 // Good conditions
        } else if features.sinr_pusch_avg > 10.0 {
            5.0 // Fair conditions
        } else {
            8.0 // Poor conditions
        };
        
        // Processing delay based on load
        let load_ratio = features.rrc_connected_users_avg / 200.0;
        let processing_delay = self.processing_model.base_processing_delay 
            + (load_ratio * self.processing_model.load_factor_impact * 100.0)
            + self.processing_model.mimo_processing_overhead;
        
        // Transport delay (simplified - would need transport network data)
        let transport_delay = 3.0; // Assumed 3ms transport delay
        
        // Handover delay based on handover performance
        let ho_success_rate = (features.lte_intra_freq_ho_sr + features.lte_inter_freq_ho_sr) / 2.0;
        let handover_delay = if ho_success_rate < 90.0 {
            self.handover_latency_model.failure_retry_penalty
        } else {
            self.handover_latency_model.measurement_reporting_delay 
                + self.handover_latency_model.decision_delay 
                + self.handover_latency_model.execution_delay
        };
        
        // Queuing delay based on traffic load
        let traffic_load = features.eric_traff_erab_erl / 50.0; // Normalize
        let queuing_delay = traffic_load * 2.0; // Max 2ms additional queuing
        
        let total_latency = air_interface_latency + processing_delay + transport_delay + handover_delay + queuing_delay;
        
        LatencyComponents {
            air_interface_latency,
            processing_delay,
            transport_delay,
            handover_delay,
            queuing_delay,
            total_latency,
        }
    }
    
    /// Calculate QCI-specific latency predictions
    fn calculate_qci_latencies(&self, features: &CsvKpiFeatures) -> HashMap<u8, f32> {
        let base_components = self.calculate_latency_components(features);
        let mut qci_latencies = HashMap::new();
        
        for (&qci, &target) in &self.qci_latency_requirements {
            // Adjust latency based on QCI priority and requirements
            let qci_adjustment = match qci {
                1 => 0.8, // Voice gets priority treatment
                2 => 0.9, // Video calls
                5 => 1.0, // IMS signaling
                8 => 1.2, // Best effort data
                9 => 1.5, // Background traffic
                _ => 1.0,
            };
            
            let adjusted_latency = base_components.total_latency * qci_adjustment;
            qci_latencies.insert(qci, adjusted_latency);
        }
        
        qci_latencies
    }
    
    /// Identify optimization opportunities
    fn identify_optimization_opportunities(
        &self, 
        features: &CsvKpiFeatures, 
        components: &LatencyComponents
    ) -> Vec<OptimizationOpportunity> {
        let mut opportunities = Vec::new();
        
        // Air interface optimization
        if components.air_interface_latency > 4.0 {
            opportunities.push(OptimizationOpportunity {
                component: "Air Interface".to_string(),
                current_value: components.air_interface_latency,
                target_value: 2.0,
                potential_improvement: components.air_interface_latency - 2.0,
                implementation_complexity: ComplexityLevel::Medium,
                recommendations: vec![
                    "Improve SINR through antenna optimization".to_string(),
                    "Reduce interference sources".to_string(),
                    "Optimize power control parameters".to_string(),
                ],
            });
        }
        
        // Processing delay optimization
        if components.processing_delay > 4.0 {
            opportunities.push(OptimizationOpportunity {
                component: "Processing Delay".to_string(),
                current_value: components.processing_delay,
                target_value: 2.0,
                potential_improvement: components.processing_delay - 2.0,
                implementation_complexity: ComplexityLevel::High,
                recommendations: vec![
                    "Upgrade baseband processing capacity".to_string(),
                    "Optimize scheduler algorithms".to_string(),
                    "Implement load balancing".to_string(),
                ],
            });
        }
        
        // Handover optimization
        if components.handover_delay > 15.0 {
            opportunities.push(OptimizationOpportunity {
                component: "Handover Performance".to_string(),
                current_value: components.handover_delay,
                target_value: 10.0,
                potential_improvement: components.handover_delay - 10.0,
                implementation_complexity: ComplexityLevel::Low,
                recommendations: vec![
                    "Optimize handover thresholds".to_string(),
                    "Reduce measurement reporting intervals".to_string(),
                    "Improve neighbor cell configuration".to_string(),
                ],
            });
        }
        
        // Queuing optimization
        if components.queuing_delay > 1.5 {
            opportunities.push(OptimizationOpportunity {
                component: "Queue Management".to_string(),
                current_value: components.queuing_delay,
                target_value: 0.5,
                potential_improvement: components.queuing_delay - 0.5,
                implementation_complexity: ComplexityLevel::Low,
                recommendations: vec![
                    "Implement priority queuing for latency-sensitive traffic".to_string(),
                    "Optimize buffer sizes".to_string(),
                    "Enable traffic shaping".to_string(),
                ],
            });
        }
        
        opportunities
    }
    
    /// Grade latency performance
    fn grade_latency_performance(&self, latency: f32) -> LatencyGrade {
        if latency < 10.0 {
            LatencyGrade::Excellent
        } else if latency < 20.0 {
            LatencyGrade::Good
        } else if latency < 50.0 {
            LatencyGrade::Acceptable
        } else if latency < 100.0 {
            LatencyGrade::Poor
        } else {
            LatencyGrade::Critical
        }
    }
    
    /// Predict optimal handover thresholds for latency minimization
    pub fn predict_optimal_handover_thresholds(&self, features: &CsvKpiFeatures) -> Result<HashMap<String, f32>, String> {
        let mut thresholds = HashMap::new();
        
        // A3 threshold for handover trigger (RSRP based)
        let a3_threshold = if features.sinr_pusch_avg > 15.0 {
            3.0 // Higher threshold in good conditions to avoid ping-pong
        } else {
            1.0 // Lower threshold in poor conditions for faster handover
        };
        thresholds.insert("a3_rsrp_threshold_db".to_string(), a3_threshold);
        
        // Time to trigger
        let time_to_trigger = if features.lte_intra_freq_ho_sr < 90.0 {
            160.0 // Longer TTT to avoid failed handovers
        } else {
            80.0 // Shorter TTT for faster handovers
        };
        thresholds.insert("time_to_trigger_ms".to_string(), time_to_trigger);
        
        // Hysteresis
        let hysteresis = if features.inter_freq_ho_attempts > features.intra_freq_ho_attempts {
            2.0 // Higher hysteresis to avoid unnecessary inter-freq handovers
        } else {
            1.0
        };
        thresholds.insert("hysteresis_db".to_string(), hysteresis);
        
        Ok(thresholds)
    }
    
    /// Predict latency optimization using handover patterns
    pub fn predict_handover_latency_optimization(&self, features: &CsvKpiFeatures) -> Result<f32, String> {
        let current_prediction = self.predict_latency(features)?;
        
        // Simulate optimized handover parameters
        let mut optimized_features = features.clone();
        
        // Improve handover success rates by 5%
        optimized_features.lte_intra_freq_ho_sr = (optimized_features.lte_intra_freq_ho_sr + 5.0).min(100.0);
        optimized_features.lte_inter_freq_ho_sr = (optimized_features.lte_inter_freq_ho_sr + 5.0).min(100.0);
        
        // Reduce handover attempts by optimizing thresholds
        optimized_features.inter_freq_ho_attempts *= 0.8;
        optimized_features.intra_freq_ho_attempts *= 0.9;
        
        let optimized_prediction = self.predict_latency(&optimized_features)?;
        
        // Return potential latency improvement
        Ok(current_prediction.predicted_latency - optimized_prediction.predicted_latency)
    }
    
    /// Generate comprehensive latency optimization plan
    pub fn generate_optimization_plan(&self, features: &CsvKpiFeatures) -> Result<LatencyOptimizationPlan, String> {
        let current_prediction = self.predict_latency(features)?;
        let handover_optimization = self.predict_handover_latency_optimization(features)?;
        let optimal_thresholds = self.predict_optimal_handover_thresholds(features)?;
        
        Ok(LatencyOptimizationPlan {
            current_latency: current_prediction.predicted_latency,
            target_latency: self.optimization_targets.voice_target_ms, // Use most stringent target
            optimization_opportunities: current_prediction.optimization_opportunities,
            handover_optimization_potential: handover_optimization,
            recommended_thresholds: optimal_thresholds,
            implementation_priority: self.prioritize_optimizations(&current_prediction.optimization_opportunities),
            expected_improvement: current_prediction.optimization_opportunities.iter()
                .map(|opp| opp.potential_improvement)
                .sum(),
        })
    }
    
    /// Prioritize optimization opportunities
    fn prioritize_optimizations(&self, opportunities: &[OptimizationOpportunity]) -> Vec<String> {
        let mut prioritized = opportunities.to_vec();
        
        // Sort by potential improvement and implementation complexity
        prioritized.sort_by(|a, b| {
            let a_score = a.potential_improvement / match a.implementation_complexity {
                ComplexityLevel::Low => 1.0,
                ComplexityLevel::Medium => 2.0,
                ComplexityLevel::High => 4.0,
            };
            let b_score = b.potential_improvement / match b.implementation_complexity {
                ComplexityLevel::Low => 1.0,
                ComplexityLevel::Medium => 2.0,
                ComplexityLevel::High => 4.0,
            };
            b_score.partial_cmp(&a_score).unwrap()
        });
        
        prioritized.into_iter().map(|opp| opp.component).collect()
    }
}

/// Complete latency optimization plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyOptimizationPlan {
    pub current_latency: f32,
    pub target_latency: f32,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub handover_optimization_potential: f32,
    pub recommended_thresholds: HashMap<String, f32>,
    pub implementation_priority: Vec<String>,
    pub expected_improvement: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_latency_optimizer_creation() {
        let optimizer = LatencyOptimizer::new("test_latency_optimizer".to_string());
        assert_eq!(optimizer.base_predictor.model_id, "test_latency_optimizer");
        assert_eq!(optimizer.base_predictor.kpi_type, KpiType::Latency);
    }
    
    #[test]
    fn test_latency_components_calculation() {
        let optimizer = LatencyOptimizer::new("test".to_string());
        let features = create_test_features();
        
        let components = optimizer.calculate_latency_components(&features);
        assert!(components.total_latency > 0.0);
        assert!(components.air_interface_latency > 0.0);
        assert!(components.processing_delay > 0.0);
    }
    
    #[test]
    fn test_latency_grading() {
        let optimizer = LatencyOptimizer::new("test".to_string());
        
        assert!(matches!(optimizer.grade_latency_performance(5.0), LatencyGrade::Excellent));
        assert!(matches!(optimizer.grade_latency_performance(15.0), LatencyGrade::Good));
        assert!(matches!(optimizer.grade_latency_performance(30.0), LatencyGrade::Acceptable));
        assert!(matches!(optimizer.grade_latency_performance(75.0), LatencyGrade::Poor));
        assert!(matches!(optimizer.grade_latency_performance(150.0), LatencyGrade::Critical));
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