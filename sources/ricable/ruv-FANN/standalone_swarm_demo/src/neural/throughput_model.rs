//! Throughput Prediction Model
//! 
//! Specialized neural network for predicting cell throughput based on signal quality,
//! traffic load, and network conditions from real CSV data.

use crate::neural::kpi_predictor::{CsvKpiFeatures, KpiPredictor, KpiType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Throughput prediction model with advanced signal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputModel {
    pub base_predictor: KpiPredictor,
    pub shannon_capacity_weights: Vec<f32>,
    pub traffic_load_model: TrafficLoadModel,
    pub mimo_efficiency_factor: f32,
    pub interference_model: InterferenceModel,
    pub modulation_scheme_impact: HashMap<String, f32>,
}

/// Traffic load modeling for throughput prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficLoadModel {
    pub saturation_threshold: f32,
    pub load_efficiency_curve: Vec<(f32, f32)>, // (load_ratio, efficiency)
    pub user_scaling_factor: f32,
    pub qci_priority_weights: HashMap<u8, f32>,
}

/// Interference modeling for realistic throughput prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceModel {
    pub thermal_noise_floor: f32,
    pub interference_margin: f32,
    pub adjacent_channel_leakage: f32,
    pub co_channel_interference: f32,
}

/// Throughput prediction with detailed breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputPrediction {
    pub predicted_dl_throughput: f32,
    pub predicted_ul_throughput: f32,
    pub theoretical_max_throughput: f32,
    pub efficiency_ratio: f32,
    pub limiting_factors: Vec<LimitingFactor>,
    pub optimization_suggestions: Vec<String>,
    pub confidence: f32,
    pub prediction_breakdown: ThroughputBreakdown,
}

/// Factor limiting throughput performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitingFactor {
    pub factor_name: String,
    pub impact_percentage: f32,
    pub current_value: f32,
    pub optimal_range: (f32, f32),
    pub severity: LimitingSeverity,
}

/// Severity of limiting factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitingSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Detailed breakdown of throughput prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputBreakdown {
    pub shannon_capacity: f32,
    pub modulation_efficiency: f32,
    pub mimo_gain: f32,
    pub traffic_load_impact: f32,
    pub interference_impact: f32,
    pub protocol_overhead: f32,
    pub final_throughput: f32,
}

impl ThroughputModel {
    /// Create new throughput model
    pub fn new(model_id: String) -> Self {
        let base_predictor = KpiPredictor::new(KpiType::Throughput, model_id);
        
        Self {
            base_predictor,
            shannon_capacity_weights: vec![0.3, 0.4, 0.2, 0.1], // SINR, Bandwidth, Power, Interference
            traffic_load_model: TrafficLoadModel {
                saturation_threshold: 80.0,
                load_efficiency_curve: vec![
                    (0.0, 1.0),
                    (0.3, 0.98),
                    (0.5, 0.95),
                    (0.7, 0.85),
                    (0.8, 0.70),
                    (0.9, 0.50),
                    (1.0, 0.30),
                ],
                user_scaling_factor: 0.02,
                qci_priority_weights: HashMap::from([
                    (1, 1.0),  // Voice
                    (5, 0.8),  // Video
                    (8, 0.6),  // Best effort
                    (9, 0.4),  // Background
                ]),
            },
            mimo_efficiency_factor: 1.8, // Typical 2x2 MIMO gain
            interference_model: InterferenceModel {
                thermal_noise_floor: -104.0, // dBm
                interference_margin: 3.0,    // dB
                adjacent_channel_leakage: 0.1,
                co_channel_interference: 0.15,
            },
            modulation_scheme_impact: HashMap::from([
                ("QPSK".to_string(), 1.0),
                ("16QAM".to_string(), 1.5),
                ("64QAM".to_string(), 2.2),
                ("256QAM".to_string(), 3.0),
            ]),
        }
    }
    
    /// Train the throughput model with historical data
    pub fn train(&mut self, training_data: &[(CsvKpiFeatures, f32, f32)]) -> Result<(), String> {
        if training_data.is_empty() {
            return Err("Training data cannot be empty".to_string());
        }
        
        // Prepare training data for base predictor (using combined throughput)
        let mut base_training_data = Vec::new();
        for (features, dl_throughput, ul_throughput) in training_data {
            let combined_throughput = dl_throughput + ul_throughput;
            base_training_data.push((features.clone(), combined_throughput));
        }
        
        // Train base KPI predictor
        self.base_predictor.train(&base_training_data)?;
        
        // Calibrate advanced models based on training data
        self.calibrate_traffic_load_model(training_data)?;
        self.calibrate_interference_model(training_data)?;
        
        Ok(())
    }
    
    /// Calibrate traffic load model based on training data
    fn calibrate_traffic_load_model(&mut self, training_data: &[(CsvKpiFeatures, f32, f32)]) -> Result<(), String> {
        // Analyze relationship between user count and throughput efficiency
        let mut load_efficiency_points = Vec::new();
        
        for (features, dl_throughput, _ul_throughput) in training_data {
            let theoretical_capacity = self.calculate_theoretical_capacity(features);
            if theoretical_capacity > 0.0 {
                let efficiency = dl_throughput / theoretical_capacity;
                let load_ratio = features.rrc_connected_users_avg / 200.0; // Normalize to typical max
                load_efficiency_points.push((load_ratio.min(1.0), efficiency.min(1.0)));
            }
        }
        
        // Update efficiency curve with real data (simplified)
        if !load_efficiency_points.is_empty() {
            load_efficiency_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            self.traffic_load_model.load_efficiency_curve = load_efficiency_points;
        }
        
        Ok(())
    }
    
    /// Calibrate interference model based on training data
    fn calibrate_interference_model(&mut self, training_data: &[(CsvKpiFeatures, f32, f32)]) -> Result<(), String> {
        // Analyze SINR vs throughput relationship to estimate interference impact
        let mut sinr_throughput_pairs = Vec::new();
        
        for (features, dl_throughput, _ul_throughput) in training_data {
            sinr_throughput_pairs.push((features.sinr_pusch_avg, *dl_throughput));
        }
        
        // Calculate average interference impact (simplified)
        if !sinr_throughput_pairs.is_empty() {
            let avg_sinr: f32 = sinr_throughput_pairs.iter().map(|(sinr, _)| sinr).sum::<f32>() 
                / sinr_throughput_pairs.len() as f32;
            
            // Adjust interference margin based on SINR distribution
            if avg_sinr < 10.0 {
                self.interference_model.interference_margin = 5.0; // High interference environment
            } else if avg_sinr > 20.0 {
                self.interference_model.interference_margin = 1.0; // Low interference
            }
        }
        
        Ok(())
    }
    
    /// Calculate theoretical Shannon capacity
    fn calculate_theoretical_capacity(&self, features: &CsvKpiFeatures) -> f32 {
        // Shannon capacity: C = B * log2(1 + SINR)
        let bandwidth_mhz = match features.frequency_band.as_str() {
            "LTE800" => 10.0,
            "LTE1800" => 15.0,
            "LTE2100" => 20.0,
            "LTE2600" => 20.0,
            _ => 15.0, // Default
        };
        
        let sinr_linear = 10.0_f32.powf(features.sinr_pusch_avg / 10.0);
        let capacity_mbps = bandwidth_mhz * (1.0 + sinr_linear).log2();
        
        capacity_mbps * self.mimo_efficiency_factor
    }
    
    /// Predict throughput with detailed analysis
    pub fn predict_throughput(&self, features: &CsvKpiFeatures) -> Result<ThroughputPrediction, String> {
        if !self.base_predictor.is_trained {
            return Err("Model must be trained before prediction".to_string());
        }
        
        // Get base prediction
        let base_prediction = self.base_predictor.predict(features)?;
        let base_throughput = base_prediction.predicted_value;
        
        // Calculate detailed throughput breakdown
        let breakdown = self.calculate_throughput_breakdown(features);
        
        // Split into DL/UL based on typical ratios
        let dl_ul_ratio = 3.0; // Typical 3:1 DL:UL ratio
        let predicted_dl_throughput = base_throughput * dl_ul_ratio / (dl_ul_ratio + 1.0);
        let predicted_ul_throughput = base_throughput / (dl_ul_ratio + 1.0);
        
        // Calculate theoretical maximum
        let theoretical_max_throughput = self.calculate_theoretical_capacity(features);
        let efficiency_ratio = if theoretical_max_throughput > 0.0 {
            base_throughput / theoretical_max_throughput
        } else {
            0.0
        };
        
        // Identify limiting factors
        let limiting_factors = self.identify_limiting_factors(features);
        
        // Generate optimization suggestions
        let optimization_suggestions = self.generate_optimization_suggestions(features, &limiting_factors);
        
        Ok(ThroughputPrediction {
            predicted_dl_throughput,
            predicted_ul_throughput,
            theoretical_max_throughput,
            efficiency_ratio,
            limiting_factors,
            optimization_suggestions,
            confidence: base_prediction.confidence,
            prediction_breakdown: breakdown,
        })
    }
    
    /// Calculate detailed throughput breakdown
    fn calculate_throughput_breakdown(&self, features: &CsvKpiFeatures) -> ThroughputBreakdown {
        let shannon_capacity = self.calculate_theoretical_capacity(features);
        
        // Estimate modulation efficiency based on SINR
        let modulation_efficiency = if features.sinr_pusch_avg > 20.0 {
            0.9 // Can use high-order modulation
        } else if features.sinr_pusch_avg > 15.0 {
            0.75
        } else if features.sinr_pusch_avg > 10.0 {
            0.6
        } else {
            0.4
        };
        
        // MIMO gain
        let mimo_gain = self.mimo_efficiency_factor;
        
        // Traffic load impact
        let load_ratio = features.rrc_connected_users_avg / 200.0;
        let traffic_load_impact = self.get_efficiency_for_load(load_ratio);
        
        // Interference impact based on error rates
        let interference_impact = if features.mac_dl_bler > 10.0 {
            0.6 // High interference
        } else if features.mac_dl_bler > 5.0 {
            0.8
        } else {
            0.95
        };
        
        // Protocol overhead (typical LTE overhead)
        let protocol_overhead = 0.85;
        
        let final_throughput = shannon_capacity 
            * modulation_efficiency 
            * traffic_load_impact 
            * interference_impact 
            * protocol_overhead;
        
        ThroughputBreakdown {
            shannon_capacity,
            modulation_efficiency,
            mimo_gain,
            traffic_load_impact,
            interference_impact,
            protocol_overhead,
            final_throughput,
        }
    }
    
    /// Get efficiency factor for given load ratio
    fn get_efficiency_for_load(&self, load_ratio: f32) -> f32 {
        // Interpolate efficiency from load curve
        for window in self.traffic_load_model.load_efficiency_curve.windows(2) {
            if let [point1, point2] = window {
                if load_ratio >= point1.0 && load_ratio <= point2.0 {
                    let ratio = (load_ratio - point1.0) / (point2.0 - point1.0);
                    return point1.1 + ratio * (point2.1 - point1.1);
                }
            }
        }
        
        // Default to last point if beyond curve
        self.traffic_load_model.load_efficiency_curve.last()
            .map(|(_, eff)| *eff)
            .unwrap_or(0.3)
    }
    
    /// Identify factors limiting throughput
    fn identify_limiting_factors(&self, features: &CsvKpiFeatures) -> Vec<LimitingFactor> {
        let mut factors = Vec::new();
        
        // Check SINR
        if features.sinr_pusch_avg < 15.0 {
            let severity = if features.sinr_pusch_avg < 5.0 {
                LimitingSeverity::Critical
            } else if features.sinr_pusch_avg < 10.0 {
                LimitingSeverity::High
            } else {
                LimitingSeverity::Medium
            };
            
            factors.push(LimitingFactor {
                factor_name: "Poor Signal Quality (SINR)".to_string(),
                impact_percentage: (15.0 - features.sinr_pusch_avg) * 5.0,
                current_value: features.sinr_pusch_avg,
                optimal_range: (15.0, 30.0),
                severity,
            });
        }
        
        // Check error rates
        if features.mac_dl_bler > 5.0 {
            let severity = if features.mac_dl_bler > 15.0 {
                LimitingSeverity::Critical
            } else if features.mac_dl_bler > 10.0 {
                LimitingSeverity::High
            } else {
                LimitingSeverity::Medium
            };
            
            factors.push(LimitingFactor {
                factor_name: "High Error Rate (BLER)".to_string(),
                impact_percentage: features.mac_dl_bler * 2.0,
                current_value: features.mac_dl_bler,
                optimal_range: (0.0, 3.0),
                severity,
            });
        }
        
        // Check user load
        if features.rrc_connected_users_avg > 150.0 {
            let severity = if features.rrc_connected_users_avg > 300.0 {
                LimitingSeverity::Critical
            } else if features.rrc_connected_users_avg > 200.0 {
                LimitingSeverity::High
            } else {
                LimitingSeverity::Medium
            };
            
            factors.push(LimitingFactor {
                factor_name: "High User Load".to_string(),
                impact_percentage: (features.rrc_connected_users_avg - 150.0) / 10.0,
                current_value: features.rrc_connected_users_avg,
                optimal_range: (0.0, 150.0),
                severity,
            });
        }
        
        factors
    }
    
    /// Generate optimization suggestions
    fn generate_optimization_suggestions(&self, features: &CsvKpiFeatures, limiting_factors: &[LimitingFactor]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        for factor in limiting_factors {
            match factor.factor_name.as_str() {
                name if name.contains("Signal Quality") => {
                    suggestions.push("Consider antenna tilt optimization".to_string());
                    suggestions.push("Check for interference sources".to_string());
                    suggestions.push("Evaluate power level adjustments".to_string());
                },
                name if name.contains("Error Rate") => {
                    suggestions.push("Optimize modulation and coding scheme".to_string());
                    suggestions.push("Check RF interference".to_string());
                    suggestions.push("Consider adaptive power control".to_string());
                },
                name if name.contains("User Load") => {
                    suggestions.push("Implement load balancing to neighboring cells".to_string());
                    suggestions.push("Consider cell splitting or additional carriers".to_string());
                    suggestions.push("Optimize scheduler algorithms".to_string());
                },
                _ => {}
            }
        }
        
        // Additional suggestions based on feature values
        if features.ave_4g_lte_dl_user_thrput < 20.0 {
            suggestions.push("Enable carrier aggregation if available".to_string());
            suggestions.push("Optimize MIMO configuration".to_string());
        }
        
        if features.endc_setup_sr < 80.0 && features.endc_establishment_att > 0.0 {
            suggestions.push("Improve 5G anchor configuration for ENDC".to_string());
        }
        
        suggestions.sort();
        suggestions.dedup();
        suggestions
    }
    
    /// Predict cell performance degradation
    pub fn predict_degradation_risk(&self, features: &CsvKpiFeatures) -> Result<f32, String> {
        let current_prediction = self.predict_throughput(features)?;
        
        // Calculate degradation risk based on multiple factors
        let mut risk_score = 0.0;
        
        // SINR degradation risk
        if features.sinr_pusch_avg < 10.0 {
            risk_score += 0.3;
        } else if features.sinr_pusch_avg < 15.0 {
            risk_score += 0.15;
        }
        
        // Error rate risk
        if features.mac_dl_bler > 10.0 {
            risk_score += 0.25;
        } else if features.mac_dl_bler > 5.0 {
            risk_score += 0.1;
        }
        
        // Load risk
        let load_ratio = features.rrc_connected_users_avg / 200.0;
        if load_ratio > 0.8 {
            risk_score += 0.2;
        } else if load_ratio > 0.6 {
            risk_score += 0.1;
        }
        
        // Handover performance risk
        if features.lte_intra_freq_ho_sr < 90.0 {
            risk_score += 0.15;
        }
        
        // Efficiency risk
        if current_prediction.efficiency_ratio < 0.5 {
            risk_score += 0.2;
        } else if current_prediction.efficiency_ratio < 0.7 {
            risk_score += 0.1;
        }
        
        Ok(risk_score.min(1.0))
    }
    
    /// Predict optimal resource allocation
    pub fn predict_optimal_allocation(&self, features: &CsvKpiFeatures) -> Result<HashMap<String, f32>, String> {
        let mut allocation = HashMap::new();
        
        // Power allocation based on SINR and interference
        let optimal_power = if features.sinr_pusch_avg < 10.0 {
            30.0 // Increase power for poor coverage
        } else if features.sinr_pusch_avg > 20.0 {
            15.0 // Reduce power in good coverage areas
        } else {
            20.0 // Standard power
        };
        allocation.insert("power_level".to_string(), optimal_power);
        
        // Bandwidth allocation based on traffic
        let traffic_ratio = features.eric_traff_erab_erl / 50.0; // Normalize
        let optimal_bandwidth = if traffic_ratio > 0.8 {
            40.0 // Full bandwidth for high traffic
        } else if traffic_ratio < 0.3 {
            20.0 // Reduced bandwidth for low traffic
        } else {
            30.0 // Standard bandwidth
        };
        allocation.insert("bandwidth_mhz".to_string(), optimal_bandwidth);
        
        // User scheduling weight based on QoS requirements
        let scheduling_weight = if features.volte_traffic_erl > 2.0 {
            1.2 // Higher priority for voice traffic
        } else {
            1.0
        };
        allocation.insert("scheduling_weight".to_string(), scheduling_weight);
        
        Ok(allocation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_throughput_model_creation() {
        let model = ThroughputModel::new("test_throughput_model".to_string());
        assert_eq!(model.base_predictor.model_id, "test_throughput_model");
        assert_eq!(model.base_predictor.kpi_type, KpiType::Throughput);
    }
    
    #[test]
    fn test_theoretical_capacity_calculation() {
        let model = ThroughputModel::new("test".to_string());
        let features = create_test_features();
        
        let capacity = model.calculate_theoretical_capacity(&features);
        assert!(capacity > 0.0);
    }
    
    #[test]
    fn test_limiting_factors_identification() {
        let model = ThroughputModel::new("test".to_string());
        let mut features = create_test_features();
        features.sinr_pusch_avg = 8.0; // Poor SINR
        features.mac_dl_bler = 12.0; // High error rate
        
        let factors = model.identify_limiting_factors(&features);
        assert!(!factors.is_empty());
        assert!(factors.iter().any(|f| f.factor_name.contains("Signal Quality")));
        assert!(factors.iter().any(|f| f.factor_name.contains("Error Rate")));
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