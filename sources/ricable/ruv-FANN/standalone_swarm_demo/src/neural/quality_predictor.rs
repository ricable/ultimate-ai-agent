//! Quality Prediction Model
//! 
//! Neural network for predicting signal quality metrics including SINR, BLER,
//! and overall quality indicators based on network conditions.

use crate::neural::kpi_predictor::{CsvKpiFeatures, KpiPredictor, KpiType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quality prediction model for signal metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPredictor {
    pub base_predictor: KpiPredictor,
    pub sinr_model: SinrPredictionModel,
    pub bler_model: BlerPredictionModel,
    pub interference_analyzer: InterferenceAnalyzer,
    pub quality_thresholds: QualityThresholds,
}

/// SINR prediction and modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinrPredictionModel {
    pub path_loss_model: PathLossModel,
    pub interference_model: InterferenceModel,
    pub power_control_model: PowerControlModel,
    pub fading_margin: f32,
}

/// BLER prediction model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlerPredictionModel {
    pub sinr_bler_curve: Vec<(f32, f32)>, // (SINR, BLER) pairs
    pub modulation_bler_impact: HashMap<String, f32>,
    pub interference_bler_factor: f32,
    pub retransmission_impact: f32,
}

/// Path loss modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathLossModel {
    pub frequency_factor: f32,
    pub distance_exponent: f32,
    pub shadowing_std: f32,
    pub building_penetration_loss: f32,
}

/// Interference analysis and modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceModel {
    pub co_channel_interference: f32,
    pub adjacent_channel_interference: f32,
    pub inter_system_interference: f32,
    pub noise_floor: f32,
}

/// Power control modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerControlModel {
    pub target_sinr: f32,
    pub step_size: f32,
    pub max_power: f32,
    pub min_power: f32,
}

/// Interference source analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceAnalyzer {
    pub dominant_interferers: Vec<InterfererSource>,
    pub interference_patterns: Vec<InterferencePattern>,
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

/// Quality thresholds for different services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub excellent_sinr: f32,
    pub good_sinr: f32,
    pub poor_sinr: f32,
    pub acceptable_bler: f32,
    pub poor_bler: f32,
    pub critical_bler: f32,
}

/// Quality prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPrediction {
    pub predicted_sinr: f32,
    pub predicted_bler: f32,
    pub quality_index: f32,
    pub quality_grade: QualityGrade,
    pub interference_analysis: InterferenceAnalysis,
    pub optimization_recommendations: Vec<String>,
    pub confidence: f32,
}

/// Interference source identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfererSource {
    pub source_type: InterferenceType,
    pub strength: f32,
    pub frequency_overlap: f32,
    pub mitigation_priority: Priority,
}

/// Types of interference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterferenceType {
    CoChannel,
    AdjacentChannel,
    IntermodulationDistortion,
    ExternalSystem,
    ThermalNoise,
}

/// Interference patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferencePattern {
    pub pattern_type: String,
    pub temporal_characteristics: TemporalPattern,
    pub spatial_characteristics: SpatialPattern,
    pub correlation_factor: f32,
}

/// Temporal interference patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalPattern {
    Constant,
    Periodic,
    Burst,
    Random,
}

/// Spatial interference patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialPattern {
    Localized,
    Distributed,
    Directional,
    Omnidirectional,
}

/// Mitigation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    pub strategy_name: String,
    pub effectiveness: f32,
    pub implementation_cost: Priority,
    pub expected_improvement: f32,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Quality grading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityGrade {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Interference analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceAnalysis {
    pub total_interference_level: f32,
    pub dominant_sources: Vec<InterfererSource>,
    pub interference_ratio: f32,
    pub quality_impact: f32,
}

impl QualityPredictor {
    /// Create new quality predictor
    pub fn new(model_id: String) -> Self {
        let base_predictor = KpiPredictor::new(KpiType::SignalQuality, model_id);
        
        Self {
            base_predictor,
            sinr_model: SinrPredictionModel {
                path_loss_model: PathLossModel {
                    frequency_factor: 0.1,
                    distance_exponent: 3.5,
                    shadowing_std: 8.0,
                    building_penetration_loss: 15.0,
                },
                interference_model: InterferenceModel {
                    co_channel_interference: -100.0, // dBm
                    adjacent_channel_interference: -110.0,
                    inter_system_interference: -115.0,
                    noise_floor: -104.0,
                },
                power_control_model: PowerControlModel {
                    target_sinr: 15.0,
                    step_size: 1.0,
                    max_power: 23.0,
                    min_power: -40.0,
                },
                fading_margin: 10.0,
            },
            bler_model: BlerPredictionModel {
                sinr_bler_curve: vec![
                    (-5.0, 50.0),
                    (0.0, 20.0),
                    (5.0, 8.0),
                    (10.0, 3.0),
                    (15.0, 1.0),
                    (20.0, 0.3),
                    (25.0, 0.1),
                ],
                modulation_bler_impact: HashMap::from([
                    ("QPSK".to_string(), 1.0),
                    ("16QAM".to_string(), 1.5),
                    ("64QAM".to_string(), 2.2),
                    ("256QAM".to_string(), 3.5),
                ]),
                interference_bler_factor: 1.2,
                retransmission_impact: 0.8,
            },
            interference_analyzer: InterferenceAnalyzer {
                dominant_interferers: Vec::new(),
                interference_patterns: Vec::new(),
                mitigation_strategies: Vec::new(),
            },
            quality_thresholds: QualityThresholds {
                excellent_sinr: 20.0,
                good_sinr: 15.0,
                poor_sinr: 5.0,
                acceptable_bler: 3.0,
                poor_bler: 10.0,
                critical_bler: 20.0,
            },
        }
    }
    
    /// Train quality prediction model
    pub fn train(&mut self, training_data: &[(CsvKpiFeatures, f32, f32)]) -> Result<(), String> {
        if training_data.is_empty() {
            return Err("Training data cannot be empty".to_string());
        }
        
        // Prepare training data for base predictor (using SINR as target)
        let mut base_training_data = Vec::new();
        for (features, sinr, _bler) in training_data {
            base_training_data.push((features.clone(), *sinr));
        }
        
        // Train base predictor
        self.base_predictor.train(&base_training_data)?;
        
        // Calibrate SINR-BLER curve from training data
        self.calibrate_bler_model(training_data)?;
        
        // Analyze interference patterns
        self.analyze_interference_patterns(training_data)?;
        
        Ok(())
    }
    
    /// Calibrate BLER model from training data
    fn calibrate_bler_model(&mut self, training_data: &[(CsvKpiFeatures, f32, f32)]) -> Result<(), String> {
        let mut sinr_bler_pairs = Vec::new();
        
        for (features, sinr, bler) in training_data {
            sinr_bler_pairs.push((*sinr, *bler));
        }
        
        // Sort by SINR for curve fitting
        sinr_bler_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Update SINR-BLER curve with real data points (simplified)
        if sinr_bler_pairs.len() > 5 {
            self.bler_model.sinr_bler_curve = sinr_bler_pairs;
        }
        
        Ok(())
    }
    
    /// Analyze interference patterns from training data
    fn analyze_interference_patterns(&mut self, training_data: &[(CsvKpiFeatures, f32, f32)]) -> Result<(), String> {
        // Analyze correlation between features and quality degradation
        let mut poor_quality_samples = Vec::new();
        
        for (features, sinr, bler) in training_data {
            if *sinr < 10.0 || *bler > 5.0 {
                poor_quality_samples.push(features);
            }
        }
        
        // Identify common patterns in poor quality samples
        if !poor_quality_samples.is_empty() {
            self.identify_interference_sources(&poor_quality_samples);
        }
        
        Ok(())
    }
    
    /// Identify interference sources from poor quality samples
    fn identify_interference_sources(&mut self, poor_samples: &[&CsvKpiFeatures]) {
        let mut interferer_sources = Vec::new();
        
        // Check for high RSSI with poor SINR (indicates interference)
        let high_rssi_poor_sinr_count = poor_samples.iter()
            .filter(|sample| sample.ul_rssi_total > -90.0 && sample.sinr_pusch_avg < 10.0)
            .count();
        
        if high_rssi_poor_sinr_count > poor_samples.len() / 3 {
            interferer_sources.push(InterfererSource {
                source_type: InterferenceType::CoChannel,
                strength: 0.7,
                frequency_overlap: 0.8,
                mitigation_priority: Priority::High,
            });
        }
        
        // Check for high error rates with decent SINR (indicates other issues)
        let high_bler_good_sinr_count = poor_samples.iter()
            .filter(|sample| sample.mac_dl_bler > 10.0 && sample.sinr_pusch_avg > 10.0)
            .count();
        
        if high_bler_good_sinr_count > poor_samples.len() / 4 {
            interferer_sources.push(InterfererSource {
                source_type: InterferenceType::IntermodulationDistortion,
                strength: 0.5,
                frequency_overlap: 0.3,
                mitigation_priority: Priority::Medium,
            });
        }
        
        self.interference_analyzer.dominant_interferers = interferer_sources;
    }
    
    /// Predict signal quality with detailed analysis
    pub fn predict_quality(&self, features: &CsvKpiFeatures) -> Result<QualityPrediction, String> {
        if !self.base_predictor.is_trained {
            return Err("Model must be trained before prediction".to_string());
        }
        
        // Get base SINR prediction
        let base_prediction = self.base_predictor.predict(features)?;
        let predicted_sinr = base_prediction.predicted_value;
        
        // Predict BLER based on SINR
        let predicted_bler = self.predict_bler_from_sinr(predicted_sinr, features);
        
        // Calculate quality index (0-100)
        let quality_index = self.calculate_quality_index(predicted_sinr, predicted_bler);
        
        // Grade quality
        let quality_grade = self.grade_quality(quality_index);
        
        // Analyze interference
        let interference_analysis = self.analyze_current_interference(features);
        
        // Generate optimization recommendations
        let optimization_recommendations = self.generate_quality_optimization_recommendations(
            features, predicted_sinr, predicted_bler, &interference_analysis
        );
        
        Ok(QualityPrediction {
            predicted_sinr,
            predicted_bler,
            quality_index,
            quality_grade,
            interference_analysis,
            optimization_recommendations,
            confidence: base_prediction.confidence,
        })
    }
    
    /// Predict BLER from SINR using calibrated curve
    fn predict_bler_from_sinr(&self, sinr: f32, features: &CsvKpiFeatures) -> f32 {
        // Interpolate BLER from SINR-BLER curve
        let mut bler = 0.0;
        
        for window in self.bler_model.sinr_bler_curve.windows(2) {
            if let [point1, point2] = window {
                if sinr >= point1.0 && sinr <= point2.0 {
                    let ratio = (sinr - point1.0) / (point2.0 - point1.0);
                    bler = point1.1 + ratio * (point2.1 - point1.1);
                    break;
                }
            }
        }
        
        // If SINR is outside curve range, use edge values
        if sinr < self.bler_model.sinr_bler_curve.first().unwrap().0 {
            bler = self.bler_model.sinr_bler_curve.first().unwrap().1;
        } else if sinr > self.bler_model.sinr_bler_curve.last().unwrap().0 {
            bler = self.bler_model.sinr_bler_curve.last().unwrap().1;
        }
        
        // Apply modulation impact (if we can infer modulation from conditions)
        let modulation_factor = if sinr > 20.0 {
            self.bler_model.modulation_bler_impact.get("64QAM").copied().unwrap_or(1.0)
        } else if sinr > 15.0 {
            self.bler_model.modulation_bler_impact.get("16QAM").copied().unwrap_or(1.0)
        } else {
            self.bler_model.modulation_bler_impact.get("QPSK").copied().unwrap_or(1.0)
        };
        
        // Apply interference factor
        let interference_factor = if features.ul_rssi_total > -90.0 && sinr < 15.0 {
            self.bler_model.interference_bler_factor
        } else {
            1.0
        };
        
        bler * modulation_factor * interference_factor
    }
    
    /// Calculate overall quality index
    fn calculate_quality_index(&self, sinr: f32, bler: f32) -> f32 {
        // Normalize SINR (0-30 dB range to 0-100 scale)
        let sinr_score = ((sinr + 5.0) / 35.0 * 100.0).clamp(0.0, 100.0);
        
        // Normalize BLER (0-20% range to 100-0 scale, inverted)
        let bler_score = ((20.0 - bler) / 20.0 * 100.0).clamp(0.0, 100.0);
        
        // Weighted combination (SINR 60%, BLER 40%)
        sinr_score * 0.6 + bler_score * 0.4
    }
    
    /// Grade quality performance
    fn grade_quality(&self, quality_index: f32) -> QualityGrade {
        if quality_index >= 85.0 {
            QualityGrade::Excellent
        } else if quality_index >= 70.0 {
            QualityGrade::Good
        } else if quality_index >= 50.0 {
            QualityGrade::Fair
        } else if quality_index >= 30.0 {
            QualityGrade::Poor
        } else {
            QualityGrade::Critical
        }
    }
    
    /// Analyze current interference conditions
    fn analyze_current_interference(&self, features: &CsvKpiFeatures) -> InterferenceAnalysis {
        let mut dominant_sources = Vec::new();
        
        // Check for co-channel interference indicators
        if features.ul_rssi_total > -95.0 && features.sinr_pusch_avg < 10.0 {
            dominant_sources.push(InterfererSource {
                source_type: InterferenceType::CoChannel,
                strength: ((-85.0 - features.ul_rssi_total) / -10.0).clamp(0.0, 1.0),
                frequency_overlap: 0.9,
                mitigation_priority: Priority::High,
            });
        }
        
        // Check for thermal noise floor issues
        if features.ul_rssi_total < -110.0 {
            dominant_sources.push(InterfererSource {
                source_type: InterferenceType::ThermalNoise,
                strength: ((-120.0 - features.ul_rssi_total) / -10.0).clamp(0.0, 1.0),
                frequency_overlap: 1.0,
                mitigation_priority: Priority::Medium,
            });
        }
        
        // Calculate total interference level
        let total_interference_level = dominant_sources.iter()
            .map(|source| source.strength)
            .sum::<f32>() / dominant_sources.len().max(1) as f32;
        
        // Calculate interference ratio (I/N)
        let signal_power = features.ul_rssi_total;
        let noise_floor = self.sinr_model.interference_model.noise_floor;
        let interference_ratio = if signal_power > noise_floor {
            (signal_power - noise_floor).abs() / 10.0 // Rough I/N calculation
        } else {
            0.0
        };
        
        // Calculate quality impact
        let quality_impact = if features.sinr_pusch_avg < 10.0 {
            (10.0 - features.sinr_pusch_avg) / 10.0
        } else {
            0.0
        };
        
        InterferenceAnalysis {
            total_interference_level,
            dominant_sources,
            interference_ratio,
            quality_impact,
        }
    }
    
    /// Generate quality optimization recommendations
    fn generate_quality_optimization_recommendations(
        &self,
        features: &CsvKpiFeatures,
        predicted_sinr: f32,
        predicted_bler: f32,
        interference_analysis: &InterferenceAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // SINR-based recommendations
        if predicted_sinr < self.quality_thresholds.good_sinr {
            recommendations.push("Consider increasing transmit power".to_string());
            recommendations.push("Optimize antenna tilt and azimuth".to_string());
            recommendations.push("Check for physical obstructions".to_string());
        }
        
        // BLER-based recommendations
        if predicted_bler > self.quality_thresholds.acceptable_bler {
            recommendations.push("Implement adaptive modulation and coding".to_string());
            recommendations.push("Enable HARQ retransmissions".to_string());
            recommendations.push("Optimize scheduler algorithms".to_string());
        }
        
        // Interference-specific recommendations
        for source in &interference_analysis.dominant_sources {
            match source.source_type {
                InterferenceType::CoChannel => {
                    recommendations.push("Implement frequency reuse optimization".to_string());
                    recommendations.push("Consider cell splitting or sectorization".to_string());
                },
                InterferenceType::AdjacentChannel => {
                    recommendations.push("Improve transmitter filtering".to_string());
                    recommendations.push("Adjust carrier spacing".to_string());
                },
                InterferenceType::ThermalNoise => {
                    recommendations.push("Check receiver sensitivity".to_string());
                    recommendations.push("Implement noise reduction techniques".to_string());
                },
                _ => {}
            }
        }
        
        // Load-based recommendations
        if features.rrc_connected_users_avg > 100.0 && predicted_sinr < 15.0 {
            recommendations.push("Implement load balancing to reduce interference".to_string());
            recommendations.push("Consider additional carrier deployment".to_string());
        }
        
        recommendations.sort();
        recommendations.dedup();
        recommendations
    }
    
    /// Predict BLER packet loss probability
    pub fn predict_packet_loss_probability(&self, features: &CsvKpiFeatures) -> Result<f32, String> {
        let quality_prediction = self.predict_quality(features)?;
        
        // Convert BLER to packet loss probability
        // BLER is block error rate, packet loss is typically related but not identical
        let packet_loss_probability = quality_prediction.predicted_bler / 100.0;
        
        // Adjust for retransmission mechanisms
        let effective_packet_loss = packet_loss_probability * self.bler_model.retransmission_impact;
        
        Ok(effective_packet_loss.min(1.0))
    }
    
    /// Predict optimal resource allocation for quality improvement
    pub fn predict_optimal_quality_allocation(&self, features: &CsvKpiFeatures) -> Result<HashMap<String, f32>, String> {
        let quality_prediction = self.predict_quality(features)?;
        let mut allocation = HashMap::new();
        
        // Power allocation based on SINR prediction
        let optimal_power = if quality_prediction.predicted_sinr < 10.0 {
            30.0 // Increase power for poor SINR
        } else if quality_prediction.predicted_sinr > 25.0 {
            15.0 // Reduce power in excellent conditions
        } else {
            20.0 // Standard power
        };
        allocation.insert("tx_power_dbm".to_string(), optimal_power);
        
        // Antenna tilt optimization
        let optimal_tilt = if quality_prediction.predicted_sinr < 12.0 {
            -2.0 // Downtilt to improve coverage
        } else if quality_prediction.predicted_sinr > 20.0 {
            2.0 // Uptilt to reduce interference
        } else {
            0.0 // Standard tilt
        };
        allocation.insert("antenna_tilt_degrees".to_string(), optimal_tilt);
        
        // Modulation and coding scheme
        let optimal_mcs = if quality_prediction.predicted_sinr > 20.0 {
            20.0 // High MCS for good conditions
        } else if quality_prediction.predicted_sinr > 15.0 {
            15.0 // Medium MCS
        } else {
            8.0 // Conservative MCS for poor conditions
        };
        allocation.insert("mcs_index".to_string(), optimal_mcs);
        
        Ok(allocation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quality_predictor_creation() {
        let predictor = QualityPredictor::new("test_quality_predictor".to_string());
        assert_eq!(predictor.base_predictor.model_id, "test_quality_predictor");
        assert_eq!(predictor.base_predictor.kpi_type, KpiType::SignalQuality);
    }
    
    #[test]
    fn test_bler_prediction() {
        let predictor = QualityPredictor::new("test".to_string());
        let features = create_test_features();
        
        let bler = predictor.predict_bler_from_sinr(15.0, &features);
        assert!(bler >= 0.0 && bler <= 100.0);
    }
    
    #[test]
    fn test_quality_grading() {
        let predictor = QualityPredictor::new("test".to_string());
        
        assert!(matches!(predictor.grade_quality(90.0), QualityGrade::Excellent));
        assert!(matches!(predictor.grade_quality(75.0), QualityGrade::Good));
        assert!(matches!(predictor.grade_quality(60.0), QualityGrade::Fair));
        assert!(matches!(predictor.grade_quality(40.0), QualityGrade::Poor));
        assert!(matches!(predictor.grade_quality(20.0), QualityGrade::Critical));
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