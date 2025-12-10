//! Feature Engineering Module
//! 
//! Advanced feature engineering for network KPI prediction, including
//! statistical features, temporal patterns, and domain-specific transformations.

use crate::neural::kpi_predictor::CsvKpiFeatures;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Feature engineering pipeline for network data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineering {
    pub statistical_features: StatisticalFeatures,
    pub temporal_features: TemporalFeatures,
    pub ratio_features: RatioFeatures,
    pub categorical_encoders: CategoricalEncoders,
    pub feature_scalers: FeatureScalers,
    pub history_buffer: VecDeque<CsvKpiFeatures>,
    pub buffer_size: usize,
}

/// Statistical feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFeatures {
    pub rolling_means: HashMap<String, f32>,
    pub rolling_stds: HashMap<String, f32>,
    pub rolling_mins: HashMap<String, f32>,
    pub rolling_maxs: HashMap<String, f32>,
    pub percentiles: HashMap<String, (f32, f32, f32)>, // P25, P50, P75
    pub z_scores: HashMap<String, f32>,
}

/// Temporal pattern features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFeatures {
    pub trends: HashMap<String, f32>,
    pub seasonality_indicators: HashMap<String, f32>,
    pub change_rates: HashMap<String, f32>,
    pub volatility_measures: HashMap<String, f32>,
    pub autocorrelations: HashMap<String, f32>,
}

/// Ratio and derived features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatioFeatures {
    pub efficiency_ratios: HashMap<String, f32>,
    pub performance_indices: HashMap<String, f32>,
    pub load_indicators: HashMap<String, f32>,
    pub quality_metrics: HashMap<String, f32>,
}

/// Categorical variable encoders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalEncoders {
    pub frequency_band_encoder: HashMap<String, f32>,
    pub one_hot_encodings: HashMap<String, Vec<f32>>,
    pub target_encodings: HashMap<String, f32>,
}

/// Feature scaling and normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureScalers {
    pub min_max_scalers: HashMap<String, (f32, f32)>, // (min, max)
    pub standard_scalers: HashMap<String, (f32, f32)>, // (mean, std)
    pub robust_scalers: HashMap<String, (f32, f32)>,   // (median, iqr)
}

/// Engineered feature vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineeredFeatures {
    pub raw_features: Vec<f32>,
    pub statistical_features: Vec<f32>,
    pub temporal_features: Vec<f32>,
    pub ratio_features: Vec<f32>,
    pub categorical_features: Vec<f32>,
    pub combined_features: Vec<f32>,
    pub feature_names: Vec<String>,
}

impl FeatureEngineering {
    /// Create new feature engineering pipeline
    pub fn new(buffer_size: usize) -> Self {
        Self {
            statistical_features: StatisticalFeatures {
                rolling_means: HashMap::new(),
                rolling_stds: HashMap::new(),
                rolling_mins: HashMap::new(),
                rolling_maxs: HashMap::new(),
                percentiles: HashMap::new(),
                z_scores: HashMap::new(),
            },
            temporal_features: TemporalFeatures {
                trends: HashMap::new(),
                seasonality_indicators: HashMap::new(),
                change_rates: HashMap::new(),
                volatility_measures: HashMap::new(),
                autocorrelations: HashMap::new(),
            },
            ratio_features: RatioFeatures {
                efficiency_ratios: HashMap::new(),
                performance_indices: HashMap::new(),
                load_indicators: HashMap::new(),
                quality_metrics: HashMap::new(),
            },
            categorical_encoders: CategoricalEncoders {
                frequency_band_encoder: HashMap::from([
                    ("LTE800".to_string(), 0.2),
                    ("LTE1800".to_string(), 0.4),
                    ("LTE2100".to_string(), 0.6),
                    ("LTE2600".to_string(), 0.8),
                ]),
                one_hot_encodings: HashMap::new(),
                target_encodings: HashMap::new(),
            },
            feature_scalers: FeatureScalers {
                min_max_scalers: HashMap::new(),
                standard_scalers: HashMap::new(),
                robust_scalers: HashMap::new(),
            },
            history_buffer: VecDeque::new(),
            buffer_size,
        }
    }
    
    /// Process new data point and extract engineered features
    pub fn engineer_features(&mut self, features: &CsvKpiFeatures) -> EngineeredFeatures {
        // Add to history buffer
        self.add_to_history(features.clone());
        
        // Extract raw features
        let raw_features = self.extract_raw_features(features);
        
        // Calculate statistical features
        let statistical_features = self.calculate_statistical_features(features);
        
        // Calculate temporal features
        let temporal_features = self.calculate_temporal_features(features);
        
        // Calculate ratio features
        let ratio_features = self.calculate_ratio_features(features);
        
        // Encode categorical features
        let categorical_features = self.encode_categorical_features(features);
        
        // Combine all features
        let mut combined_features = Vec::new();
        combined_features.extend(&raw_features);
        combined_features.extend(&statistical_features);
        combined_features.extend(&temporal_features);
        combined_features.extend(&ratio_features);
        combined_features.extend(&categorical_features);
        
        // Generate feature names
        let feature_names = self.generate_feature_names();
        
        EngineeredFeatures {
            raw_features,
            statistical_features,
            temporal_features,
            ratio_features,
            categorical_features,
            combined_features,
            feature_names,
        }
    }
    
    /// Add data point to history buffer
    fn add_to_history(&mut self, features: CsvKpiFeatures) {
        self.history_buffer.push_back(features);
        
        // Maintain buffer size
        while self.history_buffer.len() > self.buffer_size {
            self.history_buffer.pop_front();
        }
    }
    
    /// Extract raw features from CSV data
    fn extract_raw_features(&self, features: &CsvKpiFeatures) -> Vec<f32> {
        vec![
            features.sinr_pusch_avg,
            features.sinr_pucch_avg,
            features.ul_rssi_total,
            features.mac_dl_bler,
            features.mac_ul_bler,
            features.rrc_connected_users_avg,
            features.ul_volume_pdcp_gbytes,
            features.dl_volume_pdcp_gbytes,
            features.volte_traffic_erl,
            features.eric_traff_erab_erl,
            features.ave_4g_lte_dl_user_thrput,
            features.ave_4g_lte_ul_user_thrput,
            features.dl_latency_avg,
            features.erab_drop_rate_qci_5,
            features.lte_dcr_volte,
            features.lte_intra_freq_ho_sr,
            features.lte_inter_freq_ho_sr,
            features.inter_freq_ho_attempts,
            features.intra_freq_ho_attempts,
            features.endc_establishment_att,
            features.endc_establishment_succ,
            features.endc_setup_sr,
            features.endc_scg_failure_ratio,
            features.band_count,
            features.active_ues_dl,
            features.active_ues_ul,
        ]
    }
    
    /// Calculate statistical features from history
    fn calculate_statistical_features(&mut self, _current: &CsvKpiFeatures) -> Vec<f32> {
        let mut stats_features = Vec::new();
        
        if self.history_buffer.len() < 2 {
            // Return zeros if insufficient history
            return vec![0.0; 20]; // Placeholder for statistical features
        }
        
        // Calculate rolling statistics for key metrics
        let sinr_values: Vec<f32> = self.history_buffer.iter()
            .map(|f| f.sinr_pusch_avg)
            .collect();
        
        let throughput_values: Vec<f32> = self.history_buffer.iter()
            .map(|f| f.ave_4g_lte_dl_user_thrput)
            .collect();
        
        let bler_values: Vec<f32> = self.history_buffer.iter()
            .map(|f| f.mac_dl_bler)
            .collect();
        
        let latency_values: Vec<f32> = self.history_buffer.iter()
            .map(|f| f.dl_latency_avg)
            .collect();
        
        // SINR statistics
        stats_features.push(Self::calculate_mean(&sinr_values));
        stats_features.push(Self::calculate_std(&sinr_values));
        stats_features.push(Self::calculate_min(&sinr_values));
        stats_features.push(Self::calculate_max(&sinr_values));
        stats_features.push(Self::calculate_percentile(&sinr_values, 0.5)); // Median
        
        // Throughput statistics
        stats_features.push(Self::calculate_mean(&throughput_values));
        stats_features.push(Self::calculate_std(&throughput_values));
        stats_features.push(Self::calculate_min(&throughput_values));
        stats_features.push(Self::calculate_max(&throughput_values));
        stats_features.push(Self::calculate_percentile(&throughput_values, 0.5));
        
        // BLER statistics
        stats_features.push(Self::calculate_mean(&bler_values));
        stats_features.push(Self::calculate_std(&bler_values));
        stats_features.push(Self::calculate_min(&bler_values));
        stats_features.push(Self::calculate_max(&bler_values));
        stats_features.push(Self::calculate_percentile(&bler_values, 0.5));
        
        // Latency statistics
        stats_features.push(Self::calculate_mean(&latency_values));
        stats_features.push(Self::calculate_std(&latency_values));
        stats_features.push(Self::calculate_min(&latency_values));
        stats_features.push(Self::calculate_max(&latency_values));
        stats_features.push(Self::calculate_percentile(&latency_values, 0.5));
        
        // Update internal state
        self.statistical_features.rolling_means.insert("sinr".to_string(), stats_features[0]);
        self.statistical_features.rolling_stds.insert("sinr".to_string(), stats_features[1]);
        
        stats_features
    }
    
    /// Calculate temporal pattern features
    fn calculate_temporal_features(&mut self, current: &CsvKpiFeatures) -> Vec<f32> {
        let mut temporal_features = Vec::new();
        
        if self.history_buffer.len() < 3 {
            return vec![0.0; 15]; // Placeholder for temporal features
        }
        
        // Calculate trends (linear regression slope)
        let sinr_trend = self.calculate_trend("sinr");
        let throughput_trend = self.calculate_trend("throughput");
        let bler_trend = self.calculate_trend("bler");
        
        temporal_features.push(sinr_trend);
        temporal_features.push(throughput_trend);
        temporal_features.push(bler_trend);
        
        // Calculate change rates (current vs previous)
        if let Some(previous) = self.history_buffer.get(self.history_buffer.len() - 2) {
            let sinr_change_rate = (current.sinr_pusch_avg - previous.sinr_pusch_avg) / previous.sinr_pusch_avg.abs().max(0.1);
            let throughput_change_rate = (current.ave_4g_lte_dl_user_thrput - previous.ave_4g_lte_dl_user_thrput) / previous.ave_4g_lte_dl_user_thrput.abs().max(0.1);
            let bler_change_rate = (current.mac_dl_bler - previous.mac_dl_bler) / previous.mac_dl_bler.abs().max(0.1);
            
            temporal_features.push(sinr_change_rate);
            temporal_features.push(throughput_change_rate);
            temporal_features.push(bler_change_rate);
        } else {
            temporal_features.extend(vec![0.0; 3]);
        }
        
        // Calculate volatility measures (coefficient of variation)
        let sinr_values: Vec<f32> = self.history_buffer.iter().map(|f| f.sinr_pusch_avg).collect();
        let throughput_values: Vec<f32> = self.history_buffer.iter().map(|f| f.ave_4g_lte_dl_user_thrput).collect();
        let bler_values: Vec<f32> = self.history_buffer.iter().map(|f| f.mac_dl_bler).collect();
        
        temporal_features.push(Self::calculate_coefficient_of_variation(&sinr_values));
        temporal_features.push(Self::calculate_coefficient_of_variation(&throughput_values));
        temporal_features.push(Self::calculate_coefficient_of_variation(&bler_values));
        
        // Calculate autocorrelations (lag-1)
        temporal_features.push(Self::calculate_autocorrelation(&sinr_values, 1));
        temporal_features.push(Self::calculate_autocorrelation(&throughput_values, 1));
        temporal_features.push(Self::calculate_autocorrelation(&bler_values, 1));
        
        // Seasonality indicators (simplified - detect periodic patterns)
        temporal_features.push(self.detect_periodicity(&sinr_values));
        temporal_features.push(self.detect_periodicity(&throughput_values));
        temporal_features.push(self.detect_periodicity(&bler_values));
        
        temporal_features
    }
    
    /// Calculate ratio and derived features
    fn calculate_ratio_features(&mut self, features: &CsvKpiFeatures) -> Vec<f32> {
        let mut ratio_features = Vec::new();
        
        // Efficiency ratios
        let spectral_efficiency = (features.ave_4g_lte_dl_user_thrput + features.ave_4g_lte_ul_user_thrput) / 
                                 (features.band_count * 20.0).max(1.0); // Assume 20MHz per band
        ratio_features.push(spectral_efficiency);
        
        let user_efficiency = (features.ave_4g_lte_dl_user_thrput + features.ave_4g_lte_ul_user_thrput) / 
                              features.rrc_connected_users_avg.max(1.0);
        ratio_features.push(user_efficiency);
        
        let energy_efficiency = (features.ave_4g_lte_dl_user_thrput + features.ave_4g_lte_ul_user_thrput) / 
                               (features.sinr_pusch_avg + 10.0); // Rough power proxy
        ratio_features.push(energy_efficiency);
        
        // Performance indices
        let quality_index = (features.sinr_pusch_avg / 30.0) * (1.0 - features.mac_dl_bler / 100.0);
        ratio_features.push(quality_index.clamp(0.0, 1.0));
        
        let mobility_performance = (features.lte_intra_freq_ho_sr + features.lte_inter_freq_ho_sr) / 200.0;
        ratio_features.push(mobility_performance.clamp(0.0, 1.0));
        
        let endc_efficiency = if features.endc_establishment_att > 0.0 {
            features.endc_establishment_succ / features.endc_establishment_att
        } else {
            0.0
        };
        ratio_features.push(endc_efficiency);
        
        // Load indicators
        let traffic_load_ratio = features.eric_traff_erab_erl / 100.0; // Normalize to typical max
        ratio_features.push(traffic_load_ratio.clamp(0.0, 2.0));
        
        let user_density = features.rrc_connected_users_avg / 200.0; // Normalize to typical max
        ratio_features.push(user_density.clamp(0.0, 2.0));
        
        let volume_ratio = features.dl_volume_pdcp_gbytes / features.ul_volume_pdcp_gbytes.max(0.1);
        ratio_features.push(volume_ratio.clamp(0.1, 10.0));
        
        // Quality metrics
        let sinr_balance = (features.sinr_pusch_avg - features.sinr_pucch_avg).abs();
        ratio_features.push(sinr_balance);
        
        let error_rate_balance = (features.mac_dl_bler - features.mac_ul_bler).abs();
        ratio_features.push(error_rate_balance);
        
        let handover_balance = (features.lte_intra_freq_ho_sr - features.lte_inter_freq_ho_sr).abs();
        ratio_features.push(handover_balance);
        
        // Update internal state
        self.ratio_features.efficiency_ratios.insert("spectral".to_string(), spectral_efficiency);
        self.ratio_features.performance_indices.insert("quality".to_string(), quality_index);
        
        ratio_features
    }
    
    /// Encode categorical features
    fn encode_categorical_features(&mut self, features: &CsvKpiFeatures) -> Vec<f32> {
        let mut categorical_features = Vec::new();
        
        // Frequency band encoding
        let band_encoding = self.categorical_encoders.frequency_band_encoder
            .get(&features.frequency_band)
            .copied()
            .unwrap_or(0.5); // Default for unknown bands
        categorical_features.push(band_encoding);
        
        // Band count (ordinal encoding)
        let normalized_band_count = (features.band_count - 1.0) / 7.0; // Normalize 1-8 bands to 0-1
        categorical_features.push(normalized_band_count.clamp(0.0, 1.0));
        
        // Binary features
        let has_volte = if features.volte_traffic_erl > 0.0 { 1.0 } else { 0.0 };
        categorical_features.push(has_volte);
        
        let has_endc = if features.endc_establishment_att > 0.0 { 1.0 } else { 0.0 };
        categorical_features.push(has_endc);
        
        let high_load = if features.rrc_connected_users_avg > 100.0 { 1.0 } else { 0.0 };
        categorical_features.push(high_load);
        
        // Performance categories
        let sinr_category = if features.sinr_pusch_avg > 20.0 {
            1.0 // Excellent
        } else if features.sinr_pusch_avg > 15.0 {
            0.75 // Good
        } else if features.sinr_pusch_avg > 10.0 {
            0.5 // Fair
        } else {
            0.25 // Poor
        };
        categorical_features.push(sinr_category);
        
        let throughput_category = if features.ave_4g_lte_dl_user_thrput > 50.0 {
            1.0 // High
        } else if features.ave_4g_lte_dl_user_thrput > 20.0 {
            0.66 // Medium
        } else {
            0.33 // Low
        };
        categorical_features.push(throughput_category);
        
        categorical_features
    }
    
    /// Calculate trend using linear regression
    fn calculate_trend(&self, metric: &str) -> f32 {
        if self.history_buffer.len() < 3 {
            return 0.0;
        }
        
        let values: Vec<f32> = match metric {
            "sinr" => self.history_buffer.iter().map(|f| f.sinr_pusch_avg).collect(),
            "throughput" => self.history_buffer.iter().map(|f| f.ave_4g_lte_dl_user_thrput).collect(),
            "bler" => self.history_buffer.iter().map(|f| f.mac_dl_bler).collect(),
            _ => return 0.0,
        };
        
        let n = values.len() as f32;
        let x_values: Vec<f32> = (0..values.len()).map(|i| i as f32).collect();
        
        let sum_x = x_values.iter().sum::<f32>();
        let sum_y = values.iter().sum::<f32>();
        let sum_xy = x_values.iter().zip(values.iter()).map(|(x, y)| x * y).sum::<f32>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f32>();
        
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }
        
        (n * sum_xy - sum_x * sum_y) / denominator
    }
    
    /// Generate feature names for interpretability
    fn generate_feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        
        // Raw feature names
        let raw_names = vec![
            "sinr_pusch", "sinr_pucch", "ul_rssi_total", "mac_dl_bler", "mac_ul_bler",
            "rrc_connected_users", "ul_volume_gbytes", "dl_volume_gbytes", "volte_traffic",
            "eric_traff_erab", "dl_user_thrput", "ul_user_thrput", "dl_latency",
            "erab_drop_rate", "lte_dcr_volte", "intra_ho_sr", "inter_ho_sr",
            "inter_ho_attempts", "intra_ho_attempts", "endc_att", "endc_succ",
            "endc_setup_sr", "endc_scg_failure", "band_count", "active_ues_dl", "active_ues_ul"
        ];
        names.extend(raw_names.iter().map(|s| s.to_string()));
        
        // Statistical feature names
        for metric in &["sinr", "throughput", "bler", "latency"] {
            names.push(format!("{}_mean", metric));
            names.push(format!("{}_std", metric));
            names.push(format!("{}_min", metric));
            names.push(format!("{}_max", metric));
            names.push(format!("{}_median", metric));
        }
        
        // Temporal feature names
        for metric in &["sinr", "throughput", "bler"] {
            names.push(format!("{}_trend", metric));
        }
        for metric in &["sinr", "throughput", "bler"] {
            names.push(format!("{}_change_rate", metric));
        }
        for metric in &["sinr", "throughput", "bler"] {
            names.push(format!("{}_volatility", metric));
        }
        for metric in &["sinr", "throughput", "bler"] {
            names.push(format!("{}_autocorr", metric));
        }
        for metric in &["sinr", "throughput", "bler"] {
            names.push(format!("{}_periodicity", metric));
        }
        
        // Ratio feature names
        names.extend(vec![
            "spectral_efficiency".to_string(),
            "user_efficiency".to_string(),
            "energy_efficiency".to_string(),
            "quality_index".to_string(),
            "mobility_performance".to_string(),
            "endc_efficiency".to_string(),
            "traffic_load_ratio".to_string(),
            "user_density".to_string(),
            "volume_ratio".to_string(),
            "sinr_balance".to_string(),
            "error_rate_balance".to_string(),
            "handover_balance".to_string(),
        ]);
        
        // Categorical feature names
        names.extend(vec![
            "band_encoding".to_string(),
            "normalized_band_count".to_string(),
            "has_volte".to_string(),
            "has_endc".to_string(),
            "high_load".to_string(),
            "sinr_category".to_string(),
            "throughput_category".to_string(),
        ]);
        
        names
    }
    
    // Statistical helper functions
    fn calculate_mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }
    
    fn calculate_std(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = Self::calculate_mean(values);
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }
    
    fn calculate_min(values: &[f32]) -> f32 {
        values.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }
    
    fn calculate_max(values: &[f32]) -> f32 {
        values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }
    
    fn calculate_percentile(values: &[f32], percentile: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile * (sorted_values.len() - 1) as f32) as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }
    
    fn calculate_coefficient_of_variation(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = Self::calculate_mean(values);
        if mean.abs() < 1e-10 {
            return 0.0;
        }
        
        let std = Self::calculate_std(values);
        std / mean.abs()
    }
    
    fn calculate_autocorrelation(values: &[f32], lag: usize) -> f32 {
        if values.len() <= lag {
            return 0.0;
        }
        
        let n = values.len() - lag;
        let x1: Vec<f32> = values[..n].to_vec();
        let x2: Vec<f32> = values[lag..].to_vec();
        
        let mean1 = Self::calculate_mean(&x1);
        let mean2 = Self::calculate_mean(&x2);
        
        let numerator: f32 = x1.iter().zip(x2.iter())
            .map(|(&a, &b)| (a - mean1) * (b - mean2))
            .sum();
        
        let denom1: f32 = x1.iter().map(|&x| (x - mean1).powi(2)).sum();
        let denom2: f32 = x2.iter().map(|&x| (x - mean2).powi(2)).sum();
        
        let denominator = (denom1 * denom2).sqrt();
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    fn detect_periodicity(&self, values: &[f32]) -> f32 {
        if values.len() < 6 {
            return 0.0;
        }
        
        // Simple periodicity detection using autocorrelation at different lags
        let mut max_autocorr = 0.0;
        let max_lag = values.len() / 3;
        
        for lag in 2..=max_lag {
            let autocorr = Self::calculate_autocorrelation(values, lag);
            max_autocorr = max_autocorr.max(autocorr.abs());
        }
        
        max_autocorr
    }
    
    /// Get feature importance based on engineered features
    pub fn get_feature_importance(&self) -> HashMap<String, f32> {
        let mut importance = HashMap::new();
        
        // Assign importance scores based on domain knowledge
        // Raw features
        importance.insert("sinr_pusch".to_string(), 0.9);
        importance.insert("mac_dl_bler".to_string(), 0.85);
        importance.insert("dl_user_thrput".to_string(), 0.8);
        importance.insert("rrc_connected_users".to_string(), 0.75);
        
        // Statistical features
        importance.insert("sinr_std".to_string(), 0.7);
        importance.insert("throughput_mean".to_string(), 0.75);
        importance.insert("bler_max".to_string(), 0.7);
        
        // Temporal features
        importance.insert("sinr_trend".to_string(), 0.65);
        importance.insert("throughput_volatility".to_string(), 0.6);
        
        // Ratio features
        importance.insert("spectral_efficiency".to_string(), 0.8);
        importance.insert("quality_index".to_string(), 0.85);
        importance.insert("user_efficiency".to_string(), 0.7);
        
        importance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_engineering_creation() {
        let fe = FeatureEngineering::new(10);
        assert_eq!(fe.buffer_size, 10);
        assert!(fe.history_buffer.is_empty());
    }
    
    #[test]
    fn test_raw_feature_extraction() {
        let fe = FeatureEngineering::new(10);
        let features = create_test_features();
        
        let raw_features = fe.extract_raw_features(&features);
        assert_eq!(raw_features.len(), 26); // Expected number of raw features
        assert_eq!(raw_features[0], features.sinr_pusch_avg);
    }
    
    #[test]
    fn test_ratio_features() {
        let mut fe = FeatureEngineering::new(10);
        let features = create_test_features();
        
        let ratio_features = fe.calculate_ratio_features(&features);
        assert!(!ratio_features.is_empty());
        assert!(ratio_features.iter().all(|&f| f.is_finite()));
    }
    
    #[test]
    fn test_categorical_encoding() {
        let mut fe = FeatureEngineering::new(10);
        let features = create_test_features();
        
        let categorical_features = fe.encode_categorical_features(&features);
        assert!(!categorical_features.is_empty());
        assert!(categorical_features.iter().all(|&f| f >= 0.0 && f <= 1.0));
    }
    
    #[test]
    fn test_statistical_calculations() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(FeatureEngineering::calculate_mean(&values), 3.0);
        assert!((FeatureEngineering::calculate_std(&values) - 1.5811).abs() < 0.01);
        assert_eq!(FeatureEngineering::calculate_min(&values), 1.0);
        assert_eq!(FeatureEngineering::calculate_max(&values), 5.0);
        assert_eq!(FeatureEngineering::calculate_percentile(&values, 0.5), 3.0);
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