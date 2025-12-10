//! Enhanced Anomaly Detector with Real KPI Data Integration
//! 
//! Migrated and enhanced from AFM detection modules to work with real fanndata.csv
//! Combines autoencoder-based reconstruction with statistical anomaly detection
//! and integrates seamlessly with the standalone swarm demo architecture.

use crate::utils::csv_data_parser::{RealNetworkData, CsvDataParser};
use crate::neural::ml_model::{MLModel, ModelType, TrainingConfig};
use crate::utils::metrics::PerformanceMetrics;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::time::Instant;

/// Enhanced anomaly detector using real network data
pub struct EnhancedAnomalyDetector {
    /// Autoencoder model for reconstruction-based detection
    autoencoder: Box<dyn MLModel>,
    
    /// Statistical thresholds calculated from real data
    thresholds: AnomalyThresholds,
    
    /// Feature processor for neural networks
    feature_processor: FeatureProcessor,
    
    /// Performance metrics tracker
    metrics: PerformanceMetrics,
    
    /// Historical anomaly patterns
    anomaly_history: Vec<AnomalyEvent>,
    
    /// Model configuration
    config: AnomalyDetectorConfig,
}

/// Statistical thresholds derived from real data analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyThresholds {
    // Availability thresholds
    pub min_cell_availability: f64,
    pub max_drop_rate_qci5: f64,
    pub max_drop_rate_qci8: f64,
    
    // Performance thresholds
    pub min_throughput_dl: f64,
    pub min_throughput_ul: f64,
    pub max_latency_avg: f64,
    pub max_latency_qci1: f64,
    
    // Quality thresholds
    pub min_sinr_pusch: f64,
    pub min_sinr_pucch: f64,
    pub max_bler_dl: f64,
    pub max_bler_ul: f64,
    
    // 5G specific thresholds
    pub min_endc_setup_sr: f64,
    pub max_endc_failure_rate: f64,
    
    // Statistical bounds (mean ± 3*std)
    pub statistical_bounds: HashMap<String, (f64, f64)>,
}

impl Default for AnomalyThresholds {
    fn default() -> Self {
        Self {
            min_cell_availability: 95.0,
            max_drop_rate_qci5: 2.0,
            max_drop_rate_qci8: 1.0,
            min_throughput_dl: 1000.0,
            min_throughput_ul: 500.0,
            max_latency_avg: 50.0,
            max_latency_qci1: 30.0,
            min_sinr_pusch: -5.0,
            min_sinr_pucch: -8.0,
            max_bler_dl: 10.0,
            max_bler_ul: 15.0,
            min_endc_setup_sr: 85.0,
            max_endc_failure_rate: 15.0,
            statistical_bounds: HashMap::new(),
        }
    }
}

/// Feature processor for neural network input
#[derive(Debug, Clone)]
pub struct FeatureProcessor {
    /// Feature scaling parameters
    scaling_params: HashMap<String, (f64, f64)>, // (mean, std)
    
    /// Feature importance weights
    feature_weights: Vec<f64>,
    
    /// Temporal window size for sequence features
    window_size: usize,
}

impl FeatureProcessor {
    pub fn new(window_size: usize) -> Self {
        Self {
            scaling_params: HashMap::new(),
            feature_weights: vec![1.0; 33], // Default equal weights for 33 features
            window_size,
        }
    }
    
    /// Process real network data into neural network features
    pub fn process_features(&self, data: &RealNetworkData) -> Vec<f64> {
        let mut features = data.to_neural_features();
        
        // Apply feature scaling if parameters are available
        for (i, feature) in features.iter_mut().enumerate() {
            if let Some((mean, std)) = self.scaling_params.get(&format!("feature_{}", i)) {
                if *std > 0.0 {
                    *feature = (*feature - mean) / std;
                }
            }
        }
        
        // Apply feature importance weights
        for (i, feature) in features.iter_mut().enumerate() {
            if i < self.feature_weights.len() {
                *feature *= self.feature_weights[i];
            }
        }
        
        features
    }
    
    /// Fit scaling parameters from training data
    pub fn fit_scaling(&mut self, training_data: &[RealNetworkData]) {
        if training_data.is_empty() {
            return;
        }
        
        // Extract all features
        let all_features: Vec<Vec<f64>> = training_data
            .iter()
            .map(|data| data.to_neural_features())
            .collect();
        
        // Calculate mean and std for each feature
        for feature_idx in 0..all_features[0].len() {
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
            
            self.scaling_params.insert(format!("feature_{}", feature_idx), (mean, std));
        }
        
        println!("✅ Feature scaling parameters fitted for {} features", all_features[0].len());
    }
}

/// Anomaly event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub timestamp: String,
    pub cell_id: String,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub reconstruction_error: f64,
    pub statistical_score: f64,
    pub combined_score: f64,
    pub contributing_factors: Vec<String>,
    pub raw_data: RealNetworkData,
}

/// Types of anomalies detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    PerformanceDegradation,
    QualityIssue,
    ConnectivityProblem,
    CapacityOverload,
    SignalQualityIssue,
    EndcSetupFailure,
    HandoverProblem,
    MultipleIssues,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Configuration for the anomaly detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectorConfig {
    pub reconstruction_threshold: f64,
    pub statistical_threshold: f64,
    pub combined_threshold: f64,
    pub enable_autoencoder: bool,
    pub enable_statistical: bool,
    pub temporal_smoothing: bool,
    pub alert_threshold: f64,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            reconstruction_threshold: 0.1,
            statistical_threshold: 3.0, // 3-sigma threshold
            combined_threshold: 0.6,
            enable_autoencoder: true,
            enable_statistical: true,
            temporal_smoothing: true,
            alert_threshold: 0.8,
        }
    }
}

impl EnhancedAnomalyDetector {
    /// Create new anomaly detector with real data integration
    pub fn new(config: AnomalyDetectorConfig) -> Self {
        let autoencoder = Box::new(MLModel::new(
            ModelType::Autoencoder,
            TrainingConfig {
                epochs: 100,
                learning_rate: 0.001,
                batch_size: 32,
                validation_split: 0.2,
                early_stopping: true,
                patience: 10,
            }
        ));
        
        Self {
            autoencoder,
            thresholds: AnomalyThresholds::default(),
            feature_processor: FeatureProcessor::new(10),
            metrics: PerformanceMetrics::new(),
            anomaly_history: Vec::new(),
            config,
        }
    }
    
    /// Train the anomaly detector on real network data
    pub fn train_on_real_data(&mut self, csv_file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("🧠 Training enhanced anomaly detector on real data...");
        let start_time = Instant::now();
        
        // Parse real data from CSV
        let mut parser = CsvDataParser::new(csv_file_path.to_string());
        let training_data = parser.parse_real_data()?;
        
        println!("📊 Training data: {} records", training_data.len());
        
        // Filter for normal operation data (for autoencoder training)
        let normal_data: Vec<&RealNetworkData> = training_data
            .iter()
            .filter(|data| {
                data.cell_availability >= 98.0 &&
                data.erab_drop_rate_qci5 <= 1.0 &&
                data.mac_dl_bler <= 5.0 &&
                data.sinr_pusch >= 5.0
            })
            .collect();
        
        println!("✅ Normal operation data: {} records ({:.1}%)", 
            normal_data.len(), 
            normal_data.len() as f64 / training_data.len() as f64 * 100.0
        );
        
        // Fit feature scaling parameters
        self.feature_processor.fit_scaling(&training_data);
        
        // Calculate statistical thresholds from real data
        self.calculate_real_data_thresholds(&training_data);
        
        // Train autoencoder on normal data if enabled
        if self.config.enable_autoencoder && !normal_data.is_empty() {
            let features: Vec<Vec<f64>> = normal_data
                .iter()
                .map(|data| self.feature_processor.process_features(data))
                .collect();
            
            self.autoencoder.train(&features, &features)?; // Autoencoder: input = output
            println!("🤖 Autoencoder trained successfully");
        }
        
        let training_time = start_time.elapsed();
        println!("⏱️  Training completed in {:.2}s", training_time.as_secs_f64());
        
        Ok(())
    }
    
    /// Calculate thresholds from real data statistics
    fn calculate_real_data_thresholds(&mut self, data: &[RealNetworkData]) {
        if data.is_empty() {
            return;
        }
        
        // Calculate percentile-based thresholds
        let mut availability_values: Vec<f64> = data.iter().map(|d| d.cell_availability).collect();
        let mut drop_rate_qci5: Vec<f64> = data.iter().map(|d| d.erab_drop_rate_qci5).collect();
        let mut throughput_dl: Vec<f64> = data.iter().map(|d| d.dl_user_throughput).collect();
        let mut latency_values: Vec<f64> = data.iter().map(|d| d.dl_latency_avg).collect();
        let mut sinr_values: Vec<f64> = data.iter().map(|d| d.sinr_pusch).collect();
        
        availability_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        drop_rate_qci5.sort_by(|a, b| a.partial_cmp(b).unwrap());
        throughput_dl.sort_by(|a, b| a.partial_cmp(b).unwrap());
        latency_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sinr_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Use 5th percentile for minimum thresholds, 95th for maximum
        let p5_idx = (data.len() as f64 * 0.05) as usize;
        let p95_idx = (data.len() as f64 * 0.95) as usize;
        
        self.thresholds.min_cell_availability = availability_values[p5_idx];
        self.thresholds.max_drop_rate_qci5 = drop_rate_qci5[p95_idx];
        self.thresholds.min_throughput_dl = throughput_dl[p5_idx];
        self.thresholds.max_latency_avg = latency_values[p95_idx];
        self.thresholds.min_sinr_pusch = sinr_values[p5_idx];
        
        println!("📊 Real data thresholds calculated:");
        println!("   📶 Min Availability: {:.1}%", self.thresholds.min_cell_availability);
        println!("   📉 Max Drop Rate QCI5: {:.2}%", self.thresholds.max_drop_rate_qci5);
        println!("   🚀 Min DL Throughput: {:.0} Mbps", self.thresholds.min_throughput_dl);
        println!("   ⏱️  Max Latency: {:.1} ms", self.thresholds.max_latency_avg);
        println!("   📡 Min SINR: {:.1} dB", self.thresholds.min_sinr_pusch);
    }
    
    /// Detect anomalies in real network data
    pub fn detect_anomalies(&mut self, data: &RealNetworkData) -> AnomalyEvent {
        let start_time = Instant::now();
        
        // Calculate reconstruction error if autoencoder is enabled
        let reconstruction_error = if self.config.enable_autoencoder {
            let features = self.feature_processor.process_features(data);
            match self.autoencoder.predict(&[features]) {
                Ok(reconstructed) => {
                    if let Some(recon_features) = reconstructed.first() {
                        let original_features = self.feature_processor.process_features(data);
                        let mse: f64 = original_features
                            .iter()
                            .zip(recon_features.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>() / original_features.len() as f64;
                        mse.sqrt()
                    } else {
                        0.0
                    }
                }
                Err(_) => 0.0,
            }
        } else {
            0.0
        };
        
        // Calculate statistical anomaly score
        let statistical_score = if self.config.enable_statistical {
            self.calculate_statistical_score(data)
        } else {
            0.0
        };
        
        // Combine scores
        let combined_score = if self.config.enable_autoencoder && self.config.enable_statistical {
            (reconstruction_error * 0.6 + statistical_score * 0.4)
        } else if self.config.enable_autoencoder {
            reconstruction_error
        } else {
            statistical_score
        };
        
        // Determine anomaly type and severity
        let (anomaly_type, contributing_factors) = self.classify_anomaly(data);
        let severity = self.determine_severity(combined_score, data);
        
        let anomaly_event = AnomalyEvent {
            timestamp: data.timestamp.clone(),
            cell_id: format!("{}_{}", data.enodeb_name, data.cell_name),
            anomaly_type,
            severity,
            reconstruction_error,
            statistical_score,
            combined_score,
            contributing_factors,
            raw_data: data.clone(),
        };
        
        // Store in history
        self.anomaly_history.push(anomaly_event.clone());
        
        // Keep only recent history
        if self.anomaly_history.len() > 10000 {
            self.anomaly_history.drain(0..1000);
        }
        
        self.metrics.record_detection_time(start_time.elapsed().as_millis() as f64);
        
        anomaly_event
    }
    
    /// Calculate statistical anomaly score based on thresholds
    fn calculate_statistical_score(&self, data: &RealNetworkData) -> f64 {
        let mut violations = 0;
        let mut severity_sum = 0.0;
        let checks = 10;
        
        // Availability check
        if data.cell_availability < self.thresholds.min_cell_availability {
            violations += 1;
            severity_sum += (self.thresholds.min_cell_availability - data.cell_availability) / 100.0;
        }
        
        // Drop rate checks
        if data.erab_drop_rate_qci5 > self.thresholds.max_drop_rate_qci5 {
            violations += 1;
            severity_sum += data.erab_drop_rate_qci5 / 10.0;
        }
        
        // Throughput checks
        if data.dl_user_throughput < self.thresholds.min_throughput_dl {
            violations += 1;
            severity_sum += (self.thresholds.min_throughput_dl - data.dl_user_throughput) / 10000.0;
        }
        
        // Latency checks
        if data.dl_latency_avg > self.thresholds.max_latency_avg {
            violations += 1;
            severity_sum += (data.dl_latency_avg - self.thresholds.max_latency_avg) / 100.0;
        }
        
        // Signal quality checks
        if data.sinr_pusch < self.thresholds.min_sinr_pusch {
            violations += 1;
            severity_sum += (self.thresholds.min_sinr_pusch - data.sinr_pusch) / 20.0;
        }
        
        // BLER checks
        if data.mac_dl_bler > self.thresholds.max_bler_dl {
            violations += 1;
            severity_sum += data.mac_dl_bler / 50.0;
        }
        
        // Additional quality checks...
        let violation_rate = violations as f64 / checks as f64;
        let avg_severity = if violations > 0 { severity_sum / violations as f64 } else { 0.0 };
        
        (violation_rate + avg_severity) / 2.0
    }
    
    /// Classify the type of anomaly based on KPI patterns
    fn classify_anomaly(&self, data: &RealNetworkData) -> (AnomalyType, Vec<String>) {
        let mut factors = Vec::new();
        let mut issue_counts = HashMap::new();
        
        // Check availability issues
        if data.cell_availability < self.thresholds.min_cell_availability {
            factors.push("Low cell availability".to_string());
            *issue_counts.entry("connectivity").or_insert(0) += 1;
        }
        
        // Check quality issues
        if data.erab_drop_rate_qci5 > self.thresholds.max_drop_rate_qci5 {
            factors.push("High ERAB drop rate".to_string());
            *issue_counts.entry("quality").or_insert(0) += 1;
        }
        
        if data.mac_dl_bler > self.thresholds.max_bler_dl {
            factors.push("High downlink BLER".to_string());
            *issue_counts.entry("quality").or_insert(0) += 1;
        }
        
        // Check performance issues
        if data.dl_user_throughput < self.thresholds.min_throughput_dl {
            factors.push("Low downlink throughput".to_string());
            *issue_counts.entry("performance").or_insert(0) += 1;
        }
        
        if data.dl_latency_avg > self.thresholds.max_latency_avg {
            factors.push("High latency".to_string());
            *issue_counts.entry("performance").or_insert(0) += 1;
        }
        
        // Check signal quality
        if data.sinr_pusch < self.thresholds.min_sinr_pusch {
            factors.push("Poor SINR".to_string());
            *issue_counts.entry("signal").or_insert(0) += 1;
        }
        
        // Check 5G ENDC issues
        if data.endc_setup_sr < self.thresholds.min_endc_setup_sr {
            factors.push("Low ENDC setup success rate".to_string());
            *issue_counts.entry("endc").or_insert(0) += 1;
        }
        
        // Determine primary anomaly type
        let anomaly_type = if issue_counts.len() > 2 {
            AnomalyType::MultipleIssues
        } else if issue_counts.contains_key("endc") {
            AnomalyType::EndcSetupFailure
        } else if issue_counts.contains_key("signal") {
            AnomalyType::SignalQualityIssue
        } else if issue_counts.contains_key("quality") {
            AnomalyType::QualityIssue
        } else if issue_counts.contains_key("connectivity") {
            AnomalyType::ConnectivityProblem
        } else if issue_counts.contains_key("performance") {
            AnomalyType::PerformanceDegradation
        } else {
            AnomalyType::PerformanceDegradation
        };
        
        (anomaly_type, factors)
    }
    
    /// Determine severity based on combined score and KPI values
    fn determine_severity(&self, combined_score: f64, data: &RealNetworkData) -> AnomalySeverity {
        // Check for critical conditions
        if data.cell_availability < 50.0 || 
           data.erab_drop_rate_qci5 > 20.0 ||
           combined_score > 0.9 {
            return AnomalySeverity::Critical;
        }
        
        // Check for high severity
        if data.cell_availability < 80.0 ||
           data.erab_drop_rate_qci5 > 10.0 ||
           combined_score > 0.7 {
            return AnomalySeverity::High;
        }
        
        // Check for medium severity
        if data.cell_availability < 95.0 ||
           data.erab_drop_rate_qci5 > 5.0 ||
           combined_score > 0.4 {
            return AnomalySeverity::Medium;
        }
        
        AnomalySeverity::Low
    }
    
    /// Get anomaly detection performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }
    
    /// Get recent anomaly history
    pub fn get_recent_anomalies(&self, limit: usize) -> Vec<&AnomalyEvent> {
        self.anomaly_history
            .iter()
            .rev()
            .take(limit)
            .collect()
    }
    
    /// Export anomaly detection results
    pub fn export_results(&self, format: &str) -> Result<String, Box<dyn std::error::Error>> {
        match format {
            "json" => {
                let results = serde_json::to_string_pretty(&self.anomaly_history)?;
                Ok(results)
            }
            "csv" => {
                let mut csv_output = String::from("timestamp,cell_id,anomaly_type,severity,combined_score,factors\n");
                for event in &self.anomaly_history {
                    csv_output.push_str(&format!(
                        "{},{},{:?},{:?},{:.3},{}\n",
                        event.timestamp,
                        event.cell_id,
                        event.anomaly_type,
                        event.severity,
                        event.combined_score,
                        event.contributing_factors.join("; ")
                    ));
                }
                Ok(csv_output)
            }
            _ => Err("Unsupported export format".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detection() {
        let config = AnomalyDetectorConfig::default();
        let mut detector = EnhancedAnomalyDetector::new(config);
        
        // Test with normal data
        let normal_data = RealNetworkData {
            timestamp: "2025-06-27 00".to_string(),
            cell_availability: 99.5,
            erab_drop_rate_qci5: 0.5,
            dl_user_throughput: 25000.0,
            sinr_pusch: 12.0,
            mac_dl_bler: 2.0,
            dl_latency_avg: 15.0,
            endc_setup_sr: 98.0,
            ..Default::default()
        };
        
        let anomaly = detector.detect_anomalies(&normal_data);
        assert!(matches!(anomaly.severity, AnomalySeverity::Low));
        
        // Test with anomalous data
        let anomalous_data = RealNetworkData {
            timestamp: "2025-06-27 01".to_string(),
            cell_availability: 70.0, // Low availability
            erab_drop_rate_qci5: 15.0, // High drop rate
            dl_user_throughput: 500.0, // Low throughput
            sinr_pusch: -10.0, // Poor signal
            mac_dl_bler: 25.0, // High error rate
            dl_latency_avg: 100.0, // High latency
            endc_setup_sr: 60.0, // Poor 5G performance
            ..Default::default()
        };
        
        let anomaly = detector.detect_anomalies(&anomalous_data);
        assert!(matches!(anomaly.severity, AnomalySeverity::High) || 
                matches!(anomaly.severity, AnomalySeverity::Critical));
        assert!(!anomaly.contributing_factors.is_empty());
    }
}

impl Default for RealNetworkData {
    fn default() -> Self {
        Self {
            timestamp: String::new(),
            enodeb_code: String::new(),
            enodeb_name: String::new(),
            cell_code: String::new(),
            cell_name: String::new(),
            band: String::new(),
            num_bands: 0,
            cell_availability: 100.0,
            volte_traffic: 0.0,
            erab_traffic: 0.0,
            rrc_connected_users: 0.0,
            ul_volume_gbytes: 0.0,
            dl_volume_gbytes: 0.0,
            dcr_volte: 0.0,
            erab_drop_rate_qci5: 0.0,
            erab_drop_rate_qci8: 0.0,
            ue_context_attempts: 0,
            ue_context_abnormal_rel: 0.0,
            volte_radio_drop: 0.0,
            cssr_volte: 100.0,
            cssr_end_user: 100.0,
            erab_qci1_ssr: 100.0,
            erab_init_setup_sr: 100.0,
            dl_user_throughput: 0.0,
            ul_user_throughput: 0.0,
            dl_cell_throughput: 0.0,
            ul_cell_throughput: 0.0,
            sinr_pusch: 0.0,
            sinr_pucch: 0.0,
            ul_rssi_pucch: -120.0,
            ul_rssi_pusch: -120.0,
            ul_rssi_total: -120.0,
            mac_dl_bler: 0.0,
            mac_ul_bler: 0.0,
            dl_packet_error_rate: 0.0,
            dl_packet_error_qci1: 0.0,
            dl_packet_loss_qci5: 0.0,
            ul_packet_loss_rate: 0.0,
            dl_latency_avg: 0.0,
            dl_latency_qci1: 0.0,
            endc_establishment_attempts: 0,
            endc_establishment_success: 0,
            endc_setup_sr: 100.0,
            endc_capable_ues: 0,
            lte_intra_freq_ho_sr: 100.0,
            lte_inter_freq_ho_sr: 100.0,
            intra_freq_ho_attempts: 0,
            inter_freq_ho_attempts: 0,
            active_users_dl: 0,
            active_users_ul: 0,
            voip_integrity_rate: 100.0,
            ue_power_limited_percent: 0.0,
        }
    }
}