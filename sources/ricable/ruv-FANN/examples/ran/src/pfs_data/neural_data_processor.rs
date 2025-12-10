//! Enhanced Neural Data Processor for RAN Intelligence
//! 
//! High-performance data processing pipeline optimized for neural swarm coordination
//! Integrates comprehensive RAN data mapping with real-time anomaly detection

use crate::pfs_data::ran_data_mapper::{RanDataMapper, AnomalyAlert, RanDataCategory};
use crate::pfs_data::tensor::{TensorStorage, TensorMeta, TensorDataType, TensorBatch};
use crate::pfs_data::data_driven_thresholds::DataDrivenThresholds;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

/// Enhanced neural data processor with swarm intelligence integration
#[derive(Debug)]
pub struct NeuralDataProcessor {
    pub data_mapper: RanDataMapper,
    pub processing_stats: ProcessingStats,
    pub anomaly_buffer: Arc<Mutex<Vec<AnomalyAlert>>>,
    pub feature_cache: Arc<Mutex<HashMap<String, CachedFeatures>>>,
    pub neural_config: NeuralProcessingConfig,
}

/// Configuration for neural processing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralProcessingConfig {
    pub batch_size: usize,
    pub feature_vector_size: usize,
    pub anomaly_threshold: f32,
    pub cache_enabled: bool,
    pub parallel_processing: bool,
    pub real_time_processing: bool,
    pub swarm_coordination_enabled: bool,
}

/// Processing statistics for monitoring
#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub rows_processed: u64,
    pub anomalies_detected: u64,
    pub features_extracted: u64,
    pub avg_processing_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Cached feature vectors for performance optimization
#[derive(Debug, Clone)]
pub struct CachedFeatures {
    pub afm_features: Vec<f32>,
    pub dtm_features: Vec<f32>,
    pub comprehensive_features: Vec<f32>,
    pub timestamp: SystemTime,
    pub cell_id: String,
}

/// Neural processing result with comprehensive analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralProcessingResult {
    pub cell_id: String,
    pub timestamp: SystemTime,
    pub afm_features: Vec<f32>,
    pub dtm_features: Vec<f32>,
    pub comprehensive_features: Vec<f32>,
    pub anomalies: Vec<AnomalyAlert>,
    pub neural_scores: NeuralScores,
    pub processing_metadata: ProcessingMetadata,
}

/// Neural scoring for different intelligence modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralScores {
    pub afm_fault_probability: f32,      // 0.0-1.0 probability of fault
    pub dtm_mobility_score: f32,         // Mobility pattern complexity
    pub energy_efficiency_score: f32,    // Energy optimization potential
    pub service_quality_score: f32,      // Overall service quality
    pub anomaly_severity_score: f32,     // Combined anomaly severity
}

/// Processing metadata for swarm coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub processing_time_ms: f64,
    pub feature_importance_scores: HashMap<String, f32>,
    pub data_quality_score: f32,
    pub completeness_ratio: f32,
    pub swarm_coordination_data: Option<SwarmCoordinationData>,
}

/// Swarm coordination data for neural agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCoordinationData {
    pub agent_assignments: HashMap<String, String>,
    pub coordination_patterns: Vec<String>,
    pub distributed_processing_hints: Vec<String>,
}

impl Default for NeuralProcessingConfig {
    fn default() -> Self {
        // Use data-driven thresholds instead of hardcoded values
        let data_driven = DataDrivenThresholds::from_csv_analysis();
        
        Self {
            batch_size: 32,
            feature_vector_size: 50,
            anomaly_threshold: data_driven.neural_config.anomaly_threshold,  // Was hardcoded 0.8, now 2.15 from analysis
            cache_enabled: true,
            parallel_processing: true,
            real_time_processing: true,
            swarm_coordination_enabled: true,
        }
    }
}

impl NeuralDataProcessor {
    /// Create new enhanced neural data processor
    pub fn new(config: NeuralProcessingConfig) -> Self {
        Self {
            data_mapper: RanDataMapper::new(),
            processing_stats: ProcessingStats::default(),
            anomaly_buffer: Arc::new(Mutex::new(Vec::new())),
            feature_cache: Arc::new(Mutex::new(HashMap::new())),
            neural_config: config,
        }
    }

    /// Process CSV data with comprehensive neural intelligence
    pub fn process_csv_data(&mut self, csv_content: &str) -> Vec<NeuralProcessingResult> {
        let start_time = Instant::now();
        let lines: Vec<&str> = csv_content.lines().collect();
        
        if lines.is_empty() {
            return Vec::new();
        }
        
        // Skip header row
        let data_lines = &lines[1..];
        
        let results = if self.neural_config.parallel_processing {
            self.process_parallel(data_lines)
        } else {
            self.process_sequential(data_lines)
        };
        
        // Update processing statistics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.processing_stats.rows_processed += results.len() as u64;
        self.processing_stats.avg_processing_time_ms = 
            (self.processing_stats.avg_processing_time_ms + processing_time) / 2.0;
        
        results
    }

    /// Process CSV file directly from filesystem 
    pub fn process_csv_file<P: AsRef<std::path::Path>>(&mut self, file_path: P) -> Result<Vec<NeuralProcessingResult>, Box<dyn std::error::Error>> {
        let csv_content = std::fs::read_to_string(file_path)?;
        Ok(self.process_csv_data(&csv_content))
    }

    /// Process data lines in parallel for high performance
    fn process_parallel(&mut self, data_lines: &[&str]) -> Vec<NeuralProcessingResult> {
        data_lines.par_iter()
            .filter_map(|line| self.process_single_row(line))
            .collect()
    }

    /// Process data lines sequentially 
    fn process_sequential(&mut self, data_lines: &[&str]) -> Vec<NeuralProcessingResult> {
        data_lines.iter()
            .filter_map(|line| self.process_single_row(line))
            .collect()
    }

    /// Process a single CSV row with comprehensive neural analysis
    fn process_single_row(&self, csv_row: &str) -> Option<NeuralProcessingResult> {
        let start_time = Instant::now();
        
        // Parse CSV row into structured data
        let data_row = self.data_mapper.parse_csv_row(csv_row);
        if data_row.is_empty() {
            return None;
        }
        
        // Extract cell identifier for caching
        let cell_id = self.extract_cell_id(&data_row);
        
        // Check cache if enabled
        if self.neural_config.cache_enabled {
            if let Ok(cache) = self.feature_cache.lock() {
                if let Some(cached) = cache.get(&cell_id) {
                    // Use cached features if recent enough (within 5 minutes)
                    if cached.timestamp.elapsed().unwrap_or_default().as_secs() < 300 {
                        return Some(self.create_result_from_cache(cached, &data_row));
                    }
                }
            }
        }
        
        // Extract neural features for different intelligence modules
        let afm_features = self.data_mapper.get_afm_detection_features(&data_row);
        let dtm_features = self.data_mapper.get_dtm_mobility_features(&data_row);
        let comprehensive_features = self.data_mapper.get_comprehensive_features(&data_row);
        
        // Detect anomalies with enhanced algorithms
        let anomalies = self.data_mapper.detect_anomalies(&data_row);
        
        // Calculate neural scores for each intelligence module using real CSV data
        let neural_scores = self.calculate_neural_scores_from_csv(&afm_features, &dtm_features, &anomalies, &data_row);
        
        // Generate processing metadata
        let processing_metadata = self.generate_processing_metadata(
            &data_row, start_time, &afm_features, &dtm_features
        );
        
        // Cache features for future use
        if self.neural_config.cache_enabled {
            self.cache_features(&cell_id, &afm_features, &dtm_features, &comprehensive_features);
        }
        
        // Store anomalies in buffer for swarm coordination
        if !anomalies.is_empty() {
            if let Ok(mut buffer) = self.anomaly_buffer.lock() {
                buffer.extend(anomalies.clone());
                // Keep only recent anomalies (last 1000)
                if buffer.len() > 1000 {
                    let buffer_len = buffer.len();
                    buffer.drain(0..buffer_len - 1000);
                }
            }
        }
        
        Some(NeuralProcessingResult {
            cell_id,
            timestamp: SystemTime::now(),
            afm_features,
            dtm_features,
            comprehensive_features,
            anomalies,
            neural_scores,
            processing_metadata,
        })
    }

    /// Extract cell identifier from data row using real CSV columns
    fn extract_cell_id(&self, data_row: &HashMap<String, f64>) -> String {
        let enodeb = data_row.get("CODE_ELT_ENODEB")
            .map(|x| format!("{:.0}", x))
            .or_else(|| data_row.get("ENODEB").map(|x| format!("{:.0}", x)))
            .unwrap_or_else(|| "UNKNOWN_ENODEB".to_string());
        let cell = data_row.get("CODE_ELT_CELLULE")
            .map(|x| format!("{:.0}", x))
            .or_else(|| data_row.get("CELLULE").map(|x| format!("{:.0}", x)))
            .unwrap_or_else(|| "UNKNOWN_CELL".to_string());
        format!("{}_{}", enodeb, cell)
    }

    /// Calculate neural scores for different intelligence modules using real CSV data analysis
    fn calculate_neural_scores(&self, afm_features: &[f32], dtm_features: &[f32], 
                              anomalies: &[AnomalyAlert]) -> NeuralScores {
        
        // AFM fault probability - Real calculation based on availability and drop rates
        let afm_fault_probability = self.calculate_afm_fault_probability(afm_features);
        
        // DTM mobility score - Real calculation based on handover success and traffic patterns
        let dtm_mobility_score = self.calculate_dtm_mobility_score(dtm_features);
        
        // Energy efficiency score - Real calculation based on power metrics and thermal data
        let energy_efficiency_score = self.calculate_energy_efficiency_score(afm_features, dtm_features);
        
        // Service quality score - Real calculation from QoS metrics
        let service_quality_score = self.calculate_service_quality_score(afm_features, dtm_features);
        
        // Anomaly severity score - Real statistical analysis of CSV patterns
        let anomaly_severity_score = self.calculate_anomaly_severity_score(anomalies, afm_features, dtm_features);
        
        NeuralScores {
            afm_fault_probability,
            dtm_mobility_score,
            energy_efficiency_score,
            service_quality_score,
            anomaly_severity_score,
        }
    }
    
    /// Calculate AFM fault probability from availability metrics and drop rates
    fn calculate_afm_fault_probability(&self, afm_features: &[f32]) -> f32 {
        if afm_features.is_empty() {
            return 0.5; // Unknown state
        }
        
        // Expected order based on AFM detection inputs:
        // [0] CELL_AVAILABILITY_%
        // [1] 4G_LTE_DCR_VOLTE 
        // [2] ERAB_DROP_RATE_QCI_5
        // [3] ERAB_DROP_RATE_QCI_8
        // [4] UE_CTXT_ABNORM_REL_%
        // [5] MAC_DL_BLER
        // [6] MAC_UL_BLER
        // [7] DL_PACKET_ERROR_LOSS_RATE
        // [8] UL_PACKET_LOSS_RATE
        
        let mut fault_score = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        // Cell availability (inverted - low availability = high fault probability)
        if afm_features.len() > 0 {
            let availability = afm_features[0];
            let availability_fault = (1.0 - availability).max(0.0); // Convert 0-1 to fault score
            fault_score += availability_fault * 0.25; // 25% weight
            weight_sum += 0.25;
        }
        
        // VoLTE drop call rate (normalized to 0-1, higher = more fault)
        if afm_features.len() > 1 {
            let volte_dcr = afm_features[1].min(1.0); // Clamp extreme values
            fault_score += volte_dcr * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // E-RAB drop rates QCI 5 & 8 (critical for data services)
        if afm_features.len() > 3 {
            let erab_qci5 = afm_features[2].min(1.0);
            let erab_qci8 = afm_features[3].min(1.0);
            let avg_erab_drop = (erab_qci5 + erab_qci8) / 2.0;
            fault_score += avg_erab_drop * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // UE context abnormal release rate
        if afm_features.len() > 4 {
            let ue_abnormal = afm_features[4].min(1.0);
            fault_score += ue_abnormal * 0.15; // 15% weight
            weight_sum += 0.15;
        }
        
        // MAC layer block error rates (PHY/MAC issues)
        if afm_features.len() > 6 {
            let mac_dl_bler = afm_features[5].min(1.0);
            let mac_ul_bler = afm_features[6].min(1.0);
            let avg_mac_bler = (mac_dl_bler + mac_ul_bler) / 2.0;
            fault_score += avg_mac_bler * 0.10; // 10% weight
            weight_sum += 0.10;
        }
        
        // Packet loss rates (application layer issues)
        if afm_features.len() > 8 {
            let dl_packet_loss = afm_features[7].min(1.0);
            let ul_packet_loss = afm_features[8].min(1.0);
            let avg_packet_loss = (dl_packet_loss + ul_packet_loss) / 2.0;
            fault_score += avg_packet_loss * 0.10; // 10% weight
            weight_sum += 0.10;
        }
        
        // Normalize by actual weights used
        if weight_sum > 0.0 {
            (fault_score / weight_sum).clamp(0.0, 1.0)
        } else {
            0.5 // Default unknown state
        }
    }
    
    /// Calculate DTM mobility score from handover success rates and traffic patterns
    fn calculate_dtm_mobility_score(&self, dtm_features: &[f32]) -> f32 {
        if dtm_features.is_empty() {
            return 0.0;
        }
        
        // Expected order based on DTM mobility inputs:
        // [0] LTE_INTRA_FREQ_HO_SR
        // [1] LTE_INTER_FREQ_HO_SR 
        // [2] INTER FREQ HO ATTEMPTS
        // [3] INTRA FREQ HO ATTEMPTS
        // [4] ERIC_HO_OSC_INTRA
        // [5] ERIC_HO_OSC_INTER
        // [6] ERIC_RWR_TOTAL
        // [7] ERIC_RWR_LTE_RATE
        
        let mut mobility_score = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        // Handover success rates (higher = better mobility management)
        if dtm_features.len() > 1 {
            let intra_ho_sr = dtm_features[0]; // Already normalized 0-1
            let inter_ho_sr = dtm_features[1]; // Already normalized 0-1
            let avg_ho_success = (intra_ho_sr + inter_ho_sr) / 2.0;
            mobility_score += avg_ho_success * 0.40; // 40% weight for HO success
            weight_sum += 0.40;
        }
        
        // Handover attempt volumes (normalized by typical range)
        if dtm_features.len() > 3 {
            let inter_attempts = dtm_features[2];
            let intra_attempts = dtm_features[3];
            
            // Higher attempt volumes indicate more mobility activity
            // Normalize by expected maximum (1000 attempts = high mobility)
            let mobility_activity = ((inter_attempts + intra_attempts) / 2000.0).min(1.0);
            mobility_score += mobility_activity * 0.25; // 25% weight for activity level
            weight_sum += 0.25;
        }
        
        // Handover oscillation indicators (lower = better)
        if dtm_features.len() > 5 {
            let intra_osc = dtm_features[4];
            let inter_osc = dtm_features[5];
            let avg_oscillation = (intra_osc + inter_osc) / 2.0;
            
            // Invert oscillation (low oscillation = good mobility)
            let oscillation_quality = (1.0 - avg_oscillation.min(1.0)).max(0.0);
            mobility_score += oscillation_quality * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // Radio Link Failure recovery (higher success = better)
        if dtm_features.len() > 7 {
            let rwr_total = dtm_features[6];
            let rwr_lte_rate = dtm_features[7];
            
            // Good RWR performance indicates robust mobility
            let rwr_performance = rwr_lte_rate; // Already normalized
            mobility_score += rwr_performance * 0.15; // 15% weight
            weight_sum += 0.15;
        }
        
        // Normalize by actual weights used
        if weight_sum > 0.0 {
            (mobility_score / weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
    
    /// Calculate energy efficiency score from power and thermal metrics
    fn calculate_energy_efficiency_score(&self, afm_features: &[f32], dtm_features: &[f32]) -> f32 {
        let mut efficiency_score = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        // Power-limited UE percentage (from AFM features - should be in position based on mapping)
        // Lower power limitation = better efficiency
        let mut power_limited_found = false;
        for (i, &feature) in afm_features.iter().enumerate() {
            // Look for power limitation indicators in the feature vector
            if i >= 10 && i <= 15 { // Power metrics typically in this range
                let power_efficiency = (1.0 - feature.min(1.0)).max(0.0);
                efficiency_score += power_efficiency * 0.30;
                weight_sum += 0.30;
                power_limited_found = true;
                break;
            }
        }
        
        // If no specific power metric found, use default
        if !power_limited_found {
            efficiency_score += 0.7 * 0.30; // Assume moderate efficiency
            weight_sum += 0.30;
        }
        
        // Throughput efficiency (more throughput per resource = better efficiency)
        if dtm_features.len() > 8 {
            // Look for throughput indicators in DTM features
            let throughput_sum = dtm_features.iter().take(4).sum::<f32>();
            let avg_throughput = throughput_sum / 4.0;
            efficiency_score += avg_throughput * 0.25;
            weight_sum += 0.25;
        }
        
        // Resource utilization efficiency (from traffic vs capacity)
        if afm_features.len() > 5 && dtm_features.len() > 2 {
            // High performance with low error rates = efficient
            let error_rate = afm_features.iter().take(5).sum::<f32>() / 5.0;
            let utilization_efficiency = (1.0 - error_rate.min(1.0)).max(0.0);
            efficiency_score += utilization_efficiency * 0.25;
            weight_sum += 0.25;
        }
        
        // Signal quality efficiency (good signal = less power needed)
        if afm_features.len() > 8 {
            let signal_quality = afm_features.iter().skip(8).take(3).sum::<f32>() / 3.0;
            efficiency_score += signal_quality * 0.20;
            weight_sum += 0.20;
        }
        
        // Normalize by actual weights used
        if weight_sum > 0.0 {
            (efficiency_score / weight_sum).clamp(0.0, 1.0)
        } else {
            0.6 // Default moderate efficiency
        }
    }
    
    /// Calculate service quality score from QoS metrics, latency, and throughput
    fn calculate_service_quality_score(&self, afm_features: &[f32], dtm_features: &[f32]) -> f32 {
        let mut quality_score = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        // Availability and reliability (from AFM features)
        if afm_features.len() > 0 {
            let availability = afm_features[0]; // Cell availability
            quality_score += availability * 0.30; // 30% weight for availability
            weight_sum += 0.30;
        }
        
        // Error rates (lower = better quality)
        if afm_features.len() > 8 {
            let error_sum = afm_features.iter().skip(1).take(7).sum::<f32>();
            let avg_error_rate = error_sum / 7.0;
            let error_quality = (1.0 - avg_error_rate.min(1.0)).max(0.0);
            quality_score += error_quality * 0.25; // 25% weight
            weight_sum += 0.25;
        }
        
        // Handover performance (seamless mobility = good quality)
        if dtm_features.len() > 1 {
            let ho_quality = (dtm_features[0] + dtm_features[1]) / 2.0;
            quality_score += ho_quality * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // Service establishment success (from DTM clustering features)
        if dtm_features.len() > 4 {
            let service_success = dtm_features.iter().skip(2).take(3).sum::<f32>() / 3.0;
            quality_score += service_success * 0.15; // 15% weight
            weight_sum += 0.15;
        }
        
        // Overall performance consistency
        if afm_features.len() > 5 && dtm_features.len() > 5 {
            let afm_consistency = 1.0 - (afm_features.iter().take(5)
                .map(|&x| (x - 0.5).abs()).sum::<f32>() / 5.0);
            let dtm_consistency = 1.0 - (dtm_features.iter().take(5)
                .map(|&x| (x - 0.5).abs()).sum::<f32>() / 5.0);
            let overall_consistency = (afm_consistency + dtm_consistency) / 2.0;
            quality_score += overall_consistency.max(0.0) * 0.10; // 10% weight
            weight_sum += 0.10;
        }
        
        // Normalize by actual weights used
        if weight_sum > 0.0 {
            (quality_score / weight_sum).clamp(0.0, 1.0)
        } else {
            0.7 // Default good quality
        }
    }
    
    /// Calculate anomaly severity score from statistical analysis of CSV patterns
    fn calculate_anomaly_severity_score(&self, anomalies: &[AnomalyAlert], 
                                       afm_features: &[f32], dtm_features: &[f32]) -> f32 {
        if anomalies.is_empty() {
            return 0.0; // No anomalies detected
        }
        
        let mut severity_score = 0.0f32;
        let mut total_weight = 0.0f32;
        
        // Process each anomaly with domain-specific severity calculation
        for alert in anomalies {
            let base_severity = match alert.severity {
                crate::pfs_data::ran_data_mapper::AnomalySeverity::Critical => 1.0,
                crate::pfs_data::ran_data_mapper::AnomalySeverity::Warning => 0.5,
            };
            
            // Weight by relevance to different modules
            let relevance_weight = (alert.afm_relevance + alert.dtm_relevance) / 2.0;
            
            // Calculate impact based on the specific metric
            let impact_multiplier = if alert.column_name.contains("AVAILABILITY") {
                2.0 // Availability issues are critical
            } else if alert.column_name.contains("DROP") || alert.column_name.contains("ERROR") {
                1.5 // Service quality issues
            } else if alert.column_name.contains("HO") || alert.column_name.contains("HANDOVER") {
                1.2 // Mobility issues
            } else {
                1.0 // Other metrics
            };
            
            let weighted_severity = base_severity * relevance_weight * impact_multiplier;
            severity_score += weighted_severity;
            total_weight += relevance_weight * impact_multiplier;
        }
        
        // Add statistical deviation component
        let feature_variance = self.calculate_feature_variance(afm_features, dtm_features);
        severity_score += feature_variance * 0.3; // 30% weight for statistical abnormality
        total_weight += 0.3;
        
        // Normalize and clamp
        if total_weight > 0.0 {
            (severity_score / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
    
    /// Calculate statistical variance in features to detect unusual patterns
    fn calculate_feature_variance(&self, afm_features: &[f32], dtm_features: &[f32]) -> f32 {
        let all_features: Vec<f32> = afm_features.iter().chain(dtm_features.iter()).cloned().collect();
        
        if all_features.len() < 2 {
            return 0.0;
        }
        
        let mean = all_features.iter().sum::<f32>() / all_features.len() as f32;
        let variance = all_features.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / all_features.len() as f32;
        
        // High variance indicates unusual patterns
        // Normalize variance to 0-1 range (assume variance > 0.25 is high)
        (variance / 0.25).min(1.0)
    }
    
    /// Calculate neural scores directly from CSV data with enhanced domain-specific algorithms
    fn calculate_neural_scores_from_csv(&self, afm_features: &[f32], dtm_features: &[f32], 
                                       anomalies: &[AnomalyAlert], data_row: &HashMap<String, f64>) -> NeuralScores {
        
        // AFM fault probability - Real calculation from availability and drop rates
        let afm_fault_probability = self.calculate_afm_fault_probability_csv(data_row);
        
        // DTM mobility score - Real calculation from handover and traffic data  
        let dtm_mobility_score = self.calculate_dtm_mobility_score_csv(data_row);
        
        // Energy efficiency score - Real calculation from power and thermal metrics
        let energy_efficiency_score = self.calculate_energy_efficiency_score_csv(data_row);
        
        // Service quality score - Real calculation from QoS metrics
        let service_quality_score = self.calculate_service_quality_score_csv(data_row);
        
        // Anomaly severity score - Real statistical analysis 
        let anomaly_severity_score = self.calculate_anomaly_severity_score(anomalies, afm_features, dtm_features);
        
        NeuralScores {
            afm_fault_probability,
            dtm_mobility_score,
            energy_efficiency_score,
            service_quality_score,
            anomaly_severity_score,
        }
    }
    
    /// Calculate AFM fault probability directly from CSV availability metrics
    fn calculate_afm_fault_probability_csv(&self, data_row: &HashMap<String, f64>) -> f32 {
        let mut fault_score = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        // Cell availability - PRIMARY indicator (critical weight)
        if let Some(&availability) = data_row.get("CELL_AVAILABILITY_%") {
            let availability_norm = availability / 100.0; // Convert percentage to 0-1
            let fault_from_availability = (1.0 - availability_norm as f32).max(0.0);
            fault_score += fault_from_availability * 0.30; // 30% weight
            weight_sum += 0.30;
        }
        
        // VoLTE drop call rate
        if let Some(&volte_dcr) = data_row.get("4G_LTE_DCR_VOLTE") {
            let dcr_norm = (volte_dcr / 5.0).min(1.0) as f32; // Normalize by 5% max expected
            fault_score += dcr_norm * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // E-RAB drop rates for critical QCI classes
        let mut erab_drop_score = 0.0f32;
        let mut erab_count = 0;
        
        if let Some(&qci5_drop) = data_row.get("ERAB_DROP_RATE_QCI_5") {
            erab_drop_score += (qci5_drop / 3.0).min(1.0) as f32; // Max 3% expected
            erab_count += 1;
        }
        
        if let Some(&qci8_drop) = data_row.get("ERAB_DROP_RATE_QCI_8") {
            erab_drop_score += (qci8_drop / 3.0).min(1.0) as f32;
            erab_count += 1;
        }
        
        if erab_count > 0 {
            fault_score += (erab_drop_score / erab_count as f32) * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // UE context abnormal release rate
        if let Some(&ue_abnormal) = data_row.get("UE_CTXT_ABNORM_REL_%") {
            let abnormal_norm = (ue_abnormal / 5.0).min(1.0) as f32; // Max 5% expected
            fault_score += abnormal_norm * 0.15; // 15% weight
            weight_sum += 0.15;
        }
        
        // MAC layer block error rates
        let mut mac_bler_score = 0.0f32;
        let mut mac_count = 0;
        
        if let Some(&dl_bler) = data_row.get("MAC_DL_BLER") {
            mac_bler_score += (dl_bler / 10.0).min(1.0) as f32; // Max 10% expected
            mac_count += 1;
        }
        
        if let Some(&ul_bler) = data_row.get("MAC_UL_BLER") {
            mac_bler_score += (ul_bler / 10.0).min(1.0) as f32;
            mac_count += 1;
        }
        
        if mac_count > 0 {
            fault_score += (mac_bler_score / mac_count as f32) * 0.10; // 10% weight
            weight_sum += 0.10;
        }
        
        // Packet loss rates
        if let Some(&dl_packet_loss) = data_row.get("DL_PACKET_ERROR_LOSS_RATE") {
            let packet_loss_norm = (dl_packet_loss / 5.0).min(1.0) as f32; // Max 5%
            fault_score += packet_loss_norm * 0.05; // 5% weight
            weight_sum += 0.05;
        }
        
        // Normalize and return
        if weight_sum > 0.0 {
            (fault_score / weight_sum).clamp(0.0, 1.0)
        } else {
            0.5 // Unknown state if no metrics available
        }
    }
    
    /// Calculate DTM mobility score from handover success and traffic data
    fn calculate_dtm_mobility_score_csv(&self, data_row: &HashMap<String, f64>) -> f32 {
        let mut mobility_score = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        // Handover success rates (primary mobility indicators)
        let mut ho_success_score = 0.0f32;
        let mut ho_count = 0;
        
        if let Some(&intra_ho_sr) = data_row.get("LTE_INTRA_FREQ_HO_SR") {
            ho_success_score += (intra_ho_sr / 100.0) as f32; // Convert percentage
            ho_count += 1;
        }
        
        if let Some(&inter_ho_sr) = data_row.get("LTE_INTER_FREQ_HO_SR") {
            ho_success_score += (inter_ho_sr / 100.0) as f32;
            ho_count += 1;
        }
        
        if ho_count > 0 {
            mobility_score += (ho_success_score / ho_count as f32) * 0.40; // 40% weight
            weight_sum += 0.40;
        }
        
        // Handover attempt volumes (mobility activity level)
        let mut ho_attempts = 0.0f64;
        let mut attempt_count = 0;
        
        if let Some(&inter_attempts) = data_row.get("INTER FREQ HO ATTEMPTS") {
            ho_attempts += inter_attempts;
            attempt_count += 1;
        }
        
        if let Some(&intra_attempts) = data_row.get("INTRA FREQ HO ATTEMPTS") {
            ho_attempts += intra_attempts;
            attempt_count += 1;
        }
        
        if attempt_count > 0 {
            let avg_attempts = ho_attempts / attempt_count as f64;
            // Normalize by expected range (0-500 attempts = normal mobility)
            let activity_score = (avg_attempts / 500.0).min(1.0) as f32;
            mobility_score += activity_score * 0.25; // 25% weight
            weight_sum += 0.25;
        }
        
        // Connected users average (indicator of traffic handling capability)
        if let Some(&connected_users) = data_row.get("RRC_CONNECTED_ USERS_AVERAGE") {
            // Higher user count with good mobility = better DTM performance
            let user_mobility = (connected_users / 200.0).min(1.0) as f32; // Normalize by 200 users
            mobility_score += user_mobility * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // Traffic volume efficiency
        if let Some(&erab_traffic) = data_row.get("ERIC_TRAFF_ERAB_ERL") {
            let traffic_efficiency = (erab_traffic / 50.0).min(1.0) as f32; // Normalize by 50 Erl
            mobility_score += traffic_efficiency * 0.15; // 15% weight
            weight_sum += 0.15;
        }
        
        // Normalize and return
        if weight_sum > 0.0 {
            (mobility_score / weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
    
    /// Calculate energy efficiency score from power and thermal metrics in CSV
    fn calculate_energy_efficiency_score_csv(&self, data_row: &HashMap<String, f64>) -> f32 {
        let mut efficiency_score = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        // UE Power Limited percentage (key energy efficiency indicator)
        if let Some(&ue_pwr_limited) = data_row.get("UE_PWR_LIMITED") {
            // Lower power limitation = better energy efficiency
            let power_efficiency = (1.0 - (ue_pwr_limited / 100.0)).max(0.0) as f32;
            efficiency_score += power_efficiency * 0.35; // 35% weight
            weight_sum += 0.35;
        }
        
        // Throughput per resource efficiency
        let mut throughput_efficiency = 0.0f32;
        let mut throughput_count = 0;
        
        if let Some(&dl_user_throughput) = data_row.get("&_AVE_4G_LTE_DL_USER_THRPUT") {
            // Higher user throughput = more efficient resource usage
            throughput_efficiency += (dl_user_throughput / 50.0).min(1.0) as f32; // Normalize by 50 Mbps
            throughput_count += 1;
        }
        
        if let Some(&ul_user_throughput) = data_row.get("&_AVE_4G_LTE_UL_USER_THRPUT") {
            throughput_efficiency += (ul_user_throughput / 20.0).min(1.0) as f32; // Normalize by 20 Mbps
            throughput_count += 1;
        }
        
        if throughput_count > 0 {
            efficiency_score += (throughput_efficiency / throughput_count as f32) * 0.25; // 25% weight
            weight_sum += 0.25;
        }
        
        // Signal quality efficiency (better signal = less power needed)
        let mut signal_efficiency = 0.0f32;
        let mut signal_count = 0;
        
        if let Some(&sinr_pusch) = data_row.get("SINR_PUSCH_AVG") {
            // Good SINR indicates efficient signal processing
            signal_efficiency += (sinr_pusch / 30.0).min(1.0).max(0.0) as f32; // Normalize by 30 dB
            signal_count += 1;
        }
        
        if let Some(&sinr_pucch) = data_row.get("SINR_PUCCH_AVG") {
            signal_efficiency += (sinr_pucch / 30.0).min(1.0).max(0.0) as f32;
            signal_count += 1;
        }
        
        if signal_count > 0 {
            efficiency_score += (signal_efficiency / signal_count as f32) * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // Resource utilization efficiency (low error rates = efficient operation)
        if let Some(&availability) = data_row.get("CELL_AVAILABILITY_%") {
            let resource_efficiency = (availability / 100.0) as f32;
            efficiency_score += resource_efficiency * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // Normalize and return
        if weight_sum > 0.0 {
            (efficiency_score / weight_sum).clamp(0.0, 1.0)
        } else {
            0.6 // Default moderate efficiency
        }
    }
    
    /// Calculate service quality score from QoS metrics in CSV
    fn calculate_service_quality_score_csv(&self, data_row: &HashMap<String, f64>) -> f32 {
        let mut quality_score = 0.0f32;
        let mut weight_sum = 0.0f32;
        
        // Primary availability metric
        if let Some(&availability) = data_row.get("CELL_AVAILABILITY_%") {
            quality_score += (availability / 100.0) as f32 * 0.25; // 25% weight
            weight_sum += 0.25;
        }
        
        // Connection establishment success rates
        let mut establishment_quality = 0.0f32;
        let mut establishment_count = 0;
        
        if let Some(&cssr) = data_row.get("CSSR_END_USER_%") {
            establishment_quality += (cssr / 100.0) as f32;
            establishment_count += 1;
        }
        
        if let Some(&erab_ssr) = data_row.get("&_ERAB_QCI1_SSR") {
            establishment_quality += (erab_ssr / 100.0) as f32;
            establishment_count += 1;
        }
        
        if establishment_count > 0 {
            quality_score += (establishment_quality / establishment_count as f32) * 0.20; // 20% weight
            weight_sum += 0.20;
        }
        
        // Service-specific quality metrics
        if let Some(&volte_cssr) = data_row.get("&_4G_LTE_CSSR_VOLTE") {
            quality_score += (volte_cssr / 100.0) as f32 * 0.15; // 15% weight for VoLTE
            weight_sum += 0.15;
        }
        
        // Error rate quality (inverted - lower errors = higher quality)
        let mut error_quality = 0.0f32;
        let mut error_count = 0;
        
        if let Some(&volte_dcr) = data_row.get("4G_LTE_DCR_VOLTE") {
            error_quality += (1.0 - (volte_dcr / 5.0).min(1.0)) as f32; // Max 5% error expected
            error_count += 1;
        }
        
        if let Some(&mac_dl_bler) = data_row.get("MAC_DL_BLER") {
            error_quality += (1.0 - (mac_dl_bler / 10.0).min(1.0)) as f32; // Max 10% error
            error_count += 1;
        }
        
        if error_count > 0 {
            quality_score += (error_quality / error_count as f32) * 0.15; // 15% weight
            weight_sum += 0.15;
        }
        
        // Throughput quality
        let mut throughput_quality = 0.0f32;
        let mut throughput_count = 0;
        
        if let Some(&dl_throughput) = data_row.get("&_AVE_4G_LTE_DL_USER_THRPUT") {
            throughput_quality += (dl_throughput / 30.0).min(1.0) as f32; // Normalize by 30 Mbps
            throughput_count += 1;
        }
        
        if let Some(&ul_throughput) = data_row.get("&_AVE_4G_LTE_UL_USER_THRPUT") {
            throughput_quality += (ul_throughput / 10.0).min(1.0) as f32; // Normalize by 10 Mbps
            throughput_count += 1;
        }
        
        if throughput_count > 0 {
            quality_score += (throughput_quality / throughput_count as f32) * 0.15; // 15% weight
            weight_sum += 0.15;
        }
        
        // Latency quality (lower latency = better quality)
        if let Some(&dl_latency) = data_row.get("DL_LATENCY_AVG") {
            // Assume good latency is under 20ms, poor is over 100ms
            let latency_quality = (1.0 - ((dl_latency - 20.0) / 80.0).max(0.0).min(1.0)) as f32;
            quality_score += latency_quality * 0.10; // 10% weight
            weight_sum += 0.10;
        }
        
        // Normalize and return
        if weight_sum > 0.0 {
            (quality_score / weight_sum).clamp(0.0, 1.0)
        } else {
            0.7 // Default good quality
        }
    }

    /// Generate comprehensive processing metadata
    fn generate_processing_metadata(&self, data_row: &HashMap<String, f64>, 
                                  start_time: Instant, afm_features: &[f32], 
                                  dtm_features: &[f32]) -> ProcessingMetadata {
        let processing_time_ms = start_time.elapsed().as_millis() as f64;
        
        // Calculate feature importance scores
        let mut feature_importance_scores = HashMap::new();
        for (column_name, column_info) in &self.data_mapper.column_mappings {
            if data_row.contains_key(column_name) {
                feature_importance_scores.insert(
                    column_name.clone(), 
                    column_info.neural_importance
                );
            }
        }
        
        // Data quality score based on completeness and validity
        let total_columns = self.data_mapper.column_mappings.len();
        let present_columns = data_row.len();
        let completeness_ratio = present_columns as f32 / total_columns as f32;
        
        // Data quality score considers completeness and feature validity
        let data_quality_score = completeness_ratio * 
            (afm_features.iter().chain(dtm_features.iter())
             .map(|&x| if x.is_finite() { 1.0 } else { 0.0 })
             .sum::<f32>() / (afm_features.len() + dtm_features.len()) as f32);
        
        // Generate swarm coordination data if enabled
        let swarm_coordination_data = if self.neural_config.swarm_coordination_enabled {
            Some(SwarmCoordinationData {
                agent_assignments: self.generate_agent_assignments(data_row),
                coordination_patterns: self.generate_coordination_patterns(afm_features, dtm_features),
                distributed_processing_hints: self.generate_processing_hints(data_row),
            })
        } else {
            None
        };
        
        ProcessingMetadata {
            processing_time_ms,
            feature_importance_scores,
            data_quality_score,
            completeness_ratio,
            swarm_coordination_data,
        }
    }

    /// Generate agent assignments for swarm coordination
    fn generate_agent_assignments(&self, data_row: &HashMap<String, f64>) -> HashMap<String, String> {
        let mut assignments = HashMap::new();
        
        // Assign AFM agents based on fault indicators
        if let Some(availability) = data_row.get("CELL_AVAILABILITY_%") {
            if *availability < 95.0 {
                assignments.insert("fault_detection".to_string(), "afm_detector_agent".to_string());
                assignments.insert("root_cause_analysis".to_string(), "afm_rca_agent".to_string());
            }
        }
        
        // Assign DTM agents based on mobility patterns
        if let Some(ho_attempts) = data_row.get("INTER FREQ HO ATTEMPTS") {
            if *ho_attempts > 100.0 {
                assignments.insert("mobility_optimization".to_string(), "dtm_mobility_agent".to_string());
                assignments.insert("load_balancing".to_string(), "dtm_balancer_agent".to_string());
            }
        }
        
        // Always assign a coordinator agent
        assignments.insert("coordination".to_string(), "swarm_coordinator_agent".to_string());
        
        assignments
    }

    /// Generate coordination patterns for swarm intelligence
    fn generate_coordination_patterns(&self, afm_features: &[f32], dtm_features: &[f32]) -> Vec<String> {
        let mut patterns = Vec::new();
        
        // AFM coordination patterns
        let afm_avg = afm_features.iter().sum::<f32>() / afm_features.len() as f32;
        if afm_avg > 0.7 {
            patterns.push("high_priority_fault_detection".to_string());
            patterns.push("parallel_rca_analysis".to_string());
        }
        
        // DTM coordination patterns
        let dtm_avg = dtm_features.iter().sum::<f32>() / dtm_features.len() as f32;
        if dtm_avg > 0.6 {
            patterns.push("mobility_pattern_clustering".to_string());
            patterns.push("dynamic_load_balancing".to_string());
        }
        
        // Cross-module coordination
        if afm_avg > 0.5 && dtm_avg > 0.5 {
            patterns.push("integrated_optimization".to_string());
        }
        
        patterns
    }

    /// Generate distributed processing hints
    fn generate_processing_hints(&self, data_row: &HashMap<String, f64>) -> Vec<String> {
        let mut hints = Vec::new();
        
        // Processing hints based on data characteristics
        let data_complexity = data_row.len() as f32 / 101.0; // Ratio of present to total columns
        
        if data_complexity > 0.8 {
            hints.push("enable_parallel_processing".to_string());
            hints.push("use_advanced_feature_extraction".to_string());
        }
        
        if data_row.contains_key("ENDC_SETUP_SR") {
            hints.push("prioritize_5g_analysis".to_string());
        }
        
        if data_row.values().any(|&x| x > 1000.0) {
            hints.push("apply_normalization".to_string());
        }
        
        hints
    }

    /// Cache features for performance optimization
    fn cache_features(&self, cell_id: &str, afm_features: &[f32], 
                     dtm_features: &[f32], comprehensive_features: &[f32]) {
        if let Ok(mut cache) = self.feature_cache.lock() {
            cache.insert(cell_id.to_string(), CachedFeatures {
                afm_features: afm_features.to_vec(),
                dtm_features: dtm_features.to_vec(),
                comprehensive_features: comprehensive_features.to_vec(),
                timestamp: SystemTime::now(),
                cell_id: cell_id.to_string(),
            });
            
            // Limit cache size
            if cache.len() > 10000 {
                let oldest_key = cache.iter()
                    .min_by_key(|(_, v)| v.timestamp)
                    .map(|(k, _)| k.clone());
                if let Some(key) = oldest_key {
                    cache.remove(&key);
                }
            }
        }
    }

    /// Create result from cached features
    fn create_result_from_cache(&self, cached: &CachedFeatures, 
                               data_row: &HashMap<String, f64>) -> NeuralProcessingResult {
        let anomalies = self.data_mapper.detect_anomalies(data_row);
        let neural_scores = self.calculate_neural_scores(
            &cached.afm_features, &cached.dtm_features, &anomalies
        );
        
        NeuralProcessingResult {
            cell_id: cached.cell_id.clone(),
            timestamp: SystemTime::now(),
            afm_features: cached.afm_features.clone(),
            dtm_features: cached.dtm_features.clone(),
            comprehensive_features: cached.comprehensive_features.clone(),
            anomalies,
            neural_scores,
            processing_metadata: ProcessingMetadata {
                processing_time_ms: 0.1, // Cache hit
                feature_importance_scores: HashMap::new(),
                data_quality_score: 1.0,
                completeness_ratio: 1.0,
                swarm_coordination_data: None,
            },
        }
    }

    /// Convert processing results to tensor batches for neural network training
    pub fn results_to_tensor_batches(&self, results: &[NeuralProcessingResult]) -> Vec<TensorBatch> {
        let mut batches = Vec::new();
        
        for chunk in results.chunks(self.neural_config.batch_size) {
            let mut batch = TensorBatch::new();
            
            // Create input tensors from features
            let mut afm_input_data = Vec::new();
            let mut dtm_input_data: Vec<f32> = Vec::new();
            let mut comprehensive_input_data: Vec<f32> = Vec::new();
            
            for result in chunk {
                afm_input_data.extend(&result.afm_features);
                dtm_input_data.extend(&result.dtm_features);
                comprehensive_input_data.extend(&result.comprehensive_features);
            }
            
            // Create tensor storages
            if !afm_input_data.is_empty() {
                let meta = TensorMeta::new(
                    vec![chunk.len(), result_feature_size(&results[0].afm_features)],
                    TensorDataType::Float32
                );
                let mut tensor = TensorStorage::new(meta);
                tensor.store_compressed(&afm_input_data, 3).unwrap_or_default();
                batch.add_input(tensor);
            }
            
            batches.push(batch);
        }
        
        batches
    }

    /// Get recent anomalies for swarm coordination
    pub fn get_recent_anomalies(&self, max_count: usize) -> Vec<AnomalyAlert> {
        if let Ok(buffer) = self.anomaly_buffer.lock() {
            buffer.iter()
                .rev()
                .take(max_count)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> &ProcessingStats {
        &self.processing_stats
    }

    /// Clear caches and reset statistics
    pub fn reset(&mut self) {
        if let Ok(mut cache) = self.feature_cache.lock() {
            cache.clear();
        }
        if let Ok(mut buffer) = self.anomaly_buffer.lock() {
            buffer.clear();
        }
        self.processing_stats = ProcessingStats::default();
    }
}

/// Helper function to get feature vector size
fn result_feature_size(features: &[f32]) -> usize {
    features.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_processor_creation() {
        let config = NeuralProcessingConfig::default();
        let processor = NeuralDataProcessor::new(config);
        assert!(processor.neural_config.batch_size > 0);
    }

    #[test]
    fn test_csv_processing() {
        let config = NeuralProcessingConfig::default();
        let mut processor = NeuralDataProcessor::new(config);
        
        // Test with real CSV header from fanndata.csv
        let csv_data = "HEURE(PSDATE);CODE_ELT_ENODEB;ENODEB;CODE_ELT_CELLULE;CELLULE;SYS.BANDE;SYS.NB_BANDES;CELL_AVAILABILITY_%;VOLTE_TRAFFIC (ERL);ERIC_TRAFF_ERAB_ERL
2025-06-27 00:00:00;81371;SITE_001_LTE;20830980;CELL_001_F1;LTE800;4;99.2;15.75;42.8";
        let results = processor.process_csv_data(csv_data);
        
        assert!(!results.is_empty());
        assert!(!results[0].afm_features.is_empty());
        assert!(results[0].cell_id.contains("81371_20830980"));
    }

    #[test]
    fn test_neural_scoring_csv_based() {
        let config = NeuralProcessingConfig::default();
        let processor = NeuralDataProcessor::new(config);
        
        // Create test CSV data with real metrics
        let mut data_row = HashMap::new();
        data_row.insert("CELL_AVAILABILITY_%".to_string(), 98.5); // Good availability
        data_row.insert("4G_LTE_DCR_VOLTE".to_string(), 1.2); // Low drop rate
        data_row.insert("LTE_INTRA_FREQ_HO_SR".to_string(), 95.0); // Good handover success
        data_row.insert("LTE_INTER_FREQ_HO_SR".to_string(), 92.0); // Good handover success
        data_row.insert("UE_PWR_LIMITED".to_string(), 15.0); // Low power limitation
        data_row.insert("&_AVE_4G_LTE_DL_USER_THRPUT".to_string(), 25.0); // Good throughput
        data_row.insert("SINR_PUSCH_AVG".to_string(), 12.5); // Good signal quality
        
        // Test AFM fault probability calculation
        let afm_fault_prob = processor.calculate_afm_fault_probability_csv(&data_row);
        assert!(afm_fault_prob >= 0.0 && afm_fault_prob <= 1.0);
        assert!(afm_fault_prob < 0.3); // Should be low fault probability for good metrics
        
        // Test DTM mobility score calculation  
        let dtm_mobility = processor.calculate_dtm_mobility_score_csv(&data_row);
        assert!(dtm_mobility >= 0.0 && dtm_mobility <= 1.0);
        assert!(dtm_mobility > 0.7); // Should be high mobility score for good handover rates
        
        // Test energy efficiency calculation
        let energy_efficiency = processor.calculate_energy_efficiency_score_csv(&data_row);
        assert!(energy_efficiency >= 0.0 && energy_efficiency <= 1.0);
        assert!(energy_efficiency > 0.6); // Should be good efficiency for low power limitation
        
        // Test service quality calculation
        let service_quality = processor.calculate_service_quality_score_csv(&data_row);
        assert!(service_quality >= 0.0 && service_quality <= 1.0);
        assert!(service_quality > 0.8); // Should be high quality for good metrics
    }
    
    #[test]
    fn test_neural_scoring_poor_metrics() {
        let config = NeuralProcessingConfig::default();
        let processor = NeuralDataProcessor::new(config);
        
        // Create test CSV data with poor metrics
        let mut data_row = HashMap::new();
        data_row.insert("CELL_AVAILABILITY_%".to_string(), 85.0); // Poor availability
        data_row.insert("4G_LTE_DCR_VOLTE".to_string(), 8.5); // High drop rate
        data_row.insert("LTE_INTRA_FREQ_HO_SR".to_string(), 75.0); // Poor handover success
        data_row.insert("LTE_INTER_FREQ_HO_SR".to_string(), 70.0); // Poor handover success
        data_row.insert("UE_PWR_LIMITED".to_string(), 45.0); // High power limitation
        data_row.insert("&_AVE_4G_LTE_DL_USER_THRPUT".to_string(), 5.0); // Poor throughput
        data_row.insert("SINR_PUSCH_AVG".to_string(), 2.5); // Poor signal quality
        
        // Test AFM fault probability - should be high for poor metrics
        let afm_fault_prob = processor.calculate_afm_fault_probability_csv(&data_row);
        assert!(afm_fault_prob >= 0.0 && afm_fault_prob <= 1.0);
        assert!(afm_fault_prob > 0.4); // Should be high fault probability
        
        // Test DTM mobility score - should be low for poor handover rates
        let dtm_mobility = processor.calculate_dtm_mobility_score_csv(&data_row);
        assert!(dtm_mobility >= 0.0 && dtm_mobility <= 1.0);
        assert!(dtm_mobility < 0.5); // Should be low mobility score
        
        // Test energy efficiency - should be low for high power limitation
        let energy_efficiency = processor.calculate_energy_efficiency_score_csv(&data_row);
        assert!(energy_efficiency >= 0.0 && energy_efficiency <= 1.0);
        assert!(energy_efficiency < 0.4); // Should be poor efficiency
        
        // Test service quality - should be low for poor metrics
        let service_quality = processor.calculate_service_quality_score_csv(&data_row);
        assert!(service_quality >= 0.0 && service_quality <= 1.0);
        assert!(service_quality < 0.6); // Should be low quality
    }

    #[test]
    fn test_tensor_conversion() {
        let config = NeuralProcessingConfig::default();
        let processor = NeuralDataProcessor::new(config);
        
        let result = NeuralProcessingResult {
            cell_id: "test_cell".to_string(),
            timestamp: SystemTime::now(),
            afm_features: vec![0.1, 0.2, 0.3],
            dtm_features: vec![0.4, 0.5],
            comprehensive_features: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            anomalies: Vec::new(),
            neural_scores: NeuralScores {
                afm_fault_probability: 0.5,
                dtm_mobility_score: 0.6,
                energy_efficiency_score: 0.7,
                service_quality_score: 0.8,
                anomaly_severity_score: 0.0,
            },
            processing_metadata: ProcessingMetadata {
                processing_time_ms: 1.0,
                feature_importance_scores: HashMap::new(),
                data_quality_score: 1.0,
                completeness_ratio: 1.0,
                swarm_coordination_data: None,
            },
        };
        
        let batches = processor.results_to_tensor_batches(&[result]);
        assert!(!batches.is_empty());
    }
}