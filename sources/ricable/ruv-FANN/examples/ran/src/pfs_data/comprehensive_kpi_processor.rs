//! Comprehensive KPI Data Processor for Network Intelligence
//! 
//! Advanced data processing pipeline that handles the complete fanndata.csv format
//! with 100+ metrics including signal quality, performance, mobility, and ENDC capabilities.
//! Integrates with neural networks for real-time anomaly detection and feature extraction.

use crate::pfs_data::csv_data_parser::{CsvDataParser, ParsedCsvDataset, CsvParsingConfig};
use crate::pfs_data::neural_data_processor::{NeuralDataProcessor, NeuralProcessingConfig};
use crate::pfs_data::kpi::{KpiMappings, KpiCalculator};
use std::collections::HashMap;
use std::path::Path;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH, Instant};

/// Comprehensive KPI data processor for network intelligence
#[derive(Debug)]
pub struct ComprehensiveKpiProcessor {
    pub csv_parser: CsvDataParser,
    pub neural_processor: NeuralDataProcessor,
    pub kpi_calculator: KpiCalculator,
    pub kpi_mappings: KpiMappings,
    pub processing_config: ProcessingConfig,
    pub real_time_buffer: Vec<ProcessedKpiRow>,
    pub anomaly_detection_enabled: bool,
    pub feature_extraction_enabled: bool,
}

/// Configuration for comprehensive processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub enable_neural_processing: bool,
    pub enable_anomaly_detection: bool,
    pub enable_feature_extraction: bool,
    pub enable_temporal_analysis: bool,
    pub batch_size: usize,
    pub parallel_processing: bool,
    pub real_time_buffer_size: usize,
    pub anomaly_threshold: f32,
    pub feature_vector_dimensions: usize,
}

/// Processed KPI row with enhanced analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedKpiRow {
    pub timestamp: String,
    pub cell_id: String,
    pub enodeb_name: String,
    pub cell_name: String,
    pub frequency_band: String,
    
    // Signal Quality Metrics
    pub signal_quality: SignalQualityMetrics,
    
    // Performance KPIs
    pub performance_kpis: PerformanceKpis,
    
    // Mobility Metrics
    pub mobility_metrics: MobilityMetrics,
    
    // Quality Metrics (BLER, Packet Loss)
    pub quality_metrics: QualityMetrics,
    
    // ENDC Capabilities and Performance
    pub endc_metrics: EndcMetrics,
    
    // Neural Network Features
    pub neural_features: NeuralFeatures,
    
    // Anomaly Detection Results
    pub anomaly_detection: AnomalyDetection,
    
    // Calculated KPIs
    pub calculated_kpis: HashMap<String, f64>,
    
    // Temporal Analysis
    pub temporal_analysis: Option<TemporalAnalysis>,
}

/// Signal quality metrics from CSV
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQualityMetrics {
    pub sinr_pusch_avg: f64,
    pub sinr_pucch_avg: f64,
    pub ul_rssi_pucch: f64,
    pub ul_rssi_pusch: f64,
    pub ul_rssi_total: f64,
    pub signal_quality_score: f64, // Calculated composite score
}

/// Performance KPIs extracted and calculated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceKpis {
    pub cell_availability_pct: f64,
    pub volte_traffic_erl: f64,
    pub eric_traff_erab_erl: f64,
    pub rrc_connected_users_avg: f64,
    pub ul_volume_pdcp_gbytes: f64,
    pub dl_volume_pdcp_gbytes: f64,
    pub lte_dcr_volte: f64,
    pub erab_drop_rate_qci_5: f64,
    pub erab_drop_rate_qci_8: f64,
    pub ue_context_abnormal_rel_pct: f64,
    pub cssr_end_user_pct: f64,
    pub eric_erab_init_setup_sr: f64,
    pub rrc_reestab_sr: f64,
    pub ave_4g_lte_dl_user_thrput: f64,
    pub ave_4g_lte_ul_user_thrput: f64,
    pub performance_score: f64, // Calculated composite score
}

/// Mobility and handover metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobilityMetrics {
    pub lte_intra_freq_ho_sr: f64,
    pub lte_inter_freq_ho_sr: f64,
    pub inter_freq_ho_attempts: u32,
    pub intra_freq_ho_attempts: u32,
    pub eric_ho_osc_intra: f64,
    pub eric_ho_osc_inter: f64,
    pub eric_rwr_total: f64,
    pub eric_rwr_gsm_rate: f64,
    pub eric_rwr_lte_rate: f64,
    pub eric_rwr_wcdma_rate: f64,
    pub eric_srvcc3g_exesr: f64,
    pub eric_srvcc3g_intens: f64,
    pub eric_srvcc3g_prepsr: f64,
    pub eric_srvcc2g_exesr: f64,
    pub eric_srvcc2g_intens: f64,
    pub eric_srvcc2g_prepsr: f64,
    pub mobility_score: f64, // Calculated composite score
}

/// Quality metrics (BLER, packet loss, latency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub mac_dl_bler: f64,
    pub mac_ul_bler: f64,
    pub dl_packet_error_loss_rate: f64,
    pub dl_packet_error_loss_rate_qci_1: f64,
    pub dl_packet_loss_qci_5: f64,
    pub dl_packet_error_loss_rate_qci_8: f64,
    pub ul_packet_loss_rate: f64,
    pub ul_packet_error_loss_qci_1: f64,
    pub ul_packet_error_loss_qci_5: f64,
    pub ul_packet_loss_rate_qci_8: f64,
    pub dl_latency_avg: f64,
    pub dl_latency_avg_qci_1: f64,
    pub dl_latency_avg_qci_5: f64,
    pub dl_latency_avg_qci_8: f64,
    pub ue_pwr_limited: f64,
    pub voip_integrity_cell_rate: f64,
    pub quality_score: f64, // Calculated composite score
}

/// ENDC capabilities and performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndcMetrics {
    pub sum_pmmeasconfigb1endc: u32,
    pub sum_pmendcsetupuesucc: u32,
    pub sum_pmendcsetupueatt: u32,
    pub sum_pmb1measrependcconfigrestart: u32,
    pub sum_pmb1measrependcconfig: u32,
    pub sum_pmendccapableue: u32,
    pub sum_pmendcsetupfailnrra: u32,
    pub active_ues_dl: u32,
    pub active_ues_ul: u32,
    pub active_user_dl_qci_1: u32,
    pub active_user_dl_qci_5: u32,
    pub active_user_dl_qci_8: u32,
    pub endc_establishment_att: u32,
    pub endc_establishment_succ: u32,
    pub endc_nb_received_b1_reports: u32,
    pub endc_nb_ue_config_b1_reports: u32,
    pub endc_nr_ra_scg_failures: u32,
    pub endc_scg_failure_ratio: f64,
    pub endc_setup_sr: f64,
    pub nb_endc_capables_ue_setup: u32,
    pub endc_mn_mcg_bearer_relocation_att: u32,
    pub endc_mn_mcg_bearer_relocation_sr: f64,
    pub endc_performance_score: f64, // Calculated composite score
}

/// Neural network features for ML training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralFeatures {
    pub afm_features: Vec<f32>,        // Anomaly/Fault Management features
    pub dtm_features: Vec<f32>,        // Data Traffic Management features
    pub comprehensive_features: Vec<f32>, // Combined feature vector
    pub feature_importance: HashMap<String, f32>,
    pub dimensionality_reduction: Option<Vec<f32>>, // PCA/t-SNE reduced features
}

/// Anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub availability_anomaly: bool,
    pub throughput_anomaly: bool,
    pub quality_anomaly: bool,
    pub error_rate_anomaly: bool,
    pub handover_anomaly: bool,
    pub endc_anomaly: bool,
    pub critical_fault_detected: bool,
    pub anomaly_severity_score: f32,
    pub anomaly_explanation: Vec<String>,
    pub recommended_actions: Vec<String>,
}

/// Temporal analysis for time-series patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    pub trend_direction: TrendDirection,
    pub seasonality_detected: bool,
    pub change_points: Vec<ChangePoint>,
    pub prediction_confidence: f32,
    pub next_hour_prediction: HashMap<String, f64>,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Change point detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    pub timestamp: String,
    pub metric_name: String,
    pub change_magnitude: f64,
    pub change_type: ChangeType,
}

/// Change type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    StepChange,
    GradualChange,
    AnomalousSpike,
    AnomalousDropout,
}

/// Comprehensive processing results
#[derive(Debug, Clone, Serialize)]
pub struct ComprehensiveProcessingResult {
    pub processed_rows: Vec<ProcessedKpiRow>,
    pub summary_statistics: SummaryStatistics,
    pub global_anomalies: Vec<GlobalAnomaly>,
    pub network_insights: NetworkInsights,
    pub recommendations: Vec<NetworkRecommendation>,
    pub processing_metadata: ProcessingMetadata,
}

/// Summary statistics for the entire dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryStatistics {
    pub total_cells: usize,
    pub total_enodebs: usize,
    pub time_span: String,
    pub average_availability: f64,
    pub average_throughput_dl: f64,
    pub average_throughput_ul: f64,
    pub total_anomalies: usize,
    pub critical_cells: usize,
    pub degraded_cells: usize,
    pub healthy_cells: usize,
    pub endc_capable_cells: usize,
    pub endc_performance_avg: f64,
}

/// Global anomalies affecting multiple cells
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalAnomaly {
    pub anomaly_type: String,
    pub affected_cells: Vec<String>,
    pub severity: GlobalAnomalySeverity,
    pub description: String,
    pub recommended_action: String,
    pub estimated_impact: f64,
}

/// Global anomaly severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GlobalAnomalySeverity {
    Critical,
    Warning,
    Info,
}

/// Network insights from comprehensive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInsights {
    pub coverage_quality: f64,
    pub capacity_utilization: f64,
    pub mobility_performance: f64,
    pub service_quality: f64,
    pub energy_efficiency: f64,
    pub endc_readiness: f64,
    pub predicted_growth: f64,
    pub optimization_potential: f64,
}

/// Network recommendations based on analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub description: String,
    pub affected_cells: Vec<String>,
    pub expected_improvement: f64,
    pub implementation_effort: ImplementationEffort,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Coverage,
    Capacity,
    Quality,
    Mobility,
    Energy,
    FiveG,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation effort estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,     // Configuration change
    Medium,  // Parameter optimization
    High,    // Hardware upgrade/replacement
}

/// Processing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub processing_time_ms: f64,
    pub rows_processed: usize,
    pub rows_per_second: f64,
    pub memory_usage_mb: f64,
    pub neural_processing_time_ms: f64,
    pub anomaly_detection_time_ms: f64,
    pub feature_extraction_time_ms: f64,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            enable_neural_processing: true,
            enable_anomaly_detection: true,
            enable_feature_extraction: true,
            enable_temporal_analysis: true,
            batch_size: 32,
            parallel_processing: true,
            real_time_buffer_size: 1000,
            anomaly_threshold: 0.75,
            feature_vector_dimensions: 50,
        }
    }
}

impl ComprehensiveKpiProcessor {
    /// Create new comprehensive KPI processor
    pub fn new() -> Self {
        let config = ProcessingConfig::default();
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: ProcessingConfig) -> Self {
        // Configure CSV parser
        let csv_config = CsvParsingConfig {
            delimiter: b';',
            has_headers: true,
            batch_size: config.batch_size,
            max_errors_before_abort: 100,
            parallel_processing: config.parallel_processing,
            validate_data_ranges: true,
            skip_empty_rows: true,
            strict_column_count: true,
            expected_column_count: 101,
        };

        // Configure neural processor
        let neural_config = NeuralProcessingConfig {
            batch_size: config.batch_size,
            feature_vector_size: config.feature_vector_dimensions,
            anomaly_threshold: config.anomaly_threshold,
            cache_enabled: true,
            parallel_processing: config.parallel_processing,
            real_time_processing: true,
            swarm_coordination_enabled: true,
        };

        Self {
            csv_parser: CsvDataParser::with_config(csv_config),
            neural_processor: NeuralDataProcessor::new(neural_config),
            kpi_calculator: KpiCalculator::new(),
            kpi_mappings: KpiMappings::new(),
            processing_config: config.clone(),
            real_time_buffer: Vec::with_capacity(config.real_time_buffer_size),
            anomaly_detection_enabled: config.enable_anomaly_detection,
            feature_extraction_enabled: config.enable_feature_extraction,
        }
    }

    /// Process CSV file with comprehensive analysis
    pub fn process_csv_file<P: AsRef<Path>>(&mut self, file_path: P) -> Result<ComprehensiveProcessingResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        println!("🚀 Starting comprehensive KPI processing...");
        
        // Step 1: Parse CSV data
        println!("📊 Step 1: Parsing CSV data...");
        let csv_dataset = self.csv_parser.parse_csv_file(&file_path)?;
        
        // Step 2: Process with neural intelligence
        println!("🧠 Step 2: Neural processing...");
        let neural_start = Instant::now();
        let neural_results = if self.processing_config.enable_neural_processing {
            let csv_content = self.convert_parsed_to_csv(&csv_dataset);
            self.neural_processor.process_csv_data(&csv_content)
        } else {
            Vec::new()
        };
        let neural_time = neural_start.elapsed().as_millis() as f64;

        // Step 3: Extract comprehensive features and detect anomalies
        println!("🔍 Step 3: Feature extraction and anomaly detection...");
        let feature_start = Instant::now();
        let processed_rows = self.process_rows_comprehensive(&csv_dataset, &neural_results)?;
        let feature_time = feature_start.elapsed().as_millis() as f64;

        // Step 4: Generate summary statistics
        println!("📈 Step 4: Generating summary statistics...");
        let summary_statistics = self.calculate_summary_statistics(&processed_rows);

        // Step 5: Detect global anomalies
        println!("🚨 Step 5: Detecting global anomalies...");
        let anomaly_start = Instant::now();
        let global_anomalies = self.detect_global_anomalies(&processed_rows);
        let anomaly_time = anomaly_start.elapsed().as_millis() as f64;

        // Step 6: Generate network insights
        println!("💡 Step 6: Generating network insights...");
        let network_insights = self.generate_network_insights(&processed_rows, &summary_statistics);

        // Step 7: Generate recommendations
        println!("📋 Step 7: Generating recommendations...");
        let recommendations = self.generate_recommendations(&processed_rows, &global_anomalies, &network_insights);

        // Calculate processing metadata
        let total_time = start_time.elapsed().as_millis() as f64;
        let processing_metadata = ProcessingMetadata {
            processing_time_ms: total_time,
            rows_processed: processed_rows.len(),
            rows_per_second: processed_rows.len() as f64 / (total_time / 1000.0),
            memory_usage_mb: self.estimate_memory_usage(&processed_rows),
            neural_processing_time_ms: neural_time,
            anomaly_detection_time_ms: anomaly_time,
            feature_extraction_time_ms: feature_time,
        };

        println!("✅ Comprehensive processing complete!");
        println!("   📊 Processed: {} rows", processed_rows.len());
        println!("   ⏱️ Total time: {:.2}s", total_time / 1000.0);
        println!("   🧠 Neural time: {:.2}s", neural_time / 1000.0);
        println!("   🔍 Feature time: {:.2}s", feature_time / 1000.0);
        println!("   🚨 Anomaly time: {:.2}s", anomaly_time / 1000.0);
        println!("   💾 Memory usage: {:.2} MB", processing_metadata.memory_usage_mb);

        Ok(ComprehensiveProcessingResult {
            processed_rows,
            summary_statistics,
            global_anomalies,
            network_insights,
            recommendations,
            processing_metadata,
        })
    }

    /// Process rows with comprehensive analysis
    fn process_rows_comprehensive(&mut self, csv_dataset: &ParsedCsvDataset, neural_results: &[crate::pfs_data::neural_data_processor::NeuralProcessingResult]) -> Result<Vec<ProcessedKpiRow>, Box<dyn std::error::Error>> {
        let mut processed_rows = Vec::new();

        for (i, csv_row) in csv_dataset.rows.iter().enumerate() {
            // Get corresponding neural result if available
            let neural_result = neural_results.get(i);

            // Extract signal quality metrics
            let signal_quality = SignalQualityMetrics {
                sinr_pusch_avg: csv_row.quality_indicators.sinr_pusch_avg,
                sinr_pucch_avg: csv_row.quality_indicators.sinr_pucch_avg,
                ul_rssi_pucch: csv_row.quality_indicators.ul_rssi_pucch,
                ul_rssi_pusch: csv_row.quality_indicators.ul_rssi_pusch,
                ul_rssi_total: csv_row.quality_indicators.ul_rssi_total,
                signal_quality_score: self.calculate_signal_quality_score(&csv_row.quality_indicators),
            };

            // Extract performance KPIs
            let performance_kpis = PerformanceKpis {
                cell_availability_pct: csv_row.metrics.cell_availability_pct,
                volte_traffic_erl: csv_row.metrics.volte_traffic_erl,
                eric_traff_erab_erl: csv_row.metrics.eric_traff_erab_erl,
                rrc_connected_users_avg: csv_row.metrics.rrc_connected_users_avg,
                ul_volume_pdcp_gbytes: csv_row.metrics.ul_volume_pdcp_gbytes,
                dl_volume_pdcp_gbytes: csv_row.metrics.dl_volume_pdcp_gbytes,
                lte_dcr_volte: csv_row.performance_kpis.lte_dcr_volte,
                erab_drop_rate_qci_5: csv_row.performance_kpis.erab_drop_rate_qci_5,
                erab_drop_rate_qci_8: csv_row.performance_kpis.erab_drop_rate_qci_8,
                ue_context_abnormal_rel_pct: csv_row.performance_kpis.ue_context_abnormal_rel_pct,
                cssr_end_user_pct: csv_row.performance_kpis.cssr_end_user_pct,
                eric_erab_init_setup_sr: csv_row.performance_kpis.eric_erab_init_setup_sr,
                rrc_reestab_sr: csv_row.performance_kpis.rrc_reestab_sr,
                ave_4g_lte_dl_user_thrput: csv_row.performance_kpis.ave_4g_lte_dl_user_thrput,
                ave_4g_lte_ul_user_thrput: csv_row.performance_kpis.ave_4g_lte_ul_user_thrput,
                performance_score: self.calculate_performance_score(&csv_row.performance_kpis),
            };

            // Extract mobility metrics
            let mobility_metrics = MobilityMetrics {
                lte_intra_freq_ho_sr: csv_row.handover_metrics.lte_intra_freq_ho_sr,
                lte_inter_freq_ho_sr: csv_row.handover_metrics.lte_inter_freq_ho_sr,
                inter_freq_ho_attempts: csv_row.handover_metrics.inter_freq_ho_attempts,
                intra_freq_ho_attempts: csv_row.handover_metrics.intra_freq_ho_attempts,
                eric_ho_osc_intra: csv_row.handover_metrics.eric_ho_osc_intra,
                eric_ho_osc_inter: csv_row.handover_metrics.eric_ho_osc_inter,
                eric_rwr_total: csv_row.handover_metrics.eric_rwr_total,
                eric_rwr_gsm_rate: 0.0, // Not in parsed data, would need additional CSV columns
                eric_rwr_lte_rate: csv_row.handover_metrics.eric_rwr_lte_rate,
                eric_rwr_wcdma_rate: 0.0, // Not in parsed data
                eric_srvcc3g_exesr: 0.0, // Not in parsed data
                eric_srvcc3g_intens: 0.0, // Not in parsed data
                eric_srvcc3g_prepsr: 0.0, // Not in parsed data
                eric_srvcc2g_exesr: 0.0, // Not in parsed data
                eric_srvcc2g_intens: 0.0, // Not in parsed data
                eric_srvcc2g_prepsr: 0.0, // Not in parsed data
                mobility_score: self.calculate_mobility_score(&csv_row.handover_metrics),
            };

            // Extract quality metrics
            let quality_metrics = QualityMetrics {
                mac_dl_bler: csv_row.quality_indicators.mac_dl_bler,
                mac_ul_bler: csv_row.quality_indicators.mac_ul_bler,
                dl_packet_error_loss_rate: csv_row.traffic_data.dl_packet_error_loss_rate,
                dl_packet_error_loss_rate_qci_1: 0.0, // Would need additional parsing
                dl_packet_loss_qci_5: 0.0, // Would need additional parsing
                dl_packet_error_loss_rate_qci_8: 0.0, // Would need additional parsing
                ul_packet_loss_rate: csv_row.traffic_data.ul_packet_loss_rate,
                ul_packet_error_loss_qci_1: 0.0, // Would need additional parsing
                ul_packet_error_loss_qci_5: 0.0, // Would need additional parsing
                ul_packet_loss_rate_qci_8: 0.0, // Would need additional parsing
                dl_latency_avg: csv_row.traffic_data.dl_latency_avg,
                dl_latency_avg_qci_1: csv_row.traffic_data.dl_latency_avg_qci_1,
                dl_latency_avg_qci_5: csv_row.traffic_data.dl_latency_avg_qci_5,
                dl_latency_avg_qci_8: csv_row.traffic_data.dl_latency_avg_qci_8,
                ue_pwr_limited: 0.0, // Would need additional parsing
                voip_integrity_cell_rate: 0.0, // Would need additional parsing
                quality_score: self.calculate_quality_score(&csv_row.quality_indicators, &csv_row.traffic_data),
            };

            // Extract ENDC metrics
            let endc_metrics = EndcMetrics {
                sum_pmmeasconfigb1endc: csv_row.endc_metrics.pmmeasconfigb1endc,
                sum_pmendcsetupuesucc: csv_row.endc_metrics.pmendcsetupuesucc,
                sum_pmendcsetupueatt: csv_row.endc_metrics.pmendcsetupueatt,
                sum_pmb1measrependcconfigrestart: 0, // Would need additional parsing
                sum_pmb1measrependcconfig: 0, // Would need additional parsing
                sum_pmendccapableue: 0, // Would need additional parsing
                sum_pmendcsetupfailnrra: 0, // Would need additional parsing
                active_ues_dl: csv_row.metrics.active_ues_dl,
                active_ues_ul: csv_row.metrics.active_ues_ul,
                active_user_dl_qci_1: csv_row.traffic_data.active_user_dl_qci_1,
                active_user_dl_qci_5: csv_row.traffic_data.active_user_dl_qci_5,
                active_user_dl_qci_8: csv_row.traffic_data.active_user_dl_qci_8,
                endc_establishment_att: csv_row.endc_metrics.endc_establishment_att,
                endc_establishment_succ: csv_row.endc_metrics.endc_establishment_succ,
                endc_nb_received_b1_reports: 0, // Would need additional parsing
                endc_nb_ue_config_b1_reports: 0, // Would need additional parsing
                endc_nr_ra_scg_failures: 0, // Would need additional parsing
                endc_scg_failure_ratio: csv_row.endc_metrics.endc_scg_failure_ratio,
                endc_setup_sr: csv_row.endc_metrics.endc_setup_sr,
                nb_endc_capables_ue_setup: csv_row.endc_metrics.nb_endc_capables_ue_setup,
                endc_mn_mcg_bearer_relocation_att: 0, // Would need additional parsing
                endc_mn_mcg_bearer_relocation_sr: 0.0, // Would need additional parsing
                endc_performance_score: self.calculate_endc_score(&csv_row.endc_metrics),
            };

            // Extract neural features
            let neural_features = if let Some(neural_result) = neural_result {
                NeuralFeatures {
                    afm_features: neural_result.afm_features.clone(),
                    dtm_features: neural_result.dtm_features.clone(),
                    comprehensive_features: neural_result.comprehensive_features.clone(),
                    feature_importance: neural_result.processing_metadata.feature_importance_scores.clone(),
                    dimensionality_reduction: None, // Could implement PCA/t-SNE here
                }
            } else {
                NeuralFeatures {
                    afm_features: Vec::new(),
                    dtm_features: Vec::new(),
                    comprehensive_features: Vec::new(),
                    feature_importance: HashMap::new(),
                    dimensionality_reduction: None,
                }
            };

            // Perform anomaly detection
            let anomaly_detection = self.detect_comprehensive_anomalies(
                &signal_quality,
                &performance_kpis,
                &mobility_metrics,
                &quality_metrics,
                &endc_metrics,
                &csv_row.anomaly_flags,
            );

            // Calculate additional KPIs
            let calculated_kpis = self.calculate_additional_kpis(&csv_row);

            // Create processed row
            let processed_row = ProcessedKpiRow {
                timestamp: csv_row.timestamp.clone(),
                cell_id: format!("{}_{}", csv_row.cell_identifier.enodeb_code, csv_row.cell_identifier.cell_code),
                enodeb_name: csv_row.cell_identifier.enodeb_name.clone(),
                cell_name: csv_row.cell_identifier.cell_name.clone(),
                frequency_band: csv_row.cell_identifier.frequency_band.clone(),
                signal_quality,
                performance_kpis,
                mobility_metrics,
                quality_metrics,
                endc_metrics,
                neural_features,
                anomaly_detection,
                calculated_kpis,
                temporal_analysis: None, // Would implement temporal analysis here
            };

            processed_rows.push(processed_row);
        }

        Ok(processed_rows)
    }

    /// Calculate signal quality composite score
    fn calculate_signal_quality_score(&self, quality: &crate::pfs_data::csv_data_parser::QualityMetrics) -> f64 {
        let sinr_avg = (quality.sinr_pusch_avg + quality.sinr_pucch_avg) / 2.0;
        let rssi_avg = (quality.ul_rssi_pucch + quality.ul_rssi_pusch) / 2.0;
        
        // Normalize SINR (good: >15 dB, poor: <5 dB)
        let sinr_score = ((sinr_avg - 5.0) / 10.0).clamp(0.0, 1.0);
        
        // Normalize RSSI (good: >-100 dBm, poor: <-120 dBm)
        let rssi_score = ((rssi_avg + 120.0) / 20.0).clamp(0.0, 1.0);
        
        (sinr_score * 0.7 + rssi_score * 0.3) * 100.0
    }

    /// Calculate performance composite score
    fn calculate_performance_score(&self, performance: &crate::pfs_data::csv_data_parser::PerformanceKpis) -> f64 {
        let availability_score = performance.ave_4g_lte_dl_user_thrput / 100.0;
        let throughput_score = (performance.ave_4g_lte_dl_user_thrput / 50.0).min(1.0);
        let error_score = 1.0 - (performance.erab_drop_rate_qci_5 / 5.0).min(1.0);
        
        (availability_score * 0.4 + throughput_score * 0.4 + error_score * 0.2) * 100.0
    }

    /// Calculate mobility composite score
    fn calculate_mobility_score(&self, mobility: &crate::pfs_data::csv_data_parser::HandoverData) -> f64 {
        let intra_score = mobility.lte_intra_freq_ho_sr / 100.0;
        let inter_score = mobility.lte_inter_freq_ho_sr / 100.0;
        
        (intra_score * 0.6 + inter_score * 0.4) * 100.0
    }

    /// Calculate quality composite score
    fn calculate_quality_score(&self, quality: &crate::pfs_data::csv_data_parser::QualityMetrics, traffic: &crate::pfs_data::csv_data_parser::TrafficData) -> f64 {
        let bler_score = 1.0 - ((quality.mac_dl_bler + quality.mac_ul_bler) / 2.0 / 10.0).min(1.0);
        let latency_score = 1.0 - (traffic.dl_latency_avg / 50.0).min(1.0);
        let packet_loss_score = 1.0 - (traffic.dl_packet_error_loss_rate / 5.0).min(1.0);
        
        (bler_score * 0.4 + latency_score * 0.3 + packet_loss_score * 0.3) * 100.0
    }

    /// Calculate ENDC composite score
    fn calculate_endc_score(&self, endc: &crate::pfs_data::csv_data_parser::EndcData) -> f64 {
        if endc.endc_establishment_att == 0 {
            return 0.0;
        }
        
        let setup_success_rate = endc.endc_establishment_succ as f64 / endc.endc_establishment_att as f64;
        let failure_penalty = 1.0 - (endc.endc_scg_failure_ratio / 10.0).min(1.0);
        
        (setup_success_rate * 0.7 + failure_penalty * 0.3) * 100.0
    }

    /// Detect comprehensive anomalies
    fn detect_comprehensive_anomalies(
        &self,
        signal_quality: &SignalQualityMetrics,
        performance: &PerformanceKpis,
        mobility: &MobilityMetrics,
        quality: &QualityMetrics,
        endc: &EndcMetrics,
        original_flags: &crate::pfs_data::csv_data_parser::AnomalyFlags,
    ) -> AnomalyDetection {
        let mut anomaly_explanations = Vec::new();
        let mut recommended_actions = Vec::new();
        let mut severity_score = 0.0f32;

        // Check availability anomaly
        let availability_anomaly = performance.cell_availability_pct < 95.0;
        if availability_anomaly {
            severity_score += 0.3;
            anomaly_explanations.push(format!("Low cell availability: {:.1}%", performance.cell_availability_pct));
            recommended_actions.push("Investigate hardware failures and maintenance schedules".to_string());
        }

        // Check throughput anomaly
        let throughput_anomaly = performance.ave_4g_lte_dl_user_thrput < 10.0 || performance.ave_4g_lte_ul_user_thrput < 5.0;
        if throughput_anomaly {
            severity_score += 0.2;
            anomaly_explanations.push(format!("Low throughput - DL: {:.1} Mbps, UL: {:.1} Mbps", 
                performance.ave_4g_lte_dl_user_thrput, performance.ave_4g_lte_ul_user_thrput));
            recommended_actions.push("Check RF conditions and capacity planning".to_string());
        }

        // Check quality anomaly
        let quality_anomaly = quality.mac_dl_bler > 8.0 || quality.mac_ul_bler > 8.0 || quality.dl_latency_avg > 50.0;
        if quality_anomaly {
            severity_score += 0.2;
            anomaly_explanations.push("Poor service quality detected".to_string());
            recommended_actions.push("Optimize RF parameters and check interference".to_string());
        }

        // Check error rate anomaly
        let error_rate_anomaly = performance.erab_drop_rate_qci_5 > 3.0 || performance.erab_drop_rate_qci_8 > 3.0;
        if error_rate_anomaly {
            severity_score += 0.15;
            anomaly_explanations.push("High service drop rates detected".to_string());
            recommended_actions.push("Check transport network and core network connectivity".to_string());
        }

        // Check handover anomaly
        let handover_anomaly = mobility.lte_intra_freq_ho_sr < 90.0 || mobility.lte_inter_freq_ho_sr < 85.0;
        if handover_anomaly {
            severity_score += 0.1;
            anomaly_explanations.push("Poor handover performance".to_string());
            recommended_actions.push("Optimize neighbor relationships and handover parameters".to_string());
        }

        // Check ENDC anomaly
        let endc_anomaly = endc.endc_setup_sr < 90.0 && endc.endc_establishment_att > 0;
        if endc_anomaly {
            severity_score += 0.05;
            anomaly_explanations.push("Poor 5G NSA performance".to_string());
            recommended_actions.push("Check 5G radio conditions and inter-RAT parameters".to_string());
        }

        let critical_fault_detected = severity_score > 0.5;

        AnomalyDetection {
            availability_anomaly,
            throughput_anomaly,
            quality_anomaly,
            error_rate_anomaly,
            handover_anomaly,
            endc_anomaly,
            critical_fault_detected,
            anomaly_severity_score: severity_score.min(1.0),
            anomaly_explanation: anomaly_explanations,
            recommended_actions,
        }
    }

    /// Calculate additional KPIs using the KPI calculator
    fn calculate_additional_kpis(&self, csv_row: &crate::pfs_data::csv_data_parser::ParsedCsvRow) -> HashMap<String, f64> {
        let mut counters = HashMap::new();
        
        // Convert CSV row metrics to counter format
        counters.insert("cell_availability".to_string(), csv_row.metrics.cell_availability_pct);
        counters.insert("dl_throughput".to_string(), csv_row.performance_kpis.ave_4g_lte_dl_user_thrput);
        counters.insert("ul_throughput".to_string(), csv_row.performance_kpis.ave_4g_lte_ul_user_thrput);
        counters.insert("connected_users".to_string(), csv_row.metrics.rrc_connected_users_avg);
        
        // Calculate KPIs
        self.kpi_calculator.calculate_all_kpis(&counters)
    }

    /// Calculate summary statistics
    fn calculate_summary_statistics(&self, processed_rows: &[ProcessedKpiRow]) -> SummaryStatistics {
        if processed_rows.is_empty() {
            return SummaryStatistics {
                total_cells: 0,
                total_enodebs: 0,
                time_span: "No data".to_string(),
                average_availability: 0.0,
                average_throughput_dl: 0.0,
                average_throughput_ul: 0.0,
                total_anomalies: 0,
                critical_cells: 0,
                degraded_cells: 0,
                healthy_cells: 0,
                endc_capable_cells: 0,
                endc_performance_avg: 0.0,
            };
        }

        let total_cells = processed_rows.len();
        let unique_enodebs: std::collections::HashSet<_> = processed_rows.iter()
            .map(|row| &row.enodeb_name)
            .collect();
        let total_enodebs = unique_enodebs.len();

        let average_availability = processed_rows.iter()
            .map(|row| row.performance_kpis.cell_availability_pct)
            .sum::<f64>() / total_cells as f64;

        let average_throughput_dl = processed_rows.iter()
            .map(|row| row.performance_kpis.ave_4g_lte_dl_user_thrput)
            .sum::<f64>() / total_cells as f64;

        let average_throughput_ul = processed_rows.iter()
            .map(|row| row.performance_kpis.ave_4g_lte_ul_user_thrput)
            .sum::<f64>() / total_cells as f64;

        let total_anomalies = processed_rows.iter()
            .filter(|row| row.anomaly_detection.anomaly_severity_score > 0.0)
            .count();

        let critical_cells = processed_rows.iter()
            .filter(|row| row.anomaly_detection.critical_fault_detected)
            .count();

        let degraded_cells = processed_rows.iter()
            .filter(|row| row.anomaly_detection.anomaly_severity_score > 0.3 && !row.anomaly_detection.critical_fault_detected)
            .count();

        let healthy_cells = total_cells - critical_cells - degraded_cells;

        let endc_capable_cells = processed_rows.iter()
            .filter(|row| row.endc_metrics.endc_establishment_att > 0)
            .count();

        let endc_performance_avg = if endc_capable_cells > 0 {
            processed_rows.iter()
                .filter(|row| row.endc_metrics.endc_establishment_att > 0)
                .map(|row| row.endc_metrics.endc_performance_score)
                .sum::<f64>() / endc_capable_cells as f64
        } else {
            0.0
        };

        SummaryStatistics {
            total_cells,
            total_enodebs,
            time_span: "Single snapshot".to_string(), // Would need temporal analysis for proper time span
            average_availability,
            average_throughput_dl,
            average_throughput_ul,
            total_anomalies,
            critical_cells,
            degraded_cells,
            healthy_cells,
            endc_capable_cells,
            endc_performance_avg,
        }
    }

    /// Detect global anomalies affecting multiple cells
    fn detect_global_anomalies(&self, processed_rows: &[ProcessedKpiRow]) -> Vec<GlobalAnomaly> {
        let mut global_anomalies = Vec::new();

        // Check for widespread availability issues
        let low_availability_cells: Vec<_> = processed_rows.iter()
            .filter(|row| row.performance_kpis.cell_availability_pct < 95.0)
            .map(|row| row.cell_id.clone())
            .collect();

        if low_availability_cells.len() > processed_rows.len() / 10 { // More than 10% of cells
            global_anomalies.push(GlobalAnomaly {
                anomaly_type: "Widespread Availability Issues".to_string(),
                affected_cells: low_availability_cells,
                severity: GlobalAnomalySeverity::Critical,
                description: "Multiple cells showing availability degradation".to_string(),
                recommended_action: "Check core network and power systems".to_string(),
                estimated_impact: 0.8,
            });
        }

        // Check for throughput degradation pattern
        let low_throughput_cells: Vec<_> = processed_rows.iter()
            .filter(|row| row.performance_kpis.ave_4g_lte_dl_user_thrput < 10.0)
            .map(|row| row.cell_id.clone())
            .collect();

        if low_throughput_cells.len() > processed_rows.len() / 20 { // More than 5% of cells
            global_anomalies.push(GlobalAnomaly {
                anomaly_type: "Network Congestion Pattern".to_string(),
                affected_cells: low_throughput_cells,
                severity: GlobalAnomalySeverity::Warning,
                description: "Multiple cells showing poor throughput performance".to_string(),
                recommended_action: "Analyze traffic patterns and consider capacity expansion".to_string(),
                estimated_impact: 0.5,
            });
        }

        global_anomalies
    }

    /// Generate network insights
    fn generate_network_insights(&self, processed_rows: &[ProcessedKpiRow], summary: &SummaryStatistics) -> NetworkInsights {
        if processed_rows.is_empty() {
            return NetworkInsights {
                coverage_quality: 0.0,
                capacity_utilization: 0.0,
                mobility_performance: 0.0,
                service_quality: 0.0,
                energy_efficiency: 0.0,
                endc_readiness: 0.0,
                predicted_growth: 0.0,
                optimization_potential: 0.0,
            };
        }

        let coverage_quality = summary.average_availability;

        let capacity_utilization = processed_rows.iter()
            .map(|row| (row.performance_kpis.ave_4g_lte_dl_user_thrput + row.performance_kpis.ave_4g_lte_ul_user_thrput) / 100.0)
            .sum::<f64>() / processed_rows.len() as f64;

        let mobility_performance = processed_rows.iter()
            .map(|row| row.mobility_metrics.mobility_score)
            .sum::<f64>() / processed_rows.len() as f64;

        let service_quality = processed_rows.iter()
            .map(|row| row.quality_metrics.quality_score)
            .sum::<f64>() / processed_rows.len() as f64;

        let energy_efficiency = processed_rows.iter()
            .map(|row| row.signal_quality.signal_quality_score)
            .sum::<f64>() / processed_rows.len() as f64;

        let endc_readiness = if summary.endc_capable_cells > 0 {
            summary.endc_performance_avg
        } else {
            0.0
        };

        let predicted_growth = 0.0; // Would implement growth prediction algorithm

        let optimization_potential = 100.0 - (coverage_quality + service_quality + mobility_performance) / 3.0;

        NetworkInsights {
            coverage_quality,
            capacity_utilization,
            mobility_performance,
            service_quality,
            energy_efficiency,
            endc_readiness,
            predicted_growth,
            optimization_potential,
        }
    }

    /// Generate recommendations
    fn generate_recommendations(&self, processed_rows: &[ProcessedKpiRow], global_anomalies: &[GlobalAnomaly], insights: &NetworkInsights) -> Vec<NetworkRecommendation> {
        let mut recommendations = Vec::new();

        // Coverage recommendations
        if insights.coverage_quality < 95.0 {
            let affected_cells: Vec<_> = processed_rows.iter()
                .filter(|row| row.performance_kpis.cell_availability_pct < 95.0)
                .map(|row| row.cell_id.clone())
                .collect();

            recommendations.push(NetworkRecommendation {
                category: RecommendationCategory::Coverage,
                priority: RecommendationPriority::Critical,
                description: "Improve cell availability to meet SLA requirements".to_string(),
                affected_cells,
                expected_improvement: 95.0 - insights.coverage_quality,
                implementation_effort: ImplementationEffort::Medium,
            });
        }

        // Capacity recommendations
        if insights.capacity_utilization > 0.8 {
            recommendations.push(NetworkRecommendation {
                category: RecommendationCategory::Capacity,
                priority: RecommendationPriority::High,
                description: "Consider capacity expansion due to high utilization".to_string(),
                affected_cells: Vec::new(), // Network-wide recommendation
                expected_improvement: 0.2,
                implementation_effort: ImplementationEffort::High,
            });
        }

        // 5G recommendations
        if insights.endc_readiness < 80.0 && processed_rows.iter().any(|row| row.endc_metrics.endc_establishment_att > 0) {
            let endc_cells: Vec<_> = processed_rows.iter()
                .filter(|row| row.endc_metrics.endc_establishment_att > 0 && row.endc_metrics.endc_performance_score < 80.0)
                .map(|row| row.cell_id.clone())
                .collect();

            recommendations.push(NetworkRecommendation {
                category: RecommendationCategory::FiveG,
                priority: RecommendationPriority::Medium,
                description: "Optimize 5G NSA performance for better user experience".to_string(),
                affected_cells: endc_cells,
                expected_improvement: 80.0 - insights.endc_readiness,
                implementation_effort: ImplementationEffort::Medium,
            });
        }

        recommendations
    }

    /// Convert parsed dataset back to CSV format for neural processing
    fn convert_parsed_to_csv(&self, dataset: &ParsedCsvDataset) -> String {
        let mut csv_content = String::new();
        
        // Add simplified header (would need full column mapping for complete conversion)
        csv_content.push_str("HEURE(PSDATE);CODE_ELT_ENODEB;ENODEB;CODE_ELT_CELLULE;CELLULE;SYS.BANDE;SYS.NB_BANDES;CELL_AVAILABILITY_%\n");
        
        // Add data rows (simplified)
        for row in &dataset.rows {
            csv_content.push_str(&format!(
                "{};{};{};{};{};{};{};{}\n",
                row.timestamp,
                row.cell_identifier.enodeb_code,
                row.cell_identifier.enodeb_name,
                row.cell_identifier.cell_code,
                row.cell_identifier.cell_name,
                row.cell_identifier.frequency_band,
                row.cell_identifier.band_count,
                row.metrics.cell_availability_pct
            ));
        }
        
        csv_content
    }

    /// Estimate memory usage in MB
    fn estimate_memory_usage(&self, processed_rows: &[ProcessedKpiRow]) -> f64 {
        let row_size = std::mem::size_of::<ProcessedKpiRow>();
        (processed_rows.len() * row_size) as f64 / (1024.0 * 1024.0)
    }

    /// Export comprehensive results to JSON
    pub fn export_results(&self, result: &ComprehensiveProcessingResult, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(result)?;
        std::fs::write(file_path, json)?;
        Ok(())
    }

    /// Get real-time processing capability
    pub fn process_real_time_row(&mut self, csv_row: &str) -> Option<ProcessedKpiRow> {
        // Would implement real-time processing here
        // This is a placeholder for streaming data processing
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::io::Write;

    #[test]
    fn test_comprehensive_processor_creation() {
        let processor = ComprehensiveKpiProcessor::new();
        assert!(processor.processing_config.enable_neural_processing);
        assert!(processor.anomaly_detection_enabled);
        assert!(processor.feature_extraction_enabled);
    }

    #[test]
    fn test_signal_quality_score_calculation() {
        let processor = ComprehensiveKpiProcessor::new();
        let quality = crate::pfs_data::csv_data_parser::QualityMetrics {
            sinr_pusch_avg: 15.0,
            sinr_pucch_avg: 14.0,
            ul_rssi_pucch: -110.0,
            ul_rssi_pusch: -105.0,
            ul_rssi_total: -108.0,
            mac_dl_bler: 2.0,
            mac_ul_bler: 1.5,
        };

        let score = processor.calculate_signal_quality_score(&quality);
        assert!(score >= 0.0 && score <= 100.0);
        assert!(score > 50.0); // Should be good score for these values
    }

    #[test]
    fn test_performance_score_calculation() {
        let processor = ComprehensiveKpiProcessor::new();
        let performance = crate::pfs_data::csv_data_parser::PerformanceKpis {
            lte_dcr_volte: 0.5,
            erab_drop_rate_qci_5: 1.0,
            erab_drop_rate_qci_8: 1.5,
            ue_context_abnormal_rel_pct: 2.0,
            cssr_end_user_pct: 98.0,
            eric_erab_init_setup_sr: 95.0,
            rrc_reestab_sr: 90.0,
            ave_4g_lte_dl_user_thrput: 50.0,
            ave_4g_lte_ul_user_thrput: 25.0,
        };

        let score = processor.calculate_performance_score(&performance);
        assert!(score >= 0.0 && score <= 100.0);
    }

    #[test]
    fn test_anomaly_detection() {
        let processor = ComprehensiveKpiProcessor::new();
        
        // Create test metrics with poor values
        let signal_quality = SignalQualityMetrics {
            sinr_pusch_avg: 2.0,  // Poor signal
            sinr_pucch_avg: 1.5,
            ul_rssi_pucch: -125.0, // Poor RSSI
            ul_rssi_pusch: -130.0,
            ul_rssi_total: -127.5,
            signal_quality_score: 10.0,
        };

        let performance = PerformanceKpis {
            cell_availability_pct: 85.0, // Poor availability
            volte_traffic_erl: 1.0,
            eric_traff_erab_erl: 10.0,
            rrc_connected_users_avg: 50.0,
            ul_volume_pdcp_gbytes: 1.0,
            dl_volume_pdcp_gbytes: 5.0,
            lte_dcr_volte: 0.5,
            erab_drop_rate_qci_5: 5.0, // High drop rate
            erab_drop_rate_qci_8: 4.0,
            ue_context_abnormal_rel_pct: 2.0,
            cssr_end_user_pct: 98.0,
            eric_erab_init_setup_sr: 95.0,
            rrc_reestab_sr: 90.0,
            ave_4g_lte_dl_user_thrput: 5.0, // Poor throughput
            ave_4g_lte_ul_user_thrput: 2.0,
            performance_score: 40.0,
        };

        let mobility = MobilityMetrics {
            lte_intra_freq_ho_sr: 80.0, // Poor handover success
            lte_inter_freq_ho_sr: 75.0,
            inter_freq_ho_attempts: 50,
            intra_freq_ho_attempts: 100,
            eric_ho_osc_intra: 2.0,
            eric_ho_osc_inter: 1.5,
            eric_rwr_total: 3.0,
            eric_rwr_gsm_rate: 0.0,
            eric_rwr_lte_rate: 1.0,
            eric_rwr_wcdma_rate: 0.0,
            eric_srvcc3g_exesr: 0.0,
            eric_srvcc3g_intens: 0.0,
            eric_srvcc3g_prepsr: 0.0,
            eric_srvcc2g_exesr: 0.0,
            eric_srvcc2g_intens: 0.0,
            eric_srvcc2g_prepsr: 0.0,
            mobility_score: 70.0,
        };

        let quality = QualityMetrics {
            mac_dl_bler: 12.0, // High error rate
            mac_ul_bler: 10.0,
            dl_packet_error_loss_rate: 0.8,
            dl_packet_error_loss_rate_qci_1: 0.0,
            dl_packet_loss_qci_5: 0.0,
            dl_packet_error_loss_rate_qci_8: 0.0,
            ul_packet_loss_rate: 0.5,
            ul_packet_error_loss_qci_1: 0.0,
            ul_packet_error_loss_qci_5: 0.0,
            ul_packet_loss_rate_qci_8: 0.0,
            dl_latency_avg: 60.0, // High latency
            dl_latency_avg_qci_1: 12.0,
            dl_latency_avg_qci_5: 18.0,
            dl_latency_avg_qci_8: 20.0,
            ue_pwr_limited: 0.0,
            voip_integrity_cell_rate: 0.0,
            quality_score: 30.0,
        };

        let endc = EndcMetrics {
            sum_pmmeasconfigb1endc: 0,
            sum_pmendcsetupuesucc: 0,
            sum_pmendcsetupueatt: 0,
            sum_pmb1measrependcconfigrestart: 0,
            sum_pmb1measrependcconfig: 0,
            sum_pmendccapableue: 0,
            sum_pmendcsetupfailnrra: 0,
            active_ues_dl: 20,
            active_ues_ul: 15,
            active_user_dl_qci_1: 5,
            active_user_dl_qci_5: 10,
            active_user_dl_qci_8: 5,
            endc_establishment_att: 0,
            endc_establishment_succ: 0,
            endc_nb_received_b1_reports: 0,
            endc_nb_ue_config_b1_reports: 0,
            endc_nr_ra_scg_failures: 0,
            endc_scg_failure_ratio: 0.0,
            endc_setup_sr: 0.0,
            nb_endc_capables_ue_setup: 0,
            endc_mn_mcg_bearer_relocation_att: 0,
            endc_mn_mcg_bearer_relocation_sr: 0.0,
            endc_performance_score: 0.0,
        };

        let original_flags = crate::pfs_data::csv_data_parser::AnomalyFlags {
            availability_anomaly: false,
            throughput_anomaly: false,
            quality_anomaly: false,
            error_rate_anomaly: false,
            handover_anomaly: false,
            critical_fault_detected: false,
            anomaly_severity_score: 0.0,
        };

        let anomaly_detection = processor.detect_comprehensive_anomalies(
            &signal_quality,
            &performance,
            &mobility,
            &quality,
            &endc,
            &original_flags,
        );

        // Should detect multiple anomalies
        assert!(anomaly_detection.availability_anomaly);
        assert!(anomaly_detection.throughput_anomaly);
        assert!(anomaly_detection.quality_anomaly);
        assert!(anomaly_detection.error_rate_anomaly);
        assert!(anomaly_detection.handover_anomaly);
        assert!(anomaly_detection.critical_fault_detected);
        assert!(anomaly_detection.anomaly_severity_score > 0.5);
        assert!(!anomaly_detection.anomaly_explanation.is_empty());
        assert!(!anomaly_detection.recommended_actions.is_empty());
    }

    #[test]
    fn test_summary_statistics_calculation() {
        let processor = ComprehensiveKpiProcessor::new();
        let processed_rows = vec![
            create_test_processed_row("cell1", 99.0, 50.0, 25.0, false),
            create_test_processed_row("cell2", 95.0, 30.0, 15.0, true),
            create_test_processed_row("cell3", 98.0, 40.0, 20.0, false),
        ];

        let summary = processor.calculate_summary_statistics(&processed_rows);

        assert_eq!(summary.total_cells, 3);
        assert_eq!(summary.critical_cells, 1);
        assert_eq!(summary.healthy_cells, 2);
        assert!((summary.average_availability - 97.33).abs() < 0.1);
        assert!((summary.average_throughput_dl - 40.0).abs() < 0.1);
    }

    fn create_test_processed_row(cell_id: &str, availability: f64, dl_throughput: f64, ul_throughput: f64, critical: bool) -> ProcessedKpiRow {
        ProcessedKpiRow {
            timestamp: "2025-06-27 00:00:00".to_string(),
            cell_id: cell_id.to_string(),
            enodeb_name: "TEST_ENODEB".to_string(),
            cell_name: "TEST_CELL".to_string(),
            frequency_band: "LTE1800".to_string(),
            signal_quality: SignalQualityMetrics {
                sinr_pusch_avg: 15.0,
                sinr_pucch_avg: 14.0,
                ul_rssi_pucch: -110.0,
                ul_rssi_pusch: -105.0,
                ul_rssi_total: -108.0,
                signal_quality_score: 80.0,
            },
            performance_kpis: PerformanceKpis {
                cell_availability_pct: availability,
                volte_traffic_erl: 1.0,
                eric_traff_erab_erl: 10.0,
                rrc_connected_users_avg: 50.0,
                ul_volume_pdcp_gbytes: 1.0,
                dl_volume_pdcp_gbytes: 5.0,
                lte_dcr_volte: 0.5,
                erab_drop_rate_qci_5: 1.0,
                erab_drop_rate_qci_8: 1.5,
                ue_context_abnormal_rel_pct: 2.0,
                cssr_end_user_pct: 98.0,
                eric_erab_init_setup_sr: 95.0,
                rrc_reestab_sr: 90.0,
                ave_4g_lte_dl_user_thrput: dl_throughput,
                ave_4g_lte_ul_user_thrput: ul_throughput,
                performance_score: 85.0,
            },
            mobility_metrics: MobilityMetrics {
                lte_intra_freq_ho_sr: 95.0,
                lte_inter_freq_ho_sr: 93.0,
                inter_freq_ho_attempts: 50,
                intra_freq_ho_attempts: 100,
                eric_ho_osc_intra: 2.0,
                eric_ho_osc_inter: 1.5,
                eric_rwr_total: 3.0,
                eric_rwr_gsm_rate: 0.0,
                eric_rwr_lte_rate: 1.0,
                eric_rwr_wcdma_rate: 0.0,
                eric_srvcc3g_exesr: 0.0,
                eric_srvcc3g_intens: 0.0,
                eric_srvcc3g_prepsr: 0.0,
                eric_srvcc2g_exesr: 0.0,
                eric_srvcc2g_intens: 0.0,
                eric_srvcc2g_prepsr: 0.0,
                mobility_score: 94.0,
            },
            quality_metrics: QualityMetrics {
                mac_dl_bler: 2.0,
                mac_ul_bler: 1.5,
                dl_packet_error_loss_rate: 0.1,
                dl_packet_error_loss_rate_qci_1: 0.0,
                dl_packet_loss_qci_5: 0.0,
                dl_packet_error_loss_rate_qci_8: 0.0,
                ul_packet_loss_rate: 0.05,
                ul_packet_error_loss_qci_1: 0.0,
                ul_packet_error_loss_qci_5: 0.0,
                ul_packet_loss_rate_qci_8: 0.0,
                dl_latency_avg: 15.0,
                dl_latency_avg_qci_1: 12.0,
                dl_latency_avg_qci_5: 18.0,
                dl_latency_avg_qci_8: 20.0,
                ue_pwr_limited: 10.0,
                voip_integrity_cell_rate: 100.0,
                quality_score: 90.0,
            },
            endc_metrics: EndcMetrics {
                sum_pmmeasconfigb1endc: 100,
                sum_pmendcsetupuesucc: 95,
                sum_pmendcsetupueatt: 100,
                sum_pmb1measrependcconfigrestart: 5,
                sum_pmb1measrependcconfig: 90,
                sum_pmendccapableue: 80,
                sum_pmendcsetupfailnrra: 3,
                active_ues_dl: 20,
                active_ues_ul: 15,
                active_user_dl_qci_1: 5,
                active_user_dl_qci_5: 10,
                active_user_dl_qci_8: 5,
                endc_establishment_att: 100,
                endc_establishment_succ: 95,
                endc_nb_received_b1_reports: 80,
                endc_nb_ue_config_b1_reports: 75,
                endc_nr_ra_scg_failures: 2,
                endc_scg_failure_ratio: 2.0,
                endc_setup_sr: 95.0,
                nb_endc_capables_ue_setup: 80,
                endc_mn_mcg_bearer_relocation_att: 10,
                endc_mn_mcg_bearer_relocation_sr: 90.0,
                endc_performance_score: 95.0,
            },
            neural_features: NeuralFeatures {
                afm_features: vec![0.1, 0.2, 0.3],
                dtm_features: vec![0.4, 0.5],
                comprehensive_features: vec![0.1, 0.2, 0.3, 0.4, 0.5],
                feature_importance: HashMap::new(),
                dimensionality_reduction: None,
            },
            anomaly_detection: AnomalyDetection {
                availability_anomaly: false,
                throughput_anomaly: false,
                quality_anomaly: false,
                error_rate_anomaly: false,
                handover_anomaly: false,
                endc_anomaly: false,
                critical_fault_detected: critical,
                anomaly_severity_score: if critical { 0.8 } else { 0.1 },
                anomaly_explanation: if critical { vec!["Test critical fault".to_string()] } else { Vec::new() },
                recommended_actions: if critical { vec!["Test action".to_string()] } else { Vec::new() },
            },
            calculated_kpis: HashMap::new(),
            temporal_analysis: None,
        }
    }
}