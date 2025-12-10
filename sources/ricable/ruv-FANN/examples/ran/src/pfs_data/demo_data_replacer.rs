//! Demo Data Replacer - Replace Mock Data with Real CSV Data
//! 
//! This module provides functions to replace all mock/dummy data in the 
//! enhanced_neural_swarm_demo.rs with real data parsed from fanndata.csv.

use crate::pfs_data::csv_data_parser::{CsvDataParser, ParsedCsvDataset, RealCellDataCollection, RealCellData};
use crate::pfs_data::neural_data_processor::NeuralProcessingResult;
use std::collections::HashMap;
use std::path::Path;
use serde::{Serialize, Deserialize};
use rand::Rng;

/// Replacement data structure containing all real values to replace mock data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoDataReplacements {
    pub network_kpis: NetworkKpiReplacements,
    pub cell_performance: CellPerformanceReplacements,
    pub traffic_patterns: TrafficPatternReplacements,
    pub quality_metrics: QualityMetricReplacements,
    pub handover_metrics: HandoverMetricReplacements,
    pub fault_scenarios: FaultScenarioReplacements,
    pub optimization_targets: OptimizationTargetReplacements,
    pub neural_features: NeuralFeatureReplacements,
}

/// Network-level KPI replacements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkKpiReplacements {
    pub overall_availability: f64,
    pub network_throughput: f64,
    pub total_connected_users: u32,
    pub average_sinr: f64,
    pub network_efficiency: f64,
    pub critical_alarms: u32,
    pub performance_index: f64,
}

/// Cell-level performance replacements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellPerformanceReplacements {
    pub sample_cells: Vec<CellPerformanceData>,
    pub top_performers: Vec<CellPerformanceData>,
    pub underperformers: Vec<CellPerformanceData>,
    pub baseline_metrics: CellPerformanceData,
}

/// Individual cell performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellPerformanceData {
    pub cell_id: String,
    pub enodeb_name: String,
    pub availability: f64,
    pub dl_throughput: f64,
    pub ul_throughput: f64,
    pub connected_users: u32,
    pub sinr: f64,
    pub error_rate: f64,
    pub handover_success_rate: f64,
    pub optimization_priority: f64,
}

/// Traffic pattern replacements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficPatternReplacements {
    pub peak_hour_traffic: Vec<f64>,
    pub daily_patterns: Vec<f64>,
    pub user_distribution: Vec<u32>,
    pub qci_distributions: QciTrafficData,
    pub volte_traffic_patterns: Vec<f64>,
}

/// QCI-specific traffic data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QciTrafficData {
    pub qci_1_users: Vec<u32>,
    pub qci_5_users: Vec<u32>,
    pub qci_8_users: Vec<u32>,
    pub qci_1_latency: Vec<f64>,
    pub qci_5_latency: Vec<f64>,
    pub qci_8_latency: Vec<f64>,
}

/// Quality metric replacements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetricReplacements {
    pub sinr_distributions: SinrData,
    pub rssi_distributions: RssiData,
    pub bler_statistics: BlerData,
    pub packet_loss_rates: PacketLossData,
}

/// SINR distribution data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinrData {
    pub pusch_values: Vec<f64>,
    pub pucch_values: Vec<f64>,
    pub average_sinr: f64,
    pub min_sinr: f64,
    pub max_sinr: f64,
}

/// RSSI distribution data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RssiData {
    pub pucch_values: Vec<f64>,
    pub pusch_values: Vec<f64>,
    pub total_values: Vec<f64>,
    pub average_rssi: f64,
}

/// BLER statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlerData {
    pub dl_bler_values: Vec<f64>,
    pub ul_bler_values: Vec<f64>,
    pub average_dl_bler: f64,
    pub average_ul_bler: f64,
}

/// Packet loss data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketLossData {
    pub dl_packet_loss: Vec<f64>,
    pub ul_packet_loss: Vec<f64>,
    pub qci_specific_loss: HashMap<u8, Vec<f64>>,
}

/// Handover metric replacements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoverMetricReplacements {
    pub intra_freq_success_rates: Vec<f64>,
    pub inter_freq_success_rates: Vec<f64>,
    pub handover_attempts: HandoverAttemptData,
    pub oscillation_rates: OscillationData,
    pub average_ho_success_rate: f64,
}

/// Handover attempt data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoverAttemptData {
    pub intra_freq_attempts: Vec<u32>,
    pub inter_freq_attempts: Vec<u32>,
    pub total_attempts: u32,
}

/// Oscillation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationData {
    pub intra_freq_oscillations: Vec<f64>,
    pub inter_freq_oscillations: Vec<f64>,
    pub rwr_rates: Vec<f64>,
}

/// Fault scenario replacements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultScenarioReplacements {
    pub critical_cells: Vec<String>,
    pub fault_types: HashMap<String, u32>,
    pub anomaly_scores: Vec<f64>,
    pub degraded_performance_cells: Vec<CellPerformanceData>,
    pub fault_correlation_data: FaultCorrelationData,
}

/// Fault correlation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultCorrelationData {
    pub availability_issues: Vec<String>,
    pub throughput_issues: Vec<String>,
    pub quality_issues: Vec<String>,
    pub handover_issues: Vec<String>,
    pub root_causes: HashMap<String, Vec<String>>,
}

/// Optimization target replacements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTargetReplacements {
    pub throughput_targets: Vec<f64>,
    pub availability_targets: Vec<f64>,
    pub latency_targets: Vec<f64>,
    pub efficiency_targets: Vec<f64>,
    pub optimization_priorities: Vec<OptimizationPriority>,
}

/// Optimization priority data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPriority {
    pub cell_id: String,
    pub priority_score: f64,
    pub optimization_type: String,
    pub expected_improvement: f64,
    pub investment_required: f64,
}

/// Neural feature replacements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralFeatureReplacements {
    pub afm_features: Vec<Vec<f32>>,
    pub dtm_features: Vec<Vec<f32>>,
    pub comprehensive_features: Vec<Vec<f32>>,
    pub target_labels: Vec<Vec<f32>>,
    pub neural_scores: Vec<NeuralScoreData>,
}

/// Neural score data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralScoreData {
    pub cell_id: String,
    pub fault_probability: f32,
    pub mobility_score: f32,
    pub efficiency_score: f32,
    pub quality_score: f32,
    pub anomaly_severity: f32,
}

/// Main demo data replacer
pub struct DemoDataReplacer {
    csv_parser: CsvDataParser,
    replacements: Option<DemoDataReplacements>,
}

impl DemoDataReplacer {
    /// Create new demo data replacer
    pub fn new() -> Self {
        Self {
            csv_parser: CsvDataParser::new(),
            replacements: None,
        }
    }

    /// Generate all replacement data from CSV file
    pub fn generate_replacements<P: AsRef<Path>>(&mut self, csv_file_path: P) -> Result<&DemoDataReplacements, Box<dyn std::error::Error>> {
        println!("🔄 Generating demo data replacements from CSV...");
        
        // Parse CSV file
        let dataset = self.csv_parser.parse_csv_file(csv_file_path)?;
        
        // Get real cell data collection
        let real_data = self.csv_parser.get_real_cell_data(&dataset);
        
        // Generate all replacement categories
        let replacements = DemoDataReplacements {
            network_kpis: self.generate_network_kpi_replacements(&real_data),
            cell_performance: self.generate_cell_performance_replacements(&real_data),
            traffic_patterns: self.generate_traffic_pattern_replacements(&dataset),
            quality_metrics: self.generate_quality_metric_replacements(&dataset),
            handover_metrics: self.generate_handover_metric_replacements(&dataset),
            fault_scenarios: self.generate_fault_scenario_replacements(&dataset),
            optimization_targets: self.generate_optimization_target_replacements(&real_data),
            neural_features: self.generate_neural_feature_replacements(&dataset),
        };
        
        self.replacements = Some(replacements);
        
        println!("✅ Demo data replacements generated successfully!");
        self.print_replacement_summary();
        
        Ok(self.replacements.as_ref().unwrap())
    }

    /// Generate network-level KPI replacements
    fn generate_network_kpi_replacements(&self, real_data: &RealCellDataCollection) -> NetworkKpiReplacements {
        let stats = &real_data.statistics;
        
        NetworkKpiReplacements {
            overall_availability: stats.avg_availability,
            network_throughput: stats.avg_throughput,
            total_connected_users: stats.avg_users as u32 * stats.total_cells as u32,
            average_sinr: stats.avg_sinr,
            network_efficiency: stats.healthy_cells as f64 / stats.total_cells as f64,
            critical_alarms: stats.critical_issues as u32,
            performance_index: (stats.avg_availability + stats.avg_sinr * 10.0 + 
                              (stats.healthy_cells as f64 / stats.total_cells as f64) * 100.0) / 3.0,
        }
    }

    /// Generate cell performance replacements
    fn generate_cell_performance_replacements(&self, real_data: &RealCellDataCollection) -> CellPerformanceReplacements {
        // Get sample cells
        let sample_cells: Vec<CellPerformanceData> = real_data.get_random_sample(20)
            .iter()
            .map(|cell| self.convert_to_performance_data(cell))
            .collect();

        // Get top performers (high availability, low anomaly score)
        let top_performers: Vec<CellPerformanceData> = real_data.get_cells_by_criteria(98.0, 50.0, 0.3)
            .iter()
            .take(10)
            .map(|cell| self.convert_to_performance_data(cell))
            .collect();

        // Get underperformers
        let underperformers: Vec<CellPerformanceData> = real_data.get_problematic_cells()
            .iter()
            .take(10)
            .map(|cell| self.convert_to_performance_data(cell))
            .collect();

        // Calculate baseline metrics
        let baseline_metrics = CellPerformanceData {
            cell_id: "baseline".to_string(),
            enodeb_name: "baseline".to_string(),
            availability: real_data.statistics.avg_availability,
            dl_throughput: real_data.statistics.avg_throughput / 2.0,
            ul_throughput: real_data.statistics.avg_throughput / 4.0,
            connected_users: real_data.statistics.avg_users as u32,
            sinr: real_data.statistics.avg_sinr,
            error_rate: 2.0, // Typical baseline
            handover_success_rate: 95.0, // Typical baseline
            optimization_priority: 0.5,
        };

        CellPerformanceReplacements {
            sample_cells,
            top_performers,
            underperformers,
            baseline_metrics,
        }
    }

    /// Convert real cell data to performance data format
    fn convert_to_performance_data(&self, cell: &RealCellData) -> CellPerformanceData {
        CellPerformanceData {
            cell_id: cell.cell_id.clone(),
            enodeb_name: cell.enodeb_name.clone(),
            availability: cell.availability,
            dl_throughput: cell.throughput_dl,
            ul_throughput: cell.throughput_ul,
            connected_users: cell.connected_users,
            sinr: cell.sinr_avg,
            error_rate: cell.error_rate,
            handover_success_rate: cell.handover_success_rate,
            optimization_priority: cell.anomaly_score,
        }
    }

    /// Generate traffic pattern replacements
    fn generate_traffic_pattern_replacements(&self, dataset: &ParsedCsvDataset) -> TrafficPatternReplacements {
        let mut peak_hour_traffic = Vec::new();
        let mut daily_patterns = Vec::new();
        let mut user_distribution = Vec::new();
        let mut volte_traffic_patterns = Vec::new();
        let mut qci_1_users = Vec::new();
        let mut qci_5_users = Vec::new();
        let mut qci_8_users = Vec::new();
        let mut qci_1_latency = Vec::new();
        let mut qci_5_latency = Vec::new();
        let mut qci_8_latency = Vec::new();

        for row in &dataset.rows {
            peak_hour_traffic.push(row.metrics.eric_traff_erab_erl);
            daily_patterns.push(row.metrics.volte_traffic_erl * 24.0); // Simulate daily pattern
            user_distribution.push(row.metrics.rrc_connected_users_avg as u32);
            volte_traffic_patterns.push(row.metrics.volte_traffic_erl);
            
            qci_1_users.push(row.traffic_data.active_user_dl_qci_1);
            qci_5_users.push(row.traffic_data.active_user_dl_qci_5);
            qci_8_users.push(row.traffic_data.active_user_dl_qci_8);
            
            qci_1_latency.push(row.traffic_data.dl_latency_avg_qci_1);
            qci_5_latency.push(row.traffic_data.dl_latency_avg_qci_5);
            qci_8_latency.push(row.traffic_data.dl_latency_avg_qci_8);
        }

        TrafficPatternReplacements {
            peak_hour_traffic,
            daily_patterns,
            user_distribution,
            qci_distributions: QciTrafficData {
                qci_1_users,
                qci_5_users,
                qci_8_users,
                qci_1_latency,
                qci_5_latency,
                qci_8_latency,
            },
            volte_traffic_patterns,
        }
    }

    /// Generate quality metric replacements
    fn generate_quality_metric_replacements(&self, dataset: &ParsedCsvDataset) -> QualityMetricReplacements {
        let mut pusch_values = Vec::new();
        let mut pucch_values = Vec::new();
        let mut pucch_rssi = Vec::new();
        let mut pusch_rssi = Vec::new();
        let mut total_rssi = Vec::new();
        let mut dl_bler = Vec::new();
        let mut ul_bler = Vec::new();
        let mut dl_packet_loss = Vec::new();
        let mut ul_packet_loss = Vec::new();

        for row in &dataset.rows {
            pusch_values.push(row.quality_indicators.sinr_pusch_avg);
            pucch_values.push(row.quality_indicators.sinr_pucch_avg);
            pucch_rssi.push(row.quality_indicators.ul_rssi_pucch);
            pusch_rssi.push(row.quality_indicators.ul_rssi_pusch);
            total_rssi.push(row.quality_indicators.ul_rssi_total);
            dl_bler.push(row.quality_indicators.mac_dl_bler);
            ul_bler.push(row.quality_indicators.mac_ul_bler);
            dl_packet_loss.push(row.traffic_data.dl_packet_error_loss_rate);
            ul_packet_loss.push(row.traffic_data.ul_packet_loss_rate);
        }

        let avg_sinr = (pusch_values.iter().sum::<f64>() + pucch_values.iter().sum::<f64>()) / 
                      (pusch_values.len() + pucch_values.len()) as f64;
        let avg_rssi = total_rssi.iter().sum::<f64>() / total_rssi.len() as f64;
        let avg_dl_bler = dl_bler.iter().sum::<f64>() / dl_bler.len() as f64;
        let avg_ul_bler = ul_bler.iter().sum::<f64>() / ul_bler.len() as f64;

        QualityMetricReplacements {
            sinr_distributions: SinrData {
                pusch_values: pusch_values.clone(),
                pucch_values: pucch_values.clone(),
                average_sinr: avg_sinr,
                min_sinr: pusch_values.iter().chain(pucch_values.iter()).fold(f64::INFINITY, |a, &b| a.min(b)),
                max_sinr: pusch_values.iter().chain(pucch_values.iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            },
            rssi_distributions: RssiData {
                pucch_values: pucch_rssi,
                pusch_values: pusch_rssi,
                total_values: total_rssi,
                average_rssi: avg_rssi,
            },
            bler_statistics: BlerData {
                dl_bler_values: dl_bler,
                ul_bler_values: ul_bler,
                average_dl_bler: avg_dl_bler,
                average_ul_bler: avg_ul_bler,
            },
            packet_loss_rates: PacketLossData {
                dl_packet_loss,
                ul_packet_loss,
                qci_specific_loss: HashMap::new(), // Could be extended with QCI-specific data
            },
        }
    }

    /// Generate handover metric replacements
    fn generate_handover_metric_replacements(&self, dataset: &ParsedCsvDataset) -> HandoverMetricReplacements {
        let mut intra_freq_success_rates = Vec::new();
        let mut inter_freq_success_rates = Vec::new();
        let mut intra_freq_attempts = Vec::new();
        let mut inter_freq_attempts = Vec::new();
        let mut intra_oscillations = Vec::new();
        let mut inter_oscillations = Vec::new();
        let mut rwr_rates = Vec::new();

        for row in &dataset.rows {
            intra_freq_success_rates.push(row.handover_metrics.lte_intra_freq_ho_sr);
            inter_freq_success_rates.push(row.handover_metrics.lte_inter_freq_ho_sr);
            intra_freq_attempts.push(row.handover_metrics.intra_freq_ho_attempts);
            inter_freq_attempts.push(row.handover_metrics.inter_freq_ho_attempts);
            intra_oscillations.push(row.handover_metrics.eric_ho_osc_intra);
            inter_oscillations.push(row.handover_metrics.eric_ho_osc_inter);
            rwr_rates.push(row.handover_metrics.eric_rwr_lte_rate);
        }

        let avg_ho_success = (intra_freq_success_rates.iter().sum::<f64>() + 
                             inter_freq_success_rates.iter().sum::<f64>()) / 
                            (intra_freq_success_rates.len() + inter_freq_success_rates.len()) as f64;
        
        let total_attempts = intra_freq_attempts.iter().sum::<u32>() + inter_freq_attempts.iter().sum::<u32>();

        HandoverMetricReplacements {
            intra_freq_success_rates,
            inter_freq_success_rates,
            handover_attempts: HandoverAttemptData {
                intra_freq_attempts,
                inter_freq_attempts,
                total_attempts,
            },
            oscillation_rates: OscillationData {
                intra_freq_oscillations: intra_oscillations,
                inter_freq_oscillations: inter_oscillations,
                rwr_rates,
            },
            average_ho_success_rate: avg_ho_success,
        }
    }

    /// Generate fault scenario replacements
    fn generate_fault_scenario_replacements(&self, dataset: &ParsedCsvDataset) -> FaultScenarioReplacements {
        let mut critical_cells = Vec::new();
        let mut fault_types = HashMap::new();
        let mut anomaly_scores = Vec::new();
        let mut degraded_cells = Vec::new();
        
        // Analyze anomalies and faults
        for row in &dataset.rows {
            if row.anomaly_flags.critical_fault_detected {
                critical_cells.push(format!("{}_{}", 
                    row.cell_identifier.enodeb_code, 
                    row.cell_identifier.cell_code));
                
                degraded_cells.push(CellPerformanceData {
                    cell_id: format!("{}_{}", row.cell_identifier.enodeb_code, row.cell_identifier.cell_code),
                    enodeb_name: row.cell_identifier.enodeb_name.clone(),
                    availability: row.metrics.cell_availability_pct,
                    dl_throughput: row.performance_kpis.ave_4g_lte_dl_user_thrput,
                    ul_throughput: row.performance_kpis.ave_4g_lte_ul_user_thrput,
                    connected_users: row.metrics.rrc_connected_users_avg as u32,
                    sinr: (row.quality_indicators.sinr_pusch_avg + row.quality_indicators.sinr_pucch_avg) / 2.0,
                    error_rate: (row.quality_indicators.mac_dl_bler + row.quality_indicators.mac_ul_bler) / 2.0,
                    handover_success_rate: (row.handover_metrics.lte_intra_freq_ho_sr + row.handover_metrics.lte_inter_freq_ho_sr) / 2.0,
                    optimization_priority: row.anomaly_flags.anomaly_severity_score as f64,
                });
            }
            
            // Categorize fault types
            if row.anomaly_flags.availability_anomaly {
                *fault_types.entry("availability_fault".to_string()).or_insert(0) += 1;
            }
            if row.anomaly_flags.throughput_anomaly {
                *fault_types.entry("throughput_degradation".to_string()).or_insert(0) += 1;
            }
            if row.anomaly_flags.quality_anomaly {
                *fault_types.entry("signal_quality_issue".to_string()).or_insert(0) += 1;
            }
            if row.anomaly_flags.error_rate_anomaly {
                *fault_types.entry("high_error_rate".to_string()).or_insert(0) += 1;
            }
            if row.anomaly_flags.handover_anomaly {
                *fault_types.entry("handover_failure".to_string()).or_insert(0) += 1;
            }
            
            anomaly_scores.push(row.anomaly_flags.anomaly_severity_score as f64);
        }

        // Create fault correlation data
        let fault_correlation = FaultCorrelationData {
            availability_issues: vec![
                "Power supply failure".to_string(),
                "Hardware malfunction".to_string(),
                "Software crash".to_string(),
            ],
            throughput_issues: vec![
                "Congestion".to_string(),
                "Poor radio conditions".to_string(),
                "Interference".to_string(),
            ],
            quality_issues: vec![
                "Antenna misalignment".to_string(),
                "Interference from neighboring cells".to_string(),
                "Environmental factors".to_string(),
            ],
            handover_issues: vec![
                "Neighbor list misconfiguration".to_string(),
                "Threshold misalignment".to_string(),
                "Load imbalance".to_string(),
            ],
            root_causes: HashMap::from([
                ("availability_fault".to_string(), vec!["Hardware failure".to_string(), "Power issues".to_string()]),
                ("throughput_degradation".to_string(), vec!["Congestion".to_string(), "Poor coverage".to_string()]),
                ("signal_quality_issue".to_string(), vec!["Interference".to_string(), "Antenna issues".to_string()]),
            ]),
        };

        FaultScenarioReplacements {
            critical_cells,
            fault_types,
            anomaly_scores,
            degraded_performance_cells: degraded_cells,
            fault_correlation_data: fault_correlation,
        }
    }

    /// Generate optimization target replacements
    fn generate_optimization_target_replacements(&self, real_data: &RealCellDataCollection) -> OptimizationTargetReplacements {
        let mut throughput_targets = Vec::new();
        let mut availability_targets = Vec::new();
        let mut latency_targets = Vec::new();
        let mut efficiency_targets = Vec::new();
        let mut optimization_priorities = Vec::new();

        for cell in &real_data.cells {
            // Set targets based on current performance and industry benchmarks
            throughput_targets.push((cell.throughput_dl + cell.throughput_ul) * 1.2); // 20% improvement target
            availability_targets.push((cell.availability + 2.0).min(100.0)); // Improve availability
            latency_targets.push(15.0); // Target latency in ms
            efficiency_targets.push(cell.traffic_load * 0.8); // Improve efficiency by 20%

            // Calculate optimization priority
            let priority_score = cell.anomaly_score + 
                               (if cell.availability < 95.0 { 0.3 } else { 0.0 }) +
                               (if cell.throughput_dl < 10.0 { 0.2 } else { 0.0 });

            optimization_priorities.push(OptimizationPriority {
                cell_id: cell.cell_id.clone(),
                priority_score,
                optimization_type: if cell.availability < 95.0 {
                    "availability_improvement".to_string()
                } else if cell.throughput_dl < 10.0 {
                    "throughput_enhancement".to_string()
                } else {
                    "efficiency_optimization".to_string()
                },
                expected_improvement: priority_score * 0.5, // Expected improvement percentage
                investment_required: priority_score * 1000.0, // Investment in arbitrary units
            });
        }

        OptimizationTargetReplacements {
            throughput_targets,
            availability_targets,
            latency_targets,
            efficiency_targets,
            optimization_priorities,
        }
    }

    /// Generate neural feature replacements
    fn generate_neural_feature_replacements(&self, dataset: &ParsedCsvDataset) -> NeuralFeatureReplacements {
        let mut neural_scores = Vec::new();

        for (row, neural_result) in dataset.rows.iter().zip(dataset.neural_results.iter()) {
            neural_scores.push(NeuralScoreData {
                cell_id: format!("{}_{}", row.cell_identifier.enodeb_code, row.cell_identifier.cell_code),
                fault_probability: neural_result.neural_scores.afm_fault_probability,
                mobility_score: neural_result.neural_scores.dtm_mobility_score,
                efficiency_score: neural_result.neural_scores.energy_efficiency_score,
                quality_score: neural_result.neural_scores.service_quality_score,
                anomaly_severity: neural_result.neural_scores.anomaly_severity_score,
            });
        }

        NeuralFeatureReplacements {
            afm_features: dataset.feature_vectors.afm_features.clone(),
            dtm_features: dataset.feature_vectors.dtm_features.clone(),
            comprehensive_features: dataset.feature_vectors.comprehensive_features.clone(),
            target_labels: dataset.feature_vectors.target_labels.clone(),
            neural_scores,
        }
    }

    /// Print summary of generated replacements
    fn print_replacement_summary(&self) {
        if let Some(replacements) = &self.replacements {
            println!("\n📋 Demo Data Replacement Summary:");
            println!("   🌐 Network KPIs: Overall availability {:.2}%, {} total users", 
                    replacements.network_kpis.overall_availability,
                    replacements.network_kpis.total_connected_users);
            
            println!("   📱 Cell Performance: {} sample cells, {} top performers, {} underperformers", 
                    replacements.cell_performance.sample_cells.len(),
                    replacements.cell_performance.top_performers.len(),
                    replacements.cell_performance.underperformers.len());
            
            println!("   📊 Traffic Patterns: {} peak hour samples, {} QCI distributions", 
                    replacements.traffic_patterns.peak_hour_traffic.len(),
                    replacements.traffic_patterns.qci_distributions.qci_1_users.len());
            
            println!("   📡 Quality Metrics: {} SINR samples, avg {:.2} dB", 
                    replacements.quality_metrics.sinr_distributions.pusch_values.len(),
                    replacements.quality_metrics.sinr_distributions.average_sinr);
            
            println!("   🔄 Handover Metrics: Avg success rate {:.2}%", 
                    replacements.handover_metrics.average_ho_success_rate);
            
            println!("   ⚠️ Fault Scenarios: {} critical cells, {} fault types", 
                    replacements.fault_scenarios.critical_cells.len(),
                    replacements.fault_scenarios.fault_types.len());
            
            println!("   🎯 Optimization: {} targets, {} priorities", 
                    replacements.optimization_targets.throughput_targets.len(),
                    replacements.optimization_targets.optimization_priorities.len());
            
            println!("   🧠 Neural Features: {} AFM vectors, {} DTM vectors, {} neural scores", 
                    replacements.neural_features.afm_features.len(),
                    replacements.neural_features.dtm_features.len(),
                    replacements.neural_features.neural_scores.len());
        }
    }

    /// Get generated replacements
    pub fn get_replacements(&self) -> Option<&DemoDataReplacements> {
        self.replacements.as_ref()
    }

    /// Export replacements to JSON file for use in demo
    pub fn export_to_json<P: AsRef<Path>>(&self, output_path: P) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(replacements) = &self.replacements {
            let json_content = serde_json::to_string_pretty(replacements)?;
            std::fs::write(output_path, json_content)?;
            println!("✅ Replacements exported to JSON file");
        }
        Ok(())
    }

    /// Generate specific replacement values for common mock patterns
    pub fn get_mock_replacements(&self) -> MockValueReplacements {
        if let Some(replacements) = &self.replacements {
            MockValueReplacements {
                // Replace common mock values like 0.5, 1.0, 2.0, etc.
                availability_values: replacements.cell_performance.sample_cells.iter()
                    .map(|c| c.availability / 100.0)
                    .collect(),
                
                throughput_values: replacements.cell_performance.sample_cells.iter()
                    .map(|c| c.dl_throughput)
                    .collect(),
                
                user_count_values: replacements.cell_performance.sample_cells.iter()
                    .map(|c| c.connected_users as f64)
                    .collect(),
                
                sinr_values: replacements.quality_metrics.sinr_distributions.pusch_values.clone(),
                
                error_rate_values: replacements.quality_metrics.bler_statistics.dl_bler_values.clone(),
                
                handover_success_values: replacements.handover_metrics.intra_freq_success_rates.iter()
                    .map(|v| v / 100.0)
                    .collect(),
                
                traffic_load_values: replacements.traffic_patterns.peak_hour_traffic.clone(),
                
                latency_values: replacements.traffic_patterns.qci_distributions.qci_1_latency.clone(),
                
                anomaly_scores: replacements.fault_scenarios.anomaly_scores.clone(),
                
                optimization_priorities: replacements.optimization_targets.optimization_priorities.iter()
                    .map(|p| p.priority_score)
                    .collect(),
            }
        } else {
            MockValueReplacements::default()
        }
    }
}

/// Mock value replacements for common patterns
#[derive(Debug, Clone, Default)]
pub struct MockValueReplacements {
    pub availability_values: Vec<f64>,
    pub throughput_values: Vec<f64>,
    pub user_count_values: Vec<f64>,
    pub sinr_values: Vec<f64>,
    pub error_rate_values: Vec<f64>,
    pub handover_success_values: Vec<f64>,
    pub traffic_load_values: Vec<f64>,
    pub latency_values: Vec<f64>,
    pub anomaly_scores: Vec<f64>,
    pub optimization_priorities: Vec<f64>,
}

impl MockValueReplacements {
    /// Get a random value from availability values to replace mock 0.95, 0.99, etc.
    pub fn get_random_availability(&self) -> f64 {
        if self.availability_values.is_empty() {
            0.95 // Fallback
        } else {
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..self.availability_values.len());
            self.availability_values[index]
        }
    }

    /// Get a random throughput value to replace mock 50.0, 100.0, etc.
    pub fn get_random_throughput(&self) -> f64 {
        if self.throughput_values.is_empty() {
            50.0 // Fallback
        } else {
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..self.throughput_values.len());
            self.throughput_values[index]
        }
    }

    /// Get a random user count to replace mock 100, 200, etc.
    pub fn get_random_user_count(&self) -> u32 {
        if self.user_count_values.is_empty() {
            100 // Fallback
        } else {
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..self.user_count_values.len());
            self.user_count_values[index] as u32
        }
    }

    /// Get a vector of real values to replace mock arrays like vec![0.5, 1.0, 2.0]
    pub fn get_real_value_vector(&self, vector_type: &str, size: usize) -> Vec<f64> {
        let source_values = match vector_type {
            "availability" => &self.availability_values,
            "throughput" => &self.throughput_values,
            "sinr" => &self.sinr_values,
            "error_rate" => &self.error_rate_values,
            "handover" => &self.handover_success_values,
            "traffic" => &self.traffic_load_values,
            "latency" => &self.latency_values,
            "anomaly" => &self.anomaly_scores,
            _ => &self.availability_values,
        };

        if source_values.is_empty() {
            vec![0.5; size] // Fallback
        } else {
            let mut rng = rand::thread_rng();
            (0..size).map(|_| {
                let index = rng.gen_range(0..source_values.len());
                source_values[index]
            }).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_demo_data_replacer_creation() {
        let replacer = DemoDataReplacer::new();
        assert!(replacer.replacements.is_none());
    }

    #[test]
    fn test_mock_value_replacements() {
        let mock_replacements = MockValueReplacements {
            availability_values: vec![0.95, 0.98, 0.99],
            throughput_values: vec![25.0, 50.0, 75.0],
            user_count_values: vec![50.0, 100.0, 150.0],
            ..Default::default()
        };

        let availability = mock_replacements.get_random_availability();
        assert!(availability >= 0.9 && availability <= 1.0);

        let throughput = mock_replacements.get_random_throughput();
        assert!(throughput >= 20.0 && throughput <= 80.0);

        let user_count = mock_replacements.get_random_user_count();
        assert!(user_count >= 40 && user_count <= 160);
    }

    #[test]
    fn test_real_value_vector_generation() {
        let mock_replacements = MockValueReplacements {
            sinr_values: vec![5.0, 10.0, 15.0, 20.0],
            ..Default::default()
        };

        let sinr_vector = mock_replacements.get_real_value_vector("sinr", 5);
        assert_eq!(sinr_vector.len(), 5);
        
        for value in sinr_vector {
            assert!(value >= 5.0 && value <= 20.0);
        }
    }
}