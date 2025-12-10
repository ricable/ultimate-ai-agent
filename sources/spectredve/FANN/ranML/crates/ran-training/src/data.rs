//! Data loading and management for telecom neural network training

use crate::error::{TrainingError, TrainingResult};
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Telecom data record from CSV
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelecomRecord {
    /// Timestamp
    pub timestamp: String,
    /// eNodeB identifier
    pub enodeb_code: String,
    /// eNodeB name
    pub enodeb_name: String,
    /// Cell identifier
    pub cell_code: String,
    /// Cell name
    pub cell_name: String,
    /// LTE band
    pub lte_band: String,
    /// Number of bands
    pub num_bands: u8,
    /// Cell availability percentage
    pub cell_availability: f32,
    /// VoLTE traffic in Erlangs
    pub volte_traffic: f32,
    /// ERAB traffic in Erlangs
    pub erab_traffic: f32,
    /// Average connected users
    pub connected_users_avg: f32,
    /// Uplink volume in GB
    pub ul_volume_gb: f32,
    /// Downlink volume in GB
    pub dl_volume_gb: f32,
    /// 4G LTE drop call rate VoLTE
    pub dcr_volte: f32,
    /// ERAB drop rate QCI 5
    pub erab_drop_qci5: f32,
    /// ERAB drop rate QCI 8
    pub erab_drop_qci8: f32,
    /// UE context attempts
    pub ue_context_att: u32,
    /// UE context abnormal release percentage
    pub ue_context_abnorm_rel_pct: f32,
    /// Average DL user throughput
    pub avg_dl_user_throughput: f32,
    /// Average UL user throughput
    pub avg_ul_user_throughput: f32,
    /// SINR PUSCH average
    pub sinr_pusch_avg: f32,
    /// SINR PUCCH average
    pub sinr_pucch_avg: f32,
    /// UL RSSI total
    pub ul_rssi_total: f32,
    /// MAC DL BLER
    pub mac_dl_bler: f32,
    /// MAC UL BLER
    pub mac_ul_bler: f32,
    /// DL packet error loss rate
    pub dl_packet_error_loss_rate: f32,
    /// UL packet loss rate
    pub ul_packet_loss_rate: f32,
    /// DL latency average
    pub dl_latency_avg: f32,
    /// Handover success rate
    pub handover_success_rate: f32,
}

impl TelecomRecord {
    /// Extract features for neural network input
    pub fn to_features(&self) -> Vec<f32> {
        vec![
            self.cell_availability,
            self.volte_traffic,
            self.erab_traffic,
            self.connected_users_avg,
            self.ul_volume_gb,
            self.dl_volume_gb,
            self.dcr_volte,
            self.erab_drop_qci5,
            self.erab_drop_qci8,
            self.ue_context_abnorm_rel_pct,
            self.avg_dl_user_throughput / 1000.0, // Normalize to Kbps
            self.avg_ul_user_throughput / 1000.0, // Normalize to Kbps
            self.sinr_pusch_avg,
            self.sinr_pucch_avg,
            self.ul_rssi_total / 100.0, // Normalize RSSI
            self.mac_dl_bler,
            self.mac_ul_bler,
            self.dl_packet_error_loss_rate,
            self.ul_packet_loss_rate,
            self.dl_latency_avg,
            self.handover_success_rate,
        ]
    }

    /// Get target value (can be customized based on use case)
    pub fn get_target(&self, target_type: &TargetType) -> f32 {
        match target_type {
            TargetType::CellAvailability => self.cell_availability,
            TargetType::VoLTETraffic => self.volte_traffic,
            TargetType::UserThroughput => (self.avg_dl_user_throughput + self.avg_ul_user_throughput) / 2.0,
            TargetType::QualityScore => {
                // Composite quality score based on multiple metrics
                let availability_score = self.cell_availability / 100.0;
                let error_score = 1.0 - (self.dl_packet_error_loss_rate + self.ul_packet_loss_rate) / 2.0;
                let latency_score = 1.0 - (self.dl_latency_avg / 100.0).min(1.0);
                let handover_score = self.handover_success_rate / 100.0;
                
                (availability_score + error_score + latency_score + handover_score) / 4.0
            }
        }
    }
}

/// Target types for training
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TargetType {
    /// Predict cell availability percentage
    CellAvailability,
    /// Predict VoLTE traffic volume
    VoLTETraffic,
    /// Predict user throughput
    UserThroughput,
    /// Predict composite quality score
    QualityScore,
}

/// Dataset container for training and validation
#[derive(Debug, Clone)]
pub struct TelecomDataset {
    /// Raw records
    pub records: Vec<TelecomRecord>,
    /// Feature matrix (samples x features)
    pub features: Vec<Vec<f32>>,
    /// Target values
    pub targets: Vec<f32>,
    /// Feature statistics for normalization
    pub feature_stats: FeatureStats,
    /// Target type being predicted
    pub target_type: TargetType,
}

/// Feature statistics for normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    /// Mean values for each feature
    pub means: Vec<f32>,
    /// Standard deviations for each feature
    pub stds: Vec<f32>,
    /// Minimum values for each feature
    pub mins: Vec<f32>,
    /// Maximum values for each feature
    pub maxs: Vec<f32>,
}

impl FeatureStats {
    /// Create statistics from feature matrix
    pub fn from_features(features: &[Vec<f32>]) -> Self {
        if features.is_empty() {
            return Self {
                means: Vec::new(),
                stds: Vec::new(),
                mins: Vec::new(),
                maxs: Vec::new(),
            };
        }

        let num_features = features[0].len();
        let num_samples = features.len() as f32;
        
        let mut means = vec![0.0; num_features];
        let mut mins = vec![f32::INFINITY; num_features];
        let mut maxs = vec![f32::NEG_INFINITY; num_features];

        // Calculate means, mins, maxs
        for sample in features {
            for (i, &value) in sample.iter().enumerate() {
                if !value.is_finite() {
                    continue; // Skip NaN/infinite values
                }
                means[i] += value;
                mins[i] = mins[i].min(value);
                maxs[i] = maxs[i].max(value);
            }
        }

        for mean in &mut means {
            *mean /= num_samples;
        }

        // Calculate standard deviations
        let mut stds = vec![0.0; num_features];
        for sample in features {
            for (i, &value) in sample.iter().enumerate() {
                if value.is_finite() {
                    let diff = value - means[i];
                    stds[i] += diff * diff;
                }
            }
        }

        for std in &mut stds {
            *std = (*std / num_samples).sqrt();
            if *std == 0.0 {
                *std = 1.0; // Avoid division by zero
            }
        }

        Self { means, stds, mins, maxs }
    }

    /// Normalize features using z-score normalization
    pub fn normalize(&self, features: &mut [f32]) {
        for (i, feature) in features.iter_mut().enumerate() {
            if i < self.means.len() && feature.is_finite() {
                *feature = (*feature - self.means[i]) / self.stds[i];
            }
        }
    }

    /// Denormalize features back to original scale
    pub fn denormalize(&self, features: &mut [f32]) {
        for (i, feature) in features.iter_mut().enumerate() {
            if i < self.means.len() {
                *feature = *feature * self.stds[i] + self.means[i];
            }
        }
    }
}

impl TelecomDataset {
    /// Load dataset from CSV file
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        target_type: TargetType,
        max_records: Option<usize>,
    ) -> TrainingResult<Self> {
        let file = File::open(path)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b';')
            .from_reader(file);

        let mut records = Vec::new();
        let mut skipped_count = 0;

        for (i, result) in reader.deserialize().enumerate() {
            if let Some(max) = max_records {
                if records.len() >= max {
                    break;
                }
            }

            match result {
                Ok(record) => {
                    let telecom_record: TelecomRecord = Self::parse_csv_record(record)?;
                    
                    // Basic validation - skip records with invalid data
                    if Self::is_valid_record(&telecom_record) {
                        records.push(telecom_record);
                    } else {
                        skipped_count += 1;
                    }
                }
                Err(e) => {
                    log::warn!("Skipping invalid record at line {}: {}", i + 2, e);
                    skipped_count += 1;
                }
            }
        }

        if records.is_empty() {
            return Err(TrainingError::DataError("No valid records found in CSV".into()));
        }

        log::info!("Loaded {} records, skipped {} invalid records", records.len(), skipped_count);

        Self::from_records(records, target_type)
    }

    /// Create dataset from existing records
    pub fn from_records(records: Vec<TelecomRecord>, target_type: TargetType) -> TrainingResult<Self> {
        let features: Vec<Vec<f32>> = records.iter().map(|r| r.to_features()).collect();
        let targets: Vec<f32> = records.iter().map(|r| r.get_target(&target_type)).collect();

        let feature_stats = FeatureStats::from_features(&features);

        Ok(Self {
            records,
            features,
            targets,
            feature_stats,
            target_type,
        })
    }

    /// Parse raw CSV record into TelecomRecord
    fn parse_csv_record(record: HashMap<String, String>) -> TrainingResult<TelecomRecord> {
        // Helper function to parse float with fallback
        let parse_float = |key: &str, default: f32| -> f32 {
            record.get(key)
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(default)
        };

        let parse_u32 = |key: &str, default: u32| -> u32 {
            record.get(key)
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(default)
        };

        let parse_u8 = |key: &str, default: u8| -> u8 {
            record.get(key)
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(default)
        };

        let get_string = |key: &str| -> String {
            record.get(key).cloned().unwrap_or_default()
        };

        Ok(TelecomRecord {
            timestamp: get_string("HEURE(PSDATE)"),
            enodeb_code: get_string("CODE_ELT_ENODEB"),
            enodeb_name: get_string("ENODEB"),
            cell_code: get_string("CODE_ELT_CELLULE"),
            cell_name: get_string("CELLULE"),
            lte_band: get_string("SYS.BANDE"),
            num_bands: parse_u8("SYS.NB_BANDES", 4),
            cell_availability: parse_float("CELL_AVAILABILITY_%", 0.0),
            volte_traffic: parse_float("VOLTE_TRAFFIC (ERL)", 0.0),
            erab_traffic: parse_float("ERIC_TRAFF_ERAB_ERL", 0.0),
            connected_users_avg: parse_float("RRC_CONNECTED_ USERS_AVERAGE", 0.0),
            ul_volume_gb: parse_float("UL_VOLUME_PDCP_GBYTES", 0.0),
            dl_volume_gb: parse_float("DL_VOLUME_PDCP_GBYTES", 0.0),
            dcr_volte: parse_float("4G_LTE_DCR_VOLTE", 0.0),
            erab_drop_qci5: parse_float("ERAB_DROP_RATE_QCI_5", 0.0),
            erab_drop_qci8: parse_float("ERAB_DROP_RATE_QCI_8", 0.0),
            ue_context_att: parse_u32("NB_UE_CTXT_ATT", 0),
            ue_context_abnorm_rel_pct: parse_float("UE_CTXT_ABNORM_REL_%", 0.0),
            avg_dl_user_throughput: parse_float("&_AVE_4G_LTE_DL_USER_THRPUT", 0.0),
            avg_ul_user_throughput: parse_float("&_AVE_4G_LTE_UL_USER_THRPUT", 0.0),
            sinr_pusch_avg: parse_float("SINR_PUSCH_AVG", 0.0),
            sinr_pucch_avg: parse_float("SINR_PUCCH_AVG", 0.0),
            ul_rssi_total: parse_float("UL_RSSI_TOTAL", -120.0),
            mac_dl_bler: parse_float("MAC_DL_BLER", 0.0),
            mac_ul_bler: parse_float("MAC_UL_BLER", 0.0),
            dl_packet_error_loss_rate: parse_float("DL_PACKET_ERROR_LOSS_RATE", 0.0),
            ul_packet_loss_rate: parse_float("UL_PACKET_LOSS_RATE", 0.0),
            dl_latency_avg: parse_float("DL_LATENCY_AVG", 0.0),
            handover_success_rate: parse_float("LTE_INTRA_FREQ_HO_SR", 100.0),
        })
    }

    /// Validate record for basic sanity
    fn is_valid_record(record: &TelecomRecord) -> bool {
        // Basic validation rules
        record.cell_availability >= 0.0 && record.cell_availability <= 100.0
            && record.volte_traffic >= 0.0
            && record.erab_traffic >= 0.0
            && record.connected_users_avg >= 0.0
    }

    /// Normalize all features in the dataset
    pub fn normalize_features(&mut self) {
        for features in &mut self.features {
            self.feature_stats.normalize(features);
        }
    }

    /// Split dataset into training and validation sets
    pub fn train_val_split(&self, val_ratio: f32) -> TrainingResult<(Self, Self)> {
        if !(0.0..=1.0).contains(&val_ratio) {
            return Err(TrainingError::InvalidInput("Validation ratio must be between 0.0 and 1.0".into()));
        }

        let total_samples = self.records.len();
        let val_size = (total_samples as f32 * val_ratio) as usize;
        let train_size = total_samples - val_size;

        let train_records = self.records[..train_size].to_vec();
        let val_records = self.records[train_size..].to_vec();

        let train_features = self.features[..train_size].to_vec();
        let val_features = self.features[train_size..].to_vec();

        let train_targets = self.targets[..train_size].to_vec();
        let val_targets = self.targets[train_size..].to_vec();

        let train_dataset = Self {
            records: train_records,
            features: train_features,
            targets: train_targets,
            feature_stats: self.feature_stats.clone(),
            target_type: self.target_type,
        };

        let val_dataset = Self {
            records: val_records,
            features: val_features,
            targets: val_targets,
            feature_stats: self.feature_stats.clone(),
            target_type: self.target_type,
        };

        Ok((train_dataset, val_dataset))
    }

    /// Get feature count
    pub fn feature_count(&self) -> usize {
        if self.features.is_empty() {
            0
        } else {
            self.features[0].len()
        }
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_feature_stats() {
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];

        let stats = FeatureStats::from_features(&features);
        
        assert_eq!(stats.means, vec![2.0, 4.0, 6.0]);
        assert_eq!(stats.mins, vec![1.0, 2.0, 3.0]);
        assert_eq!(stats.maxs, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_normalization() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let stats = FeatureStats::from_features(&features);
        
        let mut test_features = vec![2.0, 3.0];
        stats.normalize(&mut test_features);
        
        // Should be normalized around 0
        assert!(test_features[0].abs() < 1.0);
        assert!(test_features[1].abs() < 1.0);
    }

    #[test]
    fn test_target_types() {
        let record = TelecomRecord {
            timestamp: "2025-06-27 00".to_string(),
            enodeb_code: "81371".to_string(),
            enodeb_name: "TEST_NR".to_string(),
            cell_code: "20830980".to_string(),
            cell_name: "TEST_F1".to_string(),
            lte_band: "LTE800".to_string(),
            num_bands: 4,
            cell_availability: 95.5,
            volte_traffic: 10.0,
            erab_traffic: 20.0,
            connected_users_avg: 15.0,
            ul_volume_gb: 1.0,
            dl_volume_gb: 2.0,
            dcr_volte: 0.1,
            erab_drop_qci5: 0.05,
            erab_drop_qci8: 0.02,
            ue_context_att: 1000,
            ue_context_abnorm_rel_pct: 0.5,
            avg_dl_user_throughput: 50000.0,
            avg_ul_user_throughput: 25000.0,
            sinr_pusch_avg: 15.0,
            sinr_pucch_avg: 12.0,
            ul_rssi_total: -110.0,
            mac_dl_bler: 0.01,
            mac_ul_bler: 0.02,
            dl_packet_error_loss_rate: 0.001,
            ul_packet_loss_rate: 0.002,
            dl_latency_avg: 20.0,
            handover_success_rate: 98.0,
        };

        assert_eq!(record.get_target(&TargetType::CellAvailability), 95.5);
        assert_eq!(record.get_target(&TargetType::VoLTETraffic), 10.0);
        assert_eq!(record.get_target(&TargetType::UserThroughput), 37500.0);
        
        let quality_score = record.get_target(&TargetType::QualityScore);
        assert!(quality_score > 0.0 && quality_score <= 1.0);
    }
}