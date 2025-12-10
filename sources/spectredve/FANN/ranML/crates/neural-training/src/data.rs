//! Data loading and preprocessing for telecom neural network training

use anyhow::{Context, Result};
use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Telecom dataset record matching fanndata.csv structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelecomRecord {
    pub timestamp: String,
    pub enodeb_code: String, 
    pub enodeb_name: String,
    pub cell_code: String,
    pub cell_name: String,
    pub lte_band: String,
    pub num_bands: u32,
    pub cell_availability: f32,
    pub volte_traffic: f32,
    pub erab_traffic: f32,
    pub connected_users_avg: f32,
    pub ul_volume_gb: f32,
    pub dl_volume_gb: f32,
    pub dcr_volte: f32,
    pub erab_drop_qci5: f32,
    pub erab_drop_qci8: f32,
    pub ue_context_att: u32,
    pub ue_context_abnorm_rel_pct: f32,
    pub avg_dl_user_throughput: f32,
    pub avg_ul_user_throughput: f32,
    pub sinr_pusch_avg: f32,
    pub sinr_pucch_avg: f32,
    pub ul_rssi_total: f32,
    pub mac_dl_bler: f32,
    pub mac_ul_bler: f32,
    pub dl_packet_error_loss_rate: f32,
    pub ul_packet_loss_rate: f32,
    pub dl_latency_avg: f32,
    pub handover_success_rate: f32,
    // Add other fields as needed for complete CSV coverage
}

/// Processed dataset for neural network training
#[derive(Debug, Clone)]
pub struct TelecomDataset {
    pub features: Array2<f32>,
    pub targets: Array1<f32>,
    pub feature_names: Vec<String>,
    pub target_name: String,
    pub normalization_stats: Option<NormalizationStats>,
}

/// Statistics for feature normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub means: Array1<f32>,
    pub stds: Array1<f32>,
    pub mins: Array1<f32>,
    pub maxs: Array1<f32>,
}

/// Data split for training and testing
#[derive(Debug, Clone)]
pub struct DataSplit {
    pub train: TelecomDataset,
    pub test: TelecomDataset,
}

/// Telecom data loader with preprocessing capabilities
pub struct TelecomDataLoader;

impl TelecomDataLoader {
    /// Load telecom data from CSV file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<TelecomDataset> {
        log::info!("Loading telecom data from: {:?}", path.as_ref());
        
        let mut reader = ReaderBuilder::new()
            .delimiter(b';')
            .has_headers(true)
            .from_path(path)?;
        
        let mut records = Vec::new();
        let headers = reader.headers()?.clone();
        
        for result in reader.records() {
            let record = result?;
            if let Ok(telecom_record) = Self::parse_record(&record, &headers) {
                records.push(telecom_record);
            }
        }
        
        log::info!("Loaded {} records", records.len());
        
        // Convert to feature matrix and target vector
        let dataset = Self::records_to_dataset(records)?;
        Ok(dataset)
    }
    
    /// Parse a CSV record into TelecomRecord
    fn parse_record(record: &csv::StringRecord, headers: &csv::StringRecord) -> Result<TelecomRecord> {
        let mut field_map = HashMap::new();
        for (i, header) in headers.iter().enumerate() {
            if let Some(value) = record.get(i) {
                field_map.insert(header.to_lowercase(), value);
            }
        }
        
        // Parse required fields with error handling
        let cell_availability = field_map.get("cell_availability_%")
            .or_else(|| field_map.get("cell_availability"))
            .unwrap_or("0")
            .parse::<f32>()
            .unwrap_or(0.0);
            
        let volte_traffic = field_map.get("volte_traffic (erl)")
            .or_else(|| field_map.get("volte_traffic"))
            .unwrap_or("0")
            .parse::<f32>()
            .unwrap_or(0.0);
            
        Ok(TelecomRecord {
            timestamp: field_map.get("heure(psdate)").unwrap_or("").to_string(),
            enodeb_code: field_map.get("code_elt_enodeb").unwrap_or("").to_string(),
            enodeb_name: field_map.get("enodeb").unwrap_or("").to_string(),
            cell_code: field_map.get("code_elt_cellule").unwrap_or("").to_string(),
            cell_name: field_map.get("cellule").unwrap_or("").to_string(),
            lte_band: field_map.get("sys.bande").unwrap_or("").to_string(),
            num_bands: field_map.get("sys.nb_bandes")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0),
            cell_availability,
            volte_traffic,
            erab_traffic: field_map.get("eric_traff_erab_erl")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            connected_users_avg: field_map.get("rrc_connected_ users_average")
                .or_else(|| field_map.get("rrc_connected_users_average"))
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            ul_volume_gb: field_map.get("ul_volume_pdcp_gbytes")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            dl_volume_gb: field_map.get("dl_volume_pdcp_gbytes")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            dcr_volte: field_map.get("4g_lte_dcr_volte")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            erab_drop_qci5: field_map.get("erab_drop_rate_qci_5")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            erab_drop_qci8: field_map.get("erab_drop_rate_qci_8")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            ue_context_att: field_map.get("nb_ue_ctxt_att")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0),
            ue_context_abnorm_rel_pct: field_map.get("ue_ctxt_abnorm_rel_%")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            avg_dl_user_throughput: field_map.get("&_ave_4g_lte_dl_user_thrput")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            avg_ul_user_throughput: field_map.get("&_ave_4g_lte_ul_user_thrput")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            sinr_pusch_avg: field_map.get("sinr_pusch_avg")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            sinr_pucch_avg: field_map.get("sinr_pucch_avg")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            ul_rssi_total: field_map.get("ul_rssi_total")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            mac_dl_bler: field_map.get("mac_dl_bler")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            mac_ul_bler: field_map.get("mac_ul_bler")
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            dl_packet_error_loss_rate: field_map.get("dl_packet_error_loss_rate")
                .map(|s| s.parse().unwrap_or(0.0))
                .unwrap_or(0.0),
            ul_packet_loss_rate: field_map.get("ul_packet_loss_rate")
                .map(|s| s.parse().unwrap_or(0.0))
                .unwrap_or(0.0),
            dl_latency_avg: field_map.get("dl_latency_avg")
                .map(|s| s.parse().unwrap_or(0.0))
                .unwrap_or(0.0),
            handover_success_rate: field_map.get("lte_intra_freq_ho_sr")
                .map(|s| s.parse().unwrap_or(0.0))
                .unwrap_or(0.0),
        })
    }
    
    /// Convert records to feature matrix and target vector
    fn records_to_dataset(records: Vec<TelecomRecord>) -> Result<TelecomDataset> {
        if records.is_empty() {
            return Err(anyhow::anyhow!("No valid records found"));
        }
        
        let num_samples = records.len();
        
        // Define feature extraction function
        let extract_features = |record: &TelecomRecord| -> Vec<f32> {
            vec![
                record.num_bands as f32,
                record.volte_traffic,
                record.erab_traffic,
                record.connected_users_avg,
                record.ul_volume_gb,
                record.dl_volume_gb,
                record.dcr_volte,
                record.erab_drop_qci5,
                record.erab_drop_qci8,
                record.ue_context_att as f32,
                record.ue_context_abnorm_rel_pct,
                record.avg_dl_user_throughput,
                record.avg_ul_user_throughput,
                record.sinr_pusch_avg,
                record.sinr_pucch_avg,
                record.ul_rssi_total,
                record.mac_dl_bler,
                record.mac_ul_bler,
                record.dl_packet_error_loss_rate,
                record.ul_packet_loss_rate,
                record.dl_latency_avg,
                record.handover_success_rate,
            ]
        };
        
        // Extract features and targets
        let feature_vecs: Vec<Vec<f32>> = records.iter().map(extract_features).collect();
        let num_features = feature_vecs[0].len();
        
        // Create feature matrix
        let mut features = Array2::zeros((num_samples, num_features));
        for (i, feature_vec) in feature_vecs.iter().enumerate() {
            for (j, &value) in feature_vec.iter().enumerate() {
                features[[i, j]] = value;
            }
        }
        
        // Use cell availability as target
        let targets: Array1<f32> = records.iter()
            .map(|r| r.cell_availability)
            .collect::<Vec<f32>>()
            .into();
        
        let feature_names = vec![
            "num_bands", "volte_traffic", "erab_traffic", "connected_users_avg",
            "ul_volume_gb", "dl_volume_gb", "dcr_volte", "erab_drop_qci5",
            "erab_drop_qci8", "ue_context_att", "ue_context_abnorm_rel_pct",
            "avg_dl_user_throughput", "avg_ul_user_throughput", "sinr_pusch_avg",
            "sinr_pucch_avg", "ul_rssi_total", "mac_dl_bler", "mac_ul_bler",
            "dl_packet_error_loss_rate", "ul_packet_loss_rate", "dl_latency_avg",
            "handover_success_rate"
        ].iter().map(|s| s.to_string()).collect();
        
        Ok(TelecomDataset {
            features,
            targets,
            feature_names,
            target_name: "cell_availability".to_string(),
            normalization_stats: None,
        })
    }
}

impl TelecomDataset {
    /// Split dataset into training and testing sets
    pub fn split_train_test(&self, train_ratio: f32) -> Result<DataSplit> {
        let num_samples = self.features.nrows();
        let train_size = (num_samples as f32 * train_ratio) as usize;
        
        let mut indices: Vec<usize> = (0..num_samples).collect();
        indices.shuffle(&mut thread_rng());
        
        let train_indices = &indices[..train_size];
        let test_indices = &indices[train_size..];
        
        let train_features = self.features.select(ndarray::Axis(0), train_indices);
        let test_features = self.features.select(ndarray::Axis(0), test_indices);
        
        let train_targets = self.targets.select(ndarray::Axis(0), train_indices);
        let test_targets = self.targets.select(ndarray::Axis(0), test_indices);
        
        let train = TelecomDataset {
            features: train_features,
            targets: train_targets,
            feature_names: self.feature_names.clone(),
            target_name: self.target_name.clone(),
            normalization_stats: self.normalization_stats.clone(),
        };
        
        let test = TelecomDataset {
            features: test_features,
            targets: test_targets,
            feature_names: self.feature_names.clone(),
            target_name: self.target_name.clone(),
            normalization_stats: self.normalization_stats.clone(),
        };
        
        Ok(DataSplit { train, test })
    }
    
    /// Normalize features using z-score normalization
    pub fn normalize(&mut self) -> Result<()> {
        let means = self.features.mean_axis(ndarray::Axis(0)).unwrap();
        let stds = self.features.std_axis(ndarray::Axis(0), 0.0);
        let mins = self.features.fold_axis(ndarray::Axis(0), f32::INFINITY, |&a, &b| a.min(b));
        let maxs = self.features.fold_axis(ndarray::Axis(0), f32::NEG_INFINITY, |&a, &b| a.max(b));
        
        // Normalize features
        for mut row in self.features.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                if stds[i] > 0.0 {
                    *val = (*val - means[i]) / stds[i];
                }
            }
        }
        
        self.normalization_stats = Some(NormalizationStats {
            means,
            stds,
            mins,
            maxs,
        });
        
        Ok(())
    }
    
    /// Save dataset to JSON files
    pub fn save_splits<P: AsRef<Path>>(&self, train_path: P, test_path: P, split: &DataSplit) -> Result<()> {
        // Convert to serializable format
        let train_data = SerializableDataset::from_dataset(&split.train);
        let test_data = SerializableDataset::from_dataset(&split.test);
        
        // Save to JSON files
        let train_json = serde_json::to_string_pretty(&train_data)
            .context("Failed to serialize training data")?;
        std::fs::write(train_path, train_json)
            .context("Failed to write training data file")?;
        
        let test_json = serde_json::to_string_pretty(&test_data)
            .context("Failed to serialize test data")?;
        std::fs::write(test_path, test_json)
            .context("Failed to write test data file")?;
        
        log::info!("Saved train data: {} samples", split.train.features.nrows());
        log::info!("Saved test data: {} samples", split.test.features.nrows());
        
        Ok(())
    }
}

/// Serializable version of dataset for JSON export
#[derive(Serialize, Deserialize)]
struct SerializableDataset {
    features: Vec<Vec<f32>>,
    targets: Vec<f32>,
    feature_names: Vec<String>,
    target_name: String,
    normalization_stats: Option<NormalizationStats>,
}

impl SerializableDataset {
    fn from_dataset(dataset: &TelecomDataset) -> Self {
        let features = dataset.features.rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        
        Self {
            features,
            targets: dataset.targets.to_vec(),
            feature_names: dataset.feature_names.clone(),
            target_name: dataset.target_name.clone(),
            normalization_stats: dataset.normalization_stats.clone(),
        }
    }
}