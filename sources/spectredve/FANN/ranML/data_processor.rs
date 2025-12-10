//! Standalone data processor to create train.json and test.json from fanndata.csv

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::error::Error;
use serde::{Deserialize, Serialize};

/// Configuration for data splitting
#[derive(Debug, Clone)]
struct DataSplitConfig {
    train_ratio: f32,
    shuffle: bool,
    seed: u64,
}

impl Default for DataSplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.8,
            shuffle: true,
            seed: 42,
        }
    }
}

/// JSON format for serialized datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetJson {
    features: Vec<Vec<f32>>,
    targets: Vec<f32>,
    feature_names: Vec<String>,
    target_name: String,
    normalization_stats: Option<NormalizationStats>,
}

/// Normalization statistics for features
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NormalizationStats {
    means: Vec<f32>,
    stds: Vec<f32>,
    mins: Vec<f32>,
    maxs: Vec<f32>,
}

/// Simplified telecom record structure
#[derive(Debug, Clone)]
struct TelecomRecord {
    num_bands: f32,
    cell_availability: f32,
    volte_traffic: f32,
    erab_traffic: f32,
    connected_users_avg: f32,
    ul_volume_gb: f32,
    dl_volume_gb: f32,
    dcr_volte: f32,
    erab_drop_qci5: f32,
    erab_drop_qci8: f32,
    ue_context_att: f32,
    ue_context_abnorm_rel_pct: f32,
    avg_dl_user_throughput: f32,
    avg_ul_user_throughput: f32,
    sinr_pusch_avg: f32,
    sinr_pucch_avg: f32,
    ul_rssi_total: f32,
    mac_dl_bler: f32,
    mac_ul_bler: f32,
    dl_packet_error_loss_rate: f32,
    ul_packet_loss_rate: f32,
    dl_latency_avg: f32,
}

impl TelecomRecord {
    /// Parse from CSV row
    fn from_csv_row(headers: &[String], values: &[String]) -> Option<Self> {
        let get_value = |key: &str| -> f32 {
            headers.iter()
                .position(|h| h == key)
                .and_then(|i| values.get(i))
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.0)
        };

        let get_value_default = |key: &str, default: f32| -> f32 {
            headers.iter()
                .position(|h| h == key)
                .and_then(|i| values.get(i))
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(default)
        };

        // Only process records with valid cell availability
        let cell_availability = get_value("CELL_AVAILABILITY_%");
        if cell_availability < 0.0 || cell_availability > 100.0 {
            return None;
        }

        Some(Self {
            num_bands: get_value_default("SYS.NB_BANDES", 4.0),
            cell_availability,
            volte_traffic: get_value("VOLTE_TRAFFIC (ERL)"),
            erab_traffic: get_value("ERIC_TRAFF_ERAB_ERL"),
            connected_users_avg: get_value("RRC_CONNECTED_ USERS_AVERAGE"),
            ul_volume_gb: get_value("UL_VOLUME_PDCP_GBYTES"),
            dl_volume_gb: get_value("DL_VOLUME_PDCP_GBYTES"),
            dcr_volte: get_value("4G_LTE_DCR_VOLTE"),
            erab_drop_qci5: get_value("ERAB_DROP_RATE_QCI_5"),
            erab_drop_qci8: get_value("ERAB_DROP_RATE_QCI_8"),
            ue_context_att: get_value("NB_UE_CTXT_ATT"),
            ue_context_abnorm_rel_pct: get_value("UE_CTXT_ABNORM_REL_%"),
            avg_dl_user_throughput: get_value("&_AVE_4G_LTE_DL_USER_THRPUT"),
            avg_ul_user_throughput: get_value("&_AVE_4G_LTE_UL_USER_THRPUT"),
            sinr_pusch_avg: get_value("SINR_PUSCH_AVG"),
            sinr_pucch_avg: get_value("SINR_PUCCH_AVG"),
            ul_rssi_total: get_value_default("UL_RSSI_TOTAL", -120.0),
            mac_dl_bler: get_value("MAC_DL_BLER"),
            mac_ul_bler: get_value("MAC_UL_BLER"),
            dl_packet_error_loss_rate: get_value("DL_PACKET_ERROR_LOSS_RATE"),
            ul_packet_loss_rate: get_value("UL_PACKET_LOSS_RATE"),
            dl_latency_avg: get_value("DL_LATENCY_AVG"),
        })
    }

    /// Convert to feature vector
    fn to_features(&self) -> Vec<f32> {
        vec![
            self.num_bands,
            self.volte_traffic,
            self.erab_traffic,
            self.connected_users_avg,
            self.ul_volume_gb,
            self.dl_volume_gb,
            self.dcr_volte,
            self.erab_drop_qci5,
            self.erab_drop_qci8,
            self.ue_context_att / 1000.0, // Normalize large values
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
        ]
    }

    /// Get target value (cell availability)
    fn get_target(&self) -> f32 {
        self.cell_availability
    }
}

/// Load CSV and parse records
fn load_csv_data<P: AsRef<Path>>(path: P) -> Result<Vec<TelecomRecord>, Box<dyn Error>> {
    println!("Loading CSV data from: {:?}", path.as_ref());
    
    let content = std::fs::read_to_string(path)?;
    let mut lines = content.lines();
    
    // Parse headers
    let header_line = lines.next().ok_or("Empty CSV file")?;
    let headers: Vec<String> = header_line.split(';').map(|s| s.to_string()).collect();
    
    println!("Found {} columns in CSV", headers.len());
    
    let mut records = Vec::new();
    let mut skipped = 0;
    let mut processed = 0;
    
    for (line_num, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        
        let values: Vec<String> = line.split(';').map(|s| s.to_string()).collect();
        
        if values.len() != headers.len() {
            skipped += 1;
            continue;
        }
        
        if let Some(record) = TelecomRecord::from_csv_row(&headers, &values) {
            records.push(record);
            processed += 1;
        } else {
            skipped += 1;
        }
        
        if processed % 10000 == 0 {
            println!("Processed {} records...", processed);
        }
    }
    
    println!("Loaded {} valid records, skipped {} invalid records", records.len(), skipped);
    Ok(records)
}

/// Calculate feature statistics
fn calculate_feature_stats(features: &[Vec<f32>]) -> NormalizationStats {
    if features.is_empty() {
        return NormalizationStats {
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
            if value.is_finite() {
                means[i] += value;
                mins[i] = mins[i].min(value);
                maxs[i] = maxs[i].max(value);
            }
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
    
    NormalizationStats { means, stds, mins, maxs }
}

/// Shuffle data using Fisher-Yates algorithm
fn shuffle_data(features: &mut Vec<Vec<f32>>, targets: &mut Vec<f32>, seed: u64) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut rng_state = seed;
    
    let n = features.len();
    for i in (1..n).rev() {
        // Simple LCG for deterministic shuffling
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let j = (rng_state as usize) % (i + 1);
        
        features.swap(i, j);
        targets.swap(i, j);
    }
}

/// Split data into train and test sets
fn split_data(
    mut features: Vec<Vec<f32>>,
    mut targets: Vec<f32>,
    config: &DataSplitConfig,
) -> (DatasetJson, DatasetJson) {
    println!("Splitting {} samples with ratio {:.1}%", features.len(), config.train_ratio * 100.0);
    
    // Shuffle if requested
    if config.shuffle {
        shuffle_data(&mut features, &mut targets, config.seed);
        println!("Data shuffled with seed {}", config.seed);
    }
    
    // Calculate split point
    let total_samples = features.len();
    let train_size = (total_samples as f32 * config.train_ratio) as usize;
    
    // Split features and targets
    let train_features = features[..train_size].to_vec();
    let test_features = features[train_size..].to_vec();
    let train_targets = targets[..train_size].to_vec();
    let test_targets = targets[train_size..].to_vec();
    
    // Calculate normalization stats on training data only
    let normalization_stats = calculate_feature_stats(&train_features);
    
    let feature_names = vec![
        "num_bands".to_string(),
        "volte_traffic".to_string(),
        "erab_traffic".to_string(),
        "connected_users_avg".to_string(),
        "ul_volume_gb".to_string(),
        "dl_volume_gb".to_string(),
        "dcr_volte".to_string(),
        "erab_drop_qci5".to_string(),
        "erab_drop_qci8".to_string(),
        "ue_context_att".to_string(),
        "ue_context_abnorm_rel_pct".to_string(),
        "avg_dl_user_throughput".to_string(),
        "avg_ul_user_throughput".to_string(),
        "sinr_pusch_avg".to_string(),
        "sinr_pucch_avg".to_string(),
        "ul_rssi_total".to_string(),
        "mac_dl_bler".to_string(),
        "mac_ul_bler".to_string(),
        "dl_packet_error_loss_rate".to_string(),
        "ul_packet_loss_rate".to_string(),
        "dl_latency_avg".to_string(),
    ];
    
    let train_dataset = DatasetJson {
        features: train_features,
        targets: train_targets,
        feature_names: feature_names.clone(),
        target_name: "cell_availability".to_string(),
        normalization_stats: Some(normalization_stats.clone()),
    };
    
    let test_dataset = DatasetJson {
        features: test_features,
        targets: test_targets,
        feature_names,
        target_name: "cell_availability".to_string(),
        normalization_stats: Some(normalization_stats),
    };
    
    println!("Train set: {} samples", train_dataset.features.len());
    println!("Test set: {} samples", test_dataset.features.len());
    
    (train_dataset, test_dataset)
}

/// Save dataset to JSON file
fn save_json<P: AsRef<Path>>(dataset: &DatasetJson, path: P) -> Result<(), Box<dyn Error>> {
    let json_str = serde_json::to_string_pretty(dataset)?;
    let mut file = File::create(path)?;
    file.write_all(json_str.as_bytes())?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== FANN Data Processor ===");
    
    let csv_path = "data/pm/fanndata.csv";
    let train_path = "data/pm/train.json";
    let test_path = "data/pm/test.json";
    
    println!("Input CSV: {}", csv_path);
    println!("Output train: {}", train_path);
    println!("Output test: {}", test_path);
    
    // Load CSV data
    let records = load_csv_data(csv_path)?;
    
    if records.is_empty() {
        return Err("No valid records found in CSV".into());
    }
    
    // Convert to features and targets
    let features: Vec<Vec<f32>> = records.iter().map(|r| r.to_features()).collect();
    let targets: Vec<f32> = records.iter().map(|r| r.get_target()).collect();
    
    println!("Extracted {} features per sample", features[0].len());
    
    // Split data
    let config = DataSplitConfig::default();
    let (train_dataset, test_dataset) = split_data(features, targets, &config);
    
    // Save datasets
    save_json(&train_dataset, train_path)?;
    save_json(&test_dataset, test_path)?;
    
    println!("\n=== Success! ===");
    println!("Training data saved to: {}", train_path);
    println!("Test data saved to: {}", test_path);
    println!("Features per sample: {}", train_dataset.feature_names.len());
    println!("Train samples: {}", train_dataset.features.len());
    println!("Test samples: {}", test_dataset.features.len());
    
    // Calculate some basic statistics
    let train_avg_target: f32 = train_dataset.targets.iter().sum::<f32>() / train_dataset.targets.len() as f32;
    let test_avg_target: f32 = test_dataset.targets.iter().sum::<f32>() / test_dataset.targets.len() as f32;
    
    println!("Train avg cell availability: {:.2}%", train_avg_target);
    println!("Test avg cell availability: {:.2}%", test_avg_target);
    
    Ok(())
}

// Include serde dependency
use serde;
use serde_json;