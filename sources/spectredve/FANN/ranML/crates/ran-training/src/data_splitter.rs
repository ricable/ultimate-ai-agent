//! Data splitting utilities for creating train/test datasets from CSV

use crate::data::{TelecomDataset, TelecomRecord, TargetType};
use crate::error::{TrainingError, TrainingResult};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Configuration for data splitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSplitConfig {
    /// Percentage of data to use for training (0.0 to 1.0)
    pub train_ratio: f32,
    /// Target type for prediction
    pub target_type: TargetType,
    /// Maximum number of records to process (None for all)
    pub max_records: Option<usize>,
    /// Random seed for reproducible splits
    pub seed: Option<u64>,
    /// Whether to shuffle data before splitting
    pub shuffle: bool,
    /// Whether to stratify the split (balance target distribution)
    pub stratify: bool,
}

impl Default for DataSplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.8,
            target_type: TargetType::CellAvailability,
            max_records: None,
            seed: Some(42),
            shuffle: true,
            stratify: false,
        }
    }
}

/// JSON format for serialized datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetJson {
    /// Feature matrix (samples x features)
    pub features: Vec<Vec<f32>>,
    /// Target values
    pub targets: Vec<f32>,
    /// Feature names for reference
    pub feature_names: Vec<String>,
    /// Target variable name
    pub target_name: String,
    /// Normalization statistics (optional)
    pub normalization_stats: Option<NormalizationStats>,
}

/// Normalization statistics for features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    /// Mean values for each feature
    pub means: Vec<f32>,
    /// Standard deviations for each feature
    pub stds: Vec<f32>,
    /// Minimum values for each feature
    pub mins: Vec<f32>,
    /// Maximum values for each feature
    pub maxs: Vec<f32>,
}

/// Data splitter for creating train/test datasets
pub struct DataSplitter {
    config: DataSplitConfig,
}

impl DataSplitter {
    /// Create new data splitter with configuration
    pub fn new(config: DataSplitConfig) -> Self {
        Self { config }
    }

    /// Create data splitter with default configuration
    pub fn default() -> Self {
        Self::new(DataSplitConfig::default())
    }

    /// Split CSV data into train and test sets, saving as JSON files
    pub fn split_csv_to_json<P: AsRef<Path>>(
        &self,
        csv_path: P,
        train_output: P,
        test_output: P,
    ) -> TrainingResult<SplitInfo> {
        // Load dataset from CSV
        let mut dataset = TelecomDataset::from_csv(
            csv_path,
            self.config.target_type,
            self.config.max_records,
        )?;

        // Shuffle if requested
        if self.config.shuffle {
            self.shuffle_dataset(&mut dataset)?;
        }

        // Split dataset
        let (train_dataset, test_dataset) = if self.config.stratify {
            self.stratified_split(&dataset)?
        } else {
            self.simple_split(&dataset)?
        };

        // Convert to JSON format
        let train_json = self.dataset_to_json(&train_dataset)?;
        let test_json = self.dataset_to_json(&test_dataset)?;

        // Save to files
        self.save_json(&train_json, train_output)?;
        self.save_json(&test_json, test_output)?;

        // Return split information
        Ok(SplitInfo {
            total_samples: dataset.sample_count(),
            train_samples: train_dataset.sample_count(),
            test_samples: test_dataset.sample_count(),
            feature_count: dataset.feature_count(),
            target_type: self.config.target_type,
        })
    }

    /// Simple random split based on train_ratio
    fn simple_split(&self, dataset: &TelecomDataset) -> TrainingResult<(TelecomDataset, TelecomDataset)> {
        let total_samples = dataset.sample_count();
        let train_size = (total_samples as f32 * self.config.train_ratio) as usize;
        
        let train_records = dataset.records[..train_size].to_vec();
        let test_records = dataset.records[train_size..].to_vec();

        let train_features = dataset.features[..train_size].to_vec();
        let test_features = dataset.features[train_size..].to_vec();

        let train_targets = dataset.targets[..train_size].to_vec();
        let test_targets = dataset.targets[train_size..].to_vec();

        let train_dataset = TelecomDataset {
            records: train_records,
            features: train_features,
            targets: train_targets,
            feature_stats: dataset.feature_stats.clone(),
            target_type: dataset.target_type,
        };

        let test_dataset = TelecomDataset {
            records: test_records,
            features: test_features,
            targets: test_targets,
            feature_stats: dataset.feature_stats.clone(),
            target_type: dataset.target_type,
        };

        Ok((train_dataset, test_dataset))
    }

    /// Stratified split to maintain target distribution
    fn stratified_split(&self, dataset: &TelecomDataset) -> TrainingResult<(TelecomDataset, TelecomDataset)> {
        // For continuous targets, we'll bin them into quartiles for stratification
        let mut target_indices: Vec<(f32, usize)> = dataset.targets
            .iter()
            .enumerate()
            .map(|(i, &target)| (target, i))
            .collect();
        
        target_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let total_samples = dataset.sample_count();
        let train_size = (total_samples as f32 * self.config.train_ratio) as usize;
        
        let quartile_size = total_samples / 4;
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        // Select proportionally from each quartile
        for quartile in 0..4 {
            let start = quartile * quartile_size;
            let end = if quartile == 3 { total_samples } else { (quartile + 1) * quartile_size };
            
            let quartile_train_size = ((end - start) as f32 * self.config.train_ratio) as usize;
            
            for i in start..end {
                if i - start < quartile_train_size {
                    train_indices.push(target_indices[i].1);
                } else {
                    test_indices.push(target_indices[i].1);
                }
            }
        }

        // Create datasets from selected indices
        let train_records: Vec<TelecomRecord> = train_indices
            .iter()
            .map(|&i| dataset.records[i].clone())
            .collect();
        
        let test_records: Vec<TelecomRecord> = test_indices
            .iter()
            .map(|&i| dataset.records[i].clone())
            .collect();

        let train_features: Vec<Vec<f32>> = train_indices
            .iter()
            .map(|&i| dataset.features[i].clone())
            .collect();
        
        let test_features: Vec<Vec<f32>> = test_indices
            .iter()
            .map(|&i| dataset.features[i].clone())
            .collect();

        let train_targets: Vec<f32> = train_indices
            .iter()
            .map(|&i| dataset.targets[i])
            .collect();
        
        let test_targets: Vec<f32> = test_indices
            .iter()
            .map(|&i| dataset.targets[i])
            .collect();

        let train_dataset = TelecomDataset {
            records: train_records,
            features: train_features,
            targets: train_targets,
            feature_stats: dataset.feature_stats.clone(),
            target_type: dataset.target_type,
        };

        let test_dataset = TelecomDataset {
            records: test_records,
            features: test_features,
            targets: test_targets,
            feature_stats: dataset.feature_stats.clone(),
            target_type: dataset.target_type,
        };

        Ok((train_dataset, test_dataset))
    }

    /// Shuffle dataset using Fisher-Yates algorithm
    fn shuffle_dataset(&self, dataset: &mut TelecomDataset) -> TrainingResult<()> {
        use rand::prelude::*;
        
        let mut rng = if let Some(seed) = self.config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let n = dataset.sample_count();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            
            // Swap records
            dataset.records.swap(i, j);
            dataset.features.swap(i, j);
            dataset.targets.swap(i, j);
        }

        Ok(())
    }

    /// Convert dataset to JSON format
    fn dataset_to_json(&self, dataset: &TelecomDataset) -> TrainingResult<DatasetJson> {
        let feature_names = self.get_feature_names();
        let target_name = self.get_target_name();

        let normalization_stats = Some(NormalizationStats {
            means: dataset.feature_stats.means.clone(),
            stds: dataset.feature_stats.stds.clone(),
            mins: dataset.feature_stats.mins.clone(),
            maxs: dataset.feature_stats.maxs.clone(),
        });

        Ok(DatasetJson {
            features: dataset.features.clone(),
            targets: dataset.targets.clone(),
            feature_names,
            target_name,
            normalization_stats,
        })
    }

    /// Get feature names based on TelecomRecord structure
    fn get_feature_names(&self) -> Vec<String> {
        vec![
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
        ]
    }

    /// Get target name based on target type
    fn get_target_name(&self) -> String {
        match self.config.target_type {
            TargetType::CellAvailability => "cell_availability".to_string(),
            TargetType::VoLTETraffic => "volte_traffic".to_string(),
            TargetType::UserThroughput => "user_throughput".to_string(),
            TargetType::QualityScore => "quality_score".to_string(),
        }
    }

    /// Save JSON dataset to file
    fn save_json<P: AsRef<Path>>(&self, dataset: &DatasetJson, path: P) -> TrainingResult<()> {
        let mut file = File::create(path)?;
        let json_str = serde_json::to_string_pretty(dataset)?;
        file.write_all(json_str.as_bytes())?;
        Ok(())
    }

    /// Load JSON dataset from file
    pub fn load_json<P: AsRef<Path>>(path: P) -> TrainingResult<DatasetJson> {
        let content = std::fs::read_to_string(path)?;
        let dataset: DatasetJson = serde_json::from_str(&content)?;
        Ok(dataset)
    }
}

/// Information about the data split
#[derive(Debug, Clone)]
pub struct SplitInfo {
    /// Total number of samples in original dataset
    pub total_samples: usize,
    /// Number of training samples
    pub train_samples: usize,
    /// Number of test samples
    pub test_samples: usize,
    /// Number of features
    pub feature_count: usize,
    /// Target type used for prediction
    pub target_type: TargetType,
}

impl SplitInfo {
    /// Display split information
    pub fn display(&self) {
        println!("=== Data Split Information ===");
        println!("Total samples: {}", self.total_samples);
        println!("Training samples: {} ({:.1}%)", 
                 self.train_samples, 
                 (self.train_samples as f32 / self.total_samples as f32) * 100.0);
        println!("Test samples: {} ({:.1}%)", 
                 self.test_samples, 
                 (self.test_samples as f32 / self.total_samples as f32) * 100.0);
        println!("Feature count: {}", self.feature_count);
        println!("Target type: {:?}", self.target_type);
    }
}

/// Utility function to create train/test split from CSV
pub fn create_train_test_split<P: AsRef<Path>>(
    csv_path: P,
    train_output: P,
    test_output: P,
    config: Option<DataSplitConfig>,
) -> TrainingResult<SplitInfo> {
    let config = config.unwrap_or_default();
    let splitter = DataSplitter::new(config);
    splitter.split_csv_to_json(csv_path, train_output, test_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[test]
    fn test_data_split_config() {
        let config = DataSplitConfig::default();
        assert_eq!(config.train_ratio, 0.8);
        assert_eq!(config.seed, Some(42));
        assert!(config.shuffle);
    }

    #[test]
    fn test_feature_names() {
        let splitter = DataSplitter::default();
        let feature_names = splitter.get_feature_names();
        assert_eq!(feature_names.len(), 21);
        assert!(feature_names.contains(&"cell_availability".to_string()));
        assert!(feature_names.contains(&"volte_traffic".to_string()));
    }

    #[test]
    fn test_target_names() {
        let splitter = DataSplitter::default();
        assert_eq!(splitter.get_target_name(), "cell_availability");
        
        let mut config = DataSplitConfig::default();
        config.target_type = TargetType::VoLTETraffic;
        let splitter = DataSplitter::new(config);
        assert_eq!(splitter.get_target_name(), "volte_traffic");
    }

    #[test]
    fn test_json_serialization() {
        let dataset = DatasetJson {
            features: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            targets: vec![100.0, 95.0],
            feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
            target_name: "target".to_string(),
            normalization_stats: None,
        };

        let json_str = serde_json::to_string(&dataset).unwrap();
        let deserialized: DatasetJson = serde_json::from_str(&json_str).unwrap();
        
        assert_eq!(dataset.features, deserialized.features);
        assert_eq!(dataset.targets, deserialized.targets);
    }
}