//! Data preprocessing utilities for neural network training

use crate::data::{FeatureStats, TelecomDataset};
use crate::error::{TrainingError, TrainingResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Preprocessing pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Whether to normalize features (z-score)
    pub normalize_features: bool,
    /// Whether to remove outliers
    pub remove_outliers: bool,
    /// Outlier threshold (standard deviations from mean)
    pub outlier_threshold: f32,
    /// Minimum samples required per class/bin
    pub min_samples_per_bin: usize,
    /// Handle missing values strategy
    pub missing_value_strategy: MissingValueStrategy,
    /// Feature selection method
    pub feature_selection: FeatureSelectionMethod,
    /// Target preprocessing
    pub target_preprocessing: TargetPreprocessing,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            normalize_features: true,
            remove_outliers: true,
            outlier_threshold: 3.0,
            min_samples_per_bin: 10,
            missing_value_strategy: MissingValueStrategy::Mean,
            feature_selection: FeatureSelectionMethod::None,
            target_preprocessing: TargetPreprocessing::None,
        }
    }
}

/// Strategy for handling missing values
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    /// Replace with mean value
    Mean,
    /// Replace with median value
    Median,
    /// Replace with zero
    Zero,
    /// Remove samples with missing values
    Remove,
    /// Forward fill from previous value
    ForwardFill,
}

/// Feature selection methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    /// No feature selection
    None,
    /// Select top K features by variance
    VarianceThreshold(f32),
    /// Select top K features by correlation with target
    Correlation(usize),
    /// Remove highly correlated features
    RemoveCorrelated(f32),
}

/// Target preprocessing options
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TargetPreprocessing {
    /// No preprocessing
    None,
    /// Log transform (for skewed distributions)
    Log,
    /// Square root transform
    Sqrt,
    /// Min-max scaling to [0, 1]
    MinMax,
    /// Z-score normalization
    ZScore,
}

/// Data preprocessing pipeline
pub struct DataPreprocessor {
    config: PreprocessingConfig,
    feature_stats: Option<FeatureStats>,
    target_stats: Option<TargetStats>,
    selected_features: Option<Vec<usize>>,
}

/// Statistics for target preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

impl DataPreprocessor {
    /// Create new preprocessor with configuration
    pub fn new(config: PreprocessingConfig) -> Self {
        Self {
            config,
            feature_stats: None,
            target_stats: None,
            selected_features: None,
        }
    }

    /// Create preprocessor with default configuration
    pub fn default() -> Self {
        Self::new(PreprocessingConfig::default())
    }

    /// Fit preprocessor to training data and transform
    pub fn fit_transform(&mut self, dataset: &mut TelecomDataset) -> TrainingResult<()> {
        self.fit(dataset)?;
        self.transform(dataset)
    }

    /// Fit preprocessor to training data (learn statistics)
    pub fn fit(&mut self, dataset: &TelecomDataset) -> TrainingResult<()> {
        // Handle missing values first
        let cleaned_features = self.handle_missing_values(&dataset.features)?;
        
        // Compute feature statistics
        self.feature_stats = Some(FeatureStats::from_features(&cleaned_features));
        
        // Compute target statistics
        self.target_stats = Some(self.compute_target_stats(&dataset.targets));
        
        // Feature selection
        self.selected_features = self.select_features(&cleaned_features, &dataset.targets)?;
        
        Ok(())
    }

    /// Transform dataset using fitted preprocessor
    pub fn transform(&self, dataset: &mut TelecomDataset) -> TrainingResult<()> {
        if self.feature_stats.is_none() {
            return Err(TrainingError::ConfigError("Preprocessor not fitted".into()));
        }

        // Handle missing values
        let mut cleaned_features = self.handle_missing_values(&dataset.features)?;
        
        // Remove outliers
        if self.config.remove_outliers {
            self.remove_outliers(&mut cleaned_features, &mut dataset.targets)?;
        }
        
        // Feature selection
        if let Some(ref selected_indices) = self.selected_features {
            cleaned_features = self.apply_feature_selection(&cleaned_features, selected_indices);
        }
        
        // Normalize features
        if self.config.normalize_features {
            if let Some(ref stats) = self.feature_stats {
                for features in &mut cleaned_features {
                    stats.normalize(features);
                }
            }
        }
        
        // Preprocess targets
        let transformed_targets = self.preprocess_targets(&dataset.targets)?;
        
        // Update dataset
        dataset.features = cleaned_features;
        dataset.targets = transformed_targets;
        
        Ok(())
    }

    /// Handle missing values in features
    fn handle_missing_values(&self, features: &[Vec<f32>]) -> TrainingResult<Vec<Vec<f32>>> {
        let mut cleaned = features.to_vec();
        
        match self.config.missing_value_strategy {
            MissingValueStrategy::Mean => {
                let stats = FeatureStats::from_features(&cleaned);
                for sample in &mut cleaned {
                    for (i, feature) in sample.iter_mut().enumerate() {
                        if !feature.is_finite() && i < stats.means.len() {
                            *feature = stats.means[i];
                        }
                    }
                }
            }
            MissingValueStrategy::Median => {
                // Compute medians
                let num_features = if cleaned.is_empty() { 0 } else { cleaned[0].len() };
                let mut medians = vec![0.0; num_features];
                
                for feat_idx in 0..num_features {
                    let mut values: Vec<f32> = cleaned
                        .iter()
                        .map(|sample| sample[feat_idx])
                        .filter(|v| v.is_finite())
                        .collect();
                    
                    if !values.is_empty() {
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        medians[feat_idx] = values[values.len() / 2];
                    }
                }
                
                // Replace missing values with medians
                for sample in &mut cleaned {
                    for (i, feature) in sample.iter_mut().enumerate() {
                        if !feature.is_finite() && i < medians.len() {
                            *feature = medians[i];
                        }
                    }
                }
            }
            MissingValueStrategy::Zero => {
                for sample in &mut cleaned {
                    for feature in sample {
                        if !feature.is_finite() {
                            *feature = 0.0;
                        }
                    }
                }
            }
            MissingValueStrategy::Remove => {
                cleaned.retain(|sample| sample.iter().all(|f| f.is_finite()));
            }
            MissingValueStrategy::ForwardFill => {
                for feat_idx in 0..cleaned[0].len() {
                    let mut last_valid = 0.0;
                    for sample in &mut cleaned {
                        if sample[feat_idx].is_finite() {
                            last_valid = sample[feat_idx];
                        } else {
                            sample[feat_idx] = last_valid;
                        }
                    }
                }
            }
        }
        
        Ok(cleaned)
    }

    /// Remove outliers from dataset
    fn remove_outliers(&self, features: &mut Vec<Vec<f32>>, targets: &mut Vec<f32>) -> TrainingResult<()> {
        let stats = FeatureStats::from_features(features);
        let mut keep_indices = Vec::new();
        
        for (i, sample) in features.iter().enumerate() {
            let mut is_outlier = false;
            
            for (j, &value) in sample.iter().enumerate() {
                if j < stats.means.len() && stats.stds[j] > 0.0 {
                    let z_score = (value - stats.means[j]).abs() / stats.stds[j];
                    if z_score > self.config.outlier_threshold {
                        is_outlier = true;
                        break;
                    }
                }
            }
            
            if !is_outlier {
                keep_indices.push(i);
            }
        }
        
        // Filter features and targets
        let filtered_features: Vec<Vec<f32>> = keep_indices.iter()
            .map(|&i| features[i].clone())
            .collect();
        let filtered_targets: Vec<f32> = keep_indices.iter()
            .map(|&i| targets[i])
            .collect();
        
        *features = filtered_features;
        *targets = filtered_targets;
        
        log::info!("Removed {} outliers, {} samples remaining", 
                  features.len() - keep_indices.len(), keep_indices.len());
        
        Ok(())
    }

    /// Select features based on configuration
    fn select_features(&self, features: &[Vec<f32>], targets: &[f32]) -> TrainingResult<Option<Vec<usize>>> {
        match self.config.feature_selection {
            FeatureSelectionMethod::None => Ok(None),
            FeatureSelectionMethod::VarianceThreshold(threshold) => {
                let stats = FeatureStats::from_features(features);
                let selected: Vec<usize> = stats.stds.iter()
                    .enumerate()
                    .filter(|(_, &std)| std > threshold)
                    .map(|(i, _)| i)
                    .collect();
                Ok(Some(selected))
            }
            FeatureSelectionMethod::Correlation(top_k) => {
                let correlations = self.compute_correlations(features, targets)?;
                let mut indexed_corrs: Vec<(usize, f32)> = correlations.into_iter().enumerate().collect();
                indexed_corrs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                
                let selected: Vec<usize> = indexed_corrs.into_iter()
                    .take(top_k)
                    .map(|(i, _)| i)
                    .collect();
                Ok(Some(selected))
            }
            FeatureSelectionMethod::RemoveCorrelated(threshold) => {
                let selected = self.remove_correlated_features(features, threshold)?;
                Ok(Some(selected))
            }
        }
    }

    /// Apply feature selection to dataset
    fn apply_feature_selection(&self, features: &[Vec<f32>], selected_indices: &[usize]) -> Vec<Vec<f32>> {
        features.iter()
            .map(|sample| {
                selected_indices.iter()
                    .map(|&i| sample.get(i).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect()
    }

    /// Compute correlations between features and target
    fn compute_correlations(&self, features: &[Vec<f32>], targets: &[f32]) -> TrainingResult<Vec<f32>> {
        if features.is_empty() {
            return Ok(Vec::new());
        }
        
        let num_features = features[0].len();
        let mut correlations = vec![0.0; num_features];
        
        let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;
        let target_std = {
            let var = targets.iter()
                .map(|&t| (t - target_mean).powi(2))
                .sum::<f32>() / targets.len() as f32;
            var.sqrt()
        };
        
        for feat_idx in 0..num_features {
            let feat_values: Vec<f32> = features.iter().map(|s| s[feat_idx]).collect();
            let feat_mean = feat_values.iter().sum::<f32>() / feat_values.len() as f32;
            let feat_std = {
                let var = feat_values.iter()
                    .map(|&f| (f - feat_mean).powi(2))
                    .sum::<f32>() / feat_values.len() as f32;
                var.sqrt()
            };
            
            if feat_std > 0.0 && target_std > 0.0 {
                let covariance = feat_values.iter()
                    .zip(targets.iter())
                    .map(|(&f, &t)| (f - feat_mean) * (t - target_mean))
                    .sum::<f32>() / feat_values.len() as f32;
                
                correlations[feat_idx] = covariance / (feat_std * target_std);
            }
        }
        
        Ok(correlations)
    }

    /// Remove highly correlated features
    fn remove_correlated_features(&self, features: &[Vec<f32>], threshold: f32) -> TrainingResult<Vec<usize>> {
        let num_features = if features.is_empty() { 0 } else { features[0].len() };
        let mut selected = Vec::new();
        
        // Compute feature correlation matrix
        let mut correlation_matrix = vec![vec![0.0; num_features]; num_features];
        
        for i in 0..num_features {
            for j in i..num_features {
                if i == j {
                    correlation_matrix[i][j] = 1.0;
                } else {
                    let corr = self.compute_feature_correlation(features, i, j);
                    correlation_matrix[i][j] = corr;
                    correlation_matrix[j][i] = corr;
                }
            }
        }
        
        // Greedy selection to remove highly correlated features
        let mut feature_selected = vec![false; num_features];
        
        for i in 0..num_features {
            if feature_selected[i] {
                continue;
            }
            
            selected.push(i);
            feature_selected[i] = true;
            
            // Mark highly correlated features as excluded
            for j in (i + 1)..num_features {
                if correlation_matrix[i][j].abs() > threshold {
                    feature_selected[j] = true;
                }
            }
        }
        
        Ok(selected)
    }

    /// Compute correlation between two features
    fn compute_feature_correlation(&self, features: &[Vec<f32>], idx1: usize, idx2: usize) -> f32 {
        let values1: Vec<f32> = features.iter().map(|s| s[idx1]).collect();
        let values2: Vec<f32> = features.iter().map(|s| s[idx2]).collect();
        
        let mean1 = values1.iter().sum::<f32>() / values1.len() as f32;
        let mean2 = values2.iter().sum::<f32>() / values2.len() as f32;
        
        let std1 = {
            let var = values1.iter().map(|&v| (v - mean1).powi(2)).sum::<f32>() / values1.len() as f32;
            var.sqrt()
        };
        let std2 = {
            let var = values2.iter().map(|&v| (v - mean2).powi(2)).sum::<f32>() / values2.len() as f32;
            var.sqrt()
        };
        
        if std1 > 0.0 && std2 > 0.0 {
            let covariance = values1.iter()
                .zip(values2.iter())
                .map(|(&v1, &v2)| (v1 - mean1) * (v2 - mean2))
                .sum::<f32>() / values1.len() as f32;
            
            covariance / (std1 * std2)
        } else {
            0.0
        }
    }

    /// Compute target statistics
    fn compute_target_stats(&self, targets: &[f32]) -> TargetStats {
        let valid_targets: Vec<f32> = targets.iter()
            .copied()
            .filter(|t| t.is_finite())
            .collect();
        
        if valid_targets.is_empty() {
            return TargetStats {
                mean: 0.0,
                std: 1.0,
                min: 0.0,
                max: 1.0,
            };
        }
        
        let mean = valid_targets.iter().sum::<f32>() / valid_targets.len() as f32;
        let variance = valid_targets.iter()
            .map(|&t| (t - mean).powi(2))
            .sum::<f32>() / valid_targets.len() as f32;
        let std = variance.sqrt();
        let min = valid_targets.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = valid_targets.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        TargetStats { mean, std, min, max }
    }

    /// Preprocess targets based on configuration
    fn preprocess_targets(&self, targets: &[f32]) -> TrainingResult<Vec<f32>> {
        let stats = self.target_stats.as_ref()
            .ok_or(TrainingError::ConfigError("Target stats not computed".into()))?;
        
        let transformed = match self.config.target_preprocessing {
            TargetPreprocessing::None => targets.to_vec(),
            TargetPreprocessing::Log => {
                targets.iter()
                    .map(|&t| if t > 0.0 { t.ln() } else { 0.0 })
                    .collect()
            }
            TargetPreprocessing::Sqrt => {
                targets.iter()
                    .map(|&t| if t >= 0.0 { t.sqrt() } else { 0.0 })
                    .collect()
            }
            TargetPreprocessing::MinMax => {
                let range = stats.max - stats.min;
                if range > 0.0 {
                    targets.iter()
                        .map(|&t| (t - stats.min) / range)
                        .collect()
                } else {
                    vec![0.5; targets.len()]
                }
            }
            TargetPreprocessing::ZScore => {
                if stats.std > 0.0 {
                    targets.iter()
                        .map(|&t| (t - stats.mean) / stats.std)
                        .collect()
                } else {
                    vec![0.0; targets.len()]
                }
            }
        };
        
        Ok(transformed)
    }

    /// Get configuration
    pub fn config(&self) -> &PreprocessingConfig {
        &self.config
    }

    /// Get selected feature indices
    pub fn selected_features(&self) -> Option<&Vec<usize>> {
        self.selected_features.as_ref()
    }

    /// Get feature statistics
    pub fn feature_stats(&self) -> Option<&FeatureStats> {
        self.feature_stats.as_ref()
    }

    /// Get target statistics
    pub fn target_stats(&self) -> Option<&TargetStats> {
        self.target_stats.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{TargetType, TelecomRecord};

    fn create_test_dataset() -> TelecomDataset {
        let records = vec![
            TelecomRecord {
                timestamp: "2025-06-27 00".to_string(),
                enodeb_code: "1".to_string(),
                enodeb_name: "TEST1".to_string(),
                cell_code: "1".to_string(),
                cell_name: "CELL1".to_string(),
                lte_band: "LTE800".to_string(),
                num_bands: 4,
                cell_availability: 95.0,
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
            },
        ];
        
        TelecomDataset::from_records(records, TargetType::CellAvailability).unwrap()
    }

    #[test]
    fn test_preprocessor_creation() {
        let config = PreprocessingConfig::default();
        let preprocessor = DataPreprocessor::new(config);
        assert!(preprocessor.feature_stats.is_none());
    }

    #[test]
    fn test_missing_value_handling() {
        let features = vec![
            vec![1.0, f32::NAN, 3.0],
            vec![2.0, 4.0, f32::INFINITY],
            vec![3.0, 6.0, 9.0],
        ];
        
        let preprocessor = DataPreprocessor::default();
        let cleaned = preprocessor.handle_missing_values(&features).unwrap();
        
        // All values should be finite
        for sample in &cleaned {
            for &value in sample {
                assert!(value.is_finite());
            }
        }
    }

    #[test]
    fn test_feature_selection() {
        let features = vec![
            vec![1.0, 100.0, 1.1],  // Low variance, high variance, correlated with first
            vec![1.0, 200.0, 1.2],
            vec![1.0, 300.0, 1.3],
        ];
        
        let preprocessor = DataPreprocessor::new(PreprocessingConfig {
            feature_selection: FeatureSelectionMethod::VarianceThreshold(1.0),
            ..Default::default()
        });
        
        let selected = preprocessor.select_features(&features, &[1.0, 2.0, 3.0]).unwrap();
        assert!(selected.is_some());
        
        let indices = selected.unwrap();
        assert!(indices.contains(&1)); // High variance feature should be selected
        assert!(!indices.contains(&0)); // Low variance feature should be removed
    }
}