//! GPU-Optimized Data Loading and Preprocessing
//! 
//! This module provides efficient data loading, preprocessing, and batching
//! for GPU training with memory management and parallel processing.

use std::sync::Arc;
use std::collections::VecDeque;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2, Array3, Axis};
use tokio::sync::mpsc;
use rayon::prelude::*;

#[cfg(feature = "gpu")]
use candle_core::{Device, Tensor, DType};

use crate::data::TelecomDataset;
use crate::gpu_training::MemoryOptimization;

/// Data preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Normalization strategy
    pub normalization: NormalizationType,
    /// Handle missing values
    pub missing_value_strategy: MissingValueStrategy,
    /// Feature selection threshold (correlation)
    pub feature_selection_threshold: f32,
    /// Apply feature scaling
    pub apply_scaling: bool,
    /// Remove outliers (z-score threshold)
    pub outlier_threshold: f32,
    /// Create polynomial features
    pub polynomial_features: bool,
    /// Polynomial degree
    pub polynomial_degree: usize,
    /// Time series window size for LSTM
    pub time_window_size: usize,
    /// Overlap between time windows
    pub time_window_overlap: usize,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            normalization: NormalizationType::StandardScore,
            missing_value_strategy: MissingValueStrategy::Mean,
            feature_selection_threshold: 0.01,
            apply_scaling: true,
            outlier_threshold: 3.0,
            polynomial_features: false,
            polynomial_degree: 2,
            time_window_size: 10,
            time_window_overlap: 5,
        }
    }
}

/// Normalization types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationType {
    /// Z-score normalization (mean=0, std=1)
    StandardScore,
    /// Min-max normalization (0-1 range)
    MinMax,
    /// Robust scaling (median and IQR)
    RobustScaling,
    /// Unit vector scaling
    L2Norm,
    /// No normalization
    None,
}

/// Missing value handling strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    /// Fill with mean
    Mean,
    /// Fill with median
    Median,
    /// Fill with mode
    Mode,
    /// Fill with zero
    Zero,
    /// Forward fill
    ForwardFill,
    /// Backward fill
    BackwardFill,
    /// Linear interpolation
    Interpolation,
    /// Remove samples
    Remove,
}

/// GPU Data Loader for efficient batch processing
pub struct GpuDataLoader {
    /// Preprocessing configuration
    config: PreprocessingConfig,
    /// Batch size
    batch_size: usize,
    /// Number of worker threads
    num_workers: usize,
    /// Memory optimization level
    memory_optimization: MemoryOptimization,
    /// Prefetch factor (number of batches to prefetch)
    prefetch_factor: usize,
    /// Device for tensor operations
    #[cfg(feature = "gpu")]
    device: Option<Device>,
    #[cfg(not(feature = "gpu"))]
    device: Option<String>,
}

impl GpuDataLoader {
    /// Create new GPU data loader
    pub fn new(
        config: PreprocessingConfig,
        batch_size: usize,
        num_workers: usize,
        memory_optimization: MemoryOptimization,
        #[cfg(feature = "gpu")]
    device: Option<Device>,
    #[cfg(not(feature = "gpu"))]
    device: Option<String>,
    ) -> Self {
        let prefetch_factor = match memory_optimization {
            MemoryOptimization::Performance => 4,
            MemoryOptimization::Balanced => 2,
            MemoryOptimization::Conservative => 1,
        };
        
        Self {
            config,
            batch_size,
            num_workers,
            memory_optimization,
            prefetch_factor,
            device,
        }
    }
    
    /// Preprocess telecom dataset
    pub fn preprocess_dataset(&self, mut dataset: TelecomDataset) -> Result<PreprocessedDataset> {
        log::info!("Starting data preprocessing...");
        
        // Handle missing values
        dataset = self.handle_missing_values(dataset)?;
        
        // Remove outliers
        dataset = self.remove_outliers(dataset)?;
        
        // Feature selection
        dataset = self.select_features(dataset)?;
        
        // Normalization
        let (dataset, normalization_params) = self.normalize_features(dataset)?;
        
        // Create polynomial features if enabled
        let dataset = if self.config.polynomial_features {
            self.create_polynomial_features(dataset)?
        } else {
            dataset
        };
        
        log::info!("Preprocessing completed. Final shape: {} x {}", 
            dataset.features.nrows(), dataset.features.ncols());
        
        Ok(PreprocessedDataset {
            dataset,
            normalization_params,
            preprocessing_config: self.config.clone(),
        })
    }
    
    /// Handle missing values in dataset
    fn handle_missing_values(&self, mut dataset: TelecomDataset) -> Result<TelecomDataset> {
        log::info!("Handling missing values with strategy: {:?}", self.config.missing_value_strategy);
        
        // Count missing values
        let mut missing_count = 0;
        let mut feature_means = Vec::new();
        let mut feature_medians = Vec::new();
        
        // Calculate statistics for each feature
        for col_idx in 0..dataset.features.ncols() {
            let column = dataset.features.column(col_idx);
            let values: Vec<f64> = column.iter().filter(|&&x| !x.is_nan()).cloned().collect();
            
            let mean = if !values.is_empty() {
                values.iter().sum::<f64>() / values.len() as f64
            } else {
                0.0
            };
            
            let mut sorted_values = values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if !sorted_values.is_empty() {
                if sorted_values.len() % 2 == 0 {
                    (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
                } else {
                    sorted_values[sorted_values.len() / 2]
                }
            } else {
                0.0
            };
            
            feature_means.push(mean);
            feature_medians.push(median);
            
            // Count missing values in this column
            missing_count += column.iter().filter(|&&x| x.is_nan()).count();
        }
        
        log::info!("Found {} missing values", missing_count);
        
        // Fill missing values
        if missing_count > 0 {
            for ((row_idx, col_idx), value) in dataset.features.indexed_iter_mut() {
                if value.is_nan() {
                    *value = match self.config.missing_value_strategy {
                        MissingValueStrategy::Mean => feature_means[col_idx],
                        MissingValueStrategy::Median => feature_medians[col_idx],
                        MissingValueStrategy::Zero => 0.0,
                        MissingValueStrategy::Mode => feature_means[col_idx], // Simplified to mean
                        MissingValueStrategy::ForwardFill => {
                            // Find last valid value in this column
                            let mut fill_value = 0.0;
                            for prev_row in (0..row_idx).rev() {
                                if !dataset.features[[prev_row, col_idx]].is_nan() {
                                    fill_value = dataset.features[[prev_row, col_idx]];
                                    break;
                                }
                            }
                            fill_value
                        },
                        MissingValueStrategy::BackwardFill => {
                            // Find next valid value in this column
                            let mut fill_value = 0.0;
                            for next_row in (row_idx + 1)..dataset.features.nrows() {
                                if !dataset.features[[next_row, col_idx]].is_nan() {
                                    fill_value = dataset.features[[next_row, col_idx]];
                                    break;
                                }
                            }
                            fill_value
                        },
                        MissingValueStrategy::Interpolation => {
                            // Simple linear interpolation
                            feature_means[col_idx] // Fallback to mean for simplicity
                        },
                        MissingValueStrategy::Remove => {
                            // This would require row removal, handled separately
                            0.0
                        },
                    };
                }
            }
        }
        
        Ok(dataset)
    }
    
    /// Remove outliers using z-score method
    fn remove_outliers(&self, mut dataset: TelecomDataset) -> Result<TelecomDataset> {
        if self.config.outlier_threshold <= 0.0 {
            return Ok(dataset);
        }
        
        log::info!("Removing outliers with z-score threshold: {}", self.config.outlier_threshold);
        
        let mut outlier_mask = vec![false; dataset.features.nrows()];
        
        // Calculate z-scores for each feature
        for col_idx in 0..dataset.features.ncols() {
            let column = dataset.features.column(col_idx);
            let mean = column.mean().unwrap_or(0.0);
            let std = column.std(1.0);
            
            if std > 1e-8 {
                for (row_idx, &value) in column.iter().enumerate() {
                    let z_score = (value - mean).abs() / std;
                    if z_score > self.config.outlier_threshold as f64 {
                        outlier_mask[row_idx] = true;
                    }
                }
            }
        }
        
        let outlier_count = outlier_mask.iter().filter(|&&x| x).count();
        log::info!("Identified {} outlier samples", outlier_count);
        
        if outlier_count > 0 {
            // Remove outlier rows
            let valid_indices: Vec<usize> = outlier_mask.iter()
                .enumerate()
                .filter_map(|(idx, &is_outlier)| if !is_outlier { Some(idx) } else { None })
                .collect();
            
            // Create new arrays without outliers
            let new_features = Array2::from_shape_fn(
                (valid_indices.len(), dataset.features.ncols()),
                |(i, j)| dataset.features[[valid_indices[i], j]]
            );
            
            let new_targets = Array1::from_shape_fn(
                valid_indices.len(),
                |i| dataset.targets[valid_indices[i]]
            );
            
            dataset.features = new_features;
            dataset.targets = new_targets;
            
            log::info!("Removed {} outliers, remaining samples: {}", 
                outlier_count, dataset.features.nrows());
        }
        
        Ok(dataset)
    }
    
    /// Select features based on correlation with target
    fn select_features(&self, mut dataset: TelecomDataset) -> Result<TelecomDataset> {
        if self.config.feature_selection_threshold <= 0.0 {
            return Ok(dataset);
        }
        
        log::info!("Performing feature selection with threshold: {}", 
            self.config.feature_selection_threshold);
        
        let mut correlations = Vec::new();
        let target_mean = dataset.targets.mean().unwrap_or(0.0);
        
        // Calculate correlation of each feature with target
        for col_idx in 0..dataset.features.ncols() {
            let feature_column = dataset.features.column(col_idx);
            let feature_mean = feature_column.mean().unwrap_or(0.0);
            
            let mut numerator = 0.0;
            let mut feature_variance = 0.0;
            let mut target_variance = 0.0;
            
            for (feature_val, target_val) in feature_column.iter().zip(dataset.targets.iter()) {
                let feature_diff = feature_val - feature_mean;
                let target_diff = target_val - target_mean;
                
                numerator += feature_diff * target_diff;
                feature_variance += feature_diff * feature_diff;
                target_variance += target_diff * target_diff;
            }
            
            let correlation = if feature_variance > 1e-8 && target_variance > 1e-8 {
                numerator / (feature_variance.sqrt() * target_variance.sqrt())
            } else {
                0.0
            };
            
            correlations.push((col_idx, correlation.abs()));
        }
        
        // Select features above threshold
        let selected_features: Vec<usize> = correlations.iter()
            .filter(|(_, corr)| *corr >= self.config.feature_selection_threshold as f64)
            .map(|(idx, _)| *idx)
            .collect();
        
        if selected_features.len() < dataset.features.ncols() {
            log::info!("Selected {} features out of {} (removed {})", 
                selected_features.len(), dataset.features.ncols(), 
                dataset.features.ncols() - selected_features.len());
            
            // Create new feature matrix with selected features
            let new_features = Array2::from_shape_fn(
                (dataset.features.nrows(), selected_features.len()),
                |(i, j)| dataset.features[[i, selected_features[j]]]
            );
            
            dataset.features = new_features;
        }
        
        Ok(dataset)
    }
    
    /// Normalize features
    fn normalize_features(&self, mut dataset: TelecomDataset) -> Result<(TelecomDataset, NormalizationParams)> {
        log::info!("Normalizing features with method: {:?}", self.config.normalization);
        
        let mut normalization_params = NormalizationParams {
            normalization_type: self.config.normalization,
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
            feature_mins: Vec::new(),
            feature_maxs: Vec::new(),
            feature_medians: Vec::new(),
            feature_iqrs: Vec::new(),
        };
        
        match self.config.normalization {
            NormalizationType::StandardScore => {
                for col_idx in 0..dataset.features.ncols() {
                    let mut column = dataset.features.column_mut(col_idx);
                    let mean = column.mean().unwrap_or(0.0);
                    let std = column.std(1.0);
                    
                    normalization_params.feature_means.push(mean);
                    normalization_params.feature_stds.push(std);
                    
                    if std > 1e-8 {
                        column.mapv_inplace(|x| (x - mean) / std);
                    }
                }
            },
            NormalizationType::MinMax => {
                for col_idx in 0..dataset.features.ncols() {
                    let mut column = dataset.features.column_mut(col_idx);
                    let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    
                    normalization_params.feature_mins.push(min_val);
                    normalization_params.feature_maxs.push(max_val);
                    
                    let range = max_val - min_val;
                    if range > 1e-8 {
                        column.mapv_inplace(|x| (x - min_val) / range);
                    }
                }
            },
            NormalizationType::RobustScaling => {
                for col_idx in 0..dataset.features.ncols() {
                    let mut column = dataset.features.column_mut(col_idx);
                    let mut values: Vec<f64> = column.iter().cloned().collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    
                    let median = if values.len() % 2 == 0 {
                        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };
                    
                    let q1 = values[values.len() / 4];
                    let q3 = values[3 * values.len() / 4];
                    let iqr = q3 - q1;
                    
                    normalization_params.feature_medians.push(median);
                    normalization_params.feature_iqrs.push(iqr);
                    
                    if iqr > 1e-8 {
                        column.mapv_inplace(|x| (x - median) / iqr);
                    }
                }
            },
            NormalizationType::L2Norm => {
                for col_idx in 0..dataset.features.ncols() {
                    let mut column = dataset.features.column_mut(col_idx);
                    let norm = column.iter().map(|&x| x * x).sum::<f64>().sqrt();
                    
                    if norm > 1e-8 {
                        column.mapv_inplace(|x| x / norm);
                    }
                }
            },
            NormalizationType::None => {
                // No normalization
            },
        }
        
        Ok((dataset, normalization_params))
    }
    
    /// Create polynomial features
    fn create_polynomial_features(&self, mut dataset: TelecomDataset) -> Result<TelecomDataset> {
        if self.config.polynomial_degree <= 1 {
            return Ok(dataset);
        }
        
        log::info!("Creating polynomial features up to degree {}", self.config.polynomial_degree);
        
        let original_features = dataset.features.clone();
        let original_cols = original_features.ncols();
        
        // Calculate new feature count (approximate for quadratic)
        let new_feature_count = if self.config.polynomial_degree == 2 {
            original_cols + (original_cols * (original_cols + 1)) / 2
        } else {
            original_cols * self.config.polynomial_degree // Simplified
        };
        
        let mut new_features = Array2::zeros((dataset.features.nrows(), new_feature_count));
        
        // Copy original features
        for i in 0..dataset.features.nrows() {
            for j in 0..original_cols {
                new_features[[i, j]] = original_features[[i, j]];
            }
        }
        
        let mut feature_idx = original_cols;
        
        // Add polynomial features
        if self.config.polynomial_degree >= 2 {
            // Add quadratic terms
            for i in 0..original_cols {
                for j in i..original_cols {
                    if feature_idx < new_feature_count {
                        for row in 0..dataset.features.nrows() {
                            new_features[[row, feature_idx]] = 
                                original_features[[row, i]] * original_features[[row, j]];
                        }
                        feature_idx += 1;
                    }
                }
            }
        }
        
        dataset.features = new_features.slice(ndarray::s![.., ..feature_idx]).to_owned();
        
        log::info!("Created polynomial features: {} -> {} features", 
            original_cols, dataset.features.ncols());
        
        Ok(dataset)
    }
    
    /// Create time series windows for LSTM training
    pub fn create_time_series_windows(&self, dataset: &TelecomDataset) -> Result<TimeSeriesDataset> {
        log::info!("Creating time series windows with size {} and overlap {}", 
            self.config.time_window_size, self.config.time_window_overlap);
        
        let step_size = self.config.time_window_size - self.config.time_window_overlap;
        let num_windows = (dataset.features.nrows() - self.config.time_window_size) / step_size + 1;
        
        let window_features = Array3::zeros((
            num_windows,
            self.config.time_window_size,
            dataset.features.ncols()
        ));
        
        let window_targets = Array1::zeros(num_windows);
        
        for window_idx in 0..num_windows {
            let start_idx = window_idx * step_size;
            let end_idx = start_idx + self.config.time_window_size;
            
            // Extract window features
            for t in 0..self.config.time_window_size {
                for f in 0..dataset.features.ncols() {
                    window_features[[window_idx, t, f]] = dataset.features[[start_idx + t, f]];
                }
            }
            
            // Use the last target in the window as the label
            window_targets[window_idx] = dataset.targets[end_idx - 1];
        }
        
        log::info!("Created {} time series windows", num_windows);
        
        Ok(TimeSeriesDataset {
            features: window_features,
            targets: window_targets,
            window_size: self.config.time_window_size,
            overlap: self.config.time_window_overlap,
        })
    }
    
    /// Convert data to GPU tensors
    #[cfg(feature = "gpu")]
    pub fn to_gpu_tensors(&self, features: &Array2<f64>, targets: &Array1<f64>) -> Result<(Tensor, Tensor)> {
        let device = self.device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU device not available"))?;
        
        // Convert to f32 for GPU efficiency
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let targets_f32: Vec<f32> = targets.iter().map(|&x| x as f32).collect();
        
        let features_tensor = Tensor::from_slice(
            &features_f32,
            (features.nrows(), features.ncols()),
            device
        )?;
        
        let targets_tensor = Tensor::from_slice(
            &targets_f32,
            (targets.len(),),
            device
        )?;
        
        Ok((features_tensor, targets_tensor))
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn to_gpu_tensors(&self, _features: &Array2<f64>, _targets: &Array1<f64>) -> Result<((), ())> {
        Err(anyhow::anyhow!("GPU features not enabled"))
    }
    
    /// Create data batches for training
    pub async fn create_batches(&self, dataset: &PreprocessedDataset) -> Result<Vec<DataBatch>> {
        let num_samples = dataset.dataset.features.nrows();
        let num_batches = (num_samples + self.batch_size - 1) / self.batch_size;
        
        log::info!("Creating {} batches of size {} for {} samples", 
            num_batches, self.batch_size, num_samples);
        
        let mut batches = Vec::new();
        
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * self.batch_size;
            let end_idx = std::cmp::min(start_idx + self.batch_size, num_samples);
            let actual_batch_size = end_idx - start_idx;
            
            // Extract batch data
            let batch_features = dataset.dataset.features
                .slice(ndarray::s![start_idx..end_idx, ..])
                .to_owned();
            
            let batch_targets = dataset.dataset.targets
                .slice(ndarray::s![start_idx..end_idx])
                .to_owned();
            
            let batch = DataBatch {
                features: batch_features,
                targets: batch_targets,
                batch_size: actual_batch_size,
                batch_idx,
            };
            
            batches.push(batch);
        }
        
        Ok(batches)
    }
}

/// Preprocessed dataset with normalization parameters
#[derive(Debug, Clone)]
pub struct PreprocessedDataset {
    pub dataset: TelecomDataset,
    pub normalization_params: NormalizationParams,
    pub preprocessing_config: PreprocessingConfig,
}

/// Normalization parameters for inverse transform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub normalization_type: NormalizationType,
    pub feature_means: Vec<f64>,
    pub feature_stds: Vec<f64>,
    pub feature_mins: Vec<f64>,
    pub feature_maxs: Vec<f64>,
    pub feature_medians: Vec<f64>,
    pub feature_iqrs: Vec<f64>,
}

/// Time series dataset for LSTM training
#[derive(Debug, Clone)]
pub struct TimeSeriesDataset {
    pub features: Array3<f64>, // (samples, time_steps, features)
    pub targets: Array1<f64>,
    pub window_size: usize,
    pub overlap: usize,
}

/// Data batch for GPU training
#[derive(Debug, Clone)]
pub struct DataBatch {
    pub features: Array2<f64>,
    pub targets: Array1<f64>,
    pub batch_size: usize,
    pub batch_idx: usize,
}

impl DataBatch {
    /// Convert batch to GPU tensors
    #[cfg(feature = "gpu")]
    pub fn to_gpu_tensors(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let features_f32: Vec<f32> = self.features.iter().map(|&x| x as f32).collect();
        let targets_f32: Vec<f32> = self.targets.iter().map(|&x| x as f32).collect();
        
        let features_tensor = Tensor::from_slice(
            &features_f32,
            (self.features.nrows(), self.features.ncols()),
            device
        )?;
        
        let targets_tensor = Tensor::from_slice(
            &targets_f32,
            (self.targets.len(),),
            device
        )?;
        
        Ok((features_tensor, targets_tensor))
    }
}

/// Data loading statistics
#[derive(Debug, Clone, Serialize)]
pub struct DataLoadingStats {
    pub original_samples: usize,
    pub final_samples: usize,
    pub original_features: usize,
    pub final_features: usize,
    pub missing_values_filled: usize,
    pub outliers_removed: usize,
    pub features_selected: usize,
    pub preprocessing_time: std::time::Duration,
}