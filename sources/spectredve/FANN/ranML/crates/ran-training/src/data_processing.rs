use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use anyhow::{Result, Context};
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use log::{info, debug, warn};

#[derive(Debug, Clone)]
pub struct TelecomDataset {
    pub features: Array2<f64>,
    pub targets: Array2<f64>,
    pub feature_names: Vec<String>,
    pub target_names: Vec<String>,
    pub metadata: DatasetMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub sample_count: usize,
    pub feature_count: usize,
    pub target_count: usize,
    pub data_quality_score: f64,
    pub preprocessing_steps: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TrainingData {
    pub features: Array2<f64>,
    pub targets: Array2<f64>,
    pub indices: Vec<usize>,
}

impl TrainingData {
    pub fn feature_count(&self) -> usize {
        self.features.ncols()
    }
    
    pub fn target_count(&self) -> usize {
        self.targets.ncols()
    }
    
    pub fn sample_count(&self) -> usize {
        self.features.nrows()
    }
}

pub struct DataPreprocessor {
    pub missing_value_strategy: MissingValueStrategy,
    pub scaling_strategy: ScalingStrategy,
    pub feature_selection_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum MissingValueStrategy {
    Mean,
    Median,
    Zero,
    Remove,
}

#[derive(Debug, Clone)]
pub enum ScalingStrategy {
    StandardScale,
    MinMaxScale,
    RobustScale,
    None,
}

impl DataPreprocessor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            missing_value_strategy: MissingValueStrategy::Mean,
            scaling_strategy: ScalingStrategy::StandardScale,
            feature_selection_threshold: 0.95,
        })
    }
    
    pub fn load_csv_data(&self, path: &Path) -> Result<TelecomDataset> {
        info!("🔄 Loading CSV data from: {:?}", path);
        
        let file = File::open(path)
            .with_context(|| format!("Failed to open CSV file: {:?}", path))?;
        let reader = BufReader::new(file);
        
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(b';')
            .has_headers(true)
            .from_reader(reader);
        
        // Get headers
        let headers = csv_reader.headers()?.clone();
        let feature_names: Vec<String> = headers.iter()
            .map(|h| h.to_string())
            .collect();
        
        info!("📊 Found {} columns in CSV", feature_names.len());
        
        // Read all records
        let mut records = Vec::new();
        for result in csv_reader.records() {
            let record = result?;
            let row: Vec<f64> = record.iter()
                .map(|field| {
                    field.parse::<f64>()
                        .unwrap_or(0.0) // Default to 0 for parsing errors
                })
                .collect();
            records.push(row);
        }
        
        info!("✅ Loaded {} records from CSV", records.len());
        
        if records.is_empty() {
            return Err(anyhow::anyhow!("No valid records found in CSV file"));
        }
        
        // Convert to ndarray
        let num_features = records[0].len();
        let num_samples = records.len();
        
        let flat_data: Vec<f64> = records.into_iter().flatten().collect();
        let features_array = Array2::from_shape_vec((num_samples, num_features), flat_data)?;
        
        // Split features and targets (last 8 columns as targets for telecom metrics)
        let target_count = 8.min(num_features);
        let feature_count = num_features - target_count;
        
        let features = features_array.slice(s![.., ..feature_count]).to_owned();
        let targets = features_array.slice(s![.., feature_count..]).to_owned();
        
        let target_names = feature_names[feature_count..].to_vec();
        let feature_names_only = feature_names[..feature_count].to_vec();
        
        // Create metadata
        let metadata = DatasetMetadata {
            sample_count: num_samples,
            feature_count,
            target_count,
            data_quality_score: self.calculate_data_quality(&features),
            preprocessing_steps: vec!["csv_load".to_string()],
        };
        
        info!("📈 Dataset created: {} samples, {} features, {} targets", 
              num_samples, feature_count, target_count);
        
        Ok(TelecomDataset {
            features,
            targets,
            feature_names: feature_names_only,
            target_names,
            metadata,
        })
    }
    
    fn calculate_data_quality(&self, features: &Array2<f64>) -> f64 {
        let total_elements = features.len() as f64;
        let valid_elements = features.iter()
            .filter(|&&x| !x.is_nan() && !x.is_infinite())
            .count() as f64;
        
        valid_elements / total_elements
    }
    
    pub fn preprocess_dataset(&self, dataset: &mut TelecomDataset) -> Result<()> {
        info!("🔧 Starting dataset preprocessing...");
        
        // Handle missing values
        self.handle_missing_values(&mut dataset.features)?;
        
        // Scale features
        self.scale_features(&mut dataset.features)?;
        
        // Update metadata
        dataset.metadata.preprocessing_steps.push("missing_values_handled".to_string());
        dataset.metadata.preprocessing_steps.push("features_scaled".to_string());
        dataset.metadata.data_quality_score = self.calculate_data_quality(&dataset.features);
        
        info!("✅ Preprocessing complete. Data quality: {:.2}%", 
              dataset.metadata.data_quality_score * 100.0);
        
        Ok(())
    }
    
    fn handle_missing_values(&self, features: &mut Array2<f64>) -> Result<()> {
        match self.missing_value_strategy {
            MissingValueStrategy::Mean => {
                for col in 0..features.ncols() {
                    let column = features.column(col);
                    let mean = column.iter()
                        .filter(|&&x| !x.is_nan())
                        .sum::<f64>() / column.len() as f64;
                    
                    for row in 0..features.nrows() {
                        if features[[row, col]].is_nan() {
                            features[[row, col]] = mean;
                        }
                    }
                }
            }
            MissingValueStrategy::Zero => {
                for elem in features.iter_mut() {
                    if elem.is_nan() {
                        *elem = 0.0;
                    }
                }
            }
            MissingValueStrategy::Median => {
                for col in 0..features.ncols() {
                    let mut column_values: Vec<f64> = features.column(col)
                        .iter()
                        .filter(|&&x| !x.is_nan())
                        .copied()
                        .collect();
                    
                    column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = if column_values.len() % 2 == 0 {
                        (column_values[column_values.len() / 2 - 1] + column_values[column_values.len() / 2]) / 2.0
                    } else {
                        column_values[column_values.len() / 2]
                    };
                    
                    for row in 0..features.nrows() {
                        if features[[row, col]].is_nan() {
                            features[[row, col]] = median;
                        }
                    }
                }
            }
            MissingValueStrategy::Remove => {
                warn!("Remove strategy not implemented in-place. Using mean instead.");
                self.handle_missing_values(features)?;
            }
        }
        
        Ok(())
    }
    
    fn scale_features(&self, features: &mut Array2<f64>) -> Result<()> {
        match self.scaling_strategy {
            ScalingStrategy::StandardScale => {
                for col in 0..features.ncols() {
                    let column = features.column(col);
                    let mean = column.mean().unwrap_or(0.0);
                    let std = column.std(1.0);
                    
                    if std > 1e-10 {
                        for row in 0..features.nrows() {
                            features[[row, col]] = (features[[row, col]] - mean) / std;
                        }
                    }
                }
            }
            ScalingStrategy::MinMaxScale => {
                for col in 0..features.ncols() {
                    let column = features.column(col);
                    let min_val = column.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_val = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    
                    if (max_val - min_val) > 1e-10 {
                        for row in 0..features.nrows() {
                            features[[row, col]] = (features[[row, col]] - min_val) / (max_val - min_val);
                        }
                    }
                }
            }
            ScalingStrategy::RobustScale => {
                for col in 0..features.ncols() {
                    let mut column_values: Vec<f64> = features.column(col).to_vec();
                    column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    
                    let q1 = column_values[column_values.len() / 4];
                    let q3 = column_values[3 * column_values.len() / 4];
                    let iqr = q3 - q1;
                    let median = column_values[column_values.len() / 2];
                    
                    if iqr > 1e-10 {
                        for row in 0..features.nrows() {
                            features[[row, col]] = (features[[row, col]] - median) / iqr;
                        }
                    }
                }
            }
            ScalingStrategy::None => {
                // No scaling
            }
        }
        
        Ok(())
    }
}

impl TelecomDataset {
    pub fn len(&self) -> usize {
        self.features.nrows()
    }
    
    pub fn feature_count(&self) -> usize {
        self.features.ncols()
    }
    
    pub fn target_count(&self) -> usize {
        self.targets.ncols()
    }
    
    pub fn split_data(&self, train_ratio: f64, val_ratio: f64, test_ratio: f64) -> Result<(TrainingData, TrainingData, TrainingData)> {
        let total_samples = self.len();
        let train_size = (total_samples as f64 * train_ratio) as usize;
        let val_size = (total_samples as f64 * val_ratio) as usize;
        let test_size = total_samples - train_size - val_size;
        
        info!("📊 Splitting data: train={}, val={}, test={}", train_size, val_size, test_size);
        
        // Create indices
        let mut indices: Vec<usize> = (0..total_samples).collect();
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        
        let train_indices = indices[..train_size].to_vec();
        let val_indices = indices[train_size..train_size + val_size].to_vec();
        let test_indices = indices[train_size + val_size..].to_vec();
        
        // Create training data
        let train_data = TrainingData {
            features: self.select_rows(&self.features, &train_indices)?,
            targets: self.select_rows(&self.targets, &train_indices)?,
            indices: train_indices,
        };
        
        let val_data = TrainingData {
            features: self.select_rows(&self.features, &val_indices)?,
            targets: self.select_rows(&self.targets, &val_indices)?,
            indices: val_indices,
        };
        
        let test_data = TrainingData {
            features: self.select_rows(&self.features, &test_indices)?,
            targets: self.select_rows(&self.targets, &test_indices)?,
            indices: test_indices,
        };
        
        Ok((train_data, val_data, test_data))
    }
    
    fn select_rows(&self, array: &Array2<f64>, indices: &[usize]) -> Result<Array2<f64>> {
        let mut selected = Array2::zeros((indices.len(), array.ncols()));
        
        for (i, &idx) in indices.iter().enumerate() {
            selected.row_mut(i).assign(&array.row(idx));
        }
        
        Ok(selected)
    }
}

// Import ndarray slice syntax
use ndarray::s;