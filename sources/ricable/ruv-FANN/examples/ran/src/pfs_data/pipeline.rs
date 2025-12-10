//! Data processing pipeline for neural network feature extraction
//! 
//! Implements streaming data normalization and feature extraction pipelines

use arrow::array::{Array, ArrayRef, Float32Array, Int64Array};
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Feature extraction pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Normalization method
    pub normalization: NormalizationMethod,
    /// Feature selection methods
    pub feature_selection: Vec<FeatureSelectionMethod>,
    /// Window size for streaming features
    pub window_size: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable outlier detection
    pub outlier_detection: bool,
}

/// Normalization methods
#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// Min-Max normalization (range 0-1)
    MinMax,
    /// Robust scaling using median and IQR
    Robust,
    /// No normalization
    None,
}

/// Feature selection methods
#[derive(Debug, Clone)]
pub enum FeatureSelectionMethod {
    /// Select top K features by variance
    VarianceThreshold(f64),
    /// Select features with correlation above threshold
    CorrelationThreshold(f64),
    /// Select top K features
    TopK(usize),
    /// Custom feature list
    Custom(Vec<String>),
}

/// Streaming feature extractor
pub struct FeatureExtractor {
    config: PipelineConfig,
    /// Running statistics for normalization
    statistics: HashMap<String, ColumnStats>,
    /// Feature buffer for windowed operations
    feature_buffer: Vec<HashMap<String, f64>>,
    /// Selected features
    selected_features: Vec<String>,
}

/// Column statistics for normalization
#[derive(Debug, Clone)]
struct ColumnStats {
    mean: f64,
    variance: f64,
    min: f64,
    max: f64,
    count: u64,
    median: f64,
    q1: f64,
    q3: f64,
}

impl ColumnStats {
    fn new() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            count: 0,
            median: 0.0,
            q1: 0.0,
            q3: 0.0,
        }
    }

    /// Update statistics with new value using online algorithm
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.variance += delta * delta2;
        
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn std_dev(&self) -> f64 {
        if self.count > 1 {
            (self.variance / (self.count - 1) as f64).sqrt()
        } else {
            0.0
        }
    }
}

impl FeatureExtractor {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            statistics: HashMap::new(),
            feature_buffer: Vec::new(),
            selected_features: Vec::new(),
        }
    }

    /// Extract features from a record batch
    pub fn extract_features(&mut self, batch: &RecordBatch) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let mut features = Vec::new();
        
        // Convert Arrow batch to feature vectors
        let row_count = batch.num_rows();
        
        for row_idx in 0..row_count {
            let mut row_features = HashMap::new();
            
            // Extract values from each column
            for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                let column = batch.column(col_idx);
                let field_name = field.name();
                
                if let Some(value) = self.extract_value_from_column(column, row_idx) {
                    row_features.insert(field_name.clone(), value);
                    
                    // Update statistics
                    let stats = self.statistics.entry(field_name.clone())
                        .or_insert_with(ColumnStats::new);
                    stats.update(value);
                }
            }
            
            // Add to feature buffer
            self.feature_buffer.push(row_features.clone());
            
            // Keep buffer size within limits
            if self.feature_buffer.len() > self.config.window_size {
                self.feature_buffer.remove(0);
            }
            
            // Generate features for this row
            let normalized_features = self.normalize_features(&row_features);
            let windowed_features = self.extract_windowed_features();
            
            // Combine features
            let mut combined_features = Vec::new();
            combined_features.extend(normalized_features);
            combined_features.extend(windowed_features);
            
            features.push(combined_features);
        }
        
        Ok(features)
    }

    /// Extract value from Arrow column
    fn extract_value_from_column(&self, column: &ArrayRef, row_idx: usize) -> Option<f64> {
        match column.data_type() {
            arrow::datatypes::DataType::Float32 => {
                let array = column.as_any().downcast_ref::<Float32Array>()?;
                Some(array.value(row_idx) as f64)
            }
            arrow::datatypes::DataType::Int64 => {
                let array = column.as_any().downcast_ref::<Int64Array>()?;
                Some(array.value(row_idx) as f64)
            }
            _ => None,
        }
    }

    /// Normalize features based on configuration
    fn normalize_features(&self, features: &HashMap<String, f64>) -> Vec<f32> {
        let mut normalized = Vec::new();
        
        for (feature_name, &value) in features {
            if let Some(stats) = self.statistics.get(feature_name) {
                let normalized_value = match self.config.normalization {
                    NormalizationMethod::ZScore => {
                        let std_dev = stats.std_dev();
                        if std_dev > 0.0 {
                            (value - stats.mean) / std_dev
                        } else {
                            0.0
                        }
                    }
                    NormalizationMethod::MinMax => {
                        let range = stats.max - stats.min;
                        if range > 0.0 {
                            (value - stats.min) / range
                        } else {
                            0.0
                        }
                    }
                    NormalizationMethod::Robust => {
                        let iqr = stats.q3 - stats.q1;
                        if iqr > 0.0 {
                            (value - stats.median) / iqr
                        } else {
                            0.0
                        }
                    }
                    NormalizationMethod::None => value,
                };
                
                normalized.push(normalized_value as f32);
            }
        }
        
        normalized
    }

    /// Extract windowed features (moving averages, trends, etc.)
    fn extract_windowed_features(&self) -> Vec<f32> {
        let mut windowed = Vec::new();
        
        if self.feature_buffer.len() < 2 {
            return windowed;
        }
        
        // Calculate moving averages for each feature
        for feature_name in self.get_numeric_features() {
            let values: Vec<f64> = self.feature_buffer.iter()
                .filter_map(|row| row.get(&feature_name))
                .cloned()
                .collect();
            
            if values.len() >= 2 {
                // Moving average
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                windowed.push(avg as f32);
                
                // Trend (slope of linear regression)
                let trend = self.calculate_trend(&values);
                windowed.push(trend as f32);
                
                // Volatility (standard deviation)
                let volatility = self.calculate_volatility(&values);
                windowed.push(volatility as f32);
            }
        }
        
        windowed
    }

    /// Get list of numeric features
    fn get_numeric_features(&self) -> Vec<String> {
        self.statistics.keys().cloned().collect()
    }

    /// Calculate trend using simple linear regression
    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let n = values.len() as f64;
        let sum_x = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = values.iter().enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let sum_x2 = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();
        
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator != 0.0 {
            (n * sum_xy - sum_x * sum_y) / denominator
        } else {
            0.0
        }
    }

    /// Calculate volatility (standard deviation)
    fn calculate_volatility(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }

    /// Process multiple batches in parallel
    pub fn process_batches(&mut self, batches: Vec<RecordBatch>) -> Vec<Vec<Vec<f32>>> {
        batches.into_par_iter()
            .map(|batch| {
                let mut local_extractor = self.clone();
                local_extractor.extract_features(&batch).unwrap_or_default()
            })
            .collect()
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers(&self, values: &[f64]) -> Vec<bool> {
        if values.len() < 4 {
            return vec![false; values.len()];
        }
        
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let q1_idx = sorted_values.len() / 4;
        let q3_idx = 3 * sorted_values.len() / 4;
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        values.iter()
            .map(|&v| v < lower_bound || v > upper_bound)
            .collect()
    }
}

impl Clone for FeatureExtractor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            statistics: self.statistics.clone(),
            feature_buffer: self.feature_buffer.clone(),
            selected_features: self.selected_features.clone(),
        }
    }
}

/// Data pipeline for streaming processing
pub struct StreamingPipeline {
    extractor: FeatureExtractor,
    batch_buffer: Vec<RecordBatch>,
    output_buffer: Vec<Vec<f32>>,
}

impl StreamingPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            extractor: FeatureExtractor::new(config),
            batch_buffer: Vec::new(),
            output_buffer: Vec::new(),
        }
    }

    /// Add a batch to the pipeline
    pub fn add_batch(&mut self, batch: RecordBatch) -> Result<(), Box<dyn std::error::Error>> {
        self.batch_buffer.push(batch);
        
        // Process when buffer is full
        if self.batch_buffer.len() >= self.extractor.config.batch_size {
            self.process_buffer()?;
        }
        
        Ok(())
    }

    /// Process accumulated batches
    fn process_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let batches = std::mem::take(&mut self.batch_buffer);
        
        for batch in batches {
            let features = self.extractor.extract_features(&batch)?;
            self.output_buffer.extend(features);
        }
        
        Ok(())
    }

    /// Get processed features
    pub fn get_features(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.output_buffer)
    }

    /// Flush remaining batches
    pub fn flush(&mut self) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if !self.batch_buffer.is_empty() {
            self.process_buffer()?;
        }
        Ok(self.get_features())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};

    #[test]
    fn test_feature_extractor() {
        let config = PipelineConfig {
            normalization: NormalizationMethod::ZScore,
            feature_selection: vec![],
            window_size: 10,
            batch_size: 5,
            outlier_detection: false,
        };
        
        let mut extractor = FeatureExtractor::new(config);
        
        // Create test batch
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float32, false),
        ]));
        let array = Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0]));
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();
        
        let features = extractor.extract_features(&batch).unwrap();
        assert_eq!(features.len(), 3);
    }

    #[test]
    fn test_streaming_pipeline() {
        let config = PipelineConfig {
            normalization: NormalizationMethod::MinMax,
            feature_selection: vec![],
            window_size: 5,
            batch_size: 2,
            outlier_detection: true,
        };
        
        let mut pipeline = StreamingPipeline::new(config);
        
        // Create test batch
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float32, false),
        ]));
        let array = Arc::new(Float32Array::from(vec![1.0, 2.0]));
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();
        
        pipeline.add_batch(batch).unwrap();
        let features = pipeline.get_features();
        assert!(!features.is_empty());
    }
}