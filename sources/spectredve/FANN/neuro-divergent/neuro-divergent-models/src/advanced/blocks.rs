//! Shared building blocks for advanced neural forecasting models
//! 
//! This module provides common architectural components used across different
//! advanced models like NBEATS, NHITS, and TiDE including basis functions,
//! multi-layer perceptrons, and specialized layers.

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction};

/// Errors that can occur during model operations
#[derive(Debug, Clone)]
pub enum ModelError {
    InvalidInput(String),
    NetworkError(String),
    ConfigurationError(String),
    DimensionMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ModelError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ModelError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            ModelError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for ModelError {}

/// Base model trait for all neural forecasting models
pub trait BaseModel<T: Float>: Send + Sync {
    /// Model configuration type
    type Config: Clone + Send + Sync;
    
    /// Get the model name
    fn name(&self) -> &'static str;
    
    /// Get the forecast horizon
    fn horizon(&self) -> usize;
    
    /// Get the input window size
    fn input_size(&self) -> usize;
    
    /// Fit the model to training data
    fn fit(&mut self, data: &TimeSeriesData<T>) -> Result<(), ModelError>;
    
    /// Generate forecasts
    fn predict(&self, data: &TimeSeriesData<T>) -> Result<Vec<T>, ModelError>;
    
    /// Generate probabilistic forecasts with prediction intervals
    fn predict_quantiles(&self, data: &TimeSeriesData<T>, quantiles: &[T]) -> Result<Vec<Vec<T>>, ModelError>;
    
    /// Get model parameters count
    fn parameters_count(&self) -> usize;
    
    /// Check if model is fitted
    fn is_fitted(&self) -> bool;
}

/// Time series data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData<T: Float> {
    pub series: Vec<T>,
    pub exogenous: Option<Vec<Vec<T>>>,
    pub static_features: Option<Vec<T>>,
    pub timestamps: Option<Vec<i64>>,
}

impl<T: Float> TimeSeriesData<T> {
    pub fn new(series: Vec<T>) -> Self {
        Self {
            series,
            exogenous: None,
            static_features: None,
            timestamps: None,
        }
    }
    
    pub fn with_exogenous(mut self, exogenous: Vec<Vec<T>>) -> Self {
        self.exogenous = Some(exogenous);
        self
    }
    
    pub fn with_static_features(mut self, features: Vec<T>) -> Self {
        self.static_features = Some(features);
        self
    }
    
    pub fn len(&self) -> usize {
        self.series.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }
}

/// Basis function types for interpretable models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BasisFunction<T: Float> {
    /// Generic basis (identity)
    Generic,
    /// Polynomial basis for trend modeling
    Polynomial { degree: usize },
    /// Fourier basis for seasonality modeling
    Fourier { harmonics: usize },
    /// Custom basis function
    Custom { 
        name: String,
        generator: fn(usize, usize) -> Vec<Vec<T>>
    },
}

impl<T: Float> BasisFunction<T> {
    /// Generate basis matrix for given input and output sizes
    pub fn generate_basis(&self, input_size: usize, output_size: usize) -> Vec<Vec<T>> {
        match self {
            BasisFunction::Generic => {
                // Identity matrix for generic basis
                let mut basis = vec![vec![T::zero(); output_size]; input_size];
                let min_size = input_size.min(output_size);
                for i in 0..min_size {
                    basis[i][i] = T::one();
                }
                basis
            },
            BasisFunction::Polynomial { degree } => {
                self.generate_polynomial_basis(input_size, output_size, *degree)
            },
            BasisFunction::Fourier { harmonics } => {
                self.generate_fourier_basis(input_size, output_size, *harmonics)
            },
            BasisFunction::Custom { generator, .. } => {
                generator(input_size, output_size)
            }
        }
    }
    
    fn generate_polynomial_basis(&self, input_size: usize, output_size: usize, degree: usize) -> Vec<Vec<T>> {
        let mut basis = vec![vec![T::zero(); output_size]; input_size];
        
        for i in 0..input_size {
            let t = T::from(i).unwrap() / T::from(input_size - 1).unwrap();
            for j in 0..output_size.min(degree + 1) {
                let power = T::from(j).unwrap();
                basis[i][j] = t.powf(power);
            }
        }
        
        basis
    }
    
    fn generate_fourier_basis(&self, input_size: usize, output_size: usize, harmonics: usize) -> Vec<Vec<T>> {
        let mut basis = vec![vec![T::zero(); output_size]; input_size];
        let pi = T::from(std::f64::consts::PI).unwrap();
        
        for i in 0..input_size {
            let t = T::from(i).unwrap() / T::from(input_size).unwrap();
            let mut col_idx = 0;
            
            // DC component
            if col_idx < output_size {
                basis[i][col_idx] = T::one();
                col_idx += 1;
            }
            
            // Harmonic components
            for h in 1..=harmonics {
                if col_idx < output_size {
                    let freq = T::from(h).unwrap();
                    basis[i][col_idx] = (freq * pi * t * T::from(2.0).unwrap()).cos();
                    col_idx += 1;
                }
                
                if col_idx < output_size {
                    let freq = T::from(h).unwrap();
                    basis[i][col_idx] = (freq * pi * t * T::from(2.0).unwrap()).sin();
                    col_idx += 1;
                }
            }
        }
        
        basis
    }
}

/// Stack types for NBEATS-like models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StackType {
    Generic,
    Trend,
    Seasonality,
}

/// Multi-layer perceptron block with flexible configuration
#[derive(Debug, Clone)]
pub struct MLPBlock<T: Float> {
    pub networks: Vec<Network<T>>,
    pub layer_sizes: Vec<usize>,
    pub activation: ActivationFunction,
    pub dropout_rate: Option<T>,
}

impl<T: Float> MLPBlock<T> {
    /// Create a new MLP block
    pub fn new(input_size: usize, layer_sizes: Vec<usize>, activation: ActivationFunction) -> Result<Self, ModelError> {
        if layer_sizes.is_empty() {
            return Err(ModelError::ConfigurationError("Layer sizes cannot be empty".to_string()));
        }
        
        let mut networks = Vec::new();
        let mut current_input_size = input_size;
        
        for &layer_size in &layer_sizes {
            let network = NetworkBuilder::new()
                .input_layer(current_input_size)
                .output_layer_with_activation(layer_size, activation, T::one())
                .build();
            
            networks.push(network);
            current_input_size = layer_size;
        }
        
        Ok(Self {
            networks,
            layer_sizes: layer_sizes.clone(),
            activation,
            dropout_rate: None,
        })
    }
    
    /// Forward pass through the MLP block
    pub fn forward(&mut self, input: &[T]) -> Result<Vec<T>, ModelError> {
        if input.len() != self.input_size() {
            return Err(ModelError::DimensionMismatch {
                expected: self.input_size(),
                actual: input.len(),
            });
        }
        
        let mut current_input = input.to_vec();
        
        for network in &mut self.networks {
            current_input = network.run(&current_input);
        }
        
        Ok(current_input)
    }
    
    /// Get input size
    pub fn input_size(&self) -> usize {
        self.networks.first()
            .map(|n| n.num_inputs())
            .unwrap_or(0)
    }
    
    /// Get output size
    pub fn output_size(&self) -> usize {
        self.layer_sizes.last().copied().unwrap_or(0)
    }
    
    /// Set dropout rate
    pub fn with_dropout(mut self, dropout_rate: T) -> Self {
        self.dropout_rate = Some(dropout_rate);
        self
    }
}

/// Residual connection wrapper
#[derive(Debug, Clone)]
pub struct ResidualBlock<T: Float> {
    pub main_path: MLPBlock<T>,
    pub skip_connection: bool,
    pub input_size: usize,
    pub output_size: usize,
}

impl<T: Float> ResidualBlock<T> {
    /// Create a new residual block
    pub fn new(main_path: MLPBlock<T>, skip_connection: bool) -> Self {
        let input_size = main_path.input_size();
        let output_size = main_path.output_size();
        
        Self {
            main_path,
            skip_connection,
            input_size,
            output_size,
        }
    }
    
    /// Forward pass with optional residual connection
    pub fn forward(&mut self, input: &[T]) -> Result<Vec<T>, ModelError> {
        let main_output = self.main_path.forward(input)?;
        
        if self.skip_connection && self.input_size == self.output_size {
            // Add residual connection
            let mut output = main_output;
            for (i, &input_val) in input.iter().enumerate().take(output.len()) {
                output[i] = output[i] + input_val;
            }
            Ok(output)
        } else {
            Ok(main_output)
        }
    }
}

/// Dense layer with configurable activation
#[derive(Debug, Clone)]
pub struct DenseLayer<T: Float> {
    pub network: Network<T>,
    pub input_size: usize,
    pub output_size: usize,
}

impl<T: Float> DenseLayer<T> {
    /// Create a new dense layer
    pub fn new(input_size: usize, output_size: usize, activation: ActivationFunction) -> Self {
        let network = NetworkBuilder::new()
            .input_layer(input_size)
            .output_layer_with_activation(output_size, activation, T::one())
            .build();
        
        Self {
            network,
            input_size,
            output_size,
        }
    }
    
    /// Forward pass through the layer
    pub fn forward(&mut self, input: &[T]) -> Result<Vec<T>, ModelError> {
        if input.len() != self.input_size {
            return Err(ModelError::DimensionMismatch {
                expected: self.input_size,
                actual: input.len(),
            });
        }
        
        Ok(self.network.run(input))
    }
}

/// Pooling operations for multi-resolution processing
#[derive(Debug, Clone, Copy)]
pub enum PoolingType {
    Max,
    Average,
    AdaptiveAverage,
}

/// Pooling layer for downsampling
#[derive(Debug, Clone)]
pub struct PoolingLayer<T: Float> {
    pub pool_type: PoolingType,
    pub pool_size: usize,
    pub stride: usize,
    phantom: PhantomData<T>,
}

impl<T: Float> PoolingLayer<T> {
    /// Create a new pooling layer
    pub fn new(pool_type: PoolingType, pool_size: usize, stride: Option<usize>) -> Self {
        Self {
            pool_type,
            pool_size,
            stride: stride.unwrap_or(pool_size),
            phantom: PhantomData,
        }
    }
    
    /// Apply pooling operation
    pub fn forward(&self, input: &[T]) -> Vec<T> {
        match self.pool_type {
            PoolingType::Max => self.max_pool(input),
            PoolingType::Average => self.avg_pool(input),
            PoolingType::AdaptiveAverage => self.adaptive_avg_pool(input),
        }
    }
    
    fn max_pool(&self, input: &[T]) -> Vec<T> {
        let mut output = Vec::new();
        let mut i = 0;
        
        while i + self.pool_size <= input.len() {
            let max_val = input[i..i + self.pool_size]
                .iter()
                .fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });
            
            output.push(max_val);
            i += self.stride;
        }
        
        output
    }
    
    fn avg_pool(&self, input: &[T]) -> Vec<T> {
        let mut output = Vec::new();
        let mut i = 0;
        
        while i + self.pool_size <= input.len() {
            let sum: T = input[i..i + self.pool_size].iter().sum();
            let avg = sum / T::from(self.pool_size).unwrap();
            
            output.push(avg);
            i += self.stride;
        }
        
        output
    }
    
    fn adaptive_avg_pool(&self, input: &[T]) -> Vec<T> {
        // Adaptive pooling to fixed output size
        let input_len = input.len();
        let output_size = self.pool_size;
        let mut output = vec![T::zero(); output_size];
        
        for i in 0..output_size {
            let start = (i * input_len) / output_size;
            let end = ((i + 1) * input_len) / output_size;
            
            if start < end {
                let sum: T = input[start..end].iter().sum();
                output[i] = sum / T::from(end - start).unwrap();
            }
        }
        
        output
    }
}

/// Interpolation layer for upsampling
#[derive(Debug, Clone)]
pub struct InterpolationLayer<T: Float> {
    pub interpolation_type: InterpolationType,
    pub target_size: usize,
    phantom: PhantomData<T>,
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationType {
    Linear,
    Nearest,
    Cubic,
}

impl<T: Float> InterpolationLayer<T> {
    /// Create a new interpolation layer
    pub fn new(interpolation_type: InterpolationType, target_size: usize) -> Self {
        Self {
            interpolation_type,
            target_size,
            phantom: PhantomData,
        }
    }
    
    /// Apply interpolation
    pub fn forward(&self, input: &[T]) -> Vec<T> {
        match self.interpolation_type {
            InterpolationType::Linear => self.linear_interpolate(input),
            InterpolationType::Nearest => self.nearest_interpolate(input),
            InterpolationType::Cubic => self.cubic_interpolate(input),
        }
    }
    
    fn linear_interpolate(&self, input: &[T]) -> Vec<T> {
        if input.len() <= 1 || self.target_size <= 1 {
            return vec![input.get(0).copied().unwrap_or(T::zero()); self.target_size];
        }
        
        let mut output = vec![T::zero(); self.target_size];
        let input_len = T::from(input.len() - 1).unwrap();
        let target_len = T::from(self.target_size - 1).unwrap();
        
        for i in 0..self.target_size {
            let pos = T::from(i).unwrap() * input_len / target_len;
            let idx = pos.floor().to_usize().unwrap().min(input.len() - 2);
            let frac = pos - T::from(idx).unwrap();
            
            output[i] = input[idx] * (T::one() - frac) + input[idx + 1] * frac;
        }
        
        output
    }
    
    fn nearest_interpolate(&self, input: &[T]) -> Vec<T> {
        let mut output = vec![T::zero(); self.target_size];
        let scale = input.len() as f64 / self.target_size as f64;
        
        for i in 0..self.target_size {
            let src_idx = ((i as f64 + 0.5) * scale).floor() as usize;
            let clamped_idx = src_idx.min(input.len() - 1);
            output[i] = input[clamped_idx];
        }
        
        output
    }
    
    fn cubic_interpolate(&self, input: &[T]) -> Vec<T> {
        // Simplified cubic interpolation (fallback to linear for now)
        self.linear_interpolate(input)
    }
}

/// Expression ratio calculation for multi-rate processing
pub fn calculate_expression_ratios<T: Float>(rates: &[usize]) -> Vec<T> {
    let max_rate = *rates.iter().max().unwrap_or(&1);
    rates.iter()
        .map(|&rate| T::from(rate).unwrap() / T::from(max_rate).unwrap())
        .collect()
}

/// Hierarchical sampling for multi-scale processing
pub fn hierarchical_sample<T: Float>(input: &[T], sampling_rates: &[usize]) -> Vec<Vec<T>> {
    sampling_rates.iter()
        .map(|&rate| {
            if rate == 1 {
                input.to_vec()
            } else {
                input.iter()
                    .step_by(rate)
                    .copied()
                    .collect()
            }
        })
        .collect()
}

/// Utility functions for tensor operations
pub mod tensor_ops {
    use super::*;
    
    /// Matrix multiplication for 2D tensors represented as Vec<Vec<T>>
    pub fn matmul<T: Float>(a: &[Vec<T>], b: &[Vec<T>]) -> Result<Vec<Vec<T>>, ModelError> {
        if a.is_empty() || b.is_empty() {
            return Err(ModelError::InvalidInput("Empty matrices".to_string()));
        }
        
        let rows_a = a.len();
        let cols_a = a[0].len();
        let rows_b = b.len();
        let cols_b = b[0].len();
        
        if cols_a != rows_b {
            return Err(ModelError::DimensionMismatch {
                expected: cols_a,
                actual: rows_b,
            });
        }
        
        let mut result = vec![vec![T::zero(); cols_b]; rows_a];
        
        for i in 0..rows_a {
            for j in 0..cols_b {
                for k in 0..cols_a {
                    result[i][j] = result[i][j] + a[i][k] * b[k][j];
                }
            }
        }
        
        Ok(result)
    }
    
    /// Element-wise addition
    pub fn add<T: Float>(a: &[T], b: &[T]) -> Result<Vec<T>, ModelError> {
        if a.len() != b.len() {
            return Err(ModelError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
    }
    
    /// Element-wise subtraction
    pub fn subtract<T: Float>(a: &[T], b: &[T]) -> Result<Vec<T>, ModelError> {
        if a.len() != b.len() {
            return Err(ModelError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect())
    }
    
    /// Element-wise multiplication
    pub fn multiply<T: Float>(a: &[T], b: &[T]) -> Result<Vec<T>, ModelError> {
        if a.len() != b.len() {
            return Err(ModelError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect())
    }
    
    /// Vector dot product
    pub fn dot<T: Float>(a: &[T], b: &[T]) -> Result<T, ModelError> {
        if a.len() != b.len() {
            return Err(ModelError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
    }
    
    /// Softmax activation
    pub fn softmax<T: Float>(input: &[T]) -> Vec<T> {
        let max_val = input.iter().fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });
        let exp_vals: Vec<T> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: T = exp_vals.iter().sum();
        
        exp_vals.iter().map(|&x| x / sum_exp).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mlp_block_creation() {
        let mlp = MLPBlock::<f32>::new(10, vec![64, 32, 16], ActivationFunction::ReLU);
        assert!(mlp.is_ok());
        
        let mlp = mlp.unwrap();
        assert_eq!(mlp.input_size(), 10);
        assert_eq!(mlp.output_size(), 16);
    }
    
    #[test]
    fn test_basis_function_generation() {
        let poly_basis = BasisFunction::<f32>::Polynomial { degree: 3 };
        let basis_matrix = poly_basis.generate_basis(10, 4);
        assert_eq!(basis_matrix.len(), 10);
        assert_eq!(basis_matrix[0].len(), 4);
        
        let fourier_basis = BasisFunction::<f32>::Fourier { harmonics: 2 };
        let basis_matrix = fourier_basis.generate_basis(8, 5);
        assert_eq!(basis_matrix.len(), 8);
        assert_eq!(basis_matrix[0].len(), 5);
    }
    
    #[test]
    fn test_pooling_operations() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let max_pool = PoolingLayer::<f32>::new(PoolingType::Max, 2, None);
        let result = max_pool.forward(&input);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
        
        let avg_pool = PoolingLayer::<f32>::new(PoolingType::Average, 2, None);
        let result = avg_pool.forward(&input);
        assert_eq!(result, vec![1.5, 3.5, 5.5, 7.5]);
    }
    
    #[test]
    fn test_interpolation() {
        let input = vec![1.0, 3.0, 5.0];
        let interpolation = InterpolationLayer::<f32>::new(InterpolationType::Linear, 5);
        let result = interpolation.forward(&input);
        
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[4], 5.0);
    }
    
    #[test]
    fn test_tensor_operations() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let sum = tensor_ops::add(&a, &b).unwrap();
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);
        
        let diff = tensor_ops::subtract(&a, &b).unwrap();
        assert_eq!(diff, vec![-3.0, -3.0, -3.0]);
        
        let dot = tensor_ops::dot(&a, &b).unwrap();
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }
}