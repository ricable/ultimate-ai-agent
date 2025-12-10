/*!
# Tensor Operations Compatibility Layer

This module provides a compatibility layer for tensor operations used by AFM detect module.
*/

use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

/// Simple tensor structure using ndarray backend
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Array2<f32>,
    device: Device,
}

/// Device enumeration
#[derive(Debug, Clone, Copy)]
pub enum Device {
    Cpu,
}

/// Result type for tensor operations
pub type Result<T> = std::result::Result<T, TensorError>;

/// Tensor operation errors
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
}

impl Tensor {
    /// Create a new tensor from a 2D array
    pub fn new(data: Array2<f32>, device: Device) -> Self {
        Self { data, device }
    }
    
    /// Create a tensor from a vector with specified shape
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        if shape.len() != 2 {
            return Err(TensorError::InvalidDimension(
                "Only 2D tensors supported".to_string()
            ));
        }
        
        let expected_size = shape.iter().product::<usize>();
        if data.len() != expected_size {
            return Err(TensorError::ShapeMismatch {
                expected: vec![expected_size],
                actual: vec![data.len()],
            });
        }
        
        let array = Array2::from_shape_vec((shape[0], shape[1]), data)
            .map_err(|e| TensorError::ComputationError(e.to_string()))?;
        
        Ok(Self::new(array, Device::Cpu))
    }
    
    /// Create a tensor filled with random values
    pub fn randn(mean: f32, std: f32, shape: &[usize], device: &Device) -> Result<Self> {
        use rand::prelude::*;
        use rand_distr::Normal;
        
        if shape.len() != 2 {
            return Err(TensorError::InvalidDimension(
                "Only 2D tensors supported".to_string()
            ));
        }
        
        let mut rng = thread_rng();
        let normal = Normal::new(mean, std)
            .map_err(|e| TensorError::ComputationError(e.to_string()))?;
        
        let size = shape.iter().product::<usize>();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        
        Self::from_vec(data, shape)
    }
    
    /// Get the shape of the tensor
    pub fn dims(&self) -> Vec<usize> {
        vec![self.data.nrows(), self.data.ncols()]
    }
    
    /// Get the device
    pub fn device(&self) -> Device {
        self.device
    }
    
    /// Convert to 2D vector
    pub fn to_vec2(&self) -> Result<Vec<Vec<f32>>> {
        let mut result = Vec::with_capacity(self.data.nrows());
        for row in self.data.rows() {
            result.push(row.to_vec());
        }
        Ok(result)
    }
    
    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let result = self.data.dot(&other.data);
        Ok(Tensor::new(result, self.device))
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data + &other.data;
        Ok(Tensor::new(result, self.device))
    }
    
    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data - &other.data;
        Ok(Tensor::new(result, self.device))
    }
    
    /// Mean along all dimensions
    pub fn mean_all(&self) -> Result<Scalar> {
        let mean = self.data.mean().unwrap_or(0.0);
        Ok(Scalar { value: mean })
    }
    
    /// Maximum along all dimensions
    pub fn max_all(&self) -> Result<Scalar> {
        let max = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        Ok(Scalar { value: max })
    }
    
    /// Apply a function element-wise
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        let result = self.data.mapv(f);
        Tensor::new(result, self.device)
    }
    
    /// Get a narrow slice along a dimension
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor> {
        match dim {
            0 => {
                if start + len > self.data.nrows() {
                    return Err(TensorError::InvalidDimension(
                        "Slice extends beyond tensor bounds".to_string()
                    ));
                }
                let slice = self.data.slice(ndarray::s![start..start+len, ..]);
                Ok(Tensor::new(slice.to_owned(), self.device))
            }
            1 => {
                if start + len > self.data.ncols() {
                    return Err(TensorError::InvalidDimension(
                        "Slice extends beyond tensor bounds".to_string()
                    ));
                }
                let slice = self.data.slice(ndarray::s![.., start..start+len]);
                Ok(Tensor::new(slice.to_owned(), self.device))
            }
            _ => Err(TensorError::InvalidDimension(
                "Only dimensions 0 and 1 supported".to_string()
            ))
        }
    }
    
    /// Get element at index
    pub fn i(&self, idx: usize) -> Result<Scalar> {
        if idx >= self.data.len() {
            return Err(TensorError::InvalidDimension(
                "Index out of bounds".to_string()
            ));
        }
        
        let flat_data: Vec<f32> = self.data.iter().cloned().collect();
        Ok(Scalar { value: flat_data[idx] })
    }
    
    /// Absolute value
    pub fn abs(&self) -> Result<Tensor> {
        let result = self.data.mapv(|x| x.abs());
        Ok(Tensor::new(result, self.device))
    }
    
    /// Element-wise minimum with scalar
    pub fn min(&self, value: f32) -> Tensor {
        let result = self.data.mapv(|x| x.min(value));
        Tensor::new(result, self.device)
    }
    
    /// Mean along dimension 0
    pub fn mean(&self, dim: usize) -> Result<Tensor> {
        match dim {
            0 => {
                let mean = self.data.mean_axis(ndarray::Axis(0)).unwrap();
                let reshaped = mean.insert_axis(ndarray::Axis(0));
                Ok(Tensor::new(reshaped, self.device))
            }
            _ => Err(TensorError::InvalidDimension(
                "Only dimension 0 supported for mean".to_string()
            ))
        }
    }
}

/// Scalar value from tensor operations
#[derive(Debug, Clone)]
pub struct Scalar {
    value: f32,
}

impl Scalar {
    pub fn to_scalar<T>(&self) -> Result<T> 
    where 
        T: From<f32>
    {
        Ok(T::from(self.value))
    }
}

/// Module trait for neural network layers
pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

/// Variable builder for layer parameters
pub struct VarBuilder {
    device: Device,
}

impl VarBuilder {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
    
    pub fn pp(&self, _name: &str) -> Self {
        Self { device: self.device }
    }
    
    pub fn get_with_hints(&self, shape: &[usize], _name: &str) -> Result<Tensor> {
        // Initialize with small random values
        Tensor::randn(0.0, 0.1, shape, &self.device)
    }
}

/// Variable map for storing parameters
pub struct VarMap {
    device: Device,
}

impl VarMap {
    pub fn new() -> Self {
        Self { device: Device::Cpu }
    }
}

impl Default for VarMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear layer implementation
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(&[in_dim, out_dim], "weight")?;
        let bias = Some(vb.get_with_hints(&[1, out_dim], "bias")?);
        
        Ok(Self { weight, bias })
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = input.matmul(&self.weight)?;
        
        if let Some(ref bias) = self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }
}

/// Data type enumeration
#[derive(Debug, Clone, Copy)]
pub enum DType {
    F32,
}

/// Create VarBuilder from VarMap
pub fn var_builder_from_varmap(var_map: &VarMap, dtype: DType, device: &Device) -> VarBuilder {
    VarBuilder::new(*device)
}

/// AdamW optimizer
pub struct AdamW {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

impl AdamW {
    pub fn new(params: Vec<Tensor>) -> Result<Self> {
        Ok(Self {
            params,
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        })
    }
    
    pub fn step(&mut self) -> Result<()> {
        // Simple placeholder implementation
        Ok(())
    }
}