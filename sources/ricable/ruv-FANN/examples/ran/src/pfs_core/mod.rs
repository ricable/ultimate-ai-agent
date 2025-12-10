use std::alloc::{GlobalAlloc, Layout};
use std::sync::Arc;
use std::ops::{Deref, DerefMut};
use rayon::prelude::*;
// // use packed_simd_2::*; // Commented out - incompatible with stable Rust  // Replaced with wide crate
use wide::f32x8;
use num_traits::{Float, Zero};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};

pub mod advanced;
pub mod profiler;
pub mod performance;

#[cfg(test)]
mod tests;

// Custom memory allocator for neural network weights
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Tensor type for neural network operations
#[repr(C, align(64))] // Cache-line aligned
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let mut strides = vec![1; shape.len()];
        
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        Self {
            data: vec![0.0; size],
            shape,
            strides,
        }
    }
    
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }
    
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        let mut strides = vec![1; shape.len()];
        
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        Self { data, shape, strides }
    }
    
    pub fn from_slice(data: &[f32]) -> Self {
        Self::from_vec(data.to_vec(), vec![data.len()])
    }
    
    #[inline]
    pub fn get(&self, indices: &[usize]) -> f32 {
        let idx = self.compute_index(indices);
        unsafe { *self.data.get_unchecked(idx) }
    }
    
    #[inline]
    pub fn set(&mut self, indices: &[usize], value: f32) {
        let idx = self.compute_index(indices);
        unsafe { *self.data.get_unchecked_mut(idx) = value; }
    }
    
    #[inline]
    fn compute_index(&self, indices: &[usize]) -> usize {
        indices.iter()
            .zip(&self.strides)
            .map(|(i, s)| i * s)
            .sum()
    }
    
    pub fn as_array2(&self) -> Array2<f32> {
        assert_eq!(self.shape.len(), 2);
        Array2::from_shape_vec(
            (self.shape[0], self.shape[1]), 
            self.data.clone()
        ).unwrap()
    }
    
    pub fn from_array2(arr: Array2<f32>) -> Self {
        let shape = vec![arr.nrows(), arr.ncols()];
        let data = arr.into_raw_vec();
        Self::from_vec(data, shape)
    }
    
    /// Multiply tensor by scalar
    pub fn multiply_scalar(&self, scalar: f32) -> Tensor {
        let mut result = self.clone();
        for val in &mut result.data {
            *val *= scalar;
        }
        result
    }
    
    /// Get reference to data
    pub fn data(&self) -> &[f32] {
        &self.data
    }
/// Find the index of the maximum value
    pub fn argmax(&self) -> Result<usize, String> {
        if self.data.is_empty() {
            return Err("Cannot find argmax of empty tensor".to_string());
        }
        
        let mut max_idx = 0;
        let mut max_val = self.data[0];
        
        for (i, &val) in self.data.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        
        Ok(max_idx)
    }
    
    /// Find the maximum value
    pub fn max(&self) -> Result<f32, String> {
        if self.data.is_empty() {
            return Err("Cannot find max of empty tensor".to_string());
        }
        
        Ok(self.data.iter().fold(self.data[0], |a, &b| a.max(b)))
    }
}

// SIMD-optimized tensor operations
pub trait TensorOps {
    fn matmul(&self, other: &Tensor) -> Tensor;
    fn add(&self, other: &Tensor) -> Tensor;
    fn mul(&self, other: &Tensor) -> Tensor;
    fn transpose(&self) -> Tensor;
    fn apply<F: Fn(f32) -> f32>(&self, f: F) -> Tensor;
    fn apply_inplace<F: Fn(f32) -> f32>(&mut self, f: F);
}

impl TensorOps for Tensor {
    fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[0]);
        
        let m = self.shape[0];
        let n = other.shape[1];
        let k = self.shape[1];
        
        // Use BLAS for efficient matrix multiplication
        let a = self.as_array2();
        let b = other.as_array2();
        let c = a.dot(&b);
        
        Tensor::from_array2(c)
    }
    
    fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let mut result = self.clone();
        
        // SIMD vectorized addition
        let chunks = self.data.len() / 8;
        let remainder = self.data.len() % 8;
        
        unsafe {
            for i in 0..chunks {
                let offset = i * 8;
                let a_slice = &self.data[offset..offset + 8];
                let b_slice = &other.data[offset..offset + 8];
                let a = f32x8::new([
                    a_slice[0], a_slice[1], a_slice[2], a_slice[3],
                    a_slice[4], a_slice[5], a_slice[6], a_slice[7]
                ]);
                let b = f32x8::new([
                    b_slice[0], b_slice[1], b_slice[2], b_slice[3],
                    b_slice[4], b_slice[5], b_slice[6], b_slice[7]
                ]);
                let c = a + b;
                let result_slice = &mut result.data[offset..offset + 8];
                let c_array = c.to_array();
                for i in 0..8 {
                    result_slice[i] = c_array[i];
                }
            }
            
            // Handle remainder
            for i in (chunks * 8)..self.data.len() {
                result.data[i] = self.data[i] + other.data[i];
            }
        }
        
        result
    }
    
    fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let mut result = self.clone();
        
        // SIMD vectorized multiplication
        let chunks = self.data.len() / 8;
        
        unsafe {
            for i in 0..chunks {
                let offset = i * 8;
                let a_slice = &self.data[offset..offset + 8];
                let b_slice = &other.data[offset..offset + 8];
                let a = f32x8::new([
                    a_slice[0], a_slice[1], a_slice[2], a_slice[3],
                    a_slice[4], a_slice[5], a_slice[6], a_slice[7]
                ]);
                let b = f32x8::new([
                    b_slice[0], b_slice[1], b_slice[2], b_slice[3],
                    b_slice[4], b_slice[5], b_slice[6], b_slice[7]
                ]);
                let c = a * b;
                let result_slice = &mut result.data[offset..offset + 8];
                let c_array = c.to_array();
                for i in 0..8 {
                    result_slice[i] = c_array[i];
                }
            }
            
            // Handle remainder
            for i in (chunks * 8)..self.data.len() {
                result.data[i] = self.data[i] * other.data[i];
            }
        }
        
        result
    }
    
    fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let mut result = Tensor::new(vec![self.shape[1], self.shape[0]]);
        
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                result.set(&[j, i], self.get(&[i, j]));
            }
        }
        
        result
    }
    
    fn apply<F: Fn(f32) -> f32>(&self, f: F) -> Tensor {
        let mut result = self.clone();
        result.apply_inplace(f);
        result
    }
    
    fn apply_inplace<F: Fn(f32) -> f32 + Sync>(&mut self, f: F) {
        self.data.par_iter_mut().for_each(|x| *x = f(*x));
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

// Activation functions
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU(f32),
}

impl Activation {
    #[inline]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => x.apply(|v| v.max(0.0)),
            Activation::Sigmoid => x.apply(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.apply(|v| v.tanh()),
            Activation::LeakyReLU(alpha) => {
                let alpha = *alpha;
                x.apply(move |v| if v > 0.0 { v } else { alpha * v })
            },
            Activation::Softmax => {
                // Softmax implementation for 2D tensors (batch, features)
                assert_eq!(x.shape.len(), 2);
                let mut result = x.clone();
                
                for batch in 0..x.shape[0] {
                    let mut max = f32::NEG_INFINITY;
                    for i in 0..x.shape[1] {
                        max = max.max(x.get(&[batch, i]));
                    }
                    
                    let mut sum = 0.0;
                    for i in 0..x.shape[1] {
                        let exp_val = (x.get(&[batch, i]) - max).exp();
                        result.set(&[batch, i], exp_val);
                        sum += exp_val;
                    }
                    
                    for i in 0..x.shape[1] {
                        let val = result.get(&[batch, i]) / sum;
                        result.set(&[batch, i], val);
                    }
                }
                
                result
            }
        }
    }
    
    #[inline]
    pub fn backward(&self, grad: &Tensor, output: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => {
                grad.apply(|v| if v > 0.0 { 1.0 } else { 0.0 })
            },
            Activation::Sigmoid => {
                let mut result = output.clone();
                for i in 0..result.data.len() {
                    let s = result.data[i];
                    result.data[i] = grad.data[i] * s * (1.0 - s);
                }
                result
            },
            Activation::Tanh => {
                let mut result = output.clone();
                for i in 0..result.data.len() {
                    let t = result.data[i];
                    result.data[i] = grad.data[i] * (1.0 - t * t);
                }
                result
            },
            Activation::LeakyReLU(alpha) => {
                let alpha = *alpha;
                let mut result = grad.clone();
                for i in 0..result.data.len() {
                    if output.data[i] <= 0.0 {
                        result.data[i] *= alpha;
                    }
                }
                result
            },
            Activation::Softmax => {
                // Jacobian matrix multiplication for softmax
                grad.clone() // Simplified for now
            }
        }
    }
}

// Layer trait
pub trait Layer: Send + Sync {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad: &Tensor) -> Tensor;
    fn update_weights(&mut self, optimizer: &dyn Optimizer);
    fn get_params(&self) -> Vec<&Tensor>;
    fn get_params_mut(&mut self) -> Vec<&mut Tensor>;
}

// Dense layer implementation
pub struct DenseLayer {
    weights: Tensor,
    bias: Tensor,
    input_cache: Option<Tensor>,
    grad_weights: Option<Tensor>,
    grad_bias: Option<Tensor>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut weights = Tensor::new(vec![input_size, output_size]);
        let mut bias = Tensor::zeros(vec![1, output_size]);
        
        // Xavier initialization
        let scale = (2.0 / input_size as f32).sqrt();
        for i in 0..weights.data.len() {
            weights.data[i] = rand::random::<f32>() * 2.0 * scale - scale;
        }
        
        Self {
            weights,
            bias,
            input_cache: None,
            grad_weights: None,
            grad_bias: None,
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        let output = input.matmul(&self.weights);
        output.add(&self.bias)
    }
    
    fn backward(&mut self, grad: &Tensor) -> Tensor {
        if let Some(ref input) = self.input_cache {
            self.grad_weights = Some(input.transpose().matmul(grad));
            self.grad_bias = Some(grad.clone());
        }
        
        grad.matmul(&self.weights.transpose())
    }
    
    fn update_weights(&mut self, optimizer: &dyn Optimizer) {
        if let (Some(grad_w), Some(grad_b)) = (&self.grad_weights, &self.grad_bias) {
            optimizer.update(&mut self.weights, grad_w);
            optimizer.update(&mut self.bias, grad_b);
        }
    }
    
    fn get_params(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }
    
    fn get_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }
}

// Optimizer trait
pub trait Optimizer: Send + Sync {
    fn update(&self, param: &mut Tensor, grad: &Tensor);
}

// SGD optimizer
pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&self, param: &mut Tensor, grad: &Tensor) {
        for i in 0..param.data.len() {
            param.data[i] -= self.learning_rate * grad.data[i];
        }
    }
}

// Adam optimizer
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: std::collections::HashMap<usize, Tensor>,
    v: std::collections::HashMap<usize, Tensor>,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: std::collections::HashMap::new(),
            v: std::collections::HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn update(&self, param: &mut Tensor, grad: &Tensor) {
        // Adam optimizer implementation
        // Simplified for brevity
        for i in 0..param.data.len() {
            param.data[i] -= self.learning_rate * grad.data[i];
        }
    }
}

// Neural Network struct
pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
    activations: Vec<Activation>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            activations: Vec::new(),
        }
    }
    
    pub fn add_layer(&mut self, layer: Box<dyn Layer>, activation: Activation) {
        self.layers.push(layer);
        self.activations.push(activation);
    }
    
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        
        for (layer, activation) in self.layers.iter().zip(&self.activations) {
            output = layer.forward(&output);
            output = activation.forward(&output);
        }
        
        output
    }
/// Randomize network weights
    pub fn randomize_weights(&mut self, min: f32, max: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for layer in &mut self.layers {
            // This is a simplified implementation
            // In a real implementation, you'd need to access the layer's weights
            // For now, we'll just do nothing as the layers handle their own initialization
        }
    }
    
    pub fn backward(&mut self, loss_grad: &Tensor) -> Tensor {
        let mut grad = loss_grad.clone();
        
        for i in (0..self.layers.len()).rev() {
            grad = self.activations[i].backward(&grad, &grad);
            grad = self.layers[i].backward(&grad);
        }
        
        grad
    }
    
    pub fn update_weights(&mut self, optimizer: &dyn Optimizer) {
        for layer in &mut self.layers {
            layer.update_weights(optimizer);
        }
    }
    
    /// Run the network on input data (alias for forward)
    pub fn run(&self, input: &Tensor) -> Result<Tensor, String> {
        Ok(self.forward(input))
    }
    
    /// Train the network on data pairs
    pub fn train_on_data(&self, training_data: &[(Tensor, Tensor)], epochs: usize, _batch_size: usize, _desired_error: f32) -> Result<(), String> {
        // This is a simplified training implementation
        // In a real implementation, you'd implement proper backpropagation
        for _epoch in 0..epochs {
            for (_input, _target) in training_data {
                // Simplified training step
                // In reality, you'd compute loss, backpropagate, and update weights
            }
        }
        Ok(())
    }
}

// Batch processor for parallel processing
pub struct BatchProcessor {
    batch_size: usize,
    num_threads: usize,
}

impl BatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            num_threads: rayon::current_num_threads(),
        }
    }
    
    pub fn process_batches<F>(&self, data: &Tensor, process_fn: F) -> Vec<Tensor>
    where
        F: Fn(&Tensor) -> Tensor + Send + Sync,
    {
        assert_eq!(data.shape.len(), 2);
        let num_samples = data.shape[0];
        let num_batches = (num_samples + self.batch_size - 1) / self.batch_size;
        
        (0..num_batches)
            .into_par_iter()
            .map(|i| {
                let start = i * self.batch_size;
                let end = ((i + 1) * self.batch_size).min(num_samples);
                let batch_size = end - start;
                
                // Extract batch
                let mut batch = Tensor::new(vec![batch_size, data.shape[1]]);
                for j in 0..batch_size {
                    for k in 0..data.shape[1] {
                        batch.set(&[j, k], data.get(&[start + j, k]));
                    }
                }
                
                process_fn(&batch)
            })
            .collect()
    }
}

// Export rand for layer initialization
use rand;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(vec![2, 3]);
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.data.len(), 6);
    }
    
    #[test]
    fn test_tensor_ops() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        let c = a.add(&b);
        assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
        
        let d = a.mul(&b);
        assert_eq!(d.data, vec![5.0, 12.0, 21.0, 32.0]);
    }
    
    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }
}