//! Synaptic Neural WASM - WASM-optimized neural network engine
//!
//! This crate provides high-performance neural network operations optimized
//! for WebAssembly with SIMD acceleration support.

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use ndarray::{Array2, Array1, ArrayView2, ArrayView1};
use std::fmt::Debug;

/// Neural network errors
#[derive(Debug, thiserror::Error)]
pub enum NeuralError {
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub type Result<T> = std::result::Result<T, NeuralError>;

/// Activation functions
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

/// Layer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Dense { input_size: usize, output_size: usize },
    Dropout { rate: f32 },
}

/// Neural network layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub layer_type: LayerType,
    pub activation: Activation,
    pub weights: Option<Array2<f32>>,
    pub bias: Option<Array1<f32>>,
}

impl Layer {
    /// Create a dense layer
    pub fn dense(input_size: usize, output_size: usize) -> Self {
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rand::random::<f32>() * 0.1 - 0.05
        });
        let bias = Array1::zeros(output_size);
        
        Self {
            layer_type: LayerType::Dense { input_size, output_size },
            activation: Activation::ReLU,
            weights: Some(weights),
            bias: Some(bias),
        }
    }
    
    /// Set activation function
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
    
    /// Forward pass through the layer
    pub fn forward(&self, input: &ArrayView1<f32>) -> Result<Array1<f32>> {
        match &self.layer_type {
            LayerType::Dense { .. } => {
                let weights = self.weights.as_ref()
                    .ok_or_else(|| NeuralError::ComputationError("No weights".to_string()))?;
                let bias = self.bias.as_ref()
                    .ok_or_else(|| NeuralError::ComputationError("No bias".to_string()))?;
                
                let output = weights.dot(input) + bias;
                Ok(self.apply_activation(&output))
            }
            LayerType::Dropout { rate: _ } => {
                // In inference mode, dropout is identity
                Ok(input.to_owned())
            }
        }
    }
    
    /// Apply activation function
    fn apply_activation(&self, x: &Array1<f32>) -> Array1<f32> {
        match self.activation {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Linear => x.clone(),
        }
    }
}

/// Neural network
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

#[wasm_bindgen]
impl NeuralNetwork {
    /// Create a new neural network
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
        }
    }
    
    /// Get the number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
    
    /// Predict with the network (returns JSON string for WASM compatibility)
    pub fn predict(&self, input: &[f32]) -> std::result::Result<String, JsValue> {
        if self.layers.is_empty() {
            return Err(JsValue::from_str("No layers in network"));
        }
        
        let mut current = Array1::from_vec(input.to_vec());
        
        for layer in &self.layers {
            current = layer.forward(&current.view())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        
        let output: Vec<f32> = current.to_vec();
        serde_json::to_string(&output)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl NeuralNetwork {
    /// Add a layer to the network
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }
    
    /// Forward pass through the network
    pub fn forward(&self, input: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut current = input.to_owned();
        
        for layer in &self.layers {
            current = layer.forward(&current.view())?;
        }
        
        Ok(current)
    }
    
    /// Save network to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))
    }
    
    /// Load network from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))
    }
}

/// SIMD-accelerated operations
#[cfg(feature = "simd")]
pub mod simd {
    use super::*;
    
    /// SIMD dot product for f32
    pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        
        let mut sum = 0.0;
        let chunks = a.len() / 4;
        
        // Process 4 elements at a time
        for i in 0..chunks {
            let idx = i * 4;
            sum += a[idx] * b[idx];
            sum += a[idx + 1] * b[idx + 1];
            sum += a[idx + 2] * b[idx + 2];
            sum += a[idx + 3] * b[idx + 3];
        }
        
        // Handle remaining elements
        for i in (chunks * 4)..a.len() {
            sum += a[i] * b[i];
        }
        
        sum
    }
    
    /// SIMD matrix multiplication
    pub fn matmul_simd(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2);
        
        let mut result = Array2::zeros((m, n));
        
        for i in 0..m {
            for j in 0..n {
                let a_row = a.row(i);
                let b_col = b.column(j);
                result[[i, j]] = dot_product_simd(a_row.as_slice().unwrap(), b_col.as_slice().unwrap());
            }
        }
        
        result
    }
}

/// WebAssembly utilities
#[wasm_bindgen]
pub fn init_panic_hook() {
    // Set panic hook for better error messages
    // Feature flag would be needed for console_error_panic_hook
}

/// Performance utilities
#[wasm_bindgen]
pub struct Performance;

#[wasm_bindgen]
impl Performance {
    /// Get current timestamp in milliseconds
    pub fn now() -> f64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0)
    }
}

/// Measure execution time
pub fn measure_time<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
    let start = Performance::now();
    let result = f();
    let elapsed = Performance::now() - start;
    (result, elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layer_creation() {
        let layer = Layer::dense(10, 5);
        assert_eq!(layer.weights.as_ref().unwrap().dim(), (5, 10));
        assert_eq!(layer.bias.as_ref().unwrap().len(), 5);
    }
    
    #[test]
    fn test_activation_functions() {
        let input = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let layer = Layer::dense(3, 3).with_activation(Activation::ReLU);
        
        let activated = layer.apply_activation(&input);
        assert_eq!(activated[0], 0.0); // ReLU(-1) = 0
        assert_eq!(activated[1], 0.0); // ReLU(0) = 0
        assert_eq!(activated[2], 1.0); // ReLU(1) = 1
    }
    
    #[test]
    fn test_network_creation() {
        let mut network = NeuralNetwork::new();
        network.add_layer(Layer::dense(784, 128));
        network.add_layer(Layer::dense(128, 10));
        
        assert_eq!(network.layer_count(), 2);
    }
    
    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = simd::dot_product_simd(&a, &b);
        assert_eq!(result, 70.0); // 1*5 + 2*6 + 3*7 + 4*8
    }
}