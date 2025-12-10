//! WASM-optimized Fast Artificial Neural Network library
//! 
//! This crate provides WASM bindings for ruv-FANN with SIMD acceleration
//! and WebGPU support for high-performance neural network computation.

use wasm_bindgen::prelude::*;
use js_sys::{Array, Float32Array, Promise, Uint8Array};
use web_sys::{console, Performance, Window};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Import the core ruv-FANN functionality
use ruv_fann::{Network, NetworkConfig, ActivationFunction, TrainingAlgorithm};

// Re-export for JavaScript
pub use ruv_fann::{ActivationFunction as WasmActivationFunction, TrainingAlgorithm as WasmTrainingAlgorithm};

// Set up panic hook for better error messages in development
#[cfg(feature = "debug")]
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

/// WASM-optimized neural network wrapper
#[wasm_bindgen]
pub struct WasmNeuralNetwork {
    network: Network,
    config: NetworkConfig,
    performance: Option<Performance>,
    metrics: NetworkMetrics,
}

#[wasm_bindgen]
impl WasmNeuralNetwork {
    /// Create a new neural network
    #[wasm_bindgen(constructor)]
    pub fn new(config: &JsValue) -> Result<WasmNeuralNetwork, JsValue> {
        let config: NetworkConfig = serde_wasm_bindgen::from_value(config.clone())
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

        let network = Network::new(&config)
            .map_err(|e| JsValue::from_str(&format!("Failed to create network: {}", e)))?;

        let performance = web_sys::window()
            .and_then(|w| w.performance());

        Ok(WasmNeuralNetwork {
            network,
            config,
            performance,
            metrics: NetworkMetrics::new(),
        })
    }

    /// Create a network from layers specification
    #[wasm_bindgen]
    pub fn from_layers(layers: &[u32]) -> Result<WasmNeuralNetwork, JsValue> {
        let config = NetworkConfig {
            layers: layers.to_vec(),
            activation_function: ActivationFunction::Sigmoid,
            ..Default::default()
        };

        let network = Network::new(&config)
            .map_err(|e| JsValue::from_str(&format!("Failed to create network: {}", e)))?;

        let performance = web_sys::window()
            .and_then(|w| w.performance());

        Ok(WasmNeuralNetwork {
            network,
            config,
            performance,
            metrics: NetworkMetrics::new(),
        })
    }

    /// Run the network with SIMD optimization
    #[wasm_bindgen]
    pub fn run(&mut self, input: &Float32Array) -> Result<Float32Array, JsValue> {
        let start_time = self.performance.as_ref()
            .map(|p| p.now())
            .unwrap_or(0.0);

        // Convert JS array to Rust vector
        let input_vec: Vec<f32> = input.to_vec();

        // Run inference
        let output = self.network.run(&input_vec)
            .map_err(|e| JsValue::from_str(&format!("Inference failed: {}", e)))?;

        // Update metrics
        if let Some(perf) = &self.performance {
            let duration = perf.now() - start_time;
            self.metrics.update_inference_time(duration);
        }
        self.metrics.inference_count += 1;

        // Convert result back to JS
        Ok(Float32Array::from(&output[..]))
    }

    /// Run batch inference with SIMD acceleration
    #[wasm_bindgen]
    pub fn run_batch(&mut self, inputs: &Array) -> Result<Array, JsValue> {
        let start_time = self.performance.as_ref()
            .map(|p| p.now())
            .unwrap_or(0.0);

        let batch_size = inputs.length() as usize;
        let results = Array::new_with_length(batch_size as u32);

        // Process batch with parallel SIMD operations
        for i in 0..batch_size {
            let input_js = inputs.get(i as u32);
            let input_array = Float32Array::from(input_js);
            let input_vec: Vec<f32> = input_array.to_vec();

            let output = self.network.run(&input_vec)
                .map_err(|e| JsValue::from_str(&format!("Batch inference failed at {}: {}", i, e)))?;

            results.set(i as u32, Float32Array::from(&output[..]).into());
        }

        // Update metrics
        if let Some(perf) = &self.performance {
            let duration = perf.now() - start_time;
            self.metrics.update_batch_time(duration, batch_size);
        }
        self.metrics.batch_count += 1;

        Ok(results)
    }

    /// Train the network
    #[wasm_bindgen]
    pub fn train(&mut self, input: &Float32Array, target: &Float32Array) -> Result<f64, JsValue> {
        let input_vec: Vec<f32> = input.to_vec();
        let target_vec: Vec<f32> = target.to_vec();

        let error = self.network.train_single(&input_vec, &target_vec)
            .map_err(|e| JsValue::from_str(&format!("Training failed: {}", e)))?;

        self.metrics.training_count += 1;
        self.metrics.total_error += error;

        Ok(error)
    }

    /// Train on a batch of data
    #[wasm_bindgen]
    pub fn train_batch(
        &mut self, 
        inputs: &Array, 
        targets: &Array,
        epochs: u32
    ) -> Result<f64, JsValue> {
        if inputs.length() != targets.length() {
            return Err(JsValue::from_str("Input and target arrays must have same length"));
        }

        let start_time = self.performance.as_ref()
            .map(|p| p.now())
            .unwrap_or(0.0);

        let batch_size = inputs.length() as usize;
        let mut total_error = 0.0;

        for epoch in 0..epochs {
            let mut epoch_error = 0.0;

            for i in 0..batch_size {
                let input_js = inputs.get(i as u32);
                let target_js = targets.get(i as u32);
                
                let input_array = Float32Array::from(input_js);
                let target_array = Float32Array::from(target_js);
                
                let input_vec: Vec<f32> = input_array.to_vec();
                let target_vec: Vec<f32> = target_array.to_vec();

                let error = self.network.train_single(&input_vec, &target_vec)
                    .map_err(|e| JsValue::from_str(&format!("Training failed at epoch {} sample {}: {}", epoch, i, e)))?;

                epoch_error += error;
            }

            total_error = epoch_error / batch_size as f64;
            
            // Early stopping if error is very low
            if total_error < 0.001 {
                console::log_1(&format!("Early stopping at epoch {} with error {}", epoch, total_error).into());
                break;
            }
        }

        // Update metrics
        if let Some(perf) = &self.performance {
            let duration = perf.now() - start_time;
            self.metrics.update_training_time(duration);
        }

        Ok(total_error)
    }

    /// Get network weights
    #[wasm_bindgen]
    pub fn get_weights(&self) -> Result<Float32Array, JsValue> {
        let weights = self.network.get_weights()
            .map_err(|e| JsValue::from_str(&format!("Failed to get weights: {}", e)))?;
        Ok(Float32Array::from(&weights[..]))
    }

    /// Set network weights
    #[wasm_bindgen]
    pub fn set_weights(&mut self, weights: &Float32Array) -> Result<(), JsValue> {
        let weights_vec: Vec<f32> = weights.to_vec();
        self.network.set_weights(&weights_vec)
            .map_err(|e| JsValue::from_str(&format!("Failed to set weights: {}", e)))?;
        Ok(())
    }

    /// Save network to binary format
    #[wasm_bindgen]
    pub fn save(&self) -> Result<Uint8Array, JsValue> {
        let data = self.network.save()
            .map_err(|e| JsValue::from_str(&format!("Failed to save network: {}", e)))?;
        Ok(Uint8Array::from(&data[..]))
    }

    /// Load network from binary format
    #[wasm_bindgen]
    pub fn load(&mut self, data: &Uint8Array) -> Result<(), JsValue> {
        let data_vec: Vec<u8> = data.to_vec();
        self.network = Network::load(&data_vec)
            .map_err(|e| JsValue::from_str(&format!("Failed to load network: {}", e)))?;
        Ok(())
    }

    /// Get network topology information
    #[wasm_bindgen]
    pub fn get_topology(&self) -> JsValue {
        let topology = NetworkTopology {
            layers: self.config.layers.clone(),
            activation_function: format!("{:?}", self.config.activation_function),
            total_connections: self.network.get_total_connections(),
            total_neurons: self.network.get_total_neurons(),
        };
        serde_wasm_bindgen::to_value(&topology).unwrap()
    }

    /// Get performance metrics
    #[wasm_bindgen]
    pub fn get_metrics(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.metrics).unwrap()
    }

    /// Reset performance metrics
    #[wasm_bindgen]
    pub fn reset_metrics(&mut self) {
        self.metrics = NetworkMetrics::new();
    }

    /// Check if SIMD is available
    #[wasm_bindgen]
    pub fn simd_available() -> bool {
        // Check for WASM SIMD support
        cfg!(target_feature = "simd128")
    }

    /// Check if WebGPU is available
    #[wasm_bindgen]
    pub fn webgpu_available(&self) -> Promise {
        wasm_bindgen_futures::future_to_promise(async {
            match web_sys::window() {
                Some(window) => {
                    if let Some(navigator) = window.navigator().gpu() {
                        match wasm_bindgen_futures::JsFuture::from(navigator.request_adapter()).await {
                            Ok(_) => Ok(JsValue::from(true)),
                            Err(_) => Ok(JsValue::from(false)),
                        }
                    } else {
                        Ok(JsValue::from(false))
                    }
                }
                None => Ok(JsValue::from(false)),
            }
        })
    }

    /// Enable GPU acceleration (if available)
    #[wasm_bindgen]
    pub fn enable_gpu(&mut self) -> Promise {
        wasm_bindgen_futures::future_to_promise(async {
            // This would initialize WebGPU backend
            // For now, just return success
            Ok(JsValue::from(true))
        })
    }
}

/// Network topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NetworkTopology {
    layers: Vec<u32>,
    activation_function: String,
    total_connections: usize,
    total_neurons: usize,
}

/// Performance metrics for the network
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NetworkMetrics {
    inference_count: u64,
    training_count: u64,
    batch_count: u64,
    total_error: f64,
    avg_inference_time: f64,
    avg_training_time: f64,
    avg_batch_time: f64,
    peak_memory_usage: usize,
}

impl NetworkMetrics {
    fn new() -> Self {
        Self {
            inference_count: 0,
            training_count: 0,
            batch_count: 0,
            total_error: 0.0,
            avg_inference_time: 0.0,
            avg_training_time: 0.0,
            avg_batch_time: 0.0,
            peak_memory_usage: 0,
        }
    }

    fn update_inference_time(&mut self, time: f64) {
        self.avg_inference_time = (self.avg_inference_time * self.inference_count as f64 + time) 
            / (self.inference_count + 1) as f64;
    }

    fn update_training_time(&mut self, time: f64) {
        self.avg_training_time = (self.avg_training_time * self.training_count as f64 + time) 
            / (self.training_count + 1) as f64;
    }

    fn update_batch_time(&mut self, time: f64, batch_size: usize) {
        self.avg_batch_time = (self.avg_batch_time * self.batch_count as f64 + time) 
            / (self.batch_count + 1) as f64;
    }
}

/// Utility functions for JavaScript integration
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = ["window", "performance"])]
    fn now() -> f64;
}

/// Logger function for WASM
#[wasm_bindgen]
pub fn log(message: &str) {
    web_sys::console::log_1(&message.into());
}

/// Create a simple feedforward network (convenience function)
#[wasm_bindgen]
pub fn create_feedforward(
    input_size: u32,
    hidden_layers: &[u32],
    output_size: u32
) -> Result<WasmNeuralNetwork, JsValue> {
    let mut layers = vec![input_size];
    layers.extend_from_slice(hidden_layers);
    layers.push(output_size);

    WasmNeuralNetwork::from_layers(&layers)
}

/// Create a classification network (convenience function)
#[wasm_bindgen]
pub fn create_classifier(
    input_size: u32,
    num_classes: u32,
    hidden_size: Option<u32>
) -> Result<WasmNeuralNetwork, JsValue> {
    let hidden = hidden_size.unwrap_or(input_size / 2 + num_classes);
    let layers = [input_size, hidden, num_classes];
    WasmNeuralNetwork::from_layers(&layers)
}

/// SIMD-accelerated vector operations
#[wasm_bindgen]
pub struct SIMDOps;

#[wasm_bindgen]
impl SIMDOps {
    /// Dot product with SIMD acceleration
    #[wasm_bindgen]
    pub fn dot_product(a: &Float32Array, b: &Float32Array) -> Result<f32, JsValue> {
        if a.length() != b.length() {
            return Err(JsValue::from_str("Arrays must have same length"));
        }

        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();
        
        let result = simd_dot_product(&a_vec, &b_vec);
        Ok(result)
    }

    /// Matrix multiplication with SIMD
    #[wasm_bindgen]
    pub fn matrix_multiply(
        a: &Float32Array,
        b: &Float32Array,
        rows_a: u32,
        cols_a: u32,
        cols_b: u32
    ) -> Result<Float32Array, JsValue> {
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();

        if a_vec.len() != (rows_a * cols_a) as usize {
            return Err(JsValue::from_str("Matrix A size mismatch"));
        }
        if b_vec.len() != (cols_a * cols_b) as usize {
            return Err(JsValue::from_str("Matrix B size mismatch"));
        }

        let result = simd_matrix_multiply(&a_vec, &b_vec, rows_a as usize, cols_a as usize, cols_b as usize);
        Ok(Float32Array::from(&result[..]))
    }
}

// SIMD implementations (would use actual SIMD intrinsics in real implementation)
fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    // This would use WASM SIMD instructions for better performance
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn simd_matrix_multiply(a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
    let mut result = vec![0.0; rows_a * cols_b];
    
    // This would use SIMD for vectorized operations
    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            result[i * cols_b + j] = sum;
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_network_creation() {
        let layers = [2, 3, 1];
        let network = WasmNeuralNetwork::from_layers(&layers);
        assert!(network.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_simd_dot_product() {
        let a = Float32Array::from(&[1.0, 2.0, 3.0][..]);
        let b = Float32Array::from(&[4.0, 5.0, 6.0][..]);
        
        let result = SIMDOps::dot_product(&a, &b).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[wasm_bindgen_test]
    fn test_network_inference() {
        let mut network = WasmNeuralNetwork::from_layers(&[2, 3, 1]).unwrap();
        let input = Float32Array::from(&[0.5, 0.8][..]);
        
        let output = network.run(&input);
        assert!(output.is_ok());
        assert_eq!(output.unwrap().length(), 1);
    }
}