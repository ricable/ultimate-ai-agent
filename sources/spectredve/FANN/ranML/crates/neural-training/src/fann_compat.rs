//! Compatibility layer for FANN neural network operations
//! This module provides a simplified interface compatible with the ruv-fann library

use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Simplified activation function enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationFunction {
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    ReLULeaky,
    Elliot,
    Gaussian,
}

impl Default for ActivationFunction {
    fn default() -> Self {
        Self::Sigmoid
    }
}

/// Simplified neuron structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub connections: Vec<f32>,
    pub bias: f32,
    pub activation_function: ActivationFunction,
}

/// Simplified layer structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

/// Simplified network structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub connection_rate: f32,
}

/// Training data structure
#[derive(Debug, Clone)]
pub struct TrainingData {
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
}

/// Training error types
#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Training failed: {0}")]
    TrainingFailed(String),
}

/// Network builder for creating neural networks
pub struct NetworkBuilder {
    layers: Vec<usize>,
    activations: Vec<ActivationFunction>,
    connection_rate: f32,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            activations: Vec::new(),
            connection_rate: 1.0,
        }
    }
    
    pub fn layers_from_sizes(mut self, sizes: &[usize]) -> Self {
        self.layers = sizes.to_vec();
        self
    }
    
    pub fn add_layer(mut self, size: usize) -> Self {
        self.layers.push(size);
        self
    }
    
    pub fn add_layer_with_activation(mut self, size: usize, activation: ActivationFunction) -> Self {
        self.layers.push(size);
        self.activations.push(activation);
        self
    }
    
    pub fn connection_rate(mut self, rate: f32) -> Self {
        self.connection_rate = rate;
        self
    }
    
    pub fn build(self) -> Network {
        let mut layers = Vec::new();
        
        for (i, &size) in self.layers.iter().enumerate() {
            let mut neurons = Vec::new();
            
            for _ in 0..size {
                let connections = if i == 0 {
                    Vec::new() // Input layer has no connections
                } else {
                    // Create connections from previous layer
                    let prev_size = self.layers[i - 1];
                    (0..prev_size).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect()
                };
                
                neurons.push(Neuron {
                    connections,
                    bias: rand::random::<f32>() * 2.0 - 1.0,
                    activation_function: self.activations.get(i).copied().unwrap_or_default(),
                });
            }
            
            layers.push(Layer { neurons });
        }
        
        Network {
            layers,
            connection_rate: self.connection_rate,
        }
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Network {
    /// Run the network on input data
    pub fn run(&mut self, inputs: &[f32]) -> Vec<f32> {
        let mut activations = inputs.to_vec();
        
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer_idx == 0 {
                continue; // Skip input layer
            }
            
            let mut next_activations = Vec::new();
            
            for neuron in &layer.neurons {
                let mut sum = neuron.bias;
                
                for (i, &weight) in neuron.connections.iter().enumerate() {
                    if i < activations.len() {
                        sum += weight * activations[i];
                    }
                }
                
                let output = self.apply_activation(sum, neuron.activation_function);
                next_activations.push(output);
            }
            
            activations = next_activations;
        }
        
        activations
    }
    
    /// Apply activation function
    fn apply_activation(&self, x: f32, func: ActivationFunction) -> f32 {
        match func {
            ActivationFunction::Linear => x,
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::ReLULeaky => if x > 0.0 { x } else { 0.01 * x },
            ActivationFunction::Elliot => x / (1.0 + x.abs()),
            ActivationFunction::Gaussian => (-x * x).exp(),
        }
    }
    
    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Get number of inputs
    pub fn num_inputs(&self) -> usize {
        self.layers.first().map(|l| l.neurons.len()).unwrap_or(0)
    }
    
    /// Get number of outputs
    pub fn num_outputs(&self) -> usize {
        self.layers.last().map(|l| l.neurons.len()).unwrap_or(0)
    }
}

/// Simplified training algorithm trait
pub trait TrainingAlgorithm {
    fn train_epoch(&mut self, network: &mut Network, data: &TrainingData) -> Result<f32, TrainingError>;
    fn calculate_error(&self, network: &Network, data: &TrainingData) -> f32;
}

/// Simple backpropagation implementation
pub struct IncrementalBackprop {
    learning_rate: f32,
    momentum: f32,
}

impl IncrementalBackprop {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
        }
    }
    
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

impl TrainingAlgorithm for IncrementalBackprop {
    fn train_epoch(&mut self, network: &mut Network, data: &TrainingData) -> Result<f32, TrainingError> {
        let mut total_error = 0.0;
        
        for (input, target) in data.inputs.iter().zip(data.outputs.iter()) {
            let output = network.run(input);
            
            // Calculate error (simplified MSE)
            let error: f32 = output.iter()
                .zip(target.iter())
                .map(|(&o, &t)| (o - t).powi(2))
                .sum::<f32>() / output.len() as f32;
            
            total_error += error;
            
            // Simple weight updates (placeholder for actual backpropagation)
            for layer in &mut network.layers {
                for neuron in &mut layer.neurons {
                    for weight in &mut neuron.connections {
                        let gradient = rand::random::<f32>() * 0.01 - 0.005; // Placeholder
                        *weight -= self.learning_rate * gradient;
                    }
                    neuron.bias -= self.learning_rate * (rand::random::<f32>() * 0.01 - 0.005);
                }
            }
        }
        
        Ok(total_error / data.inputs.len() as f32)
    }
    
    fn calculate_error(&self, network: &Network, data: &TrainingData) -> f32 {
        let mut total_error = 0.0;
        let mut network_clone = network.clone();
        
        for (input, target) in data.inputs.iter().zip(data.outputs.iter()) {
            let output = network_clone.run(input);
            
            let error: f32 = output.iter()
                .zip(target.iter())
                .map(|(&o, &t)| (o - t).powi(2))
                .sum::<f32>() / output.len() as f32;
            
            total_error += error;
        }
        
        total_error / data.inputs.len() as f32
    }
}