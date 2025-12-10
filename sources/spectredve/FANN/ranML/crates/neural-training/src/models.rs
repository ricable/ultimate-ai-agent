//! Neural network model definitions and architectures

use crate::fann_compat::{ActivationFunction, Network, NetworkBuilder};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Neural network architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkArchitecture {
    pub name: String,
    pub layer_sizes: Vec<usize>,
    pub activation_functions: Vec<ActivationFunction>,
    pub connection_rate: f32,
    pub bias: bool,
}

/// Collection of predefined network architectures for comparison
pub struct NetworkArchitectures;

impl NetworkArchitectures {
    /// Get all predefined architectures for telecom data
    pub fn get_all_architectures(input_size: usize, output_size: usize) -> Vec<NetworkArchitecture> {
        vec![
            Self::shallow_network(input_size, output_size),
            Self::deep_network(input_size, output_size),
            Self::wide_network(input_size, output_size),
            Self::residual_like_network(input_size, output_size),
            Self::bottleneck_network(input_size, output_size),
        ]
    }
    
    /// Shallow network: Input -> Hidden(32) -> Output
    pub fn shallow_network(input_size: usize, output_size: usize) -> NetworkArchitecture {
        NetworkArchitecture {
            name: "Shallow_Network".to_string(),
            layer_sizes: vec![input_size, 32, output_size],
            activation_functions: vec![
                ActivationFunction::Linear,    // Input layer
                ActivationFunction::ReLU,      // Hidden layer
                ActivationFunction::Linear,    // Output layer
            ],
            connection_rate: 1.0,
            bias: true,
        }
    }
    
    /// Deep network: Input -> 64 -> 32 -> 16 -> 8 -> Output
    pub fn deep_network(input_size: usize, output_size: usize) -> NetworkArchitecture {
        NetworkArchitecture {
            name: "Deep_Network".to_string(),
            layer_sizes: vec![input_size, 64, 32, 16, 8, output_size],
            activation_functions: vec![
                ActivationFunction::Linear,           // Input layer
                ActivationFunction::ReLU,             // Hidden layer 1
                ActivationFunction::ReLU,             // Hidden layer 2
                ActivationFunction::ReLU,             // Hidden layer 3
                ActivationFunction::Sigmoid,          // Hidden layer 4
                ActivationFunction::Linear,           // Output layer
            ],
            connection_rate: 1.0,
            bias: true,
        }
    }
    
    /// Wide network: Input -> 128 -> 64 -> Output
    pub fn wide_network(input_size: usize, output_size: usize) -> NetworkArchitecture {
        NetworkArchitecture {
            name: "Wide_Network".to_string(),
            layer_sizes: vec![input_size, 128, 64, output_size],
            activation_functions: vec![
                ActivationFunction::Linear,    // Input layer
                ActivationFunction::ReLU,      // Hidden layer 1
                ActivationFunction::ReLU,      // Hidden layer 2
                ActivationFunction::Linear,    // Output layer
            ],
            connection_rate: 1.0,
            bias: true,
        }
    }
    
    /// Residual-like network with varying activations
    pub fn residual_like_network(input_size: usize, output_size: usize) -> NetworkArchitecture {
        NetworkArchitecture {
            name: "Residual_Like_Network".to_string(),
            layer_sizes: vec![input_size, 64, 64, 32, output_size],
            activation_functions: vec![
                ActivationFunction::Linear,           // Input layer
                ActivationFunction::ReLU,             // Hidden layer 1
                ActivationFunction::ReLULeaky,        // Hidden layer 2
                ActivationFunction::Tanh,             // Hidden layer 3
                ActivationFunction::Linear,           // Output layer
            ],
            connection_rate: 1.0,
            bias: true,
        }
    }
    
    /// Bottleneck network: Input -> 16 -> 8 -> 16 -> Output
    pub fn bottleneck_network(input_size: usize, output_size: usize) -> NetworkArchitecture {
        NetworkArchitecture {
            name: "Bottleneck_Network".to_string(),
            layer_sizes: vec![input_size, 16, 8, 16, output_size],
            activation_functions: vec![
                ActivationFunction::Linear,    // Input layer
                ActivationFunction::Tanh,      // Encoder
                ActivationFunction::Sigmoid,   // Bottleneck
                ActivationFunction::ReLU,      // Decoder
                ActivationFunction::Linear,    // Output layer
            ],
            connection_rate: 1.0,
            bias: true,
        }
    }
}

/// Neural network model wrapper
#[derive(Debug, Clone)]
pub struct NeuralModel {
    pub name: String,
    pub network: Network,
    pub architecture: NetworkArchitecture,
    pub training_params: TrainingParameters,
}

/// Training parameters for neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParameters {
    pub learning_rate: f32,
    pub momentum: f32,
    pub max_epochs: usize,
    pub target_error: f32,
    pub batch_size: Option<usize>,
    pub weight_decay: f32,
    pub dropout_rate: f32,
}

impl Default for TrainingParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            max_epochs: 1000,
            target_error: 0.001,
            batch_size: None, // None means full batch
            weight_decay: 0.0001,
            dropout_rate: 0.0,
        }
    }
}

impl NeuralModel {
    /// Create a new neural model from architecture
    pub fn from_architecture(architecture: NetworkArchitecture) -> Result<Self> {
        let mut builder = NetworkBuilder::new();
        
        // Build network with specified layer sizes
        for (i, &size) in architecture.layer_sizes.iter().enumerate() {
            if i == 0 {
                builder = builder.add_layer(size); // Input layer
            } else {
                let activation = architecture.activation_functions
                    .get(i)
                    .unwrap_or(&ActivationFunction::ReLU);
                builder = builder.add_layer_with_activation(size, *activation);
            }
        }
        
        let network = builder
            .connection_rate(architecture.connection_rate)
            .build();
        
        Ok(Self {
            name: architecture.name.clone(),
            network,
            architecture,
            training_params: TrainingParameters::default(),
        })
    }
    
    /// Create model with custom training parameters
    pub fn with_training_params(mut self, params: TrainingParameters) -> Self {
        self.training_params = params;
        self
    }
    
    /// Get model summary
    pub fn summary(&self) -> ModelSummary {
        let mut total_parameters = 0;
        let mut layer_info = Vec::new();
        
        for (i, layer) in self.network.layers.iter().enumerate() {
            let neurons = layer.neurons.len();
            let connections = if i == 0 { 0 } else {
                layer.neurons.first()
                    .map(|n| n.connections.len())
                    .unwrap_or(0) * neurons
            };
            let activation = self.architecture.activation_functions
                .get(i)
                .unwrap_or(&ActivationFunction::Linear);
            
            layer_info.push(LayerInfo {
                layer_index: i,
                neurons,
                connections,
                activation: *activation,
            });
            
            total_parameters += connections + neurons; // weights + biases
        }
        
        ModelSummary {
            name: self.name.clone(),
            total_parameters,
            layer_info,
            training_params: self.training_params.clone(),
        }
    }
    
    /// Predict on input data
    pub fn predict(&mut self, inputs: &[f32]) -> Vec<f32> {
        self.network.run(inputs)
    }
    
    /// Predict on batch of inputs
    pub fn predict_batch(&mut self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter()
            .map(|input| self.network.run(input))
            .collect()
    }
}

/// Model summary information
#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    pub name: String,
    pub total_parameters: usize,
    pub layer_info: Vec<LayerInfo>,
    pub training_params: TrainingParameters,
}

/// Layer information for model summary
#[derive(Debug, Clone, Serialize)]
pub struct LayerInfo {
    pub layer_index: usize,
    pub neurons: usize,
    pub connections: usize,
    pub activation: ActivationFunction,
}

/// Model registry for managing multiple models
#[derive(Debug)]
pub struct ModelRegistry {
    models: HashMap<String, NeuralModel>,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }
    
    /// Register a model
    pub fn register_model(&mut self, model: NeuralModel) {
        self.models.insert(model.name.clone(), model);
    }
    
    /// Get model by name
    pub fn get_model(&self, name: &str) -> Option<&NeuralModel> {
        self.models.get(name)
    }
    
    /// Get mutable model by name
    pub fn get_model_mut(&mut self, name: &str) -> Option<&mut NeuralModel> {
        self.models.get_mut(name)
    }
    
    /// Get all model names
    pub fn model_names(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
    
    /// Create models for all predefined architectures
    pub fn create_all_models(input_size: usize, output_size: usize) -> Result<Self> {
        let mut registry = Self::new();
        
        let architectures = NetworkArchitectures::get_all_architectures(input_size, output_size);
        
        for architecture in architectures {
            let model = NeuralModel::from_architecture(architecture)?;
            registry.register_model(model);
        }
        
        Ok(registry)
    }
    
    /// Get model summaries
    pub fn get_summaries(&self) -> Vec<ModelSummary> {
        self.models.values()
            .map(|model| model.summary())
            .collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Hyperparameter configuration for grid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfig {
    pub learning_rates: Vec<f32>,
    pub momentum_values: Vec<f32>,
    pub batch_sizes: Vec<Option<usize>>,
    pub weight_decay_values: Vec<f32>,
    pub max_epochs: usize,
    pub target_error: f32,
}

impl Default for HyperparameterConfig {
    fn default() -> Self {
        Self {
            learning_rates: vec![0.001, 0.01, 0.1],
            momentum_values: vec![0.0, 0.5, 0.9],
            batch_sizes: vec![None, Some(32), Some(64)],
            weight_decay_values: vec![0.0, 0.0001, 0.001],
            max_epochs: 1000,
            target_error: 0.001,
        }
    }
}

impl HyperparameterConfig {
    /// Generate all combinations of hyperparameters
    pub fn generate_combinations(&self) -> Vec<TrainingParameters> {
        let mut combinations = Vec::new();
        
        for &learning_rate in &self.learning_rates {
            for &momentum in &self.momentum_values {
                for &batch_size in &self.batch_sizes {
                    for &weight_decay in &self.weight_decay_values {
                        combinations.push(TrainingParameters {
                            learning_rate,
                            momentum,
                            max_epochs: self.max_epochs,
                            target_error: self.target_error,
                            batch_size,
                            weight_decay,
                            dropout_rate: 0.0,
                        });
                    }
                }
            }
        }
        
        combinations
    }
}