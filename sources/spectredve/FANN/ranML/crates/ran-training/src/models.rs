//! Neural network models and architectures for telecom training

use crate::error::{TrainingError, TrainingResult};
use ruv_fann::{Network, NetworkBuilder, ActivationFunction, TrainingAlgorithm};
use serde::{Deserialize, Serialize};

/// Neural network model with training configuration
#[derive(Debug, Clone)]
pub struct NeuralModel {
    /// Model name/identifier
    pub name: String,
    /// Network architecture
    pub architecture: ModelArchitecture,
    /// Training parameters
    pub training_params: TrainingParameters,
    /// The actual neural network
    pub network: Network<f32>,
}

/// Model architecture specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Layer sizes: [input, hidden1, hidden2, ..., output]
    pub layers: Vec<usize>,
    /// Activation functions for each layer
    pub activations: Vec<ActivationFunction>,
    /// Bias values for each layer
    pub biases: Vec<f32>,
}

impl ModelArchitecture {
    /// Create architecture from layer sizes with default activations
    pub fn from_layers(layers: &[usize]) -> Self {
        let mut activations = vec![ActivationFunction::Sigmoid; layers.len()];
        if !activations.is_empty() {
            // Use linear activation for output layer
            let last_index = activations.len() - 1;
            activations[last_index] = ActivationFunction::Linear;
        }
        
        let biases = vec![1.0; layers.len()];
        
        Self {
            layers: layers.to_vec(),
            activations,
            biases,
        }
    }
    
    /// Create architecture with custom activations
    pub fn with_activations(layers: Vec<usize>, activations: Vec<ActivationFunction>) -> TrainingResult<Self> {
        if layers.len() != activations.len() {
            return Err(TrainingError::InvalidArchitecture(
                "Number of layers must match number of activation functions".into()
            ));
        }
        
        let biases = vec![1.0; layers.len()];
        
        Ok(Self {
            layers,
            activations,
            biases,
        })
    }
    
    /// Get input layer size
    pub fn input_size(&self) -> usize {
        self.layers.first().copied().unwrap_or(0)
    }
    
    /// Get output layer size
    pub fn output_size(&self) -> usize {
        self.layers.last().copied().unwrap_or(0)
    }
    
    /// Get total number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
    
    /// Get hidden layers (excluding input and output)
    pub fn hidden_layers(&self) -> &[usize] {
        if self.layers.len() <= 2 {
            &[]
        } else {
            &self.layers[1..self.layers.len()-1]
        }
    }
}

/// Training parameters for neural network models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParameters {
    /// Learning rate (0.0 - 1.0)
    pub learning_rate: f32,
    /// Maximum training epochs
    pub max_epochs: u32,
    /// Target error for convergence
    pub target_error: f32,
    /// Momentum for gradient descent
    pub momentum: f32,
    /// Training algorithm
    pub algorithm: TrainingAlgorithm,
}

impl Default for TrainingParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_epochs: 1000,
            target_error: 0.001,
            momentum: 0.9,
            algorithm: TrainingAlgorithm::RProp,
        }
    }
}

impl NeuralModel {
    /// Create a new neural model
    pub fn new(
        name: String,
        architecture: ModelArchitecture,
        training_params: TrainingParameters,
    ) -> TrainingResult<Self> {
        let network = Self::build_network(&architecture)?;
        
        Ok(Self {
            name,
            architecture,
            training_params,
            network,
        })
    }
    
    /// Create a model with default parameters
    pub fn with_architecture(name: String, architecture: ModelArchitecture) -> TrainingResult<Self> {
        Self::new(name, architecture, TrainingParameters::default())
    }
    
    /// Build the neural network from architecture specification
    fn build_network(arch: &ModelArchitecture) -> TrainingResult<Network<f32>> {
        if arch.layers.len() < 2 {
            return Err(TrainingError::InvalidArchitecture(
                "Architecture must have at least 2 layers (input and output)".into()
            ));
        }
        
        let mut builder = NetworkBuilder::<f32>::new()
            .input_layer(arch.layers[0]);
        
        // Add hidden layers
        for i in 1..arch.layers.len()-1 {
            let layer_size = arch.layers[i];
            let activation = arch.activations.get(i).copied()
                .unwrap_or(ActivationFunction::Sigmoid);
            let bias = arch.biases.get(i).copied().unwrap_or(1.0);
            
            builder = builder.hidden_layer_with_activation(layer_size, activation, bias);
        }
        
        // Add output layer
        let output_size = *arch.layers.last().unwrap();
        let output_activation = arch.activations.last().copied()
            .unwrap_or(ActivationFunction::Linear);
        let output_bias = arch.biases.last().copied().unwrap_or(1.0);
        
        builder = builder.output_layer_with_activation(output_size, output_activation, output_bias);
        
        Ok(builder.build())
    }
    
    /// Update training parameters
    pub fn with_training_params(mut self, params: TrainingParameters) -> Self {
        self.training_params = params;
        self
    }
    
    /// Make a prediction with the model
    pub fn predict(&mut self, inputs: &[f32]) -> Vec<f32> {
        self.network.run(inputs)
    }
    
    /// Get model parameter count (approximate)
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        
        for i in 0..self.architecture.layers.len()-1 {
            let current_layer = self.architecture.layers[i];
            let next_layer = self.architecture.layers[i+1];
            
            // Weights + biases
            count += current_layer * next_layer + next_layer;
        }
        
        count
    }
    
    /// Get model complexity score (parameters per layer)
    pub fn complexity_score(&self) -> f32 {
        let param_count = self.parameter_count();
        let layer_count = self.architecture.layer_count();
        
        if layer_count == 0 {
            0.0
        } else {
            param_count as f32 / layer_count as f32
        }
    }
    
    /// Clone the model with a new name
    pub fn clone_with_name(&self, new_name: String) -> TrainingResult<Self> {
        Self::new(new_name, self.architecture.clone(), self.training_params.clone())
    }
    
    /// Save model architecture to JSON
    pub fn save_architecture<P: AsRef<std::path::Path>>(&self, path: P) -> TrainingResult<()> {
        let json = serde_json::to_string_pretty(&self.architecture)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    /// Load model architecture from JSON
    pub fn load_architecture<P: AsRef<std::path::Path>>(path: P) -> TrainingResult<ModelArchitecture> {
        let content = std::fs::read_to_string(path)?;
        let architecture = serde_json::from_str(&content)?;
        Ok(architecture)
    }
}

/// Model factory for creating common architectures
pub struct ModelFactory;

impl ModelFactory {
    /// Create a shallow network (good for simple patterns)
    pub fn create_shallow(input_size: usize, output_size: usize) -> TrainingResult<NeuralModel> {
        let architecture = ModelArchitecture::from_layers(&[
            input_size,
            input_size / 2 + 2,
            output_size
        ]);
        
        NeuralModel::new(
            "shallow_model".to_string(),
            architecture,
            TrainingParameters::default()
        )
    }
    
    /// Create a deep network (good for complex patterns)
    pub fn create_deep(input_size: usize, output_size: usize) -> TrainingResult<NeuralModel> {
        let hidden1 = input_size * 2 / 3;
        let hidden2 = hidden1 * 2 / 3;
        let hidden3 = hidden2 * 2 / 3;
        
        let architecture = ModelArchitecture::from_layers(&[
            input_size,
            hidden1,
            hidden2,
            hidden3,
            output_size
        ]);
        
        NeuralModel::new(
            "deep_model".to_string(),
            architecture,
            TrainingParameters::default()
        )
    }
    
    /// Create a wide network (good for diverse features)
    pub fn create_wide(input_size: usize, output_size: usize) -> TrainingResult<NeuralModel> {
        let hidden_size = input_size * 3 / 2;
        
        let architecture = ModelArchitecture::from_layers(&[
            input_size,
            hidden_size,
            hidden_size,
            output_size
        ]);
        
        NeuralModel::new(
            "wide_model".to_string(),
            architecture,
            TrainingParameters::default()
        )
    }
    
    /// Create a pyramid network (decreasing layer sizes)
    pub fn create_pyramid(input_size: usize, output_size: usize) -> TrainingResult<NeuralModel> {
        let mut layers = vec![input_size];
        let mut current_size = input_size;
        
        while current_size > output_size * 2 {
            current_size = current_size * 2 / 3;
            layers.push(current_size);
        }
        
        layers.push(output_size);
        
        let architecture = ModelArchitecture::from_layers(&layers);
        
        NeuralModel::new(
            "pyramid_model".to_string(),
            architecture,
            TrainingParameters::default()
        )
    }
    
    /// Create a telecom-optimized network
    pub fn create_telecom_optimized() -> TrainingResult<NeuralModel> {
        // Optimized for 21 telecom features -> 1 quality score
        let architecture = ModelArchitecture::from_layers(&[21, 16, 12, 8, 1]);
        
        let params = TrainingParameters {
            learning_rate: 0.05, // Conservative for stability
            max_epochs: 2000,
            target_error: 0.0001, // High precision
            momentum: 0.95,
            algorithm: TrainingAlgorithm::RProp,
        };
        
        NeuralModel::new(
            "telecom_optimized".to_string(),
            architecture,
            params
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_architecture_creation() {
        let arch = ModelArchitecture::from_layers(&[10, 8, 6, 1]);
        assert_eq!(arch.layers, vec![10, 8, 6, 1]);
        assert_eq!(arch.input_size(), 10);
        assert_eq!(arch.output_size(), 1);
        assert_eq!(arch.layer_count(), 4);
        assert_eq!(arch.hidden_layers(), &[8, 6]);
    }
    
    #[test]
    fn test_neural_model_creation() {
        let arch = ModelArchitecture::from_layers(&[5, 3, 1]);
        let model = NeuralModel::new(
            "test_model".to_string(),
            arch,
            TrainingParameters::default()
        );
        
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.name, "test_model");
    }
    
    #[test]
    fn test_parameter_count() {
        let arch = ModelArchitecture::from_layers(&[3, 2, 1]);
        let model = NeuralModel::new(
            "test".to_string(),
            arch,
            TrainingParameters::default()
        ).unwrap();
        
        // 3->2: 3*2 + 2 = 8 parameters
        // 2->1: 2*1 + 1 = 3 parameters
        // Total: 11 parameters
        assert_eq!(model.parameter_count(), 11);
    }
    
    #[test]
    fn test_model_factory() {
        let shallow = ModelFactory::create_shallow(10, 1).unwrap();
        assert_eq!(shallow.architecture.layer_count(), 3);
        
        let deep = ModelFactory::create_deep(10, 1).unwrap();
        assert!(deep.architecture.layer_count() >= 4);
        
        let telecom = ModelFactory::create_telecom_optimized().unwrap();
        assert_eq!(telecom.architecture.input_size(), 21);
        assert_eq!(telecom.architecture.output_size(), 1);
    }
}