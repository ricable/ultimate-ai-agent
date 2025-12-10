//! Neural Network Configuration Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: u32,
    pub hidden_layers: Vec<usize>,
    pub activation_function: String,
    pub dropout_rate: f32,
    pub weight_decay: f32,
    pub early_stopping: bool,
    pub early_stopping_patience: u32,
    pub validation_split: f32,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            hidden_layers: vec![64, 32, 16],
            activation_function: "relu".to_string(),
            dropout_rate: 0.2,
            weight_decay: 0.0001,
            early_stopping: true,
            early_stopping_patience: 10,
            validation_split: 0.2,
        }
    }
}

impl NeuralConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err("learning_rate must be between 0.0 and 1.0".to_string());
        }
        
        if self.batch_size == 0 {
            return Err("batch_size must be greater than 0".to_string());
        }
        
        if self.epochs == 0 {
            return Err("epochs must be greater than 0".to_string());
        }
        
        if self.hidden_layers.is_empty() {
            return Err("hidden_layers cannot be empty".to_string());
        }
        
        if self.hidden_layers.iter().any(|&size| size == 0) {
            return Err("all hidden layer sizes must be greater than 0".to_string());
        }
        
        let valid_activations = ["relu", "sigmoid", "tanh", "leaky_relu", "swish"];
        if !valid_activations.contains(&self.activation_function.as_str()) {
            return Err(format!("Invalid activation function: {}", self.activation_function));
        }
        
        if !(0.0..=1.0).contains(&self.dropout_rate) {
            return Err("dropout_rate must be between 0.0 and 1.0".to_string());
        }
        
        if self.weight_decay < 0.0 {
            return Err("weight_decay must be non-negative".to_string());
        }
        
        if !(0.0..1.0).contains(&self.validation_split) {
            return Err("validation_split must be between 0.0 and 1.0".to_string());
        }
        
        if self.early_stopping_patience == 0 {
            return Err("early_stopping_patience must be greater than 0".to_string());
        }
        
        Ok(())
    }
    
    pub fn development() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 16,
            epochs: 50,
            hidden_layers: vec![32, 16],
            activation_function: "relu".to_string(),
            dropout_rate: 0.3,
            weight_decay: 0.001,
            early_stopping: true,
            early_stopping_patience: 5,
            validation_split: 0.3,
        }
    }
    
    pub fn production() -> Self {
        Self {
            learning_rate: 0.0001,
            batch_size: 64,
            epochs: 200,
            hidden_layers: vec![128, 64, 32, 16],
            activation_function: "swish".to_string(),
            dropout_rate: 0.1,
            weight_decay: 0.00001,
            early_stopping: true,
            early_stopping_patience: 20,
            validation_split: 0.1,
        }
    }
}