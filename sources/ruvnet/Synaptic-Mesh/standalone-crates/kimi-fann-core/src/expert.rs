//! Micro-Expert Neural Network Implementation
//!
//! This module implements the core KimiMicroExpert struct, which represents
//! a specialized neural network optimized for specific tasks.

use crate::*;
use synaptic_neural_wasm::{NeuralNetwork, Layer, Activation};
use ndarray::{Array1, Array2};
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Expert domain categorization
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertDomain {
    /// Logical reasoning and problem-solving
    Reasoning,
    /// Code generation, debugging, and analysis
    Coding,
    /// Natural language processing
    Language,
    /// Mathematical computation and reasoning
    Mathematics,
    /// Tool usage and API interaction
    ToolUse,
    /// Context understanding and synthesis
    Context,
    /// Task planning and decomposition
    Planning,
    /// Creative writing and content generation
    Creativity,
    /// Data analysis and interpretation
    DataAnalysis,
    /// Decision making and evaluation
    DecisionMaking,
}

impl ExpertDomain {
    /// Get the default specializations for this domain
    pub fn default_specializations(&self) -> Vec<Specialization> {
        match self {
            ExpertDomain::Reasoning => vec![
                Specialization::LogicalInference,
                Specialization::DecisionMaking,
            ],
            ExpertDomain::Coding => vec![
                Specialization::CodeGeneration,
                Specialization::CodeDebugging,
            ],
            ExpertDomain::Language => vec![
                Specialization::LanguageUnderstanding,
                Specialization::LanguageGeneration,
            ],
            ExpertDomain::Mathematics => vec![
                Specialization::MathematicalReasoning,
                Specialization::DataAnalysis,
            ],
            ExpertDomain::ToolUse => vec![
                Specialization::ToolUsage,
                Specialization::ApiInteraction,
            ],
            ExpertDomain::Context => vec![
                Specialization::ContextUnderstanding,
                Specialization::ContextSynthesis,
            ],
            ExpertDomain::Planning => vec![
                Specialization::TaskPlanning,
            ],
            ExpertDomain::Creativity => vec![
                Specialization::CreativeWriting,
            ],
            ExpertDomain::DataAnalysis => vec![
                Specialization::DataAnalysis,
            ],
            ExpertDomain::DecisionMaking => vec![
                Specialization::DecisionMaking,
            ],
        }
    }
}

/// Expert configuration for micro-network creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    /// Expert identification
    pub id: ExpertId,
    /// Domain of expertise
    pub domain: ExpertDomain,
    /// Specific specialization
    pub specialization: Specialization,
    /// Network architecture configuration
    pub architecture: NetworkArchitecture,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
}

/// Network architecture specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkArchitecture {
    /// Input layer size
    pub input_size: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Output layer size
    pub output_size: usize,
    /// Activation functions for each layer
    pub activations: Vec<Activation>,
    /// Dropout rates (optional)
    pub dropout_rates: Vec<f32>,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Regularization parameter
    pub regularization: f32,
}

/// Performance thresholds for expert validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Minimum accuracy threshold
    pub min_accuracy: f32,
    /// Maximum inference time in milliseconds
    pub max_inference_time: f32,
    /// Maximum memory usage in bytes
    pub max_memory_usage: usize,
    /// Minimum confidence threshold
    pub min_confidence: f32,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_accuracy: 0.8,
            max_inference_time: 100.0,
            max_memory_usage: 10 * 1024 * 1024, // 10MB
            min_confidence: 0.7,
        }
    }
}

/// Core micro-expert implementation
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiMicroExpert {
    /// Unique identifier
    id: ExpertId,
    /// Domain of expertise
    domain: ExpertDomain,
    /// Specific specialization
    specialization: Specialization,
    /// Neural network instance
    #[serde(skip)]
    network: Option<NeuralNetwork>,
    /// Network weights (serializable)
    weights: Vec<f32>,
    /// Network architecture
    architecture: NetworkArchitecture,
    /// Parameter count
    parameter_count: usize,
    /// Confidence threshold
    confidence_threshold: f32,
    /// Performance metrics
    metrics: PerformanceMetrics,
    /// Creation timestamp
    created_at: f64,
    /// Last used timestamp
    last_used: f64,
}

#[wasm_bindgen]
impl KimiMicroExpert {
    /// Create a new micro-expert with the given configuration
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> std::result::Result<KimiMicroExpert, JsValue> {
        let config: ExpertConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

        let mut expert = KimiMicroExpert {
            id: config.id,
            domain: config.domain,
            specialization: config.specialization.clone(),
            network: None,
            weights: Vec::new(),
            architecture: config.architecture.clone(),
            parameter_count: 0,
            confidence_threshold: config.performance_thresholds.min_confidence,
            metrics: PerformanceMetrics::new(),
            created_at: Utils::now(),
            last_used: Utils::now(),
        };

        expert.initialize_network(&config.architecture)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(expert)
    }

    /// Get the expert ID
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> ExpertId {
        self.id
    }

    /// Get the domain as a string
    #[wasm_bindgen(getter)]
    pub fn domain(&self) -> String {
        format!("{:?}", self.domain)
    }

    /// Get the specialization as a string
    #[wasm_bindgen(getter)]
    pub fn specialization(&self) -> String {
        format!("{:?}", self.specialization)
    }

    /// Get the parameter count
    #[wasm_bindgen(getter)]
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    /// Get the confidence threshold
    #[wasm_bindgen(getter)]
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    /// Perform inference with the expert
    #[wasm_bindgen]
    pub fn predict(&mut self, input: &[f32]) -> std::result::Result<String, JsValue> {
        let start_time = Utils::now();
        
        self.last_used = start_time;
        
        let network = self.network.as_ref()
            .ok_or_else(|| JsValue::from_str("Network not initialized"))?;

        let result = network.predict(input)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let inference_time = Utils::now() - start_time;
        
        // Update metrics
        let memory_usage = self.estimate_memory_usage();
        self.metrics.update(inference_time as f32, true, memory_usage);

        // Parse the JSON result to add confidence information
        let mut output: Vec<f32> = serde_json::from_str(&result)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Calculate confidence based on output distribution
        let confidence = self.calculate_confidence(&output);
        
        // Create enhanced result with metadata
        let enhanced_result = ExpertResult {
            expert_id: self.id,
            domain: self.domain,
            output,
            confidence,
            inference_time: inference_time as f32,
            memory_usage,
            timestamp: Utils::now(),
        };

        serde_json::to_string(&enhanced_result)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the current performance metrics
    #[wasm_bindgen]
    pub fn get_metrics(&self) -> std::result::Result<String, JsValue> {
        serde_json::to_string(&self.metrics)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Check if the expert is suitable for the given context
    #[wasm_bindgen]
    pub fn is_suitable_for(&self, context_json: &str) -> std::result::Result<bool, JsValue> {
        let context: RequestContext = serde_json::from_str(context_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid context: {}", e)))?;

        let is_domain_match = context.required_capabilities.contains(&self.domain);
        let meets_performance = self.metrics.confidence_score >= context.performance_requirements.min_confidence;
        let meets_timing = self.metrics.avg_inference_time <= context.performance_requirements.max_inference_time;

        Ok(is_domain_match && meets_performance && meets_timing)
    }

    /// Get the expert's memory usage
    #[wasm_bindgen]
    pub fn memory_usage(&self) -> usize {
        self.estimate_memory_usage()
    }

    /// Serialize the expert to JSON
    #[wasm_bindgen]
    pub fn to_json(&self) -> std::result::Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create an expert from JSON
    #[wasm_bindgen]
    pub fn from_json(json: &str) -> std::result::Result<KimiMicroExpert, JsValue> {
        let mut expert: KimiMicroExpert = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

        // Reinitialize the network from saved weights
        expert.restore_network()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(expert)
    }
}

impl KimiMicroExpert {
    /// Initialize the neural network with the given architecture
    fn initialize_network(&mut self, arch: &NetworkArchitecture) -> Result<()> {
        let mut network = NeuralNetwork::new();
        
        // Add input layer implicitly handled by first hidden layer
        let mut prev_size = arch.input_size;
        
        // Add hidden layers
        for (i, &hidden_size) in arch.hidden_layers.iter().enumerate() {
            let activation = arch.activations.get(i).copied().unwrap_or(Activation::ReLU);
            let layer = Layer::dense(prev_size, hidden_size).with_activation(activation);
            network.add_layer(layer);
            prev_size = hidden_size;
        }
        
        // Add output layer
        let output_activation = arch.activations.last().copied().unwrap_or(Activation::Linear);
        let output_layer = Layer::dense(prev_size, arch.output_size).with_activation(output_activation);
        network.add_layer(output_layer);
        
        // Calculate parameter count
        self.parameter_count = self.calculate_parameter_count(arch);
        
        // Store the network
        self.network = Some(network);
        
        // Extract and store weights for serialization
        self.extract_weights()?;
        
        Ok(())
    }

    /// Calculate the total parameter count
    fn calculate_parameter_count(&self, arch: &NetworkArchitecture) -> usize {
        let mut count = 0;
        let mut prev_size = arch.input_size;
        
        // Count hidden layer parameters
        for &hidden_size in &arch.hidden_layers {
            count += prev_size * hidden_size + hidden_size; // weights + biases
            prev_size = hidden_size;
        }
        
        // Count output layer parameters
        count += prev_size * arch.output_size + arch.output_size;
        
        count
    }

    /// Extract weights from the network for serialization
    fn extract_weights(&mut self) -> Result<()> {
        // This is a simplified version - in a real implementation,
        // you'd extract the actual weights from the neural network
        self.weights = vec![0.0; self.parameter_count];
        Ok(())
    }

    /// Restore the network from saved weights
    fn restore_network(&mut self) -> Result<()> {
        if self.network.is_none() {
            self.initialize_network(&self.architecture.clone())?;
        }
        
        // In a real implementation, you'd restore the actual weights
        // For now, we just ensure the network is properly initialized
        Ok(())
    }

    /// Calculate confidence based on output distribution
    fn calculate_confidence(&self, output: &[f32]) -> f32 {
        if output.is_empty() {
            return 0.0;
        }

        // For classification tasks, use max probability
        if output.len() > 1 {
            let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = output.iter().map(|x| x.exp()).sum();
            (max_val.exp() / sum).min(1.0).max(0.0)
        } else {
            // For regression tasks, use inverse of variance or other metrics
            // For now, return a fixed confidence
            0.8
        }
    }

    /// Estimate memory usage of this expert
    fn estimate_memory_usage(&self) -> usize {
        // Base struct size
        let base_size = std::mem::size_of::<Self>();
        
        // Weights size
        let weights_size = self.weights.len() * std::mem::size_of::<f32>();
        
        // Network size (estimated)
        let network_size = self.parameter_count * 8; // Rough estimate
        
        base_size + weights_size + network_size
    }
}

/// Result from expert inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertResult {
    /// Expert that generated this result
    pub expert_id: ExpertId,
    /// Domain of the expert
    pub domain: ExpertDomain,
    /// Output values
    pub output: Vec<f32>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Inference time in milliseconds
    pub inference_time: f32,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Timestamp of the result
    pub timestamp: f64,
}

/// Expert registry for managing multiple experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRegistry {
    experts: HashMap<ExpertId, KimiMicroExpert>,
    domain_index: HashMap<ExpertDomain, Vec<ExpertId>>,
    specialization_index: HashMap<Specialization, Vec<ExpertId>>,
}

impl ExpertRegistry {
    /// Create a new expert registry
    pub fn new() -> Self {
        Self {
            experts: HashMap::new(),
            domain_index: HashMap::new(),
            specialization_index: HashMap::new(),
        }
    }

    /// Register a new expert
    pub fn register_expert(&mut self, expert: KimiMicroExpert) -> Result<()> {
        let id = expert.id;
        let domain = expert.domain;
        let specialization = expert.specialization.clone();

        // Add to main registry
        self.experts.insert(id, expert);

        // Update domain index
        self.domain_index.entry(domain).or_default().push(id);

        // Update specialization index
        self.specialization_index.entry(specialization).or_default().push(id);

        Ok(())
    }

    /// Get an expert by ID
    pub fn get_expert(&self, id: ExpertId) -> Option<&KimiMicroExpert> {
        self.experts.get(&id)
    }

    /// Get experts by domain
    pub fn get_experts_by_domain(&self, domain: ExpertDomain) -> Vec<&KimiMicroExpert> {
        self.domain_index
            .get(&domain)
            .map(|ids| ids.iter().filter_map(|id| self.experts.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get experts by specialization
    pub fn get_experts_by_specialization(&self, specialization: &Specialization) -> Vec<&KimiMicroExpert> {
        self.specialization_index
            .get(specialization)
            .map(|ids| ids.iter().filter_map(|id| self.experts.get(id)).collect())
            .unwrap_or_default()
    }

    /// Find the best expert for a given context
    pub fn find_best_expert(&self, context: &RequestContext) -> Option<ExpertId> {
        let mut best_expert = None;
        let mut best_score = 0.0;

        for domain in &context.required_capabilities {
            if let Some(expert_ids) = self.domain_index.get(domain) {
                for &expert_id in expert_ids {
                    if let Some(expert) = self.experts.get(&expert_id) {
                        let score = self.calculate_expert_score(expert, context);
                        if score > best_score {
                            best_score = score;
                            best_expert = Some(expert_id);
                        }
                    }
                }
            }
        }

        best_expert
    }

    /// Calculate suitability score for an expert given a context
    fn calculate_expert_score(&self, expert: &KimiMicroExpert, context: &RequestContext) -> f32 {
        let mut score = 0.0;

        // Domain match bonus
        if context.required_capabilities.contains(&expert.domain) {
            score += 0.4;
        }

        // Performance score
        score += expert.metrics.confidence_score * 0.3;

        // Timing score
        let timing_score = if expert.metrics.avg_inference_time <= context.performance_requirements.max_inference_time {
            1.0 - (expert.metrics.avg_inference_time / context.performance_requirements.max_inference_time).min(1.0)
        } else {
            0.0
        };
        score += timing_score * 0.2;

        // Experience score
        let experience_score = (expert.metrics.execution_count as f32 / 100.0).min(1.0);
        score += experience_score * 0.1;

        score.min(1.0)
    }
}

impl Default for ExpertRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        let config = ExpertConfig {
            id: 1,
            domain: ExpertDomain::Reasoning,
            specialization: Specialization::LogicalInference,
            architecture: NetworkArchitecture {
                input_size: 100,
                hidden_layers: vec![64, 32],
                output_size: 10,
                activations: vec![Activation::ReLU, Activation::ReLU, Activation::Linear],
                dropout_rates: vec![],
            },
            training_config: TrainingConfig {
                learning_rate: 0.001,
                batch_size: 32,
                epochs: 100,
                regularization: 0.01,
            },
            performance_thresholds: PerformanceThresholds::default(),
        };

        let config_json = serde_json::to_string(&config).unwrap();
        let expert = KimiMicroExpert::new(&config_json).unwrap();

        assert_eq!(expert.id(), 1);
        assert_eq!(expert.domain(), "Reasoning");
        assert!(expert.parameter_count() > 0);
    }

    #[test]
    fn test_expert_registry() {
        let mut registry = ExpertRegistry::new();
        
        let config = ExpertConfig {
            id: 1,
            domain: ExpertDomain::Coding,
            specialization: Specialization::CodeGeneration,
            architecture: NetworkArchitecture {
                input_size: 50,
                hidden_layers: vec![32],
                output_size: 5,
                activations: vec![Activation::ReLU, Activation::Linear],
                dropout_rates: vec![],
            },
            training_config: TrainingConfig {
                learning_rate: 0.001,
                batch_size: 16,
                epochs: 50,
                regularization: 0.01,
            },
            performance_thresholds: PerformanceThresholds::default(),
        };

        let config_json = serde_json::to_string(&config).unwrap();
        let expert = KimiMicroExpert::new(&config_json).unwrap();
        
        registry.register_expert(expert).unwrap();
        
        assert!(registry.get_expert(1).is_some());
        
        let coding_experts = registry.get_experts_by_domain(ExpertDomain::Coding);
        assert_eq!(coding_experts.len(), 1);
    }

    #[test]
    fn test_parameter_count_calculation() {
        let arch = NetworkArchitecture {
            input_size: 10,
            hidden_layers: vec![5, 3],
            output_size: 2,
            activations: vec![Activation::ReLU, Activation::ReLU, Activation::Linear],
            dropout_rates: vec![],
        };

        let config = ExpertConfig {
            id: 1,
            domain: ExpertDomain::Mathematics,
            specialization: Specialization::MathematicalReasoning,
            architecture: arch,
            training_config: TrainingConfig {
                learning_rate: 0.001,
                batch_size: 32,
                epochs: 100,
                regularization: 0.01,
            },
            performance_thresholds: PerformanceThresholds::default(),
        };

        let config_json = serde_json::to_string(&config).unwrap();
        let expert = KimiMicroExpert::new(&config_json).unwrap();

        // Expected: (10*5 + 5) + (5*3 + 3) + (3*2 + 2) = 55 + 18 + 8 = 81
        assert_eq!(expert.parameter_count(), 81);
    }
}