//! Expert definitions and micro-expert architecture

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use ndarray::{Array2, Array1};

/// Expert domain specializations based on Kimi-K2 analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertDomain {
    /// Logical reasoning and inference (10K parameters)
    Reasoning,
    /// Code generation and debugging (50K parameters)
    Coding,
    /// Natural language processing (25K parameters)
    Language,
    /// Tool usage and function calling (15K parameters)
    ToolUse,
    /// Mathematical computation (20K parameters)
    Mathematics,
    /// Long-context understanding (30K parameters)
    Context,
}

impl ExpertDomain {
    /// Get all available expert domains
    pub fn all_domains() -> Vec<Self> {
        vec![
            Self::Reasoning,
            Self::Coding,
            Self::Language,
            Self::ToolUse,
            Self::Mathematics,
            Self::Context,
        ]
    }

    /// Get the target parameter count for this domain
    pub fn target_parameters(&self) -> usize {
        match self {
            Self::Reasoning => 10_000,
            Self::Coding => 50_000,
            Self::Language => 25_000,
            Self::ToolUse => 15_000,
            Self::Mathematics => 20_000,
            Self::Context => 30_000,
        }
    }

    /// Get the expected input features for this domain
    pub fn input_features(&self) -> usize {
        match self {
            Self::Reasoning => 512,
            Self::Coding => 1024,
            Self::Language => 768,
            Self::ToolUse => 384,
            Self::Mathematics => 640,
            Self::Context => 2048,
        }
    }

    /// Get the output dimension for this domain
    pub fn output_dimension(&self) -> usize {
        match self {
            Self::Reasoning => 256,
            Self::Coding => 512,
            Self::Language => 384,
            Self::ToolUse => 192,
            Self::Mathematics => 320,
            Self::Context => 768,
        }
    }
}

/// Parameters for a micro-expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertParameters {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension  
    pub output_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Activation function
    pub activation: ActivationFunction,
    /// Learning rate for fine-tuning
    pub learning_rate: f32,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Target parameter count
    pub target_params: usize,
}

impl Default for ExpertParameters {
    fn default() -> Self {
        Self {
            input_dim: 512,
            output_dim: 256,
            hidden_dims: vec![256, 128],
            activation: ActivationFunction::ReLU,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            target_params: 10_000,
        }
    }
}

/// Activation functions for micro-experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    Swish,
}

/// Weights and biases for a micro-expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertWeights {
    /// Layer weights (layer_index -> weight_matrix)
    pub weights: HashMap<usize, Array2<f32>>,
    /// Layer biases (layer_index -> bias_vector)
    pub biases: HashMap<usize, Array1<f32>>,
    /// Total parameter count
    pub parameter_count: usize,
    /// Compression ratio from original expert
    pub compression_ratio: f32,
}

impl ExpertWeights {
    /// Create new expert weights
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            biases: HashMap::new(),
            parameter_count: 0,
            compression_ratio: 1.0,
        }
    }

    /// Add a layer's weights and biases
    pub fn add_layer(&mut self, layer_idx: usize, weights: Array2<f32>, biases: Array1<f32>) {
        let layer_params = weights.len() + biases.len();
        self.parameter_count += layer_params;
        self.weights.insert(layer_idx, weights);
        self.biases.insert(layer_idx, biases);
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.parameter_count * std::mem::size_of::<f32>()
    }

    /// Validate weight consistency
    pub fn validate(&self) -> Result<()> {
        // Check that dimensions are consistent
        for (layer_idx, weights) in &self.weights {
            if let Some(biases) = self.biases.get(layer_idx) {
                if weights.nrows() != biases.len() {
                    return Err(anyhow::anyhow!(
                        "Layer {} weight/bias dimension mismatch: {} vs {}",
                        layer_idx, weights.nrows(), biases.len()
                    ));
                }
            }
        }
        Ok(())
    }
}

/// A micro-expert extracted from Kimi-K2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroExpert {
    /// Unique expert identifier
    pub id: usize,
    /// Expert domain specialization
    pub domain: ExpertDomain,
    /// Network parameters
    pub parameters: ExpertParameters,
    /// Trained weights
    pub weights: ExpertWeights,
    /// Performance metrics
    pub metrics: ExpertMetrics,
    /// Metadata
    pub metadata: ExpertMetadata,
}

impl MicroExpert {
    /// Create a new micro-expert
    pub fn new(
        id: usize,
        domain: ExpertDomain,
        parameters: ExpertParameters,
        weights: ExpertWeights,
    ) -> Result<Self> {
        // Validate that parameters match domain expectations
        if parameters.target_params != domain.target_parameters() {
            tracing::warn!(
                "Parameter count mismatch for domain {:?}: expected {}, got {}",
                domain, domain.target_parameters(), parameters.target_params
            );
        }

        weights.validate()?;

        Ok(Self {
            id,
            domain,
            parameters,
            weights,
            metrics: ExpertMetrics::default(),
            metadata: ExpertMetadata::new(),
        })
    }

    /// Get the expert's memory footprint
    pub fn memory_footprint(&self) -> usize {
        self.weights.memory_usage()
    }

    /// Check if expert is WASM-compatible
    pub fn is_wasm_compatible(&self) -> bool {
        // Check if expert is small enough for WASM
        self.memory_footprint() < 100 * 1024 * 1024 && // < 100MB
        self.parameters.target_params <= 100_000 // <= 100K parameters
    }

    /// Get expert efficiency score
    pub fn efficiency_score(&self) -> f32 {
        // Balance between performance and size
        let performance_weight = 0.7;
        let size_weight = 0.3;
        
        performance_weight * self.metrics.accuracy + 
        size_weight * (1.0 - (self.memory_footprint() as f32 / (100.0 * 1024.0 * 1024.0)))
    }

    /// Predict output for given input (simplified interface)
    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>> {
        // This would interface with ruv-FANN or other neural network backend
        // For now, return a placeholder
        let output_dim = self.parameters.output_dim;
        Ok(vec![0.0; output_dim])
    }

    /// Get confidence score for prediction
    pub fn get_confidence(&self, input: &[f32]) -> Result<f32> {
        // Calculate prediction confidence
        // This would analyze the uncertainty of the prediction
        Ok(0.85) // Placeholder
    }
}

/// Performance metrics for a micro-expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMetrics {
    /// Accuracy on validation set
    pub accuracy: f32,
    /// Inference speed (ms)
    pub inference_speed_ms: f32,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Specialization score (how domain-specific)
    pub specialization_score: f32,
    /// Distillation quality (compared to original)
    pub distillation_quality: f32,
}

impl Default for ExpertMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            inference_speed_ms: 0.0,
            memory_usage: 0,
            specialization_score: 0.0,
            distillation_quality: 0.0,
        }
    }
}

/// Metadata for a micro-expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMetadata {
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Source expert IDs from original model
    pub source_experts: Vec<usize>,
    /// Training dataset used
    pub training_dataset: String,
    /// Version of distillation process
    pub distillation_version: String,
    /// Additional tags
    pub tags: Vec<String>,
}

impl ExpertMetadata {
    pub fn new() -> Self {
        Self {
            created_at: chrono::Utc::now(),
            source_experts: Vec::new(),
            training_dataset: "unknown".to_string(),
            distillation_version: "0.1.0".to_string(),
            tags: Vec::new(),
        }
    }
}

/// Map of all experts in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMap {
    /// All available experts
    pub experts: HashMap<usize, ExpertSpecification>,
    /// Domain to expert ID mapping
    pub domain_mapping: HashMap<ExpertDomain, Vec<usize>>,
    /// Total experts count
    pub total_experts: usize,
    /// Analysis metadata
    pub metadata: ExpertMapMetadata,
}

impl ExpertMap {
    /// Create expert map from analysis
    pub fn from_analysis(analysis: &crate::analysis::SpecializationAnalysis) -> Result<Self> {
        tracing::info!("Creating expert map from specialization analysis");
        
        let mut experts = HashMap::new();
        let mut domain_mapping = HashMap::new();
        let mut expert_id_counter = 0;
        
        // Initialize domain mapping
        for domain in ExpertDomain::all_domains() {
            domain_mapping.insert(domain, Vec::new());
        }
        
        // Process micro-expert mappings from analysis
        for mapping in &analysis.micro_expert_mappings {
            let expert_spec = ExpertSpecification {
                id: expert_id_counter,
                domain: mapping.target_domain.clone(),
                parameters: ExpertParameters {
                    input_dim: mapping.target_domain.input_features(),
                    output_dim: mapping.target_domain.output_dimension(),
                    hidden_dims: Self::determine_hidden_dims(&mapping.target_domain),
                    activation: ActivationFunction::ReLU,
                    learning_rate: 0.001,
                    dropout_rate: 0.1,
                    target_params: mapping.target_domain.target_parameters(),
                },
                source_info: SourceInfo {
                    original_expert_ids: mapping.source_experts.iter().map(|(_, id)| *id).collect(),
                    layer_indices: mapping.source_experts.iter().map(|(layer, _)| *layer).collect(),
                    parameter_reduction_ratio: mapping.parameter_reduction,
                    extraction_confidence: mapping.performance_retention,
                },
                extraction_difficulty: Self::determine_extraction_difficulty(mapping),
            };
            
            // Add to domain mapping
            domain_mapping.entry(mapping.target_domain.clone())
                .or_default()
                .push(expert_id_counter);
            
            experts.insert(expert_id_counter, expert_spec);
            expert_id_counter += 1;
        }
        
        // Add additional experts from domain clusters if needed
        for (domain, cluster) in &analysis.domain_clusters.clusters {
            if !cluster.is_empty() && domain_mapping.get(domain).map_or(0, |v| v.len()) == 0 {
                // Create a basic expert for domains that don't have mapped experts
                let expert_spec = ExpertSpecification {
                    id: expert_id_counter,
                    domain: domain.clone(),
                    parameters: ExpertParameters {
                        input_dim: domain.input_features(),
                        output_dim: domain.output_dimension(),
                        hidden_dims: Self::determine_hidden_dims(domain),
                        activation: ActivationFunction::ReLU,
                        learning_rate: 0.001,
                        dropout_rate: 0.1,
                        target_params: domain.target_parameters(),
                    },
                    source_info: SourceInfo {
                        original_expert_ids: cluster.iter().map(|(_, id)| *id).collect(),
                        layer_indices: cluster.iter().map(|(layer, _)| *layer).collect(),
                        parameter_reduction_ratio: 0.1, // 10:1 reduction estimate
                        extraction_confidence: analysis.domain_clusters.cluster_quality
                            .get(domain).copied().unwrap_or(0.7),
                    },
                    extraction_difficulty: ExtractionDifficulty::Medium,
                };
                
                domain_mapping.entry(domain.clone())
                    .or_default()
                    .push(expert_id_counter);
                
                experts.insert(expert_id_counter, expert_spec);
                expert_id_counter += 1;
            }
        }
        
        let expert_map = Self {
            experts,
            domain_mapping,
            total_experts: expert_id_counter,
            metadata: ExpertMapMetadata {
                created_at: analysis.metadata.analysis_date,
                source_model: "Kimi-K2".to_string(),
                analysis_version: analysis.metadata.analysis_version.clone(),
                quality_metrics: HashMap::new(),
            },
        };
        
        tracing::info!("Created expert map with {} experts across {} domains", 
                      expert_map.total_experts, expert_map.domain_mapping.len());
        
        Ok(expert_map)
    }
    
    /// Determine appropriate hidden layer dimensions for a domain
    fn determine_hidden_dims(domain: &ExpertDomain) -> Vec<usize> {
        match domain {
            ExpertDomain::Reasoning => vec![256, 128],
            ExpertDomain::Coding => vec![512, 256, 128],
            ExpertDomain::Language => vec![384, 192],
            ExpertDomain::ToolUse => vec![192, 96],
            ExpertDomain::Mathematics => vec![320, 160],
            ExpertDomain::Context => vec![768, 384, 192],
        }
    }
    
    /// Determine extraction difficulty based on mapping characteristics
    fn determine_extraction_difficulty(mapping: &crate::analysis::MicroExpertMapping) -> ExtractionDifficulty {
        let source_count = mapping.source_experts.len();
        let reduction_ratio = mapping.parameter_reduction;
        let performance_retention = mapping.performance_retention;
        
        // More sources = harder extraction
        // Higher compression = harder extraction  
        // Lower performance retention = harder extraction
        let difficulty_score = 
            (source_count as f32 * 0.1) +
            (reduction_ratio * 2.0) +
            ((1.0 - performance_retention) * 3.0);
        
        match difficulty_score {
            score if score < 0.5 => ExtractionDifficulty::Easy,
            score if score < 1.0 => ExtractionDifficulty::Medium,
            score if score < 2.0 => ExtractionDifficulty::Hard,
            _ => ExtractionDifficulty::VeryHard,
        }
    }

    /// Load expert map from file
    pub async fn load(path: &std::path::Path) -> Result<Self> {
        let content = tokio::fs::read_to_string(path).await?;
        let expert_map: Self = serde_json::from_str(&content)?;
        Ok(expert_map)
    }

    /// Get expert specification by ID
    pub fn get_expert(&self, expert_id: usize) -> Option<&ExpertSpecification> {
        self.experts.get(&expert_id)
    }

    /// Get experts for a specific domain
    pub fn get_domain_experts(&self, domain: &ExpertDomain) -> Vec<usize> {
        self.domain_mapping.get(domain).cloned().unwrap_or_default()
    }

    /// Add an expert to the map
    pub fn add_expert(&mut self, expert_spec: ExpertSpecification) {
        let expert_id = expert_spec.id;
        let domain = expert_spec.domain.clone();
        
        self.experts.insert(expert_id, expert_spec);
        self.domain_mapping.entry(domain).or_default().push(expert_id);
        self.total_experts = self.experts.len();
    }
}

/// Specification for an expert (before extraction)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertSpecification {
    /// Expert ID
    pub id: usize,
    /// Domain specialization
    pub domain: ExpertDomain,
    /// Target parameters
    pub parameters: ExpertParameters,
    /// Source information from original model
    pub source_info: SourceInfo,
    /// Estimated extraction difficulty
    pub extraction_difficulty: ExtractionDifficulty,
}

/// Information about source expert in original model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Original expert IDs that contribute to this micro-expert
    pub original_expert_ids: Vec<usize>,
    /// Layer indices in original model
    pub layer_indices: Vec<usize>,
    /// Estimated parameter reduction ratio
    pub parameter_reduction_ratio: f32,
    /// Confidence in extraction quality
    pub extraction_confidence: f32,
}

/// Difficulty level for expert extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtractionDifficulty {
    Easy,
    Medium,
    Hard,
    VeryHard,
}

/// Metadata for the expert map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertMapMetadata {
    /// Analysis timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Original model information
    pub source_model: String,
    /// Analysis version
    pub analysis_version: String,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_domain_parameters() {
        let reasoning = ExpertDomain::Reasoning;
        assert_eq!(reasoning.target_parameters(), 10_000);
        assert_eq!(reasoning.input_features(), 512);
    }

    #[test]
    fn test_expert_weights_validation() {
        let mut weights = ExpertWeights::new();
        
        // Add valid layer
        let weight_matrix = Array2::zeros((10, 5));
        let bias_vector = Array1::zeros(10);
        weights.add_layer(0, weight_matrix, bias_vector);
        
        assert!(weights.validate().is_ok());
        assert_eq!(weights.parameter_count, 60); // 10*5 + 10
    }

    #[test]
    fn test_micro_expert_creation() {
        let domain = ExpertDomain::Reasoning;
        let mut params = ExpertParameters::default();
        params.target_params = domain.target_parameters();
        
        let weights = ExpertWeights::new();
        
        let expert = MicroExpert::new(1, domain, params, weights);
        assert!(expert.is_ok());
    }
}