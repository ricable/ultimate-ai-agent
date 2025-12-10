//! Expert routing and feature extraction for micro-expert selection

use crate::expert::{ExpertDomain, MicroExpert, ExpertWeights};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use ndarray::{Array1, Array2};

/// Expert routing engine for selecting appropriate micro-experts
#[derive(Debug, Clone)]
pub struct ExpertRoutingEngine {
    /// Routing network for expert selection
    pub routing_network: RoutingNetwork,
    /// Expert profiles for each available expert
    pub expert_profiles: HashMap<usize, ExpertProfile>,
    /// Performance history for route optimization
    pub performance_history: PerformanceTracker,
    /// Feature extraction pipeline
    pub feature_extractor: FeatureExtractor,
    /// Routing configuration
    pub config: RoutingConfig,
}

impl ExpertRoutingEngine {
    /// Create a new routing engine
    pub fn new(config: RoutingConfig) -> Result<Self> {
        let routing_network = RoutingNetwork::new(&config.network_config)?;
        let feature_extractor = FeatureExtractor::new(&config.feature_config)?;
        
        Ok(Self {
            routing_network,
            expert_profiles: HashMap::new(),
            performance_history: PerformanceTracker::new(),
            feature_extractor,
            config,
        })
    }

    /// Register a micro-expert with the routing engine
    pub fn register_expert(&mut self, expert: &MicroExpert) -> Result<()> {
        let profile = ExpertProfile::from_expert(expert)?;
        self.expert_profiles.insert(expert.id, profile);
        
        // Update routing network if needed
        if self.expert_profiles.len() % 10 == 0 {
            self.retrain_routing_network()?;
        }
        
        tracing::debug!("Registered expert {} for domain {:?}", expert.id, expert.domain);
        Ok(())
    }

    /// Select experts for a given request context
    pub fn select_experts(&self, context: &RequestContext) -> Result<Vec<ExpertId>> {
        tracing::debug!("Selecting experts for request: {}", context.prompt.chars().take(50).collect::<String>());
        
        // Extract features from the request
        let features = self.feature_extractor.extract_features(context)?;
        
        // Get expert scores from routing network
        let expert_scores = self.routing_network.predict(&features)?;
        
        // Apply performance-based adjustments
        let adjusted_scores = self.adjust_for_performance(&expert_scores)?;
        
        // Select top-k experts
        let selected_experts = self.top_k_selection(&adjusted_scores, self.config.max_experts_per_request)?;
        
        // Apply diversity constraints
        let diverse_experts = self.apply_diversity_constraints(&selected_experts, context)?;
        
        tracing::debug!("Selected {} experts: {:?}", diverse_experts.len(), diverse_experts);
        Ok(diverse_experts)
    }

    /// Update performance history after expert execution
    pub fn update_performance(&mut self, execution_result: &ExecutionResult) -> Result<()> {
        self.performance_history.record_execution(execution_result)?;
        
        // Update expert profiles based on performance
        for expert_id in &execution_result.experts_used {
            if let Some(profile) = self.expert_profiles.get_mut(expert_id) {
                profile.update_performance(&execution_result.metrics)?;
            }
        }
        
        // Trigger retraining if performance degrades
        if self.performance_history.should_retrain(&self.config.retraining_config) {
            tracing::info!("Performance degradation detected, retraining routing network");
            self.retrain_routing_network()?;
        }
        
        Ok(())
    }

    /// Retrain the routing network based on accumulated performance data
    fn retrain_routing_network(&mut self) -> Result<()> {
        tracing::info!("Retraining routing network with {} historical executions", 
                      self.performance_history.execution_count());

        let training_data = self.performance_history.get_training_data()?;
        self.routing_network.retrain(&training_data)?;
        
        tracing::info!("Routing network retraining completed");
        Ok(())
    }

    /// Adjust expert scores based on historical performance
    fn adjust_for_performance(&self, base_scores: &ExpertScores) -> Result<ExpertScores> {
        let mut adjusted_scores = base_scores.clone();
        
        for (expert_id, score) in adjusted_scores.scores.iter_mut() {
            if let Some(profile) = self.expert_profiles.get(expert_id) {
                let performance_multiplier = profile.get_performance_multiplier();
                *score *= performance_multiplier;
            }
        }
        
        Ok(adjusted_scores)
    }

    /// Select top-k experts from scores
    fn top_k_selection(&self, scores: &ExpertScores, k: usize) -> Result<Vec<ExpertId>> {
        let mut expert_score_pairs: Vec<_> = scores.scores.iter().collect();
        expert_score_pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let selected = expert_score_pairs.into_iter()
            .take(k)
            .filter(|(_, score)| **score > self.config.min_activation_threshold)
            .map(|(id, _)| *id)
            .collect();
        
        Ok(selected)
    }

    /// Apply diversity constraints to expert selection
    fn apply_diversity_constraints(
        &self,
        experts: &[ExpertId],
        context: &RequestContext,
    ) -> Result<Vec<ExpertId>> {
        if !self.config.enforce_diversity {
            return Ok(experts.to_vec());
        }

        let mut diverse_experts = Vec::new();
        let mut used_domains = std::collections::HashSet::new();
        
        for &expert_id in experts {
            if let Some(profile) = self.expert_profiles.get(&expert_id) {
                // Enforce domain diversity if configured
                if self.config.max_experts_per_domain > 0 {
                    let domain_count = diverse_experts.iter()
                        .filter_map(|&id| self.expert_profiles.get(&id))
                        .filter(|p| p.primary_domain == profile.primary_domain)
                        .count();
                    
                    if domain_count >= self.config.max_experts_per_domain {
                        continue;
                    }
                }
                
                diverse_experts.push(expert_id);
                used_domains.insert(profile.primary_domain.clone());
                
                if diverse_experts.len() >= self.config.max_experts_per_request {
                    break;
                }
            }
        }
        
        Ok(diverse_experts)
    }

    /// Get routing statistics
    pub fn get_routing_stats(&self) -> RoutingStats {
        RoutingStats {
            total_experts: self.expert_profiles.len(),
            total_requests: self.performance_history.execution_count(),
            average_experts_per_request: self.performance_history.average_experts_per_request(),
            domain_distribution: self.get_domain_distribution(),
            performance_metrics: self.performance_history.get_aggregate_metrics(),
        }
    }

    fn get_domain_distribution(&self) -> HashMap<ExpertDomain, usize> {
        let mut distribution = HashMap::new();
        for profile in self.expert_profiles.values() {
            *distribution.entry(profile.primary_domain.clone()).or_insert(0) += 1;
        }
        distribution
    }
}

/// Routing network for expert selection
#[derive(Debug, Clone)]
pub struct RoutingNetwork {
    /// Network layers
    pub layers: Vec<RoutingLayer>,
    /// Network configuration
    pub config: NetworkConfig,
    /// Training state
    pub training_state: NetworkTrainingState,
}

impl RoutingNetwork {
    /// Create a new routing network
    pub fn new(config: &NetworkConfig) -> Result<Self> {
        let mut layers = Vec::new();
        
        // Input layer
        let input_size = config.input_features;
        let mut current_size = input_size;
        
        // Hidden layers
        for &hidden_size in &config.hidden_sizes {
            layers.push(RoutingLayer::new(current_size, hidden_size)?);
            current_size = hidden_size;
        }
        
        // Output layer (one output per expert)
        layers.push(RoutingLayer::new(current_size, config.num_experts)?);
        
        Ok(Self {
            layers,
            config: config.clone(),
            training_state: NetworkTrainingState::new(),
        })
    }

    /// Predict expert scores for given features
    pub fn predict(&self, features: &FeatureVector) -> Result<ExpertScores> {
        let mut activations = features.values.clone();
        
        // Forward pass through all layers
        for layer in &self.layers {
            activations = layer.forward(&activations)?;
        }
        
        // Apply softmax to get probabilities
        let probabilities = self.softmax(&activations);
        
        // Convert to expert scores
        let mut scores = HashMap::new();
        for (i, &prob) in probabilities.iter().enumerate() {
            scores.insert(i, prob);
        }
        
        Ok(ExpertScores { scores })
    }

    /// Retrain the network with new data
    pub fn retrain(&mut self, training_data: &RoutingTrainingData) -> Result<()> {
        tracing::info!("Retraining routing network with {} samples", training_data.samples.len());
        
        let mut optimizer = RoutingOptimizer::new(&self.config.optimizer_config)?;
        
        for epoch in 0..self.config.training_epochs {
            let mut epoch_loss = 0.0;
            
            for batch in training_data.iter_batches(self.config.batch_size) {
                let loss = self.train_batch(batch, &mut optimizer)?;
                epoch_loss += loss;
            }
            
            let avg_loss = epoch_loss / training_data.num_batches(self.config.batch_size) as f32;
            
            if epoch % 10 == 0 {
                tracing::debug!("Epoch {}: Loss = {:.4}", epoch, avg_loss);
            }
            
            // Early stopping
            if avg_loss < self.config.convergence_threshold {
                tracing::info!("Converged at epoch {} with loss {:.4}", epoch, avg_loss);
                break;
            }
        }
        
        self.training_state.last_training = chrono::Utc::now();
        self.training_state.training_count += 1;
        
        Ok(())
    }

    fn softmax(&self, logits: &Array1<f32>) -> Array1<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Array1<f32> = logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        exp_logits / sum_exp
    }

    fn train_batch(&mut self, batch: &[RoutingTrainingSample], optimizer: &mut RoutingOptimizer) -> Result<f32> {
        // Implementation would include actual backpropagation
        // For now, return a placeholder loss
        Ok(0.5)
    }
}

/// Individual layer in the routing network
#[derive(Debug, Clone)]
pub struct RoutingLayer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: LayerActivation,
}

impl RoutingLayer {
    pub fn new(input_size: usize, output_size: usize) -> Result<Self> {
        // Initialize weights with Xavier/Glorot initialization
        let scale = (2.0 / (input_size + output_size) as f32).sqrt();
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            scale * (rand::random::<f32>() - 0.5) * 2.0
        });
        let biases = Array1::zeros(output_size);
        
        Ok(Self {
            weights,
            biases,
            activation: LayerActivation::ReLU,
        })
    }

    pub fn forward(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let linear_output = self.weights.dot(input) + &self.biases;
        let activated_output = self.activation.apply(&linear_output);
        Ok(activated_output)
    }
}

/// Activation functions for routing layers
#[derive(Debug, Clone)]
pub enum LayerActivation {
    ReLU,
    Tanh,
    Sigmoid,
    Linear,
}

impl LayerActivation {
    pub fn apply(&self, input: &Array1<f32>) -> Array1<f32> {
        match self {
            Self::ReLU => input.mapv(|x| x.max(0.0)),
            Self::Tanh => input.mapv(|x| x.tanh()),
            Self::Sigmoid => input.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Self::Linear => input.clone(),
        }
    }
}

/// Feature extraction pipeline
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Text processing pipeline
    pub text_processor: TextProcessor,
    /// Context analysis pipeline
    pub context_analyzer: ContextAnalyzer,
    /// Feature combination strategy
    pub combination_strategy: FeatureCombinationStrategy,
    /// Feature extraction configuration
    pub config: FeatureConfig,
}

impl FeatureExtractor {
    pub fn new(config: &FeatureConfig) -> Result<Self> {
        Ok(Self {
            text_processor: TextProcessor::new(&config.text_config)?,
            context_analyzer: ContextAnalyzer::new(&config.context_config)?,
            combination_strategy: config.combination_strategy.clone(),
            config: config.clone(),
        })
    }

    /// Extract features from a request context
    pub fn extract_features(&self, context: &RequestContext) -> Result<FeatureVector> {
        // Extract text features
        let text_features = self.text_processor.extract_features(&context.prompt)?;
        
        // Extract context features
        let context_features = self.context_analyzer.extract_features(context)?;
        
        // Combine features
        let combined_features = self.combination_strategy.combine(
            &text_features,
            &context_features,
        )?;
        
        Ok(FeatureVector {
            values: combined_features,
            feature_names: self.get_feature_names(),
        })
    }

    fn get_feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        names.extend(self.text_processor.get_feature_names());
        names.extend(self.context_analyzer.get_feature_names());
        names
    }
}

/// Text processing for feature extraction
#[derive(Debug, Clone)]
pub struct TextProcessor {
    pub tokenizer: SimpleTokenizer,
    pub embedding_model: Option<EmbeddingModel>,
    pub config: TextConfig,
}

impl TextProcessor {
    pub fn new(config: &TextConfig) -> Result<Self> {
        Ok(Self {
            tokenizer: SimpleTokenizer::new(),
            embedding_model: None, // Would load actual embedding model
            config: config.clone(),
        })
    }

    pub fn extract_features(&self, text: &str) -> Result<Array1<f32>> {
        let mut features = Vec::new();
        
        // Basic text statistics
        features.push(text.len() as f32);
        features.push(text.split_whitespace().count() as f32);
        features.push(text.chars().filter(|c| c.is_alphabetic()).count() as f32);
        features.push(text.chars().filter(|c| c.is_numeric()).count() as f32);
        features.push(text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32);
        
        // Domain-specific patterns
        features.extend(self.extract_domain_patterns(text));
        
        // Complexity metrics
        features.extend(self.extract_complexity_metrics(text));
        
        // Pad or truncate to fixed size
        features.resize(self.config.feature_size, 0.0);
        
        Ok(Array1::from_vec(features))
    }

    fn extract_domain_patterns(&self, text: &str) -> Vec<f32> {
        let mut patterns = Vec::new();
        
        // Code-like patterns
        patterns.push(if text.contains("def ") || text.contains("function") || text.contains("class ") { 1.0 } else { 0.0 });
        patterns.push(if text.contains("if ") || text.contains("while ") || text.contains("for ") { 1.0 } else { 0.0 });
        patterns.push(if text.contains("import ") || text.contains("include") { 1.0 } else { 0.0 });
        
        // Math patterns
        patterns.push(if text.contains("=") || text.contains("+") || text.contains("*") { 1.0 } else { 0.0 });
        patterns.push(if text.contains("solve") || text.contains("calculate") { 1.0 } else { 0.0 });
        
        // Question patterns
        patterns.push(if text.contains("?") { 1.0 } else { 0.0 });
        patterns.push(if text.starts_with("What") || text.starts_with("How") || text.starts_with("Why") { 1.0 } else { 0.0 });
        
        // Tool usage patterns
        patterns.push(if text.contains("call") || text.contains("use") || text.contains("API") { 1.0 } else { 0.0 });
        
        patterns
    }

    fn extract_complexity_metrics(&self, text: &str) -> Vec<f32> {
        vec![
            // Sentence count
            text.split('.').count() as f32,
            // Average word length
            text.split_whitespace().map(|w| w.len()).sum::<usize>() as f32 / text.split_whitespace().count().max(1) as f32,
            // Vocabulary richness (simplified)
            text.split_whitespace().collect::<std::collections::HashSet<_>>().len() as f32,
        ]
    }

    pub fn get_feature_names(&self) -> Vec<String> {
        (0..self.config.feature_size).map(|i| format!("text_feature_{}", i)).collect()
    }
}

/// Context analysis for feature extraction
#[derive(Debug, Clone)]
pub struct ContextAnalyzer {
    pub config: ContextConfig,
}

impl ContextAnalyzer {
    pub fn new(config: &ContextConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn extract_features(&self, context: &RequestContext) -> Result<Array1<f32>> {
        let mut features = Vec::new();
        
        // Time-based features
        features.push(context.timestamp.hour() as f32 / 24.0);
        features.push(context.timestamp.minute() as f32 / 60.0);
        
        // User context features
        features.push(if context.user_id.is_some() { 1.0 } else { 0.0 });
        features.push(context.session_length as f32);
        features.push(context.conversation_turns as f32);
        
        // Priority and urgency
        features.push(context.priority as u8 as f32 / 3.0); // Assuming 3 priority levels
        features.push(if context.requires_tools { 1.0 } else { 0.0 });
        
        // Context length features
        features.push(context.conversation_history.len() as f32);
        features.push(context.available_tools.len() as f32);
        
        // Pad to fixed size
        features.resize(self.config.feature_size, 0.0);
        
        Ok(Array1::from_vec(features))
    }

    pub fn get_feature_names(&self) -> Vec<String> {
        (0..self.config.feature_size).map(|i| format!("context_feature_{}", i)).collect()
    }
}

/// Simple tokenizer for text processing
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    // Placeholder for actual tokenizer implementation
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        Self {}
    }
}

/// Embedding model placeholder
#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    // Placeholder for actual embedding model
}

/// Feature combination strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureCombinationStrategy {
    Concatenate,
    WeightedSum,
    AttentionBased,
    CrossProduct,
}

impl FeatureCombinationStrategy {
    pub fn combine(&self, text_features: &Array1<f32>, context_features: &Array1<f32>) -> Result<Array1<f32>> {
        match self {
            Self::Concatenate => {
                let mut combined = text_features.to_vec();
                combined.extend(context_features.iter());
                Ok(Array1::from_vec(combined))
            },
            Self::WeightedSum => {
                // Simple weighted average
                let text_weight = 0.7;
                let context_weight = 0.3;
                let min_len = text_features.len().min(context_features.len());
                let combined: Vec<f32> = (0..min_len)
                    .map(|i| text_weight * text_features[i] + context_weight * context_features[i])
                    .collect();
                Ok(Array1::from_vec(combined))
            },
            _ => {
                // Fallback to concatenation for now
                let mut combined = text_features.to_vec();
                combined.extend(context_features.iter());
                Ok(Array1::from_vec(combined))
            }
        }
    }
}

/// Expert profile for routing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertProfile {
    /// Expert identifier
    pub expert_id: usize,
    /// Primary domain
    pub primary_domain: ExpertDomain,
    /// Secondary domains
    pub secondary_domains: Vec<ExpertDomain>,
    /// Performance metrics
    pub performance_metrics: ProfilePerformanceMetrics,
    /// Usage statistics
    pub usage_stats: UsageStatistics,
    /// Specialization patterns
    pub specialization_patterns: SpecializationPatterns,
}

impl ExpertProfile {
    pub fn from_expert(expert: &MicroExpert) -> Result<Self> {
        Ok(Self {
            expert_id: expert.id,
            primary_domain: expert.domain.clone(),
            secondary_domains: Vec::new(), // Would be derived from analysis
            performance_metrics: ProfilePerformanceMetrics::from_expert_metrics(&expert.metrics),
            usage_stats: UsageStatistics::new(),
            specialization_patterns: SpecializationPatterns::analyze_expert(expert)?,
        })
    }

    pub fn update_performance(&mut self, metrics: &ExecutionMetrics) -> Result<()> {
        self.performance_metrics.update(metrics);
        self.usage_stats.record_usage();
        Ok(())
    }

    pub fn get_performance_multiplier(&self) -> f32 {
        // Calculate performance-based multiplier
        let accuracy_component = self.performance_metrics.recent_accuracy;
        let speed_component = 1.0 - (self.performance_metrics.recent_latency / 100.0).min(1.0);
        let reliability_component = self.performance_metrics.success_rate;
        
        (accuracy_component + speed_component + reliability_component) / 3.0
    }
}

/// Performance metrics for expert profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilePerformanceMetrics {
    pub recent_accuracy: f32,
    pub recent_latency: f32,
    pub success_rate: f32,
    pub total_executions: usize,
    pub average_confidence: f32,
}

impl ProfilePerformanceMetrics {
    pub fn from_expert_metrics(metrics: &ExpertMetrics) -> Self {
        Self {
            recent_accuracy: metrics.accuracy,
            recent_latency: metrics.inference_speed_ms,
            success_rate: 1.0, // Would be tracked from actual executions
            total_executions: 0,
            average_confidence: 0.8, // Default confidence
        }
    }

    pub fn update(&mut self, metrics: &ExecutionMetrics) {
        // Exponential moving average for recent metrics
        let alpha = 0.1;
        self.recent_accuracy = alpha * metrics.accuracy + (1.0 - alpha) * self.recent_accuracy;
        self.recent_latency = alpha * metrics.latency_ms + (1.0 - alpha) * self.recent_latency;
        
        self.total_executions += 1;
        if metrics.success {
            self.success_rate = alpha + (1.0 - alpha) * self.success_rate;
        } else {
            self.success_rate = (1.0 - alpha) * self.success_rate;
        }
    }
}

/// Usage statistics for experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStatistics {
    pub total_calls: usize,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
    pub usage_frequency: f32,
    pub peak_usage_hour: Option<u32>,
}

impl UsageStatistics {
    pub fn new() -> Self {
        Self {
            total_calls: 0,
            last_used: None,
            usage_frequency: 0.0,
            peak_usage_hour: None,
        }
    }

    pub fn record_usage(&mut self) {
        self.total_calls += 1;
        self.last_used = Some(chrono::Utc::now());
        // Update frequency and peak hour calculations
    }
}

/// Specialization patterns for experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializationPatterns {
    pub preferred_input_patterns: Vec<String>,
    pub typical_output_styles: Vec<String>,
    pub context_preferences: Vec<String>,
    pub anti_patterns: Vec<String>, // Patterns this expert performs poorly on
}

impl SpecializationPatterns {
    pub fn analyze_expert(expert: &MicroExpert) -> Result<Self> {
        // This would analyze the expert's training data and performance
        // to identify specialization patterns
        Ok(Self {
            preferred_input_patterns: Vec::new(),
            typical_output_styles: Vec::new(),
            context_preferences: Vec::new(),
            anti_patterns: Vec::new(),
        })
    }
}

/// Request context for routing decisions
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub prompt: String,
    pub conversation_history: Vec<String>,
    pub user_id: Option<String>,
    pub session_length: usize,
    pub conversation_turns: usize,
    pub priority: RequestPriority,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub requires_tools: bool,
    pub available_tools: Vec<String>,
    pub performance_requirements: PerformanceRequirements,
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPriority {
    Low = 1,
    Medium = 2,
    High = 3,
}

/// Performance requirements for requests
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    pub max_latency_ms: Option<f32>,
    pub min_accuracy: Option<f32>,
    pub max_memory_mb: Option<f32>,
}

/// Feature vector extracted from request
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub values: Array1<f32>,
    pub feature_names: Vec<String>,
}

/// Expert scores from routing network
#[derive(Debug, Clone)]
pub struct ExpertScores {
    pub scores: HashMap<ExpertId, f32>,
}

/// Expert identifier
pub type ExpertId = usize;

/// Execution result for performance tracking
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub experts_used: Vec<ExpertId>,
    pub metrics: ExecutionMetrics,
    pub success: bool,
    pub output_quality: f32,
}

/// Execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    pub latency_ms: f32,
    pub accuracy: f32,
    pub memory_usage_mb: f32,
    pub success: bool,
    pub confidence: f32,
}

/// Performance tracker for routing optimization
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    execution_history: Vec<ExecutionResult>,
    aggregate_metrics: AggregateMetrics,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            execution_history: Vec::new(),
            aggregate_metrics: AggregateMetrics::new(),
        }
    }

    pub fn record_execution(&mut self, result: &ExecutionResult) -> Result<()> {
        self.execution_history.push(result.clone());
        self.aggregate_metrics.update(result);
        
        // Keep only recent history to manage memory
        if self.execution_history.len() > 10000 {
            self.execution_history.drain(0..1000);
        }
        
        Ok(())
    }

    pub fn execution_count(&self) -> usize {
        self.execution_history.len()
    }

    pub fn average_experts_per_request(&self) -> f32 {
        if self.execution_history.is_empty() {
            return 0.0;
        }
        
        let total_experts: usize = self.execution_history.iter()
            .map(|r| r.experts_used.len())
            .sum();
        
        total_experts as f32 / self.execution_history.len() as f32
    }

    pub fn should_retrain(&self, config: &RetrainingConfig) -> bool {
        if self.execution_history.len() < config.min_samples_for_retraining {
            return false;
        }
        
        // Check if performance has degraded
        let recent_accuracy = self.get_recent_accuracy(config.recent_window_size);
        recent_accuracy < config.performance_threshold
    }

    pub fn get_recent_accuracy(&self, window_size: usize) -> f32 {
        if self.execution_history.is_empty() {
            return 0.0;
        }
        
        let start_idx = self.execution_history.len().saturating_sub(window_size);
        let recent_executions = &self.execution_history[start_idx..];
        
        let total_accuracy: f32 = recent_executions.iter()
            .map(|r| r.metrics.accuracy)
            .sum();
        
        total_accuracy / recent_executions.len() as f32
    }

    pub fn get_training_data(&self) -> Result<RoutingTrainingData> {
        // Convert execution history to training samples
        let samples = self.execution_history.iter()
            .map(|r| RoutingTrainingSample {
                features: Array1::zeros(100), // Placeholder - would extract actual features
                expert_scores: r.experts_used.iter().map(|&id| (id, 1.0)).collect(),
                success: r.success,
                quality_score: r.output_quality,
            })
            .collect();
        
        Ok(RoutingTrainingData { samples })
    }

    pub fn get_aggregate_metrics(&self) -> &AggregateMetrics {
        &self.aggregate_metrics
    }
}

/// Aggregate performance metrics
#[derive(Debug, Clone)]
pub struct AggregateMetrics {
    pub average_accuracy: f32,
    pub average_latency: f32,
    pub success_rate: f32,
    pub total_requests: usize,
}

impl AggregateMetrics {
    pub fn new() -> Self {
        Self {
            average_accuracy: 0.0,
            average_latency: 0.0,
            success_rate: 0.0,
            total_requests: 0,
        }
    }

    pub fn update(&mut self, result: &ExecutionResult) {
        let alpha = 1.0 / (self.total_requests + 1) as f32;
        
        self.average_accuracy = alpha * result.metrics.accuracy + (1.0 - alpha) * self.average_accuracy;
        self.average_latency = alpha * result.metrics.latency_ms + (1.0 - alpha) * self.average_latency;
        
        if result.success {
            self.success_rate = alpha + (1.0 - alpha) * self.success_rate;
        } else {
            self.success_rate = (1.0 - alpha) * self.success_rate;
        }
        
        self.total_requests += 1;
    }
}

/// Training data for routing network
#[derive(Debug, Clone)]
pub struct RoutingTrainingData {
    pub samples: Vec<RoutingTrainingSample>,
}

impl RoutingTrainingData {
    pub fn iter_batches(&self, batch_size: usize) -> impl Iterator<Item = &[RoutingTrainingSample]> {
        self.samples.chunks(batch_size)
    }

    pub fn num_batches(&self, batch_size: usize) -> usize {
        (self.samples.len() + batch_size - 1) / batch_size
    }
}

/// Individual training sample for routing network
#[derive(Debug, Clone)]
pub struct RoutingTrainingSample {
    pub features: Array1<f32>,
    pub expert_scores: HashMap<ExpertId, f32>,
    pub success: bool,
    pub quality_score: f32,
}

/// Routing optimizer
#[derive(Debug)]
pub struct RoutingOptimizer {
    config: OptimizerConfig,
    state: OptimizerState,
}

impl RoutingOptimizer {
    pub fn new(config: &OptimizerConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            state: OptimizerState::new(),
        })
    }
}

/// Optimizer state for routing network training
#[derive(Debug)]
pub struct OptimizerState {
    pub step_count: usize,
    pub momentum_buffers: HashMap<String, Array2<f32>>,
}

impl OptimizerState {
    pub fn new() -> Self {
        Self {
            step_count: 0,
            momentum_buffers: HashMap::new(),
        }
    }
}

/// Configuration structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub network_config: NetworkConfig,
    pub feature_config: FeatureConfig,
    pub retraining_config: RetrainingConfig,
    pub max_experts_per_request: usize,
    pub max_experts_per_domain: usize,
    pub min_activation_threshold: f32,
    pub enforce_diversity: bool,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            network_config: NetworkConfig::default(),
            feature_config: FeatureConfig::default(),
            retraining_config: RetrainingConfig::default(),
            max_experts_per_request: 3,
            max_experts_per_domain: 2,
            min_activation_threshold: 0.1,
            enforce_diversity: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub input_features: usize,
    pub hidden_sizes: Vec<usize>,
    pub num_experts: usize,
    pub training_epochs: usize,
    pub batch_size: usize,
    pub convergence_threshold: f32,
    pub optimizer_config: OptimizerConfig,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            input_features: 200,
            hidden_sizes: vec![128, 64],
            num_experts: 50,
            training_epochs: 100,
            batch_size: 32,
            convergence_threshold: 0.001,
            optimizer_config: OptimizerConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub text_config: TextConfig,
    pub context_config: ContextConfig,
    pub combination_strategy: FeatureCombinationStrategy,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            text_config: TextConfig::default(),
            context_config: ContextConfig::default(),
            combination_strategy: FeatureCombinationStrategy::Concatenate,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    pub feature_size: usize,
    pub max_sequence_length: usize,
    pub enable_embeddings: bool,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            feature_size: 100,
            max_sequence_length: 512,
            enable_embeddings: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub feature_size: usize,
    pub max_history_length: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            feature_size: 50,
            max_history_length: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainingConfig {
    pub min_samples_for_retraining: usize,
    pub performance_threshold: f32,
    pub recent_window_size: usize,
    pub retraining_frequency_hours: usize,
}

impl Default for RetrainingConfig {
    fn default() -> Self {
        Self {
            min_samples_for_retraining: 1000,
            performance_threshold: 0.8,
            recent_window_size: 100,
            retraining_frequency_hours: 24,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
        }
    }
}

/// Network training state
#[derive(Debug, Clone)]
pub struct NetworkTrainingState {
    pub last_training: chrono::DateTime<chrono::Utc>,
    pub training_count: usize,
    pub best_validation_loss: f32,
}

impl NetworkTrainingState {
    pub fn new() -> Self {
        Self {
            last_training: chrono::Utc::now(),
            training_count: 0,
            best_validation_loss: f32::INFINITY,
        }
    }
}

/// Routing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingStats {
    pub total_experts: usize,
    pub total_requests: usize,
    pub average_experts_per_request: f32,
    pub domain_distribution: HashMap<ExpertDomain, usize>,
    pub performance_metrics: AggregateMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_config_default() {
        let config = RoutingConfig::default();
        assert_eq!(config.max_experts_per_request, 3);
        assert!(config.enforce_diversity);
    }

    #[test]
    fn test_feature_extraction() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(&config).unwrap();
        assert_eq!(extractor.config.text_config.feature_size, 100);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new();
        assert_eq!(tracker.execution_count(), 0);
        assert_eq!(tracker.average_experts_per_request(), 0.0);
    }
}