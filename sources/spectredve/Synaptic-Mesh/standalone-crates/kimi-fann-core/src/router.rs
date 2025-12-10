//! Expert Router System
//!
//! This module implements the expert routing system that determines which
//! micro-experts should be activated for a given request.

use crate::*;
use synaptic_neural_wasm::{NeuralNetwork, Layer, Activation};
use ndarray::Array1;
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Expert routing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Select single best expert
    SingleBest,
    /// Select top-k experts
    TopK { k: usize },
    /// Ensemble of experts
    Ensemble { max_experts: usize },
    /// Parallel execution with voting
    ParallelVoting { min_votes: usize },
    /// Sequential fallback chain
    SequentialFallback,
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        RoutingStrategy::TopK { k: 3 }
    }
}

/// Expert execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Selected experts in order
    pub experts: Vec<ExpertId>,
    /// Execution groups (experts that can run in parallel)
    pub parallel_groups: Vec<ParallelGroup>,
    /// Sequential experts (must run after parallel groups)
    pub sequential_experts: Vec<ExpertId>,
    /// Fallback strategy if primary experts fail
    pub fallback_strategy: Option<FallbackStrategy>,
    /// Estimated total execution time
    pub estimated_time: f32,
    /// Estimated memory usage
    pub estimated_memory: usize,
}

/// Group of experts that can execute in parallel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelGroup {
    /// Expert IDs in this group
    pub experts: Vec<ExpertId>,
    /// Expected execution time for this group
    pub estimated_time: f32,
    /// Memory requirements
    pub memory_requirement: usize,
}

/// Fallback strategy when primary experts fail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackStrategy {
    /// Backup experts to try
    pub backup_experts: Vec<ExpertId>,
    /// Simplified execution plan
    pub simplified_plan: bool,
    /// Emergency expert (always available)
    pub emergency_expert: Option<ExpertId>,
}

/// Expert router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Routing strategy to use
    pub strategy: RoutingStrategy,
    /// Maximum number of experts to activate
    pub max_experts: usize,
    /// Confidence threshold for expert selection
    pub confidence_threshold: f32,
    /// Enable parallel execution
    pub enable_parallel: bool,
    /// Performance optimization settings
    pub optimization: OptimizationSettings,
}

/// Performance optimization settings for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    /// Prefer faster experts
    pub prefer_speed: bool,
    /// Prefer more accurate experts
    pub prefer_accuracy: bool,
    /// Maximum allowed memory usage
    pub max_memory_usage: usize,
    /// Maximum allowed total time
    pub max_total_time: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::default(),
            max_experts: 5,
            confidence_threshold: 0.7,
            enable_parallel: true,
            optimization: OptimizationSettings {
                prefer_speed: true,
                prefer_accuracy: true,
                max_memory_usage: 100 * 1024 * 1024, // 100MB
                max_total_time: 500.0, // 500ms
            },
        }
    }
}

/// Core expert router implementation
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRouter {
    /// Router configuration
    config: RouterConfig,
    /// Routing neural network (small network for expert selection)
    #[serde(skip)]
    routing_network: Option<NeuralNetwork>,
    /// Expert profiles for quick lookup
    expert_profiles: HashMap<ExpertId, ExpertProfile>,
    /// Performance history tracker
    performance_history: PerformanceHistory,
    /// Feature extraction configuration
    feature_config: FeatureConfig,
    /// Routing cache for common patterns
    routing_cache: HashMap<String, CachedRoute>,
}

/// Expert profile for routing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertProfile {
    /// Expert identifier
    pub id: ExpertId,
    /// Domain and specialization
    pub domain: ExpertDomain,
    pub specialization: Specialization,
    /// Performance characteristics
    pub avg_execution_time: f32,
    pub success_rate: f32,
    pub memory_usage: usize,
    /// Capability scores (0.0 to 1.0)
    pub capability_scores: HashMap<String, f32>,
    /// Dependencies on other experts
    pub dependencies: Vec<ExpertId>,
    /// Complementary experts
    pub complements: Vec<ExpertId>,
}

/// Performance history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    /// Recent routing decisions and their outcomes
    recent_routes: VecDeque<RouteOutcome>,
    /// Success rates by expert combination
    combination_success: HashMap<Vec<ExpertId>, f32>,
    /// Average execution times by context type
    context_times: HashMap<String, f32>,
    /// Maximum history size
    max_history_size: usize,
}

/// Outcome of a routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteOutcome {
    /// Context hash for this routing decision
    pub context_hash: String,
    /// Selected experts
    pub experts: Vec<ExpertId>,
    /// Actual execution time
    pub execution_time: f32,
    /// Success indicator
    pub success: bool,
    /// Final confidence score
    pub final_confidence: f32,
    /// Timestamp
    pub timestamp: f64,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Number of features to extract
    pub feature_count: usize,
    /// Include token-based features
    pub include_token_features: bool,
    /// Include semantic features
    pub include_semantic_features: bool,
    /// Include context features
    pub include_context_features: bool,
    /// Feature normalization
    pub normalize_features: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            feature_count: 64,
            include_token_features: true,
            include_semantic_features: true,
            include_context_features: true,
            normalize_features: true,
        }
    }
}

/// Cached routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRoute {
    /// Execution plan
    pub plan: ExecutionPlan,
    /// Cache timestamp
    pub timestamp: f64,
    /// Hit count
    pub hit_count: u32,
    /// Success rate for this cached route
    pub success_rate: f32,
}

#[wasm_bindgen]
impl ExpertRouter {
    /// Create a new expert router
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<ExpertRouter, JsValue> {
        let config: RouterConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

        let mut router = ExpertRouter {
            config,
            routing_network: None,
            expert_profiles: HashMap::new(),
            performance_history: PerformanceHistory {
                recent_routes: VecDeque::new(),
                combination_success: HashMap::new(),
                context_times: HashMap::new(),
                max_history_size: 1000,
            },
            feature_config: FeatureConfig::default(),
            routing_cache: HashMap::new(),
        };

        router.initialize_routing_network()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(router)
    }

    /// Plan expert execution for a given context
    #[wasm_bindgen]
    pub fn plan_execution(&mut self, context_json: &str) -> Result<String, JsValue> {
        let context: RequestContext = serde_json::from_str(context_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid context: {}", e)))?;

        // Check cache first
        let context_hash = self.calculate_context_hash(&context);
        if let Some(cached) = self.routing_cache.get_mut(&context_hash) {
            cached.hit_count += 1;
            let plan_json = serde_json::to_string(&cached.plan)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            return Ok(plan_json);
        }

        // Extract features from the context
        let features = self.extract_routing_features(&context)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Get expert scores from routing network
        let expert_scores = self.predict_expert_scores(&features)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Select experts based on strategy
        let selected_experts = self.select_experts(&expert_scores, &context)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Create execution plan
        let plan = self.create_execution_plan(selected_experts, &context)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Cache the plan
        self.cache_route(context_hash, plan.clone());

        let plan_json = serde_json::to_string(&plan)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(plan_json)
    }

    /// Register an expert profile for routing
    #[wasm_bindgen]
    pub fn register_expert(&mut self, profile_json: &str) -> Result<(), JsValue> {
        let profile: ExpertProfile = serde_json::from_str(profile_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid profile: {}", e)))?;

        self.expert_profiles.insert(profile.id, profile);
        Ok(())
    }

    /// Update routing performance with execution outcome
    #[wasm_bindgen]
    pub fn update_performance(&mut self, outcome_json: &str) -> Result<(), JsValue> {
        let outcome: RouteOutcome = serde_json::from_str(outcome_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid outcome: {}", e)))?;

        // Update performance history
        self.performance_history.recent_routes.push_back(outcome.clone());
        
        // Maintain history size
        if self.performance_history.recent_routes.len() > self.performance_history.max_history_size {
            self.performance_history.recent_routes.pop_front();
        }

        // Update combination success rates
        let success_rate = self.performance_history.combination_success
            .entry(outcome.experts.clone())
            .or_insert(1.0);
        
        // Exponential moving average
        *success_rate = 0.9 * *success_rate + 0.1 * if outcome.success { 1.0 } else { 0.0 };

        // Update cache success rate if applicable
        if let Some(cached) = self.routing_cache.get_mut(&outcome.context_hash) {
            cached.success_rate = 0.9 * cached.success_rate + 0.1 * if outcome.success { 1.0 } else { 0.0 };
        }

        Ok(())
    }

    /// Get routing statistics
    #[wasm_bindgen]
    pub fn get_statistics(&self) -> Result<String, JsValue> {
        let stats = RoutingStatistics {
            total_routes: self.performance_history.recent_routes.len(),
            cache_hit_rate: self.calculate_cache_hit_rate(),
            avg_execution_time: self.calculate_avg_execution_time(),
            success_rate: self.calculate_overall_success_rate(),
            expert_usage: self.calculate_expert_usage_stats(),
            top_combinations: self.get_top_expert_combinations(),
        };

        serde_json::to_string(&stats)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Clear routing cache
    #[wasm_bindgen]
    pub fn clear_cache(&mut self) {
        self.routing_cache.clear();
    }
}

impl ExpertRouter {
    /// Initialize the routing neural network
    fn initialize_routing_network(&mut self) -> Result<()> {
        let mut network = NeuralNetwork::new();
        
        // Small network for routing decisions
        // Input: context features, Output: expert selection scores
        let input_size = self.feature_config.feature_count;
        let hidden_size = 32;
        let output_size = 64; // Support up to 64 experts
        
        // Hidden layer
        let hidden_layer = Layer::dense(input_size, hidden_size)
            .with_activation(Activation::ReLU);
        network.add_layer(hidden_layer);
        
        // Output layer with sigmoid for probability scores
        let output_layer = Layer::dense(hidden_size, output_size)
            .with_activation(Activation::Sigmoid);
        network.add_layer(output_layer);
        
        self.routing_network = Some(network);
        Ok(())
    }

    /// Extract features from request context
    fn extract_routing_features(&self, context: &RequestContext) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(self.feature_config.feature_count);

        // Basic context features
        features.push(context.complexity);
        features.push(context.token_count as f32 / 1000.0); // Normalize token count
        features.push(context.history_length as f32 / 100.0); // Normalize history length

        // Required capabilities (one-hot encoding)
        for domain in [
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Language,
            ExpertDomain::Mathematics,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
            ExpertDomain::Planning,
            ExpertDomain::Creativity,
            ExpertDomain::DataAnalysis,
            ExpertDomain::DecisionMaking,
        ] {
            features.push(if context.required_capabilities.contains(&domain) { 1.0 } else { 0.0 });
        }

        // Performance requirements
        features.push(context.performance_requirements.max_inference_time / 1000.0);
        features.push(context.performance_requirements.min_confidence);
        features.push(context.performance_requirements.max_memory_usage as f32 / (1024.0 * 1024.0)); // MB

        // Text-based features (simplified)
        if self.feature_config.include_token_features {
            let request_words: Vec<&str> = context.request.split_whitespace().collect();
            features.push(request_words.len() as f32 / 100.0); // Word count
            features.push(context.request.chars().count() as f32 / 1000.0); // Char count
            
            // Simple keyword detection
            let keywords = [
                "code", "function", "algorithm", "debug", "fix", "implement",
                "explain", "analyze", "summarize", "translate", "write",
                "calculate", "solve", "compute", "math", "equation",
                "plan", "strategy", "organize", "structure", "design",
                "create", "generate", "invent", "imagine", "story",
            ];
            
            for keyword in keywords {
                features.push(if context.request.to_lowercase().contains(keyword) { 1.0 } else { 0.0 });
            }
        }

        // Pad or truncate to exact feature count
        features.resize(self.feature_config.feature_count, 0.0);

        // Normalize if configured
        if self.feature_config.normalize_features {
            let max_val = features.iter().copied().fold(0.0, f32::max);
            if max_val > 0.0 {
                for feature in &mut features {
                    *feature /= max_val;
                }
            }
        }

        Ok(features)
    }

    /// Predict expert scores using the routing network
    fn predict_expert_scores(&self, features: &[f32]) -> Result<Vec<f32>> {
        let network = self.routing_network.as_ref()
            .ok_or_else(|| KimiError::RoutingError("Routing network not initialized".to_string()))?;

        let result_json = network.predict(features)
            .map_err(|e| KimiError::RoutingError(e.to_string()))?;

        let scores: Vec<f32> = serde_json::from_str(&result_json)
            .map_err(|e| KimiError::RoutingError(e.to_string()))?;

        Ok(scores)
    }

    /// Select experts based on scores and strategy
    fn select_experts(&self, scores: &[f32], context: &RequestContext) -> Result<Vec<ExpertId>> {
        let mut expert_candidates: Vec<(ExpertId, f32)> = self.expert_profiles
            .iter()
            .enumerate()
            .filter_map(|(idx, (id, profile))| {
                if idx < scores.len() {
                    let base_score = scores[idx];
                    let adjusted_score = self.adjust_score_for_context(base_score, profile, context);
                    Some((*id, adjusted_score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score (descending)
        expert_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply strategy
        let selected = match &self.config.strategy {
            RoutingStrategy::SingleBest => {
                expert_candidates.into_iter().take(1).map(|(id, _)| id).collect()
            }
            RoutingStrategy::TopK { k } => {
                expert_candidates.into_iter().take(*k).map(|(id, _)| id).collect()
            }
            RoutingStrategy::Ensemble { max_experts } => {
                expert_candidates.into_iter()
                    .take(*max_experts)
                    .filter(|(_, score)| *score >= self.config.confidence_threshold)
                    .map(|(id, _)| id)
                    .collect()
            }
            RoutingStrategy::ParallelVoting { min_votes: _ } => {
                // Select top experts for parallel voting
                expert_candidates.into_iter().take(5).map(|(id, _)| id).collect()
            }
            RoutingStrategy::SequentialFallback => {
                // Select with fallback chain
                expert_candidates.into_iter().take(3).map(|(id, _)| id).collect()
            }
        };

        // Limit by max_experts config
        let limited: Vec<ExpertId> = selected.into_iter()
            .take(self.config.max_experts)
            .collect();

        Ok(limited)
    }

    /// Adjust expert score based on context
    fn adjust_score_for_context(&self, base_score: f32, profile: &ExpertProfile, context: &RequestContext) -> f32 {
        let mut adjusted_score = base_score;

        // Domain match bonus
        if context.required_capabilities.contains(&profile.domain) {
            adjusted_score += 0.2;
        }

        // Performance requirements adjustment
        if profile.avg_execution_time <= context.performance_requirements.max_inference_time {
            adjusted_score += 0.1;
        } else {
            adjusted_score -= 0.2;
        }

        // Memory usage adjustment
        if profile.memory_usage <= context.performance_requirements.max_memory_usage {
            adjusted_score += 0.05;
        } else {
            adjusted_score -= 0.1;
        }

        // Success rate bonus
        adjusted_score += profile.success_rate * 0.1;

        // Historical performance adjustment
        if let Some(combo_success) = self.performance_history.combination_success.get(&vec![profile.id]) {
            adjusted_score += (combo_success - 0.5) * 0.2;
        }

        adjusted_score.min(1.0).max(0.0)
    }

    /// Create execution plan from selected experts
    fn create_execution_plan(&self, experts: Vec<ExpertId>, context: &RequestContext) -> Result<ExecutionPlan> {
        let mut parallel_groups = Vec::new();
        let mut sequential_experts = Vec::new();
        let mut estimated_time = 0.0;
        let mut estimated_memory = 0;

        if self.config.enable_parallel && context.performance_requirements.allow_parallel {
            // Group experts that can run in parallel
            let mut current_group = Vec::new();
            let mut current_memory = 0;

            for expert_id in &experts {
                if let Some(profile) = self.expert_profiles.get(expert_id) {
                    // Check if adding this expert would exceed memory limit
                    if current_memory + profile.memory_usage <= context.performance_requirements.max_memory_usage {
                        current_group.push(*expert_id);
                        current_memory += profile.memory_usage;
                        estimated_time = estimated_time.max(profile.avg_execution_time);
                    } else {
                        // Start new group or make sequential
                        if !current_group.is_empty() {
                            parallel_groups.push(ParallelGroup {
                                experts: current_group.clone(),
                                estimated_time,
                                memory_requirement: current_memory,
                            });
                            current_group.clear();
                            current_memory = 0;
                            estimated_time = 0.0;
                        }
                        sequential_experts.push(*expert_id);
                        estimated_time += profile.avg_execution_time;
                    }
                    estimated_memory += profile.memory_usage;
                }
            }

            // Add remaining group
            if !current_group.is_empty() {
                parallel_groups.push(ParallelGroup {
                    experts: current_group,
                    estimated_time,
                    memory_requirement: current_memory,
                });
            }
        } else {
            // All experts run sequentially
            for expert_id in &experts {
                if let Some(profile) = self.expert_profiles.get(expert_id) {
                    estimated_time += profile.avg_execution_time;
                    estimated_memory += profile.memory_usage;
                }
            }
            sequential_experts = experts.clone();
        }

        // Create fallback strategy
        let fallback_strategy = self.create_fallback_strategy(&experts, context);

        Ok(ExecutionPlan {
            experts,
            parallel_groups,
            sequential_experts,
            fallback_strategy,
            estimated_time,
            estimated_memory,
        })
    }

    /// Create fallback strategy
    fn create_fallback_strategy(&self, _primary_experts: &[ExpertId], _context: &RequestContext) -> Option<FallbackStrategy> {
        // Simplified fallback - in a real implementation, this would be more sophisticated
        let backup_experts: Vec<ExpertId> = self.expert_profiles
            .iter()
            .filter(|(_, profile)| profile.success_rate > 0.9)
            .map(|(id, _)| *id)
            .take(2)
            .collect();

        if backup_experts.is_empty() {
            None
        } else {
            Some(FallbackStrategy {
                emergency_expert: backup_experts.first().copied(),
                backup_experts,
                simplified_plan: true,
            })
        }
    }

    /// Calculate context hash for caching
    fn calculate_context_hash(&self, context: &RequestContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        context.request.hash(&mut hasher);
        context.required_capabilities.hash(&mut hasher);
        ((context.complexity * 100.0) as u32).hash(&mut hasher);
        context.token_count.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Cache a routing decision
    fn cache_route(&mut self, context_hash: String, plan: ExecutionPlan) {
        let cached_route = CachedRoute {
            plan,
            timestamp: Utils::now(),
            hit_count: 0,
            success_rate: 1.0,
        };

        self.routing_cache.insert(context_hash, cached_route);

        // Limit cache size
        if self.routing_cache.len() > 1000 {
            // Remove oldest entries (simplified)
            let mut to_remove = Vec::new();
            let cutoff_time = Utils::now() - 3600000.0; // 1 hour

            for (key, cached) in &self.routing_cache {
                if cached.timestamp < cutoff_time && cached.hit_count < 5 {
                    to_remove.push(key.clone());
                }
            }

            for key in to_remove {
                self.routing_cache.remove(&key);
            }
        }
    }

    /// Calculate cache hit rate
    fn calculate_cache_hit_rate(&self) -> f32 {
        let total_hits: u32 = self.routing_cache.values().map(|c| c.hit_count).sum();
        let total_routes = self.performance_history.recent_routes.len() as u32;
        
        if total_routes > 0 {
            total_hits as f32 / total_routes as f32
        } else {
            0.0
        }
    }

    /// Calculate average execution time
    fn calculate_avg_execution_time(&self) -> f32 {
        let times: Vec<f32> = self.performance_history.recent_routes
            .iter()
            .map(|r| r.execution_time)
            .collect();

        if times.is_empty() {
            0.0
        } else {
            times.iter().sum::<f32>() / times.len() as f32
        }
    }

    /// Calculate overall success rate
    fn calculate_overall_success_rate(&self) -> f32 {
        let successes = self.performance_history.recent_routes
            .iter()
            .filter(|r| r.success)
            .count();

        let total = self.performance_history.recent_routes.len();

        if total > 0 {
            successes as f32 / total as f32
        } else {
            1.0
        }
    }

    /// Calculate expert usage statistics
    fn calculate_expert_usage_stats(&self) -> HashMap<ExpertId, u32> {
        let mut usage = HashMap::new();

        for route in &self.performance_history.recent_routes {
            for &expert_id in &route.experts {
                *usage.entry(expert_id).or_insert(0) += 1;
            }
        }

        usage
    }

    /// Get top expert combinations
    fn get_top_expert_combinations(&self) -> Vec<(Vec<ExpertId>, f32)> {
        let mut combinations: Vec<_> = self.performance_history.combination_success
            .iter()
            .map(|(combo, success)| (combo.clone(), *success))
            .collect();

        combinations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        combinations.into_iter().take(10).collect()
    }
}

/// Routing statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct RoutingStatistics {
    pub total_routes: usize,
    pub cache_hit_rate: f32,
    pub avg_execution_time: f32,
    pub success_rate: f32,
    pub expert_usage: HashMap<ExpertId, u32>,
    pub top_combinations: Vec<(Vec<ExpertId>, f32)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let config = RouterConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        let router = ExpertRouter::new(&config_json).unwrap();
        
        assert_eq!(router.config.max_experts, 5);
        assert!(router.routing_network.is_some());
    }

    #[test]
    fn test_feature_extraction() {
        let config = RouterConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        let router = ExpertRouter::new(&config_json).unwrap();

        let context = RequestContext {
            request: "Write a function to sort an array".to_string(),
            complexity: 0.6,
            token_count: 50,
            history_length: 5,
            required_capabilities: vec![ExpertDomain::Coding],
            performance_requirements: PerformanceRequirements::default(),
            metadata: HashMap::new(),
        };

        let features = router.extract_routing_features(&context).unwrap();
        assert_eq!(features.len(), router.feature_config.feature_count);
    }

    #[test]
    fn test_expert_profile_registration() {
        let config = RouterConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        let mut router = ExpertRouter::new(&config_json).unwrap();

        let profile = ExpertProfile {
            id: 1,
            domain: ExpertDomain::Coding,
            specialization: Specialization::CodeGeneration,
            avg_execution_time: 50.0,
            success_rate: 0.9,
            memory_usage: 1024 * 1024,
            capability_scores: HashMap::new(),
            dependencies: vec![],
            complements: vec![],
        };

        let profile_json = serde_json::to_string(&profile).unwrap();
        router.register_expert(&profile_json).unwrap();

        assert!(router.expert_profiles.contains_key(&1));
    }
}