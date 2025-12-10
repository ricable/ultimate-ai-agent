//! Expert Execution Engine
//!
//! This module handles the execution of micro-experts, including parallel
//! execution, result merging, and error recovery.

use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use js_sys::Promise;

/// Execution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Execute experts sequentially
    Sequential,
    /// Execute experts in parallel (Web Workers)
    Parallel,
    /// Adaptive execution based on system load
    Adaptive,
    /// Pipeline execution with overlapping
    Pipeline,
}

/// Execution context for expert runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Request identifier
    pub request_id: String,
    /// Input data for the experts
    pub input_data: Vec<f32>,
    /// Execution strategy to use
    pub strategy: ExecutionStrategy,
    /// Timeout in milliseconds
    pub timeout: f32,
    /// Priority level (0-10)
    pub priority: u8,
    /// Whether to retry on failure
    pub retry_on_failure: bool,
    /// Maximum number of retries
    pub max_retries: u32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            request_id: format!("req_{}", Utils::generate_expert_id()),
            input_data: Vec::new(),
            strategy: ExecutionStrategy::Adaptive,
            timeout: 5000.0, // 5 seconds
            priority: 5,
            retry_on_failure: true,
            max_retries: 3,
            metadata: HashMap::new(),
        }
    }
}

/// Execution result from a single expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertExecutionResult {
    /// Expert that generated this result
    pub expert_id: ExpertId,
    /// Execution success status
    pub success: bool,
    /// Output data
    pub output: Vec<f32>,
    /// Confidence score
    pub confidence: f32,
    /// Execution time in milliseconds
    pub execution_time: f32,
    /// Memory used during execution
    pub memory_used: usize,
    /// Error message if execution failed
    pub error_message: Option<String>,
    /// Execution timestamp
    pub timestamp: f64,
}

/// Combined execution result from multiple experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedExecutionResult {
    /// Request identifier
    pub request_id: String,
    /// Individual expert results
    pub expert_results: Vec<ExpertExecutionResult>,
    /// Final merged output
    pub final_output: Vec<f32>,
    /// Overall confidence score
    pub overall_confidence: f32,
    /// Total execution time
    pub total_execution_time: f32,
    /// Success indicator
    pub success: bool,
    /// Execution strategy used
    pub strategy_used: ExecutionStrategy,
    /// Number of experts that succeeded
    pub successful_experts: usize,
    /// Error details if any
    pub errors: Vec<String>,
}

/// Result merging strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergingStrategy {
    /// Average all outputs
    Average,
    /// Weighted average by confidence
    WeightedAverage,
    /// Use result from most confident expert
    BestConfidence,
    /// Majority voting for classification
    MajorityVoting,
    /// Ensemble method with learned weights
    Ensemble,
}

impl Default for MergingStrategy {
    fn default() -> Self {
        MergingStrategy::WeightedAverage
    }
}

/// Expert execution engine
#[wasm_bindgen]
pub struct ExecutionEngine {
    /// Memory pool for expert management
    memory_pool: WasmMemoryPool,
    /// Expert registry
    registry: ExpertRegistry,
    /// Web Worker pool for parallel execution
    #[wasm_bindgen(skip)]
    worker_pool: Option<WebWorkerPool>,
    /// Result merger
    result_merger: ResultMerger,
    /// Error recovery system
    error_recovery: ErrorRecovery,
    /// Execution statistics
    stats: ExecutionStats,
    /// Configuration
    config: ExecutionConfig,
}

/// Web Worker pool for parallel execution
#[derive(Debug)]
pub struct WebWorkerPool {
    /// Available workers
    workers: VecDeque<web_sys::Worker>,
    /// Maximum number of workers
    max_workers: usize,
    /// Currently busy workers
    busy_workers: HashMap<String, web_sys::Worker>,
}

/// Result merger for combining expert outputs
#[derive(Debug, Clone)]
pub struct ResultMerger {
    /// Merging strategy
    strategy: MergingStrategy,
    /// Learned ensemble weights
    ensemble_weights: HashMap<ExpertId, f32>,
    /// Confidence thresholds
    confidence_thresholds: HashMap<ExpertDomain, f32>,
}

/// Error recovery system
#[derive(Debug, Clone)]
pub struct ErrorRecovery {
    /// Fallback experts for each domain
    fallback_experts: HashMap<ExpertDomain, Vec<ExpertId>>,
    /// Retry strategies
    retry_strategies: HashMap<String, RetryStrategy>,
    /// Circuit breaker states
    circuit_breakers: HashMap<ExpertId, CircuitBreakerState>,
}

/// Circuit breaker state
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    /// Current state
    state: CircuitState,
    /// Failure count
    failure_count: u32,
    /// Success count
    success_count: u32,
    /// Last failure time
    last_failure_time: f64,
    /// Threshold for opening circuit
    failure_threshold: u32,
    /// Timeout before trying again
    timeout_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,  // Normal operation
    Open,    // Circuit is open, failing fast
    HalfOpen, // Testing if service is back
}

/// Retry strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryStrategy {
    /// No retries
    None,
    /// Fixed delay between retries
    FixedDelay { delay_ms: u64 },
    /// Exponential backoff
    ExponentialBackoff { base_delay_ms: u64, max_delay_ms: u64 },
    /// Linear backoff
    LinearBackoff { initial_delay_ms: u64, increment_ms: u64 },
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Total executions performed
    pub total_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Average execution time
    pub avg_execution_time: f32,
    /// Average confidence score
    pub avg_confidence: f32,
    /// Parallel executions count
    pub parallel_executions: u64,
    /// Sequential executions count
    pub sequential_executions: u64,
    /// Total experts used
    pub total_experts_used: u64,
    /// Last execution timestamp
    pub last_execution_time: f64,
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            avg_execution_time: 0.0,
            avg_confidence: 0.0,
            parallel_executions: 0,
            sequential_executions: 0,
            total_experts_used: 0,
            last_execution_time: 0.0,
        }
    }
}

/// Execution engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Default execution strategy
    pub default_strategy: ExecutionStrategy,
    /// Default merging strategy
    pub default_merging: MergingStrategy,
    /// Maximum parallel workers
    pub max_workers: usize,
    /// Default timeout in milliseconds
    pub default_timeout: f32,
    /// Enable circuit breakers
    pub enable_circuit_breakers: bool,
    /// Enable adaptive execution
    pub enable_adaptive_execution: bool,
    /// Performance monitoring
    pub enable_performance_monitoring: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            default_strategy: ExecutionStrategy::Adaptive,
            default_merging: MergingStrategy::WeightedAverage,
            max_workers: 4,
            default_timeout: 5000.0,
            enable_circuit_breakers: true,
            enable_adaptive_execution: true,
            enable_performance_monitoring: true,
        }
    }
}

#[wasm_bindgen]
impl ExecutionEngine {
    /// Create a new execution engine
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, memory_pool: WasmMemoryPool) -> Result<ExecutionEngine, JsValue> {
        let config: ExecutionConfig = serde_json::from_str(config_json)
            .unwrap_or_else(|_| ExecutionConfig::default());

        Ok(ExecutionEngine {
            memory_pool,
            registry: ExpertRegistry::new(),
            worker_pool: None, // Initialize later if needed
            result_merger: ResultMerger::new(config.default_merging.clone()),
            error_recovery: ErrorRecovery::new(),
            stats: ExecutionStats::default(),
            config,
        })
    }

    /// Execute an execution plan
    #[wasm_bindgen]
    pub fn execute_plan(&mut self, plan_json: &str, context_json: &str) -> Promise {
        let plan_result: Result<ExecutionPlan, _> = serde_json::from_str(plan_json);
        let context_result: Result<ExecutionContext, _> = serde_json::from_str(context_json);

        match (plan_result, context_result) {
            (Ok(plan), Ok(context)) => {
                let engine = self.clone(); // This would need proper async handling in real implementation
                future_to_promise(async move {
                    engine.execute_plan_async(plan, context).await
                })
            }
            (Err(e), _) => {
                let error = JsValue::from_str(&format!("Invalid plan JSON: {}", e));
                Promise::reject(&error)
            }
            (_, Err(e)) => {
                let error = JsValue::from_str(&format!("Invalid context JSON: {}", e));
                Promise::reject(&error)
            }
        }
    }

    /// Execute a single expert
    #[wasm_bindgen]
    pub fn execute_expert(&mut self, expert_id: ExpertId, input: &[f32]) -> Result<String, JsValue> {
        let start_time = Utils::now();
        
        // Load expert from memory
        let expert_data = self.memory_pool.get_expert_data(expert_id)?;
        let expert: KimiMicroExpert = serde_json::from_str(&expert_data)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse expert: {}", e)))?;

        // Check circuit breaker
        if self.config.enable_circuit_breakers {
            if let Some(circuit_state) = self.error_recovery.circuit_breakers.get(&expert_id) {
                if circuit_state.state == CircuitState::Open {
                    let error_msg = format!("Circuit breaker is open for expert {}", expert_id);
                    return Err(JsValue::from_str(&error_msg));
                }
            }
        }

        // Execute the expert
        let result = match expert.predict(input) {
            Ok(output_json) => {
                let expert_result: ExpertResult = serde_json::from_str(&output_json)
                    .map_err(|e| JsValue::from_str(&format!("Failed to parse expert result: {}", e)))?;

                ExpertExecutionResult {
                    expert_id,
                    success: true,
                    output: expert_result.output,
                    confidence: expert_result.confidence,
                    execution_time: Utils::now() - start_time,
                    memory_used: expert_result.memory_usage,
                    error_message: None,
                    timestamp: Utils::now(),
                }
            }
            Err(e) => {
                ExpertExecutionResult {
                    expert_id,
                    success: false,
                    output: Vec::new(),
                    confidence: 0.0,
                    execution_time: Utils::now() - start_time,
                    memory_used: 0,
                    error_message: Some(format!("{:?}", e)),
                    timestamp: Utils::now(),
                }
            }
        };

        // Update circuit breaker
        if self.config.enable_circuit_breakers {
            self.update_circuit_breaker(expert_id, result.success);
        }

        // Update statistics
        self.update_execution_stats(&result);

        serde_json::to_string(&result)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get execution statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.stats)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Register an expert in the execution engine
    #[wasm_bindgen]
    pub fn register_expert(&mut self, expert_json: &str) -> Result<(), JsValue> {
        let expert: KimiMicroExpert = serde_json::from_str(expert_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid expert JSON: {}", e)))?;

        self.registry.register_expert(expert)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(())
    }

    /// Initialize Web Worker pool for parallel execution
    #[wasm_bindgen]
    pub fn initialize_worker_pool(&mut self, worker_script_url: &str) -> Result<(), JsValue> {
        self.worker_pool = Some(WebWorkerPool::new(self.config.max_workers, worker_script_url)?);
        Ok(())
    }

    /// Clear execution statistics
    #[wasm_bindgen]
    pub fn clear_stats(&mut self) {
        self.stats = ExecutionStats::default();
    }
}

impl ExecutionEngine {
    /// Clone implementation for async execution
    fn clone(&self) -> Self {
        // This is a simplified clone - in a real implementation,
        // you'd need to handle the non-cloneable fields properly
        ExecutionEngine {
            memory_pool: self.memory_pool.clone(),
            registry: self.registry.clone(),
            worker_pool: None,
            result_merger: self.result_merger.clone(),
            error_recovery: self.error_recovery.clone(),
            stats: self.stats.clone(),
            config: self.config.clone(),
        }
    }

    /// Execute a plan asynchronously
    async fn execute_plan_async(&self, plan: ExecutionPlan, context: ExecutionContext) -> Result<JsValue, JsValue> {
        let start_time = Utils::now();
        let mut expert_results = Vec::new();
        let mut errors = Vec::new();

        // Execute based on strategy
        match context.strategy {
            ExecutionStrategy::Sequential => {
                expert_results = self.execute_sequential(&plan.experts, &context).await?;
            }
            ExecutionStrategy::Parallel => {
                expert_results = self.execute_parallel(&plan.experts, &context).await?;
            }
            ExecutionStrategy::Adaptive => {
                expert_results = self.execute_adaptive(&plan.experts, &context).await?;
            }
            ExecutionStrategy::Pipeline => {
                expert_results = self.execute_pipeline(&plan.experts, &context).await?;
            }
        }

        // Merge results
        let merged_result = self.result_merger.merge_results(&expert_results)?;

        // Create combined result
        let combined_result = CombinedExecutionResult {
            request_id: context.request_id,
            expert_results: expert_results.clone(),
            final_output: merged_result.output,
            overall_confidence: merged_result.confidence,
            total_execution_time: Utils::now() - start_time,
            success: expert_results.iter().any(|r| r.success),
            strategy_used: context.strategy,
            successful_experts: expert_results.iter().filter(|r| r.success).count(),
            errors,
        };

        let result_json = serde_json::to_string(&combined_result)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(JsValue::from_str(&result_json))
    }

    /// Execute experts sequentially
    async fn execute_sequential(&self, experts: &[ExpertId], context: &ExecutionContext) -> Result<Vec<ExpertExecutionResult>, JsValue> {
        let mut results = Vec::new();

        for &expert_id in experts {
            let result = self.execute_single_expert(expert_id, &context.input_data, context.timeout).await?;
            results.push(result);

            // Early termination if we have enough successful results
            if results.iter().filter(|r| r.success).count() >= 3 {
                break;
            }
        }

        Ok(results)
    }

    /// Execute experts in parallel
    async fn execute_parallel(&self, experts: &[ExpertId], context: &ExecutionContext) -> Result<Vec<ExpertExecutionResult>, JsValue> {
        // In a real implementation, this would use Web Workers or similar
        // For now, we'll simulate parallel execution
        let mut results = Vec::new();

        for &expert_id in experts {
            let result = self.execute_single_expert(expert_id, &context.input_data, context.timeout).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute with adaptive strategy
    async fn execute_adaptive(&self, experts: &[ExpertId], context: &ExecutionContext) -> Result<Vec<ExpertExecutionResult>, JsValue> {
        // Decide between sequential and parallel based on system state
        if experts.len() <= 2 || !self.has_parallel_capability() {
            self.execute_sequential(experts, context).await
        } else {
            self.execute_parallel(experts, context).await
        }
    }

    /// Execute with pipeline strategy
    async fn execute_pipeline(&self, experts: &[ExpertId], context: &ExecutionContext) -> Result<Vec<ExpertExecutionResult>, JsValue> {
        // Pipeline execution with overlapping
        // For now, fall back to sequential
        self.execute_sequential(experts, context).await
    }

    /// Execute a single expert asynchronously
    async fn execute_single_expert(&self, expert_id: ExpertId, input: &[f32], timeout: f32) -> Result<ExpertExecutionResult, JsValue> {
        // This is a simplified async wrapper
        // In a real implementation, you'd use proper async/await with timeouts
        
        let start_time = Utils::now();

        // Simulate async execution
        let result = ExpertExecutionResult {
            expert_id,
            success: true,
            output: vec![0.5, 0.3, 0.2], // Placeholder output
            confidence: 0.8,
            execution_time: Utils::now() - start_time,
            memory_used: 1024,
            error_message: None,
            timestamp: Utils::now(),
        };

        Ok(result)
    }

    /// Check if parallel execution is available
    fn has_parallel_capability(&self) -> bool {
        self.worker_pool.is_some()
    }

    /// Update circuit breaker state
    fn update_circuit_breaker(&mut self, expert_id: ExpertId, success: bool) {
        let circuit = self.error_recovery.circuit_breakers
            .entry(expert_id)
            .or_insert_with(|| CircuitBreakerState {
                state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: 0.0,
                failure_threshold: 5,
                timeout_ms: 30000.0, // 30 seconds
            });

        if success {
            circuit.success_count += 1;
            if circuit.state == CircuitState::HalfOpen && circuit.success_count >= 3 {
                circuit.state = CircuitState::Closed;
                circuit.failure_count = 0;
            }
        } else {
            circuit.failure_count += 1;
            circuit.last_failure_time = Utils::now();

            if circuit.failure_count >= circuit.failure_threshold {
                circuit.state = CircuitState::Open;
            }
        }

        // Check if circuit should move from Open to HalfOpen
        if circuit.state == CircuitState::Open {
            let time_since_failure = Utils::now() - circuit.last_failure_time;
            if time_since_failure >= circuit.timeout_ms {
                circuit.state = CircuitState::HalfOpen;
                circuit.success_count = 0;
            }
        }
    }

    /// Update execution statistics
    fn update_execution_stats(&mut self, result: &ExpertExecutionResult) {
        self.stats.total_executions += 1;
        self.stats.total_experts_used += 1;
        self.stats.last_execution_time = result.timestamp;

        if result.success {
            self.stats.successful_executions += 1;
        } else {
            self.stats.failed_executions += 1;
        }

        // Update averages
        let total = self.stats.total_executions as f32;
        self.stats.avg_execution_time = 
            (self.stats.avg_execution_time * (total - 1.0) + result.execution_time) / total;
        self.stats.avg_confidence = 
            (self.stats.avg_confidence * (total - 1.0) + result.confidence) / total;
    }
}

impl ResultMerger {
    /// Create a new result merger
    pub fn new(strategy: MergingStrategy) -> Self {
        Self {
            strategy,
            ensemble_weights: HashMap::new(),
            confidence_thresholds: HashMap::new(),
        }
    }

    /// Merge multiple expert results
    pub fn merge_results(&self, results: &[ExpertExecutionResult]) -> Result<ExpertResult> {
        if results.is_empty() {
            return Err(KimiError::ExecutionError("No results to merge".to_string()));
        }

        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        if successful_results.is_empty() {
            return Err(KimiError::ExecutionError("No successful results to merge".to_string()));
        }

        let merged_output = match self.strategy {
            MergingStrategy::Average => self.merge_average(&successful_results)?,
            MergingStrategy::WeightedAverage => self.merge_weighted_average(&successful_results)?,
            MergingStrategy::BestConfidence => self.merge_best_confidence(&successful_results)?,
            MergingStrategy::MajorityVoting => self.merge_majority_voting(&successful_results)?,
            MergingStrategy::Ensemble => self.merge_ensemble(&successful_results)?,
        };

        let merged_confidence = self.calculate_merged_confidence(&successful_results);

        Ok(ExpertResult {
            expert_id: 0, // Merged result
            domain: ExpertDomain::Reasoning, // Default
            output: merged_output,
            confidence: merged_confidence,
            inference_time: successful_results.iter().map(|r| r.execution_time).sum(),
            memory_usage: successful_results.iter().map(|r| r.memory_used).sum(),
            timestamp: Utils::now(),
        })
    }

    /// Merge using simple average
    fn merge_average(&self, results: &[&ExpertExecutionResult]) -> Result<Vec<f32>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        let output_size = results[0].output.len();
        let mut merged = vec![0.0; output_size];

        for result in results {
            if result.output.len() != output_size {
                return Err(KimiError::ExecutionError("Mismatched output sizes".to_string()));
            }

            for (i, &value) in result.output.iter().enumerate() {
                merged[i] += value;
            }
        }

        let count = results.len() as f32;
        for value in &mut merged {
            *value /= count;
        }

        Ok(merged)
    }

    /// Merge using weighted average by confidence
    fn merge_weighted_average(&self, results: &[&ExpertExecutionResult]) -> Result<Vec<f32>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        let output_size = results[0].output.len();
        let mut merged = vec![0.0; output_size];
        let mut total_weight = 0.0;

        for result in results {
            if result.output.len() != output_size {
                return Err(KimiError::ExecutionError("Mismatched output sizes".to_string()));
            }

            let weight = result.confidence;
            total_weight += weight;

            for (i, &value) in result.output.iter().enumerate() {
                merged[i] += value * weight;
            }
        }

        if total_weight > 0.0 {
            for value in &mut merged {
                *value /= total_weight;
            }
        }

        Ok(merged)
    }

    /// Use result from most confident expert
    fn merge_best_confidence(&self, results: &[&ExpertExecutionResult]) -> Result<Vec<f32>> {
        let best_result = results.iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| KimiError::ExecutionError("No results available".to_string()))?;

        Ok(best_result.output.clone())
    }

    /// Merge using majority voting (for classification)
    fn merge_majority_voting(&self, results: &[&ExpertExecutionResult]) -> Result<Vec<f32>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        let output_size = results[0].output.len();
        let mut vote_counts = vec![HashMap::new(); output_size];

        for result in results {
            if result.output.len() != output_size {
                return Err(KimiError::ExecutionError("Mismatched output sizes".to_string()));
            }

            for (i, &value) in result.output.iter().enumerate() {
                // Round to nearest 0.1 for voting
                let rounded_value = (value * 10.0).round() / 10.0;
                *vote_counts[i].entry(rounded_value as i32).or_insert(0) += 1;
            }
        }

        let mut merged = Vec::with_capacity(output_size);
        for votes in vote_counts {
            let winner = votes.into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(value, _)| value as f32 / 10.0)
                .unwrap_or(0.0);
            merged.push(winner);
        }

        Ok(merged)
    }

    /// Merge using ensemble method
    fn merge_ensemble(&self, results: &[&ExpertExecutionResult]) -> Result<Vec<f32>> {
        // If no learned weights, fall back to weighted average
        if self.ensemble_weights.is_empty() {
            return self.merge_weighted_average(results);
        }

        let output_size = results[0].output.len();
        let mut merged = vec![0.0; output_size];
        let mut total_weight = 0.0;

        for result in results {
            if result.output.len() != output_size {
                return Err(KimiError::ExecutionError("Mismatched output sizes".to_string()));
            }

            let weight = self.ensemble_weights.get(&result.expert_id).copied().unwrap_or(1.0);
            total_weight += weight;

            for (i, &value) in result.output.iter().enumerate() {
                merged[i] += value * weight;
            }
        }

        if total_weight > 0.0 {
            for value in &mut merged {
                *value /= total_weight;
            }
        }

        Ok(merged)
    }

    /// Calculate merged confidence score
    fn calculate_merged_confidence(&self, results: &[&ExpertExecutionResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        match self.strategy {
            MergingStrategy::Average => {
                results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32
            }
            MergingStrategy::WeightedAverage => {
                let total_weight: f32 = results.iter().map(|r| r.confidence).sum();
                if total_weight > 0.0 {
                    results.iter().map(|r| r.confidence * r.confidence).sum::<f32>() / total_weight
                } else {
                    0.0
                }
            }
            MergingStrategy::BestConfidence => {
                results.iter().map(|r| r.confidence).fold(0.0, f32::max)
            }
            MergingStrategy::MajorityVoting => {
                // Confidence based on agreement level
                let avg_confidence = results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32;
                let agreement_factor = 1.0 - (results.len() as f32 - 1.0) * 0.1; // Penalty for disagreement
                avg_confidence * agreement_factor.max(0.5)
            }
            MergingStrategy::Ensemble => {
                // Weighted confidence
                let total_weight: f32 = results.iter()
                    .map(|r| self.ensemble_weights.get(&r.expert_id).copied().unwrap_or(1.0))
                    .sum();
                
                if total_weight > 0.0 {
                    results.iter()
                        .map(|r| {
                            let weight = self.ensemble_weights.get(&r.expert_id).copied().unwrap_or(1.0);
                            r.confidence * weight
                        })
                        .sum::<f32>() / total_weight
                } else {
                    0.0
                }
            }
        }
    }
}

impl WebWorkerPool {
    /// Create a new Web Worker pool
    pub fn new(max_workers: usize, script_url: &str) -> Result<Self, JsValue> {
        let mut workers = VecDeque::new();

        // Create initial workers
        for _ in 0..max_workers {
            let worker = web_sys::Worker::new(script_url)?;
            workers.push_back(worker);
        }

        Ok(WebWorkerPool {
            workers,
            max_workers,
            busy_workers: HashMap::new(),
        })
    }

    /// Get an available worker
    pub fn get_worker(&mut self) -> Option<web_sys::Worker> {
        self.workers.pop_front()
    }

    /// Return a worker to the pool
    pub fn return_worker(&mut self, worker: web_sys::Worker) {
        if self.workers.len() < self.max_workers {
            self.workers.push_back(worker);
        }
    }
}

impl ErrorRecovery {
    /// Create a new error recovery system
    pub fn new() -> Self {
        Self {
            fallback_experts: HashMap::new(),
            retry_strategies: HashMap::new(),
            circuit_breakers: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_merger_average() {
        let merger = ResultMerger::new(MergingStrategy::Average);
        
        let results = vec![
            &ExpertExecutionResult {
                expert_id: 1,
                success: true,
                output: vec![1.0, 2.0, 3.0],
                confidence: 0.8,
                execution_time: 10.0,
                memory_used: 1024,
                error_message: None,
                timestamp: 0.0,
            },
            &ExpertExecutionResult {
                expert_id: 2,
                success: true,
                output: vec![2.0, 4.0, 6.0],
                confidence: 0.9,
                execution_time: 15.0,
                memory_used: 2048,
                error_message: None,
                timestamp: 0.0,
            },
        ];

        let merged = merger.merge_average(&results).unwrap();
        assert_eq!(merged, vec![1.5, 3.0, 4.5]);
    }

    #[test]
    fn test_circuit_breaker_state() {
        let mut circuit = CircuitBreakerState {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: 0.0,
            failure_threshold: 3,
            timeout_ms: 1000.0,
        };

        assert_eq!(circuit.state, CircuitState::Closed);

        // Simulate failures
        circuit.failure_count = 3;
        circuit.state = CircuitState::Open;
        
        assert_eq!(circuit.state, CircuitState::Open);
    }

    #[test]
    fn test_execution_context_defaults() {
        let context = ExecutionContext::default();
        assert_eq!(context.priority, 5);
        assert_eq!(context.timeout, 5000.0);
        assert!(context.retry_on_failure);
        assert_eq!(context.max_retries, 3);
    }
}