//! Kimi-K2 Micro-Expert Neural Networks
//!
//! This crate implements the core micro-expert neural networks for the Kimi-K2 
//! Rust-WASM conversion project. It provides lightweight, specialized neural 
//! networks optimized for WebAssembly deployment with sub-100ms inference times.

#![allow(clippy::module_inception)]

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

pub mod expert;
pub mod router;
pub mod memory;
pub mod compression;
pub mod execution;
pub mod context;
pub mod error;

pub use expert::*;
pub use router::*;
pub use memory::*;
pub use compression::*;
pub use execution::*;
pub use context::*;
pub use error::*;

/// Expert identification type
pub type ExpertId = u32;

/// Performance metrics for micro-experts
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average inference time in milliseconds
    pub avg_inference_time: f32,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Number of executions
    pub execution_count: u64,
    /// Confidence score (0.0 to 1.0)
    pub confidence_score: f32,
}

#[wasm_bindgen]
impl PerformanceMetrics {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            avg_inference_time: 0.0,
            success_rate: 1.0,
            memory_usage: 0,
            execution_count: 0,
            confidence_score: 1.0,
        }
    }

    #[wasm_bindgen]
    pub fn update(&mut self, inference_time: f32, success: bool, memory_usage: usize) {
        // Update average inference time
        let total_time = self.avg_inference_time * self.execution_count as f32;
        self.execution_count += 1;
        self.avg_inference_time = (total_time + inference_time) / self.execution_count as f32;

        // Update success rate
        let total_successes = self.success_rate * (self.execution_count - 1) as f32;
        let new_successes = total_successes + if success { 1.0 } else { 0.0 };
        self.success_rate = new_successes / self.execution_count as f32;

        // Update memory usage (moving average)
        self.memory_usage = (self.memory_usage + memory_usage) / 2;

        // Update confidence based on performance
        self.confidence_score = self.calculate_confidence_score();
    }

    fn calculate_confidence_score(&self) -> f32 {
        // Confidence based on success rate, inference time, and execution count
        let time_factor = if self.avg_inference_time < 50.0 { 1.0 } 
                         else if self.avg_inference_time < 100.0 { 0.8 } 
                         else { 0.6 };
        
        let experience_factor = (self.execution_count as f32 / 100.0).min(1.0);
        
        self.success_rate * time_factor * (0.5 + 0.5 * experience_factor)
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Expert specialization metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Specialization {
    /// Logical reasoning and problem-solving
    LogicalInference,
    /// Code generation and analysis
    CodeGeneration,
    /// Code debugging and error detection
    CodeDebugging,
    /// Natural language understanding
    LanguageUnderstanding,
    /// Natural language generation
    LanguageGeneration,
    /// Mathematical computation
    MathematicalReasoning,
    /// Function calling and tool use
    ToolUsage,
    /// API interaction and integration
    ApiInteraction,
    /// Long-context understanding
    ContextUnderstanding,
    /// Context synthesis and summarization
    ContextSynthesis,
    /// Task planning and decomposition
    TaskPlanning,
    /// Creative writing and content generation
    CreativeWriting,
    /// Data analysis and interpretation
    DataAnalysis,
    /// Decision making and evaluation
    DecisionMaking,
}

impl Specialization {
    /// Get the expected parameter count range for this specialization
    pub fn parameter_range(&self) -> (usize, usize) {
        match self {
            Specialization::LogicalInference => (5_000, 15_000),
            Specialization::CodeGeneration => (20_000, 60_000),
            Specialization::CodeDebugging => (15_000, 40_000),
            Specialization::LanguageUnderstanding => (10_000, 30_000),
            Specialization::LanguageGeneration => (15_000, 35_000),
            Specialization::MathematicalReasoning => (8_000, 25_000),
            Specialization::ToolUsage => (5_000, 20_000),
            Specialization::ApiInteraction => (3_000, 15_000),
            Specialization::ContextUnderstanding => (20_000, 50_000),
            Specialization::ContextSynthesis => (15_000, 40_000),
            Specialization::TaskPlanning => (10_000, 30_000),
            Specialization::CreativeWriting => (12_000, 35_000),
            Specialization::DataAnalysis => (8_000, 25_000),
            Specialization::DecisionMaking => (6_000, 20_000),
        }
    }

    /// Get the priority level for this specialization
    pub fn priority_level(&self) -> u8 {
        match self {
            Specialization::LogicalInference => 9,
            Specialization::CodeGeneration => 8,
            Specialization::MathematicalReasoning => 8,
            Specialization::ContextUnderstanding => 7,
            Specialization::TaskPlanning => 7,
            Specialization::CodeDebugging => 6,
            Specialization::LanguageUnderstanding => 6,
            Specialization::ToolUsage => 6,
            Specialization::ContextSynthesis => 5,
            Specialization::LanguageGeneration => 5,
            Specialization::DataAnalysis => 5,
            Specialization::DecisionMaking => 4,
            Specialization::ApiInteraction => 4,
            Specialization::CreativeWriting => 3,
        }
    }
}

/// Request context for expert routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestContext {
    /// The original request text
    pub request: String,
    /// Estimated complexity (0.0 to 1.0)
    pub complexity: f32,
    /// Number of tokens in the request
    pub token_count: usize,
    /// Conversation history length
    pub history_length: usize,
    /// Required capabilities
    pub required_capabilities: Vec<ExpertDomain>,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Context metadata
    pub metadata: HashMap<String, String>,
}

/// Performance requirements for expert execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum allowed inference time in milliseconds
    pub max_inference_time: f32,
    /// Minimum required confidence score
    pub min_confidence: f32,
    /// Maximum memory usage in bytes
    pub max_memory_usage: usize,
    /// Whether parallel execution is allowed
    pub allow_parallel: bool,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_inference_time: 100.0,
            min_confidence: 0.7,
            max_memory_usage: 50 * 1024 * 1024, // 50MB
            allow_parallel: true,
        }
    }
}

/// Initialize the WASM module with panic hooks
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    
    // Uncomment this line if you add console_log to features
    // console_log::init_with_level(log::Level::Info).expect("error initializing log");
}

/// Get the version of the kimi-fann-core crate
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Memory usage statistics for WASM
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: usize,
    /// Active experts in memory
    pub active_experts: usize,
    /// Cached experts count
    pub cached_experts: usize,
    /// Memory utilization ratio (0.0 to 1.0)
    pub utilization: f32,
}

#[wasm_bindgen]
impl MemoryStats {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            active_experts: 0,
            cached_experts: 0,
            utilization: 0.0,
        }
    }

    /// Convert to JSON string for JavaScript interop
    #[wasm_bindgen]
    pub fn to_json(&self) -> std::result::Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for WASM interop
#[wasm_bindgen]
pub struct Utils;

#[wasm_bindgen]
impl Utils {
    /// Get current timestamp in milliseconds
    #[wasm_bindgen]
    pub fn now() -> f64 {
        js_sys::Date::now()
    }

    /// Log a message to the console
    #[wasm_bindgen]
    pub fn log(message: &str) {
        web_sys::console::log_1(&message.into());
    }

    /// Generate a random expert ID
    #[wasm_bindgen]
    pub fn generate_expert_id() -> ExpertId {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        assert_eq!(metrics.execution_count, 0);
        assert_eq!(metrics.success_rate, 1.0);

        metrics.update(50.0, true, 1024);
        assert_eq!(metrics.execution_count, 1);
        assert_eq!(metrics.avg_inference_time, 50.0);
        assert_eq!(metrics.success_rate, 1.0);

        metrics.update(100.0, false, 2048);
        assert_eq!(metrics.execution_count, 2);
        assert_eq!(metrics.avg_inference_time, 75.0);
        assert_eq!(metrics.success_rate, 0.5);
    }

    #[test]
    fn test_specialization_parameter_ranges() {
        let spec = Specialization::CodeGeneration;
        let (min, max) = spec.parameter_range();
        assert!(min < max);
        assert!(min >= 1000);
        assert!(max <= 100_000);
    }

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats::new();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.active_experts, 0);
        assert_eq!(stats.utilization, 0.0);
    }
}