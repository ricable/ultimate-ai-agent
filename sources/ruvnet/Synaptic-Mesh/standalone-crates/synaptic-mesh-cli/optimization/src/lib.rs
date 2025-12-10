//! # Kimi-K2 Deep Optimization Suite
//! 
//! A comprehensive optimization framework for Rust-WASM neural network conversion
//! using real ruv-FANN implementations and advanced performance analysis.
//!
//! ## Features
//!
//! - **Real Neural Networks**: Actual ruv-FANN implementations, not mocks
//! - **WASM Optimization**: Advanced WebAssembly optimization techniques
//! - **Performance Benchmarking**: Comprehensive performance measurement suite
//! - **Memory Analysis**: Real-time memory usage tracking and optimization
//! - **Browser Compatibility**: Cross-browser validation and optimization
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  Optimization Suite                        │
//! ├─────────────────┬─────────────────┬─────────────────────────┤
//! │  Neural Engine  │  WASM Bindings  │   Performance Suite     │
//! │  (ruv-FANN)     │  (wasm-bindgen) │   (Benchmarks)          │
//! ├─────────────────┼─────────────────┼─────────────────────────┤
//! │  Memory Mgmt    │  Browser Compat │   Profiling Tools       │
//! │  (Optimization) │  (Validation)   │   (Real-time Metrics)   │
//! └─────────────────┴─────────────────┴─────────────────────────┘
//! ```

use wasm_bindgen::prelude::*;

// Re-export core functionality
pub use kimi_fann_core::{
    domains::ExpertDomain,
    expert::Expert,
    memory::MemoryManager,
    router::ExpertRouter,
    runtime::Runtime,
};

// Internal modules
pub mod neural;
pub mod wasm;
pub mod memory;
pub mod performance;
pub mod browser;

// Error handling
pub mod error;
pub use error::{OptimizationError, Result};

// Performance monitoring
pub mod metrics;
pub use metrics::{PerformanceMetrics, MetricsCollector};

// Benchmarking suite
#[cfg(feature = "benchmarking")]
pub mod benchmarks;

// Initialize WASM environment
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    
    #[cfg(feature = "debug-info")]
    console_log::init_with_level(log::Level::Debug).unwrap();
    
    #[cfg(not(feature = "debug-info"))]
    console_log::init_with_level(log::Level::Info).unwrap();
    
    log::info!("Kimi-K2 Optimization Suite initialized");
}

/// Main optimization coordinator
#[wasm_bindgen]
pub struct OptimizationSuite {
    neural_engine: neural::NeuralOptimizer,
    wasm_optimizer: wasm::WasmOptimizer,
    memory_tracker: memory::MemoryTracker,
    performance_monitor: performance::PerformanceMonitor,
    metrics_collector: MetricsCollector,
}

#[wasm_bindgen]
impl OptimizationSuite {
    /// Create a new optimization suite instance
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<OptimizationSuite> {
        Ok(OptimizationSuite {
            neural_engine: neural::NeuralOptimizer::new()?,
            wasm_optimizer: wasm::WasmOptimizer::new(),
            memory_tracker: memory::MemoryTracker::new(),
            performance_monitor: performance::PerformanceMonitor::new(),
            metrics_collector: MetricsCollector::new(),
        })
    }

    /// Run comprehensive optimization analysis
    #[wasm_bindgen]
    pub async fn run_optimization(&mut self) -> Result<JsValue> {
        log::info!("Starting comprehensive optimization analysis");
        
        let start_time = instant::Instant::now();
        
        // Neural network optimization
        let neural_metrics = self.neural_engine.optimize().await?;
        
        // WASM optimization
        let wasm_metrics = self.wasm_optimizer.optimize().await?;
        
        // Memory optimization
        let memory_metrics = self.memory_tracker.optimize().await?;
        
        // Performance analysis
        let performance_metrics = self.performance_monitor.analyze().await?;
        
        let total_time = start_time.elapsed();
        
        let results = OptimizationResults {
            neural: neural_metrics,
            wasm: wasm_metrics,
            memory: memory_metrics,
            performance: performance_metrics,
            total_duration_ms: total_time.as_millis() as u32,
        };
        
        log::info!("Optimization analysis completed in {}ms", total_time.as_millis());
        
        Ok(serde_wasm_bindgen::to_value(&results)?)
    }

    /// Get real-time performance metrics
    #[wasm_bindgen]
    pub fn get_metrics(&self) -> Result<JsValue> {
        let metrics = self.metrics_collector.current_metrics();
        Ok(serde_wasm_bindgen::to_value(&metrics)?)
    }

    /// Validate browser compatibility
    #[wasm_bindgen]
    pub async fn validate_browser_compatibility(&self) -> Result<JsValue> {
        let compatibility = browser::BrowserValidator::new().validate().await?;
        Ok(serde_wasm_bindgen::to_value(&compatibility)?)
    }
}

/// Results from optimization analysis
#[derive(serde::Serialize, serde::Deserialize)]
pub struct OptimizationResults {
    pub neural: neural::NeuralMetrics,
    pub wasm: wasm::WasmMetrics,
    pub memory: memory::MemoryMetrics,
    pub performance: performance::PerformanceData,
    pub total_duration_ms: u32,
}

/// Utility functions for optimization
pub mod utils {
    use super::*;
    
    /// Format file size in human-readable format
    pub fn format_bytes(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        format!("{:.2} {}", size, UNITS[unit_index])
    }
    
    /// Calculate percentage improvement
    pub fn calculate_improvement(old_value: f64, new_value: f64) -> f64 {
        if old_value == 0.0 {
            return 0.0;
        }
        ((old_value - new_value) / old_value) * 100.0
    }
}

// Global allocator for memory tracking
#[cfg(feature = "memory-tracking")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;