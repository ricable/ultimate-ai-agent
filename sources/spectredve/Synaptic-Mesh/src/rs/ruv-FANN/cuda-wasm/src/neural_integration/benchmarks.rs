//! Comprehensive Benchmarks for Neural Integration
//!
//! This module provides extensive benchmarking capabilities to measure
//! and compare performance between CPU and GPU implementations.

use super::{
    ActivationFunction, BridgeConfig, NeuralBridge, NeuralOperation, NeuralResult,
    GpuDevice, Precision,
};
use std::time::{Duration, Instant};

/// Benchmark results for a single operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation_name: String,
    pub input_size: usize,
    pub gpu_time: Duration,
    pub cpu_time: Duration,
    pub gpu_throughput: f64,
    pub cpu_throughput: f64,
    pub speedup_factor: f64,
    pub memory_usage: usize,
}

/// Comprehensive benchmark suite
pub struct BenchmarkSuite {
    gpu_bridge: NeuralBridge,
    cpu_bridge: NeuralBridge,
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> NeuralResult<Self> {
        // Create GPU-enabled bridge
        let gpu_config = BridgeConfig {
            enable_gpu: true,
            gpu_device: GpuDevice::HighPerformance,
            memory_pool_size: 1024, // 1GB
            enable_monitoring: true,
            auto_fallback: false, // Don't fallback to CPU
            batch_size: 64,
            precision: Precision::Float32,
        };
        
        // Create CPU-only bridge
        let cpu_config = BridgeConfig {
            enable_gpu: false,
            auto_fallback: false,
            enable_monitoring: true,
            ..gpu_config.clone()
        };
        
        let gpu_bridge = NeuralBridge::with_config(gpu_config)?;
        let cpu_bridge = NeuralBridge::with_config(cpu_config)?;
        
        Ok(Self {
            gpu_bridge,
            cpu_bridge,
            results: Vec::new(),
        })
    }
    
    /// Run comprehensive benchmarks
    pub fn run_comprehensive_benchmarks(&mut self) -> NeuralResult<()> {
        println!("Running Comprehensive Neural Integration Benchmarks");
        println!("=================================================");
        
        self.benchmark_vector_operations()?;
        self.benchmark_matrix_operations()?;
        self.benchmark_activation_functions()?;
        self.benchmark_neural_networks()?;
        self.benchmark_batch_operations()?;
        
        self.print_summary();
        
        Ok(())
    }
    
    /// Benchmark vector operations
    fn benchmark_vector_operations(&mut self) -> NeuralResult<()> {
        println!("\n--- Vector Operations Benchmark ---");
        
        let sizes = vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000];
        
        for size in sizes {
            let result = self.benchmark_vector_add(size)?;
            self.results.push(result.clone());
            
            println!(
                "Vector Add ({}): GPU {:.2}ms ({:.0} Mops/s), CPU {:.2}ms ({:.0} Mops/s), Speedup: {:.2}x",
                format_size(size),
                result.gpu_time.as_secs_f64() * 1000.0,
                result.gpu_throughput / 1e6,
                result.cpu_time.as_secs_f64() * 1000.0,
                result.cpu_throughput / 1e6,
                result.speedup_factor
            );
        }
        
        Ok(())
    }
    
    /// Benchmark matrix operations
    fn benchmark_matrix_operations(&mut self) -> NeuralResult<()> {
        println!("\n--- Matrix Operations Benchmark ---");
        
        let sizes = vec![64, 128, 256, 512, 1024];
        
        for size in sizes {
            let result = self.benchmark_matrix_multiply(size)?;
            self.results.push(result.clone());
            
            let gflops_gpu = calculate_matrix_gflops(size, result.gpu_time);
            let gflops_cpu = calculate_matrix_gflops(size, result.cpu_time);
            
            println!(
                "Matrix {}x{}: GPU {:.2}ms ({:.1} GFLOPS), CPU {:.2}ms ({:.1} GFLOPS), Speedup: {:.2}x",
                size, size,
                result.gpu_time.as_secs_f64() * 1000.0,
                gflops_gpu,
                result.cpu_time.as_secs_f64() * 1000.0,
                gflops_cpu,
                result.speedup_factor
            );
        }
        
        Ok(())
    }
    
    /// Benchmark activation functions
    fn benchmark_activation_functions(&mut self) -> NeuralResult<()> {
        println!("\n--- Activation Functions Benchmark ---");
        
        let functions = vec![
            ("Sigmoid", ActivationFunction::Sigmoid),
            ("ReLU", ActivationFunction::ReLU),
            ("Tanh", ActivationFunction::Tanh),
            ("GELU", ActivationFunction::GELU),
            ("Swish", ActivationFunction::Swish),
        ];
        
        let size = 1_000_000;
        
        for (name, function) in functions {
            let result = self.benchmark_activation_function(function, size)?;
            self.results.push(result.clone());
            
            println!(
                "{:>8} ({}): GPU {:.2}ms ({:.0} Mops/s), CPU {:.2}ms ({:.0} Mops/s), Speedup: {:.2}x",
                name,
                format_size(size),
                result.gpu_time.as_secs_f64() * 1000.0,
                result.gpu_throughput / 1e6,
                result.cpu_time.as_secs_f64() * 1000.0,
                result.cpu_throughput / 1e6,
                result.speedup_factor
            );
        }
        
        Ok(())
    }
    
    /// Benchmark neural network operations
    fn benchmark_neural_networks(&mut self) -> NeuralResult<()> {
        println!("\n--- Neural Network Benchmark ---");
        
        let networks = vec![
            ("Small", vec![10, 20, 10]),
            ("Medium", vec![100, 200, 100, 50]),
            ("Large", vec![784, 1000, 500, 250, 10]),
            ("Deep", vec![100, 100, 100, 100, 100, 100, 10]),
        ];
        
        for (name, layer_sizes) in networks {
            let result = self.benchmark_neural_network(&layer_sizes)?;
            self.results.push(result.clone());
            
            println!(
                "{:>6} ({:?}): GPU {:.2}ms, CPU {:.2}ms, Speedup: {:.2}x",
                name,
                layer_sizes,
                result.gpu_time.as_secs_f64() * 1000.0,
                result.cpu_time.as_secs_f64() * 1000.0,
                result.speedup_factor
            );
        }
        
        Ok(())
    }
    
    /// Benchmark batch operations
    fn benchmark_batch_operations(&mut self) -> NeuralResult<()> {
        println!("\n--- Batch Operations Benchmark ---");
        
        let batch_sizes = vec![1, 8, 32, 128, 512];
        let operation_size = 10_000;
        
        for batch_size in batch_sizes {
            let result = self.benchmark_batch_processing(batch_size, operation_size)?;
            self.results.push(result.clone());
            
            println!(
                "Batch size {:3}: GPU {:.2}ms ({:.0} ops/s), CPU {:.2}ms ({:.0} ops/s), Speedup: {:.2}x",
                batch_size,
                result.gpu_time.as_secs_f64() * 1000.0,
                result.gpu_throughput,
                result.cpu_time.as_secs_f64() * 1000.0,
                result.cpu_throughput,
                result.speedup_factor
            );
        }
        
        Ok(())
    }
    
    /// Benchmark vector addition
    fn benchmark_vector_add(&self, size: usize) -> NeuralResult<BenchmarkResult> {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        let mut input_data = a;
        input_data.extend(b);
        
        let operation = NeuralOperation::VectorAdd { size, _phantom: std::marker::PhantomData };
        
        // GPU benchmark
        let gpu_time = self.time_operation(&self.gpu_bridge, operation.clone(), &input_data)?;
        
        // CPU benchmark
        let cpu_time = self.time_operation(&self.cpu_bridge, operation, &input_data)?;
        
        let gpu_throughput = size as f64 / gpu_time.as_secs_f64();
        let cpu_throughput = size as f64 / cpu_time.as_secs_f64();
        let speedup_factor = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        
        Ok(BenchmarkResult {
            operation_name: "vector_add".to_string(),
            input_size: size,
            gpu_time,
            cpu_time,
            gpu_throughput,
            cpu_throughput,
            speedup_factor,
            memory_usage: size * 4 * 3, // 3 vectors of f32
        })
    }
    
    /// Benchmark matrix multiplication
    fn benchmark_matrix_multiply(&self, size: usize) -> NeuralResult<BenchmarkResult> {
        let matrix_a: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
        let matrix_b: Vec<f32> = (0..size * size).map(|i| (i * 2) as f32).collect();
        let mut input_data = matrix_a;
        input_data.extend(matrix_b);
        
        let operation = NeuralOperation::MatrixMultiply {
            a_rows: size,
            a_cols: size,
            b_cols: size,
            _phantom: std::marker::PhantomData,
        };
        
        // GPU benchmark
        let gpu_time = self.time_operation(&self.gpu_bridge, operation.clone(), &input_data)?;
        
        // CPU benchmark
        let cpu_time = self.time_operation(&self.cpu_bridge, operation, &input_data)?;
        
        let operations = 2.0 * (size as f64).powi(3); // 2 * n^3 FLOPs
        let gpu_throughput = operations / gpu_time.as_secs_f64();
        let cpu_throughput = operations / cpu_time.as_secs_f64();
        let speedup_factor = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        
        Ok(BenchmarkResult {
            operation_name: "matrix_multiply".to_string(),
            input_size: size * size,
            gpu_time,
            cpu_time,
            gpu_throughput,
            cpu_throughput,
            speedup_factor,
            memory_usage: size * size * 4 * 3, // 3 matrices of f32
        })
    }
    
    /// Benchmark activation function
    fn benchmark_activation_function(&self, function: ActivationFunction, size: usize) -> NeuralResult<BenchmarkResult> {
        let input_data: Vec<f32> = (0..size).map(|i| (i as f32) / 1000.0 - 5.0).collect();
        
        let operation = NeuralOperation::ActivationFunction { function, size, _phantom: std::marker::PhantomData };
        
        // GPU benchmark
        let gpu_time = self.time_operation(&self.gpu_bridge, operation.clone(), &input_data)?;
        
        // CPU benchmark
        let cpu_time = self.time_operation(&self.cpu_bridge, operation, &input_data)?;
        
        let gpu_throughput = size as f64 / gpu_time.as_secs_f64();
        let cpu_throughput = size as f64 / cpu_time.as_secs_f64();
        let speedup_factor = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        
        Ok(BenchmarkResult {
            operation_name: format!("activation_{function:?}"),
            input_size: size,
            gpu_time,
            cpu_time,
            gpu_throughput,
            cpu_throughput,
            speedup_factor,
            memory_usage: size * 4 * 2, // Input and output f32 vectors
        })
    }
    
    /// Benchmark neural network forward propagation
    fn benchmark_neural_network(&self, layer_sizes: &[usize]) -> NeuralResult<BenchmarkResult> {
        let input_size = layer_sizes[0];
        let input_data: Vec<f32> = (0..input_size).map(|i| (i as f32) / input_size as f32).collect();
        
        let operation = NeuralOperation::ForwardPropagation {
            layer_sizes: layer_sizes.to_vec(),
            _phantom: std::marker::PhantomData,
        };
        
        // GPU benchmark
        let gpu_time = self.time_operation(&self.gpu_bridge, operation.clone(), &input_data)?;
        
        // CPU benchmark
        let cpu_time = self.time_operation(&self.cpu_bridge, operation, &input_data)?;
        
        let total_params: usize = layer_sizes.windows(2).map(|w| w[0] * w[1]).sum();
        let gpu_throughput = total_params as f64 / gpu_time.as_secs_f64();
        let cpu_throughput = total_params as f64 / cpu_time.as_secs_f64();
        let speedup_factor = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        
        Ok(BenchmarkResult {
            operation_name: "neural_network".to_string(),
            input_size: total_params,
            gpu_time,
            cpu_time,
            gpu_throughput,
            cpu_throughput,
            speedup_factor,
            memory_usage: total_params * 4, // Approximate memory usage
        })
    }
    
    /// Benchmark batch processing
    fn benchmark_batch_processing(&self, batch_size: usize, operation_size: usize) -> NeuralResult<BenchmarkResult> {
        let operations: Vec<_> = (0..batch_size)
            .map(|_| NeuralOperation::VectorAdd { size: operation_size, _phantom: std::marker::PhantomData })
            .collect();
        
        let inputs: Vec<_> = (0..batch_size)
            .map(|_| {
                let mut data: Vec<f32> = (0..operation_size).map(|i| i as f32).collect();
                data.extend((0..operation_size).map(|i| (i * 2) as f32));
                data
            })
            .collect();
        
        // GPU benchmark
        let gpu_start = Instant::now();
        let gpu_processor = self.gpu_bridge.create_batch_processor();
        let _gpu_results = gpu_processor.process_batch(operations.clone(), inputs.clone())?;
        let gpu_time = gpu_start.elapsed();
        
        // CPU benchmark  
        let cpu_start = Instant::now();
        let cpu_processor = self.cpu_bridge.create_batch_processor();
        let _cpu_results = cpu_processor.process_batch(operations, inputs)?;
        let cpu_time = cpu_start.elapsed();
        
        let gpu_throughput = batch_size as f64 / gpu_time.as_secs_f64();
        let cpu_throughput = batch_size as f64 / cpu_time.as_secs_f64();
        let speedup_factor = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        
        Ok(BenchmarkResult {
            operation_name: "batch_processing".to_string(),
            input_size: batch_size,
            gpu_time,
            cpu_time,
            gpu_throughput,
            cpu_throughput,
            speedup_factor,
            memory_usage: batch_size * operation_size * 4 * 3,
        })
    }
    
    /// Time a single operation
    fn time_operation(
        &self,
        bridge: &NeuralBridge,
        operation: NeuralOperation<f32>,
        input_data: &[f32],
    ) -> NeuralResult<Duration> {
        // Warm up
        for _ in 0..3 {
            let _ = bridge.execute_neural_operation(operation.clone(), input_data)?;
        }
        
        // Actual timing
        let iterations = 10;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = bridge.execute_neural_operation(operation.clone(), input_data)?;
        }
        
        let total_time = start.elapsed();
        Ok(total_time / iterations)
    }
    
    /// Print benchmark summary
    fn print_summary(&self) {
        println!("\n=== Benchmark Summary ===");
        
        let mut operation_groups: std::collections::HashMap<String, Vec<&BenchmarkResult>> = std::collections::HashMap::new();
        
        for result in &self.results {
            let base_name = result.operation_name.split('_').next().unwrap_or(&result.operation_name);
            operation_groups.entry(base_name.to_string()).or_default().push(result);
        }
        
        for (operation_type, results) in operation_groups {
            let avg_speedup: f64 = results.iter().map(|r| r.speedup_factor).sum::<f64>() / results.len() as f64;
            let max_speedup = results.iter().map(|r| r.speedup_factor).fold(0.0, f64::max);
            let min_speedup = results.iter().map(|r| r.speedup_factor).fold(f64::INFINITY, f64::min);
            
            println!(
                "{:>15}: Avg {:.2}x, Max {:.2}x, Min {:.2}x speedup ({} tests)",
                operation_type, avg_speedup, max_speedup, min_speedup, results.len()
            );
        }
        
        let overall_avg_speedup: f64 = self.results.iter().map(|r| r.speedup_factor).sum::<f64>() / self.results.len() as f64;
        println!("\nOverall Average Speedup: {overall_avg_speedup:.2}x");
        
        // Memory usage summary
        let total_memory: usize = self.results.iter().map(|r| r.memory_usage).sum();
        println!("Total Memory Tested: {}", format_bytes(total_memory));
        
        // Performance recommendations
        println!("\n=== Performance Recommendations ===");
        let best_operations: Vec<_> = self.results.iter()
            .filter(|r| r.speedup_factor > 5.0)
            .collect();
        
        if !best_operations.is_empty() {
            println!("Best GPU operations (>5x speedup):");
            for result in best_operations {
                println!("  - {} ({:.1}x speedup)", result.operation_name, result.speedup_factor);
            }
        }
        
        let poor_operations: Vec<_> = self.results.iter()
            .filter(|r| r.speedup_factor < 1.5)
            .collect();
        
        if !poor_operations.is_empty() {
            println!("Operations better on CPU (<1.5x speedup):");
            for result in poor_operations {
                println!("  - {} ({:.1}x speedup)", result.operation_name, result.speedup_factor);
            }
        }
    }
    
    /// Export results as CSV
    pub fn export_csv(&self, filename: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(filename)?;
        
        // CSV header
        writeln!(
            file,
            "Operation,InputSize,GPUTimeMs,CPUTimeMs,GPUThroughput,CPUThroughput,SpeedupFactor,MemoryUsage"
        )?;
        
        // CSV data
        for result in &self.results {
            writeln!(
                file,
                "{},{},{:.6},{:.6},{:.2},{:.2},{:.2},{}",
                result.operation_name,
                result.input_size,
                result.gpu_time.as_secs_f64() * 1000.0,
                result.cpu_time.as_secs_f64() * 1000.0,
                result.gpu_throughput,
                result.cpu_throughput,
                result.speedup_factor,
                result.memory_usage
            )?;
        }
        
        println!("Benchmark results exported to {filename}");
        Ok(())
    }
}

/// Helper function to calculate matrix multiplication GFLOPS
fn calculate_matrix_gflops(size: usize, time: Duration) -> f64 {
    let flops = 2.0 * (size as f64).powi(3); // 2 * n^3 operations
    flops / time.as_secs_f64() / 1e9
}

/// Helper function to format sizes
fn format_size(size: usize) -> String {
    if size >= 1_000_000 {
        format!("{}M", size / 1_000_000)
    } else if size >= 1_000 {
        format!("{}K", size / 1_000)
    } else {
        size.to_string()
    }
}

/// Helper function to format bytes
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.1} {}", size, UNITS[unit_index])
}

/// Run quick benchmark (for testing)
pub fn run_quick_benchmark() -> NeuralResult<()> {
    println!("Running Quick Benchmark...");
    
    let suite = BenchmarkSuite::new()?;
    
    // Run a subset of benchmarks
    let result = suite.benchmark_vector_add(10_000)?;
    println!("Vector Add 10K: {:.2}x speedup", result.speedup_factor);
    
    let result = suite.benchmark_matrix_multiply(128)?;
    println!("Matrix 128x128: {:.2}x speedup", result.speedup_factor);
    
    let result = suite.benchmark_activation_function(ActivationFunction::ReLU, 100_000)?;
    println!("ReLU 100K: {:.2}x speedup", result.speedup_factor);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new();
        assert!(suite.is_ok(), "Failed to create benchmark suite");
    }
    
    #[test]
    fn test_quick_benchmark() {
        let result = run_quick_benchmark();
        assert!(result.is_ok(), "Quick benchmark failed: {result:?}");
    }
    
    #[test]
    fn test_format_functions() {
        assert_eq!(format_size(1_000), "1K");
        assert_eq!(format_size(1_500_000), "1M");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
    }
}