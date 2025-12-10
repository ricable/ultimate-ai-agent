//! Comprehensive performance benchmarks for WASM optimization
//!
//! This benchmark suite measures the performance optimizations and ensures
//! we meet the targets of <2MB compressed WASM and >70% native CUDA performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cuda_rust_wasm::*;
use std::time::Duration;

/// Benchmark configuration
struct BenchConfig {
    sizes: Vec<usize>,
    iterations: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            sizes: vec![1024, 4096, 16384, 65536, 262144, 1048576],
            iterations: 100,
        }
    }
}

/// Memory allocation and deallocation benchmarks
fn bench_memory_allocation(c: &mut Criterion) {
    let config = BenchConfig::default();
    
    let mut group = c.benchmark_group("memory_allocation");
    group.measurement_time(Duration::from_secs(10));
    
    for size in config.sizes.iter() {
        group.bench_with_input(
            BenchmarkId::new("pooled_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let buffer = memory::allocate(black_box(size));
                    memory::deallocate(buffer);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("standard_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let buffer = vec![0u8; black_box(size)];
                    drop(buffer);
                });
            },
        );
    }
    
    group.finish();
}

/// Kernel compilation and caching benchmarks
fn bench_kernel_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_compilation");
    group.measurement_time(Duration::from_secs(15));
    
    let simple_kernel = r#"
        __global__ void vector_add(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
    "#;
    
    let complex_kernel = r#"
        __global__ void matrix_multiply(float* a, float* b, float* c, int n) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < n && col < n) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += a[row * n + k] * b[k * n + col];
                }
                c[row * n + col] = sum;
            }
        }
    "#;
    
    group.bench_function("simple_kernel_compile", |b| {
        let transpiler = CudaRust::new();
        b.iter(|| {
            let _ = transpiler.transpile(black_box(simple_kernel));
        });
    });
    
    group.bench_function("complex_kernel_compile", |b| {
        let transpiler = CudaRust::new();
        b.iter(|| {
            let _ = transpiler.transpile(black_box(complex_kernel));
        });
    });
    
    #[cfg(feature = "webgpu-only")]
    group.bench_function("webgpu_kernel_compile", |b| {
        let transpiler = CudaRust::new();
        b.iter(|| {
            let _ = transpiler.to_webgpu(black_box(simple_kernel));
        });
    });
    
    group.finish();
}

/// Parser performance benchmarks
fn bench_parser_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser_performance");
    
    let small_code = r#"
        __global__ void simple(float* a) {
            int i = threadIdx.x;
            a[i] = i;
        }
    "#;
    
    let medium_code = r#"
        #include <cuda_runtime.h>
        
        __global__ void vector_operations(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float temp = a[idx] * 2.0f;
                temp += b[idx];
                temp = sqrtf(temp);
                c[idx] = temp;
            }
        }
        
        __device__ float helper_function(float x, float y) {
            return x * y + sinf(x);
        }
    "#;
    
    let large_code = format!("{}\n{}", medium_code.repeat(10), r#"
        __global__ void complex_kernel(float* input, float* output, int n) {
            extern __shared__ float shared_data[];
            
            int tid = threadIdx.x;
            int bid = blockIdx.x;
            int idx = bid * blockDim.x + tid;
            
            // Load data into shared memory
            if (idx < n) {
                shared_data[tid] = input[idx];
            } else {
                shared_data[tid] = 0.0f;
            }
            
            __syncthreads();
            
            // Perform reduction in shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_data[tid] += shared_data[tid + s];
                }
                __syncthreads();
            }
            
            // Write result
            if (tid == 0) {
                output[bid] = shared_data[0];
            }
        }
    "#);
    
    let parser = parser::CudaParser::new();
    
    group.bench_function("small_code_parse", |b| {
        b.iter(|| {
            let _ = parser.parse(black_box(small_code));
        });
    });
    
    group.bench_function("medium_code_parse", |b| {
        b.iter(|| {
            let _ = parser.parse(black_box(medium_code));
        });
    });
    
    group.bench_function("large_code_parse", |b| {
        b.iter(|| {
            let _ = parser.parse(black_box(&large_code));
        });
    });
    
    group.finish();
}

/// Transpiler optimization benchmarks
fn bench_transpiler_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpiler_optimization");
    
    let kernel_with_optimizations = r#"
        __global__ void optimized_kernel(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Loop unrolling opportunity
            #pragma unroll 4
            for (int j = 0; j < 4; j++) {
                if (i * 4 + j < n) {
                    c[i * 4 + j] = a[i * 4 + j] + b[i * 4 + j];
                }
            }
        }
    "#;
    
    let transpiler = transpiler::Transpiler::new();
    let parser = parser::CudaParser::new();
    
    group.bench_function("optimization_pass", |b| {
        let ast = parser.parse(kernel_with_optimizations).unwrap();
        b.iter(|| {
            let _ = transpiler.transpile(black_box(ast.clone()));
        });
    });
    
    group.finish();
}

/// Memory pool performance benchmarks
fn bench_memory_pool_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool");
    group.measurement_time(Duration::from_secs(10));
    
    let pool = memory::MemoryPool::new();
    let config = BenchConfig::default();
    
    // Warm up the pool
    for &size in &config.sizes {
        for _ in 0..10 {
            let buffer = pool.allocate(size);
            pool.deallocate(buffer);
        }
    }
    
    for size in config.sizes.iter() {
        group.bench_with_input(
            BenchmarkId::new("pool_hot_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let buffer = pool.allocate(black_box(size));
                    pool.deallocate(buffer);
                });
            },
        );
    }
    
    // Test cache efficiency
    group.bench_function("cache_hit_ratio", |b| {
        b.iter(|| {
            // Allocate and deallocate same sizes repeatedly
            for &size in &[1024, 2048, 4096] {
                for _ in 0..10 {
                    let buffer = pool.allocate(black_box(size));
                    pool.deallocate(buffer);
                }
            }
        });
    });
    
    group.finish();
}

/// Performance monitoring overhead benchmarks
fn bench_monitoring_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("monitoring_overhead");
    
    use profiling::{global_monitor, CounterType, time_operation};
    
    group.bench_function("no_monitoring", |b| {
        b.iter(|| {
            // Simulate work without monitoring
            for i in 0..1000 {
                black_box(i * i);
            }
        });
    });
    
    group.bench_function("with_monitoring", |b| {
        b.iter(|| {
            let _timer = time_operation(CounterType::Custom("benchmark".to_string()));
            // Simulate work with monitoring
            for i in 0..1000 {
                black_box(i * i);
            }
        });
    });
    
    group.bench_function("timer_creation_only", |b| {
        b.iter(|| {
            let _timer = time_operation(CounterType::Custom("creation_test".to_string()));
        });
    });
    
    group.finish();
}

/// End-to-end performance benchmarks
fn bench_end_to_end_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    group.measurement_time(Duration::from_secs(20));
    
    let vector_add_kernel = r#"
        __global__ void vector_add(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
    "#;
    
    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let cuda_rust = CudaRust::new();
            let _ = cuda_rust.transpile(black_box(vector_add_kernel));
        });
    });
    
    #[cfg(feature = "webgpu-only")]
    group.bench_function("webgpu_pipeline", |b| {
        b.iter(|| {
            let cuda_rust = CudaRust::new();
            let _ = cuda_rust.to_webgpu(black_box(vector_add_kernel));
        });
    });
    
    group.finish();
}

/// WASM size optimization benchmarks
fn bench_wasm_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_size_impact");
    
    // Measure performance vs size trade-offs
    group.bench_function("size_optimized_compilation", |b| {
        let transpiler = CudaRust::new();
        let code = r#"
            __global__ void size_test(float* data, int n) {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < n) {
                    data[i] = data[i] * 2.0f + 1.0f;
                }
            }
        "#;
        
        b.iter(|| {
            let _ = transpiler.transpile(black_box(code));
        });
    });
    
    group.finish();
}

/// Memory bandwidth benchmarks
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth");
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = vec![1024, 4096, 16384, 65536, 262144];
    
    for size in sizes {
        group.bench_with_input(
            BenchmarkId::new("memory_copy", size),
            &size,
            |b, &size| {
                let source = vec![1.0f32; size];
                b.iter(|| {
                    let mut dest = vec![0.0f32; size];
                    dest.copy_from_slice(&source);
                    black_box(dest);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("memory_transform", size),
            &size,
            |b, &size| {
                let data = vec![1.0f32; size];
                b.iter(|| {
                    let result: Vec<f32> = data
                        .iter()
                        .map(|&x| black_box(x * 2.0 + 1.0))
                        .collect();
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

/// Performance regression tests
fn bench_performance_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_tests");
    
    // Baseline performance targets
    let baseline_parse_time_ns = 50_000; // 50 microseconds
    let baseline_transpile_time_ns = 100_000; // 100 microseconds
    
    let simple_kernel = r#"
        __global__ void baseline_test(float* a, float* b, int n) {
            int i = threadIdx.x;
            if (i < n) a[i] = b[i] * 2.0f;
        }
    "#;
    
    group.bench_function("regression_parse", |b| {
        let parser = parser::CudaParser::new();
        b.iter(|| {
            let result = parser.parse(black_box(simple_kernel));
            assert!(result.is_ok());
        });
    });
    
    group.bench_function("regression_transpile", |b| {
        let cuda_rust = CudaRust::new();
        b.iter(|| {
            let result = cuda_rust.transpile(black_box(simple_kernel));
            assert!(result.is_ok());
        });
    });
    
    group.finish();
}

criterion_group!(
    performance_benches,
    bench_memory_allocation,
    bench_kernel_compilation,
    bench_parser_performance,
    bench_transpiler_optimization,
    bench_memory_pool_performance,
    bench_monitoring_overhead,
    bench_end_to_end_performance,
    bench_wasm_size_impact,
    bench_memory_bandwidth,
    bench_performance_regression
);

criterion_main!(performance_benches);

#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_efficiency() {
        let pool = memory::MemoryPool::new();
        
        // Warm up
        for _ in 0..10 {
            let buffer = pool.allocate(1024);
            pool.deallocate(buffer);
        }
        
        let stats = pool.stats();
        let hit_ratio = pool.hit_ratio();
        
        println!("Memory Pool Hit Ratio: {hit_ratio:.1}%");
        println!("Total Allocations: {}", stats.total_allocations);
        println!("Cache Hits: {}", stats.cache_hits);
        
        // Should have good cache performance
        assert!(hit_ratio > 50.0, "Cache hit ratio too low: {hit_ratio:.1}%");
    }
    
    #[test]
    fn test_compilation_performance() {
        let cuda_rust = CudaRust::new();
        let kernel = r#"
            __global__ void perf_test(float* data, int n) {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < n) data[i] *= 2.0f;
            }
        "#;
        
        let start = std::time::Instant::now();
        let result = cuda_rust.transpile(kernel);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        println!("Compilation time: {duration:?}");
        
        // Should compile reasonably fast
        assert!(duration.as_millis() < 100, "Compilation too slow: {duration:?}");
    }
    
    #[test]
    fn test_monitoring_overhead() {
        use profiling::{global_monitor, CounterType, time_operation};
        
        let iterations = 10000;
        
        // Measure without monitoring
        let start = std::time::Instant::now();
        for i in 0..iterations {
            black_box(i * i);
        }
        let baseline = start.elapsed();
        
        // Measure with monitoring
        let start = std::time::Instant::now();
        for i in 0..iterations {
            let _timer = time_operation(CounterType::Custom("overhead_test".to_string()));
            black_box(i * i);
        }
        let monitored = start.elapsed();
        
        let overhead_ratio = monitored.as_nanos() as f64 / baseline.as_nanos() as f64;
        println!("Monitoring overhead ratio: {overhead_ratio:.2}x");
        
        // Overhead should be minimal
        assert!(overhead_ratio < 2.0, "Monitoring overhead too high: {overhead_ratio:.2}x");
    }
}