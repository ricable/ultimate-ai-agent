//! Performance regression detection benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cuda_rust_wasm::{
    transpiler::{CudaTranspiler, TranspilerOptions, OptimizationLevel},
    runtime::{WasmRuntime, RuntimeOptions},
    kernel::{KernelLauncher, LaunchConfig},
    memory::{DeviceMemory, MemoryPool, AllocationStrategy},
};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct PerformanceBaseline {
    test_name: String,
    mean_time_ns: u64,
    std_dev_ns: u64,
    throughput_ops_per_sec: f64,
    timestamp: u64,
    git_commit: Option<String>,
    build_config: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct BaselineDatabase {
    baselines: HashMap<String, PerformanceBaseline>,
    last_updated: u64,
}

const BASELINE_FILE: &str = "target/performance_baselines.json";
const REGRESSION_THRESHOLD: f64 = 0.05; // 5% regression threshold

fn load_baselines() -> BaselineDatabase {
    if Path::new(BASELINE_FILE).exists() {
        let content = fs::read_to_string(BASELINE_FILE).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        BaselineDatabase {
            baselines: HashMap::new(),
            last_updated: 0,
        }
    }
}

fn save_baselines(db: &BaselineDatabase) {
    if let Some(parent) = Path::new(BASELINE_FILE).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let content = serde_json::to_string_pretty(db).unwrap();
    fs::write(BASELINE_FILE, content).ok();
}

fn record_baseline(test_name: &str, duration: Duration, throughput: f64) {
    let mut db = load_baselines();
    
    let git_commit = std::process::Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|s| s.trim().to_string());
    
    let baseline = PerformanceBaseline {
        test_name: test_name.to_string(),
        mean_time_ns: duration.as_nanos() as u64,
        std_dev_ns: duration.as_nanos() as u64 / 20, // Estimate 5% std dev
        throughput_ops_per_sec: throughput,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        git_commit,
        build_config: if cfg!(debug_assertions) { "debug" } else { "release" }.to_string(),
    };
    
    db.baselines.insert(test_name.to_string(), baseline);
    db.last_updated = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    save_baselines(&db);
}

fn check_regression(test_name: &str, duration: Duration, throughput: f64) -> bool {
    let db = load_baselines();
    
    if let Some(baseline) = db.baselines.get(test_name) {
        let current_time_ns = duration.as_nanos() as u64;
        let baseline_time_ns = baseline.mean_time_ns;
        
        let regression_ratio = (current_time_ns as f64) / (baseline_time_ns as f64);
        let throughput_ratio = throughput / baseline.throughput_ops_per_sec;
        
        println!("\n=== Performance Regression Check ===");
        println!("Test: {}", test_name);
        println!("Baseline time: {} ns", baseline_time_ns);
        println!("Current time:  {} ns", current_time_ns);
        println!("Time ratio: {:.3}x", regression_ratio);
        println!("Baseline throughput: {:.2} ops/sec", baseline.throughput_ops_per_sec);
        println!("Current throughput:  {:.2} ops/sec", throughput);
        println!("Throughput ratio: {:.3}x", throughput_ratio);
        
        if regression_ratio > (1.0 + REGRESSION_THRESHOLD) {
            println!("⚠️  PERFORMANCE REGRESSION DETECTED! {:.1}% slower", 
                     (regression_ratio - 1.0) * 100.0);
            return true;
        } else if regression_ratio < (1.0 - REGRESSION_THRESHOLD) {
            println!("✅ Performance improvement: {:.1}% faster", 
                     (1.0 - regression_ratio) * 100.0);
        } else {
            println!("✅ Performance within acceptable range");
        }
    } else {
        println!("No baseline found for {}, recording new baseline", test_name);
        record_baseline(test_name, duration, throughput);
    }
    
    false
}

fn benchmark_with_regression_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_detection");
    
    // Standard vector addition benchmark
    let vector_add_code = r#"
        __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    "#;
    
    let size = 1000000; // 1M elements
    
    group.bench_function("vector_add_1m", |b| {
        let transpiler = CudaTranspiler::new(TranspilerOptions::default());
        let wasm_bytes = transpiler.transpile(vector_add_code).unwrap();
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let module = runtime.load_module(&wasm_bytes).unwrap();
        let launcher = KernelLauncher::new(module);
        
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 100 * 1024 * 1024).unwrap();
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        
        let d_a = pool.allocate_and_copy(&data).unwrap();
        let d_b = pool.allocate_and_copy(&data).unwrap();
        let d_c: DeviceMemory<f32> = pool.allocate(size).unwrap();
        
        let config = LaunchConfig {
            grid_size: ((size + 255) / 256, 1, 1),
            block_size: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let mut iteration_count = 0;
        let start_time = Instant::now();
        
        b.iter(|| {
            launcher.launch(
                "vector_add",
                config,
                &[d_a.as_arg(), d_b.as_arg(), d_c.as_arg(), size.as_arg()],
            ).unwrap();
            runtime.synchronize().unwrap();
            iteration_count += 1;
        });
        
        let total_duration = start_time.elapsed();
        let avg_duration = total_duration / iteration_count;
        let throughput = (size as f64) / avg_duration.as_secs_f64();
        
        check_regression("vector_add_1m", avg_duration, throughput);
    });
    
    // Matrix multiplication benchmark
    let matrix_mult_code = r#"
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
    
    let matrix_size = 256;
    
    group.bench_function("matrix_multiply_256x256", |b| {
        let transpiler = CudaTranspiler::new(TranspilerOptions::default());
        let wasm_bytes = transpiler.transpile(matrix_mult_code).unwrap();
        let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
        let module = runtime.load_module(&wasm_bytes).unwrap();
        let launcher = KernelLauncher::new(module);
        
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 200 * 1024 * 1024).unwrap();
        let elements = matrix_size * matrix_size;
        let data: Vec<f32> = (0..elements).map(|i| (i % 100) as f32).collect();
        
        let d_a = pool.allocate_and_copy(&data).unwrap();
        let d_b = pool.allocate_and_copy(&data).unwrap();
        let d_c: DeviceMemory<f32> = pool.allocate(elements).unwrap();
        
        let config = LaunchConfig {
            grid_size: ((matrix_size + 15) / 16, (matrix_size + 15) / 16, 1),
            block_size: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        
        let mut iteration_count = 0;
        let start_time = Instant::now();
        
        b.iter(|| {
            launcher.launch(
                "matrix_multiply",
                config,
                &[d_a.as_arg(), d_b.as_arg(), d_c.as_arg(), matrix_size.as_arg()],
            ).unwrap();
            runtime.synchronize().unwrap();
            iteration_count += 1;
        });
        
        let total_duration = start_time.elapsed();
        let avg_duration = total_duration / iteration_count;
        let operations = (matrix_size * matrix_size * matrix_size) as f64; // O(n³) operations
        let throughput = operations / avg_duration.as_secs_f64();
        
        check_regression("matrix_multiply_256x256", avg_duration, throughput);
    });
    
    // Memory bandwidth benchmark
    group.bench_function("memory_bandwidth", |b| {
        let pool = MemoryPool::new(AllocationStrategy::BestFit, 100 * 1024 * 1024).unwrap();
        let size = 10 * 1024 * 1024 / 4; // 10MB of f32 data
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        
        let d_src = pool.allocate_and_copy(&data).unwrap();
        let d_dst: DeviceMemory<f32> = pool.allocate(size).unwrap();
        
        let mut iteration_count = 0;
        let start_time = Instant::now();
        
        b.iter(|| {
            d_src.copy_to(black_box(&d_dst)).unwrap();
            iteration_count += 1;
        });
        
        let total_duration = start_time.elapsed();
        let avg_duration = total_duration / iteration_count;
        let bytes_transferred = (size * std::mem::size_of::<f32>()) as f64;
        let throughput = bytes_transferred / avg_duration.as_secs_f64(); // bytes/sec
        
        check_regression("memory_bandwidth", avg_duration, throughput);
    });
    
    // Compilation speed benchmark
    group.bench_function("compilation_speed", |b| {
        let complex_kernel = r#"
            #define BLOCK_SIZE 16
            
            __global__ void complex_computation(float* input, float* output, 
                                               int width, int height) {
                __shared__ float shared_data[BLOCK_SIZE][BLOCK_SIZE];
                
                int tx = threadIdx.x;
                int ty = threadIdx.y;
                int bx = blockIdx.x;
                int by = blockIdx.y;
                
                int x = bx * BLOCK_SIZE + tx;
                int y = by * BLOCK_SIZE + ty;
                int idx = y * width + x;
                
                if (x < width && y < height) {
                    shared_data[ty][tx] = input[idx];
                }
                __syncthreads();
                
                if (x < width && y < height) {
                    float result = 0.0f;
                    
                    // Complex computation
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        for (int j = 0; j < BLOCK_SIZE; j++) {
                            if (ty + i < BLOCK_SIZE && tx + j < BLOCK_SIZE) {
                                float val = shared_data[ty + i][tx + j];
                                result += __sinf(val) * __cosf(val * 2.0f);
                                result += __expf(-val * val) * __logf(val + 1.0f);
                            }
                        }
                    }
                    
                    output[idx] = result;
                }
            }
        "#;
        
        let mut iteration_count = 0;
        let start_time = Instant::now();
        
        b.iter(|| {
            let transpiler = CudaTranspiler::new(TranspilerOptions {
                optimization_level: OptimizationLevel::Aggressive,
                ..Default::default()
            });
            let _wasm_bytes = transpiler.transpile(black_box(complex_kernel)).unwrap();
            iteration_count += 1;
        });
        
        let total_duration = start_time.elapsed();
        let avg_duration = total_duration / iteration_count;
        let lines_of_code = complex_kernel.lines().count() as f64;
        let throughput = lines_of_code / avg_duration.as_secs_f64(); // lines/sec
        
        check_regression("compilation_speed", avg_duration, throughput);
    });
    
    group.finish();
}

fn benchmark_optimization_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_regression");
    
    let kernel_code = r#"
        __global__ void optimization_test(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float x = data[idx];
                
                // Operations that can be optimized
                x = x * 2.0f + 1.0f;
                x = __powf(x, 2.0f);
                x = __sqrtf(x);
                x = x / 2.0f - 0.5f;
                
                // Loop that can be unrolled
                for (int i = 0; i < 4; i++) {
                    x = x * 0.9f + 0.1f;
                }
                
                data[idx] = x;
            }
        }
    "#;
    
    let optimization_levels = vec![
        ("none", OptimizationLevel::None),
        ("basic", OptimizationLevel::Basic),
        ("aggressive", OptimizationLevel::Aggressive),
    ];
    
    for (name, level) in optimization_levels {
        group.bench_with_input(
            BenchmarkId::new("optimization_level", name),
            &level,
            |b, &level| {
                let options = TranspilerOptions {
                    optimization_level: level,
                    ..Default::default()
                };
                
                let transpiler = CudaTranspiler::new(options);
                let wasm_bytes = transpiler.transpile(kernel_code).unwrap();
                let runtime = WasmRuntime::new(RuntimeOptions::default()).unwrap();
                let module = runtime.load_module(&wasm_bytes).unwrap();
                let launcher = KernelLauncher::new(module);
                
                let size = 100000;
                let pool = MemoryPool::new(AllocationStrategy::BestFit, 10 * 1024 * 1024).unwrap();
                let data: Vec<f32> = (0..size).map(|i| (i as f32) / 1000.0).collect();
                let d_data = pool.allocate_and_copy(&data).unwrap();
                
                let config = LaunchConfig {
                    grid_size: ((size + 255) / 256, 1, 1),
                    block_size: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                
                let mut iteration_count = 0;
                let start_time = Instant::now();
                
                b.iter(|| {
                    launcher.launch(
                        "optimization_test",
                        config,
                        &[d_data.as_arg(), size.as_arg()],
                    ).unwrap();
                    runtime.synchronize().unwrap();
                    iteration_count += 1;
                });
                
                let total_duration = start_time.elapsed();
                let avg_duration = total_duration / iteration_count;
                let throughput = (size as f64) / avg_duration.as_secs_f64();
                
                let test_name = format!("optimization_{}", name);
                check_regression(&test_name, avg_duration, throughput);
            },
        );
    }
    
    group.finish();
}

fn generate_performance_report() {
    let db = load_baselines();
    
    println!("\n" + "=".repeat(60).as_str());
    println!("        PERFORMANCE REGRESSION REPORT");
    println!("=" .repeat(60));
    
    if db.baselines.is_empty() {
        println!("No performance baselines found.");
        println!("Run benchmarks to establish baselines.");
        return;
    }
    
    println!("Total benchmarks: {}", db.baselines.len());
    println!("Last updated: {}", 
             chrono::DateTime::from_timestamp(db.last_updated as i64, 0)
                 .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                 .unwrap_or_else(|| "Unknown".to_string()));
    
    println!("\nBenchmark Details:");
    println!("{:<25} {:<15} {:<15} {:<15} {:<10}", 
             "Test Name", "Mean Time", "Throughput", "Std Dev", "Commit");
    println!("-".repeat(80));
    
    for (name, baseline) in &db.baselines {
        let mean_time = if baseline.mean_time_ns > 1_000_000 {
            format!("{:.2} ms", baseline.mean_time_ns as f64 / 1_000_000.0)
        } else if baseline.mean_time_ns > 1_000 {
            format!("{:.2} μs", baseline.mean_time_ns as f64 / 1_000.0)
        } else {
            format!("{} ns", baseline.mean_time_ns)
        };
        
        let throughput = if baseline.throughput_ops_per_sec > 1_000_000.0 {
            format!("{:.2} Mops/s", baseline.throughput_ops_per_sec / 1_000_000.0)
        } else if baseline.throughput_ops_per_sec > 1_000.0 {
            format!("{:.2} Kops/s", baseline.throughput_ops_per_sec / 1_000.0)
        } else {
            format!("{:.2} ops/s", baseline.throughput_ops_per_sec)
        };
        
        let std_dev = if baseline.std_dev_ns > 1_000_000 {
            format!("{:.2} ms", baseline.std_dev_ns as f64 / 1_000_000.0)
        } else if baseline.std_dev_ns > 1_000 {
            format!("{:.2} μs", baseline.std_dev_ns as f64 / 1_000.0)
        } else {
            format!("{} ns", baseline.std_dev_ns)
        };
        
        let commit = baseline.git_commit.as_ref()
            .map(|c| &c[..8.min(c.len())])
            .unwrap_or("unknown");
        
        println!("{:<25} {:<15} {:<15} {:<15} {:<10}", 
                 name, mean_time, throughput, std_dev, commit);
    }
    
    println!("\nRegression threshold: {:.1}%", REGRESSION_THRESHOLD * 100.0);
    println!("Baseline file: {}", BASELINE_FILE);
    println!("=" .repeat(60));
}

criterion_group!(
    regression_benches,
    benchmark_with_regression_check,
    benchmark_optimization_regression
);

fn custom_main() {
    // Generate report before running benchmarks
    generate_performance_report();
    
    // Run the actual benchmarks
    regression_benches();
    
    // Generate report after running benchmarks
    println!("\n\nPost-benchmark report:");
    generate_performance_report();
}

criterion_main!(regression_benches);