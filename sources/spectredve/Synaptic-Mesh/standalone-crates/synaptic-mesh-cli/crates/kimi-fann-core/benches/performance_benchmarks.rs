//! Performance Benchmarks for Kimi-FANN Core
//! 
//! Comprehensive benchmarks testing neural network performance,
//! routing efficiency, and WASM execution speed.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kimi_fann_core::*;

fn expert_processing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("expert_processing");
    
    let domains = vec![
        ExpertDomain::Reasoning,
        ExpertDomain::Coding,
        ExpertDomain::Mathematics,
        ExpertDomain::Language,
        ExpertDomain::ToolUse,
        ExpertDomain::Context,
    ];
    
    for domain in domains {
        group.bench_with_input(
            BenchmarkId::new("single_expert", format!("{:?}", domain)),
            &domain,
            |b, &domain| {
                let expert = MicroExpert::new(domain);
                let query = "Benchmark test query for performance measurement";
                
                b.iter(|| {
                    black_box(expert.process(black_box(query)))
                });
            },
        );
    }
    
    group.finish();
}

fn router_performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("router_performance");
    
    // Test with different numbers of experts
    let expert_counts = vec![1, 5, 10, 25, 50];
    
    for count in expert_counts {
        group.bench_with_input(
            BenchmarkId::new("routing", count),
            &count,
            |b, &count| {
                let mut router = ExpertRouter::new();
                
                // Add experts
                for i in 0..count {
                    let domain = match i % 6 {
                        0 => ExpertDomain::Reasoning,
                        1 => ExpertDomain::Coding,
                        2 => ExpertDomain::Mathematics,
                        3 => ExpertDomain::Language,
                        4 => ExpertDomain::ToolUse,
                        _ => ExpertDomain::Context,
                    };
                    router.add_expert(MicroExpert::new(domain));
                }
                
                let query = "Route this query to the appropriate expert for processing";
                
                b.iter(|| {
                    black_box(router.route(black_box(query)))
                });
            },
        );
    }
    
    group.finish();
}

fn runtime_processing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_processing");
    
    let config = ProcessingConfig::new();
    let runtime = KimiRuntime::new(config);
    
    let queries = vec![
        "Simple query",
        "Complex mathematical calculation: solve the differential equation dy/dx = x^2 + 2x + 1",
        "Programming task: implement a binary search tree with insertion and deletion",
        "Reasoning problem: analyze the ethical implications of artificial intelligence",
        "Language task: translate and explain the cultural context of this phrase",
    ];
    
    for (i, query) in queries.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("query_complexity", i),
            query,
            |b, &query| {
                b.iter(|| {
                    black_box(runtime.process(black_box(query)))
                });
            },
        );
    }
    
    group.finish();
}

fn memory_efficiency_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    group.bench_function("expert_creation", |b| {
        b.iter(|| {
            black_box(MicroExpert::new(black_box(ExpertDomain::Coding)))
        });
    });
    
    group.bench_function("router_creation", |b| {
        b.iter(|| {
            black_box(ExpertRouter::new())
        });
    });
    
    group.bench_function("runtime_creation", |b| {
        let config = ProcessingConfig::new();
        b.iter(|| {
            black_box(KimiRuntime::new(black_box(config.clone())))
        });
    });
    
    group.finish();
}

fn throughput_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    
    let runtime = KimiRuntime::new(ProcessingConfig::new());
    
    group.bench_function("sequential_processing", |b| {
        let queries = vec![
            "Query 1", "Query 2", "Query 3", "Query 4", "Query 5",
            "Query 6", "Query 7", "Query 8", "Query 9", "Query 10",
        ];
        
        b.iter(|| {
            for query in &queries {
                black_box(runtime.process(black_box(query)));
            }
        });
    });
    
    group.finish();
}

fn scalability_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    
    // Test processing with different batch sizes
    let batch_sizes = vec![1, 10, 50, 100];
    
    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("batch_processing", batch_size),
            &batch_size,
            |b, &batch_size| {
                let runtime = KimiRuntime::new(ProcessingConfig::new());
                
                b.iter(|| {
                    for i in 0..batch_size {
                        let query = format!("Batch query number {}", i);
                        black_box(runtime.process(black_box(&query)));
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn expert_specialization_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("expert_specialization");
    
    let test_cases = vec![
        (ExpertDomain::Coding, "def quicksort(arr): pass"),
        (ExpertDomain::Mathematics, "Calculate the integral of sin(x) from 0 to pi"),
        (ExpertDomain::Reasoning, "What are the ethical implications of AI?"),
        (ExpertDomain::Language, "Translate this to French: Hello world"),
        (ExpertDomain::ToolUse, "How do I configure nginx for HTTPS?"),
        (ExpertDomain::Context, "Remember my previous conversation about databases"),
    ];
    
    for (domain, query) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("domain_specific", format!("{:?}", domain)),
            &(domain, query),
            |b, &(domain, query)| {
                let expert = MicroExpert::new(domain);
                
                b.iter(|| {
                    black_box(expert.process(black_box(query)))
                });
            },
        );
    }
    
    group.finish();
}

fn concurrent_processing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_processing");
    
    group.bench_function("parallel_experts", |b| {
        use std::sync::Arc;
        use std::thread;
        
        let runtime = Arc::new(KimiRuntime::new(ProcessingConfig::new()));
        
        b.iter(|| {
            let mut handles = vec![];
            
            for i in 0..4 {
                let runtime_clone = Arc::clone(&runtime);
                let handle = thread::spawn(move || {
                    let query = format!("Concurrent query {}", i);
                    runtime_clone.process(&query)
                });
                handles.push(handle);
            }
            
            let results: Vec<_> = handles.into_iter()
                .map(|h| h.join().unwrap())
                .collect();
            
            black_box(results)
        });
    });
    
    group.finish();
}

fn configuration_impact_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("configuration_impact");
    
    let configs = vec![
        ProcessingConfig { max_experts: 1, timeout_ms: 1000 },
        ProcessingConfig { max_experts: 3, timeout_ms: 5000 },
        ProcessingConfig { max_experts: 5, timeout_ms: 10000 },
        ProcessingConfig { max_experts: 10, timeout_ms: 30000 },
    ];
    
    for (i, config) in configs.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("max_experts", config.max_experts),
            config,
            |b, config| {
                let runtime = KimiRuntime::new(config.clone());
                let query = "Test query for configuration impact measurement";
                
                b.iter(|| {
                    black_box(runtime.process(black_box(query)))
                });
            },
        );
    }
    
    group.finish();
}

fn wasm_specific_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_specific");
    
    // Test serialization performance (important for WASM)
    group.bench_function("config_serialization", |b| {
        let config = ExpertConfig {
            domain: ExpertDomain::Coding,
            parameter_count: 50000,
            learning_rate: 0.01,
        };
        
        b.iter(|| {
            let serialized = serde_json::to_string(&config).unwrap();
            let _deserialized: ExpertConfig = serde_json::from_str(&serialized).unwrap();
            black_box(())
        });
    });
    
    // Test domain enum operations (frequent in WASM)
    group.bench_function("domain_operations", |b| {
        let domains = vec![
            ExpertDomain::Reasoning,
            ExpertDomain::Coding,
            ExpertDomain::Mathematics,
            ExpertDomain::Language,
            ExpertDomain::ToolUse,
            ExpertDomain::Context,
        ];
        
        b.iter(|| {
            for domain in &domains {
                black_box(*domain);
                let serialized = serde_json::to_string(domain).unwrap();
                let _deserialized: ExpertDomain = serde_json::from_str(&serialized).unwrap();
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    expert_processing_benchmark,
    router_performance_benchmark,
    runtime_processing_benchmark,
    memory_efficiency_benchmark,
    throughput_benchmark,
    scalability_benchmark,
    expert_specialization_benchmark,
    concurrent_processing_benchmark,
    configuration_impact_benchmark,
    wasm_specific_benchmark
);

criterion_main!(benches);