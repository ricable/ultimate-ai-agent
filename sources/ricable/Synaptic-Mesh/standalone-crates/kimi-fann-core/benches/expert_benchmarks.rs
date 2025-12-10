//! Performance benchmarks for Kimi-FANN Core
//!
//! This benchmark suite measures the performance of various components
//! including expert execution, routing, memory management, and compression.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kimi_fann_core::*;
use std::collections::HashMap;
use std::time::Duration;

/// Benchmark expert creation and initialization
fn bench_expert_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("expert_creation");
    
    // Test different expert sizes
    let sizes = vec![
        ("small", 32, vec![16], 8),
        ("medium", 128, vec![64, 32], 16),
        ("large", 512, vec![256, 128, 64], 32),
    ];

    for (name, input_size, hidden_layers, output_size) in sizes {
        group.bench_with_input(
            BenchmarkId::new("create_expert", name),
            &(input_size, hidden_layers.clone(), output_size),
            |b, (input_size, hidden_layers, output_size)| {
                b.iter(|| {
                    let config = ExpertConfig {
                        id: 1,
                        domain: ExpertDomain::Coding,
                        specialization: Specialization::CodeGeneration,
                        architecture: NetworkArchitecture {
                            input_size: *input_size,
                            hidden_layers: hidden_layers.clone(),
                            output_size: *output_size,
                            activations: vec![
                                synaptic_neural_wasm::Activation::ReLU; 
                                hidden_layers.len() + 1
                            ],
                            dropout_rates: vec![],
                        },
                        training_config: TrainingConfig {
                            learning_rate: 0.001,
                            batch_size: 32,
                            epochs: 100,
                            regularization: 0.01,
                        },
                        performance_thresholds: PerformanceThresholds::default(),
                    };

                    let config_json = serde_json::to_string(&config).unwrap();
                    black_box(KimiMicroExpert::new(&config_json).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark expert inference performance
fn bench_expert_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("expert_inference");
    group.measurement_time(Duration::from_secs(10));

    // Create test experts
    let experts = create_benchmark_experts();
    
    for (name, mut expert) in experts {
        let input_size = expert.architecture.input_size;
        let input_data: Vec<f32> = (0..input_size).map(|i| i as f32 * 0.01).collect();

        group.bench_with_input(
            BenchmarkId::new("inference", name),
            &input_data,
            |b, input| {
                b.iter(|| {
                    black_box(expert.predict(input).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark expert routing performance
fn bench_expert_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("expert_routing");

    // Create router with multiple experts
    let router_config = RouterConfig::default();
    let router_json = serde_json::to_string(&router_config).unwrap();
    let mut router = ExpertRouter::new(&router_json).unwrap();

    // Register multiple expert profiles
    for i in 1..=10 {
        let profile = ExpertProfile {
            id: i,
            domain: match i % 4 {
                0 => ExpertDomain::Coding,
                1 => ExpertDomain::Reasoning,
                2 => ExpertDomain::Language,
                _ => ExpertDomain::Mathematics,
            },
            specialization: Specialization::CodeGeneration,
            avg_execution_time: 50.0 + (i as f32 * 10.0),
            success_rate: 0.9 + (i as f32 * 0.01),
            memory_usage: 1024 * 1024 * i as usize,
            capability_scores: HashMap::new(),
            dependencies: vec![],
            complements: vec![],
        };

        let profile_json = serde_json::to_string(&profile).unwrap();
        router.register_expert(&profile_json).unwrap();
    }

    // Test different request types
    let test_contexts = vec![
        ("simple_coding", create_coding_context()),
        ("complex_reasoning", create_reasoning_context()),
        ("language_processing", create_language_context()),
        ("multi_domain", create_multi_domain_context()),
    ];

    for (name, context) in test_contexts {
        let context_json = serde_json::to_string(&context).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("route_request", name),
            &context_json,
            |b, context_json| {
                b.iter(|| {
                    black_box(router.plan_execution(context_json).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory pool operations
fn bench_memory_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool");

    let memory_config = MemoryPoolConfig {
        max_memory_bytes: 128 * 1024 * 1024, // 128MB
        max_active_experts: 50,
        expert_cache_size: 100,
        enable_compaction: true,
        gc_threshold: 0.8,
        memory_alignment: 32,
    };

    let memory_json = serde_json::to_string(&memory_config).unwrap();
    let mut pool = WasmMemoryPool::new(&memory_json).unwrap();

    // Create test expert
    let expert = create_test_expert(1, ExpertDomain::Coding, 128, vec![64, 32], 16);
    let expert_json = expert.to_json().unwrap();

    group.bench_function("load_expert", |b| {
        b.iter(|| {
            black_box(pool.load_expert(&expert_json).unwrap())
        });
    });

    group.bench_function("unload_expert", |b| {
        b.iter_batched(
            || {
                let block_id = pool.load_expert(&expert_json).unwrap();
                block_id
            },
            |block_id| {
                black_box(pool.unload_expert(expert.id()).unwrap())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("get_memory_stats", |b| {
        b.iter(|| {
            black_box(pool.get_memory_stats().unwrap())
        });
    });

    group.finish();
}

/// Benchmark compression performance
fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    let mut compressor = ExpertCompressor::default();
    
    // Create experts of different sizes for compression testing
    let experts = vec![
        ("small", create_test_expert(1, ExpertDomain::Coding, 64, vec![32], 8)),
        ("medium", create_test_expert(2, ExpertDomain::Reasoning, 256, vec![128, 64], 32)),
        ("large", create_test_expert(3, ExpertDomain::Language, 512, vec![256, 128, 64], 64)),
    ];

    for (name, expert) in &experts {
        group.bench_with_input(
            BenchmarkId::new("compress_expert", name),
            expert,
            |b, expert| {
                b.iter(|| {
                    black_box(compressor.compress_expert(expert).unwrap())
                });
            },
        );
    }

    // Test decompression
    let compressed_experts: Vec<_> = experts.iter()
        .map(|(name, expert)| {
            let compressed = compressor.compress_expert(expert).unwrap();
            (name.clone(), compressed)
        })
        .collect();

    for (name, compressed) in &compressed_experts {
        group.bench_with_input(
            BenchmarkId::new("decompress_expert", name),
            compressed,
            |b, compressed| {
                b.iter(|| {
                    black_box(compressor.decompress_expert(compressed).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark context window operations
fn bench_context_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_window");

    let context_config = ContextConfig {
        max_tokens: 10_000,
        compression_threshold: 0.8,
        enable_summarization: true,
        max_summary_ratio: 0.3,
        enable_importance_tracking: true,
        sliding_window_overlap: 256,
    };

    let context_json = serde_json::to_string(&context_config).unwrap();
    let mut context_window = ContextWindow::new(&context_json).unwrap();

    // Test different content sizes
    let content_sizes = vec![
        ("short", "This is a short message for testing."),
        ("medium", &"This is a medium length message for testing context window performance. ".repeat(10)),
        ("long", &"This is a longer message with much more content to test how the context window handles larger inputs and performs compression when needed. ".repeat(50)),
    ];

    for (name, content) in content_sizes {
        group.bench_with_input(
            BenchmarkId::new("add_content", name),
            &content,
            |b, content| {
                let mut cw = context_window.clone();
                b.iter(|| {
                    black_box(cw.add_content(content, "user").unwrap())
                });
            },
        );
    }

    // Fill context window for compression testing
    for i in 0..100 {
        let content = format!("Test message number {} with some content to fill the context window.", i);
        context_window.add_content(&content, "user").unwrap();
    }

    group.bench_function("compress_context", |b| {
        let mut cw = context_window.clone();
        b.iter(|| {
            black_box(cw.compress_context().unwrap())
        });
    });

    group.bench_function("get_context", |b| {
        b.iter(|| {
            black_box(context_window.get_context())
        });
    });

    group.finish();
}

/// Benchmark parallel execution scenarios
fn bench_parallel_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_execution");
    group.measurement_time(Duration::from_secs(15));

    // Create execution engine
    let exec_config = ExecutionConfig {
        default_strategy: ExecutionStrategy::Parallel,
        max_workers: 4,
        enable_circuit_breakers: true,
        enable_adaptive_execution: true,
        ..Default::default()
    };

    let exec_json = serde_json::to_string(&exec_config).unwrap();
    let memory_config = MemoryPoolConfig::default();
    let memory_json = serde_json::to_string(&memory_config).unwrap();
    let memory_pool = WasmMemoryPool::new(&memory_json).unwrap();
    let mut execution_engine = ExecutionEngine::new(&exec_json, memory_pool).unwrap();

    // Register multiple experts
    for i in 1..=8 {
        let expert = create_test_expert(
            i,
            match i % 4 {
                0 => ExpertDomain::Coding,
                1 => ExpertDomain::Reasoning,
                2 => ExpertDomain::Language,
                _ => ExpertDomain::Mathematics,
            },
            128,
            vec![64, 32],
            16,
        );
        let expert_json = expert.to_json().unwrap();
        execution_engine.register_expert(&expert_json).unwrap();
    }

    // Test different execution scenarios
    let execution_plans = vec![
        ("single_expert", vec![1]),
        ("two_experts", vec![1, 2]),
        ("four_experts", vec![1, 2, 3, 4]),
        ("eight_experts", vec![1, 2, 3, 4, 5, 6, 7, 8]),
    ];

    for (name, expert_ids) in execution_plans {
        let plan = ExecutionPlan {
            experts: expert_ids.clone(),
            parallel_groups: vec![ParallelGroup {
                experts: expert_ids.clone(),
                estimated_time: 50.0,
                memory_requirement: expert_ids.len() * 1024 * 1024,
            }],
            sequential_experts: vec![],
            fallback_strategy: None,
            estimated_time: 50.0,
            estimated_memory: expert_ids.len() * 1024 * 1024,
        };

        let plan_json = serde_json::to_string(&plan).unwrap();
        let exec_context = ExecutionContext {
            request_id: format!("bench_{}", name),
            input_data: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            strategy: ExecutionStrategy::Parallel,
            timeout: 5000.0,
            priority: 5,
            retry_on_failure: false,
            max_retries: 0,
            metadata: HashMap::new(),
        };
        let context_json = serde_json::to_string(&exec_context).unwrap();

        group.bench_with_input(
            BenchmarkId::new("execute_plan", name),
            &(plan_json, context_json),
            |b, (plan_json, context_json)| {
                b.iter(|| {
                    // Note: In a real benchmark, you'd need to handle the Promise
                    // For now, we just benchmark the plan creation
                    black_box(execution_engine.execute_plan(plan_json, context_json))
                });
            },
        );
    }

    group.finish();
}

// Helper functions

fn create_benchmark_experts() -> Vec<(&'static str, KimiMicroExpert)> {
    vec![
        ("small_coding", create_test_expert(1, ExpertDomain::Coding, 64, vec![32], 8)),
        ("medium_reasoning", create_test_expert(2, ExpertDomain::Reasoning, 128, vec![64, 32], 16)),
        ("large_language", create_test_expert(3, ExpertDomain::Language, 256, vec![128, 64, 32], 32)),
    ]
}

fn create_test_expert(
    id: ExpertId,
    domain: ExpertDomain,
    input_size: usize,
    hidden_layers: Vec<usize>,
    output_size: usize,
) -> KimiMicroExpert {
    let config = ExpertConfig {
        id,
        domain,
        specialization: Specialization::CodeGeneration,
        architecture: NetworkArchitecture {
            input_size,
            hidden_layers,
            output_size,
            activations: vec![synaptic_neural_wasm::Activation::ReLU; 3],
            dropout_rates: vec![],
        },
        training_config: TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            regularization: 0.01,
        },
        performance_thresholds: PerformanceThresholds::default(),
    };

    let config_json = serde_json::to_string(&config).unwrap();
    KimiMicroExpert::new(&config_json).unwrap()
}

fn create_coding_context() -> RequestContext {
    RequestContext {
        request: "Write a Python function to sort a list".to_string(),
        complexity: 0.5,
        token_count: 30,
        history_length: 0,
        required_capabilities: vec![ExpertDomain::Coding],
        performance_requirements: PerformanceRequirements::default(),
        metadata: HashMap::new(),
    }
}

fn create_reasoning_context() -> RequestContext {
    RequestContext {
        request: "Solve this logic puzzle with multiple constraints".to_string(),
        complexity: 0.8,
        token_count: 50,
        history_length: 2,
        required_capabilities: vec![ExpertDomain::Reasoning],
        performance_requirements: PerformanceRequirements {
            max_inference_time: 200.0,
            min_confidence: 0.85,
            ..Default::default()
        },
        metadata: HashMap::new(),
    }
}

fn create_language_context() -> RequestContext {
    RequestContext {
        request: "Summarize this long document and extract key points".to_string(),
        complexity: 0.7,
        token_count: 200,
        history_length: 5,
        required_capabilities: vec![ExpertDomain::Language, ExpertDomain::Context],
        performance_requirements: PerformanceRequirements::default(),
        metadata: HashMap::new(),
    }
}

fn create_multi_domain_context() -> RequestContext {
    RequestContext {
        request: "Analyze this code, explain the algorithm, and suggest improvements".to_string(),
        complexity: 0.9,
        token_count: 100,
        history_length: 3,
        required_capabilities: vec![
            ExpertDomain::Coding,
            ExpertDomain::Language,
            ExpertDomain::Reasoning,
        ],
        performance_requirements: PerformanceRequirements {
            max_inference_time: 300.0,
            min_confidence: 0.8,
            allow_parallel: true,
            ..Default::default()
        },
        metadata: HashMap::new(),
    }
}

criterion_group!(
    benches,
    bench_expert_creation,
    bench_expert_inference,
    bench_expert_routing,
    bench_memory_pool,
    bench_compression,
    bench_context_window,
    bench_parallel_execution
);

criterion_main!(benches);