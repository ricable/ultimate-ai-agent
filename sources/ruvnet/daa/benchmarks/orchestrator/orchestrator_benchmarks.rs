//! Orchestrator benchmarks for workflow execution and coordination
//! Measures performance of DAA orchestrator operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use daa_orchestrator::{Orchestrator, WorkflowEngine, WorkflowDefinition, Action, Context};
use tokio::runtime::Runtime;
use std::time::Duration;

/// Benchmark workflow creation
fn bench_workflow_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Workflow Creation");

    group.bench_function("create_simple_workflow", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = WorkflowEngine::new();
            let definition = WorkflowDefinition::builder()
                .name("test_workflow")
                .add_step("step1", Action::Monitor)
                .add_step("step2", Action::Reason)
                .add_step("step3", Action::Act)
                .build()
                .unwrap();

            let workflow = engine.create_workflow(black_box(definition)).await.unwrap();
            black_box(workflow)
        });
    });

    group.bench_function("create_complex_workflow", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = WorkflowEngine::new();
            let definition = WorkflowDefinition::builder()
                .name("complex_workflow")
                .add_step("monitor", Action::Monitor)
                .add_step("reason", Action::Reason)
                .add_step("act", Action::Act)
                .add_step("reflect", Action::Reflect)
                .add_step("plan", Action::Plan)
                .add_dependency("reason", "monitor")
                .add_dependency("act", "reason")
                .add_dependency("reflect", "act")
                .add_dependency("plan", "reflect")
                .build()
                .unwrap();

            let workflow = engine.create_workflow(black_box(definition)).await.unwrap();
            black_box(workflow)
        });
    });

    group.finish();
}

/// Benchmark workflow execution
fn bench_workflow_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Workflow Execution");

    group.bench_function("execute_mrap_loop", |b| {
        b.to_async(&rt).iter(|| async {
            let orchestrator = Orchestrator::new(Default::default()).unwrap();

            // Execute one MRAP cycle
            let context = Context::default();
            let _ = orchestrator.monitor().await;
            let _ = orchestrator.reason(black_box(context)).await;
            let action = Action::default();
            let result = orchestrator.act(black_box(action)).await.unwrap();
            let _ = orchestrator.reflect(black_box(result)).await;

            black_box(())
        });
    });

    group.bench_function("execute_parallel_workflows", |b| {
        b.to_async(&rt).iter(|| async {
            let engine = WorkflowEngine::new();
            let mut handles = vec![];

            // Create and execute 10 workflows in parallel
            for i in 0..10 {
                let definition = WorkflowDefinition::builder()
                    .name(&format!("parallel_workflow_{}", i))
                    .add_step("step1", Action::Monitor)
                    .add_step("step2", Action::Act)
                    .build()
                    .unwrap();

                let workflow = engine.create_workflow(definition).await.unwrap();
                let handle = tokio::spawn(async move {
                    engine.execute_workflow(&workflow.id, serde_json::json!({})).await
                });
                handles.push(handle);
            }

            // Wait for all workflows to complete
            for handle in handles {
                let _ = handle.await;
            }

            black_box(())
        });
    });

    group.finish();
}

/// Benchmark rules evaluation
#[cfg(feature = "rules-integration")]
fn bench_rules_evaluation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Rules Evaluation");

    group.bench_function("evaluate_simple_rule", |b| {
        b.to_async(&rt).iter(|| async {
            use daa_rules::{RulesEngine, RuleContext};

            let engine = RulesEngine::new();
            let context = RuleContext::default();

            let result = engine.evaluate(black_box(context)).await.unwrap();
            black_box(result)
        });
    });

    group.bench_function("evaluate_complex_ruleset", |b| {
        b.to_async(&rt).iter(|| async {
            use daa_rules::{RulesEngine, RuleContext, RuleDef};

            let engine = RulesEngine::new();

            // Add multiple rules
            for i in 0..100 {
                let rule = RuleDef::builder()
                    .name(&format!("rule_{}", i))
                    .condition("context.value > 10")
                    .action("increment")
                    .build()
                    .unwrap();

                engine.add_rule(rule).await.unwrap();
            }

            let context = RuleContext::default();
            let result = engine.evaluate(black_box(context)).await.unwrap();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark event processing throughput
fn bench_event_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Event Processing");

    for event_count in [10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*event_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_events", event_count)),
            event_count,
            |b, &event_count| {
                b.to_async(&rt).iter(|| async {
                    let orchestrator = Orchestrator::new(Default::default()).unwrap();

                    // Process events
                    for _ in 0..event_count {
                        let context = Context::default();
                        let _ = orchestrator.reason(black_box(context)).await;
                    }

                    black_box(())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark orchestrator startup and teardown
fn bench_orchestrator_lifecycle(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("Orchestrator Lifecycle");

    group.bench_function("start_and_stop", |b| {
        b.to_async(&rt).iter(|| async {
            let orchestrator = Orchestrator::new(Default::default()).unwrap();
            orchestrator.start().await.unwrap();

            // Run for a short duration
            tokio::time::sleep(Duration::from_millis(10)).await;

            orchestrator.stop().await.unwrap();
            black_box(())
        });
    });

    group.finish();
}

/// Benchmark state monitoring
fn bench_state_monitoring(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("State Monitoring");

    group.bench_function("monitor_system_state", |b| {
        b.to_async(&rt).iter(|| async {
            let orchestrator = Orchestrator::new(Default::default()).unwrap();
            orchestrator.start().await.unwrap();

            let state = orchestrator.monitor().await.unwrap();
            black_box(state)
        });
    });

    group.finish();
}

#[cfg(feature = "rules-integration")]
criterion_group!(
    orchestrator_benches,
    bench_workflow_creation,
    bench_workflow_execution,
    bench_rules_evaluation,
    bench_event_processing,
    bench_orchestrator_lifecycle,
    bench_state_monitoring
);

#[cfg(not(feature = "rules-integration"))]
criterion_group!(
    orchestrator_benches,
    bench_workflow_creation,
    bench_workflow_execution,
    bench_event_processing,
    bench_orchestrator_lifecycle,
    bench_state_monitoring
);

criterion_main!(orchestrator_benches);
