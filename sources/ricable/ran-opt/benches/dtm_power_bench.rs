use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use ran_opt::{DtmPowerManager, PowerStateFeatures};

fn benchmark_energy_prediction(c: &mut Criterion) {
    let mut manager = DtmPowerManager::new();
    
    // Setup realistic features
    let mut features = PowerStateFeatures::new();
    features.cpu_utilization = 0.65;
    features.memory_pressure = 0.45;
    features.io_wait_ratio = 0.2;
    features.thermal_state = 0.7;
    features.frequency_scaling = 0.9;
    features.voltage_level = 0.85;
    features.cache_miss_rate = 0.15;
    features.network_activity = 0.3;
    
    manager.update_features(features);
    
    c.bench_function("energy_prediction", |b| {
        b.iter(|| {
            black_box(manager.predict_energy_consumption())
        })
    });
}

fn benchmark_scheduler_decision(c: &mut Criterion) {
    let manager = DtmPowerManager::new();
    
    let mut features = PowerStateFeatures::new();
    features.cpu_utilization = 0.3;
    features.io_wait_ratio = 0.6;
    features.thermal_state = 0.5;
    
    c.bench_function("scheduler_decision", |b| {
        b.iter(|| {
            black_box(manager.get_scheduler_decision())
        })
    });
}

fn benchmark_arrival_prediction(c: &mut Criterion) {
    let mut manager = DtmPowerManager::new();
    
    c.bench_function("arrival_prediction", |b| {
        b.iter(|| {
            black_box(manager.predict_next_event_time(black_box(1.0)))
        })
    });
}

fn benchmark_full_pipeline(c: &mut Criterion) {
    let mut manager = DtmPowerManager::new();
    
    let mut features = PowerStateFeatures::new();
    features.cpu_utilization = 0.55;
    features.memory_pressure = 0.4;
    features.io_wait_ratio = 0.3;
    features.thermal_state = 0.6;
    features.frequency_scaling = 0.95;
    features.voltage_level = 0.9;
    features.cache_miss_rate = 0.12;
    features.network_activity = 0.25;
    
    c.bench_function("full_pipeline", |b| {
        b.iter(|| {
            manager.update_features(features.clone());
            let energy = manager.predict_energy_consumption();
            let scheduler = manager.get_scheduler_decision();
            let arrival = manager.predict_next_event_time(1.0);
            black_box((energy, scheduler, arrival))
        })
    });
}

fn benchmark_model_compression(c: &mut Criterion) {
    let mut manager = DtmPowerManager::new();
    
    c.bench_function("model_compression", |b| {
        b.iter(|| {
            manager.compress_models(black_box(0.01));
        })
    });
}

fn benchmark_memory_usage(c: &mut Criterion) {
    c.bench_function("memory_footprint", |b| {
        b.iter(|| {
            let manager = DtmPowerManager::new();
            let stats = manager.get_model_stats();
            black_box(stats)
        })
    });
}

criterion_group!(
    benches,
    benchmark_energy_prediction,
    benchmark_scheduler_decision,
    benchmark_arrival_prediction,
    benchmark_full_pipeline,
    benchmark_model_compression,
    benchmark_memory_usage
);
criterion_main!(benches);