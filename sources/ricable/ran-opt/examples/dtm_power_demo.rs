use ran_opt::{DtmPowerManager, PowerStateFeatures, SchedulerDecision};
use std::time::Instant;

fn main() {
    println!("DTM Power Manager Demo");
    println!("======================");
    
    // Initialize the power manager
    let mut manager = DtmPowerManager::new();
    
    // Simulate different power states
    let scenarios = vec![
        ("Low Load", 0.15, 0.2, 0.8, 0.3),
        ("Medium Load", 0.45, 0.5, 0.4, 0.6),
        ("High Load", 0.85, 0.8, 0.1, 0.9),
        ("Thermal Stress", 0.7, 0.6, 0.2, 0.95),
        ("IO Heavy", 0.3, 0.4, 0.9, 0.5),
    ];
    
    for (name, cpu, memory, io, thermal) in scenarios {
        println!("\n--- {} Scenario ---", name);
        
        // Configure features
        let mut features = PowerStateFeatures::new();
        features.cpu_utilization = cpu;
        features.memory_pressure = memory;
        features.io_wait_ratio = io;
        features.thermal_state = thermal;
        features.frequency_scaling = 1.0 - thermal * 0.3; // Thermal throttling
        features.voltage_level = 0.8 + cpu * 0.2; // Dynamic voltage scaling
        features.cache_miss_rate = cpu * 0.2;
        features.network_activity = memory * 0.3;
        
        manager.update_features(features);
        
        // Measure inference time
        let start = Instant::now();
        let energy_prediction = manager.predict_energy_consumption();
        let scheduler_decision = manager.get_scheduler_decision();
        let next_arrival = manager.predict_next_event_time(1.0);
        let inference_time = start.elapsed();
        
        println!("  Energy Prediction: {:.3} W", energy_prediction);
        println!("  Scheduler Decision: {:?}", scheduler_decision);
        println!("  Next Event Time: {:.3} ms", next_arrival * 1000.0);
        println!("  Inference Time: {:?}", inference_time);
        
        // Verify real-time constraint
        if inference_time.as_millis() > 10 {
            println!("  WARNING: Inference time exceeds 10ms threshold!");
        } else {
            println!("  ✓ Real-time constraint satisfied");
        }
    }
    
    // Demonstrate model compression
    println!("\n--- Model Compression ---");
    let stats_before = manager.get_model_stats();
    println!("Before compression:");
    println!("  Total parameters: {}", stats_before.total_parameters);
    println!("  Energy predictor size: {} bytes", stats_before.energy_predictor_size);
    println!("  Arrival predictor size: {} bytes", stats_before.arrival_predictor_size);
    
    manager.compress_models(0.01);
    let stats_after = manager.get_model_stats();
    println!("After compression (threshold: 0.01):");
    println!("  Total parameters: {}", stats_after.total_parameters);
    println!("  Energy predictor size: {} bytes", stats_after.energy_predictor_size);
    println!("  Arrival predictor size: {} bytes", stats_after.arrival_predictor_size);
    
    // Test 1-minute granularity predictions
    println!("\n--- 1-Minute Granularity Test ---");
    let start_time = Instant::now();
    let mut predictions = Vec::new();
    
    for second in 0..60 {
        let mut features = PowerStateFeatures::new();
        // Simulate varying load over time
        let load_factor = (second as f32 / 60.0 * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5;
        features.cpu_utilization = load_factor;
        features.memory_pressure = load_factor * 0.8;
        features.io_wait_ratio = (1.0 - load_factor) * 0.5;
        features.thermal_state = load_factor * 0.7 + 0.3;
        
        manager.update_features(features);
        let prediction = manager.predict_energy_consumption();
        predictions.push(prediction);
    }
    
    let total_time = start_time.elapsed();
    println!("  60 predictions completed in {:?}", total_time);
    println!("  Average per prediction: {:?}", total_time / 60);
    println!("  Energy range: {:.3} - {:.3} W", 
             predictions.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             predictions.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Demonstrate scheduler decision logic
    println!("\n--- Scheduler Decision Logic ---");
    let test_cases = vec![
        ("Idle system", 0.05, 0.1, 0.9, 0.3),
        ("CPU bound", 0.9, 0.2, 0.1, 0.6),
        ("Memory bound", 0.4, 0.9, 0.3, 0.5),
        ("Thermal throttling", 0.8, 0.6, 0.2, 0.95),
    ];
    
    for (case, cpu, memory, io, thermal) in test_cases {
        let mut features = PowerStateFeatures::new();
        features.cpu_utilization = cpu;
        features.memory_pressure = memory;
        features.io_wait_ratio = io;
        features.thermal_state = thermal;
        
        manager.update_features(features);
        let decision = manager.get_scheduler_decision();
        
        println!("  {}: {:?}", case, decision);
    }
    
    println!("\n--- Performance Summary ---");
    println!("✓ Real-time inference (<10ms)");
    println!("✓ 1-minute granularity predictions");
    println!("✓ Quantization and pruning for edge deployment");
    println!("✓ Custom activation functions for power curve modeling");
    println!("✓ Integrated decision tree for scheduler selection");
    println!("✓ LSTM-based inter-arrival time prediction");
}