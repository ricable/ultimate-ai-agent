//! Additional functions for the Enhanced Neural Swarm Demo

use crate::*;
use std::fs;

pub fn demonstrate_neural_predictions(coordinator: &mut SwarmCoordinator, rng: &mut StdRng) -> Result<(), String> {
    println!("🧠 Neural Network Prediction Demonstration");
    println!("==========================================");
    
    // Generate test scenarios
    let test_scenarios = vec![
        ("High Power Scenario", RANConfiguration {
            cell_id: 999,
            frequency_band: 3500.0,
            power_level: 35.0,
            antenna_tilt: 2.0,
            bandwidth: 80.0,
            modulation_scheme: "256QAM".to_string(),
            mimo_config: "8x8".to_string(),
            beamforming_enabled: true,
        }),
        ("Low Power Scenario", RANConfiguration {
            cell_id: 998,
            frequency_band: 2400.0,
            power_level: 15.0,
            antenna_tilt: -3.0,
            bandwidth: 40.0,
            modulation_scheme: "QPSK".to_string(),
            mimo_config: "2x2".to_string(),
            beamforming_enabled: false,
        }),
        ("Balanced Scenario", RANConfiguration {
            cell_id: 997,
            frequency_band: 2800.0,
            power_level: 25.0,
            antenna_tilt: 0.0,
            bandwidth: 60.0,
            modulation_scheme: "64QAM".to_string(),
            mimo_config: "4x4".to_string(),
            beamforming_enabled: true,
        }),
    ];
    
    for (scenario_name, config) in test_scenarios {
        println!("\n📊 Testing {}", scenario_name);
        let metrics = coordinator.simulate_ran_environment(&config, rng);
        
        println!("  ├─ Simulated Metrics:");
        println!("  │   • Throughput: {:.2} Mbps", metrics.throughput);
        println!("  │   • Latency: {:.2} ms", metrics.latency);
        println!("  │   • Energy Efficiency: {:.3}", metrics.energy_efficiency);
        println!("  │   • Interference Level: {:.3}", metrics.interference_level);
        
        println!("  └─ Agent Predictions:");
        for agent in &coordinator.agents {
            match agent.predict_fitness(&metrics) {
                Ok(predicted_fitness) => {
                    let actual_fitness = agent.evaluate_fitness(&metrics);
                    let error = (predicted_fitness - actual_fitness).abs();
                    println!("      • Agent {} ({:?}): Predicted {:.4}, Actual {:.4}, Error {:.4}",
                             agent.id, agent.specialization, predicted_fitness, actual_fitness, error);
                },
                Err(e) => println!("      • Agent {}: Prediction error: {}", agent.id, e),
            }
        }
    }
    
    Ok(())
}

pub fn save_results_to_file(summary: &OptimizationSummary) -> Result<(), Box<dyn std::error::Error>> {
    let json_data = serde_json::to_string_pretty(summary)?;
    fs::write("optimization_results.json", json_data)?;
    Ok(())
}

pub fn print_convergence_analysis(summary: &OptimizationSummary) {
    println!("📊 Convergence Analysis:");
    if summary.convergence_history.len() > 10 {
        let early_avg = summary.convergence_history[0..10].iter().sum::<f32>() / 10.0;
        let late_avg = summary.convergence_history[summary.convergence_history.len()-10..]
            .iter().sum::<f32>() / 10.0;
        let improvement = late_avg - early_avg;
        
        println!("  ├─ Early Average (iterations 1-10): {:.4}", early_avg);
        println!("  ├─ Late Average (last 10 iterations): {:.4}", late_avg);
        println!("  ├─ Total Improvement: {:.4}", improvement);
        println!("  └─ Improvement Rate: {:.2}%", (improvement / early_avg.abs()) * 100.0);
    }
}

pub fn print_specialization_analysis(summary: &OptimizationSummary) {
    println!("🎯 Specialization Performance Analysis:");
    
    let mut spec_performance: HashMap<String, Vec<f32>> = HashMap::new();
    for agent in &summary.agent_performances {
        spec_performance.entry(agent.specialization.clone())
            .or_insert_with(Vec::new)
            .push(agent.personal_best_fitness);
    }
    
    for (specialization, performances) in spec_performance {
        let avg_performance = performances.iter().sum::<f32>() / performances.len() as f32;
        let max_performance = performances.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        println!("  ├─ {}: Avg {:.4}, Max {:.4}", specialization, avg_performance, max_performance);
    }
}