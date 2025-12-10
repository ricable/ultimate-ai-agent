//! Enhanced Neural Swarm Demo - Standalone Modular Version
//! 
//! This is a complete reimplementation of the neural swarm optimization system
//! with modular architecture, advanced neural networks, and comprehensive
//! RAN optimization capabilities.

pub mod models;
pub mod neural;
pub mod swarm;
pub mod config;
pub mod utils;
pub mod performance;
pub mod pfs; // Performance Monitoring System
pub mod dtm; // Digital Twin Mobility System

use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::models::{RANConfiguration, RANMetrics, AgentSpecialization, OptimizationSummary};
use crate::neural::{NeuralAgent, MLModel, DemandPredictor};
use crate::swarm::{SwarmCoordinator, SwarmParameters};
use crate::config::SwarmConfig;
use crate::utils::{Timer, ProgressTracker, StatUtils};
use crate::performance::{PerformanceMetrics, ResourceMonitor};
use crate::pfs::{PFSSystem, pfs_utils};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Enhanced Neural Swarm Demo - Standalone Modular Version");
    println!("============================================================");
    println!("🧠 Features: Advanced Neural Networks, Swarm Intelligence, RAN Optimization");
    println!("📊 Enhanced with PFS (Performance Monitoring System)");
    println!();
    
    // Initialize PFS system first
    let runtime = tokio::runtime::Runtime::new()?;
    let mut pfs_system = PFSSystem::new();
    runtime.block_on(async {
        pfs_system.start().await
    })?;
    println!();
    
    // Load configuration
    let config = SwarmConfig::development();
    config.validate()?;
    
    println!("📋 Configuration loaded:");
    println!("  ├─ Population Size: {}", config.optimization.population_size);
    println!("  ├─ Max Iterations: {}", config.optimization.max_iterations);
    println!("  ├─ Neural Learning Rate: {}", config.neural.learning_rate);
    println!("  └─ System Threads: {}", config.system.max_threads);
    println!();
    
    // Initialize the swarm coordinator
    let swarm_params = SwarmParameters {
        population_size: config.optimization.population_size,
        max_iterations: config.optimization.max_iterations,
        inertia_weight: config.optimization.inertia_weight,
        cognitive_weight: config.optimization.cognitive_weight,
        social_weight: config.optimization.social_weight,
        convergence_threshold: config.optimization.convergence_threshold,
        elite_size: config.optimization.elite_size,
    };
    
    let mut coordinator = SwarmCoordinator::new(swarm_params);
    
    // Initialize swarm with 4-dimensional optimization space
    coordinator.initialize_swarm(4);
    
    println!("🐝 Swarm initialized with {} agents", coordinator.agents.len());
    println!("   Agent Specializations:");
    for agent in &coordinator.agents {
        println!("   ├─ {}: {:?}", agent.id, agent.neural_agent.specialization);
    }
    println!();
    
    // Start optimization with timer
    let _optimization_timer = Timer::new("Neural Swarm Optimization".to_string());
    
    println!("🔄 Starting neural swarm optimization...");
    
    // Create RNG for the fitness evaluator - we'll move it into the closure
    let mut rng = StdRng::seed_from_u64(42);
    
    let result = {
        let result = coordinator.optimize(|config: &RANConfiguration| -> (f32, RANMetrics) {
            // Create a fresh RNG instance for each evaluation to avoid borrow issues
            let mut local_rng = StdRng::seed_from_u64(42);
            let mut metrics = RANMetrics::new();
            
            // Enhanced RAN simulation with realistic constraints
            let power_factor = (config.power_level - 5.0) / 35.0; // Normalize to 0-1
            let bandwidth_factor = config.bandwidth / 80.0;
            let freq_factor = (config.frequency_band - 2400.0) / 1100.0;
            
            // Throughput calculation
            metrics.throughput = (50.0 + power_factor * 30.0 + bandwidth_factor * 20.0) 
                * (1.0 + local_rng.gen_range(-0.1..0.1));
            
            // Latency calculation (inverse relationship with some parameters)
            metrics.latency = (20.0 - power_factor * 5.0 + freq_factor * 10.0)
                * (1.0 + local_rng.gen_range(-0.2..0.2));
            metrics.latency = metrics.latency.max(1.0);
            
            // Energy efficiency
            metrics.energy_efficiency = (0.6 + bandwidth_factor * 0.2 - power_factor * 0.1)
                * (1.0 + local_rng.gen_range(-0.1..0.1));
            metrics.energy_efficiency = metrics.energy_efficiency.clamp(0.1, 1.0);
            
            // Interference level
            metrics.interference_level = (power_factor * 0.3 + freq_factor * 0.2)
                * (1.0 + local_rng.gen_range(-0.1..0.1));
            metrics.interference_level = metrics.interference_level.clamp(0.0, 1.0);
            
            let fitness = metrics.calculate_fitness();
            (fitness, metrics)
        });
        result
    };
    
    println!();
    println!("🎯 OPTIMIZATION RESULTS");
    println!("=======================");
    println!("⏱️  Execution Time: {:.2}s", result.execution_time_ms as f64 / 1000.0);
    println!("🔄 Iterations Completed: {}", result.iterations_completed);
    println!("🏆 Best Fitness: {:.6}", result.best_fitness);
    println!();
    
    println!("📊 Optimal RAN Configuration:");
    println!("  ├─ Cell ID: {}", result.best_configuration.cell_id);
    println!("  ├─ Power Level: {:.1} dBm", result.best_configuration.power_level);
    println!("  ├─ Antenna Tilt: {:.1}°", result.best_configuration.antenna_tilt);
    println!("  ├─ Bandwidth: {:.0} MHz", result.best_configuration.bandwidth);
    println!("  ├─ Frequency: {:.0} MHz", result.best_configuration.frequency_band);
    println!("  ├─ Modulation: {}", result.best_configuration.modulation_scheme);
    println!("  ├─ MIMO: {}", result.best_configuration.mimo_config);
    println!("  └─ Beamforming: {}", if result.best_configuration.beamforming_enabled { "Enabled" } else { "Disabled" });
    println!();
    
    println!("📈 Performance Metrics:");
    println!("  ├─ Throughput: {:.2} Mbps", result.best_metrics.throughput);
    println!("  ├─ Latency: {:.2} ms", result.best_metrics.latency);
    println!("  ├─ Energy Efficiency: {:.3}", result.best_metrics.energy_efficiency);
    println!("  └─ Interference Level: {:.3}", result.best_metrics.interference_level);
    println!();
    
    // Convergence analysis
    if result.convergence_history.len() > 10 {
        let early_avg = StatUtils::mean(&result.convergence_history[0..10]);
        let late_avg = StatUtils::mean(&result.convergence_history[result.convergence_history.len()-10..]);
        let improvement = late_avg - early_avg;
        
        println!("📊 Convergence Analysis:");
        println!("  ├─ Early Average (iterations 1-10): {:.4}", early_avg);
        println!("  ├─ Late Average (last 10 iterations): {:.4}", late_avg);
        println!("  ├─ Total Improvement: {:.4}", improvement);
        println!("  └─ Improvement Rate: {:.2}%", (improvement / early_avg.abs()) * 100.0);
        println!();
    }
    
    // Agent performance analysis
    println!("🎯 Agent Performance Analysis:");
    let mut agent_fitness: Vec<f32> = result.agent_performances.values().cloned().collect();
    if !agent_fitness.is_empty() {
        agent_fitness.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        println!("  ├─ Best Agent Fitness: {:.4}", agent_fitness[0]);
        println!("  ├─ Worst Agent Fitness: {:.4}", agent_fitness[agent_fitness.len()-1]);
        println!("  ├─ Average Fitness: {:.4}", StatUtils::mean(&agent_fitness));
        println!("  └─ Standard Deviation: {:.4}", StatUtils::std_dev(&agent_fitness));
    }
    println!();
    
    // Demonstrate neural network predictions
    demonstrate_neural_predictions(&coordinator, &mut rng)?;
    
    // Demonstrate demand prediction
    demonstrate_demand_prediction(&mut rng)?;
    
    // Generate comprehensive PFS system report
    runtime.block_on(async {
        println!("\n📊 PFS SYSTEM PERFORMANCE REPORT");
        println!("=================================");
        
        let system_status = pfs_system.get_system_status().await;
        println!("🎯 PFS System Status:");
        println!("  ├─ Active Metrics: {}", system_status.active_metrics_count);
        println!("  ├─ Active Agents: {}", system_status.active_agents_count);
        println!("  ├─ Total Operations: {}", system_status.total_operations);
        println!("  ├─ Performance Score: {:.3}", system_status.average_performance_score);
        println!("  └─ System Uptime: {:?}", system_status.system_uptime);
        println!();
        
        // Generate and display comprehensive report
        let pfs_report = pfs_system.generate_system_report().await;
        println!("{}", pfs_report);
        
        // Export PFS data
        match pfs_system.export_system_data("json").await {
            Ok(_) => println!("✅ PFS system data exported successfully"),
            Err(e) => println!("⚠️ Failed to export PFS data: {}", e),
        }
    });
    
    // Save comprehensive results with PFS integration
    let summary = result.to_optimization_summary();
    save_results(&summary)?;
    
    println!("✅ Enhanced Neural Swarm Demo with PFS completed successfully!");
    println!("📁 Optimization results saved to 'swarm_optimization_results.json'");
    println!("📊 PFS system data exported with comprehensive performance analytics");
    
    Ok(())
}

fn demonstrate_neural_predictions(coordinator: &SwarmCoordinator, rng: &mut StdRng) -> Result<(), String> {
    println!("🧠 Neural Network Prediction Demonstration");
    println!("==========================================");
    
    // Test scenarios for neural prediction
    let test_scenarios = vec![
        ("High Throughput Scenario", RANConfiguration {
            cell_id: 999,
            frequency_band: 3500.0,
            power_level: 35.0,
            antenna_tilt: 2.0,
            bandwidth: 80.0,
            modulation_scheme: "256QAM".to_string(),
            mimo_config: "8x8".to_string(),
            beamforming_enabled: true,
        }),
        ("Low Latency Scenario", RANConfiguration {
            cell_id: 998,
            frequency_band: 2400.0,
            power_level: 25.0,
            antenna_tilt: 0.0,
            bandwidth: 40.0,
            modulation_scheme: "64QAM".to_string(),
            mimo_config: "4x4".to_string(),
            beamforming_enabled: true,
        }),
        ("Energy Efficient Scenario", RANConfiguration {
            cell_id: 997,
            frequency_band: 2800.0,
            power_level: 15.0,
            antenna_tilt: -2.0,
            bandwidth: 20.0,
            modulation_scheme: "QPSK".to_string(),
            mimo_config: "2x2".to_string(),
            beamforming_enabled: false,
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
            match agent.neural_agent.predict_fitness(&metrics) {
                Ok(predicted_fitness) => {
                    let actual_fitness = agent.neural_agent.evaluate_fitness(&metrics);
                    let error = (predicted_fitness - actual_fitness).abs();
                    println!("      • Agent {} ({:?}): Predicted {:.4}, Actual {:.4}, Error {:.4}",
                             agent.id, agent.neural_agent.specialization, predicted_fitness, actual_fitness, error);
                },
                Err(e) => println!("      • Agent {}: Prediction error: {}", agent.id, e),
            }
        }
    }
    
    Ok(())
}

fn demonstrate_demand_prediction(rng: &mut StdRng) -> Result<(), String> {
    println!("\n🔮 Demand Prediction Demonstration");
    println!("==================================");
    
    // Create demand predictor
    let mut predictor = DemandPredictor::new(24);
    
    // Generate sample historical data (simulating 7 days of hourly data)
    let mut historical_data = Vec::new();
    for day in 0..7 {
        for hour in 0..24 {
            // Simulate daily pattern with some randomness
            let base_demand = 50.0;
            let daily_pattern = 20.0 * (2.0 * std::f32::consts::PI * (hour as f32) / 24.0).sin();
            let weekend_factor = if day >= 5 { 0.8 } else { 1.0 };
            let noise = rng.gen_range(-5.0..5.0);
            
            let demand = (base_demand + daily_pattern) * weekend_factor + noise;
            historical_data.push(demand.max(0.0));
        }
    }
    
    // Add historical data to predictor
    predictor.add_historical_data(historical_data.clone());
    
    println!("📈 Historical Data Statistics:");
    println!("  ├─ Data Points: {}", historical_data.len());
    println!("  ├─ Average Demand: {:.2}", StatUtils::mean(&historical_data));
    println!("  ├─ Standard Deviation: {:.2}", StatUtils::std_dev(&historical_data));
    println!("  └─ Min/Max: {:.2}/{:.2}", 
             historical_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             historical_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Make predictions for next 24 hours
    match predictor.predict_demand(24) {
        Ok(predictions) => {
            println!("\n🔮 24-Hour Demand Forecast:");
            for (hour, &prediction) in predictions.iter().enumerate() {
                let (lower, upper) = predictor.get_confidence_interval(prediction);
                println!("  Hour {:2}: {:.2} (CI: {:.2} - {:.2})", 
                         hour, prediction, lower, upper);
                
                if hour % 6 == 5 { println!(); } // Add spacing every 6 hours
            }
            
            println!("📊 Prediction Statistics:");
            println!("  ├─ Average Predicted Demand: {:.2}", StatUtils::mean(&predictions));
            println!("  ├─ Prediction Range: {:.2}", 
                     predictions.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) - 
                     predictions.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
            println!("  └─ Confidence Level: {:.0}%", predictor.confidence_level * 100.0);
        },
        Err(e) => println!("❌ Prediction failed: {}", e),
    }
    
    Ok(())
}

fn save_results(summary: &OptimizationSummary) -> Result<(), Box<dyn std::error::Error>> {
    let json_data = serde_json::to_string_pretty(summary)?;
    std::fs::write("swarm_optimization_results.json", json_data)?;
    
    // Also save a CSV summary for easy analysis
    let csv_content = format!(
        "timestamp,iterations,best_fitness,throughput,latency,energy_efficiency,interference,execution_time\n{},{},{},{},{},{},{},{}\n",
        summary.timestamp.format("%Y-%m-%d %H:%M:%S"),
        summary.total_iterations,
        summary.best_fitness,
        summary.best_metrics.throughput,
        summary.best_metrics.latency,
        summary.best_metrics.energy_efficiency,
        summary.best_metrics.interference_level,
        summary.execution_time_seconds
    );
    std::fs::write("swarm_results_summary.csv", csv_content)?;
    
    Ok(())
}
