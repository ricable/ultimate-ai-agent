//! Enhanced PSO Network Optimization Demo
//! 
//! This demo showcases the enhanced PSO implementation with multi-objective
//! optimization for network performance including throughput, latency, energy
//! efficiency, and other key performance indicators.

use std::collections::HashMap;
use std::time::Instant;

use standalone_swarm_demo::models::{RANConfiguration, RANMetrics, AgentSpecialization};
use standalone_swarm_demo::swarm::{
    SwarmAgent, SwarmParameters, ParticleSwarmOptimizer,
    NetworkFitnessScores, OptimizationObjective, NetworkConstraints,
    NetworkConditions, TrafficPattern, WeatherConditions,
    NetworkFitnessEvaluator, OptimizationReport,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Enhanced PSO Network Optimization Demo");
    println!("==========================================");
    
    // Initialize network constraints for realistic optimization
    let network_constraints = NetworkConstraints {
        max_power_consumption: 25.0,
        min_coverage_area: 2.0,
        max_interference_threshold: 0.25,
        required_throughput: 100.0,
        max_latency: 5.0,
        min_handover_success_rate: 0.98,
        energy_budget: 150.0,
    };
    
    // Define multi-objective optimization goals
    let optimization_objective = OptimizationObjective::MultiObjective(vec![
        (OptimizationObjective::MaximizeThroughput, 0.35),
        (OptimizationObjective::MinimizeLatency, 0.30),
        (OptimizationObjective::OptimizeEnergyEfficiency, 0.20),
        (OptimizationObjective::MinimizeInterference, 0.10),
        (OptimizationObjective::MaximizeHandoverSuccess, 0.05),
    ]);
    
    // Configure swarm parameters
    let swarm_params = SwarmParameters {
        population_size: 30,
        max_iterations: 200,
        inertia_weight: 0.8,
        cognitive_weight: 1.8,
        social_weight: 1.8,
        convergence_threshold: 0.001,
        elite_size: 6,
    };
    
    // Initialize enhanced PSO optimizer
    let mut pso = ParticleSwarmOptimizer::new_with_objectives(
        swarm_params,
        4, // 4 dimensions: power, tilt, bandwidth, frequency
        optimization_objective,
        network_constraints,
    );
    
    println!("✅ Initialized Enhanced PSO with multi-objective optimization");
    
    // Create specialized swarm agents for different optimization aspects
    let mut agents = create_specialized_agents(30);
    
    // Initialize sub-swarms for multi-layer optimization
    pso.initialize_sub_swarms(&agents);
    
    println!("✅ Created {} specialized agents across {} sub-swarms", 
             agents.len(), pso.sub_swarms.len());
    
    // Display initial network conditions
    display_network_conditions(&pso.current_network_conditions);
    
    // Run optimization scenarios
    run_optimization_scenarios(&mut pso, &mut agents)?;
    
    Ok(())
}

fn create_specialized_agents(count: usize) -> Vec<SwarmAgent> {
    let specializations = vec![
        AgentSpecialization::ThroughputOptimizer,
        AgentSpecialization::LatencyMinimizer,
        AgentSpecialization::EnergyEfficiencyExpert,
        AgentSpecialization::InterferenceAnalyst,
        AgentSpecialization::GeneralPurpose,
    ];
    
    let mut agents = Vec::new();
    
    for i in 0..count {
        let specialization = specializations[i % specializations.len()].clone();
        let agent = SwarmAgent::new(
            format!("agent_{}", i),
            specialization,
            4, // 4 dimensions
        );
        agents.push(agent);
    }
    
    agents
}

fn display_network_conditions(conditions: &NetworkConditions) {
    println!("\n📊 Current Network Conditions:");
    println!("   Load Factor: {:.2}", conditions.load_factor);
    println!("   Interference Level: {:.2}", conditions.interference_level);
    println!("   Mobility Factor: {:.2}", conditions.mobility_factor);
    println!("   Traffic Pattern: {:?}", conditions.traffic_pattern);
    println!("   Time of Day: {:.1}h", conditions.time_of_day);
    println!("   Weather: {:?}", conditions.weather_conditions);
}

fn run_optimization_scenarios(
    pso: &mut ParticleSwarmOptimizer,
    agents: &mut [SwarmAgent],
) -> Result<(), Box<dyn std::error::Error>> {
    let scenarios = vec![
        ("🎯 High Throughput Scenario", TrafficPattern::DataTransfer),
        ("⚡ Low Latency Scenario", TrafficPattern::Gaming),
        ("🔋 Energy Efficiency Scenario", TrafficPattern::IoT),
        ("📞 VoIP Optimization Scenario", TrafficPattern::VoIP),
        ("📺 Video Streaming Scenario", TrafficPattern::Video),
    ];
    
    for (scenario_name, traffic_pattern) in scenarios {
        println!("\n{}", scenario_name);
        println!("{}", "=".repeat(50));
        
        // Update network conditions for scenario
        pso.current_network_conditions.traffic_pattern = traffic_pattern.clone();
        
        // Adapt optimization strategy based on scenario
        pso.adapt_optimization_strategy();
        
        // Run optimization
        let start_time = Instant::now();
        run_single_optimization(pso, agents, 50)?;
        let optimization_time = start_time.elapsed();
        
        // Display results
        display_optimization_results(pso, optimization_time);
        
        // Generate detailed report
        let report = pso.generate_optimization_report();
        display_detailed_report(&report);
    }
    
    // Final comprehensive analysis
    display_final_analysis(pso, agents);
    
    Ok(())
}

fn run_single_optimization(
    pso: &mut ParticleSwarmOptimizer,
    agents: &mut [SwarmAgent],
    iterations: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut evaluator = NetworkFitnessEvaluator::new();
    
    for iteration in 0..iterations {
        // Update particles with multi-objective optimization
        pso.update_particles(agents);
        
        // Apply mutation for diversity
        if iteration % 20 == 0 {
            pso.apply_mutation(agents, 0.05);
        }
        
        // Display progress every 10 iterations
        if iteration % 10 == 0 {
            let diversity = pso.get_swarm_diversity(agents);
            let best_fitness = pso.global_best_fitness;
            let pareto_size = pso.get_pareto_front().len();
            
            println!("   Iteration {}: Best={:.4}, Diversity={:.4}, Pareto={}", 
                     iteration, best_fitness, diversity, pareto_size);
        }
    }
    
    Ok(())
}

fn display_optimization_results(pso: &ParticleSwarmOptimizer, optimization_time: std::time::Duration) {
    let pareto_front = pso.get_pareto_front();
    let metrics = pso.get_network_metrics();
    
    println!("⏱️  Optimization completed in {:.2}s", optimization_time.as_secs_f64());
    println!("📈 Best Fitness: {:.4}", pso.global_best_fitness);
    println!("🎯 Pareto Front Size: {}", pareto_front.len());
    println!("📊 Convergence Rate: {:.4}", metrics.convergence_rate);
    println!("⚠️  Constraint Violations: {:.2}%", metrics.constraint_violation_rate * 100.0);
    
    if !pareto_front.is_empty() {
        println!("\n🏆 Best Solutions per Objective:");
        let best_solutions = pso.get_best_solutions_per_objective();
        
        for (objective, solution) in best_solutions.iter() {
            println!("   {}: {:.4}", objective, 
                     get_objective_score(objective, &solution.fitness_scores));
        }
    }
}

fn get_objective_score(objective: &str, scores: &NetworkFitnessScores) -> f32 {
    match objective {
        "throughput" => scores.throughput,
        "latency" => scores.latency,
        "energy_efficiency" => scores.energy_efficiency,
        "interference" => scores.interference_level,
        _ => scores.weighted_composite,
    }
}

fn display_detailed_report(report: &OptimizationReport) {
    println!("\n📋 Detailed Optimization Report:");
    println!("   Hypervolume: {:.6}", report.hypervolume);
    println!("   Iterations: {}", report.iterations_completed);
    
    println!("\n🌐 Network Performance Analysis:");
    println!("   Current Load: {:.2}", report.network_conditions.load_factor);
    println!("   Interference: {:.2}", report.network_conditions.interference_level);
    println!("   Mobility: {:.2}", report.network_conditions.mobility_factor);
}

fn display_final_analysis(pso: &ParticleSwarmOptimizer, agents: &[SwarmAgent]) {
    println!("\n" + &"=".repeat(60));
    println!("🎯 FINAL COMPREHENSIVE ANALYSIS");
    println!("{}", "=".repeat(60));
    
    // Agent performance analysis
    let mut specialization_performance: HashMap<AgentSpecialization, Vec<f32>> = HashMap::new();
    
    for agent in agents {
        specialization_performance
            .entry(agent.neural_agent.specialization.clone())
            .or_insert_with(Vec::new)
            .push(agent.current_fitness);
    }
    
    println!("\n📊 Agent Specialization Performance:");
    for (specialization, fitnesses) in specialization_performance {
        let avg_fitness = fitnesses.iter().sum::<f32>() / fitnesses.len() as f32;
        let max_fitness = fitnesses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        println!("   {:?}: Avg={:.4}, Max={:.4}, Count={}", 
                 specialization, avg_fitness, max_fitness, fitnesses.len());
    }
    
    // Sub-swarm analysis
    println!("\n🔗 Sub-Swarm Performance:");
    for sub_swarm in &pso.sub_swarms {
        println!("   {:?} Layer: Best={:.4}, Agents={}", 
                 sub_swarm.layer_type, sub_swarm.local_best_fitness, sub_swarm.agents.len());
    }
    
    // Network optimization summary
    let final_metrics = pso.get_network_metrics();
    println!("\n🌐 Final Network Optimization Summary:");
    println!("   Pareto Solutions: {}", final_metrics.pareto_front_size);
    println!("   Convergence: {:.4}", final_metrics.convergence_rate);
    println!("   Constraint Compliance: {:.2}%", 
             (1.0 - final_metrics.constraint_violation_rate) * 100.0);
    
    // Best configuration analysis
    if let Some(best_solution) = pso.get_pareto_front().first() {
        println!("\n⭐ Optimal Network Configuration:");
        let config = &best_solution.configuration;
        println!("   Cell ID: {}", config.cell_id);
        println!("   Power Level: {:.1} dBm", config.power_level);
        println!("   Antenna Tilt: {:.1}°", config.antenna_tilt);
        println!("   Bandwidth: {:.0} MHz", config.bandwidth);
        println!("   Frequency: {:.0} MHz", config.frequency_band);
        println!("   MIMO: {}", config.mimo_config);
        println!("   Beamforming: {}", if config.beamforming_enabled { "Enabled" } else { "Disabled" });
        
        println!("\n📈 Performance Metrics:");
        let scores = &best_solution.fitness_scores;
        println!("   Throughput Score: {:.4}", scores.throughput);
        println!("   Latency Score: {:.4}", scores.latency);
        println!("   Energy Efficiency: {:.4}", scores.energy_efficiency);
        println!("   Interference Level: {:.4}", scores.interference_level);
        println!("   Handover Success: {:.4}", scores.handover_success_rate);
        println!("   ENDC Success: {:.4}", scores.endc_establishment_success);
        println!("   User Satisfaction: {:.4}", scores.user_satisfaction);
    }
    
    println!("\n✅ Enhanced PSO Network Optimization Complete!");
    println!("🚀 Achieved multi-objective optimization across all network layers");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_specialized_agents_creation() {
        let agents = create_specialized_agents(10);
        assert_eq!(agents.len(), 10);
        
        // Check that we have different specializations
        let mut specializations = std::collections::HashSet::new();
        for agent in &agents {
            specializations.insert(agent.neural_agent.specialization.clone());
        }
        assert!(specializations.len() > 1);
    }
    
    #[test]
    fn test_optimization_scenario_setup() {
        let constraints = NetworkConstraints::default();
        let objective = OptimizationObjective::MaximizeThroughput;
        let params = SwarmParameters::default();
        
        let pso = ParticleSwarmOptimizer::new_with_objectives(
            params, 4, objective, constraints
        );
        
        assert_eq!(pso.global_best_position.len(), 4);
    }
}