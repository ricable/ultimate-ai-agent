//! Swarm Coordinator Implementation
//! 
//! This module implements the main coordination logic for the swarm optimization system.

use crate::models::{RANConfiguration, RANMetrics, AgentSpecialization, OptimizationSummary, AgentPerformance};
use crate::neural::NeuralAgent;
use crate::swarm::{SwarmAgent, SwarmParameters, SwarmOptimizationResult, SwarmDiversity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use rand::Rng;
use chrono::Utc;

#[derive(Debug, Clone)]
pub struct SwarmCoordinator {
    pub agents: Vec<SwarmAgent>,
    pub global_best_position: Vec<f32>,
    pub global_best_fitness: f32,
    pub global_best_configuration: RANConfiguration,
    pub global_best_metrics: RANMetrics,
    pub parameters: SwarmParameters,
    pub convergence_history: Vec<f32>,
    pub iteration_count: u32,
    pub diversity_history: Vec<SwarmDiversity>,
}

impl SwarmCoordinator {
    pub fn new(parameters: SwarmParameters) -> Self {
        Self {
            agents: Vec::new(),
            global_best_position: Vec::new(),
            global_best_fitness: f32::NEG_INFINITY,
            global_best_configuration: RANConfiguration::new(0),
            global_best_metrics: RANMetrics::new(),
            parameters,
            convergence_history: Vec::new(),
            iteration_count: 0,
            diversity_history: Vec::new(),
        }
    }
    
    pub fn initialize_swarm(&mut self, dimensions: usize) {
        self.agents.clear();
        self.global_best_position = vec![0.0; dimensions];
        
        // Create agents with diverse specializations
        let specializations = [
            AgentSpecialization::ThroughputOptimizer,
            AgentSpecialization::LatencyMinimizer,
            AgentSpecialization::EnergyEfficiencyExpert,
            AgentSpecialization::InterferenceAnalyst,
            AgentSpecialization::GeneralPurpose,
        ];
        
        for i in 0..self.parameters.population_size {
            let specialization = specializations[i % specializations.len()].clone();
            let agent = SwarmAgent::new(
                format!("agent_{}", i),
                specialization,
                dimensions,
            );
            self.agents.push(agent);
        }
        
        println!("Initialized swarm with {} agents", self.agents.len());
    }
    
    pub fn optimize<F>(&mut self, mut fitness_evaluator: F) -> SwarmOptimizationResult 
    where 
        F: FnMut(&RANConfiguration) -> (f32, RANMetrics),
    {
        let start_time = Instant::now();
        let mut result = SwarmOptimizationResult::new();
        
        println!("Starting swarm optimization...");
        
        // Initial fitness evaluation
        self.evaluate_all_agents(&mut fitness_evaluator);
        
        for iteration in 0..self.parameters.max_iterations {
            self.iteration_count = iteration;
            
            // Update velocities and positions
            self.update_swarm();
            
            // Evaluate fitness
            self.evaluate_all_agents(&mut fitness_evaluator);
            
            // Update global best
            self.update_global_best();
            
            // Track convergence
            self.convergence_history.push(self.global_best_fitness);
            
            // Calculate and store diversity
            let diversity = SwarmDiversity::calculate(&self.agents);
            self.diversity_history.push(diversity);
            
            // Check convergence
            if self.check_convergence() {
                println!("Convergence reached at iteration {}", iteration);
                break;
            }
            
            // Apply adaptive mechanisms
            self.apply_adaptive_mechanisms(iteration);
            
            // Progress reporting
            if iteration % 10 == 0 {
                println!("Iteration {}: Best fitness = {:.6}", iteration, self.global_best_fitness);
            }
        }
        
        let execution_time = start_time.elapsed();
        
        // Prepare result
        result.best_fitness = self.global_best_fitness;
        result.best_configuration = self.global_best_configuration.clone();
        result.best_metrics = self.global_best_metrics.clone();
        result.convergence_history = self.convergence_history.clone();
        result.iterations_completed = self.iteration_count;
        result.execution_time_ms = execution_time.as_millis();
        
        // Collect agent performances
        for agent in &self.agents {
            result.agent_performances.insert(
                agent.id.clone(),
                agent.personal_best_fitness,
            );
        }
        
        println!("Optimization completed in {:.2}s", execution_time.as_secs_f64());
        println!("Best fitness: {:.6}", result.best_fitness);
        
        result
    }
    
    fn evaluate_all_agents<F>(&mut self, fitness_evaluator: &mut F) 
    where 
        F: FnMut(&RANConfiguration) -> (f32, RANMetrics),
    {
        for agent in &mut self.agents {
            let config = agent.position_to_configuration();
            let (fitness, metrics) = fitness_evaluator(&config);
            agent.update_fitness(fitness);
            
            // Store the best configuration and metrics
            if fitness > self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best_position = agent.position.clone();
                self.global_best_configuration = config;
                self.global_best_metrics = metrics;
            }
        }
    }
    
    fn update_swarm(&mut self) {
        for agent in &mut self.agents {
            // Update velocity
            agent.update_velocity(&self.global_best_position, &self.parameters);
            
            // Update position
            agent.update_position(&self.parameters);
        }
    }
    
    fn update_global_best(&mut self) {
        for agent in &self.agents {
            if agent.personal_best_fitness > self.global_best_fitness {
                self.global_best_fitness = agent.personal_best_fitness;
                self.global_best_position = agent.personal_best_position.clone();
                self.global_best_configuration = agent.position_to_configuration();
            }
        }
    }
    
    fn check_convergence(&self) -> bool {
        if self.convergence_history.len() < 20 {
            return false;
        }
        
        let recent_history = &self.convergence_history[self.convergence_history.len() - 20..];
        let max_fitness = recent_history.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_fitness = recent_history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        
        (max_fitness - min_fitness) < self.parameters.convergence_threshold
    }
    
    fn apply_adaptive_mechanisms(&mut self, iteration: u32) {
        // Adaptive inertia weight
        let max_iter = self.parameters.max_iterations as f32;
        let current_iter = iteration as f32;
        self.parameters.inertia_weight = 0.9 - (0.5 * current_iter / max_iter);
        
        // Diversity-based mutation
        if let Some(diversity) = self.diversity_history.last() {
            if diversity.position_diversity < 0.1 {
                self.apply_mutation(0.05);
            }
        }
        
        // Elite preservation and replacement
        if iteration % 20 == 0 {
            self.apply_elite_replacement();
        }
        
        // Adaptive specialization
        self.balance_specializations();
    }
    
    fn apply_mutation(&mut self, mutation_rate: f32) {
        let worst_agents = self.get_worst_agents(self.parameters.population_size / 4);
        
        for agent_idx in worst_agents {
            if agent_idx < self.agents.len() {
                self.agents[agent_idx].mutate(mutation_rate);
            }
        }
    }
    
    fn apply_elite_replacement(&mut self) {
        let elite_count = self.parameters.elite_size;
        let worst_agents = self.get_worst_agents(elite_count);
        let best_agents = self.get_best_agents(elite_count);
        
        for (worst_idx, best_idx) in worst_agents.iter().zip(best_agents.iter()) {
            if *worst_idx < self.agents.len() && *best_idx < self.agents.len() {
                let cloned_agent = self.agents[*best_idx].clone_agent();
                self.agents[*worst_idx] = cloned_agent;
            }
        }
    }
    
    fn balance_specializations(&mut self) {
        let current_diversity = SwarmDiversity::calculate(&self.agents);
        let total_agents = self.agents.len() as f32;
        
        // Target distribution (roughly equal)
        let target_per_spec = total_agents / 5.0;
        
        for (spec, count) in &current_diversity.specialization_distribution {
            if *count as f32 > target_per_spec * 1.5 {
                // Too many of this specialization, convert some to underrepresented ones
                self.rebalance_specialization(spec.clone(), *count);
            }
        }
    }
    
    fn rebalance_specialization(&mut self, over_spec: AgentSpecialization, count: usize) {
        let target_specs = [
            AgentSpecialization::ThroughputOptimizer,
            AgentSpecialization::LatencyMinimizer,
            AgentSpecialization::EnergyEfficiencyExpert,
            AgentSpecialization::InterferenceAnalyst,
        ];
        
        let mut converted = 0;
        let max_convert = count / 3; // Convert up to 1/3 of the excess
        
        for agent in &mut self.agents {
            if agent.neural_agent.specialization == over_spec && converted < max_convert {
                // Find a less represented specialization
                let new_spec = target_specs[converted % target_specs.len()].clone();
                agent.neural_agent.specialization = new_spec;
                converted += 1;
            }
        }
    }
    
    fn get_worst_agents(&self, count: usize) -> Vec<usize> {
        let mut agent_indices: Vec<usize> = (0..self.agents.len()).collect();
        agent_indices.sort_by(|&a, &b| {
            self.agents[a].personal_best_fitness
                .partial_cmp(&self.agents[b].personal_best_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        agent_indices.into_iter().take(count).collect()
    }
    
    fn get_best_agents(&self, count: usize) -> Vec<usize> {
        let mut agent_indices: Vec<usize> = (0..self.agents.len()).collect();
        agent_indices.sort_by(|&a, &b| {
            self.agents[b].personal_best_fitness
                .partial_cmp(&self.agents[a].personal_best_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        agent_indices.into_iter().take(count).collect()
    }
    
    pub fn simulate_ran_environment(&self, config: &RANConfiguration, rng: &mut impl Rng) -> RANMetrics {
        // Enhanced RAN simulation with realistic constraints
        let mut metrics = RANMetrics::new();
        
        // Base metrics influenced by configuration
        let power_factor = (config.power_level - 5.0) / 35.0; // Normalize to 0-1
        let bandwidth_factor = config.bandwidth / 80.0;
        let freq_factor = (config.frequency_band - 2400.0) / 1100.0;
        
        // Throughput calculation
        metrics.throughput = (50.0 + power_factor * 30.0 + bandwidth_factor * 20.0) 
            * (1.0 + rng.gen_range(-0.1..0.1));
        
        // Latency calculation (inverse relationship with some parameters)
        metrics.latency = (20.0 - power_factor * 5.0 + freq_factor * 10.0)
            * (1.0 + rng.gen_range(-0.2..0.2));
        metrics.latency = metrics.latency.max(1.0);
        
        // Energy efficiency
        metrics.energy_efficiency = (0.6 + bandwidth_factor * 0.2 - power_factor * 0.1)
            * (1.0 + rng.gen_range(-0.1..0.1));
        metrics.energy_efficiency = metrics.energy_efficiency.clamp(0.1, 1.0);
        
        // Interference level
        metrics.interference_level = (power_factor * 0.3 + freq_factor * 0.2)
            * (1.0 + rng.gen_range(-0.1..0.1));
        metrics.interference_level = metrics.interference_level.clamp(0.0, 1.0);
        
        metrics
    }
    
    pub fn get_optimization_summary(&self) -> OptimizationSummary {
        let mut summary = OptimizationSummary::new();
        summary.timestamp = Utc::now();
        summary.total_iterations = self.iteration_count;
        summary.best_fitness = self.global_best_fitness;
        summary.best_configuration = self.global_best_configuration.clone();
        summary.best_metrics = self.global_best_metrics.clone();
        summary.convergence_history = self.convergence_history.clone();
        
        // Collect agent performances
        for agent in &self.agents {
            let performance = AgentPerformance {
                agent_id: agent.id.clone(),
                specialization: format!("{:?}", agent.neural_agent.specialization),
                personal_best_fitness: agent.personal_best_fitness,
                iterations_without_improvement: agent.iterations_without_improvement,
                total_evaluations: self.iteration_count,
                success_rate: if agent.personal_best_fitness > 0.0 { 1.0 } else { 0.0 },
            };
            summary.agent_performances.push(performance);
        }
        
        summary
    }
}