//! Additional PSO Methods for Multi-Objective Network Optimization
//! 
//! This module contains the implementation of multi-objective and Pareto optimization
//! methods for the enhanced PSO system.

use crate::models::{RANConfiguration, RANMetrics, AgentSpecialization};
use crate::swarm::{SwarmAgent, SwarmParameters};
use crate::swarm::pso::*;
use crate::swarm::multi_objective_fitness::NetworkFitnessEvaluator;
use std::collections::HashMap;

impl ParticleSwarmOptimizer {
    /// Evaluate multi-objective fitness for all agents
    pub fn evaluate_multi_objective_fitness(&mut self, agents: &mut [SwarmAgent]) {
        let mut evaluator = NetworkFitnessEvaluator::new();
        
        for agent in agents.iter_mut() {
            let config = agent.position_to_configuration();
            let fitness_scores = evaluator.evaluate_fitness(
                &config,
                &self.current_network_conditions,
                &agent.neural_agent.specialization,
            );
            
            // Update agent fitness with weighted composite score
            agent.update_fitness(fitness_scores.weighted_composite);
            
            // Store detailed fitness scores for analysis
            self.store_agent_fitness_details(&agent.id, fitness_scores);
        }
    }
    
    /// Store detailed fitness scores for later analysis
    fn store_agent_fitness_details(&mut self, agent_id: &str, scores: NetworkFitnessScores) {
        // In a real implementation, this would store to a database or file
        // For now, we'll just track in memory
    }
    
    /// Update Pareto archive with non-dominated solutions
    pub fn update_pareto_archive(&mut self, agents: &[SwarmAgent]) {
        for agent in agents {
            let config = agent.position_to_configuration();
            let metrics = self.calculate_ran_metrics(&config);
            let fitness_scores = self.calculate_fitness_scores(&config);
            
            let solution = ParetoSolution {
                position: agent.position.clone(),
                fitness_scores,
                configuration: config,
                metrics,
                dominance_rank: 0,
            };
            
            // Check if this solution is non-dominated
            if self.is_non_dominated(&solution) {
                self.add_to_pareto_archive(solution);
            }
        }
        
        // Clean up archive to maintain only non-dominated solutions
        self.clean_pareto_archive();
    }
    
    /// Check if a solution is non-dominated by existing archive
    fn is_non_dominated(&self, candidate: &ParetoSolution) -> bool {
        for archived_solution in &self.multi_objective_archive {
            if self.dominates(&archived_solution.fitness_scores, &candidate.fitness_scores) {
                return false;
            }
        }
        true
    }
    
    /// Check if solution A dominates solution B
    fn dominates(&self, a: &NetworkFitnessScores, b: &NetworkFitnessScores) -> bool {
        let objectives = [
            (a.throughput, b.throughput, true),              // maximize
            (a.latency, b.latency, false),                   // minimize (inverted in score)
            (a.energy_efficiency, b.energy_efficiency, true), // maximize
            (a.interference_level, b.interference_level, false), // minimize (inverted in score)
            (a.handover_success_rate, b.handover_success_rate, true), // maximize
            (a.endc_establishment_success, b.endc_establishment_success, true), // maximize
        ];
        
        let mut better_in_at_least_one = false;
        
        for (a_val, b_val, maximize) in objectives {
            if maximize {
                if a_val < b_val {
                    return false; // A is worse in this objective
                }
                if a_val > b_val {
                    better_in_at_least_one = true;
                }
            } else {
                if a_val > b_val {
                    return false; // A is worse in this objective (higher is worse for minimize)
                }
                if a_val < b_val {
                    better_in_at_least_one = true;
                }
            }
        }
        
        better_in_at_least_one
    }
    
    /// Add solution to Pareto archive
    fn add_to_pareto_archive(&mut self, solution: ParetoSolution) {
        self.multi_objective_archive.push(solution);
        
        // Limit archive size
        if self.multi_objective_archive.len() > 100 {
            self.multi_objective_archive.sort_by(|a, b| {
                b.fitness_scores.weighted_composite
                    .partial_cmp(&a.fitness_scores.weighted_composite)
                    .unwrap()
            });
            self.multi_objective_archive.truncate(50);
        }
    }
    
    /// Clean Pareto archive to remove dominated solutions
    fn clean_pareto_archive(&mut self) {
        let mut non_dominated = Vec::new();
        
        for i in 0..self.multi_objective_archive.len() {
            let mut is_dominated = false;
            
            for j in 0..self.multi_objective_archive.len() {
                if i != j && self.dominates(
                    &self.multi_objective_archive[j].fitness_scores,
                    &self.multi_objective_archive[i].fitness_scores,
                ) {
                    is_dominated = true;
                    break;
                }
            }
            
            if !is_dominated {
                non_dominated.push(self.multi_objective_archive[i].clone());
            }
        }
        
        self.multi_objective_archive = non_dominated;
    }
    
    /// Update global best from Pareto front
    pub fn update_global_best_from_pareto(&mut self) {
        if let Some(best_solution) = self.multi_objective_archive
            .iter()
            .max_by(|a, b| {
                a.fitness_scores.weighted_composite
                    .partial_cmp(&b.fitness_scores.weighted_composite)
                    .unwrap()
            })
        {
            self.global_best_position = best_solution.position.clone();
            self.global_best_fitness = best_solution.fitness_scores.weighted_composite;
        }
    }
    
    /// Update sub-swarm local bests
    pub fn update_sub_swarm_bests(&mut self, agents: &[SwarmAgent]) {
        for sub_swarm in &mut self.sub_swarms {
            let mut best_fitness = f32::NEG_INFINITY;
            let mut best_position = vec![0.0; self.global_best_position.len()];
            
            for agent_id in &sub_swarm.agents {
                if let Some(agent) = agents.iter().find(|a| a.id == *agent_id) {
                    if agent.current_fitness > best_fitness {
                        best_fitness = agent.current_fitness;
                        best_position = agent.position.clone();
                    }
                }
            }
            
            sub_swarm.local_best_fitness = best_fitness;
            sub_swarm.local_best_position = best_position;
        }
    }
    
    /// Calculate RAN metrics from configuration
    fn calculate_ran_metrics(&self, config: &RANConfiguration) -> RANMetrics {
        let mut evaluator = NetworkFitnessEvaluator::new();
        let scores = evaluator.evaluate_fitness(
            config,
            &self.current_network_conditions,
            &AgentSpecialization::GeneralPurpose,
        );
        
        RANMetrics {
            throughput: scores.throughput * 100.0, // Convert to Mbps
            latency: (1.0 - scores.latency) * 50.0, // Convert back to ms
            energy_efficiency: scores.energy_efficiency,
            interference_level: 1.0 - scores.interference_level, // Convert back to interference level
        }
    }
    
    /// Calculate fitness scores from configuration
    fn calculate_fitness_scores(&self, config: &RANConfiguration) -> NetworkFitnessScores {
        let mut evaluator = NetworkFitnessEvaluator::new();
        evaluator.evaluate_fitness(
            config,
            &self.current_network_conditions,
            &AgentSpecialization::GeneralPurpose,
        )
    }
    
    /// Get best solutions for each objective
    pub fn get_best_solutions_per_objective(&self) -> HashMap<String, ParetoSolution> {
        let mut best_solutions = HashMap::new();
        
        if !self.multi_objective_archive.is_empty() {
            // Best throughput solution
            if let Some(best_throughput) = self.multi_objective_archive
                .iter()
                .max_by(|a, b| {
                    a.fitness_scores.throughput
                        .partial_cmp(&b.fitness_scores.throughput)
                        .unwrap()
                })
            {
                best_solutions.insert("throughput".to_string(), best_throughput.clone());
            }
            
            // Best latency solution
            if let Some(best_latency) = self.multi_objective_archive
                .iter()
                .max_by(|a, b| {
                    a.fitness_scores.latency
                        .partial_cmp(&b.fitness_scores.latency)
                        .unwrap()
                })
            {
                best_solutions.insert("latency".to_string(), best_latency.clone());
            }
            
            // Best energy efficiency solution
            if let Some(best_energy) = self.multi_objective_archive
                .iter()
                .max_by(|a, b| {
                    a.fitness_scores.energy_efficiency
                        .partial_cmp(&b.fitness_scores.energy_efficiency)
                        .unwrap()
                })
            {
                best_solutions.insert("energy_efficiency".to_string(), best_energy.clone());
            }
            
            // Best interference solution
            if let Some(best_interference) = self.multi_objective_archive
                .iter()
                .max_by(|a, b| {
                    a.fitness_scores.interference_level
                        .partial_cmp(&b.fitness_scores.interference_level)
                        .unwrap()
                })
            {
                best_solutions.insert("interference".to_string(), best_interference.clone());
            }
        }
        
        best_solutions
    }
    
    /// Calculate hypervolume indicator for Pareto front quality
    pub fn calculate_hypervolume(&self, reference_point: &[f32]) -> f32 {
        if self.multi_objective_archive.is_empty() {
            return 0.0;
        }
        
        let mut hypervolume = 0.0;
        
        // Simple hypervolume calculation (for demonstration)
        for solution in &self.multi_objective_archive {
            let objectives = [
                solution.fitness_scores.throughput,
                solution.fitness_scores.latency,
                solution.fitness_scores.energy_efficiency,
                solution.fitness_scores.interference_level,
            ];
            
            let mut volume = 1.0;
            for (i, &obj_val) in objectives.iter().enumerate() {
                if i < reference_point.len() {
                    volume *= (obj_val - reference_point[i]).max(0.0);
                }
            }
            hypervolume += volume;
        }
        
        hypervolume
    }
    
    /// Generate optimization report
    pub fn generate_optimization_report(&self) -> OptimizationReport {
        let best_solutions = self.get_best_solutions_per_objective();
        let hypervolume = self.calculate_hypervolume(&[0.0, 0.0, 0.0, 0.0]);
        
        OptimizationReport {
            pareto_front_size: self.multi_objective_archive.len(),
            hypervolume,
            best_solutions,
            convergence_rate: self.get_convergence_rate(),
            iterations_completed: self.iteration,
            network_conditions: self.current_network_conditions.clone(),
            constraint_violations: self.calculate_total_constraint_violations(),
        }
    }
    
    /// Calculate total constraint violations across all solutions
    fn calculate_total_constraint_violations(&self) -> f32 {
        let mut total_violations = 0.0;
        let evaluator = NetworkFitnessEvaluator::new();
        
        for solution in &self.multi_objective_archive {
            total_violations += evaluator.calculate_constraint_violations(
                &solution.configuration,
                &self.network_constraints,
            );
        }
        
        if !self.multi_objective_archive.is_empty() {
            total_violations / self.multi_objective_archive.len() as f32
        } else {
            0.0
        }
    }
    
    /// Adapt optimization strategy based on network conditions
    pub fn adapt_optimization_strategy(&mut self) {
        let conditions = &self.current_network_conditions;
        
        // Adjust sub-swarm weights based on network conditions
        match conditions.traffic_pattern {
            TrafficPattern::VoIP => {
                // Prioritize latency optimization
                self.adjust_objective_weights("latency", 0.4);
            }
            TrafficPattern::Video => {
                // Prioritize throughput and latency
                self.adjust_objective_weights("throughput", 0.35);
                self.adjust_objective_weights("latency", 0.3);
            }
            TrafficPattern::DataTransfer => {
                // Prioritize throughput
                self.adjust_objective_weights("throughput", 0.5);
            }
            TrafficPattern::Gaming => {
                // Balance latency and throughput
                self.adjust_objective_weights("latency", 0.35);
                self.adjust_objective_weights("throughput", 0.3);
            }
            TrafficPattern::IoT => {
                // Prioritize energy efficiency
                self.adjust_objective_weights("energy_efficiency", 0.4);
            }
            TrafficPattern::Mixed => {
                // Keep balanced weights
            }
        }
        
        // Adjust parameters based on load
        if conditions.load_factor > 0.8 {
            // High load: increase exploration
            self.adaptive_parameters.current_inertia *= 1.1;
            self.adaptive_parameters.diversity_threshold *= 1.2;
        } else if conditions.load_factor < 0.3 {
            // Low load: increase exploitation
            self.adaptive_parameters.current_inertia *= 0.9;
            self.adaptive_parameters.current_social *= 1.1;
        }
    }
    
    /// Adjust objective weights dynamically
    fn adjust_objective_weights(&mut self, objective: &str, weight: f32) {
        // This would update the optimization objective weights
        // Implementation depends on how objectives are stored
    }
}

/// Optimization report structure
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub pareto_front_size: usize,
    pub hypervolume: f32,
    pub best_solutions: HashMap<String, ParetoSolution>,
    pub convergence_rate: f32,
    pub iterations_completed: u32,
    pub network_conditions: NetworkConditions,
    pub constraint_violations: f32,
}

impl OptimizationReport {
    /// Format report as string for logging
    pub fn format_report(&self) -> String {
        format!(
            "Optimization Report:\n\
            - Pareto Front Size: {}\n\
            - Hypervolume: {:.4}\n\
            - Convergence Rate: {:.4}\n\
            - Iterations: {}\n\
            - Constraint Violations: {:.4}\n\
            - Load Factor: {:.2}\n\
            - Interference Level: {:.2}",
            self.pareto_front_size,
            self.hypervolume,
            self.convergence_rate,
            self.iterations_completed,
            self.constraint_violations,
            self.network_conditions.load_factor,
            self.network_conditions.interference_level
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::AgentSpecialization;
    
    #[test]
    fn test_pareto_dominance() {
        let pso = ParticleSwarmOptimizer::new(SwarmParameters::default(), 4);
        
        let mut scores_a = NetworkFitnessScores::default();
        scores_a.throughput = 0.8;
        scores_a.latency = 0.9;
        scores_a.energy_efficiency = 0.7;
        
        let mut scores_b = NetworkFitnessScores::default();
        scores_b.throughput = 0.7;
        scores_b.latency = 0.8;
        scores_b.energy_efficiency = 0.6;
        
        assert!(pso.dominates(&scores_a, &scores_b));
        assert!(!pso.dominates(&scores_b, &scores_a));
    }
    
    #[test]
    fn test_hypervolume_calculation() {
        let mut pso = ParticleSwarmOptimizer::new(SwarmParameters::default(), 4);
        
        // Add some test solutions to archive
        let solution = ParetoSolution {
            position: vec![0.5, 0.5, 0.5, 0.5],
            fitness_scores: NetworkFitnessScores {
                throughput: 0.8,
                latency: 0.9,
                energy_efficiency: 0.7,
                interference_level: 0.8,
                ..Default::default()
            },
            configuration: RANConfiguration::new(1),
            metrics: RANMetrics::new(),
            dominance_rank: 0,
        };
        
        pso.multi_objective_archive.push(solution);
        
        let hypervolume = pso.calculate_hypervolume(&[0.0, 0.0, 0.0, 0.0]);
        assert!(hypervolume > 0.0);
    }
}