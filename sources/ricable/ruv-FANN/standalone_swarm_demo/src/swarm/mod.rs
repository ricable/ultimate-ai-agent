//! Swarm Coordination Module
//! 
//! This module implements the core swarm intelligence algorithms and coordination
//! mechanisms for the neural swarm optimization system.

use crate::models::{RANConfiguration, RANMetrics, AgentSpecialization, OptimizationSummary};
use crate::neural::NeuralAgent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use rand::Rng;

pub mod pso;
pub mod coordinator;
pub mod communication;
pub mod multi_objective_fitness;
pub mod pso_methods;
pub mod pso_completion;

pub use pso::{ParticleSwarmOptimizer, NetworkFitnessScores, OptimizationObjective, NetworkConstraints, NetworkConditions, TrafficPattern, WeatherConditions};
pub use coordinator::SwarmCoordinator;
pub use communication::CommunicationProtocol;
pub use multi_objective_fitness::{NetworkFitnessEvaluator, TrafficModel, InterferenceModel};
pub use pso_methods::OptimizationReport;
pub use pso_completion::NetworkPerformanceMetrics;

/// Swarm optimization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub inertia_weight: f32,
    pub cognitive_weight: f32,
    pub social_weight: f32,
    pub convergence_threshold: f32,
    pub elite_size: usize,
}

impl Default for SwarmParameters {
    fn default() -> Self {
        Self {
            population_size: 20,
            max_iterations: 100,
            inertia_weight: 0.7,
            cognitive_weight: 1.5,
            social_weight: 1.5,
            convergence_threshold: 0.001,
            elite_size: 5,
        }
    }
}

/// Swarm agent with position and velocity for PSO
#[derive(Debug, Clone)]
pub struct SwarmAgent {
    pub id: String,
    pub neural_agent: NeuralAgent,
    pub position: Vec<f32>,
    pub velocity: Vec<f32>,
    pub personal_best_position: Vec<f32>,
    pub personal_best_fitness: f32,
    pub current_fitness: f32,
    pub iterations_without_improvement: u32,
}

impl SwarmAgent {
    pub fn new(id: String, specialization: AgentSpecialization, dimensions: usize) -> Self {
        let neural_agent = NeuralAgent::new(id.clone(), specialization);
        
        // Initialize position and velocity randomly
        let mut rng = rand::thread_rng();
        let position: Vec<f32> = (0..dimensions).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let velocity: Vec<f32> = (0..dimensions).map(|_| rng.gen_range(-0.1..0.1)).collect();
        
        Self {
            id,
            neural_agent,
            position: position.clone(),
            velocity,
            personal_best_position: position,
            personal_best_fitness: f32::NEG_INFINITY,
            current_fitness: f32::NEG_INFINITY,
            iterations_without_improvement: 0,
        }
    }
    
    pub fn update_position(&mut self, params: &SwarmParameters) {
        for i in 0..self.position.len() {
            self.position[i] += self.velocity[i];
            
            // Apply bounds
            self.position[i] = self.position[i].clamp(-5.0, 5.0);
        }
    }
    
    pub fn update_velocity(
        &mut self,
        global_best_position: &[f32],
        params: &SwarmParameters,
    ) {
        let mut rng = rand::thread_rng();
        
        for i in 0..self.velocity.len() {
            let r1 = rng.gen::<f32>();
            let r2 = rng.gen::<f32>();
            
            let cognitive_component = params.cognitive_weight * r1 
                * (self.personal_best_position[i] - self.position[i]);
            let social_component = params.social_weight * r2 
                * (global_best_position[i] - self.position[i]);
            
            self.velocity[i] = params.inertia_weight * self.velocity[i] 
                + cognitive_component + social_component;
            
            // Apply velocity limits
            self.velocity[i] = self.velocity[i].clamp(-1.0, 1.0);
        }
    }
    
    pub fn update_fitness(&mut self, fitness: f32) {
        self.current_fitness = fitness;
        
        if fitness > self.personal_best_fitness {
            self.personal_best_fitness = fitness;
            self.personal_best_position = self.position.clone();
            self.iterations_without_improvement = 0;
        } else {
            self.iterations_without_improvement += 1;
        }
        
        // Update neural agent performance
        self.neural_agent.add_performance_score(fitness);
    }
    
    pub fn position_to_configuration(&self) -> RANConfiguration {
        // Convert normalized position to actual RAN configuration
        let mut config = RANConfiguration::new(self.id.parse().unwrap_or(0));
        
        if self.position.len() >= 4 {
            config.power_level = (self.position[0] * 15.0 + 20.0).clamp(5.0, 40.0);
            config.antenna_tilt = (self.position[1] * 5.0).clamp(-10.0, 10.0);
            config.bandwidth = match self.position[2] {
                x if x > 0.5 => 80.0,
                x if x > 0.0 => 40.0,
                _ => 20.0,
            };
            config.frequency_band = (self.position[3] * 1000.0 + 2500.0).clamp(2400.0, 3500.0);
        }
        
        config
    }
    
    pub fn mutate(&mut self, mutation_rate: f32) {
        let mut rng = rand::thread_rng();
        
        for i in 0..self.position.len() {
            if rng.gen::<f32>() < mutation_rate {
                let mutation = rng.gen_range(-0.1..0.1);
                self.position[i] = (self.position[i] + mutation).clamp(-5.0, 5.0);
            }
        }
    }
    
    pub fn clone_agent(&self) -> Self {
        let mut clone = self.clone();
        clone.id = format!("{}_clone", self.id);
        clone.neural_agent.id = clone.id.clone();
        clone
    }
}

/// Swarm optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmOptimizationResult {
    pub best_fitness: f32,
    pub best_configuration: RANConfiguration,
    pub best_metrics: RANMetrics,
    pub convergence_history: Vec<f32>,
    pub iterations_completed: u32,
    pub execution_time_ms: u128,
    pub agent_performances: HashMap<String, f32>,
}

impl SwarmOptimizationResult {
    pub fn new() -> Self {
        Self {
            best_fitness: f32::NEG_INFINITY,
            best_configuration: RANConfiguration::new(0),
            best_metrics: RANMetrics::new(),
            convergence_history: Vec::new(),
            iterations_completed: 0,
            execution_time_ms: 0,
            agent_performances: HashMap::new(),
        }
    }
    
    pub fn to_optimization_summary(&self) -> OptimizationSummary {
        let mut summary = OptimizationSummary::new();
        summary.best_fitness = self.best_fitness;
        summary.best_configuration = self.best_configuration.clone();
        summary.best_metrics = self.best_metrics.clone();
        summary.convergence_history = self.convergence_history.clone();
        summary.total_iterations = self.iterations_completed;
        summary.execution_time_seconds = self.execution_time_ms as f64 / 1000.0;
        summary
    }
}

/// Diversity measures for swarm health
#[derive(Debug, Clone)]
pub struct SwarmDiversity {
    pub position_diversity: f32,
    pub fitness_diversity: f32,
    pub specialization_distribution: HashMap<AgentSpecialization, usize>,
}

impl SwarmDiversity {
    pub fn calculate(agents: &[SwarmAgent]) -> Self {
        let position_diversity = Self::calculate_position_diversity(agents);
        let fitness_diversity = Self::calculate_fitness_diversity(agents);
        let specialization_distribution = Self::calculate_specialization_distribution(agents);
        
        Self {
            position_diversity,
            fitness_diversity,
            specialization_distribution,
        }
    }
    
    fn calculate_position_diversity(agents: &[SwarmAgent]) -> f32 {
        if agents.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..agents.len() {
            for j in i + 1..agents.len() {
                let distance = Self::euclidean_distance(&agents[i].position, &agents[j].position);
                total_distance += distance;
                count += 1;
            }
        }
        
        if count > 0 {
            total_distance / count as f32
        } else {
            0.0
        }
    }
    
    fn calculate_fitness_diversity(agents: &[SwarmAgent]) -> f32 {
        if agents.len() < 2 {
            return 0.0;
        }
        
        let fitnesses: Vec<f32> = agents.iter().map(|a| a.current_fitness).collect();
        let mean_fitness = fitnesses.iter().sum::<f32>() / fitnesses.len() as f32;
        
        let variance = fitnesses.iter()
            .map(|&f| (f - mean_fitness).powi(2))
            .sum::<f32>() / fitnesses.len() as f32;
        
        variance.sqrt()
    }
    
    fn calculate_specialization_distribution(agents: &[SwarmAgent]) -> HashMap<AgentSpecialization, usize> {
        let mut distribution = HashMap::new();
        
        for agent in agents {
            *distribution.entry(agent.neural_agent.specialization.clone()).or_insert(0) += 1;
        }
        
        distribution
    }
    
    fn euclidean_distance(pos1: &[f32], pos2: &[f32]) -> f32 {
        pos1.iter()
            .zip(pos2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_swarm_agent_creation() {
        let agent = SwarmAgent::new(
            "test_agent".to_string(),
            AgentSpecialization::ThroughputOptimizer,
            4,
        );
        
        assert_eq!(agent.id, "test_agent");
        assert_eq!(agent.position.len(), 4);
        assert_eq!(agent.velocity.len(), 4);
        assert_eq!(agent.personal_best_fitness, f32::NEG_INFINITY);
    }
    
    #[test]
    fn test_fitness_update() {
        let mut agent = SwarmAgent::new(
            "test_agent".to_string(),
            AgentSpecialization::GeneralPurpose,
            4,
        );
        
        agent.update_fitness(0.8);
        assert_eq!(agent.current_fitness, 0.8);
        assert_eq!(agent.personal_best_fitness, 0.8);
        assert_eq!(agent.iterations_without_improvement, 0);
        
        agent.update_fitness(0.5);
        assert_eq!(agent.current_fitness, 0.5);
        assert_eq!(agent.personal_best_fitness, 0.8);
        assert_eq!(agent.iterations_without_improvement, 1);
    }
    
    #[test]
    fn test_position_to_configuration() {
        let mut agent = SwarmAgent::new(
            "123".to_string(),
            AgentSpecialization::ThroughputOptimizer,
            4,
        );
        
        agent.position = vec![0.5, -0.5, 0.8, 0.2];
        let config = agent.position_to_configuration();
        
        assert_eq!(config.cell_id, 123);
        assert!(config.power_level >= 5.0 && config.power_level <= 40.0);
        assert!(config.antenna_tilt >= -10.0 && config.antenna_tilt <= 10.0);
    }
}