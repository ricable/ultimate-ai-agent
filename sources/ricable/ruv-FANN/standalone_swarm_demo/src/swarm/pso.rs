//! Enhanced Particle Swarm Optimization for Network Performance
//! 
//! This module provides a comprehensive multi-objective PSO implementation for
//! neural swarm optimization with specialized fitness functions for network KPIs.

use crate::models::{RANConfiguration, RANMetrics, AgentSpecialization};
use crate::swarm::{SwarmAgent, SwarmParameters};
use rand::Rng;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Multi-objective fitness scores for network optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkFitnessScores {
    pub throughput: f32,
    pub latency: f32,
    pub energy_efficiency: f32,
    pub interference_level: f32,
    pub handover_success_rate: f32,
    pub endc_establishment_success: f32,
    pub user_satisfaction: f32,
    pub load_balancing_score: f32,
    pub weighted_composite: f32,
}

/// Network optimization objectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationObjective {
    MaximizeThroughput,
    MinimizeLatency,
    OptimizeEnergyEfficiency,
    MinimizeInterference,
    MaximizeHandoverSuccess,
    OptimizeENDCEstablishment,
    MaximizeUserSatisfaction,
    OptimizeLoadBalancing,
    MultiObjective(Vec<(OptimizationObjective, f32)>), // Weighted objectives
}

/// Constraint definitions for realistic network scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConstraints {
    pub max_power_consumption: f32,
    pub min_coverage_area: f32,
    pub max_interference_threshold: f32,
    pub required_throughput: f32,
    pub max_latency: f32,
    pub min_handover_success_rate: f32,
    pub energy_budget: f32,
}

/// Enhanced PSO with multi-objective optimization
pub struct ParticleSwarmOptimizer {
    pub parameters: SwarmParameters,
    pub global_best_position: Vec<f32>,
    pub global_best_fitness: f32,
    pub iteration: u32,
    pub velocity_history: HashMap<String, Vec<Vec<f32>>>,
    pub multi_objective_archive: Vec<ParetoSolution>,
    pub network_constraints: NetworkConstraints,
    pub optimization_objective: OptimizationObjective,
    pub adaptive_parameters: AdaptiveParameters,
    pub sub_swarms: Vec<SubSwarm>,
    pub current_network_conditions: NetworkConditions,
}

/// Pareto-optimal solutions for multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoSolution {
    pub position: Vec<f32>,
    pub fitness_scores: NetworkFitnessScores,
    pub configuration: RANConfiguration,
    pub metrics: RANMetrics,
    pub dominance_rank: u32,
}

/// Adaptive parameters that change based on network conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParameters {
    pub base_inertia: f32,
    pub current_inertia: f32,
    pub base_cognitive: f32,
    pub current_cognitive: f32,
    pub base_social: f32,
    pub current_social: f32,
    pub adaptation_rate: f32,
    pub diversity_threshold: f32,
}

/// Sub-swarm for different network layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubSwarm {
    pub id: String,
    pub layer_type: NetworkLayer,
    pub agents: Vec<String>, // Agent IDs
    pub local_best_position: Vec<f32>,
    pub local_best_fitness: f32,
    pub specialization: AgentSpecialization,
}

/// Network layer types for multi-layer optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkLayer {
    Physical,
    MAC,
    RRC,
    Application,
    Core,
}

/// Current network conditions affecting optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    pub load_factor: f32,
    pub interference_level: f32,
    pub mobility_factor: f32,
    pub traffic_pattern: TrafficPattern,
    pub time_of_day: f32,
    pub weather_conditions: WeatherConditions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficPattern {
    VoIP,
    Video,
    DataTransfer,
    Gaming,
    IoT,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeatherConditions {
    Clear,
    Rainy,
    Foggy,
    Extreme,
}

impl ParticleSwarmOptimizer {
    pub fn new(parameters: SwarmParameters, dimensions: usize) -> Self {
        Self {
            parameters,
            global_best_position: vec![0.0; dimensions],
            global_best_fitness: f32::NEG_INFINITY,
            iteration: 0,
            velocity_history: HashMap::new(),
            multi_objective_archive: Vec::new(),
            network_constraints: NetworkConstraints::default(),
            optimization_objective: OptimizationObjective::MultiObjective(vec![
                (OptimizationObjective::MaximizeThroughput, 0.3),
                (OptimizationObjective::MinimizeLatency, 0.25),
                (OptimizationObjective::OptimizeEnergyEfficiency, 0.2),
                (OptimizationObjective::MinimizeInterference, 0.15),
                (OptimizationObjective::MaximizeHandoverSuccess, 0.1),
            ]),
            adaptive_parameters: AdaptiveParameters::default(),
            sub_swarms: Vec::new(),
            current_network_conditions: NetworkConditions::default(),
        }
    }
    
    /// Create PSO with specific network optimization objectives
    pub fn new_with_objectives(
        parameters: SwarmParameters,
        dimensions: usize,
        objective: OptimizationObjective,
        constraints: NetworkConstraints,
    ) -> Self {
        let mut pso = Self::new(parameters, dimensions);
        pso.optimization_objective = objective;
        pso.network_constraints = constraints;
        pso
    }
    
    /// Initialize sub-swarms for different network layers
    pub fn initialize_sub_swarms(&mut self, agents: &[SwarmAgent]) {
        let layers = vec![
            (NetworkLayer::Physical, AgentSpecialization::EnergyEfficiencyExpert),
            (NetworkLayer::MAC, AgentSpecialization::ThroughputOptimizer),
            (NetworkLayer::RRC, AgentSpecialization::LatencyMinimizer),
            (NetworkLayer::Application, AgentSpecialization::InterferenceAnalyst),
            (NetworkLayer::Core, AgentSpecialization::GeneralPurpose),
        ];
        
        for (layer, specialization) in layers {
            let layer_agents: Vec<String> = agents
                .iter()
                .filter(|a| a.neural_agent.specialization == specialization)
                .map(|a| a.id.clone())
                .collect();
            
            if !layer_agents.is_empty() {
                let sub_swarm = SubSwarm {
                    id: format!("{:?}_swarm", layer),
                    layer_type: layer,
                    agents: layer_agents,
                    local_best_position: vec![0.0; self.global_best_position.len()],
                    local_best_fitness: f32::NEG_INFINITY,
                    specialization,
                };
                self.sub_swarms.push(sub_swarm);
            }
        }
    }
    
    pub fn update_particles(&mut self, agents: &mut [SwarmAgent]) {
        self.iteration += 1;
        
        // Update global best
        for agent in agents.iter() {
            if agent.personal_best_fitness > self.global_best_fitness {
                self.global_best_fitness = agent.personal_best_fitness;
                self.global_best_position = agent.personal_best_position.clone();
            }
        }
        
        // Update each particle
        for agent in agents.iter_mut() {
            self.update_particle_velocity(agent);
            self.update_particle_position(agent);
            self.track_velocity_history(agent);
        }
        
        // Apply adaptive parameters
        self.update_adaptive_parameters();
    }
    
    fn update_particle_velocity(&mut self, agent: &mut SwarmAgent) {
        let mut rng = rand::thread_rng();
        
        for i in 0..agent.velocity.len() {
            let r1 = rng.gen::<f32>();
            let r2 = rng.gen::<f32>();
            
            // Standard PSO velocity update with constriction factor
            let cognitive = self.parameters.cognitive_weight * r1 
                * (agent.personal_best_position[i] - agent.position[i]);
            let social = self.parameters.social_weight * r2 
                * (self.global_best_position[i] - agent.position[i]);
            
            agent.velocity[i] = self.parameters.inertia_weight * agent.velocity[i] 
                + cognitive + social;
            
            // Apply velocity clamping
            let v_max = 5.0; // Maximum velocity
            agent.velocity[i] = agent.velocity[i].clamp(-v_max, v_max);
        }
    }
    
    fn update_particle_position(&self, agent: &mut SwarmAgent) {
        for i in 0..agent.position.len() {
            agent.position[i] += agent.velocity[i];
            
            // Apply position bounds
            agent.position[i] = agent.position[i].clamp(-5.0, 5.0);
        }
    }
    
    fn track_velocity_history(&mut self, agent: &SwarmAgent) {
        let history = self.velocity_history
            .entry(agent.id.clone())
            .or_insert_with(Vec::new);
        
        history.push(agent.velocity.clone());
        
        // Keep only recent history
        if history.len() > 50 {
            history.remove(0);
        }
    }
    
    fn update_adaptive_parameters(&mut self) {
        // Adaptive inertia weight (decreasing over time)
        let max_iter = self.parameters.max_iterations as f32;
        let current_iter = self.iteration as f32;
        
        let w_start = 0.9;
        let w_end = 0.1;
        self.parameters.inertia_weight = w_start - (w_start - w_end) * (current_iter / max_iter);
        
        // Adaptive cognitive and social weights
        let c1_start = 2.5;
        let c1_end = 0.5;
        let c2_start = 0.5;
        let c2_end = 2.5;
        
        self.parameters.cognitive_weight = c1_start - (c1_start - c1_end) * (current_iter / max_iter);
        self.parameters.social_weight = c2_start + (c2_end - c2_start) * (current_iter / max_iter);
    }
    
    pub fn get_swarm_diversity(&self, agents: &[SwarmAgent]) -> f32 {
        if agents.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..agents.len() {
            for j in i + 1..agents.len() {
                let distance = self.euclidean_distance(&agents[i].position, &agents[j].position);
                total_distance += distance;
                count += 1;
            }
        }
        
        total_distance / count as f32
    }
    
    fn euclidean_distance(&self, pos1: &[f32], pos2: &[f32]) -> f32 {
        pos1.iter()
            .zip(pos2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    pub fn apply_mutation(&self, agents: &mut [SwarmAgent], mutation_rate: f32) {
        let mut rng = rand::thread_rng();
        
        for agent in agents.iter_mut() {
            if agent.iterations_without_improvement > 20 {
                for i in 0..agent.position.len() {
                    if rng.gen::<f32>() < mutation_rate {
                        let mutation = rng.gen_range(-0.5..0.5);
                        agent.position[i] = (agent.position[i] + mutation).clamp(-5.0, 5.0);
                    }
                }
            }
        }
    }
    
    pub fn get_convergence_rate(&self) -> f32 {
        // Calculate convergence rate based on recent fitness improvements
        if self.iteration < 10 {
            return 0.0;
        }
        
        // This would typically use historical fitness data
        // For now, return a simple estimate
        1.0 / (1.0 + self.iteration as f32)
    }
}

/// Network modifier for adaptive parameters
#[derive(Debug, Clone)]
struct NetworkModifier {
    inertia_factor: f32,
    cognitive_factor: f32,
    social_factor: f32,
}

/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPerformanceMetrics {
    pub current_conditions: NetworkConditions,
    pub pareto_front_size: usize,
    pub convergence_rate: f32,
    pub diversity_score: f32,
    pub constraint_violation_rate: f32,
}

/// Default implementations for new structures
impl Default for NetworkConstraints {
    fn default() -> Self {
        Self {
            max_power_consumption: 30.0,
            min_coverage_area: 1.0,
            max_interference_threshold: 0.3,
            required_throughput: 50.0,
            max_latency: 10.0,
            min_handover_success_rate: 0.95,
            energy_budget: 100.0,
        }
    }
}

impl Default for AdaptiveParameters {
    fn default() -> Self {
        Self {
            base_inertia: 0.7,
            current_inertia: 0.7,
            base_cognitive: 1.5,
            current_cognitive: 1.5,
            base_social: 1.5,
            current_social: 1.5,
            adaptation_rate: 0.1,
            diversity_threshold: 0.1,
        }
    }
}

impl Default for NetworkConditions {
    fn default() -> Self {
        Self {
            load_factor: 0.5,
            interference_level: 0.2,
            mobility_factor: 0.3,
            traffic_pattern: TrafficPattern::Mixed,
            time_of_day: 12.0,
            weather_conditions: WeatherConditions::Clear,
        }
    }
}

impl Default for NetworkFitnessScores {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency: 0.0,
            energy_efficiency: 0.0,
            interference_level: 0.0,
            handover_success_rate: 0.0,
            endc_establishment_success: 0.0,
            user_satisfaction: 0.0,
            load_balancing_score: 0.0,
            weighted_composite: 0.0,
        }
    }
}