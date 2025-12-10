//! Core data models and structures for the Neural Swarm Platform
//! 
//! This module defines the fundamental data structures used throughout
//! the neural swarm optimization system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// RAN (Radio Access Network) configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RANConfiguration {
    pub cell_id: u32,
    pub frequency_band: f32,
    pub power_level: f32,
    pub antenna_tilt: f32,
    pub bandwidth: f32,
    pub modulation_scheme: String,
    pub mimo_config: String,
    pub beamforming_enabled: bool,
}

impl RANConfiguration {
    pub fn new(cell_id: u32) -> Self {
        Self {
            cell_id,
            frequency_band: 2400.0,
            power_level: 20.0,
            antenna_tilt: 0.0,
            bandwidth: 20.0,
            modulation_scheme: "QPSK".to_string(),
            mimo_config: "2x2".to_string(),
            beamforming_enabled: false,
        }
    }
}

/// Performance metrics for RAN optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RANMetrics {
    pub throughput: f32,
    pub latency: f32,
    pub energy_efficiency: f32,
    pub interference_level: f32,
}

impl RANMetrics {
    pub fn new() -> Self {
        Self {
            throughput: 0.0,
            latency: 0.0,
            energy_efficiency: 0.0,
            interference_level: 0.0,
        }
    }
    
    pub fn random(rng: &mut impl rand::Rng) -> Self {
        Self {
            throughput: rng.gen_range(10.0..100.0),
            latency: rng.gen_range(1.0..50.0),
            energy_efficiency: rng.gen_range(0.5..1.0),
            interference_level: rng.gen_range(0.0..0.5),
        }
    }
    
    pub fn calculate_fitness(&self) -> f32 {
        // Weighted fitness calculation
        let throughput_score = self.throughput * 0.35;
        let latency_score = (50.0 - self.latency).max(0.0) * 0.25;
        let efficiency_score = self.energy_efficiency * 0.25;
        let interference_penalty = (1.0 - self.interference_level) * 0.15;
        
        throughput_score + latency_score + efficiency_score + interference_penalty
    }
}

/// Agent specialization types for swarm coordination
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentSpecialization {
    ThroughputOptimizer,
    LatencyMinimizer,
    EnergyEfficiencyExpert,
    InterferenceAnalyst,
    GeneralPurpose,
}

impl AgentSpecialization {
    pub fn get_fitness_weights(&self) -> (f32, f32, f32, f32) {
        match self {
            AgentSpecialization::ThroughputOptimizer => (0.7, 0.1, 0.1, 0.1),
            AgentSpecialization::LatencyMinimizer => (0.1, 0.7, 0.1, 0.1),
            AgentSpecialization::EnergyEfficiencyExpert => (0.1, 0.1, 0.7, 0.1),
            AgentSpecialization::InterferenceAnalyst => (0.1, 0.1, 0.1, 0.7),
            AgentSpecialization::GeneralPurpose => (0.25, 0.25, 0.25, 0.25),
        }
    }
}

/// Agent performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformance {
    pub agent_id: String,
    pub specialization: String,
    pub personal_best_fitness: f32,
    pub iterations_without_improvement: u32,
    pub total_evaluations: u32,
    pub success_rate: f32,
}

impl AgentPerformance {
    pub fn new(agent_id: String, specialization: String) -> Self {
        Self {
            agent_id,
            specialization,
            personal_best_fitness: f32::NEG_INFINITY,
            iterations_without_improvement: 0,
            total_evaluations: 0,
            success_rate: 0.0,
        }
    }
}

/// Optimization summary for results tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSummary {
    pub timestamp: DateTime<Utc>,
    pub total_iterations: u32,
    pub convergence_history: Vec<f32>,
    pub best_fitness: f32,
    pub best_configuration: RANConfiguration,
    pub best_metrics: RANMetrics,
    pub agent_performances: Vec<AgentPerformance>,
    pub execution_time_seconds: f64,
}

impl OptimizationSummary {
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            total_iterations: 0,
            convergence_history: Vec::new(),
            best_fitness: f32::NEG_INFINITY,
            best_configuration: RANConfiguration::new(0),
            best_metrics: RANMetrics::new(),
            agent_performances: Vec::new(),
            execution_time_seconds: 0.0,
        }
    }
}

/// Network topology representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    pub nodes: HashMap<u32, NetworkNode>,
    pub edges: Vec<NetworkEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkNode {
    pub id: u32,
    pub position: (f32, f32),
    pub node_type: String,
    pub capacity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEdge {
    pub source: u32,
    pub target: u32,
    pub weight: f32,
    pub edge_type: String,
}

impl NetworkTopology {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }
    
    pub fn add_node(&mut self, node: NetworkNode) {
        self.nodes.insert(node.id, node);
    }
    
    pub fn add_edge(&mut self, edge: NetworkEdge) {
        self.edges.push(edge);
    }
}