//! Optimization Configuration Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub population_size: usize,
    pub max_iterations: u32,
    pub convergence_threshold: f32,
    pub inertia_weight: f32,
    pub cognitive_weight: f32,
    pub social_weight: f32,
    pub elite_size: usize,
    pub mutation_rate: f32,
    pub diversification_threshold: f32,
    pub adaptive_parameters: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            max_iterations: 100,
            convergence_threshold: 0.001,
            inertia_weight: 0.7,
            cognitive_weight: 1.5,
            social_weight: 1.5,
            elite_size: 5,
            mutation_rate: 0.01,
            diversification_threshold: 0.1,
            adaptive_parameters: true,
        }
    }
}

impl OptimizationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.population_size < 2 {
            return Err("population_size must be at least 2".to_string());
        }
        
        if self.max_iterations == 0 {
            return Err("max_iterations must be greater than 0".to_string());
        }
        
        if self.convergence_threshold <= 0.0 {
            return Err("convergence_threshold must be positive".to_string());
        }
        
        if self.elite_size >= self.population_size {
            return Err("elite_size must be less than population_size".to_string());
        }
        
        if !(0.0..=1.0).contains(&self.mutation_rate) {
            return Err("mutation_rate must be between 0.0 and 1.0".to_string());
        }
        
        Ok(())
    }
    
    pub fn development() -> Self {
        Self {
            population_size: 10,
            max_iterations: 50,
            convergence_threshold: 0.01,
            inertia_weight: 0.5,
            cognitive_weight: 1.0,
            social_weight: 1.0,
            elite_size: 2,
            mutation_rate: 0.05,
            diversification_threshold: 0.2,
            adaptive_parameters: true,
        }
    }
    
    pub fn production() -> Self {
        Self {
            population_size: 50,
            max_iterations: 500,
            convergence_threshold: 0.0001,
            inertia_weight: 0.9,
            cognitive_weight: 2.0,
            social_weight: 2.0,
            elite_size: 10,
            mutation_rate: 0.005,
            diversification_threshold: 0.05,
            adaptive_parameters: true,
        }
    }
}