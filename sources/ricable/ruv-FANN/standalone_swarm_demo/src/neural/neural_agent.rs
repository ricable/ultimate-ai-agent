//! Neural Agent Implementation
//! 
//! This module defines neural agents that can learn and adapt their behavior
//! for RAN optimization tasks.

use crate::models::{RANMetrics, RANConfiguration, AgentSpecialization};
use crate::neural::{NeuralNetwork, NeuralNetworkConfig};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct NeuralAgent {
    pub id: String,
    pub specialization: AgentSpecialization,
    pub neural_network: NeuralNetwork,
    pub experience_buffer: VecDeque<Experience>,
    pub performance_history: Vec<f32>,
    pub learning_rate: f32,
    pub exploration_rate: f32,
    pub max_experience_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub timestamp: u64,
}

impl NeuralAgent {
    pub fn new(id: String, specialization: AgentSpecialization) -> Self {
        let config = Self::get_network_config(&specialization);
        let neural_network = NeuralNetwork::new(config);
        
        Self {
            id,
            specialization,
            neural_network,
            experience_buffer: VecDeque::new(),
            performance_history: Vec::new(),
            learning_rate: 0.01,
            exploration_rate: 0.1,
            max_experience_size: 1000,
        }
    }
    
    fn get_network_config(specialization: &AgentSpecialization) -> NeuralNetworkConfig {
        match specialization {
            AgentSpecialization::ThroughputOptimizer => NeuralNetworkConfig {
                input_size: 8,
                hidden_layers: vec![16, 12, 8],
                output_size: 4, // Configuration adjustments
                learning_rate: 0.01,
                activation_function: "relu".to_string(),
            },
            AgentSpecialization::LatencyMinimizer => NeuralNetworkConfig {
                input_size: 8,
                hidden_layers: vec![12, 8],
                output_size: 4,
                learning_rate: 0.005,
                activation_function: "sigmoid".to_string(),
            },
            AgentSpecialization::EnergyEfficiencyExpert => NeuralNetworkConfig {
                input_size: 8,
                hidden_layers: vec![10, 6],
                output_size: 4,
                learning_rate: 0.002,
                activation_function: "tanh".to_string(),
            },
            AgentSpecialization::InterferenceAnalyst => NeuralNetworkConfig {
                input_size: 8,
                hidden_layers: vec![20, 16, 8],
                output_size: 4,
                learning_rate: 0.01,
                activation_function: "relu".to_string(),
            },
            AgentSpecialization::GeneralPurpose => NeuralNetworkConfig::default(),
        }
    }
    
    pub fn predict_fitness(&self, metrics: &RANMetrics) -> Result<f32, String> {
        let state = self.metrics_to_state(metrics);
        let output = self.neural_network.forward(&state)?;
        
        if output.is_empty() {
            return Err("Neural network produced no output".to_string());
        }
        
        Ok(output[0])
    }
    
    pub fn evaluate_fitness(&self, metrics: &RANMetrics) -> f32 {
        let weights = self.specialization.get_fitness_weights();
        
        let throughput_score = metrics.throughput * weights.0;
        let latency_score = (50.0 - metrics.latency).max(0.0) * weights.1;
        let efficiency_score = metrics.energy_efficiency * weights.2;
        let interference_score = (1.0 - metrics.interference_level) * weights.3;
        
        throughput_score + latency_score + efficiency_score + interference_score
    }
    
    pub fn suggest_configuration(&self, current_config: &RANConfiguration) -> Result<RANConfiguration, String> {
        let current_state = self.config_to_state(current_config);
        let action = self.neural_network.forward(&current_state)?;
        
        if action.len() < 4 {
            return Err("Neural network output insufficient for configuration".to_string());
        }
        
        let mut new_config = current_config.clone();
        
        // Apply neural network suggestions with constraints
        new_config.power_level = (current_config.power_level + action[0] * 5.0).clamp(5.0, 40.0);
        new_config.antenna_tilt = (current_config.antenna_tilt + action[1] * 2.0).clamp(-10.0, 10.0);
        new_config.bandwidth = match action[2] {
            x if x > 0.5 => 80.0,
            x if x > 0.0 => 40.0,
            _ => 20.0,
        };
        
        // Update modulation scheme based on neural network output
        new_config.modulation_scheme = match action[3] {
            x if x > 0.66 => "256QAM".to_string(),
            x if x > 0.33 => "64QAM".to_string(),
            _ => "QPSK".to_string(),
        };
        
        Ok(new_config)
    }
    
    pub fn learn_from_experience(&mut self, experience: Experience) -> Result<(), String> {
        self.experience_buffer.push_back(experience);
        
        // Keep buffer size manageable
        if self.experience_buffer.len() > self.max_experience_size {
            self.experience_buffer.pop_front();
        }
        
        // Trigger learning if we have enough experiences
        if self.experience_buffer.len() >= 32 {
            self.update_neural_network()?;
        }
        
        Ok(())
    }
    
    fn update_neural_network(&mut self) -> Result<(), String> {
        if self.experience_buffer.is_empty() {
            return Err("No experience data available for learning".to_string());
        }
        
        // Create training data from experience buffer
        let mut training_data = Vec::new();
        
        for experience in self.experience_buffer.iter().rev().take(100) {
            let input = experience.state.clone();
            let target = vec![experience.reward]; // Simplified target
            training_data.push((input, target));
        }
        
        self.neural_network.train(&training_data)?;
        
        // Update exploration rate (decay over time)
        self.exploration_rate = (self.exploration_rate * 0.995).max(0.01);
        
        Ok(())
    }
    
    pub fn add_performance_score(&mut self, score: f32) {
        self.performance_history.push(score);
        
        // Keep only recent performance history
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
    }
    
    pub fn get_average_performance(&self) -> f32 {
        if self.performance_history.is_empty() {
            return 0.0;
        }
        
        self.performance_history.iter().sum::<f32>() / self.performance_history.len() as f32
    }
    
    pub fn get_recent_performance(&self, window: usize) -> f32 {
        if self.performance_history.is_empty() {
            return 0.0;
        }
        
        let start = self.performance_history.len().saturating_sub(window);
        let recent_scores = &self.performance_history[start..];
        
        recent_scores.iter().sum::<f32>() / recent_scores.len() as f32
    }
    
    pub fn should_explore(&self) -> bool {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen::<f32>() < self.exploration_rate
    }
    
    fn metrics_to_state(&self, metrics: &RANMetrics) -> Vec<f32> {
        vec![
            metrics.throughput / 100.0,
            metrics.latency / 50.0,
            metrics.energy_efficiency,
            metrics.interference_level,
            // Add normalized current time, load, etc.
            0.5, // Placeholder for time of day
            0.5, // Placeholder for network load
            0.5, // Placeholder for user density
            0.5, // Placeholder for weather conditions
        ]
    }
    
    fn config_to_state(&self, config: &RANConfiguration) -> Vec<f32> {
        vec![
            config.power_level / 40.0,
            (config.antenna_tilt + 10.0) / 20.0,
            config.bandwidth / 80.0,
            config.frequency_band / 5000.0,
            if config.beamforming_enabled { 1.0 } else { 0.0 },
            // Add more normalized configuration parameters
            0.5, // Placeholder
            0.5, // Placeholder
            0.5, // Placeholder
        ]
    }
    
    pub fn clone_with_mutation(&self, mutation_rate: f32) -> Self {
        let mut clone = self.clone();
        clone.id = format!("{}_mutated", self.id);
        
        // Apply mutation to learning parameters
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        if rng.gen::<f32>() < mutation_rate {
            clone.learning_rate *= rng.gen_range(0.8..1.2);
            clone.exploration_rate *= rng.gen_range(0.8..1.2);
        }
        
        clone
    }
}

impl Default for NeuralAgent {
    fn default() -> Self {
        Self::new(
            "default_agent".to_string(),
            AgentSpecialization::GeneralPurpose,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neural_agent_creation() {
        let agent = NeuralAgent::new(
            "test_agent".to_string(),
            AgentSpecialization::ThroughputOptimizer,
        );
        
        assert_eq!(agent.id, "test_agent");
        assert_eq!(agent.specialization, AgentSpecialization::ThroughputOptimizer);
        assert!(agent.experience_buffer.is_empty());
    }
    
    #[test]
    fn test_fitness_evaluation() {
        let agent = NeuralAgent::new(
            "test_agent".to_string(),
            AgentSpecialization::ThroughputOptimizer,
        );
        
        let metrics = RANMetrics {
            throughput: 50.0,
            latency: 10.0,
            energy_efficiency: 0.8,
            interference_level: 0.2,
        };
        
        let fitness = agent.evaluate_fitness(&metrics);
        assert!(fitness > 0.0);
    }
    
    #[test]
    fn test_experience_learning() {
        let mut agent = NeuralAgent::new(
            "test_agent".to_string(),
            AgentSpecialization::GeneralPurpose,
        );
        
        let experience = Experience {
            state: vec![0.5, 0.3, 0.8, 0.1, 0.5, 0.5, 0.5, 0.5],
            action: vec![0.1, 0.2, 0.3, 0.4],
            reward: 0.7,
            next_state: vec![0.6, 0.3, 0.8, 0.1, 0.5, 0.5, 0.5, 0.5],
            timestamp: 1234567890,
        };
        
        assert!(agent.learn_from_experience(experience).is_ok());
        assert_eq!(agent.experience_buffer.len(), 1);
    }
}