//! Multi-Objective Fitness Evaluation for Network Optimization
//! 
//! This module implements comprehensive fitness functions for network KPIs
//! including throughput, latency, energy efficiency, and other performance metrics.

use crate::models::{RANConfiguration, RANMetrics, AgentSpecialization};
use crate::swarm::pso::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive network fitness evaluator
pub struct NetworkFitnessEvaluator {
    pub kpi_weights: HashMap<String, f32>,
    pub constraint_penalties: HashMap<String, f32>,
    pub historical_performance: Vec<NetworkFitnessScores>,
    pub traffic_model: TrafficModel,
    pub interference_model: InterferenceModel,
}

/// Traffic model for realistic network simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficModel {
    pub base_load: f32,
    pub peak_hours: Vec<u8>,
    pub traffic_patterns: HashMap<TrafficPattern, f32>,
    pub user_density: f32,
}

/// Interference model for network optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceModel {
    pub co_channel_interference: f32,
    pub adjacent_channel_interference: f32,
    pub external_interference: f32,
    pub thermal_noise: f32,
}

impl NetworkFitnessEvaluator {
    pub fn new() -> Self {
        let mut kpi_weights = HashMap::new();
        kpi_weights.insert("throughput".to_string(), 0.25);
        kpi_weights.insert("latency".to_string(), 0.25);
        kpi_weights.insert("energy_efficiency".to_string(), 0.20);
        kpi_weights.insert("interference".to_string(), 0.15);
        kpi_weights.insert("handover_success".to_string(), 0.10);
        kpi_weights.insert("endc_establishment".to_string(), 0.05);
        
        Self {
            kpi_weights,
            constraint_penalties: HashMap::new(),
            historical_performance: Vec::new(),
            traffic_model: TrafficModel::default(),
            interference_model: InterferenceModel::default(),
        }
    }
    
    /// Evaluate multi-objective fitness for a configuration
    pub fn evaluate_fitness(
        &mut self,
        config: &RANConfiguration,
        network_conditions: &NetworkConditions,
        specialization: &AgentSpecialization,
    ) -> NetworkFitnessScores {
        let mut scores = NetworkFitnessScores::default();
        
        // Calculate individual KPI scores
        scores.throughput = self.calculate_throughput_score(config, network_conditions);
        scores.latency = self.calculate_latency_score(config, network_conditions);
        scores.energy_efficiency = self.calculate_energy_efficiency_score(config, network_conditions);
        scores.interference_level = self.calculate_interference_score(config, network_conditions);
        scores.handover_success_rate = self.calculate_handover_success_score(config, network_conditions);
        scores.endc_establishment_success = self.calculate_endc_establishment_score(config, network_conditions);
        scores.user_satisfaction = self.calculate_user_satisfaction_score(config, network_conditions);
        scores.load_balancing_score = self.calculate_load_balancing_score(config, network_conditions);
        
        // Calculate weighted composite score based on specialization
        scores.weighted_composite = self.calculate_weighted_composite(&scores, specialization);
        
        // Store historical performance
        self.historical_performance.push(scores.clone());
        if self.historical_performance.len() > 1000 {
            self.historical_performance.remove(0);
        }
        
        scores
    }
    
    /// Calculate throughput score based on configuration and conditions
    fn calculate_throughput_score(&self, config: &RANConfiguration, conditions: &NetworkConditions) -> f32 {
        let base_throughput = match config.bandwidth {
            20.0 => 100.0,
            40.0 => 200.0,
            80.0 => 400.0,
            _ => 50.0,
        };
        
        // MIMO gain
        let mimo_gain = match config.mimo_config.as_str() {
            "2x2" => 1.5,
            "4x4" => 2.8,
            "8x8" => 4.5,
            _ => 1.0,
        };
        
        // Modulation efficiency
        let modulation_efficiency = match config.modulation_scheme.as_str() {
            "QPSK" => 0.7,
            "16QAM" => 1.0,
            "64QAM" => 1.4,
            "256QAM" => 1.8,
            _ => 0.5,
        };
        
        // Traffic pattern impact
        let traffic_impact = match conditions.traffic_pattern {
            TrafficPattern::VoIP => 0.8,
            TrafficPattern::Video => 1.2,
            TrafficPattern::DataTransfer => 1.5,
            TrafficPattern::Gaming => 1.0,
            TrafficPattern::IoT => 0.6,
            TrafficPattern::Mixed => 1.0,
        };
        
        // Load factor impact
        let load_impact = 1.0 - (conditions.load_factor * 0.5);
        
        // Beamforming gain
        let beamforming_gain = if config.beamforming_enabled { 1.3 } else { 1.0 };
        
        let calculated_throughput = base_throughput * mimo_gain * modulation_efficiency 
            * traffic_impact * load_impact * beamforming_gain;
        
        // Normalize to 0-1 range
        (calculated_throughput / 1000.0).min(1.0)
    }
    
    /// Calculate latency score (lower is better, so we invert)
    fn calculate_latency_score(&self, config: &RANConfiguration, conditions: &NetworkConditions) -> f32 {
        let base_latency = match config.frequency_band {
            freq if freq > 3000.0 => 2.0, // mmWave
            freq if freq > 2400.0 => 5.0, // Sub-6 GHz
            _ => 10.0, // Legacy bands
        };
        
        // Processing delay based on modulation complexity
        let processing_delay = match config.modulation_scheme.as_str() {
            "QPSK" => 1.0,
            "16QAM" => 1.5,
            "64QAM" => 2.0,
            "256QAM" => 3.0,
            _ => 1.0,
        };
        
        // Load impact on latency
        let load_impact = 1.0 + (conditions.load_factor * 2.0);
        
        // Mobility impact
        let mobility_impact = 1.0 + (conditions.mobility_factor * 0.5);
        
        let calculated_latency = base_latency * processing_delay * load_impact * mobility_impact;
        
        // Invert and normalize (lower latency = higher score)
        (50.0 - calculated_latency).max(0.0) / 50.0
    }
    
    /// Calculate energy efficiency score
    fn calculate_energy_efficiency_score(&self, config: &RANConfiguration, conditions: &NetworkConditions) -> f32 {
        let base_power = config.power_level;
        let throughput_power_ratio = self.calculate_throughput_score(config, conditions) / (base_power / 100.0);
        
        // MIMO power consumption
        let mimo_power_factor = match config.mimo_config.as_str() {
            "2x2" => 1.2,
            "4x4" => 1.8,
            "8x8" => 2.5,
            _ => 1.0,
        };
        
        // Beamforming power consumption
        let beamforming_power_factor = if config.beamforming_enabled { 1.3 } else { 1.0 };
        
        let efficiency = throughput_power_ratio / (mimo_power_factor * beamforming_power_factor);
        
        efficiency.min(1.0)
    }
    
    /// Calculate interference score (lower interference = higher score)
    fn calculate_interference_score(&self, config: &RANConfiguration, conditions: &NetworkConditions) -> f32 {
        let base_interference = conditions.interference_level;
        
        // Power level contribution to interference
        let power_interference = config.power_level / 100.0;
        
        // Frequency band interference characteristics
        let freq_interference = match config.frequency_band {
            freq if freq > 3000.0 => 0.2, // mmWave has less interference
            freq if freq > 2400.0 => 0.5, // Sub-6 GHz moderate interference
            _ => 0.8, // Legacy bands more interference
        };
        
        // Antenna tilt impact on interference
        let tilt_factor = 1.0 - (config.antenna_tilt.abs() / 20.0);
        
        let total_interference = base_interference + power_interference * freq_interference * tilt_factor;
        
        // Invert and normalize (lower interference = higher score)
        (1.0 - total_interference).max(0.0)
    }
    
    /// Calculate handover success rate
    fn calculate_handover_success_score(&self, config: &RANConfiguration, conditions: &NetworkConditions) -> f32 {
        let base_success_rate = 0.95;
        
        // Mobility impact
        let mobility_penalty = conditions.mobility_factor * 0.1;
        
        // Load impact
        let load_penalty = conditions.load_factor * 0.05;
        
        // Power level impact (too high or too low power affects handover)
        let power_penalty = ((config.power_level - 20.0).abs() / 20.0) * 0.05;
        
        let success_rate = base_success_rate - mobility_penalty - load_penalty - power_penalty;
        
        success_rate.max(0.0).min(1.0)
    }
    
    /// Calculate ENDC establishment success rate
    fn calculate_endc_establishment_score(&self, config: &RANConfiguration, conditions: &NetworkConditions) -> f32 {
        let base_success_rate = 0.92;
        
        // Frequency band impact (higher frequency = better ENDC)
        let freq_bonus = if config.frequency_band > 3000.0 { 0.05 } else { 0.0 };
        
        // MIMO configuration impact
        let mimo_bonus = match config.mimo_config.as_str() {
            "4x4" => 0.03,
            "8x8" => 0.05,
            _ => 0.0,
        };
        
        // Load impact
        let load_penalty = conditions.load_factor * 0.08;
        
        let success_rate = base_success_rate + freq_bonus + mimo_bonus - load_penalty;
        
        success_rate.max(0.0).min(1.0)
    }
    
    /// Calculate user satisfaction score
    fn calculate_user_satisfaction_score(&self, config: &RANConfiguration, conditions: &NetworkConditions) -> f32 {
        let throughput_score = self.calculate_throughput_score(config, conditions);
        let latency_score = self.calculate_latency_score(config, conditions);
        let interference_score = self.calculate_interference_score(config, conditions);
        
        // Weighted combination for user satisfaction
        let satisfaction = throughput_score * 0.4 + latency_score * 0.4 + interference_score * 0.2;
        
        satisfaction.min(1.0)
    }
    
    /// Calculate load balancing score
    fn calculate_load_balancing_score(&self, config: &RANConfiguration, conditions: &NetworkConditions) -> f32 {
        let optimal_load = 0.7;
        let load_deviation = (conditions.load_factor - optimal_load).abs();
        
        // Power level affects load balancing ability
        let power_factor = if config.power_level > 25.0 { 0.8 } else { 1.0 };
        
        let balancing_score = (1.0 - load_deviation * 2.0) * power_factor;
        
        balancing_score.max(0.0).min(1.0)
    }
    
    /// Calculate weighted composite score based on agent specialization
    fn calculate_weighted_composite(&self, scores: &NetworkFitnessScores, specialization: &AgentSpecialization) -> f32 {
        let weights = specialization.get_fitness_weights();
        
        scores.throughput * weights.0 +
        scores.latency * weights.1 +
        scores.energy_efficiency * weights.2 +
        scores.interference_level * weights.3
    }
    
    /// Calculate constraint violations
    pub fn calculate_constraint_violations(&self, config: &RANConfiguration, constraints: &NetworkConstraints) -> f32 {
        let mut violations = 0.0;
        
        // Power consumption constraint
        if config.power_level > constraints.max_power_consumption {
            violations += (config.power_level - constraints.max_power_consumption) * 0.1;
        }
        
        // Add more constraint checks as needed
        
        violations
    }
    
    /// Update KPI weights dynamically
    pub fn update_kpi_weights(&mut self, new_weights: HashMap<String, f32>) {
        self.kpi_weights = new_weights;
    }
    
    /// Get performance trends
    pub fn get_performance_trends(&self) -> Vec<f32> {
        self.historical_performance
            .iter()
            .map(|score| score.weighted_composite)
            .collect()
    }
}

impl Default for TrafficModel {
    fn default() -> Self {
        Self {
            base_load: 0.5,
            peak_hours: vec![8, 9, 17, 18, 19, 20],
            traffic_patterns: HashMap::new(),
            user_density: 100.0,
        }
    }
}

impl Default for InterferenceModel {
    fn default() -> Self {
        Self {
            co_channel_interference: 0.1,
            adjacent_channel_interference: 0.05,
            external_interference: 0.02,
            thermal_noise: -104.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::RANConfiguration;
    
    #[test]
    fn test_throughput_calculation() {
        let evaluator = NetworkFitnessEvaluator::new();
        let config = RANConfiguration {
            cell_id: 1,
            frequency_band: 2600.0,
            power_level: 20.0,
            antenna_tilt: 0.0,
            bandwidth: 80.0,
            modulation_scheme: "64QAM".to_string(),
            mimo_config: "4x4".to_string(),
            beamforming_enabled: true,
        };
        let conditions = NetworkConditions::default();
        
        let score = evaluator.calculate_throughput_score(&config, &conditions);
        assert!(score > 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_latency_calculation() {
        let evaluator = NetworkFitnessEvaluator::new();
        let config = RANConfiguration {
            cell_id: 1,
            frequency_band: 3500.0, // mmWave
            power_level: 20.0,
            antenna_tilt: 0.0,
            bandwidth: 80.0,
            modulation_scheme: "QPSK".to_string(),
            mimo_config: "2x2".to_string(),
            beamforming_enabled: false,
        };
        let conditions = NetworkConditions::default();
        
        let score = evaluator.calculate_latency_score(&config, &conditions);
        assert!(score > 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_energy_efficiency_calculation() {
        let evaluator = NetworkFitnessEvaluator::new();
        let config = RANConfiguration {
            cell_id: 1,
            frequency_band: 2600.0,
            power_level: 15.0, // Lower power for efficiency
            antenna_tilt: 0.0,
            bandwidth: 40.0,
            modulation_scheme: "16QAM".to_string(),
            mimo_config: "2x2".to_string(),
            beamforming_enabled: false,
        };
        let conditions = NetworkConditions::default();
        
        let score = evaluator.calculate_energy_efficiency_score(&config, &conditions);
        assert!(score > 0.0 && score <= 1.0);
    }
}