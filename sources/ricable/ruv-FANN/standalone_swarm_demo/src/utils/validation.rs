//! Validation Utilities
//! 
//! This module provides validation functions for configurations and data.

use crate::models::{RANConfiguration, RANMetrics};
use crate::config::SwarmConfig;

pub struct Validator;

impl Validator {
    pub fn validate_ran_configuration(config: &RANConfiguration) -> Result<(), String> {
        if config.power_level < 5.0 || config.power_level > 40.0 {
            return Err("Power level must be between 5.0 and 40.0 dBm".to_string());
        }
        
        if config.antenna_tilt < -15.0 || config.antenna_tilt > 15.0 {
            return Err("Antenna tilt must be between -15.0 and 15.0 degrees".to_string());
        }
        
        if ![20.0, 40.0, 80.0].contains(&config.bandwidth) {
            return Err("Bandwidth must be 20, 40, or 80 MHz".to_string());
        }
        
        if config.frequency_band < 2400.0 || config.frequency_band > 3800.0 {
            return Err("Frequency band must be between 2400 and 3800 MHz".to_string());
        }
        
        Ok(())
    }
    
    pub fn validate_ran_metrics(metrics: &RANMetrics) -> Result<(), String> {
        if metrics.throughput < 0.0 {
            return Err("Throughput cannot be negative".to_string());
        }
        
        if metrics.latency < 0.0 {
            return Err("Latency cannot be negative".to_string());
        }
        
        if !(0.0..=1.0).contains(&metrics.energy_efficiency) {
            return Err("Energy efficiency must be between 0.0 and 1.0".to_string());
        }
        
        if !(0.0..=1.0).contains(&metrics.interference_level) {
            return Err("Interference level must be between 0.0 and 1.0".to_string());
        }
        
        Ok(())
    }
    
    pub fn validate_swarm_config(config: &SwarmConfig) -> Result<(), String> {
        config.validate()
    }
}