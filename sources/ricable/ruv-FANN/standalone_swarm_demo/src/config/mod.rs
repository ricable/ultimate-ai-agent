//! Configuration Management Module
//! 
//! This module handles all configuration aspects of the neural swarm system,
//! including system parameters, optimization settings, and runtime configuration.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub mod optimization;
pub mod system;
pub mod neural;

pub use optimization::OptimizationConfig;
pub use system::SystemConfig;
pub use neural::NeuralConfig;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub system: SystemConfig,
    pub optimization: OptimizationConfig,
    pub neural: NeuralConfig,
    pub logging: LoggingConfig,
    pub performance: PerformanceConfig,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            optimization: OptimizationConfig::default(),
            neural: NeuralConfig::default(),
            logging: LoggingConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl SwarmConfig {
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: SwarmConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
    
    /// Load configuration from environment variables
    pub fn load_from_env() -> Self {
        let mut config = Self::default();
        
        // System configuration from environment
        if let Ok(threads) = std::env::var("SWARM_THREADS") {
            if let Ok(thread_count) = threads.parse::<usize>() {
                config.system.max_threads = thread_count;
            }
        }
        
        if let Ok(memory) = std::env::var("SWARM_MEMORY_LIMIT") {
            if let Ok(memory_mb) = memory.parse::<usize>() {
                config.system.memory_limit_mb = memory_mb;
            }
        }
        
        // Optimization configuration from environment
        if let Ok(pop_size) = std::env::var("SWARM_POPULATION_SIZE") {
            if let Ok(size) = pop_size.parse::<usize>() {
                config.optimization.population_size = size;
            }
        }
        
        if let Ok(max_iter) = std::env::var("SWARM_MAX_ITERATIONS") {
            if let Ok(iterations) = max_iter.parse::<u32>() {
                config.optimization.max_iterations = iterations;
            }
        }
        
        // Neural configuration from environment
        if let Ok(learning_rate) = std::env::var("SWARM_LEARNING_RATE") {
            if let Ok(rate) = learning_rate.parse::<f32>() {
                config.neural.learning_rate = rate;
            }
        }
        
        // Logging configuration from environment
        if let Ok(log_level) = std::env::var("SWARM_LOG_LEVEL") {
            config.logging.level = log_level;
        }
        
        config
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        self.system.validate()?;
        self.optimization.validate()?;
        self.neural.validate()?;
        self.logging.validate()?;
        self.performance.validate()?;
        Ok(())
    }
    
    /// Create a configuration for development/testing
    pub fn development() -> Self {
        Self {
            system: SystemConfig::development(),
            optimization: OptimizationConfig::development(),
            neural: NeuralConfig::development(),
            logging: LoggingConfig::development(),
            performance: PerformanceConfig::development(),
        }
    }
    
    /// Create a configuration for production
    pub fn production() -> Self {
        Self {
            system: SystemConfig::production(),
            optimization: OptimizationConfig::production(),
            neural: NeuralConfig::production(),
            logging: LoggingConfig::production(),
            performance: PerformanceConfig::production(),
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file_path: Option<String>,
    pub console_output: bool,
    pub structured_logging: bool,
    pub max_file_size_mb: usize,
    pub max_files: usize,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file_path: Some("swarm.log".to_string()),
            console_output: true,
            structured_logging: false,
            max_file_size_mb: 10,
            max_files: 5,
        }
    }
}

impl LoggingConfig {
    pub fn validate(&self) -> Result<(), String> {
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.level.as_str()) {
            return Err(format!("Invalid log level: {}", self.level));
        }
        
        if self.max_file_size_mb == 0 {
            return Err("max_file_size_mb must be greater than 0".to_string());
        }
        
        if self.max_files == 0 {
            return Err("max_files must be greater than 0".to_string());
        }
        
        Ok(())
    }
    
    pub fn development() -> Self {
        Self {
            level: "debug".to_string(),
            file_path: None,
            console_output: true,
            structured_logging: false,
            max_file_size_mb: 5,
            max_files: 3,
        }
    }
    
    pub fn production() -> Self {
        Self {
            level: "info".to_string(),
            file_path: Some("swarm.log".to_string()),
            console_output: false,
            structured_logging: true,
            max_file_size_mb: 50,
            max_files: 10,
        }
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_profiling: bool,
    pub profile_interval_seconds: u64,
    pub memory_monitoring: bool,
    pub cpu_monitoring: bool,
    pub disk_monitoring: bool,
    pub network_monitoring: bool,
    pub metrics_retention_hours: u64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_profiling: false,
            profile_interval_seconds: 60,
            memory_monitoring: true,
            cpu_monitoring: true,
            disk_monitoring: false,
            network_monitoring: false,
            metrics_retention_hours: 24,
        }
    }
}

impl PerformanceConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.profile_interval_seconds == 0 {
            return Err("profile_interval_seconds must be greater than 0".to_string());
        }
        
        if self.metrics_retention_hours == 0 {
            return Err("metrics_retention_hours must be greater than 0".to_string());
        }
        
        Ok(())
    }
    
    pub fn development() -> Self {
        Self {
            enable_profiling: true,
            profile_interval_seconds: 10,
            memory_monitoring: true,
            cpu_monitoring: true,
            disk_monitoring: false,
            network_monitoring: false,
            metrics_retention_hours: 1,
        }
    }
    
    pub fn production() -> Self {
        Self {
            enable_profiling: false,
            profile_interval_seconds: 300,
            memory_monitoring: true,
            cpu_monitoring: true,
            disk_monitoring: true,
            network_monitoring: true,
            metrics_retention_hours: 72,
        }
    }
}

/// Configuration manager for runtime configuration updates
pub struct ConfigManager {
    config: SwarmConfig,
    watchers: Vec<Box<dyn Fn(&SwarmConfig) + Send + Sync>>,
}

impl ConfigManager {
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            config,
            watchers: Vec::new(),
        }
    }
    
    pub fn get_config(&self) -> &SwarmConfig {
        &self.config
    }
    
    pub fn update_config(&mut self, new_config: SwarmConfig) -> Result<(), String> {
        new_config.validate()?;
        self.config = new_config;
        
        // Notify watchers
        for watcher in &self.watchers {
            watcher(&self.config);
        }
        
        Ok(())
    }
    
    pub fn add_watcher<F>(&mut self, watcher: F) 
    where 
        F: Fn(&SwarmConfig) + Send + Sync + 'static,
    {
        self.watchers.push(Box::new(watcher));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[test]
    fn test_default_config() {
        let config = SwarmConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = SwarmConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SwarmConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.system.max_threads, deserialized.system.max_threads);
        assert_eq!(config.optimization.population_size, deserialized.optimization.population_size);
    }
    
    #[test]
    fn test_environment_config() {
        env::set_var("SWARM_THREADS", "8");
        env::set_var("SWARM_POPULATION_SIZE", "50");
        env::set_var("SWARM_LOG_LEVEL", "debug");
        
        let config = SwarmConfig::load_from_env();
        
        assert_eq!(config.system.max_threads, 8);
        assert_eq!(config.optimization.population_size, 50);
        assert_eq!(config.logging.level, "debug");
        
        // Clean up
        env::remove_var("SWARM_THREADS");
        env::remove_var("SWARM_POPULATION_SIZE");
        env::remove_var("SWARM_LOG_LEVEL");
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = SwarmConfig::default();
        
        // Invalid log level
        config.logging.level = "invalid".to_string();
        assert!(config.validate().is_err());
        
        // Fix it
        config.logging.level = "info".to_string();
        assert!(config.validate().is_ok());
    }
}