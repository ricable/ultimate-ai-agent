//! System Configuration Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub max_threads: usize,
    pub memory_limit_mb: usize,
    pub timeout_seconds: u64,
    pub checkpoint_interval: u64,
    pub enable_gpu: bool,
    pub gpu_device_id: Option<u32>,
    pub temp_directory: String,
    pub data_directory: String,
    pub backup_enabled: bool,
    pub backup_interval_hours: u64,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            max_threads: num_cpus::get(),
            memory_limit_mb: 1024,
            timeout_seconds: 300,
            checkpoint_interval: 60,
            enable_gpu: false,
            gpu_device_id: None,
            temp_directory: "/tmp/swarm".to_string(),
            data_directory: "./swarm_data".to_string(),
            backup_enabled: false,
            backup_interval_hours: 24,
        }
    }
}

impl SystemConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_threads == 0 {
            return Err("max_threads must be greater than 0".to_string());
        }
        
        if self.memory_limit_mb < 64 {
            return Err("memory_limit_mb must be at least 64".to_string());
        }
        
        if self.timeout_seconds == 0 {
            return Err("timeout_seconds must be greater than 0".to_string());
        }
        
        if self.checkpoint_interval == 0 {
            return Err("checkpoint_interval must be greater than 0".to_string());
        }
        
        if self.backup_interval_hours == 0 {
            return Err("backup_interval_hours must be greater than 0".to_string());
        }
        
        Ok(())
    }
    
    pub fn development() -> Self {
        Self {
            max_threads: 2,
            memory_limit_mb: 512,
            timeout_seconds: 60,
            checkpoint_interval: 10,
            enable_gpu: false,
            gpu_device_id: None,
            temp_directory: "./temp".to_string(),
            data_directory: "./dev_data".to_string(),
            backup_enabled: false,
            backup_interval_hours: 1,
        }
    }
    
    pub fn production() -> Self {
        Self {
            max_threads: num_cpus::get(),
            memory_limit_mb: 4096,
            timeout_seconds: 3600,
            checkpoint_interval: 300,
            enable_gpu: true,
            gpu_device_id: Some(0),
            temp_directory: "/tmp/swarm_prod".to_string(),
            data_directory: "/data/swarm".to_string(),
            backup_enabled: true,
            backup_interval_hours: 6,
        }
    }
}