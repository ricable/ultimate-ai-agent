use crate::dtm_traffic::{ForecastResult, PredictorConfig, NetworkLayer, ServiceType};
use std::collections::HashMap;
use chrono::{DateTime, Utc, TimeZone, Local};

/// AMOS (Automated Network Management Operations System) script generator
/// for load balancing based on traffic predictions
pub struct AmosScriptGenerator {
    config: PredictorConfig,
    threshold_config: ThresholdConfig,
    template_config: TemplateConfig,
}

/// Configuration for load balancing thresholds
#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    pub prb_high_threshold: f32,      // 80% PRB utilization
    pub prb_medium_threshold: f32,    // 60% PRB utilization
    pub prb_low_threshold: f32,       // 40% PRB utilization
    pub user_count_threshold: u32,    // User count threshold
    pub throughput_threshold: f32,    // Throughput threshold (Mbps)
    pub qos_degradation_threshold: f32, // QoS degradation threshold
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            prb_high_threshold: 0.8,
            prb_medium_threshold: 0.6,
            prb_low_threshold: 0.4,
            user_count_threshold: 100,
            throughput_threshold: 500.0,
            qos_degradation_threshold: 0.1,
        }
    }
}

/// Configuration for AMOS script templates
#[derive(Debug, Clone)]
pub struct TemplateConfig {
    pub cell_id_prefix: String,
    pub frequency_bands: HashMap<NetworkLayer, String>,
    pub service_priorities: HashMap<ServiceType, u8>,
    pub handover_hysteresis: f32,
    pub handover_time_to_trigger: u32,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        let mut frequency_bands = HashMap::new();
        frequency_bands.insert(NetworkLayer::L2100, "2100".to_string());
        frequency_bands.insert(NetworkLayer::N78, "3500".to_string());
        frequency_bands.insert(NetworkLayer::N258, "26000".to_string());
        
        let mut service_priorities = HashMap::new();
        service_priorities.insert(ServiceType::URLLC, 1);  // Highest priority
        service_priorities.insert(ServiceType::VoNR, 2);
        service_priorities.insert(ServiceType::EMBB, 3);
        service_priorities.insert(ServiceType::MIoT, 4);   // Lowest priority
        
        Self {
            cell_id_prefix: "CELL".to_string(),
            frequency_bands,
            service_priorities,
            handover_hysteresis: 3.0,  // dB
            handover_time_to_trigger: 160,  // ms
        }
    }
}

/// Load balancing action types
#[derive(Debug, Clone)]
pub enum LoadBalancingAction {
    /// Trigger handover to target layer
    TriggerHandover {
        source_layer: NetworkLayer,
        target_layer: NetworkLayer,
        user_percentage: f32,
        service_type: Option<ServiceType>,
    },
    /// Adjust antenna tilt
    AdjustAntennaTilt {
        layer: NetworkLayer,
        tilt_adjustment: f32,  // degrees
    },
    /// Modify transmission power
    AdjustTransmissionPower {
        layer: NetworkLayer,
        power_adjustment: f32,  // dBm
    },
    /// Update QoS parameters
    UpdateQoSParameters {
        layer: NetworkLayer,
        service_type: ServiceType,
        new_priority: u8,
        bandwidth_allocation: f32,
    },
    /// Enable/disable carrier aggregation
    ConfigureCarrierAggregation {
        primary_layer: NetworkLayer,
        secondary_layer: NetworkLayer,
        enable: bool,
    },
    /// Adjust load balancing weights
    AdjustLoadBalancingWeights {
        layer: NetworkLayer,
        weight_factor: f32,
    },
}

/// AMOS script generation result
#[derive(Debug, Clone)]
pub struct AmosScript {
    pub script_id: String,
    pub timestamp: DateTime<Utc>,
    pub actions: Vec<LoadBalancingAction>,
    pub script_content: String,
    pub validation_checks: Vec<String>,
    pub rollback_script: String,
}

impl AmosScriptGenerator {
    pub fn new(config: PredictorConfig) -> Self {
        Self {
            config,
            threshold_config: ThresholdConfig::default(),
            template_config: TemplateConfig::default(),
        }
    }
    
    pub fn with_thresholds(mut self, threshold_config: ThresholdConfig) -> Self {
        self.threshold_config = threshold_config;
        self
    }
    
    pub fn with_templates(mut self, template_config: TemplateConfig) -> Self {
        self.template_config = template_config;
        self
    }
    
    /// Generate AMOS scripts based on traffic predictions
    pub fn generate_scripts(&self, predictions: &ForecastResult) -> Vec<AmosScript> {
        let mut scripts = Vec::new();
        
        // Generate scripts for each forecast horizon
        for (horizon_idx, &timestamp) in predictions.horizons.iter().enumerate() {
            let actions = self.analyze_predictions_and_generate_actions(predictions, horizon_idx);
            
            if !actions.is_empty() {
                let script = self.create_amos_script(timestamp, actions);
                scripts.push(script);
            }
        }
        
        scripts
    }
    
    fn analyze_predictions_and_generate_actions(
        &self,
        predictions: &ForecastResult,
        horizon_idx: usize,
    ) -> Vec<LoadBalancingAction> {
        let mut actions = Vec::new();
        
        // Analyze PRB utilization predictions
        for layer_idx in 0..3 {
            let layer = match layer_idx {
                0 => NetworkLayer::L2100,
                1 => NetworkLayer::N78,
                2 => NetworkLayer::N258,
                _ => continue,
            };
            
            let prb_utilization = predictions.prb_predictions[[horizon_idx, layer_idx]];
            
            // Check if PRB utilization exceeds thresholds
            if prb_utilization > self.threshold_config.prb_high_threshold {
                actions.extend(self.generate_high_load_actions(layer, prb_utilization));
            } else if prb_utilization > self.threshold_config.prb_medium_threshold {
                actions.extend(self.generate_medium_load_actions(layer, prb_utilization));
            } else if prb_utilization < self.threshold_config.prb_low_threshold {
                actions.extend(self.generate_low_load_actions(layer, prb_utilization));
            }
        }
        
        // Analyze service type predictions
        for layer_idx in 0..3 {
            let layer = match layer_idx {
                0 => NetworkLayer::L2100,
                1 => NetworkLayer::N78,
                2 => NetworkLayer::N258,
                _ => continue,
            };
            
            for service_idx in 0..4 {
                let service_type = match service_idx {
                    0 => ServiceType::EMBB,
                    1 => ServiceType::VoNR,
                    2 => ServiceType::URLLC,
                    3 => ServiceType::MIoT,
                    _ => continue,
                };
                
                let service_demand = predictions.service_predictions[[horizon_idx, layer_idx, service_idx]];
                
                if service_demand > 0.7 {  // High demand threshold
                    actions.extend(self.generate_service_specific_actions(layer, service_type, service_demand));
                }
            }
        }
        
        // Analyze QoS predictions and generate QoS-specific actions
        for layer_idx in 0..3 {
            let layer = match layer_idx {
                0 => NetworkLayer::L2100,
                1 => NetworkLayer::N78,
                2 => NetworkLayer::N258,
                _ => continue,
            };
            
            for qos_idx in 0..11 {
                let qos_degradation = predictions.qos_predictions[[horizon_idx, layer_idx, qos_idx]];
                
                if qos_degradation > self.threshold_config.qos_degradation_threshold {
                    actions.extend(self.generate_qos_recovery_actions(layer, qos_idx, qos_degradation));
                }
            }
        }
        
        actions
    }
    
    fn generate_high_load_actions(&self, layer: NetworkLayer, prb_utilization: f32) -> Vec<LoadBalancingAction> {
        let mut actions = Vec::new();
        
        match layer {
            NetworkLayer::L2100 => {
                // Offload traffic to 5G layers
                actions.push(LoadBalancingAction::TriggerHandover {
                    source_layer: NetworkLayer::L2100,
                    target_layer: NetworkLayer::N78,
                    user_percentage: 0.3,  // Move 30% of users
                    service_type: Some(ServiceType::EMBB),
                });
                
                // Adjust antenna tilt to reduce coverage overlap
                actions.push(LoadBalancingAction::AdjustAntennaTilt {
                    layer: NetworkLayer::L2100,
                    tilt_adjustment: 2.0,  // Increase tilt by 2 degrees
                });
            }
            
            NetworkLayer::N78 => {
                // Enable carrier aggregation with mmWave
                actions.push(LoadBalancingAction::ConfigureCarrierAggregation {
                    primary_layer: NetworkLayer::N78,
                    secondary_layer: NetworkLayer::N258,
                    enable: true,
                });
                
                // Adjust load balancing weights
                actions.push(LoadBalancingAction::AdjustLoadBalancingWeights {
                    layer: NetworkLayer::N78,
                    weight_factor: 0.7,  // Reduce weight to discourage new connections
                });
            }
            
            NetworkLayer::N258 => {
                // Increase transmission power for better coverage
                actions.push(LoadBalancingAction::AdjustTransmissionPower {
                    layer: NetworkLayer::N258,
                    power_adjustment: 3.0,  // Increase by 3 dBm
                });
            }
        }
        
        actions
    }
    
    fn generate_medium_load_actions(&self, layer: NetworkLayer, prb_utilization: f32) -> Vec<LoadBalancingAction> {
        let mut actions = Vec::new();
        
        // Proactive load balancing
        actions.push(LoadBalancingAction::AdjustLoadBalancingWeights {
            layer,
            weight_factor: 0.8,  // Slightly reduce weight
        });
        
        // Prepare for potential handovers
        if layer == NetworkLayer::L2100 {
            actions.push(LoadBalancingAction::TriggerHandover {
                source_layer: NetworkLayer::L2100,
                target_layer: NetworkLayer::N78,
                user_percentage: 0.1,  // Move 10% of users
                service_type: Some(ServiceType::EMBB),
            });
        }
        
        actions
    }
    
    fn generate_low_load_actions(&self, layer: NetworkLayer, prb_utilization: f32) -> Vec<LoadBalancingAction> {
        let mut actions = Vec::new();
        
        // Increase load balancing weight to attract more users
        actions.push(LoadBalancingAction::AdjustLoadBalancingWeights {
            layer,
            weight_factor: 1.2,
        });
        
        // Optimize power consumption
        actions.push(LoadBalancingAction::AdjustTransmissionPower {
            layer,
            power_adjustment: -2.0,  // Reduce by 2 dBm
        });
        
        actions
    }
    
    fn generate_service_specific_actions(&self, layer: NetworkLayer, service_type: ServiceType, demand: f32) -> Vec<LoadBalancingAction> {
        let mut actions = Vec::new();
        
        let priority = self.template_config.service_priorities.get(&service_type).unwrap_or(&4);
        
        // Adjust QoS parameters based on service type
        actions.push(LoadBalancingAction::UpdateQoSParameters {
            layer,
            service_type,
            new_priority: *priority,
            bandwidth_allocation: demand * 0.8,  // Allocate 80% of predicted demand
        });
        
        // Service-specific optimizations
        match service_type {
            ServiceType::URLLC => {
                // Ensure URLLC gets priority on 5G layers
                if layer == NetworkLayer::L2100 {
                    actions.push(LoadBalancingAction::TriggerHandover {
                        source_layer: NetworkLayer::L2100,
                        target_layer: NetworkLayer::N78,
                        user_percentage: 0.5,
                        service_type: Some(ServiceType::URLLC),
                    });
                }
            }
            
            ServiceType::VoNR => {
                // Optimize for voice services
                actions.push(LoadBalancingAction::AdjustTransmissionPower {
                    layer,
                    power_adjustment: 1.0,  // Slight power increase for better coverage
                });
            }
            
            ServiceType::EMBB => {
                // Enable carrier aggregation for high-throughput services
                if layer == NetworkLayer::N78 {
                    actions.push(LoadBalancingAction::ConfigureCarrierAggregation {
                        primary_layer: NetworkLayer::N78,
                        secondary_layer: NetworkLayer::N258,
                        enable: true,
                    });
                }
            }
            
            ServiceType::MIoT => {
                // Optimize for massive IoT - reduce power and adjust coverage
                actions.push(LoadBalancingAction::AdjustAntennaTilt {
                    layer,
                    tilt_adjustment: -1.0,  // Reduce tilt for better coverage
                });
            }
        }
        
        actions
    }
    
    fn generate_qos_recovery_actions(&self, layer: NetworkLayer, qos_idx: usize, degradation: f32) -> Vec<LoadBalancingAction> {
        let mut actions = Vec::new();
        
        // Map QoS index to service type (simplified mapping)
        let service_type = match qos_idx {
            0..=2 => ServiceType::URLLC,
            3..=5 => ServiceType::VoNR,
            6..=8 => ServiceType::EMBB,
            _ => ServiceType::MIoT,
        };
        
        // Increase priority for degraded services
        actions.push(LoadBalancingAction::UpdateQoSParameters {
            layer,
            service_type,
            new_priority: 1,  // Highest priority
            bandwidth_allocation: 0.9,  // Allocate 90% of available bandwidth
        });
        
        // Increase transmission power to improve signal quality
        actions.push(LoadBalancingAction::AdjustTransmissionPower {
            layer,
            power_adjustment: 2.0,
        });
        
        actions
    }
    
    fn create_amos_script(&self, timestamp: i64, actions: Vec<LoadBalancingAction>) -> AmosScript {
        let datetime = Utc.timestamp_opt(timestamp, 0).single().unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            Utc.timestamp_opt(since_epoch.as_secs() as i64, 0).single().unwrap()
        });
        let script_id = format!("AMOS_LB_{}", timestamp);
        
        let mut script_content = String::new();
        let mut validation_checks = Vec::new();
        let mut rollback_commands = Vec::new();
        
        // Generate script header
        script_content.push_str(&format!(
            "#!/bin/bash\n# AMOS Load Balancing Script\n# Generated: {}\n# Script ID: {}\n\n",
            timestamp,
            script_id
        ));
        
        // Add safety checks
        script_content.push_str("# Safety checks\n");
        script_content.push_str("set -e  # Exit on error\n");
        script_content.push_str("set -u  # Exit on undefined variable\n\n");
        
        // Generate commands for each action
        for (idx, action) in actions.iter().enumerate() {
            let (command, validation, rollback) = self.generate_action_commands(action, idx);
            script_content.push_str(&command);
            validation_checks.push(validation);
            rollback_commands.push(rollback);
        }
        
        // Add validation section
        script_content.push_str("\n# Validation checks\n");
        for check in &validation_checks {
            script_content.push_str(&format!("echo \"Validating: {}\"\n", check));
            script_content.push_str(&format!("{}\n", check));
        }
        
        // Generate rollback script
        let rollback_script = self.generate_rollback_script(&rollback_commands);
        
        AmosScript {
            script_id,
            timestamp: datetime,
            actions,
            script_content,
            validation_checks,
            rollback_script,
        }
    }
    
    fn generate_action_commands(&self, action: &LoadBalancingAction, idx: usize) -> (String, String, String) {
        let mut command = String::new();
        let mut validation = String::new();
        let mut rollback = String::new();
        
        command.push_str(&format!("# Action {}: {:?}\n", idx + 1, action));
        
        match action {
            LoadBalancingAction::TriggerHandover {
                source_layer,
                target_layer,
                user_percentage,
                service_type,
            } => {
                let unknown = "unknown".to_string();
                let source_freq = self.template_config.frequency_bands.get(source_layer).unwrap_or(&unknown);
                let target_freq = self.template_config.frequency_bands.get(target_layer).unwrap_or(&unknown);
                
                command.push_str(&format!(
                    "amos_cli handover --source-freq {} --target-freq {} --percentage {} --hysteresis {}\n",
                    source_freq, target_freq, user_percentage, self.template_config.handover_hysteresis
                ));
                
                validation = format!("amos_cli verify handover --source-freq {} --target-freq {}", source_freq, target_freq);
                rollback = format!("amos_cli handover --source-freq {} --target-freq {} --percentage {}", target_freq, source_freq, user_percentage);
            }
            
            LoadBalancingAction::AdjustAntennaTilt { layer, tilt_adjustment } => {
                let unknown = "unknown".to_string();
                let freq = self.template_config.frequency_bands.get(layer).unwrap_or(&unknown);
                
                command.push_str(&format!(
                    "amos_cli antenna --freq {} --tilt-adjustment {} --apply\n",
                    freq, tilt_adjustment
                ));
                
                validation = format!("amos_cli verify antenna --freq {} --tilt-range -5:15", freq);
                rollback = format!("amos_cli antenna --freq {} --tilt-adjustment {} --apply", freq, -tilt_adjustment);
            }
            
            LoadBalancingAction::AdjustTransmissionPower { layer, power_adjustment } => {
                let unknown = "unknown".to_string();
                let freq = self.template_config.frequency_bands.get(layer).unwrap_or(&unknown);
                
                command.push_str(&format!(
                    "amos_cli power --freq {} --adjustment {} --apply\n",
                    freq, power_adjustment
                ));
                
                validation = format!("amos_cli verify power --freq {} --range 10:46", freq);
                rollback = format!("amos_cli power --freq {} --adjustment {} --apply", freq, -power_adjustment);
            }
            
            LoadBalancingAction::UpdateQoSParameters {
                layer,
                service_type,
                new_priority,
                bandwidth_allocation,
            } => {
                let unknown = "unknown".to_string();
                let freq = self.template_config.frequency_bands.get(layer).unwrap_or(&unknown);
                
                command.push_str(&format!(
                    "amos_cli qos --freq {} --service {:?} --priority {} --bandwidth {} --apply\n",
                    freq, service_type, new_priority, bandwidth_allocation
                ));
                
                validation = format!("amos_cli verify qos --freq {} --service {:?}", freq, service_type);
                rollback = format!("amos_cli qos --freq {} --service {:?} --priority 3 --bandwidth 0.5 --apply", freq, service_type);
            }
            
            LoadBalancingAction::ConfigureCarrierAggregation {
                primary_layer,
                secondary_layer,
                enable,
            } => {
                let unknown = "unknown".to_string();
                let primary_freq = self.template_config.frequency_bands.get(primary_layer).unwrap_or(&unknown);
                let secondary_freq = self.template_config.frequency_bands.get(secondary_layer).unwrap_or(&unknown);
                
                command.push_str(&format!(
                    "amos_cli carrier-aggregation --primary {} --secondary {} --enable {} --apply\n",
                    primary_freq, secondary_freq, enable
                ));
                
                validation = format!("amos_cli verify carrier-aggregation --primary {} --secondary {}", primary_freq, secondary_freq);
                rollback = format!("amos_cli carrier-aggregation --primary {} --secondary {} --enable {} --apply", primary_freq, secondary_freq, !enable);
            }
            
            LoadBalancingAction::AdjustLoadBalancingWeights { layer, weight_factor } => {
                let unknown = "unknown".to_string();
                let freq = self.template_config.frequency_bands.get(layer).unwrap_or(&unknown);
                
                command.push_str(&format!(
                    "amos_cli load-balance --freq {} --weight {} --apply\n",
                    freq, weight_factor
                ));
                
                validation = format!("amos_cli verify load-balance --freq {}", freq);
                rollback = format!("amos_cli load-balance --freq {} --weight 1.0 --apply", freq);
            }
        }
        
        command.push_str("\n");
        (command, validation, rollback)
    }
    
    fn generate_rollback_script(&self, rollback_commands: &[String]) -> String {
        let mut script = String::new();
        
        script.push_str("#!/bin/bash\n");
        script.push_str("# AMOS Load Balancing Rollback Script\n");
        script.push_str("# This script reverts the changes made by the load balancing script\n\n");
        
        script.push_str("set -e  # Exit on error\n");
        script.push_str("set -u  # Exit on undefined variable\n\n");
        
        script.push_str("echo \"Starting rollback...\"\n\n");
        
        for (idx, rollback_cmd) in rollback_commands.iter().enumerate() {
            script.push_str(&format!("# Rollback action {}\n", idx + 1));
            script.push_str(&format!("{}\n", rollback_cmd));
            script.push_str("sleep 1  # Wait between actions\n\n");
        }
        
        script.push_str("echo \"Rollback completed\"\n");
        script
    }
}

/// Generate load balancing scripts based on predictions
pub fn generate_load_balancing_scripts(predictions: &ForecastResult, config: &PredictorConfig) -> Vec<String> {
    let generator = AmosScriptGenerator::new(config.clone());
    let scripts = generator.generate_scripts(predictions);
    
    scripts.into_iter().map(|script| script.script_content).collect()
}

/// Generate emergency load balancing script for immediate execution
pub fn generate_emergency_script(layer: NetworkLayer, prb_utilization: f32) -> String {
    let mut script = String::new();
    
    script.push_str("#!/bin/bash\n");
    script.push_str("# Emergency Load Balancing Script\n");
    script.push_str("# WARNING: This script implements immediate load balancing actions\n\n");
    
    script.push_str("set -e\n");
    script.push_str("set -u\n\n");
    
    script.push_str(&format!("# Emergency response for layer {:?} with PRB utilization {:.2}\n", layer, prb_utilization));
    
    match layer {
        NetworkLayer::L2100 => {
            script.push_str("# Immediate handover to 5G\n");
            script.push_str("amos_cli handover --source-freq 2100 --target-freq 3500 --percentage 0.5 --priority emergency\n");
        }
        NetworkLayer::N78 => {
            script.push_str("# Enable carrier aggregation\n");
            script.push_str("amos_cli carrier-aggregation --primary 3500 --secondary 26000 --enable true --priority emergency\n");
        }
        NetworkLayer::N258 => {
            script.push_str("# Increase transmission power\n");
            script.push_str("amos_cli power --freq 26000 --adjustment 5.0 --apply --priority emergency\n");
        }
    }
    
    script.push_str("\necho \"Emergency load balancing completed\"\n");
    script
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_amos_script_generation() {
        let config = PredictorConfig::default();
        let generator = AmosScriptGenerator::new(config);
        
        // Create mock predictions
        let horizons = vec![1234567890, 1234567950];
        let prb_predictions = Array2::from_shape_vec((2, 3), vec![0.9, 0.5, 0.3, 0.7, 0.6, 0.4]).unwrap();
        let service_predictions = Array2::from_shape_vec((2, 12), vec![0.8; 24]).unwrap()
            .into_shape((2, 3, 4)).unwrap();
        let qos_predictions = Array2::from_shape_vec((2, 33), vec![0.2; 66]).unwrap()
            .into_shape((2, 3, 11)).unwrap();
        let confidence_intervals = Array2::from_shape_vec((2, 6), vec![0.1; 12]).unwrap()
            .into_shape((2, 3, 2)).unwrap();
        
        let predictions = ForecastResult {
            horizons,
            prb_predictions,
            service_predictions,
            qos_predictions,
            confidence_intervals,
        };
        
        let scripts = generator.generate_scripts(&predictions);
        assert!(!scripts.is_empty());
        
        // Check that high PRB utilization generates appropriate actions
        let first_script = &scripts[0];
        assert!(first_script.script_content.contains("handover"));
    }
    
    #[test]
    fn test_emergency_script_generation() {
        let script = generate_emergency_script(NetworkLayer::L2100, 0.95);
        assert!(script.contains("emergency"));
        assert!(script.contains("handover"));
    }
    
    #[test]
    fn test_threshold_configuration() {
        let thresholds = ThresholdConfig {
            prb_high_threshold: 0.9,
            prb_medium_threshold: 0.7,
            prb_low_threshold: 0.5,
            ..Default::default()
        };
        
        assert_eq!(thresholds.prb_high_threshold, 0.9);
        assert_eq!(thresholds.prb_medium_threshold, 0.7);
        assert_eq!(thresholds.prb_low_threshold, 0.5);
    }
}