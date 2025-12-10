//! Agent-specific forecasting model management
//!
//! This module provides per-agent forecasting model management with adaptive
//! configuration and specialized models based on agent type and workload.

use alloc::{
    boxed::Box,
    collections::BTreeMap as HashMap,
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use core::fmt;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::models::{ForecastModel, ModelType};

/// Manages forecasting models for individual agents
pub struct AgentForecastingManager {
    agent_models: HashMap<String, AgentForecastContext>,
    resource_limit_mb: f32,
    current_memory_usage_mb: f32,
}

/// Forecasting context for a specific agent
#[derive(Clone)]
pub struct AgentForecastContext {
    pub agent_id: String,
    pub agent_type: String,
    pub primary_model: ModelType,
    pub ensemble_models: Vec<ModelType>,
    pub model_specialization: ModelSpecialization,
    pub adaptive_config: AdaptiveModelConfig,
    pub performance_history: ModelPerformanceHistory,
}

/// Model specialization based on forecast domain
#[derive(Clone, Debug)]
pub struct ModelSpecialization {
    pub forecast_domain: ForecastDomain,
    pub temporal_patterns: Vec<TemporalPattern>,
    pub optimization_objectives: Vec<OptimizationObjective>,
}

/// Forecast domain types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ForecastDomain {
    ResourceUtilization,
    TaskCompletion,
    AgentPerformance,
    SwarmDynamics,
    AnomalyDetection,
    CapacityPlanning,
}

/// Temporal pattern in time series
#[derive(Clone, Debug)]
pub struct TemporalPattern {
    pub pattern_type: String,
    pub frequency: f32,
    pub strength: f32,
    pub confidence: f32,
}

/// Optimization objective for model training
#[derive(Clone, Debug)]
pub enum OptimizationObjective {
    MinimizeLatency,
    MaximizeAccuracy,
    BalanceAccuracyLatency,
    MinimizeMemory,
}

/// Adaptive model configuration
#[derive(Clone, Debug)]
pub struct AdaptiveModelConfig {
    pub online_learning_enabled: bool,
    pub adaptation_rate: f32,
    pub model_switching_threshold: f32,
    pub ensemble_weighting_strategy: EnsembleWeightingStrategy,
    pub retraining_frequency: u32,
}

/// Strategy for weighting ensemble models
#[derive(Clone, Debug)]
pub enum EnsembleWeightingStrategy {
    Static,
    DynamicPerformance,
    Bayesian,
    StackedGeneralization,
}

/// Performance history for model tracking
#[derive(Clone, Debug)]
pub struct ModelPerformanceHistory {
    pub total_forecasts: u64,
    pub average_confidence: f32,
    pub average_latency_ms: f32,
    pub recent_accuracies: Vec<f32>,
    pub model_switches: Vec<ModelSwitchEvent>,
}

/// Model switch event record
#[derive(Clone, Debug)]
pub struct ModelSwitchEvent {
    pub timestamp: f64,
    pub from_model: String,
    pub to_model: String,
    pub reason: String,
}

impl AgentForecastingManager {
    /// Create a new agent forecasting manager
    pub fn new(resource_limit_mb: f32) -> Self {
        Self {
            agent_models: HashMap::new(),
            resource_limit_mb,
            current_memory_usage_mb: 0.0,
        }
    }

    /// Assign a forecasting model to an agent
    pub fn assign_model(
        &mut self,
        agent_id: String,
        agent_type: String,
        requirements: ForecastRequirements,
    ) -> Result<String, String> {
        // Select optimal model based on agent type
        let primary_model = self.select_optimal_model(&agent_type, &requirements)?;

        // Create model specialization
        let model_specialization = self.create_specialization(&agent_type, &requirements);

        // Create adaptive configuration
        let adaptive_config = AdaptiveModelConfig {
            online_learning_enabled: requirements.online_learning,
            adaptation_rate: 0.01,
            model_switching_threshold: 0.85,
            ensemble_weighting_strategy: EnsembleWeightingStrategy::DynamicPerformance,
            retraining_frequency: 100,
        };

        // Initialize performance history
        let performance_history = ModelPerformanceHistory {
            total_forecasts: 0,
            average_confidence: 0.0,
            average_latency_ms: 0.0,
            recent_accuracies: Vec::new(),
            model_switches: Vec::new(),
        };

        // Create forecast context
        let context = AgentForecastContext {
            agent_id: agent_id.clone(),
            agent_type,
            primary_model,
            ensemble_models: Vec::new(),
            model_specialization,
            adaptive_config,
            performance_history,
        };

        // Store context
        self.agent_models.insert(agent_id.clone(), context);

        Ok(agent_id)
    }

    /// Get agent's current forecast state
    pub fn get_agent_state(&self, agent_id: &str) -> Option<&AgentForecastContext> {
        self.agent_models.get(agent_id)
    }

    /// Update agent model performance
    pub fn update_performance(
        &mut self,
        agent_id: &str,
        latency_ms: f32,
        accuracy: f32,
        confidence: f32,
    ) -> Result<(), String> {
        let context = self
            .agent_models
            .get_mut(agent_id)
            .ok_or_else(|| format!("Agent {} not found", agent_id))?;

        // Update performance metrics
        let history = &mut context.performance_history;
        history.total_forecasts += 1;

        // Update moving averages
        let alpha = 0.1; // Exponential moving average factor
        history.average_latency_ms =
            (1.0 - alpha) * history.average_latency_ms + alpha * latency_ms;
        history.average_confidence =
            (1.0 - alpha) * history.average_confidence + alpha * confidence;

        // Track recent accuracies
        history.recent_accuracies.push(accuracy);
        if history.recent_accuracies.len() > 100 {
            history.recent_accuracies.remove(0);
        }

        // Check if model switch is needed
        if accuracy < context.adaptive_config.model_switching_threshold {
            // TODO: Implement model switching logic
        }

        Ok(())
    }

    /// Select optimal model based on agent type and requirements
    fn select_optimal_model(
        &self,
        agent_type: &str,
        requirements: &ForecastRequirements,
    ) -> Result<ModelType, String> {
        let model = match agent_type {
            "researcher" => {
                if requirements.interpretability_needed {
                    ModelType::NHITS // Good for exploratory analysis
                } else {
                    ModelType::TFT // Temporal Fusion Transformer
                }
            }
            "coder" => ModelType::LSTM,       // Sequential task patterns
            "analyst" => ModelType::TFT,      // Interpretable attention mechanism
            "optimizer" => ModelType::NBEATS, // Pure neural architecture
            "coordinator" => ModelType::DeepAR, // Probabilistic forecasts
            _ => ModelType::MLP,              // Generic baseline
        };

        Ok(model)
    }

    /// Create model specialization based on agent type
    fn create_specialization(
        &self,
        agent_type: &str,
        requirements: &ForecastRequirements,
    ) -> ModelSpecialization {
        let forecast_domain = match agent_type {
            "researcher" => ForecastDomain::TaskCompletion,
            "coder" => ForecastDomain::TaskCompletion,
            "analyst" => ForecastDomain::AgentPerformance,
            "optimizer" => ForecastDomain::ResourceUtilization,
            "coordinator" => ForecastDomain::SwarmDynamics,
            _ => ForecastDomain::AgentPerformance,
        };

        let temporal_patterns = vec![
            TemporalPattern {
                pattern_type: "daily".to_string(),
                frequency: 24.0,
                strength: 0.8,
                confidence: 0.9,
            },
            TemporalPattern {
                pattern_type: "weekly".to_string(),
                frequency: 168.0,
                strength: 0.6,
                confidence: 0.85,
            },
        ];

        let optimization_objectives = if requirements.latency_requirement_ms < 100.0 {
            vec![OptimizationObjective::MinimizeLatency]
        } else {
            vec![OptimizationObjective::BalanceAccuracyLatency]
        };

        ModelSpecialization {
            forecast_domain,
            temporal_patterns,
            optimization_objectives,
        }
    }
}

/// Forecast requirements for model selection
pub struct ForecastRequirements {
    pub horizon: usize,
    pub frequency: String,
    pub accuracy_target: f32,
    pub latency_requirement_ms: f32,
    pub interpretability_needed: bool,
    pub online_learning: bool,
}

impl Default for ForecastRequirements {
    fn default() -> Self {
        Self {
            horizon: 24,
            frequency: "H".to_string(), // Hourly
            accuracy_target: 0.9,
            latency_requirement_ms: 200.0,
            interpretability_needed: false,
            online_learning: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_model_assignment() {
        let mut manager = AgentForecastingManager::new(100.0);
        let requirements = ForecastRequirements::default();

        let result = manager.assign_model(
            "agent_1".to_string(),
            "researcher".to_string(),
            requirements,
        );

        assert!(result.is_ok());
        assert!(manager.get_agent_state("agent_1").is_some());
    }
}
