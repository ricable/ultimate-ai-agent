//! Handover prediction engine using ruv-FANN neural networks

use anyhow::Result;
use chrono::{DateTime, Utc};
use ruv_fann::{Network, ActivationFunction, TrainingAlgorithm};
use std::sync::Arc;
use std::collections::VecDeque;
use tokio::sync::RwLock;

use crate::config::OptMobConfig;
use crate::{UeMetrics, NeighborCell, HandoverPrediction, TargetCellPrediction, HandoverAction};
use crate::processor::ProcessedMetrics;
use crate::analyzer::NeighborAnalysis;

/// Neural network-based handover predictor
pub struct HandoverPredictor {
    config: Arc<OptMobConfig>,
    network: Arc<RwLock<Network>>,
    training_history: Arc<RwLock<VecDeque<TrainingExample>>>,
    model_version: Arc<RwLock<u32>>,
}

#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input: Vec<f64>,
    pub output: Vec<f64>,
    pub timestamp: DateTime<Utc>,
    pub ue_id: String,
    pub actual_handover: bool,
}

impl HandoverPredictor {
    /// Create new handover predictor
    pub async fn new(config: Arc<OptMobConfig>) -> Result<Self> {
        let network = Self::create_network(&config).await?;
        let network = Arc::new(RwLock::new(network));
        let training_history = Arc::new(RwLock::new(VecDeque::new()));
        let model_version = Arc::new(RwLock::new(1));
        
        Ok(Self {
            config,
            network,
            training_history,
            model_version,
        })
    }
    
    /// Load predictor from saved model
    pub async fn load_from_file(config: Arc<OptMobConfig>, model_path: &str) -> Result<Self> {
        let mut network = Self::create_network(&config).await?;
        network.load(model_path)?;
        
        let network = Arc::new(RwLock::new(network));
        let training_history = Arc::new(RwLock::new(VecDeque::new()));
        let model_version = Arc::new(RwLock::new(1));
        
        log::info!("Loaded handover prediction model from {}", model_path);
        
        Ok(Self {
            config,
            network,
            training_history,
            model_version,
        })
    }
    
    /// Predict handover probability and target cells
    pub async fn predict(
        &self,
        metrics: &ProcessedMetrics,
        neighbor_analysis: &NeighborAnalysis,
    ) -> Result<HandoverPrediction> {
        let input = self.prepare_input(metrics, neighbor_analysis)?;
        
        let network = self.network.read().await;
        let output = network.run(&input)?;
        
        let handover_probability = output[0];
        let confidence_score = self.calculate_confidence(&output);
        
        // Predict target cells
        let target_cells = self.predict_target_cells(
            neighbor_analysis,
            handover_probability,
        ).await?;
        
        // Determine recommended action
        let recommended_action = self.determine_action(
            handover_probability,
            confidence_score,
            &target_cells,
        );
        
        // Calculate time to handover
        let time_to_handover = self.calculate_time_to_handover(
            metrics,
            handover_probability,
        );
        
        let prediction = HandoverPrediction {
            ue_id: metrics.ue_id.clone(),
            source_cell_id: metrics.cell_id.clone(),
            prediction_timestamp: Utc::now(),
            handover_probability,
            target_cells,
            trigger_reason: self.determine_trigger_reason(metrics, neighbor_analysis),
            confidence_score,
            time_to_handover_seconds: time_to_handover,
            recommended_action,
        };
        
        // Store prediction for training
        self.store_prediction_for_training(&input, &output, &metrics.ue_id).await?;
        
        Ok(prediction)
    }
    
    /// Retrain the model with new data
    pub async fn retrain(&mut self, training_data: &[(UeMetrics, bool)]) -> Result<()> {
        let mut network = self.network.write().await;
        let mut training_history = self.training_history.write().await;
        
        // Convert training data to network format
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        
        for (metrics, handover_occurred) in training_data {
            let input = self.prepare_input_from_ue_metrics(metrics)?;
            let output = if *handover_occurred { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
            
            inputs.push(input.clone());
            outputs.push(output.clone());
            
            // Add to training history
            training_history.push_back(TrainingExample {
                input,
                output,
                timestamp: Utc::now(),
                ue_id: metrics.ue_id.clone(),
                actual_handover: *handover_occurred,
            });
        }
        
        // Limit training history size
        const MAX_TRAINING_HISTORY: usize = 10000;
        while training_history.len() > MAX_TRAINING_HISTORY {
            training_history.pop_front();
        }
        
        // Train network
        network.train_batch(&inputs, &outputs, 100)?;
        
        // Update model version
        let mut version = self.model_version.write().await;
        *version += 1;
        
        log::info!("Retrained handover prediction model with {} examples, version: {}", 
                  training_data.len(), *version);
        
        Ok(())
    }
    
    /// Save model to file
    pub async fn save_model(&self, path: &str) -> Result<()> {
        let network = self.network.read().await;
        network.save(path)?;
        
        log::info!("Saved handover prediction model to {}", path);
        Ok(())
    }
    
    /// Get model version
    pub async fn get_model_version(&self) -> u32 {
        let version = self.model_version.read().await;
        *version
    }
    
    /// Get training history size
    pub async fn get_training_history_size(&self) -> usize {
        let history = self.training_history.read().await;
        history.len()
    }
    
    async fn create_network(config: &OptMobConfig) -> Result<Network> {
        let mut layers = vec![config.model.input_features];
        layers.extend_from_slice(&config.model.hidden_layers);
        layers.push(config.model.output_classes);
        
        let mut network = Network::new(&layers)?;
        
        // Set activation functions
        let activation = match config.model.activation_function.as_str() {
            "sigmoid" => ActivationFunction::Sigmoid,
            "tanh" => ActivationFunction::Tanh,
            "relu" => ActivationFunction::Linear, // ReLU not directly available
            _ => ActivationFunction::Sigmoid,
        };
        
        network.set_activation_function_hidden(activation);
        network.set_activation_function_output(activation);
        
        // Set training algorithm
        network.set_training_algorithm(TrainingAlgorithm::Rprop);
        network.set_learning_rate(config.model.learning_rate);
        
        Ok(network)
    }
    
    fn prepare_input(
        &self,
        metrics: &ProcessedMetrics,
        neighbor_analysis: &NeighborAnalysis,
    ) -> Result<Vec<f64>> {
        let mut input = Vec::new();
        
        // UE metrics features
        input.push(metrics.rsrp_normalized);
        input.push(metrics.sinr_normalized);
        input.push(metrics.speed_normalized);
        input.push(metrics.throughput_normalized);
        input.push(metrics.cqi_normalized);
        input.push(metrics.phr_normalized);
        input.push(metrics.ta_normalized);
        input.push(metrics.load_factor);
        
        // Neighbor analysis features
        input.push(neighbor_analysis.best_neighbor_rsrp);
        input.push(neighbor_analysis.best_neighbor_sinr);
        input.push(neighbor_analysis.rsrp_differential);
        input.push(neighbor_analysis.sinr_differential);
        input.push(neighbor_analysis.load_balance_score);
        input.push(neighbor_analysis.mobility_score);
        
        // Time-based features
        input.push(metrics.trend_rsrp);
        input.push(metrics.trend_sinr);
        input.push(metrics.variance_rsrp);
        input.push(metrics.variance_sinr);
        
        Ok(input)
    }
    
    fn prepare_input_from_ue_metrics(&self, metrics: &UeMetrics) -> Result<Vec<f64>> {
        let mut input = Vec::new();
        
        // Normalize UE metrics
        input.push(self.normalize_rsrp(metrics.rsrp_dbm));
        input.push(self.normalize_sinr(metrics.sinr_db));
        input.push(self.normalize_speed(metrics.speed_kmh));
        input.push(metrics.throughput_mbps / 1000.0); // Normalize to Gbps
        input.push(metrics.cqi as f64 / 15.0); // Normalize CQI
        input.push((metrics.phr_db + 40.0) / 80.0); // Normalize PHR
        input.push(metrics.ta_us / 1000.0); // Normalize TA
        input.push(metrics.load_factor);
        
        // Add dummy neighbor features for compatibility
        input.extend(vec![0.0; 10]);
        
        Ok(input)
    }
    
    fn normalize_rsrp(&self, rsrp: f64) -> f64 {
        let (min_rsrp, max_rsrp) = self.config.processing.rsrp_range;
        (rsrp - min_rsrp) / (max_rsrp - min_rsrp)
    }
    
    fn normalize_sinr(&self, sinr: f64) -> f64 {
        let (min_sinr, max_sinr) = self.config.processing.sinr_range;
        (sinr - min_sinr) / (max_sinr - min_sinr)
    }
    
    fn normalize_speed(&self, speed: f64) -> f64 {
        let (min_speed, max_speed) = self.config.processing.speed_range;
        (speed - min_speed) / (max_speed - min_speed)
    }
    
    fn calculate_confidence(&self, output: &[f64]) -> f64 {
        // Calculate confidence based on output distribution
        let max_prob = output.iter().copied().fold(0.0, f64::max);
        let entropy = output.iter()
            .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
            .sum::<f64>();
        
        // High confidence when max probability is high and entropy is low
        max_prob * (1.0 - entropy / output.len() as f64)
    }
    
    async fn predict_target_cells(
        &self,
        neighbor_analysis: &NeighborAnalysis,
        handover_probability: f64,
    ) -> Result<Vec<TargetCellPrediction>> {
        let mut target_cells = Vec::new();
        
        for neighbor in &neighbor_analysis.neighbors {
            let selection_probability = self.calculate_selection_probability(
                neighbor,
                handover_probability,
                &neighbor_analysis,
            );
            
            if selection_probability > 0.1 { // Minimum threshold
                let target_prediction = TargetCellPrediction {
                    cell_id: neighbor.cell_id.clone(),
                    selection_probability,
                    expected_rsrp_dbm: neighbor.rsrp_dbm,
                    expected_sinr_db: neighbor.sinr_db,
                    expected_throughput_mbps: neighbor.capacity_mbps * (1.0 - neighbor.load_factor),
                    load_factor: neighbor.load_factor,
                    handover_success_probability: self.calculate_handover_success_probability(neighbor),
                };
                
                target_cells.push(target_prediction);
            }
        }
        
        // Sort by selection probability
        target_cells.sort_by(|a, b| b.selection_probability.partial_cmp(&a.selection_probability).unwrap());
        
        // Take top candidates
        target_cells.truncate(self.config.neighbor_analysis.max_neighbors);
        
        Ok(target_cells)
    }
    
    fn calculate_selection_probability(
        &self,
        neighbor: &NeighborCell,
        handover_probability: f64,
        analysis: &NeighborAnalysis,
    ) -> f64 {
        // Combine signal quality, load balance, and distance factors
        let signal_score = (neighbor.rsrp_dbm + 140.0) / 100.0; // Normalize RSRP
        let load_score = 1.0 - neighbor.load_factor;
        let distance_score = 1.0 / (1.0 + neighbor.distance_meters / 1000.0); // Distance in km
        
        let combined_score = 
            signal_score * self.config.neighbor_analysis.signal_quality_weight +
            load_score * self.config.neighbor_analysis.load_balance_weight +
            distance_score * self.config.neighbor_analysis.distance_weight;
        
        combined_score * handover_probability
    }
    
    fn calculate_handover_success_probability(&self, neighbor: &NeighborCell) -> f64 {
        // Simple heuristic based on signal quality and load
        let signal_factor = ((neighbor.rsrp_dbm + 140.0) / 100.0).clamp(0.0, 1.0);
        let load_factor = (1.0 - neighbor.load_factor).clamp(0.0, 1.0);
        let availability_factor = neighbor.availability;
        
        (signal_factor * 0.4 + load_factor * 0.3 + availability_factor * 0.3)
            .clamp(0.0, 1.0)
    }
    
    fn determine_action(
        &self,
        handover_probability: f64,
        confidence_score: f64,
        target_cells: &[TargetCellPrediction],
    ) -> HandoverAction {
        let threshold = self.config.model.handover_threshold;
        let confidence_threshold = self.config.model.confidence_threshold;
        
        if handover_probability > threshold && confidence_score > confidence_threshold {
            if let Some(best_target) = target_cells.first() {
                if best_target.handover_success_probability > 0.8 {
                    HandoverAction::ExecuteHandover {
                        target_cell: best_target.cell_id.clone(),
                    }
                } else {
                    HandoverAction::PrepareHandover {
                        target_cell: best_target.cell_id.clone(),
                    }
                }
            } else {
                HandoverAction::NoAction
            }
        } else if handover_probability > threshold * 0.5 && confidence_score > confidence_threshold {
            if let Some(best_target) = target_cells.first() {
                HandoverAction::PrepareHandover {
                    target_cell: best_target.cell_id.clone(),
                }
            } else {
                HandoverAction::NoAction
            }
        } else if handover_probability < threshold * 0.2 {
            HandoverAction::NoAction
        } else {
            // Consider threshold adjustment
            let new_rsrp_threshold = -95.0; // Conservative threshold
            let new_sinr_threshold = 5.0;
            
            HandoverAction::ModifyThresholds {
                rsrp_threshold: new_rsrp_threshold,
                sinr_threshold: new_sinr_threshold,
            }
        }
    }
    
    fn determine_trigger_reason(
        &self,
        metrics: &ProcessedMetrics,
        neighbor_analysis: &NeighborAnalysis,
    ) -> String {
        let mut reasons = Vec::new();
        
        if metrics.rsrp_normalized < 0.3 {
            reasons.push("Poor RSRP".to_string());
        }
        
        if metrics.sinr_normalized < 0.3 {
            reasons.push("Poor SINR".to_string());
        }
        
        if metrics.load_factor > 0.8 {
            reasons.push("High cell load".to_string());
        }
        
        if neighbor_analysis.rsrp_differential > 5.0 {
            reasons.push("Better neighbor available".to_string());
        }
        
        if metrics.speed_normalized > 0.7 {
            reasons.push("High mobility".to_string());
        }
        
        if reasons.is_empty() {
            "Proactive optimization".to_string()
        } else {
            reasons.join(", ")
        }
    }
    
    fn calculate_time_to_handover(
        &self,
        metrics: &ProcessedMetrics,
        handover_probability: f64,
    ) -> f64 {
        // Estimate time to handover based on signal degradation trend
        let signal_degradation_rate = metrics.trend_rsrp.abs() + metrics.trend_sinr.abs();
        
        if signal_degradation_rate > 0.0 {
            // Time until critical threshold is reached
            let threshold_buffer = 5.0; // dB
            let time_to_critical = threshold_buffer / signal_degradation_rate;
            
            // Adjust based on handover probability
            time_to_critical * (1.0 - handover_probability)
        } else {
            // No degradation trend, use fixed prediction horizon
            self.config.processing.time_window_seconds as f64
        }
    }
    
    async fn store_prediction_for_training(
        &self,
        input: &[f64],
        output: &[f64],
        ue_id: &str,
    ) -> Result<()> {
        let mut training_history = self.training_history.write().await;
        
        training_history.push_back(TrainingExample {
            input: input.to_vec(),
            output: output.to_vec(),
            timestamp: Utc::now(),
            ue_id: ue_id.to_string(),
            actual_handover: false, // Will be updated when actual outcome is known
        });
        
        // Limit history size
        const MAX_HISTORY: usize = 10000;
        while training_history.len() > MAX_HISTORY {
            training_history.pop_front();
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptMobConfig;
    
    #[tokio::test]
    async fn test_predictor_creation() {
        let config = Arc::new(OptMobConfig::default());
        let predictor = HandoverPredictor::new(config).await;
        assert!(predictor.is_ok());
    }
    
    #[tokio::test]
    async fn test_input_preparation() {
        let config = Arc::new(OptMobConfig::default());
        let predictor = HandoverPredictor::new(config).await.unwrap();
        
        let metrics = UeMetrics::new("UE001".to_string(), "Cell001".to_string())
            .with_rsrp(-85.0)
            .with_sinr(15.0)
            .with_speed(60.0);
        
        let input = predictor.prepare_input_from_ue_metrics(&metrics).unwrap();
        assert_eq!(input.len(), 18); // 8 UE metrics + 10 neighbor features
    }
    
    #[test]
    fn test_normalize_functions() {
        let config = OptMobConfig::default();
        let predictor = HandoverPredictor {
            config: Arc::new(config),
            network: Arc::new(RwLock::new(Network::new(&[1, 1]).unwrap())),
            training_history: Arc::new(RwLock::new(VecDeque::new())),
            model_version: Arc::new(RwLock::new(1)),
        };
        
        // Test RSRP normalization
        let rsrp_norm = predictor.normalize_rsrp(-85.0);
        assert!(rsrp_norm >= 0.0 && rsrp_norm <= 1.0);
        
        // Test SINR normalization
        let sinr_norm = predictor.normalize_sinr(15.0);
        assert!(sinr_norm >= 0.0 && sinr_norm <= 1.0);
        
        // Test speed normalization
        let speed_norm = predictor.normalize_speed(60.0);
        assert!(speed_norm >= 0.0 && speed_norm <= 1.0);
    }
}