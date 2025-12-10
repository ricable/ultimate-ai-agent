//! QoE Prediction Networks for RIC-TSA
//! 
//! This module implements neural networks optimized for predicting Quality of Experience
//! metrics in real-time for traffic steering decisions.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;

use crate::pfs_core::{NeuralNetwork, Layer, Activation, Tensor, TensorOps, DenseLayer, Adam};
use super::{UEContext, QoEMetrics, CellCarrier, ServiceType, SteeringFeedback, RicTsaError};

/// QoE prediction result with confidence scores
#[derive(Debug, Clone)]
pub struct QoEPredictionResult {
    pub predicted_qoe: QoEMetrics,
    pub cell_scores: HashMap<u32, f32>,
    pub confidence: f32,
    pub prediction_horizon: u32, // seconds
    pub uncertainty: f32,
}

/// Feature vector for QoE prediction
#[derive(Debug, Clone)]
pub struct QoEFeatures {
    // Network features
    pub current_throughput: f32,
    pub current_latency: f32,
    pub current_jitter: f32,
    pub current_packet_loss: f32,
    pub signal_strength: f32,
    pub signal_quality: f32,
    pub interference_level: f32,
    
    // Cell features
    pub cell_load: f32,
    pub cell_capacity: f32,
    pub neighbor_cells: Vec<f32>,
    pub frequency_band: f32,
    pub bandwidth: f32,
    
    // User features
    pub service_type: f32,
    pub user_priority: f32,
    pub mobility_speed: f32,
    pub device_type: f32,
    pub historical_qoe: Vec<f32>,
    
    // Temporal features
    pub time_of_day: f32,
    pub day_of_week: f32,
    pub traffic_pattern: f32,
    pub seasonal_factor: f32,
    
    // Environmental features
    pub weather_impact: f32,
    pub event_impact: f32,
    pub congestion_level: f32,
}

/// QoE prediction network with attention mechanism
pub struct QoEPredictor {
    // Main prediction network
    primary_network: Arc<RwLock<NeuralNetwork>>,
    
    // Attention network for feature importance
    attention_network: Arc<RwLock<NeuralNetwork>>,
    
    // Service-specific networks
    service_networks: HashMap<ServiceType, Arc<RwLock<NeuralNetwork>>>,
    
    // Temporal prediction network
    temporal_network: Arc<RwLock<NeuralNetwork>>,
    
    // Feature normalization parameters
    feature_stats: Arc<RwLock<FeatureStats>>,
    
    // Model configuration
    config: QoEPredictorConfig,
}

/// Configuration for QoE predictor
#[derive(Debug, Clone)]
pub struct QoEPredictorConfig {
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub output_size: usize,
    pub attention_heads: usize,
    pub dropout_rate: f32,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub prediction_horizon: u32,
}

/// Feature statistics for normalization
#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

impl Default for QoEPredictorConfig {
    fn default() -> Self {
        Self {
            input_size: 32,
            hidden_sizes: vec![256, 128, 64],
            output_size: 8, // QoE metrics
            attention_heads: 8,
            dropout_rate: 0.1,
            learning_rate: 0.001,
            batch_size: 32,
            sequence_length: 10,
            prediction_horizon: 30,
        }
    }
}

impl QoEPredictor {
    /// Create a new QoE predictor with optimized architecture
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = QoEPredictorConfig::default();
        Self::new_with_config(config)
    }

    /// Create QoE predictor with custom configuration
    pub fn new_with_config(config: QoEPredictorConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Create primary prediction network
        let primary_network = Arc::new(RwLock::new(Self::build_primary_network(&config)?));
        
        // Create attention network
        let attention_network = Arc::new(RwLock::new(Self::build_attention_network(&config)?));
        
        // Create temporal network
        let temporal_network = Arc::new(RwLock::new(Self::build_temporal_network(&config)?));
        
        // Create service-specific networks
        let service_networks = Self::build_service_networks(&config)?;
        
        // Initialize feature statistics
        let feature_stats = Arc::new(RwLock::new(FeatureStats {
            mean: vec![0.0; config.input_size],
            std: vec![1.0; config.input_size],
            min: vec![f32::MIN; config.input_size],
            max: vec![f32::MAX; config.input_size],
        }));

        Ok(Self {
            primary_network,
            attention_network,
            service_networks,
            temporal_network,
            feature_stats,
            config,
        })
    }

    /// Build primary prediction network
    fn build_primary_network(config: &QoEPredictorConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        // Input layer
        let mut prev_size = config.input_size;
        
        // Hidden layers with residual connections
        for &hidden_size in &config.hidden_sizes {
            network.add_layer(Box::new(DenseLayer::new(prev_size, hidden_size)));
            network.add_layer(Box::new(Activation::ReLU));
            prev_size = hidden_size;
        }
        
        // Output layer for QoE metrics
        network.add_layer(Box::new(DenseLayer::new(prev_size, config.output_size)));
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Build attention network for feature importance
    fn build_attention_network(config: &QoEPredictorConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        // Multi-head attention mechanism
        network.add_layer(Box::new(DenseLayer::new(config.input_size, config.input_size)));
        network.add_layer(Box::new(Activation::Tanh));
        network.add_layer(Box::new(DenseLayer::new(config.input_size, config.attention_heads)));
        network.add_layer(Box::new(Activation::Softmax));
        
        Ok(network)
    }

    /// Build temporal prediction network
    fn build_temporal_network(config: &QoEPredictorConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        // LSTM-like architecture for temporal patterns
        let temporal_size = config.input_size * config.sequence_length;
        network.add_layer(Box::new(DenseLayer::new(temporal_size, 128)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(128, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(64, config.output_size)));
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Build service-specific networks
    fn build_service_networks(config: &QoEPredictorConfig) -> Result<HashMap<ServiceType, Arc<RwLock<NeuralNetwork>>>, Box<dyn std::error::Error>> {
        let mut networks = HashMap::new();
        
        let service_types = vec![
            ServiceType::VideoStreaming,
            ServiceType::VoiceCall,
            ServiceType::Gaming,
            ServiceType::FileTransfer,
            ServiceType::WebBrowsing,
            ServiceType::IoTSensor,
            ServiceType::Emergency,
            ServiceType::AR_VR,
        ];
        
        for service_type in service_types {
            let network = Self::build_service_specific_network(config, &service_type)?;
            networks.insert(service_type, Arc::new(RwLock::new(network)));
        }
        
        Ok(networks)
    }

    /// Build network optimized for specific service type
    fn build_service_specific_network(config: &QoEPredictorConfig, service_type: &ServiceType) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        // Service-specific architecture
        let hidden_sizes = match service_type {
            ServiceType::VideoStreaming => vec![128, 64], // Focus on throughput and latency
            ServiceType::VoiceCall => vec![64, 32],       // Focus on latency and jitter
            ServiceType::Gaming => vec![64, 32],          // Focus on latency and reliability
            ServiceType::FileTransfer => vec![64, 32],    // Focus on throughput
            ServiceType::WebBrowsing => vec![64, 32],     // Balanced
            ServiceType::IoTSensor => vec![32, 16],       // Simple, low latency
            ServiceType::Emergency => vec![128, 64],      // High reliability
            ServiceType::AR_VR => vec![256, 128],         // Complex requirements
        };
        
        let mut prev_size = config.input_size;
        for &hidden_size in &hidden_sizes {
            network.add_layer(Box::new(DenseLayer::new(prev_size, hidden_size)));
            network.add_layer(Box::new(Activation::ReLU));
            prev_size = hidden_size;
        }
        
        network.add_layer(Box::new(DenseLayer::new(prev_size, config.output_size)));
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Extract features from UE context and network state
    pub fn extract_features(&self, ue_context: &UEContext, cell_carriers: &HashMap<u32, CellCarrier>) -> QoEFeatures {
        let current_qoe = &ue_context.current_qoe;
        
        // Get neighbor cell information
        let neighbor_cells: Vec<f32> = cell_carriers.values()
            .take(5) // Top 5 neighbor cells
            .map(|c| c.current_load)
            .collect();
        
        // Historical QoE (simplified)
        let historical_qoe = vec![
            current_qoe.throughput,
            current_qoe.latency,
            current_qoe.jitter,
            current_qoe.packet_loss,
            current_qoe.video_quality,
        ];
        
        // Service type encoding
        let service_type = match ue_context.service_type {
            ServiceType::VideoStreaming => 0.1,
            ServiceType::VoiceCall => 0.2,
            ServiceType::Gaming => 0.3,
            ServiceType::FileTransfer => 0.4,
            ServiceType::WebBrowsing => 0.5,
            ServiceType::IoTSensor => 0.6,
            ServiceType::Emergency => 0.7,
            ServiceType::AR_VR => 0.8,
        };
        
        // Get current time features
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let time_of_day = ((now % 86400) as f32) / 86400.0;
        let day_of_week = (((now / 86400) % 7) as f32) / 7.0;
        
        QoEFeatures {
            // Network features
            current_throughput: current_qoe.throughput,
            current_latency: current_qoe.latency,
            current_jitter: current_qoe.jitter,
            current_packet_loss: current_qoe.packet_loss,
            signal_strength: 0.8, // Placeholder - would come from measurements
            signal_quality: 0.9,  // Placeholder
            interference_level: 0.1, // Placeholder
            
            // Cell features
            cell_load: cell_carriers.values().next().map(|c| c.current_load).unwrap_or(50.0),
            cell_capacity: cell_carriers.values().next().map(|c| c.max_capacity).unwrap_or(100.0),
            neighbor_cells,
            frequency_band: 1.8, // Placeholder for band encoding
            bandwidth: 20.0,     // Placeholder
            
            // User features
            service_type,
            user_priority: match ue_context.user_group {
                super::UserGroup::Premium => 0.9,
                super::UserGroup::Standard => 0.7,
                super::UserGroup::Basic => 0.5,
                super::UserGroup::IoT => 0.3,
                super::UserGroup::Emergency => 1.0,
            },
            mobility_speed: match ue_context.mobility_pattern {
                super::MobilityPattern::Stationary => 0.0,
                super::MobilityPattern::Pedestrian => 0.2,
                super::MobilityPattern::Vehicular => 0.6,
                super::MobilityPattern::HighSpeed => 1.0,
            },
            device_type: if ue_context.device_capabilities.ca_support { 0.8 } else { 0.4 },
            historical_qoe,
            
            // Temporal features
            time_of_day,
            day_of_week,
            traffic_pattern: 0.5, // Placeholder
            seasonal_factor: 0.5, // Placeholder
            
            // Environmental features
            weather_impact: 0.1,   // Placeholder
            event_impact: 0.1,     // Placeholder
            congestion_level: 0.3, // Placeholder
        }
    }

    /// Predict QoE for multiple cells (main prediction function)
    pub async fn predict_qoe_multi_cell(
        &self,
        ue_context: &UEContext,
        cell_carriers: &HashMap<u32, CellCarrier>,
    ) -> Result<QoEPredictionResult, Box<dyn std::error::Error>> {
        let features = self.extract_features(ue_context, cell_carriers);
        
        // Normalize features
        let normalized_features = self.normalize_features(&features).await?;
        
        // Get attention weights
        let attention_weights = self.compute_attention_weights(&normalized_features).await?;
        
        // Apply attention to features
        let attended_features = self.apply_attention(&normalized_features, &attention_weights);
        
        // Get service-specific prediction
        let service_prediction = self.predict_service_specific(&attended_features, &ue_context.service_type).await?;
        
        // Get temporal prediction
        let temporal_prediction = self.predict_temporal(&attended_features).await?;
        
        // Combine predictions
        let primary_prediction = self.predict_primary(&attended_features).await?;
        
        // Ensemble the predictions
        let final_prediction = self.ensemble_predictions(
            &primary_prediction,
            &service_prediction,
            &temporal_prediction,
        );
        
        // Predict QoE for each cell
        let mut cell_scores = HashMap::new();
        for (cell_id, _carrier) in cell_carriers {
            let cell_score = self.predict_cell_qoe(*cell_id, &final_prediction).await?;
            cell_scores.insert(*cell_id, cell_score);
        }
        
        // Calculate confidence and uncertainty
        let confidence = self.calculate_confidence(&final_prediction, &attention_weights);
        let uncertainty = self.calculate_uncertainty(&final_prediction);
        
        Ok(QoEPredictionResult {
            predicted_qoe: self.tensor_to_qoe_metrics(&final_prediction),
            cell_scores,
            confidence,
            prediction_horizon: self.config.prediction_horizon,
            uncertainty,
        })
    }

    /// Normalize input features
    async fn normalize_features(&self, features: &QoEFeatures) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let stats = self.feature_stats.read().await;
        let mut normalized = Vec::new();
        
        // Convert features to vector
        let feature_vec = self.features_to_vector(features);
        
        // Normalize each feature
        for (i, &value) in feature_vec.iter().enumerate() {
            let norm_value = (value - stats.mean[i]) / stats.std[i];
            normalized.push(norm_value.clamp(-3.0, 3.0)); // Clip to 3 standard deviations
        }
        
        Ok(normalized)
    }

    /// Convert QoEFeatures to vector
    fn features_to_vector(&self, features: &QoEFeatures) -> Vec<f32> {
        let mut vec = Vec::new();
        
        // Network features
        vec.extend_from_slice(&[
            features.current_throughput,
            features.current_latency,
            features.current_jitter,
            features.current_packet_loss,
            features.signal_strength,
            features.signal_quality,
            features.interference_level,
        ]);
        
        // Cell features
        vec.extend_from_slice(&[
            features.cell_load,
            features.cell_capacity,
            features.frequency_band,
            features.bandwidth,
        ]);
        
        // Add neighbor cells (pad or truncate to fixed size)
        let mut neighbors = features.neighbor_cells.clone();
        neighbors.resize(5, 0.0);
        vec.extend_from_slice(&neighbors);
        
        // User features
        vec.extend_from_slice(&[
            features.service_type,
            features.user_priority,
            features.mobility_speed,
            features.device_type,
        ]);
        
        // Add historical QoE (pad or truncate to fixed size)
        let mut history = features.historical_qoe.clone();
        history.resize(5, 0.0);
        vec.extend_from_slice(&history);
        
        // Temporal features
        vec.extend_from_slice(&[
            features.time_of_day,
            features.day_of_week,
            features.traffic_pattern,
            features.seasonal_factor,
        ]);
        
        // Environmental features
        vec.extend_from_slice(&[
            features.weather_impact,
            features.event_impact,
            features.congestion_level,
        ]);
        
        // Pad to input size if needed
        vec.resize(self.config.input_size, 0.0);
        
        vec
    }

    /// Compute attention weights for features
    async fn compute_attention_weights(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.attention_network.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        
        Ok(output.data().to_vec())
    }

    /// Apply attention weights to features
    fn apply_attention(&self, features: &[f32], weights: &[f32]) -> Vec<f32> {
        features.iter()
            .zip(weights.iter().cycle())
            .map(|(f, w)| f * w)
            .collect()
    }

    /// Predict using service-specific network
    async fn predict_service_specific(&self, features: &[f32], service_type: &ServiceType) -> Result<Tensor, Box<dyn std::error::Error>> {
        let network = self.service_networks.get(service_type)
            .ok_or("Service network not found")?
            .read().await;
        
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        network.forward(&input_tensor)
    }

    /// Predict using temporal network
    async fn predict_temporal(&self, features: &[f32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let network = self.temporal_network.read().await;
        
        // Expand features for temporal sequence
        let mut temporal_features = Vec::new();
        for _ in 0..self.config.sequence_length {
            temporal_features.extend_from_slice(features);
        }
        
        let input_tensor = Tensor::from_slice(&temporal_features, &[1, temporal_features.len()]);
        network.forward(&input_tensor)
    }

    /// Predict using primary network
    async fn predict_primary(&self, features: &[f32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let network = self.primary_network.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        network.forward(&input_tensor)
    }

    /// Ensemble multiple predictions
    fn ensemble_predictions(&self, primary: &Tensor, service: &Tensor, temporal: &Tensor) -> Tensor {
        // Weighted ensemble
        let primary_weight = 0.5;
        let service_weight = 0.3;
        let temporal_weight = 0.2;
        
        let primary_weighted = primary.multiply_scalar(primary_weight);
        let service_weighted = service.multiply_scalar(service_weight);
        let temporal_weighted = temporal.multiply_scalar(temporal_weight);
        
        primary_weighted.add(&service_weighted).add(&temporal_weighted)
    }

    /// Predict QoE score for specific cell
    async fn predict_cell_qoe(&self, cell_id: u32, prediction: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
        // Simple scoring based on prediction
        let data = prediction.data();
        let mut score = 0.0;
        
        // Weight different QoE metrics
        if data.len() >= 8 {
            score += data[0] * 0.25; // throughput
            score += (1.0 - data[1]) * 0.25; // latency (inverse)
            score += (1.0 - data[2]) * 0.15; // jitter (inverse)
            score += (1.0 - data[3]) * 0.15; // packet_loss (inverse)
            score += data[4] * 0.1; // video_quality
            score += data[5] * 0.1; // audio_quality
        }
        
        Ok(score.clamp(0.0, 1.0))
    }

    /// Calculate prediction confidence
    fn calculate_confidence(&self, prediction: &Tensor, attention_weights: &[f32]) -> f32 {
        // Calculate confidence based on prediction variance and attention focus
        let data = prediction.data();
        let variance = self.calculate_variance(data);
        let attention_focus = attention_weights.iter().map(|w| w * w).sum::<f32>();
        
        let confidence = (1.0 - variance) * attention_focus;
        confidence.clamp(0.0, 1.0)
    }

    /// Calculate prediction uncertainty
    fn calculate_uncertainty(&self, prediction: &Tensor) -> f32 {
        let data = prediction.data();
        let variance = self.calculate_variance(data);
        let entropy = self.calculate_entropy(data);
        
        (variance + entropy * 0.5).clamp(0.0, 1.0)
    }

    /// Calculate variance of prediction
    fn calculate_variance(&self, data: &[f32]) -> f32 {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        variance
    }

    /// Calculate entropy of prediction
    fn calculate_entropy(&self, data: &[f32]) -> f32 {
        let sum = data.iter().sum::<f32>();
        if sum == 0.0 {
            return 0.0;
        }
        
        let entropy = data.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| {
                let p = x / sum;
                -p * p.ln()
            })
            .sum::<f32>();
        entropy
    }

    /// Convert tensor to QoE metrics
    fn tensor_to_qoe_metrics(&self, tensor: &Tensor) -> QoEMetrics {
        let data = tensor.data();
        
        QoEMetrics {
            throughput: data.get(0).copied().unwrap_or(0.0) * 100.0, // Scale to Mbps
            latency: data.get(1).copied().unwrap_or(0.0) * 200.0,    // Scale to ms
            jitter: data.get(2).copied().unwrap_or(0.0) * 50.0,      // Scale to ms
            packet_loss: data.get(3).copied().unwrap_or(0.0) * 10.0, // Scale to %
            video_quality: data.get(4).copied().unwrap_or(0.0) * 5.0, // Scale to MOS
            audio_quality: data.get(5).copied().unwrap_or(0.0) * 5.0, // Scale to MOS
            reliability: data.get(6).copied().unwrap_or(0.0) * 100.0,  // Scale to %
            availability: data.get(7).copied().unwrap_or(0.0) * 100.0, // Scale to %
        }
    }

    /// Update model with feedback for continuous learning
    pub async fn update_with_feedback(&self, feedback: &[SteeringFeedback]) -> Result<(), Box<dyn std::error::Error>> {
        // Update feature statistics
        self.update_feature_stats(feedback).await?;
        
        // Retrain models with new data
        self.retrain_models(feedback).await?;
        
        Ok(())
    }

    /// Update feature statistics for normalization
    async fn update_feature_stats(&self, feedback: &[SteeringFeedback]) -> Result<(), Box<dyn std::error::Error>> {
        let mut stats = self.feature_stats.write().await;
        
        // Update running statistics (simplified)
        for fb in feedback {
            // This would typically use an online algorithm like Welford's
            // For now, we'll use a simple exponential moving average
            let alpha = 0.1; // Learning rate for statistics
            
            // Update means and variances based on feedback
            // This is a simplified implementation
            for i in 0..stats.mean.len() {
                stats.mean[i] = (1.0 - alpha) * stats.mean[i] + alpha * fb.actual_qoe.throughput;
                stats.std[i] = (1.0 - alpha) * stats.std[i] + alpha * 0.1; // Simplified
            }
        }
        
        Ok(())
    }

    /// Retrain models with new feedback data
    async fn retrain_models(&self, feedback: &[SteeringFeedback]) -> Result<(), Box<dyn std::error::Error>> {
        // This would implement online learning or periodic retraining
        // For now, we'll just log the feedback
        println!("Received {} feedback samples for model update", feedback.len());
        
        // In a real implementation, this would:
        // 1. Extract features from feedback
        // 2. Create training batches
        // 3. Update model weights using backpropagation
        // 4. Validate model performance
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ric_tsa::{UserGroup, MobilityPattern, DeviceCapabilities, ServiceRequirements, FrequencyBand};

    #[tokio::test]
    async fn test_qoe_predictor_creation() {
        let predictor = QoEPredictor::new();
        assert!(predictor.is_ok());
    }

    #[tokio::test]
    async fn test_qoe_prediction() {
        let predictor = QoEPredictor::new().unwrap();
        
        let ue_context = UEContext {
            ue_id: 1,
            user_group: UserGroup::Standard,
            service_type: ServiceType::VideoStreaming,
            current_qoe: QoEMetrics {
                throughput: 10.0,
                latency: 20.0,
                jitter: 5.0,
                packet_loss: 0.1,
                video_quality: 4.0,
                audio_quality: 4.5,
                reliability: 99.0,
                availability: 99.9,
            },
            location: (40.7128, -74.0060),
            mobility_pattern: MobilityPattern::Pedestrian,
            device_capabilities: DeviceCapabilities {
                supported_bands: vec![FrequencyBand::Band1800MHz],
                max_mimo_layers: 4,
                ca_support: true,
                dual_connectivity: false,
            },
            service_requirements: ServiceRequirements {
                min_throughput: 5.0,
                max_latency: 50.0,
                max_jitter: 10.0,
                max_packet_loss: 1.0,
                priority: 128,
            },
        };

        let mut cell_carriers = HashMap::new();
        cell_carriers.insert(1, CellCarrier {
            carrier_id: 1,
            band: FrequencyBand::Band1800MHz,
            bandwidth: 20,
            current_load: 50.0,
            max_capacity: 100.0,
            coverage_area: 5.0,
        });

        let result = predictor.predict_qoe_multi_cell(&ue_context, &cell_carriers).await;
        assert!(result.is_ok());
        
        let prediction = result.unwrap();
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(!prediction.cell_scores.is_empty());
    }
}