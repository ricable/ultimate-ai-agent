//! User Classification for QoE-aware Traffic Steering
//! 
//! This module implements machine learning models for classifying users into
//! different service groups based on their behavior, requirements, and context.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::pfs_core::{NeuralNetwork, Layer, Activation, Tensor, TensorOps, DenseLayer};
use super::{UEContext, UserGroup, ServiceType, SteeringFeedback, MobilityPattern};

/// User classification result with confidence scores
#[derive(Debug, Clone)]
pub struct UserClassificationResult {
    pub predicted_group: UserGroup,
    pub group_probabilities: HashMap<UserGroup, f32>,
    pub confidence: f32,
    pub reasoning: Vec<String>,
}

/// User behavior features for classification
#[derive(Debug, Clone)]
pub struct UserBehaviorFeatures {
    // Traffic patterns
    pub avg_throughput: f32,
    pub peak_throughput: f32,
    pub traffic_variance: f32,
    pub session_duration: f32,
    pub data_consumption: f32,
    
    // QoE sensitivity
    pub latency_sensitivity: f32,
    pub jitter_sensitivity: f32,
    pub packet_loss_sensitivity: f32,
    pub quality_preference: f32,
    
    // Service usage patterns
    pub video_usage_ratio: f32,
    pub voice_usage_ratio: f32,
    pub gaming_usage_ratio: f32,
    pub file_transfer_ratio: f32,
    pub web_browsing_ratio: f32,
    
    // Temporal patterns
    pub usage_consistency: f32,
    pub peak_hours: Vec<f32>,
    pub weekend_usage: f32,
    pub seasonal_patterns: f32,
    
    // Mobility patterns
    pub mobility_variance: f32,
    pub handover_frequency: f32,
    pub location_diversity: f32,
    
    // Device characteristics
    pub device_tier: f32,
    pub connectivity_features: f32,
    pub battery_constraints: f32,
    
    // Payment/subscription tier
    pub subscription_tier: f32,
    pub payment_history: f32,
    pub service_level: f32,
}

/// User classifier with multiple ML models
pub struct UserClassifier {
    // Primary classification network
    primary_classifier: Arc<RwLock<NeuralNetwork>>,
    
    // Behavior pattern analyzer
    behavior_analyzer: Arc<RwLock<NeuralNetwork>>,
    
    // Service preference predictor
    service_predictor: Arc<RwLock<NeuralNetwork>>,
    
    // Temporal pattern classifier
    temporal_classifier: Arc<RwLock<NeuralNetwork>>,
    
    // Feature extractors
    feature_extractors: HashMap<String, Arc<RwLock<NeuralNetwork>>>,
    
    // Classification thresholds
    classification_thresholds: Arc<RwLock<ClassificationThresholds>>,
    
    // User history tracking
    user_history: Arc<RwLock<HashMap<u64, UserHistory>>>,
    
    // Model configuration
    config: UserClassifierConfig,
}

/// Configuration for user classifier
#[derive(Debug, Clone)]
pub struct UserClassifierConfig {
    pub feature_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub num_classes: usize,
    pub dropout_rate: f32,
    pub learning_rate: f32,
    pub confidence_threshold: f32,
    pub history_window: usize,
}

/// Classification thresholds for different user groups
#[derive(Debug, Clone)]
pub struct ClassificationThresholds {
    pub premium_threshold: f32,
    pub standard_threshold: f32,
    pub basic_threshold: f32,
    pub iot_threshold: f32,
    pub emergency_threshold: f32,
}

/// User history for temporal analysis
#[derive(Debug, Clone)]
pub struct UserHistory {
    pub classification_history: Vec<(UserGroup, f32)>,
    pub behavior_changes: Vec<BehaviorChange>,
    pub qoe_history: Vec<super::QoEMetrics>,
    pub service_usage: HashMap<ServiceType, f32>,
    pub mobility_history: Vec<MobilityPattern>,
}

/// Detected behavior changes
#[derive(Debug, Clone)]
pub struct BehaviorChange {
    pub timestamp: std::time::Instant,
    pub change_type: BehaviorChangeType,
    pub magnitude: f32,
    pub description: String,
}

/// Types of behavior changes
#[derive(Debug, Clone)]
pub enum BehaviorChangeType {
    ServiceUsageShift,
    QoERequirementChange,
    MobilityPatternChange,
    DeviceUpgrade,
    SubscriptionChange,
}

impl Default for UserClassifierConfig {
    fn default() -> Self {
        Self {
            feature_size: 32,
            hidden_sizes: vec![128, 64, 32],
            num_classes: 5, // Number of user groups
            dropout_rate: 0.1,
            learning_rate: 0.001,
            confidence_threshold: 0.7,
            history_window: 100,
        }
    }
}

impl Default for ClassificationThresholds {
    fn default() -> Self {
        Self {
            premium_threshold: 0.8,
            standard_threshold: 0.6,
            basic_threshold: 0.4,
            iot_threshold: 0.3,
            emergency_threshold: 0.9,
        }
    }
}

impl UserClassifier {
    /// Create a new user classifier
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = UserClassifierConfig::default();
        Self::new_with_config(config)
    }

    /// Create user classifier with custom configuration
    pub fn new_with_config(config: UserClassifierConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Create primary classifier
        let primary_classifier = Arc::new(RwLock::new(Self::build_primary_classifier(&config)?));
        
        // Create behavior analyzer
        let behavior_analyzer = Arc::new(RwLock::new(Self::build_behavior_analyzer(&config)?));
        
        // Create service predictor
        let service_predictor = Arc::new(RwLock::new(Self::build_service_predictor(&config)?));
        
        // Create temporal classifier
        let temporal_classifier = Arc::new(RwLock::new(Self::build_temporal_classifier(&config)?));
        
        // Create feature extractors
        let feature_extractors = Self::build_feature_extractors(&config)?;
        
        // Initialize thresholds
        let classification_thresholds = Arc::new(RwLock::new(ClassificationThresholds::default()));
        
        // Initialize user history
        let user_history = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            primary_classifier,
            behavior_analyzer,
            service_predictor,
            temporal_classifier,
            feature_extractors,
            classification_thresholds,
            user_history,
            config,
        })
    }

    /// Build primary classification network
    fn build_primary_classifier(config: &UserClassifierConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        let mut prev_size = config.feature_size;
        
        // Hidden layers
        for &hidden_size in &config.hidden_sizes {
            network.add_layer(Box::new(DenseLayer::new(prev_size, hidden_size)));
            network.add_layer(Box::new(Activation::ReLU));
            prev_size = hidden_size;
        }
        
        // Output layer with softmax for classification
        network.add_layer(Box::new(DenseLayer::new(prev_size, config.num_classes)));
        network.add_layer(Box::new(Activation::Softmax));
        
        Ok(network)
    }

    /// Build behavior pattern analyzer
    fn build_behavior_analyzer(config: &UserClassifierConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        // Autoencoder-like architecture for behavior pattern detection
        network.add_layer(Box::new(DenseLayer::new(config.feature_size, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(64, 32)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(32, 16))); // Bottleneck
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(16, 32)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(32, config.num_classes)));
        network.add_layer(Box::new(Activation::Softmax));
        
        Ok(network)
    }

    /// Build service preference predictor
    fn build_service_predictor(config: &UserClassifierConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        // Network optimized for service prediction
        network.add_layer(Box::new(DenseLayer::new(config.feature_size, 128)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(128, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(64, 8))); // Service types
        network.add_layer(Box::new(Activation::Softmax));
        
        Ok(network)
    }

    /// Build temporal pattern classifier
    fn build_temporal_classifier(config: &UserClassifierConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        // LSTM-like architecture for temporal patterns
        let temporal_input_size = config.feature_size * 10; // 10 time steps
        network.add_layer(Box::new(DenseLayer::new(temporal_input_size, 128)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(128, 64)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(64, config.num_classes)));
        network.add_layer(Box::new(Activation::Softmax));
        
        Ok(network)
    }

    /// Build feature extractors for different aspects
    fn build_feature_extractors(config: &UserClassifierConfig) -> Result<HashMap<String, Arc<RwLock<NeuralNetwork>>>, Box<dyn std::error::Error>> {
        let mut extractors = HashMap::new();
        
        // Traffic pattern extractor
        let mut traffic_extractor = NeuralNetwork::new();
        traffic_extractor.add_layer(Box::new(DenseLayer::new(10, 32)));
        traffic_extractor.add_layer(Box::new(Activation::ReLU));
        traffic_extractor.add_layer(Box::new(DenseLayer::new(32, 16)));
        traffic_extractor.add_layer(Box::new(Activation::ReLU));
        extractors.insert("traffic".to_string(), Arc::new(RwLock::new(traffic_extractor)));
        
        // QoE sensitivity extractor
        let mut qoe_extractor = NeuralNetwork::new();
        qoe_extractor.add_layer(Box::new(DenseLayer::new(8, 16)));
        qoe_extractor.add_layer(Box::new(Activation::ReLU));
        qoe_extractor.add_layer(Box::new(DenseLayer::new(16, 8)));
        qoe_extractor.add_layer(Box::new(Activation::ReLU));
        extractors.insert("qoe".to_string(), Arc::new(RwLock::new(qoe_extractor)));
        
        // Mobility pattern extractor
        let mut mobility_extractor = NeuralNetwork::new();
        mobility_extractor.add_layer(Box::new(DenseLayer::new(6, 12)));
        mobility_extractor.add_layer(Box::new(Activation::ReLU));
        mobility_extractor.add_layer(Box::new(DenseLayer::new(12, 8)));
        mobility_extractor.add_layer(Box::new(Activation::ReLU));
        extractors.insert("mobility".to_string(), Arc::new(RwLock::new(mobility_extractor)));
        
        Ok(extractors)
    }

    /// Extract user behavior features from context and history
    pub async fn extract_user_features(&self, ue_context: &UEContext) -> Result<UserBehaviorFeatures, Box<dyn std::error::Error>> {
        let history = self.user_history.read().await;
        let user_hist = history.get(&ue_context.ue_id);
        
        // Extract traffic patterns
        let (avg_throughput, peak_throughput, traffic_variance) = self.extract_traffic_patterns(ue_context, user_hist);
        
        // Extract QoE sensitivity
        let (latency_sensitivity, jitter_sensitivity, packet_loss_sensitivity, quality_preference) = 
            self.extract_qoe_sensitivity(ue_context, user_hist);
        
        // Extract service usage patterns
        let service_ratios = self.extract_service_usage_patterns(ue_context, user_hist);
        
        // Extract temporal patterns
        let (usage_consistency, peak_hours, weekend_usage, seasonal_patterns) = 
            self.extract_temporal_patterns(ue_context, user_hist);
        
        // Extract mobility patterns
        let (mobility_variance, handover_frequency, location_diversity) = 
            self.extract_mobility_patterns(ue_context, user_hist);
        
        // Extract device characteristics
        let (device_tier, connectivity_features, battery_constraints) = 
            self.extract_device_characteristics(ue_context);
        
        // Extract subscription information
        let (subscription_tier, payment_history, service_level) = 
            self.extract_subscription_info(ue_context, user_hist);

        Ok(UserBehaviorFeatures {
            avg_throughput,
            peak_throughput,
            traffic_variance,
            session_duration: 30.0, // Placeholder
            data_consumption: 1000.0, // Placeholder
            
            latency_sensitivity,
            jitter_sensitivity,
            packet_loss_sensitivity,
            quality_preference,
            
            video_usage_ratio: service_ratios.0,
            voice_usage_ratio: service_ratios.1,
            gaming_usage_ratio: service_ratios.2,
            file_transfer_ratio: service_ratios.3,
            web_browsing_ratio: service_ratios.4,
            
            usage_consistency,
            peak_hours,
            weekend_usage,
            seasonal_patterns,
            
            mobility_variance,
            handover_frequency,
            location_diversity,
            
            device_tier,
            connectivity_features,
            battery_constraints,
            
            subscription_tier,
            payment_history,
            service_level,
        })
    }

    /// Extract traffic patterns from user data
    fn extract_traffic_patterns(&self, ue_context: &UEContext, user_hist: Option<&UserHistory>) -> (f32, f32, f32) {
        let current_throughput = ue_context.current_qoe.throughput;
        
        if let Some(hist) = user_hist {
            let throughputs: Vec<f32> = hist.qoe_history.iter().map(|q| q.throughput).collect();
            let avg_throughput = throughputs.iter().sum::<f32>() / throughputs.len() as f32;
            let peak_throughput = throughputs.iter().fold(0.0, |a, &b| a.max(b));
            
            // Calculate variance
            let mean = avg_throughput;
            let variance = throughputs.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / throughputs.len() as f32;
            
            (avg_throughput, peak_throughput, variance)
        } else {
            (current_throughput, current_throughput, 0.0)
        }
    }

    /// Extract QoE sensitivity patterns
    fn extract_qoe_sensitivity(&self, ue_context: &UEContext, user_hist: Option<&UserHistory>) -> (f32, f32, f32, f32) {
        // Analyze how sensitive the user is to different QoE parameters
        let service_sensitivity = match ue_context.service_type {
            ServiceType::VideoStreaming => (0.7, 0.8, 0.6, 0.9), // High latency and quality sensitivity
            ServiceType::VoiceCall => (0.9, 0.9, 0.7, 0.8),      // High latency and jitter sensitivity
            ServiceType::Gaming => (0.95, 0.9, 0.8, 0.7),        // Very high latency sensitivity
            ServiceType::FileTransfer => (0.3, 0.2, 0.5, 0.4),   // Low sensitivity to latency
            ServiceType::WebBrowsing => (0.5, 0.3, 0.4, 0.6),    // Moderate sensitivity
            ServiceType::IoTSensor => (0.8, 0.4, 0.9, 0.3),      // High reliability sensitivity
            ServiceType::Emergency => (0.95, 0.9, 0.95, 0.9),    // High all-around sensitivity
            ServiceType::AR_VR => (0.95, 0.95, 0.9, 0.95),       // Very high all-around sensitivity
        };
        
        // Apply user group modifiers
        let group_modifier = match ue_context.user_group {
            UserGroup::Premium => 1.2,
            UserGroup::Standard => 1.0,
            UserGroup::Basic => 0.8,
            UserGroup::IoT => 0.9,
            UserGroup::Emergency => 1.3,
        };
        
        (
            service_sensitivity.0 * group_modifier,
            service_sensitivity.1 * group_modifier,
            service_sensitivity.2 * group_modifier,
            service_sensitivity.3 * group_modifier,
        )
    }

    /// Extract service usage patterns
    fn extract_service_usage_patterns(&self, ue_context: &UEContext, user_hist: Option<&UserHistory>) -> (f32, f32, f32, f32, f32) {
        if let Some(hist) = user_hist {
            let total_usage: f32 = hist.service_usage.values().sum();
            if total_usage > 0.0 {
                let video_ratio = hist.service_usage.get(&ServiceType::VideoStreaming).unwrap_or(&0.0) / total_usage;
                let voice_ratio = hist.service_usage.get(&ServiceType::VoiceCall).unwrap_or(&0.0) / total_usage;
                let gaming_ratio = hist.service_usage.get(&ServiceType::Gaming).unwrap_or(&0.0) / total_usage;
                let file_ratio = hist.service_usage.get(&ServiceType::FileTransfer).unwrap_or(&0.0) / total_usage;
                let web_ratio = hist.service_usage.get(&ServiceType::WebBrowsing).unwrap_or(&0.0) / total_usage;
                
                (video_ratio, voice_ratio, gaming_ratio, file_ratio, web_ratio)
            } else {
                (0.2, 0.2, 0.2, 0.2, 0.2) // Default equal distribution
            }
        } else {
            // Infer from current service type
            match ue_context.service_type {
                ServiceType::VideoStreaming => (0.8, 0.1, 0.05, 0.025, 0.025),
                ServiceType::VoiceCall => (0.1, 0.8, 0.05, 0.025, 0.025),
                ServiceType::Gaming => (0.05, 0.1, 0.8, 0.025, 0.025),
                ServiceType::FileTransfer => (0.025, 0.025, 0.05, 0.8, 0.1),
                ServiceType::WebBrowsing => (0.2, 0.1, 0.1, 0.1, 0.5),
                _ => (0.2, 0.2, 0.2, 0.2, 0.2),
            }
        }
    }

    /// Extract temporal usage patterns
    fn extract_temporal_patterns(&self, _ue_context: &UEContext, _user_hist: Option<&UserHistory>) -> (f32, Vec<f32>, f32, f32) {
        // Placeholder implementation
        let usage_consistency = 0.7;
        let peak_hours = vec![0.1, 0.2, 0.8, 0.9, 0.7, 0.6]; // 6 time slots
        let weekend_usage = 0.8;
        let seasonal_patterns = 0.5;
        
        (usage_consistency, peak_hours, weekend_usage, seasonal_patterns)
    }

    /// Extract mobility patterns
    fn extract_mobility_patterns(&self, ue_context: &UEContext, user_hist: Option<&UserHistory>) -> (f32, f32, f32) {
        let mobility_variance = match ue_context.mobility_pattern {
            MobilityPattern::Stationary => 0.1,
            MobilityPattern::Pedestrian => 0.3,
            MobilityPattern::Vehicular => 0.7,
            MobilityPattern::HighSpeed => 0.9,
        };
        
        let handover_frequency = mobility_variance * 0.8; // Correlated with mobility
        let location_diversity = mobility_variance * 0.9; // Correlated with mobility
        
        (mobility_variance, handover_frequency, location_diversity)
    }

    /// Extract device characteristics
    fn extract_device_characteristics(&self, ue_context: &UEContext) -> (f32, f32, f32) {
        let device_tier = if ue_context.device_capabilities.max_mimo_layers >= 4 
            && ue_context.device_capabilities.ca_support 
            && ue_context.device_capabilities.dual_connectivity {
            0.9 // High-end device
        } else if ue_context.device_capabilities.max_mimo_layers >= 2 
            && ue_context.device_capabilities.ca_support {
            0.7 // Mid-range device
        } else {
            0.4 // Basic device
        };
        
        let connectivity_features = if ue_context.device_capabilities.ca_support { 0.8 } else { 0.4 };
        let battery_constraints = 0.5; // Placeholder
        
        (device_tier, connectivity_features, battery_constraints)
    }

    /// Extract subscription information
    fn extract_subscription_info(&self, ue_context: &UEContext, _user_hist: Option<&UserHistory>) -> (f32, f32, f32) {
        let subscription_tier = match ue_context.user_group {
            UserGroup::Premium => 0.9,
            UserGroup::Standard => 0.7,
            UserGroup::Basic => 0.4,
            UserGroup::IoT => 0.3,
            UserGroup::Emergency => 1.0,
        };
        
        let payment_history = 0.8; // Placeholder
        let service_level = subscription_tier * 0.9;
        
        (subscription_tier, payment_history, service_level)
    }

    /// Classify user based on behavior features
    pub async fn classify_user(&self, ue_context: &UEContext) -> Result<UserClassificationResult, Box<dyn std::error::Error>> {
        // Extract features
        let features = self.extract_user_features(ue_context).await?;
        
        // Convert to feature vector
        let feature_vector = self.features_to_vector(&features);
        
        // Get classifications from different models
        let primary_result = self.classify_primary(&feature_vector).await?;
        let behavior_result = self.classify_behavior(&feature_vector).await?;
        let service_result = self.classify_service(&feature_vector).await?;
        
        // Ensemble the results
        let ensemble_result = self.ensemble_classifications(&primary_result, &behavior_result, &service_result);
        
        // Determine final classification
        let predicted_group = self.determine_final_group(&ensemble_result);
        
        // Calculate confidence
        let confidence = self.calculate_classification_confidence(&ensemble_result);
        
        // Generate reasoning
        let reasoning = self.generate_reasoning(&features, &predicted_group);
        
        // Update user history
        self.update_user_history(ue_context.ue_id, &predicted_group, confidence).await?;
        
        Ok(UserClassificationResult {
            predicted_group,
            group_probabilities: ensemble_result,
            confidence,
            reasoning,
        })
    }

    /// Convert features to vector format
    fn features_to_vector(&self, features: &UserBehaviorFeatures) -> Vec<f32> {
        let mut vec = Vec::new();
        
        // Traffic patterns
        vec.extend_from_slice(&[
            features.avg_throughput,
            features.peak_throughput,
            features.traffic_variance,
            features.session_duration,
            features.data_consumption,
        ]);
        
        // QoE sensitivity
        vec.extend_from_slice(&[
            features.latency_sensitivity,
            features.jitter_sensitivity,
            features.packet_loss_sensitivity,
            features.quality_preference,
        ]);
        
        // Service usage patterns
        vec.extend_from_slice(&[
            features.video_usage_ratio,
            features.voice_usage_ratio,
            features.gaming_usage_ratio,
            features.file_transfer_ratio,
            features.web_browsing_ratio,
        ]);
        
        // Temporal patterns
        vec.push(features.usage_consistency);
        vec.extend_from_slice(&features.peak_hours);
        vec.extend_from_slice(&[features.weekend_usage, features.seasonal_patterns]);
        
        // Mobility patterns
        vec.extend_from_slice(&[
            features.mobility_variance,
            features.handover_frequency,
            features.location_diversity,
        ]);
        
        // Device characteristics
        vec.extend_from_slice(&[
            features.device_tier,
            features.connectivity_features,
            features.battery_constraints,
        ]);
        
        // Subscription information
        vec.extend_from_slice(&[
            features.subscription_tier,
            features.payment_history,
            features.service_level,
        ]);
        
        // Pad to feature size
        vec.resize(self.config.feature_size, 0.0);
        
        vec
    }

    /// Classify using primary network
    async fn classify_primary(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.primary_classifier.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Classify using behavior analyzer
    async fn classify_behavior(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.behavior_analyzer.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Classify using service predictor
    async fn classify_service(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.service_predictor.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        
        // Convert service prediction to user group probabilities
        let service_probs = output.data();
        let mut group_probs = vec![0.0; 5]; // 5 user groups
        
        // Map service preferences to user groups
        if service_probs.len() >= 8 {
            group_probs[0] = service_probs[7] * 0.8; // AR/VR -> Premium
            group_probs[1] = service_probs[4] * 0.6; // Web -> Standard
            group_probs[2] = service_probs[3] * 0.7; // File -> Basic
            group_probs[3] = service_probs[5] * 0.9; // IoT -> IoT
            group_probs[4] = service_probs[6] * 0.9; // Emergency -> Emergency
        }
        
        Ok(group_probs)
    }

    /// Ensemble multiple classification results
    fn ensemble_classifications(&self, primary: &[f32], behavior: &[f32], service: &[f32]) -> HashMap<UserGroup, f32> {
        let mut result = HashMap::new();
        
        let user_groups = [
            UserGroup::Premium,
            UserGroup::Standard,
            UserGroup::Basic,
            UserGroup::IoT,
            UserGroup::Emergency,
        ];
        
        // Weighted ensemble
        let weights = [0.5, 0.3, 0.2]; // primary, behavior, service
        
        for (i, group) in user_groups.iter().enumerate() {
            let prob = weights[0] * primary.get(i).unwrap_or(&0.0)
                + weights[1] * behavior.get(i).unwrap_or(&0.0)
                + weights[2] * service.get(i).unwrap_or(&0.0);
            result.insert(group.clone(), prob);
        }
        
        result
    }

    /// Determine final user group from probabilities
    fn determine_final_group(&self, probabilities: &HashMap<UserGroup, f32>) -> UserGroup {
        probabilities.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(group, _)| group.clone())
            .unwrap_or(UserGroup::Standard)
    }

    /// Calculate classification confidence
    fn calculate_classification_confidence(&self, probabilities: &HashMap<UserGroup, f32>) -> f32 {
        let values: Vec<f32> = probabilities.values().cloned().collect();
        let max_prob = values.iter().fold(0.0, |a, &b| a.max(b));
        let second_max = values.iter().filter(|&&x| x < max_prob).fold(0.0, |a, &b| a.max(b));
        
        // Confidence based on margin between top two predictions
        (max_prob - second_max).clamp(0.0, 1.0)
    }

    /// Generate reasoning for classification
    fn generate_reasoning(&self, features: &UserBehaviorFeatures, group: &UserGroup) -> Vec<String> {
        let mut reasoning = Vec::new();
        
        match group {
            UserGroup::Premium => {
                if features.subscription_tier > 0.8 {
                    reasoning.push("High subscription tier indicates premium user".to_string());
                }
                if features.quality_preference > 0.8 {
                    reasoning.push("High quality preference typical of premium users".to_string());
                }
                if features.device_tier > 0.8 {
                    reasoning.push("High-end device suggests premium user".to_string());
                }
            }
            UserGroup::Standard => {
                reasoning.push("Balanced usage patterns indicate standard user".to_string());
            }
            UserGroup::Basic => {
                if features.subscription_tier < 0.5 {
                    reasoning.push("Low subscription tier indicates basic user".to_string());
                }
                if features.device_tier < 0.5 {
                    reasoning.push("Basic device suggests basic user tier".to_string());
                }
            }
            UserGroup::IoT => {
                if features.data_consumption < 100.0 {
                    reasoning.push("Low data consumption typical of IoT devices".to_string());
                }
                if features.mobility_variance < 0.2 {
                    reasoning.push("Low mobility typical of IoT sensors".to_string());
                }
            }
            UserGroup::Emergency => {
                reasoning.push("High reliability requirements indicate emergency service".to_string());
            }
        }
        
        reasoning
    }

    /// Update user history with new classification
    async fn update_user_history(&self, ue_id: u64, group: &UserGroup, confidence: f32) -> Result<(), Box<dyn std::error::Error>> {
        let mut history = self.user_history.write().await;
        
        let user_hist = history.entry(ue_id).or_insert_with(|| UserHistory {
            classification_history: Vec::new(),
            behavior_changes: Vec::new(),
            qoe_history: Vec::new(),
            service_usage: HashMap::new(),
            mobility_history: Vec::new(),
        });
        
        user_hist.classification_history.push((group.clone(), confidence));
        
        // Keep only recent history
        if user_hist.classification_history.len() > self.config.history_window {
            user_hist.classification_history.remove(0);
        }
        
        Ok(())
    }

    /// Update with feedback for continuous learning
    pub async fn update_with_feedback(&self, feedback: &[SteeringFeedback]) -> Result<(), Box<dyn std::error::Error>> {
        // Update user histories with feedback
        for fb in feedback {
            let mut history = self.user_history.write().await;
            if let Some(user_hist) = history.get_mut(&fb.ue_id) {
                user_hist.qoe_history.push(fb.actual_qoe.clone());
                
                // Keep only recent history
                if user_hist.qoe_history.len() > self.config.history_window {
                    user_hist.qoe_history.remove(0);
                }
            }
        }
        
        // This would typically include model retraining
        println!("Updated user classification models with {} feedback samples", feedback.len());
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ric_tsa::{DeviceCapabilities, ServiceRequirements, FrequencyBand, QoEMetrics, MobilityPattern};

    #[tokio::test]
    async fn test_user_classifier_creation() {
        let classifier = UserClassifier::new();
        assert!(classifier.is_ok());
    }

    #[tokio::test]
    async fn test_user_classification() {
        let classifier = UserClassifier::new().unwrap();
        
        let ue_context = UEContext {
            ue_id: 1,
            user_group: UserGroup::Standard,
            service_type: ServiceType::VideoStreaming,
            current_qoe: QoEMetrics {
                throughput: 25.0,
                latency: 15.0,
                jitter: 3.0,
                packet_loss: 0.05,
                video_quality: 4.2,
                audio_quality: 4.5,
                reliability: 99.5,
                availability: 99.9,
            },
            location: (40.7128, -74.0060),
            mobility_pattern: MobilityPattern::Pedestrian,
            device_capabilities: DeviceCapabilities {
                supported_bands: vec![FrequencyBand::Band1800MHz],
                max_mimo_layers: 4,
                ca_support: true,
                dual_connectivity: true,
            },
            service_requirements: ServiceRequirements {
                min_throughput: 10.0,
                max_latency: 30.0,
                max_jitter: 8.0,
                max_packet_loss: 0.5,
                priority: 150,
            },
        };

        let result = classifier.classify_user(&ue_context).await;
        assert!(result.is_ok());
        
        let classification = result.unwrap();
        assert!(classification.confidence >= 0.0 && classification.confidence <= 1.0);
        assert!(!classification.group_probabilities.is_empty());
        assert!(!classification.reasoning.is_empty());
    }
}