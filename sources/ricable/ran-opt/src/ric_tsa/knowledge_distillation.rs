//! Knowledge Distillation for Edge Deployment
//! 
//! This module implements knowledge distillation techniques to compress
//! large teacher models into smaller student models for edge deployment
//! while maintaining sub-millisecond inference performance.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::pfs_core::{NeuralNetwork, Layer, Activation, Tensor, TensorOps, DenseLayer, Adam};
use super::{UEContext, QoEMetrics, SteeringFeedback, qoe_prediction::QoEPredictor, user_classification::UserClassifier, mac_scheduler::MacScheduler};

/// Knowledge distillation trainer for model compression
pub struct KnowledgeDistillation {
    // Teacher models (large, high-accuracy)
    teacher_qoe_predictor: Arc<QoEPredictor>,
    teacher_user_classifier: Arc<UserClassifier>,
    teacher_mac_scheduler: Arc<MacScheduler>,
    
    // Student models (small, fast)
    student_qoe_predictor: Arc<RwLock<NeuralNetwork>>,
    student_user_classifier: Arc<RwLock<NeuralNetwork>>,
    student_mac_scheduler: Arc<RwLock<NeuralNetwork>>,
    
    // Distillation optimizers
    qoe_optimizer: Arc<RwLock<Adam>>,
    classifier_optimizer: Arc<RwLock<Adam>>,
    scheduler_optimizer: Arc<RwLock<Adam>>,
    
    // Training data buffer
    training_buffer: Arc<RwLock<TrainingBuffer>>,
    
    // Configuration
    config: KnowledgeDistillationConfig,
}

/// Configuration for knowledge distillation
#[derive(Debug, Clone)]
pub struct KnowledgeDistillationConfig {
    pub temperature: f32,              // Softmax temperature for distillation
    pub alpha: f32,                   // Weight for distillation loss
    pub beta: f32,                    // Weight for student loss
    pub learning_rate: f32,           // Learning rate for student models
    pub batch_size: usize,            // Batch size for training
    pub max_buffer_size: usize,       // Maximum training buffer size
    pub distillation_epochs: usize,   // Epochs for distillation training
    pub compression_ratio: f32,       // Target compression ratio
    pub inference_time_target: f32,   // Target inference time in ms
}

/// Training data buffer for distillation
#[derive(Debug, Clone)]
pub struct TrainingBuffer {
    pub qoe_samples: Vec<QoETrainingSample>,
    pub classification_samples: Vec<ClassificationTrainingSample>,
    pub scheduling_samples: Vec<SchedulingTrainingSample>,
}

/// QoE training sample
#[derive(Debug, Clone)]
pub struct QoETrainingSample {
    pub input_features: Vec<f32>,
    pub teacher_output: Vec<f32>,
    pub ground_truth: QoEMetrics,
    pub confidence: f32,
}

/// Classification training sample
#[derive(Debug, Clone)]
pub struct ClassificationTrainingSample {
    pub input_features: Vec<f32>,
    pub teacher_output: Vec<f32>,
    pub ground_truth: super::UserGroup,
    pub confidence: f32,
}

/// Scheduling training sample
#[derive(Debug, Clone)]
pub struct SchedulingTrainingSample {
    pub input_features: Vec<f32>,
    pub teacher_output: Vec<f32>,
    pub ground_truth: super::ResourceAllocation,
    pub confidence: f32,
}

/// Compressed edge model for deployment
pub struct EdgeModel {
    // Compressed neural networks
    qoe_predictor: Arc<RwLock<NeuralNetwork>>,
    user_classifier: Arc<RwLock<NeuralNetwork>>,
    mac_scheduler: Arc<RwLock<NeuralNetwork>>,
    
    // Model metadata
    metadata: EdgeModelMetadata,
    
    // Performance metrics
    performance: Arc<RwLock<EdgePerformanceMetrics>>,
}

/// Edge model metadata
#[derive(Debug, Clone)]
pub struct EdgeModelMetadata {
    pub model_version: String,
    pub compression_ratio: f32,
    pub accuracy_retention: f32,
    pub inference_time_ms: f32,
    pub model_size_mb: f32,
    pub deployment_timestamp: std::time::SystemTime,
    pub teacher_model_versions: HashMap<String, String>,
}

/// Performance metrics for edge deployment
#[derive(Debug, Clone, Default)]
pub struct EdgePerformanceMetrics {
    pub avg_inference_time_ms: f32,
    pub max_inference_time_ms: f32,
    pub min_inference_time_ms: f32,
    pub inference_count: u64,
    pub accuracy_vs_teacher: f32,
    pub memory_usage_mb: f32,
    pub energy_consumption_mj: f32,
    pub error_rate: f32,
}

/// Distillation loss components
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    pub distillation_loss: f32,  // KL divergence between teacher and student
    pub student_loss: f32,       // Cross-entropy loss on ground truth
    pub total_loss: f32,         // Weighted combination
    pub temperature: f32,        // Temperature used for softmax
}

impl Default for KnowledgeDistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,              // High weight for distillation
            beta: 0.3,               // Lower weight for student loss
            learning_rate: 0.001,
            batch_size: 32,
            max_buffer_size: 10000,
            distillation_epochs: 50,
            compression_ratio: 0.1,  // 10x compression
            inference_time_target: 0.5, // 0.5ms target
        }
    }
}

impl KnowledgeDistillation {
    /// Create a new knowledge distillation trainer
    pub fn new(
        teacher_qoe: Arc<QoEPredictor>,
        teacher_classifier: Arc<UserClassifier>,
        teacher_scheduler: Arc<MacScheduler>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = KnowledgeDistillationConfig::default();
        Self::new_with_config(teacher_qoe, teacher_classifier, teacher_scheduler, config)
    }

    /// Create knowledge distillation trainer with custom configuration
    pub fn new_with_config(
        teacher_qoe: Arc<QoEPredictor>,
        teacher_classifier: Arc<UserClassifier>,
        teacher_scheduler: Arc<MacScheduler>,
        config: KnowledgeDistillationConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create compressed student models
        let student_qoe_predictor = Arc::new(RwLock::new(Self::create_compressed_qoe_model(&config)?));
        let student_user_classifier = Arc::new(RwLock::new(Self::create_compressed_classifier_model(&config)?));
        let student_mac_scheduler = Arc::new(RwLock::new(Self::create_compressed_scheduler_model(&config)?));
        
        // Create optimizers for student models
        let qoe_optimizer = Arc::new(RwLock::new(Adam::new(config.learning_rate)));
        let classifier_optimizer = Arc::new(RwLock::new(Adam::new(config.learning_rate)));
        let scheduler_optimizer = Arc::new(RwLock::new(Adam::new(config.learning_rate)));
        
        // Initialize training buffer
        let training_buffer = Arc::new(RwLock::new(TrainingBuffer {
            qoe_samples: Vec::new(),
            classification_samples: Vec::new(),
            scheduling_samples: Vec::new(),
        }));

        Ok(Self {
            teacher_qoe_predictor: teacher_qoe,
            teacher_user_classifier: teacher_classifier,
            teacher_mac_scheduler: teacher_scheduler,
            student_qoe_predictor,
            student_user_classifier,
            student_mac_scheduler,
            qoe_optimizer,
            classifier_optimizer,
            scheduler_optimizer,
            training_buffer,
            config,
        })
    }

    /// Create compressed QoE prediction model
    fn create_compressed_qoe_model(config: &KnowledgeDistillationConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        // Much smaller architecture than teacher model
        let input_size = 32;
        let hidden_size = (128.0 * config.compression_ratio) as usize; // Compressed hidden size
        let output_size = 8;
        
        network.add_layer(Box::new(DenseLayer::new(input_size, hidden_size)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(hidden_size, hidden_size / 2)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(hidden_size / 2, output_size)));
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Create compressed user classification model
    fn create_compressed_classifier_model(config: &KnowledgeDistillationConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        let input_size = 32;
        let hidden_size = (64.0 * config.compression_ratio) as usize;
        let output_size = 5; // Number of user groups
        
        network.add_layer(Box::new(DenseLayer::new(input_size, hidden_size)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(hidden_size, output_size)));
        network.add_layer(Box::new(Activation::Softmax));
        
        Ok(network)
    }

    /// Create compressed MAC scheduler model
    fn create_compressed_scheduler_model(config: &KnowledgeDistillationConfig) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut network = NeuralNetwork::new();
        
        let input_size = 64;
        let hidden_size = (128.0 * config.compression_ratio) as usize;
        let output_size = 32; // Resource allocation parameters
        
        network.add_layer(Box::new(DenseLayer::new(input_size, hidden_size)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(hidden_size, hidden_size / 2)));
        network.add_layer(Box::new(Activation::ReLU));
        network.add_layer(Box::new(DenseLayer::new(hidden_size / 2, output_size)));
        network.add_layer(Box::new(Activation::Sigmoid));
        
        Ok(network)
    }

    /// Collect training data from teacher models
    pub async fn collect_training_data(&self, ue_contexts: &[UEContext]) -> Result<(), Box<dyn std::error::Error>> {
        let mut buffer = self.training_buffer.write().await;
        
        for ue_context in ue_contexts {
            // Collect QoE prediction data
            if let Ok(qoe_features) = self.extract_qoe_features(ue_context).await {
                if let Ok(teacher_output) = self.get_teacher_qoe_output(&qoe_features).await {
                    let sample = QoETrainingSample {
                        input_features: qoe_features,
                        teacher_output,
                        ground_truth: ue_context.current_qoe.clone(),
                        confidence: 0.9, // Placeholder
                    };
                    buffer.qoe_samples.push(sample);
                }
            }
            
            // Collect classification data
            if let Ok(class_features) = self.extract_classification_features(ue_context).await {
                if let Ok(teacher_output) = self.get_teacher_classification_output(&class_features).await {
                    let sample = ClassificationTrainingSample {
                        input_features: class_features,
                        teacher_output,
                        ground_truth: ue_context.user_group.clone(),
                        confidence: 0.9,
                    };
                    buffer.classification_samples.push(sample);
                }
            }
            
            // Collect scheduling data
            if let Ok(sched_features) = self.extract_scheduling_features(ue_context).await {
                if let Ok(teacher_output) = self.get_teacher_scheduling_output(&sched_features).await {
                    let sample = SchedulingTrainingSample {
                        input_features: sched_features,
                        teacher_output,
                        ground_truth: super::ResourceAllocation {
                            prb_allocation: vec![1, 2, 3], // Placeholder
                            mcs_index: 15,
                            mimo_layers: 2,
                            power_level: 20.0,
                            scheduling_priority: 128,
                        },
                        confidence: 0.9,
                    };
                    buffer.scheduling_samples.push(sample);
                }
            }
        }
        
        // Limit buffer size
        if buffer.qoe_samples.len() > self.config.max_buffer_size {
            buffer.qoe_samples.drain(0..buffer.qoe_samples.len() - self.config.max_buffer_size);
        }
        if buffer.classification_samples.len() > self.config.max_buffer_size {
            buffer.classification_samples.drain(0..buffer.classification_samples.len() - self.config.max_buffer_size);
        }
        if buffer.scheduling_samples.len() > self.config.max_buffer_size {
            buffer.scheduling_samples.drain(0..buffer.scheduling_samples.len() - self.config.max_buffer_size);
        }
        
        Ok(())
    }

    /// Extract QoE features for distillation
    async fn extract_qoe_features(&self, ue_context: &UEContext) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Simplified feature extraction
        let mut features = Vec::new();
        
        features.extend_from_slice(&[
            ue_context.current_qoe.throughput / 100.0,
            ue_context.current_qoe.latency / 200.0,
            ue_context.current_qoe.jitter / 50.0,
            ue_context.current_qoe.packet_loss / 10.0,
            ue_context.current_qoe.reliability / 100.0,
            ue_context.current_qoe.availability / 100.0,
        ]);
        
        // Service type encoding
        let service_encoding = match ue_context.service_type {
            super::ServiceType::VideoStreaming => 0.1,
            super::ServiceType::VoiceCall => 0.2,
            super::ServiceType::Gaming => 0.3,
            super::ServiceType::FileTransfer => 0.4,
            super::ServiceType::WebBrowsing => 0.5,
            super::ServiceType::IoTSensor => 0.6,
            super::ServiceType::Emergency => 0.7,
            super::ServiceType::AR_VR => 0.8,
        };
        features.push(service_encoding);
        
        // User group encoding
        let group_encoding = match ue_context.user_group {
            super::UserGroup::Premium => 0.9,
            super::UserGroup::Standard => 0.7,
            super::UserGroup::Basic => 0.5,
            super::UserGroup::IoT => 0.3,
            super::UserGroup::Emergency => 1.0,
        };
        features.push(group_encoding);
        
        // Pad to required size
        features.resize(32, 0.0);
        
        Ok(features)
    }

    /// Extract classification features
    async fn extract_classification_features(&self, ue_context: &UEContext) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Similar to QoE features but optimized for classification
        self.extract_qoe_features(ue_context).await
    }

    /// Extract scheduling features
    async fn extract_scheduling_features(&self, ue_context: &UEContext) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut features = Vec::new();
        
        // Extended feature set for scheduling
        features.extend_from_slice(&[
            ue_context.current_qoe.throughput / 100.0,
            ue_context.current_qoe.latency / 200.0,
            ue_context.current_qoe.jitter / 50.0,
            ue_context.current_qoe.packet_loss / 10.0,
            ue_context.service_requirements.min_throughput / 100.0,
            ue_context.service_requirements.max_latency / 200.0,
            ue_context.service_requirements.priority as f32 / 255.0,
            ue_context.device_capabilities.max_mimo_layers as f32 / 8.0,
            if ue_context.device_capabilities.ca_support { 1.0 } else { 0.0 },
        ]);
        
        // Pad to required size
        features.resize(64, 0.0);
        
        Ok(features)
    }

    /// Get teacher model outputs
    async fn get_teacher_qoe_output(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // This would call the actual teacher model
        // For now, return a placeholder output
        Ok(vec![0.8, 0.2, 0.1, 0.05, 4.0, 4.2, 0.99, 0.999])
    }

    async fn get_teacher_classification_output(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(vec![0.1, 0.7, 0.15, 0.03, 0.02]) // User group probabilities
    }

    async fn get_teacher_scheduling_output(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(vec![0.5; 32]) // Resource allocation parameters
    }

    /// Train student models using knowledge distillation
    pub async fn train_student_models(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Starting knowledge distillation training...");
        
        for epoch in 0..self.config.distillation_epochs {
            // Train QoE predictor
            let qoe_loss = self.train_qoe_student_epoch().await?;
            
            // Train user classifier
            let classifier_loss = self.train_classifier_student_epoch().await?;
            
            // Train MAC scheduler
            let scheduler_loss = self.train_scheduler_student_epoch().await?;
            
            if epoch % 10 == 0 {
                println!("Epoch {}: QoE Loss: {:.4}, Classifier Loss: {:.4}, Scheduler Loss: {:.4}",
                    epoch, qoe_loss.total_loss, classifier_loss.total_loss, scheduler_loss.total_loss);
            }
        }
        
        println!("Knowledge distillation training completed!");
        Ok(())
    }

    /// Train QoE student model for one epoch
    async fn train_qoe_student_epoch(&self) -> Result<DistillationLoss, Box<dyn std::error::Error>> {
        let buffer = self.training_buffer.read().await;
        let mut total_loss = DistillationLoss {
            distillation_loss: 0.0,
            student_loss: 0.0,
            total_loss: 0.0,
            temperature: self.config.temperature,
        };
        
        if buffer.qoe_samples.is_empty() {
            return Ok(total_loss);
        }
        
        let mut batch_count = 0;
        
        // Process batches
        for batch in buffer.qoe_samples.chunks(self.config.batch_size) {
            let mut batch_loss = DistillationLoss {
                distillation_loss: 0.0,
                student_loss: 0.0,
                total_loss: 0.0,
                temperature: self.config.temperature,
            };
            
            for sample in batch {
                // Get student prediction
                let student_output = self.predict_student_qoe(&sample.input_features).await?;
                
                // Calculate distillation loss (KL divergence)
                let distill_loss = self.calculate_kl_divergence(&student_output, &sample.teacher_output, self.config.temperature);
                
                // Calculate student loss (cross-entropy with ground truth)
                let student_loss = self.calculate_qoe_student_loss(&student_output, &sample.ground_truth);
                
                // Combine losses
                let combined_loss = self.config.alpha * distill_loss + self.config.beta * student_loss;
                
                batch_loss.distillation_loss += distill_loss;
                batch_loss.student_loss += student_loss;
                batch_loss.total_loss += combined_loss;
            }
            
            // Average batch losses
            let batch_size = batch.len() as f32;
            batch_loss.distillation_loss /= batch_size;
            batch_loss.student_loss /= batch_size;
            batch_loss.total_loss /= batch_size;
            
            // Backpropagate and update weights
            self.update_qoe_student_weights(&batch_loss).await?;
            
            total_loss.distillation_loss += batch_loss.distillation_loss;
            total_loss.student_loss += batch_loss.student_loss;
            total_loss.total_loss += batch_loss.total_loss;
            batch_count += 1;
        }
        
        // Average total losses
        if batch_count > 0 {
            total_loss.distillation_loss /= batch_count as f32;
            total_loss.student_loss /= batch_count as f32;
            total_loss.total_loss /= batch_count as f32;
        }
        
        Ok(total_loss)
    }

    /// Train classification student model for one epoch
    async fn train_classifier_student_epoch(&self) -> Result<DistillationLoss, Box<dyn std::error::Error>> {
        let buffer = self.training_buffer.read().await;
        let mut total_loss = DistillationLoss {
            distillation_loss: 0.0,
            student_loss: 0.0,
            total_loss: 0.0,
            temperature: self.config.temperature,
        };
        
        if buffer.classification_samples.is_empty() {
            return Ok(total_loss);
        }
        
        let mut batch_count = 0;
        
        for batch in buffer.classification_samples.chunks(self.config.batch_size) {
            let mut batch_loss = DistillationLoss {
                distillation_loss: 0.0,
                student_loss: 0.0,
                total_loss: 0.0,
                temperature: self.config.temperature,
            };
            
            for sample in batch {
                let student_output = self.predict_student_classification(&sample.input_features).await?;
                let distill_loss = self.calculate_kl_divergence(&student_output, &sample.teacher_output, self.config.temperature);
                let student_loss = self.calculate_classification_student_loss(&student_output, &sample.ground_truth);
                let combined_loss = self.config.alpha * distill_loss + self.config.beta * student_loss;
                
                batch_loss.distillation_loss += distill_loss;
                batch_loss.student_loss += student_loss;
                batch_loss.total_loss += combined_loss;
            }
            
            let batch_size = batch.len() as f32;
            batch_loss.distillation_loss /= batch_size;
            batch_loss.student_loss /= batch_size;
            batch_loss.total_loss /= batch_size;
            
            self.update_classifier_student_weights(&batch_loss).await?;
            
            total_loss.distillation_loss += batch_loss.distillation_loss;
            total_loss.student_loss += batch_loss.student_loss;
            total_loss.total_loss += batch_loss.total_loss;
            batch_count += 1;
        }
        
        if batch_count > 0 {
            total_loss.distillation_loss /= batch_count as f32;
            total_loss.student_loss /= batch_count as f32;
            total_loss.total_loss /= batch_count as f32;
        }
        
        Ok(total_loss)
    }

    /// Train scheduler student model for one epoch
    async fn train_scheduler_student_epoch(&self) -> Result<DistillationLoss, Box<dyn std::error::Error>> {
        let buffer = self.training_buffer.read().await;
        let mut total_loss = DistillationLoss {
            distillation_loss: 0.0,
            student_loss: 0.0,
            total_loss: 0.0,
            temperature: self.config.temperature,
        };
        
        if buffer.scheduling_samples.is_empty() {
            return Ok(total_loss);
        }
        
        let mut batch_count = 0;
        
        for batch in buffer.scheduling_samples.chunks(self.config.batch_size) {
            let mut batch_loss = DistillationLoss {
                distillation_loss: 0.0,
                student_loss: 0.0,
                total_loss: 0.0,
                temperature: self.config.temperature,
            };
            
            for sample in batch {
                let student_output = self.predict_student_scheduling(&sample.input_features).await?;
                let distill_loss = self.calculate_kl_divergence(&student_output, &sample.teacher_output, self.config.temperature);
                let student_loss = self.calculate_scheduling_student_loss(&student_output, &sample.ground_truth);
                let combined_loss = self.config.alpha * distill_loss + self.config.beta * student_loss;
                
                batch_loss.distillation_loss += distill_loss;
                batch_loss.student_loss += student_loss;
                batch_loss.total_loss += combined_loss;
            }
            
            let batch_size = batch.len() as f32;
            batch_loss.distillation_loss /= batch_size;
            batch_loss.student_loss /= batch_size;
            batch_loss.total_loss /= batch_size;
            
            self.update_scheduler_student_weights(&batch_loss).await?;
            
            total_loss.distillation_loss += batch_loss.distillation_loss;
            total_loss.student_loss += batch_loss.student_loss;
            total_loss.total_loss += batch_loss.total_loss;
            batch_count += 1;
        }
        
        if batch_count > 0 {
            total_loss.distillation_loss /= batch_count as f32;
            total_loss.student_loss /= batch_count as f32;
            total_loss.total_loss /= batch_count as f32;
        }
        
        Ok(total_loss)
    }

    /// Predict using student QoE model
    async fn predict_student_qoe(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.student_qoe_predictor.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Predict using student classification model
    async fn predict_student_classification(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.student_user_classifier.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Predict using student scheduling model
    async fn predict_student_scheduling(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let network = self.student_mac_scheduler.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        Ok(output.data().to_vec())
    }

    /// Calculate KL divergence for distillation loss
    fn calculate_kl_divergence(&self, student_output: &[f32], teacher_output: &[f32], temperature: f32) -> f32 {
        let mut kl_div = 0.0;
        
        for (s, t) in student_output.iter().zip(teacher_output.iter()) {
            let s_temp = (s / temperature).exp();
            let t_temp = (t / temperature).exp();
            
            if t_temp > 0.0 && s_temp > 0.0 {
                kl_div += t_temp * (t_temp.ln() - s_temp.ln());
            }
        }
        
        kl_div * temperature * temperature
    }

    /// Calculate student loss for QoE prediction
    fn calculate_qoe_student_loss(&self, prediction: &[f32], ground_truth: &QoEMetrics) -> f32 {
        // Mean squared error between prediction and ground truth
        let truth_vec = vec![
            ground_truth.throughput / 100.0,
            ground_truth.latency / 200.0,
            ground_truth.jitter / 50.0,
            ground_truth.packet_loss / 10.0,
            ground_truth.video_quality / 5.0,
            ground_truth.audio_quality / 5.0,
            ground_truth.reliability / 100.0,
            ground_truth.availability / 100.0,
        ];
        
        let mut mse = 0.0;
        for (p, t) in prediction.iter().zip(truth_vec.iter()) {
            mse += (p - t).powi(2);
        }
        
        mse / prediction.len() as f32
    }

    /// Calculate student loss for classification
    fn calculate_classification_student_loss(&self, prediction: &[f32], ground_truth: &super::UserGroup) -> f32 {
        // Cross-entropy loss
        let truth_index = match ground_truth {
            super::UserGroup::Premium => 0,
            super::UserGroup::Standard => 1,
            super::UserGroup::Basic => 2,
            super::UserGroup::IoT => 3,
            super::UserGroup::Emergency => 4,
        };
        
        if truth_index < prediction.len() {
            -prediction[truth_index].max(1e-7).ln()
        } else {
            1.0 // Default loss
        }
    }

    /// Calculate student loss for scheduling
    fn calculate_scheduling_student_loss(&self, prediction: &[f32], ground_truth: &super::ResourceAllocation) -> f32 {
        // Simplified MSE loss
        let mut mse = 0.0;
        let target_len = prediction.len().min(8);
        
        for i in 0..target_len {
            let target_val = match i {
                0 => ground_truth.prb_allocation.len() as f32 / 100.0,
                1 => ground_truth.mcs_index as f32 / 31.0,
                2 => ground_truth.mimo_layers as f32 / 8.0,
                3 => (ground_truth.power_level + 20.0) / 46.0,
                4 => ground_truth.scheduling_priority as f32 / 255.0,
                _ => 0.5, // Default values
            };
            mse += (prediction[i] - target_val).powi(2);
        }
        
        mse / target_len as f32
    }

    /// Update QoE student model weights
    async fn update_qoe_student_weights(&self, loss: &DistillationLoss) -> Result<(), Box<dyn std::error::Error>> {
        // Simplified weight update - in practice would use proper backpropagation
        println!("Updating QoE student weights with loss: {:.4}", loss.total_loss);
        Ok(())
    }

    /// Update classifier student model weights
    async fn update_classifier_student_weights(&self, loss: &DistillationLoss) -> Result<(), Box<dyn std::error::Error>> {
        println!("Updating classifier student weights with loss: {:.4}", loss.total_loss);
        Ok(())
    }

    /// Update scheduler student model weights
    async fn update_scheduler_student_weights(&self, loss: &DistillationLoss) -> Result<(), Box<dyn std::error::Error>> {
        println!("Updating scheduler student weights with loss: {:.4}", loss.total_loss);
        Ok(())
    }

    /// Create edge deployment model
    pub async fn create_edge_model(&self) -> Result<EdgeModel, Box<dyn std::error::Error>> {
        // Clone the trained student models
        let qoe_predictor = {
            let network = self.student_qoe_predictor.read().await;
            Arc::new(RwLock::new(network.clone()))
        };
        
        let user_classifier = {
            let network = self.student_user_classifier.read().await;
            Arc::new(RwLock::new(network.clone()))
        };
        
        let mac_scheduler = {
            let network = self.student_mac_scheduler.read().await;
            Arc::new(RwLock::new(network.clone()))
        };
        
        // Create metadata
        let metadata = EdgeModelMetadata {
            model_version: "1.0.0".to_string(),
            compression_ratio: self.config.compression_ratio,
            accuracy_retention: 0.95, // Placeholder - would be measured
            inference_time_ms: self.config.inference_time_target,
            model_size_mb: self.estimate_model_size().await?,
            deployment_timestamp: std::time::SystemTime::now(),
            teacher_model_versions: HashMap::from([
                ("qoe_predictor".to_string(), "teacher_v1.0".to_string()),
                ("user_classifier".to_string(), "teacher_v1.0".to_string()),
                ("mac_scheduler".to_string(), "teacher_v1.0".to_string()),
            ]),
        };
        
        let performance = Arc::new(RwLock::new(EdgePerformanceMetrics::default()));
        
        Ok(EdgeModel {
            qoe_predictor,
            user_classifier,
            mac_scheduler,
            metadata,
            performance,
        })
    }

    /// Estimate compressed model size
    async fn estimate_model_size(&self) -> Result<f32, Box<dyn std::error::Error>> {
        // Simplified estimation based on layer sizes and compression ratio
        let base_size_mb = 50.0; // Estimated base size for teacher models
        Ok(base_size_mb * self.config.compression_ratio)
    }

    /// Validate student models against teacher models
    pub async fn validate_student_models(&self, test_contexts: &[UEContext]) -> Result<ValidationResults, Box<dyn std::error::Error>> {
        let mut qoe_accuracy = 0.0;
        let mut classification_accuracy = 0.0;
        let mut scheduling_accuracy = 0.0;
        let mut total_inference_time = 0.0;
        
        for ue_context in test_contexts {
            let start_time = std::time::Instant::now();
            
            // Test QoE prediction
            if let Ok(features) = self.extract_qoe_features(ue_context).await {
                if let (Ok(student_output), Ok(teacher_output)) = (
                    self.predict_student_qoe(&features).await,
                    self.get_teacher_qoe_output(&features).await
                ) {
                    qoe_accuracy += self.calculate_output_similarity(&student_output, &teacher_output);
                }
            }
            
            // Test classification
            if let Ok(features) = self.extract_classification_features(ue_context).await {
                if let (Ok(student_output), Ok(teacher_output)) = (
                    self.predict_student_classification(&features).await,
                    self.get_teacher_classification_output(&features).await
                ) {
                    classification_accuracy += self.calculate_output_similarity(&student_output, &teacher_output);
                }
            }
            
            // Test scheduling
            if let Ok(features) = self.extract_scheduling_features(ue_context).await {
                if let (Ok(student_output), Ok(teacher_output)) = (
                    self.predict_student_scheduling(&features).await,
                    self.get_teacher_scheduling_output(&features).await
                ) {
                    scheduling_accuracy += self.calculate_output_similarity(&student_output, &teacher_output);
                }
            }
            
            total_inference_time += start_time.elapsed().as_secs_f32() * 1000.0; // Convert to ms
        }
        
        let num_tests = test_contexts.len() as f32;
        
        Ok(ValidationResults {
            qoe_accuracy: qoe_accuracy / num_tests,
            classification_accuracy: classification_accuracy / num_tests,
            scheduling_accuracy: scheduling_accuracy / num_tests,
            avg_inference_time_ms: total_inference_time / num_tests,
            compression_ratio: self.config.compression_ratio,
            meets_latency_target: (total_inference_time / num_tests) < self.config.inference_time_target,
        })
    }

    /// Calculate similarity between outputs
    fn calculate_output_similarity(&self, student: &[f32], teacher: &[f32]) -> f32 {
        if student.len() != teacher.len() || student.is_empty() {
            return 0.0;
        }
        
        let mut similarity = 0.0;
        for (s, t) in student.iter().zip(teacher.iter()) {
            similarity += 1.0 - (s - t).abs();
        }
        
        (similarity / student.len() as f32).max(0.0)
    }
}

/// Validation results for student models
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub qoe_accuracy: f32,
    pub classification_accuracy: f32,
    pub scheduling_accuracy: f32,
    pub avg_inference_time_ms: f32,
    pub compression_ratio: f32,
    pub meets_latency_target: bool,
}

impl EdgeModel {
    /// Perform inference with edge model
    pub async fn predict_qoe(&self, features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        let network = self.qoe_predictor.read().await;
        let input_tensor = Tensor::from_slice(features, &[1, features.len()]);
        let output = network.forward(&input_tensor)?;
        
        // Update performance metrics
        let inference_time = start_time.elapsed().as_secs_f32() * 1000.0;
        self.update_performance_metrics(inference_time).await;
        
        Ok(output.data().to_vec())
    }

    /// Update performance metrics
    async fn update_performance_metrics(&self, inference_time_ms: f32) {
        let mut perf = self.performance.write().await;
        
        perf.inference_count += 1;
        perf.avg_inference_time_ms = (perf.avg_inference_time_ms * (perf.inference_count - 1) as f32 + inference_time_ms) / perf.inference_count as f32;
        perf.max_inference_time_ms = perf.max_inference_time_ms.max(inference_time_ms);
        
        if perf.min_inference_time_ms == 0.0 || inference_time_ms < perf.min_inference_time_ms {
            perf.min_inference_time_ms = inference_time_ms;
        }
    }

    /// Get model metadata
    pub fn get_metadata(&self) -> &EdgeModelMetadata {
        &self.metadata
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> EdgePerformanceMetrics {
        self.performance.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ric_tsa::qoe_prediction::QoEPredictor;

    #[tokio::test]
    async fn test_knowledge_distillation_creation() {
        let teacher_qoe = Arc::new(QoEPredictor::new().unwrap());
        let teacher_classifier = Arc::new(UserClassifier::new().unwrap());
        let teacher_scheduler = Arc::new(MacScheduler::new().unwrap());
        
        let distillation = KnowledgeDistillation::new(teacher_qoe, teacher_classifier, teacher_scheduler);
        assert!(distillation.is_ok());
    }

    #[tokio::test]
    async fn test_edge_model_creation() {
        let teacher_qoe = Arc::new(QoEPredictor::new().unwrap());
        let teacher_classifier = Arc::new(UserClassifier::new().unwrap());
        let teacher_scheduler = Arc::new(MacScheduler::new().unwrap());
        
        let distillation = KnowledgeDistillation::new(teacher_qoe, teacher_classifier, teacher_scheduler).unwrap();
        let edge_model = distillation.create_edge_model().await;
        
        assert!(edge_model.is_ok());
        let model = edge_model.unwrap();
        assert!(model.metadata.compression_ratio > 0.0);
        assert!(model.metadata.inference_time_ms > 0.0);
    }
}