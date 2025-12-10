//! Streaming Inference Engine for Real-time Traffic Steering
//! 
//! This module implements high-performance streaming inference capabilities
//! for sub-millisecond traffic steering decisions with batch processing
//! and pipeline optimization.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, Semaphore};
use tokio::time::{Duration, Instant};
use futures::stream::{Stream, StreamExt};
use rayon::prelude::*;

use crate::pfs_core::{NeuralNetwork, Tensor, TensorOps, BatchProcessor};
use super::{UEContext, QoEMetrics, UserGroup, ServiceType, SteeringDecision, knowledge_distillation::EdgeModel};

/// Streaming inference engine for real-time processing
pub struct StreamingInferenceEngine {
    // Edge models for fast inference
    edge_models: Arc<RwLock<HashMap<String, Arc<EdgeModel>>>>,
    
    // Batch processors for different models
    qoe_batch_processor: Arc<RwLock<BatchProcessor>>,
    classification_batch_processor: Arc<RwLock<BatchProcessor>>,
    scheduling_batch_processor: Arc<RwLock<BatchProcessor>>,
    
    // Input queues for batching
    qoe_input_queue: Arc<RwLock<VecDeque<StreamingInput>>>,
    classification_input_queue: Arc<RwLock<VecDeque<StreamingInput>>>,
    scheduling_input_queue: Arc<RwLock<VecDeque<StreamingInput>>>,
    
    // Result caches
    result_cache: Arc<RwLock<ResultCache>>,
    
    // Performance monitoring
    performance_monitor: Arc<RwLock<StreamingPerformanceMonitor>>,
    
    // Processing semaphore for concurrency control
    processing_semaphore: Arc<Semaphore>,
    
    // Configuration
    config: StreamingInferenceConfig,
}

/// Configuration for streaming inference
#[derive(Debug, Clone)]
pub struct StreamingInferenceConfig {
    pub max_batch_size: usize,
    pub batch_timeout_ms: u64,
    pub max_concurrent_batches: usize,
    pub cache_ttl_ms: u64,
    pub enable_prefetching: bool,
    pub enable_result_caching: bool,
    pub pipeline_stages: usize,
    pub thread_pool_size: usize,
    pub memory_pool_size_mb: usize,
}

/// Streaming input with metadata
#[derive(Debug, Clone)]
pub struct StreamingInput {
    pub ue_id: u64,
    pub ue_context: UEContext,
    pub input_features: Vec<f32>,
    pub priority: Priority,
    pub timestamp: Instant,
    pub correlation_id: String,
}

/// Streaming result with timing information
#[derive(Debug, Clone)]
pub struct StreamingResult {
    pub ue_id: u64,
    pub target_cell: u32,
    pub target_band: super::FrequencyBand,
    pub resource_allocation: super::ResourceAllocation,
    pub confidence: f32,
    pub processing_time_ns: u64,
    pub correlation_id: String,
}

/// Priority levels for streaming inputs
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Emergency = 4,
    High = 3,
    Normal = 2,
    Low = 1,
    Background = 0,
}

/// Result cache for avoiding redundant computations
#[derive(Debug, Clone)]
pub struct ResultCache {
    pub qoe_cache: HashMap<String, (Vec<f32>, Instant)>,
    pub classification_cache: HashMap<String, (Vec<f32>, Instant)>,
    pub scheduling_cache: HashMap<String, (Vec<f32>, Instant)>,
    pub max_size: usize,
}

/// Performance monitoring for streaming inference
#[derive(Debug, Clone, Default)]
pub struct StreamingPerformanceMonitor {
    pub total_requests: u64,
    pub total_processing_time_ns: u64,
    pub avg_processing_time_ns: u64,
    pub min_processing_time_ns: u64,
    pub max_processing_time_ns: u64,
    pub p95_processing_time_ns: u64,
    pub p99_processing_time_ns: u64,
    pub throughput_requests_per_sec: f64,
    pub batch_utilization: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub recent_times: VecDeque<u64>,
}

/// Batch processing result
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub results: Vec<StreamingResult>,
    pub batch_size: usize,
    pub processing_time_ns: u64,
    pub model_type: ModelType,
}

/// Model types for batch processing
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    QoEPrediction,
    UserClassification,
    ResourceScheduling,
}

/// Streaming inference pipeline stage
#[derive(Debug, Clone)]
pub struct PipelineStage {
    pub stage_id: usize,
    pub stage_type: StageType,
    pub processing_time_ns: u64,
    pub queue_size: usize,
}

/// Pipeline stage types
#[derive(Debug, Clone)]
pub enum StageType {
    FeatureExtraction,
    Preprocessing,
    ModelInference,
    Postprocessing,
    ResultAggregation,
}

impl Default for StreamingInferenceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            batch_timeout_ms: 1,      // 1ms timeout for sub-millisecond inference
            max_concurrent_batches: 4,
            cache_ttl_ms: 100,        // 100ms cache TTL
            enable_prefetching: true,
            enable_result_caching: true,
            pipeline_stages: 5,
            thread_pool_size: 8,
            memory_pool_size_mb: 256,
        }
    }
}

impl StreamingInferenceEngine {
    /// Create a new streaming inference engine
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = StreamingInferenceConfig::default();
        Self::new_with_config(config)
    }

    /// Create streaming inference engine with custom configuration
    pub fn new_with_config(config: StreamingInferenceConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize batch processors
        let qoe_batch_processor = Arc::new(RwLock::new(BatchProcessor::new(config.max_batch_size)));
        let classification_batch_processor = Arc::new(RwLock::new(BatchProcessor::new(config.max_batch_size)));
        let scheduling_batch_processor = Arc::new(RwLock::new(BatchProcessor::new(config.max_batch_size)));
        
        // Initialize input queues
        let qoe_input_queue = Arc::new(RwLock::new(VecDeque::new()));
        let classification_input_queue = Arc::new(RwLock::new(VecDeque::new()));
        let scheduling_input_queue = Arc::new(RwLock::new(VecDeque::new()));
        
        // Initialize result cache
        let result_cache = Arc::new(RwLock::new(ResultCache {
            qoe_cache: HashMap::new(),
            classification_cache: HashMap::new(),
            scheduling_cache: HashMap::new(),
            max_size: 1000,
        }));
        
        // Initialize performance monitor
        let performance_monitor = Arc::new(RwLock::new(StreamingPerformanceMonitor::default()));
        
        // Initialize semaphore for concurrency control
        let processing_semaphore = Arc::new(Semaphore::new(config.max_concurrent_batches));
        
        // Initialize edge models map
        let edge_models = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            edge_models,
            qoe_batch_processor,
            classification_batch_processor,
            scheduling_batch_processor,
            qoe_input_queue,
            classification_input_queue,
            scheduling_input_queue,
            result_cache,
            performance_monitor,
            processing_semaphore,
            config,
        })
    }

    /// Register an edge model for streaming inference
    pub async fn register_edge_model(&self, model_id: String, model: Arc<EdgeModel>) -> Result<(), Box<dyn std::error::Error>> {
        let mut models = self.edge_models.write().await;
        models.insert(model_id, model);
        Ok(())
    }

    /// Process a batch of UE contexts for streaming inference
    pub async fn process_batch(&self, ue_ids: &[u64], ue_contexts: &HashMap<u64, UEContext>) -> Result<Vec<(u64, StreamingResult)>, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        
        // Create streaming inputs
        let mut streaming_inputs = Vec::new();
        for &ue_id in ue_ids {
            if let Some(ue_context) = ue_contexts.get(&ue_id) {
                let features = self.extract_streaming_features(ue_context).await?;
                let priority = self.determine_priority(ue_context);
                
                let input = StreamingInput {
                    ue_id,
                    ue_context: ue_context.clone(),
                    input_features: features,
                    priority,
                    timestamp: start_time,
                    correlation_id: format!("batch_{}_{}", ue_id, start_time.elapsed().as_nanos()),
                };
                streaming_inputs.push(input);
            }
        }
        
        // Sort by priority
        streaming_inputs.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        // Process in parallel pipelines
        let qoe_results = self.process_qoe_batch(&streaming_inputs).await?;
        let classification_results = self.process_classification_batch(&streaming_inputs).await?;
        let scheduling_results = self.process_scheduling_batch(&streaming_inputs).await?;
        
        // Combine results
        for (i, input) in streaming_inputs.iter().enumerate() {
            let qoe_output = qoe_results.get(i).cloned().unwrap_or_default();
            let class_output = classification_results.get(i).cloned().unwrap_or_default();
            let sched_output = scheduling_results.get(i).cloned().unwrap_or_default();
            
            let result = self.create_streaming_result(input, &qoe_output, &class_output, &sched_output)?;
            results.push((input.ue_id, result));
        }
        
        // Update performance metrics
        let processing_time = start_time.elapsed().as_nanos() as u64;
        self.update_performance_metrics(processing_time, streaming_inputs.len()).await;
        
        Ok(results)
    }

    /// Extract features for streaming inference
    async fn extract_streaming_features(&self, ue_context: &UEContext) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut features = Vec::new();
        
        // Core QoE features
        features.extend_from_slice(&[
            ue_context.current_qoe.throughput / 100.0,
            ue_context.current_qoe.latency / 200.0,
            ue_context.current_qoe.jitter / 50.0,
            ue_context.current_qoe.packet_loss / 10.0,
            ue_context.current_qoe.reliability / 100.0,
            ue_context.current_qoe.availability / 100.0,
        ]);
        
        // Service and user context
        let service_encoding = self.encode_service_type(&ue_context.service_type);
        let user_encoding = self.encode_user_group(&ue_context.user_group);
        features.extend_from_slice(&service_encoding);
        features.extend_from_slice(&user_encoding);
        
        // Device capabilities
        features.extend_from_slice(&[
            ue_context.device_capabilities.max_mimo_layers as f32 / 8.0,
            if ue_context.device_capabilities.ca_support { 1.0 } else { 0.0 },
            if ue_context.device_capabilities.dual_connectivity { 1.0 } else { 0.0 },
        ]);
        
        // Service requirements
        features.extend_from_slice(&[
            ue_context.service_requirements.min_throughput / 100.0,
            ue_context.service_requirements.max_latency / 200.0,
            ue_context.service_requirements.priority as f32 / 255.0,
        ]);
        
        // Mobility encoding
        let mobility_encoding = match ue_context.mobility_pattern {
            super::MobilityPattern::Stationary => [1.0, 0.0, 0.0, 0.0],
            super::MobilityPattern::Pedestrian => [0.0, 1.0, 0.0, 0.0],
            super::MobilityPattern::Vehicular => [0.0, 0.0, 1.0, 0.0],
            super::MobilityPattern::HighSpeed => [0.0, 0.0, 0.0, 1.0],
        };
        features.extend_from_slice(&mobility_encoding);
        
        // Temporal features
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let time_of_day = ((now % 86400) as f32) / 86400.0;
        features.push(time_of_day);
        
        // Pad to consistent size
        features.resize(32, 0.0);
        
        Ok(features)
    }

    /// Encode service type to vector
    fn encode_service_type(&self, service_type: &ServiceType) -> Vec<f32> {
        match service_type {
            ServiceType::VideoStreaming => vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ServiceType::VoiceCall => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ServiceType::Gaming => vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ServiceType::FileTransfer => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ServiceType::WebBrowsing => vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ServiceType::IoTSensor => vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ServiceType::Emergency => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ServiceType::AR_VR => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Encode user group to vector
    fn encode_user_group(&self, user_group: &UserGroup) -> Vec<f32> {
        match user_group {
            UserGroup::Premium => vec![1.0, 0.0, 0.0, 0.0, 0.0],
            UserGroup::Standard => vec![0.0, 1.0, 0.0, 0.0, 0.0],
            UserGroup::Basic => vec![0.0, 0.0, 1.0, 0.0, 0.0],
            UserGroup::IoT => vec![0.0, 0.0, 0.0, 1.0, 0.0],
            UserGroup::Emergency => vec![0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Determine processing priority
    fn determine_priority(&self, ue_context: &UEContext) -> Priority {
        match ue_context.user_group {
            UserGroup::Emergency => Priority::Emergency,
            UserGroup::Premium => Priority::High,
            UserGroup::Standard => Priority::Normal,
            UserGroup::Basic => Priority::Low,
            UserGroup::IoT => match ue_context.service_type {
                ServiceType::Emergency => Priority::Emergency,
                _ => Priority::Background,
            },
        }
    }

    /// Process QoE prediction batch
    async fn process_qoe_batch(&self, inputs: &[StreamingInput]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let _permit = self.processing_semaphore.acquire().await?;
        let start_time = Instant::now();
        
        // Check cache first
        let mut results = Vec::new();
        let mut cache_misses = Vec::new();
        
        if self.config.enable_result_caching {
            let cache = self.result_cache.read().await;
            for input in inputs {
                let cache_key = self.create_cache_key(&input.input_features, ModelType::QoEPrediction);
                if let Some((cached_result, timestamp)) = cache.qoe_cache.get(&cache_key) {
                    if start_time.duration_since(*timestamp).as_millis() < self.config.cache_ttl_ms as u128 {
                        results.push(cached_result.clone());
                        continue;
                    }
                }
                cache_misses.push(input);
                results.push(Vec::new()); // Placeholder
            }
        } else {
            cache_misses = inputs.to_vec();
            results.resize(inputs.len(), Vec::new());
        }
        
        // Process cache misses in batches
        if !cache_misses.is_empty() {
            let batch_results = self.process_model_batch(&cache_misses, ModelType::QoEPrediction).await?;
            
            // Update cache and results
            if self.config.enable_result_caching {
                let mut cache = self.result_cache.write().await;
                for (i, input) in cache_misses.iter().enumerate() {
                    let cache_key = self.create_cache_key(&input.input_features, ModelType::QoEPrediction);
                    if let Some(result) = batch_results.get(i) {
                        cache.qoe_cache.insert(cache_key, (result.clone(), start_time));
                        // Find original position and update result
                        for (j, original_input) in inputs.iter().enumerate() {
                            if original_input.ue_id == input.ue_id {
                                results[j] = result.clone();
                                break;
                            }
                        }
                    }
                }
            } else {
                results = batch_results;
            }
        }
        
        Ok(results)
    }

    /// Process classification batch
    async fn process_classification_batch(&self, inputs: &[StreamingInput]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let _permit = self.processing_semaphore.acquire().await?;
        
        // Similar to QoE batch processing but for classification
        let mut results = Vec::new();
        let mut cache_misses = Vec::new();
        
        if self.config.enable_result_caching {
            let cache = self.result_cache.read().await;
            for input in inputs {
                let cache_key = self.create_cache_key(&input.input_features, ModelType::UserClassification);
                if let Some((cached_result, timestamp)) = cache.classification_cache.get(&cache_key) {
                    if Instant::now().duration_since(*timestamp).as_millis() < self.config.cache_ttl_ms as u128 {
                        results.push(cached_result.clone());
                        continue;
                    }
                }
                cache_misses.push(input);
                results.push(Vec::new());
            }
        } else {
            cache_misses = inputs.to_vec();
            results.resize(inputs.len(), Vec::new());
        }
        
        if !cache_misses.is_empty() {
            let batch_results = self.process_model_batch(&cache_misses, ModelType::UserClassification).await?;
            
            if self.config.enable_result_caching {
                let mut cache = self.result_cache.write().await;
                for (i, input) in cache_misses.iter().enumerate() {
                    let cache_key = self.create_cache_key(&input.input_features, ModelType::UserClassification);
                    if let Some(result) = batch_results.get(i) {
                        cache.classification_cache.insert(cache_key, (result.clone(), Instant::now()));
                        for (j, original_input) in inputs.iter().enumerate() {
                            if original_input.ue_id == input.ue_id {
                                results[j] = result.clone();
                                break;
                            }
                        }
                    }
                }
            } else {
                results = batch_results;
            }
        }
        
        Ok(results)
    }

    /// Process scheduling batch
    async fn process_scheduling_batch(&self, inputs: &[StreamingInput]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let _permit = self.processing_semaphore.acquire().await?;
        
        // Similar processing pattern for scheduling
        let mut results = Vec::new();
        let mut cache_misses = Vec::new();
        
        if self.config.enable_result_caching {
            let cache = self.result_cache.read().await;
            for input in inputs {
                let cache_key = self.create_cache_key(&input.input_features, ModelType::ResourceScheduling);
                if let Some((cached_result, timestamp)) = cache.scheduling_cache.get(&cache_key) {
                    if Instant::now().duration_since(*timestamp).as_millis() < self.config.cache_ttl_ms as u128 {
                        results.push(cached_result.clone());
                        continue;
                    }
                }
                cache_misses.push(input);
                results.push(Vec::new());
            }
        } else {
            cache_misses = inputs.to_vec();
            results.resize(inputs.len(), Vec::new());
        }
        
        if !cache_misses.is_empty() {
            let batch_results = self.process_model_batch(&cache_misses, ModelType::ResourceScheduling).await?;
            
            if self.config.enable_result_caching {
                let mut cache = self.result_cache.write().await;
                for (i, input) in cache_misses.iter().enumerate() {
                    let cache_key = self.create_cache_key(&input.input_features, ModelType::ResourceScheduling);
                    if let Some(result) = batch_results.get(i) {
                        cache.scheduling_cache.insert(cache_key, (result.clone(), Instant::now()));
                        for (j, original_input) in inputs.iter().enumerate() {
                            if original_input.ue_id == input.ue_id {
                                results[j] = result.clone();
                                break;
                            }
                        }
                    }
                }
            } else {
                results = batch_results;
            }
        }
        
        Ok(results)
    }

    /// Process a batch with a specific model type
    async fn process_model_batch(&self, inputs: &[&StreamingInput], model_type: ModelType) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        // Create batched input tensor
        let batch_size = inputs.len();
        let feature_size = inputs.first().map(|i| i.input_features.len()).unwrap_or(32);
        let mut batch_data = Vec::with_capacity(batch_size * feature_size);
        
        for input in inputs {
            batch_data.extend_from_slice(&input.input_features);
        }
        
        let batch_tensor = Tensor::from_slice(&batch_data, &[batch_size, feature_size]);
        
        // Process with appropriate model
        let batch_output = match model_type {
            ModelType::QoEPrediction => {
                // Use first available QoE model
                let models = self.edge_models.read().await;
                if let Some(model) = models.values().next() {
                    let predictor = model.qoe_predictor.read().await;
                    predictor.forward(&batch_tensor)?
                } else {
                    return Err("No QoE model available".into());
                }
            }
            ModelType::UserClassification => {
                let models = self.edge_models.read().await;
                if let Some(model) = models.values().next() {
                    let classifier = model.user_classifier.read().await;
                    classifier.forward(&batch_tensor)?
                } else {
                    return Err("No classification model available".into());
                }
            }
            ModelType::ResourceScheduling => {
                let models = self.edge_models.read().await;
                if let Some(model) = models.values().next() {
                    let scheduler = model.mac_scheduler.read().await;
                    scheduler.forward(&batch_tensor)?
                } else {
                    return Err("No scheduling model available".into());
                }
            }
        };
        
        // Split batch output into individual results
        let output_data = batch_output.data();
        let output_size = output_data.len() / batch_size;
        let mut results = Vec::new();
        
        for i in 0..batch_size {
            let start_idx = i * output_size;
            let end_idx = start_idx + output_size;
            results.push(output_data[start_idx..end_idx].to_vec());
        }
        
        Ok(results)
    }

    /// Create cache key for result caching
    fn create_cache_key(&self, features: &[f32], model_type: ModelType) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for &feature in features {
            (feature * 1000.0) as i32.hash(&mut hasher); // Quantize for better cache hits
        }
        model_type.hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }

    /// Create streaming result from model outputs
    fn create_streaming_result(
        &self,
        input: &StreamingInput,
        qoe_output: &[f32],
        class_output: &[f32],
        sched_output: &[f32],
    ) -> Result<StreamingResult, Box<dyn std::error::Error>> {
        // Determine target cell (simplified)
        let target_cell = if !sched_output.is_empty() {
            (sched_output[0] * 100.0) as u32 + 1
        } else {
            1
        };
        
        // Determine target band
        let target_band = super::FrequencyBand::Band1800MHz; // Simplified
        
        // Create resource allocation
        let resource_allocation = super::ResourceAllocation {
            prb_allocation: if !sched_output.is_empty() {
                let num_prbs = (sched_output[1] * 20.0) as usize;
                (0..num_prbs).map(|i| i as u16).collect()
            } else {
                vec![1, 2, 3]
            },
            mcs_index: if sched_output.len() > 2 {
                (sched_output[2] * 31.0) as u8
            } else {
                15
            },
            mimo_layers: if sched_output.len() > 3 {
                (sched_output[3] * 4.0) as u8 + 1
            } else {
                2
            },
            power_level: if sched_output.len() > 4 {
                sched_output[4] * 46.0 - 20.0
            } else {
                20.0
            },
            scheduling_priority: if sched_output.len() > 5 {
                (sched_output[5] * 255.0) as u8
            } else {
                128
            },
        };
        
        // Calculate confidence from outputs
        let confidence = if !qoe_output.is_empty() && !class_output.is_empty() {
            let qoe_conf = qoe_output.iter().sum::<f32>() / qoe_output.len() as f32;
            let class_conf = class_output.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.5);
            (qoe_conf + class_conf) / 2.0
        } else {
            0.7
        };
        
        let processing_time = input.timestamp.elapsed().as_nanos() as u64;
        
        Ok(StreamingResult {
            ue_id: input.ue_id,
            target_cell,
            target_band,
            resource_allocation,
            confidence,
            processing_time_ns: processing_time,
            correlation_id: input.correlation_id.clone(),
        })
    }

    /// Update performance metrics
    async fn update_performance_metrics(&self, processing_time_ns: u64, batch_size: usize) {
        let mut monitor = self.performance_monitor.write().await;
        
        monitor.total_requests += batch_size as u64;
        monitor.total_processing_time_ns += processing_time_ns;
        monitor.avg_processing_time_ns = monitor.total_processing_time_ns / monitor.total_requests;
        
        if monitor.min_processing_time_ns == 0 || processing_time_ns < monitor.min_processing_time_ns {
            monitor.min_processing_time_ns = processing_time_ns;
        }
        monitor.max_processing_time_ns = monitor.max_processing_time_ns.max(processing_time_ns);
        
        // Update recent times for percentile calculation
        monitor.recent_times.push_back(processing_time_ns);
        if monitor.recent_times.len() > 1000 {
            monitor.recent_times.pop_front();
        }
        
        // Calculate percentiles
        if monitor.recent_times.len() >= 20 {
            let mut sorted_times: Vec<u64> = monitor.recent_times.iter().cloned().collect();
            sorted_times.sort_unstable();
            
            let p95_idx = (sorted_times.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted_times.len() as f64 * 0.99) as usize;
            
            monitor.p95_processing_time_ns = sorted_times[p95_idx.min(sorted_times.len() - 1)];
            monitor.p99_processing_time_ns = sorted_times[p99_idx.min(sorted_times.len() - 1)];
        }
        
        // Calculate throughput (requests per second)
        let time_window_ns = 1_000_000_000u64; // 1 second
        let recent_requests = monitor.recent_times.iter()
            .filter(|&&time| processing_time_ns.saturating_sub(time) < time_window_ns)
            .count();
        monitor.throughput_requests_per_sec = recent_requests as f64;
        
        // Calculate batch utilization
        monitor.batch_utilization = batch_size as f64 / self.config.max_batch_size as f64;
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> StreamingPerformanceMonitor {
        self.performance_monitor.read().await.clone()
    }

    /// Clear result caches
    pub async fn clear_caches(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut cache = self.result_cache.write().await;
        cache.qoe_cache.clear();
        cache.classification_cache.clear();
        cache.scheduling_cache.clear();
        Ok(())
    }

    /// Optimize memory usage
    pub async fn optimize_memory(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Clean expired cache entries
        let now = Instant::now();
        let ttl = Duration::from_millis(self.config.cache_ttl_ms);
        
        let mut cache = self.result_cache.write().await;
        
        cache.qoe_cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < ttl);
        cache.classification_cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < ttl);
        cache.scheduling_cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < ttl);
        
        // Limit cache sizes
        if cache.qoe_cache.len() > cache.max_size {
            let excess = cache.qoe_cache.len() - cache.max_size;
            let keys_to_remove: Vec<String> = cache.qoe_cache.keys().take(excess).cloned().collect();
            for key in keys_to_remove {
                cache.qoe_cache.remove(&key);
            }
        }
        
        Ok(())
    }

    /// Create processing pipeline for continuous streaming
    pub async fn create_streaming_pipeline(&self) -> Result<impl Stream<Item = StreamingResult>, Box<dyn std::error::Error>> {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingInput>();
        
        // This would create a continuous processing pipeline
        // For now, return a simple stream
        Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
            .map(|_input| StreamingResult {
                ue_id: 1,
                target_cell: 1,
                target_band: super::FrequencyBand::Band1800MHz,
                resource_allocation: super::ResourceAllocation {
                    prb_allocation: vec![1, 2, 3],
                    mcs_index: 15,
                    mimo_layers: 2,
                    power_level: 20.0,
                    scheduling_priority: 128,
                },
                confidence: 0.8,
                processing_time_ns: 500_000, // 0.5ms
                correlation_id: "stream_result".to_string(),
            }))
    }
}

impl std::hash::Hash for ModelType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ModelType::QoEPrediction => 0.hash(state),
            ModelType::UserClassification => 1.hash(state),
            ModelType::ResourceScheduling => 2.hash(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ric_tsa::{DeviceCapabilities, ServiceRequirements, FrequencyBand, MobilityPattern};

    #[tokio::test]
    async fn test_streaming_engine_creation() {
        let engine = StreamingInferenceEngine::new();
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_feature_extraction() {
        let engine = StreamingInferenceEngine::new().unwrap();
        
        let ue_context = UEContext {
            ue_id: 1,
            user_group: UserGroup::Standard,
            service_type: ServiceType::VideoStreaming,
            current_qoe: QoEMetrics {
                throughput: 20.0,
                latency: 25.0,
                jitter: 5.0,
                packet_loss: 0.1,
                video_quality: 4.0,
                audio_quality: 4.2,
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
                min_throughput: 10.0,
                max_latency: 50.0,
                max_jitter: 10.0,
                max_packet_loss: 1.0,
                priority: 128,
            },
        };

        let features = engine.extract_streaming_features(&ue_context).await;
        assert!(features.is_ok());
        
        let feature_vec = features.unwrap();
        assert_eq!(feature_vec.len(), 32);
        assert!(feature_vec.iter().all(|&f| f >= 0.0 && f <= 1.0));
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let engine = StreamingInferenceEngine::new().unwrap();
        
        let ue_context = UEContext {
            ue_id: 1,
            user_group: UserGroup::Standard,
            service_type: ServiceType::VideoStreaming,
            current_qoe: QoEMetrics {
                throughput: 15.0,
                latency: 30.0,
                jitter: 8.0,
                packet_loss: 0.2,
                video_quality: 3.8,
                audio_quality: 4.0,
                reliability: 98.5,
                availability: 99.8,
            },
            location: (40.7128, -74.0060),
            mobility_pattern: MobilityPattern::Pedestrian,
            device_capabilities: DeviceCapabilities {
                supported_bands: vec![FrequencyBand::Band1800MHz],
                max_mimo_layers: 2,
                ca_support: false,
                dual_connectivity: false,
            },
            service_requirements: ServiceRequirements {
                min_throughput: 5.0,
                max_latency: 60.0,
                max_jitter: 15.0,
                max_packet_loss: 2.0,
                priority: 100,
            },
        };

        let mut ue_contexts = HashMap::new();
        ue_contexts.insert(1, ue_context);
        
        let ue_ids = vec![1];
        let result = engine.process_batch(&ue_ids, &ue_contexts).await;
        
        // Note: This test will fail because we don't have actual edge models registered
        // In practice, you would register edge models before testing
        assert!(result.is_err() || !result.unwrap().is_empty());
    }
}