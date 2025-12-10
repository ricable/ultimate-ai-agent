//! Batch processing utilities for RAN neural networks

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::{
    NeuralError, NeuralResult, RanNeuralNetwork, InferenceResult, 
    RanData, FeatureExtractor, ModelType,
};

/// Batch processor for running neural network inference on multiple inputs
#[derive(Debug)]
pub struct BatchProcessor {
    /// Processor identifier
    pub id: Uuid,
    /// Configuration
    pub config: BatchConfig,
    /// Processing statistics
    pub stats: BatchStats,
    /// Active batch queue
    pub queue: Arc<Mutex<Vec<BatchItem>>>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: BatchConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            stats: BatchStats::default(),
            queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Process a batch of RAN data
    pub fn process_batch(
        &mut self,
        network: &mut RanNeuralNetwork,
        data_batch: &[RanData],
    ) -> NeuralResult<Vec<InferenceResult>> {
        let start_time = Instant::now();
        let batch_id = Uuid::new_v4();
        
        tracing::debug!("Processing batch {} with {} items", batch_id, data_batch.len());
        
        if data_batch.len() > self.config.max_batch_size {
            return Err(NeuralError::InvalidInput(format!(
                "Batch size {} exceeds maximum {}",
                data_batch.len(),
                self.config.max_batch_size
            )));
        }

        let mut results = Vec::with_capacity(data_batch.len());
        let mut successful_items = 0;
        let mut failed_items = 0;

        // Process each item in the batch
        for (i, data) in data_batch.iter().enumerate() {
            match self.process_single_item(network, data, i) {
                Ok(result) => {
                    results.push(result);
                    successful_items += 1;
                }
                Err(e) => {
                    failed_items += 1;
                    tracing::warn!("Failed to process batch item {}: {}", i, e);
                    
                    if !self.config.continue_on_error {
                        return Err(e);
                    }
                    
                    // Create error result for failed item
                    let error_result = InferenceResult {
                        id: Uuid::new_v4(),
                        model_type: network.model_type,
                        input: Vec::new(),
                        output: Vec::new(),
                        confidence: 0.0,
                        inference_time: Duration::from_millis(0),
                        timestamp: Utc::now(),
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("error".to_string(), serde_json::Value::String(e.to_string()));
                            meta
                        },
                    };
                    results.push(error_result);
                }
            }
        }

        let processing_time = start_time.elapsed();
        
        // Update statistics
        self.update_stats(data_batch.len(), successful_items, failed_items, processing_time);
        
        tracing::info!(
            "Batch {} completed: {}/{} successful, time: {:?}",
            batch_id, successful_items, data_batch.len(), processing_time
        );

        Ok(results)
    }

    /// Process a single item from the batch
    fn process_single_item(
        &self,
        network: &mut RanNeuralNetwork,
        data: &RanData,
        _item_index: usize,
    ) -> NeuralResult<InferenceResult> {
        // Extract features from RAN data
        let features = network.extract_features(data)?;
        
        // Run inference
        let output = network.predict(&features)?;
        
        // Create result
        let result = InferenceResult {
            id: Uuid::new_v4(),
            model_type: network.model_type,
            input: features,
            output,
            confidence: 0.85, // Default confidence for batch processing
            inference_time: Duration::from_millis(1), // Placeholder
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        
        Ok(result)
    }

    /// Add item to processing queue
    pub fn enqueue_item(&self, item: BatchItem) -> NeuralResult<()> {
        let mut queue = self.queue.lock()
            .map_err(|_| NeuralError::Concurrency("Failed to acquire queue lock".to_string()))?;
        
        if queue.len() >= self.config.max_queue_size {
            return Err(NeuralError::Resource("Batch queue is full".to_string()));
        }
        
        queue.push(item);
        Ok(())
    }

    /// Process queued items
    pub fn process_queue(&mut self, network: &mut RanNeuralNetwork) -> NeuralResult<Vec<InferenceResult>> {
        let items = {
            let mut queue = self.queue.lock()
                .map_err(|_| NeuralError::Concurrency("Failed to acquire queue lock".to_string()))?;
            
            let batch_size = self.config.batch_size.min(queue.len());
            if batch_size == 0 {
                return Ok(Vec::new());
            }
            
            queue.drain(0..batch_size).collect::<Vec<_>>()
        };

        let data_batch: Vec<RanData> = items.into_iter().map(|item| item.data).collect();
        self.process_batch(network, &data_batch)
    }

    /// Get queue size
    pub fn queue_size(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Clear the queue
    pub fn clear_queue(&self) -> NeuralResult<()> {
        let mut queue = self.queue.lock()
            .map_err(|_| NeuralError::Concurrency("Failed to acquire queue lock".to_string()))?;
        queue.clear();
        Ok(())
    }

    /// Update processing statistics
    fn update_stats(&mut self, total_items: usize, successful: usize, failed: usize, duration: Duration) {
        self.stats.total_batches += 1;
        self.stats.total_items += total_items;
        self.stats.successful_items += successful;
        self.stats.failed_items += failed;
        self.stats.total_processing_time += duration;
        
        // Update averages
        self.stats.avg_batch_size = self.stats.total_items as f64 / self.stats.total_batches as f64;
        self.stats.avg_processing_time = Duration::from_secs_f64(
            self.stats.total_processing_time.as_secs_f64() / self.stats.total_batches as f64
        );
        
        // Update success rate
        if self.stats.total_items > 0 {
            self.stats.success_rate = self.stats.successful_items as f64 / self.stats.total_items as f64;
        }
        
        self.stats.last_batch_time = Some(Utc::now());
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = BatchStats::default();
    }

    /// Get throughput in items per second
    pub fn throughput(&self) -> f64 {
        if self.stats.avg_processing_time.as_secs_f64() > 0.0 {
            self.stats.avg_batch_size / self.stats.avg_processing_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Configuration for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Preferred batch size
    pub batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Timeout for batch processing
    pub processing_timeout: Duration,
    /// Continue processing on individual item errors
    pub continue_on_error: bool,
    /// Enable parallel processing within batch
    pub parallel_processing: bool,
    /// Number of worker threads for parallel processing
    pub worker_threads: Option<usize>,
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_batch_size: 128,
            max_queue_size: 1000,
            processing_timeout: Duration::from_secs(30),
            continue_on_error: true,
            parallel_processing: false,
            worker_threads: None,
            enable_caching: false,
            cache_size: 100,
        }
    }
}

/// Batch processing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchStats {
    /// Total number of batches processed
    pub total_batches: u64,
    /// Total number of items processed
    pub total_items: usize,
    /// Number of successful items
    pub successful_items: usize,
    /// Number of failed items
    pub failed_items: usize,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average processing time per batch
    pub avg_processing_time: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Last batch processing time
    pub last_batch_time: Option<DateTime<Utc>>,
}

impl BatchStats {
    /// Get items per second throughput
    pub fn items_per_second(&self) -> f64 {
        if self.total_processing_time.as_secs_f64() > 0.0 {
            self.total_items as f64 / self.total_processing_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get batches per second throughput
    pub fn batches_per_second(&self) -> f64 {
        if self.total_processing_time.as_secs_f64() > 0.0 {
            self.total_batches as f64 / self.total_processing_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        1.0 - self.success_rate
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Batches: {}, Items: {}, Success Rate: {:.1}%, Throughput: {:.1} items/sec",
            self.total_batches,
            self.total_items,
            self.success_rate * 100.0,
            self.items_per_second()
        )
    }
}

/// Item in the batch processing queue
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Item identifier
    pub id: Uuid,
    /// RAN data to process
    pub data: RanData,
    /// Item priority (higher = more important)
    pub priority: u8,
    /// Timestamp when item was queued
    pub queued_at: DateTime<Utc>,
    /// Optional callback ID for result delivery
    pub callback_id: Option<String>,
    /// Item metadata
    pub metadata: HashMap<String, String>,
}

impl BatchItem {
    /// Create a new batch item
    pub fn new(data: RanData) -> Self {
        Self {
            id: Uuid::new_v4(),
            data,
            priority: 5, // Medium priority
            queued_at: Utc::now(),
            callback_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a batch item with priority
    pub fn with_priority(data: RanData, priority: u8) -> Self {
        Self {
            id: Uuid::new_v4(),
            data,
            priority,
            queued_at: Utc::now(),
            callback_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Get age of the item in queue
    pub fn age(&self) -> Duration {
        let now = Utc::now();
        (now - self.queued_at).to_std().unwrap_or(Duration::from_secs(0))
    }
}

/// Parallel batch processor using worker threads
#[derive(Debug)]
pub struct ParallelBatchProcessor {
    /// Base processor
    pub processor: BatchProcessor,
    /// Number of worker threads
    pub num_workers: usize,
    /// Worker thread handles
    pub workers: Vec<std::thread::JoinHandle<()>>,
}

impl ParallelBatchProcessor {
    /// Create a new parallel batch processor
    pub fn new(config: BatchConfig, num_workers: usize) -> Self {
        let processor = BatchProcessor::new(config);
        
        Self {
            processor,
            num_workers,
            workers: Vec::new(),
        }
    }

    /// Start worker threads
    pub fn start_workers(&mut self) -> NeuralResult<()> {
        // Implementation would create worker threads
        // For now, just log that workers would be started
        tracing::info!("Starting {} worker threads", self.num_workers);
        Ok(())
    }

    /// Stop worker threads
    pub fn stop_workers(&mut self) -> NeuralResult<()> {
        // Implementation would stop worker threads
        tracing::info!("Stopping worker threads");
        Ok(())
    }
}

/// Result aggregator for batch processing
#[derive(Debug, Default)]
pub struct BatchResultAggregator {
    /// Aggregated results by model type
    pub results_by_type: HashMap<ModelType, Vec<InferenceResult>>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

impl BatchResultAggregator {
    /// Create a new result aggregator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add results from a batch
    pub fn add_batch_results(&mut self, results: Vec<InferenceResult>) {
        for result in results {
            self.results_by_type
                .entry(result.model_type)
                .or_insert_with(Vec::new)
                .push(result);
        }
    }

    /// Get results for a specific model type
    pub fn get_results(&self, model_type: ModelType) -> Option<&Vec<InferenceResult>> {
        self.results_by_type.get(&model_type)
    }

    /// Calculate aggregate metrics
    pub fn calculate_metrics(&mut self) {
        let total_results: usize = self.results_by_type.values().map(|v| v.len()).sum();
        self.metrics.insert("total_results".to_string(), total_results as f64);

        // Calculate average confidence by model type
        for (model_type, results) in &self.results_by_type {
            if !results.is_empty() {
                let avg_confidence = results.iter()
                    .map(|r| r.confidence)
                    .sum::<f64>() / results.len() as f64;
                
                self.metrics.insert(
                    format!("{:?}_avg_confidence", model_type),
                    avg_confidence
                );
            }
        }
    }

    /// Get metric value
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// Clear all results and metrics
    pub fn clear(&mut self) {
        self.results_by_type.clear();
        self.metrics.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_batch_size, 128);
        assert!(config.continue_on_error);
    }

    #[test]
    fn test_batch_stats() {
        let mut stats = BatchStats::default();
        stats.total_items = 100;
        stats.successful_items = 95;
        stats.success_rate = 0.95;
        
        assert_eq!(stats.error_rate(), 0.05);
        assert!(stats.summary().contains("95.0%"));
    }

    #[test]
    fn test_batch_item() {
        let data = RanData::new();
        let item = BatchItem::new(data);
        
        assert_eq!(item.priority, 5);
        assert!(item.age().as_secs() < 1); // Should be very recent
    }

    #[test]
    fn test_batch_processor_creation() {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config);
        
        assert_eq!(processor.queue_size(), 0);
        assert_eq!(processor.stats.total_batches, 0);
    }

    #[test]
    fn test_result_aggregator() {
        let mut aggregator = BatchResultAggregator::new();
        assert_eq!(aggregator.results_by_type.len(), 0);
        
        aggregator.calculate_metrics();
        assert_eq!(aggregator.get_metric("total_results"), Some(0.0));
    }

    #[test]
    fn test_parallel_processor() {
        let config = BatchConfig::default();
        let processor = ParallelBatchProcessor::new(config, 4);
        
        assert_eq!(processor.num_workers, 4);
        assert_eq!(processor.workers.len(), 0);
    }
}