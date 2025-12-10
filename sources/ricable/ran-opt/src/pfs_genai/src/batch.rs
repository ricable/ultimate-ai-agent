//! Batch inference queuing system for efficient LLM processing

use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, oneshot, Notify};
use std::time::{Duration, Instant};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::{LLMClient, Prompt, Response, Result, GenAIError, BatchConfig};

/// A batch request waiting in the queue
#[derive(Debug)]
pub struct BatchRequest {
    pub backend: String,
    pub prompt: Prompt,
    pub response_tx: oneshot::Sender<Result<Response>>,
    pub timestamp: Instant,
}

/// Statistics for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    pub total_requests: u64,
    pub total_batches: u64,
    pub average_batch_size: f64,
    pub average_wait_time_ms: f64,
    pub throughput_requests_per_second: f64,
}

/// Batch processing queue
pub struct BatchQueue {
    config: BatchConfig,
    queue: Arc<Mutex<VecDeque<BatchRequest>>>,
    stats: Arc<RwLock<BatchStats>>,
    notify: Arc<Notify>,
    running: Arc<RwLock<bool>>,
}

impl BatchQueue {
    pub fn new(config: &BatchConfig) -> Self {
        let queue = Self {
            config: config.clone(),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(BatchStats {
                total_requests: 0,
                total_batches: 0,
                average_batch_size: 0.0,
                average_wait_time_ms: 0.0,
                throughput_requests_per_second: 0.0,
            })),
            notify: Arc::new(Notify::new()),
            running: Arc::new(RwLock::new(false)),
        };
        
        // Start the batch processor
        queue.start_processor();
        queue
    }
    
    /// Check if we should batch the current request
    pub fn should_batch(&self) -> bool {
        self.config.enabled
    }
    
    /// Enqueue a request for batch processing
    pub async fn enqueue(
        &self,
        backend: &str,
        prompt: &Prompt,
        client: Arc<dyn LLMClient>,
    ) -> Result<Response> {
        let (response_tx, response_rx) = oneshot::channel();
        
        let request = BatchRequest {
            backend: backend.to_string(),
            prompt: prompt.clone(),
            response_tx,
            timestamp: Instant::now(),
        };
        
        // Add to queue
        {
            let mut queue = self.queue.lock().await;
            queue.push_back(request);
            
            // Update stats
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }
        
        // Notify processor
        self.notify.notify_one();
        
        // Wait for response
        response_rx.await.map_err(|_| {
            GenAIError::Backend("Batch request was cancelled".to_string())
        })?
    }
    
    /// Start the batch processor task
    fn start_processor(&self) {
        let queue = self.queue.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let notify = self.notify.clone();
        let running = self.running.clone();
        
        tokio::spawn(async move {
            {
                let mut running_guard = running.write().await;
                *running_guard = true;
            }
            
            loop {
                // Wait for requests or timeout
                let timeout = tokio::time::sleep(Duration::from_millis(config.batch_timeout_ms));
                tokio::select! {
                    _ = notify.notified() => {
                        // Process available requests
                        Self::process_batch(&queue, &config, &stats).await;
                    }
                    _ = timeout => {
                        // Timeout - process any pending requests
                        Self::process_batch(&queue, &config, &stats).await;
                    }
                }
                
                // Check if we should continue running
                let should_run = *running.read().await;
                if !should_run {
                    break;
                }
            }
        });
    }
    
    /// Process a batch of requests
    async fn process_batch(
        queue: &Arc<Mutex<VecDeque<BatchRequest>>>,
        config: &BatchConfig,
        stats: &Arc<RwLock<BatchStats>>,
    ) {
        let mut batch = Vec::new();
        
        // Extract batch from queue
        {
            let mut queue_guard = queue.lock().await;
            
            // Take up to max_batch_size requests
            while batch.len() < config.max_batch_size && !queue_guard.is_empty() {
                if let Some(request) = queue_guard.pop_front() {
                    batch.push(request);
                }
            }
        }
        
        if batch.is_empty() {
            return;
        }
        
        // Group requests by backend
        let mut backend_batches: std::collections::HashMap<String, Vec<BatchRequest>> = 
            std::collections::HashMap::new();
        
        for request in batch {
            backend_batches
                .entry(request.backend.clone())
                .or_insert_with(Vec::new)
                .push(request);
        }
        
        // Process each backend batch
        for (backend, requests) in backend_batches {
            Self::process_backend_batch(backend, requests, stats).await;
        }
    }
    
    /// Process a batch of requests for a specific backend
    async fn process_backend_batch(
        backend: String,
        requests: Vec<BatchRequest>,
        stats: &Arc<RwLock<BatchStats>>,
    ) {
        let batch_size = requests.len();
        let batch_start = Instant::now();
        
        // Extract prompts and response channels
        let mut prompts = Vec::new();
        let mut response_channels = Vec::new();
        let mut request_timestamps = Vec::new();
        
        for request in requests {
            prompts.push(request.prompt);
            response_channels.push(request.response_tx);
            request_timestamps.push(request.timestamp);
        }
        
        // TODO: Get the actual client for the backend
        // For now, simulate batch processing
        
        // Simulate batch processing time
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Create mock responses
        let mut responses = Vec::new();
        for (i, prompt) in prompts.iter().enumerate() {
            let response = Response {
                text: format!("Batch response {} to: {}", i, 
                    prompt.user.chars().take(30).collect::<String>()),
                usage: crate::TokenUsage {
                    prompt_tokens: prompt.user.len() / 4,
                    completion_tokens: 50,
                    total_tokens: prompt.user.len() / 4 + 50,
                },
                model: backend.clone(),
                metadata: crate::ResponseMetadata {
                    latency_ms: batch_start.elapsed().as_millis() as u64,
                    cached: false,
                    cache_similarity: None,
                    batched: true,
                },
            };
            responses.push(response);
        }
        
        // Send responses back
        for (response_tx, response) in response_channels.into_iter().zip(responses.into_iter()) {
            let _ = response_tx.send(Ok(response));
        }
        
        // Update statistics
        {
            let mut stats_guard = stats.write().await;
            stats_guard.total_batches += 1;
            
            // Update average batch size
            let total_requests = stats_guard.total_requests as f64;
            let total_batches = stats_guard.total_batches as f64;
            stats_guard.average_batch_size = total_requests / total_batches;
            
            // Update average wait time
            let total_wait_time: u64 = request_timestamps.iter()
                .map(|ts| batch_start.duration_since(*ts).as_millis() as u64)
                .sum();
            
            let avg_wait_time = total_wait_time as f64 / batch_size as f64;
            stats_guard.average_wait_time_ms = 
                (stats_guard.average_wait_time_ms * (total_batches - 1.0) + avg_wait_time) / total_batches;
            
            // Update throughput (requests per second)
            let total_time_seconds = batch_start.elapsed().as_secs_f64();
            if total_time_seconds > 0.0 {
                stats_guard.throughput_requests_per_second = 
                    batch_size as f64 / total_time_seconds;
            }
        }
    }
    
    /// Get current batch statistics
    pub async fn stats(&self) -> BatchStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    /// Get current queue size
    pub async fn queue_size(&self) -> usize {
        let queue = self.queue.lock().await;
        queue.len()
    }
    
    /// Stop the batch processor
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        self.notify.notify_one();
    }
}

/// Batch scheduler that manages multiple queues
pub struct BatchScheduler {
    queues: std::collections::HashMap<String, Arc<BatchQueue>>,
    config: BatchConfig,
}

impl BatchScheduler {
    pub fn new(config: BatchConfig) -> Self {
        Self {
            queues: std::collections::HashMap::new(),
            config,
        }
    }
    
    /// Get or create a batch queue for a specific backend
    pub fn get_queue(&mut self, backend: &str) -> Arc<BatchQueue> {
        if let Some(queue) = self.queues.get(backend) {
            queue.clone()
        } else {
            let queue = Arc::new(BatchQueue::new(&self.config));
            self.queues.insert(backend.to_string(), queue.clone());
            queue
        }
    }
    
    /// Get statistics for all queues
    pub async fn all_stats(&self) -> std::collections::HashMap<String, BatchStats> {
        let mut all_stats = std::collections::HashMap::new();
        
        for (backend, queue) in &self.queues {
            all_stats.insert(backend.clone(), queue.stats().await);
        }
        
        all_stats
    }
    
    /// Stop all batch processors
    pub async fn stop_all(&self) {
        for queue in self.queues.values() {
            queue.stop().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::MockClient;
    
    #[tokio::test]
    async fn test_batch_queue_creation() {
        let config = BatchConfig {
            enabled: true,
            max_batch_size: 5,
            batch_timeout_ms: 100,
        };
        
        let queue = BatchQueue::new(&config);
        assert_eq!(queue.queue_size().await, 0);
    }
    
    #[tokio::test]
    async fn test_batch_scheduler() {
        let config = BatchConfig {
            enabled: true,
            max_batch_size: 5,
            batch_timeout_ms: 100,
        };
        
        let mut scheduler = BatchScheduler::new(config);
        
        let queue1 = scheduler.get_queue("backend1");
        let queue2 = scheduler.get_queue("backend2");
        let queue1_again = scheduler.get_queue("backend1");
        
        // Should reuse the same queue for the same backend
        assert!(Arc::ptr_eq(&queue1, &queue1_again));
        assert!(!Arc::ptr_eq(&queue1, &queue2));
        
        scheduler.stop_all().await;
    }
    
    #[tokio::test]
    async fn test_batch_stats() {
        let config = BatchConfig {
            enabled: true,
            max_batch_size: 2,
            batch_timeout_ms: 50,
        };
        
        let queue = BatchQueue::new(&config);
        
        // Initial stats should be zero
        let stats = queue.stats().await;
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_batches, 0);
        
        queue.stop().await;
    }
}