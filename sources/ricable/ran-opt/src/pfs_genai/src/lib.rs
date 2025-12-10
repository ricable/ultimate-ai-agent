//! PFS-GenAI: Generative AI Abstraction Service
//! 
//! This module provides an efficient abstraction layer for LLM integration with:
//! - Async client traits for multiple backends
//! - Token-efficient prompt engineering
//! - Response caching with semantic similarity
//! - Streaming response handlers
//! - Connection pooling and batch inference

pub mod client;
pub mod cache;
pub mod compression;
pub mod streaming;
pub mod backends;
pub mod batch;
pub mod metrics;

use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GenAIError {
    #[error("Backend error: {0}")]
    Backend(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Compression error: {0}")]
    Compression(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Invalid configuration: {0}")]
    Config(String),
    
    #[error("Token limit exceeded: current {current}, max {max}")]
    TokenLimit { current: usize, max: usize },
}

pub type Result<T> = std::result::Result<T, GenAIError>;

/// Core configuration for the GenAI service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenAIConfig {
    /// Maximum tokens per request
    pub max_tokens: usize,
    
    /// Default temperature for generation
    pub temperature: f32,
    
    /// Cache configuration
    pub cache: CacheConfig,
    
    /// Batch inference configuration
    pub batch: BatchConfig,
    
    /// Connection pool size
    pub pool_size: usize,
    
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable semantic caching
    pub enabled: bool,
    
    /// Vector DB URL for semantic cache
    pub vector_db_url: String,
    
    /// Similarity threshold for cache hits
    pub similarity_threshold: f32,
    
    /// Cache TTL in seconds
    pub ttl_secs: u64,
    
    /// Maximum cache size in MB
    pub max_size_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Enable batch inference
    pub enabled: bool,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
}

/// A prompt that can be sent to an LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    /// System message (optional)
    pub system: Option<String>,
    
    /// User message
    pub user: String,
    
    /// Additional context (optional)
    pub context: Option<Vec<String>>,
    
    /// Maximum tokens for response
    pub max_tokens: Option<usize>,
    
    /// Temperature for generation
    pub temperature: Option<f32>,
}

/// Response from an LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Generated text
    pub text: String,
    
    /// Token usage statistics
    pub usage: TokenUsage,
    
    /// Model used for generation
    pub model: String,
    
    /// Response metadata
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Tokens in the prompt
    pub prompt_tokens: usize,
    
    /// Tokens in the completion
    pub completion_tokens: usize,
    
    /// Total tokens used
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Time taken for inference in milliseconds
    pub latency_ms: u64,
    
    /// Whether response was from cache
    pub cached: bool,
    
    /// Cache similarity score if applicable
    pub cache_similarity: Option<f32>,
    
    /// Whether response was batched
    pub batched: bool,
}

/// Main trait for LLM clients
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Generate a response for a single prompt
    async fn generate(&self, prompt: &Prompt) -> Result<Response>;
    
    /// Generate responses for multiple prompts (batch inference)
    async fn generate_batch(&self, prompts: &[Prompt]) -> Result<Vec<Response>>;
    
    /// Stream a response token by token
    async fn generate_stream(
        &self,
        prompt: &Prompt,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Send + Unpin>>;
    
    /// Get the model name
    fn model_name(&self) -> &str;
    
    /// Count tokens in a text
    async fn count_tokens(&self, text: &str) -> Result<usize>;
}

/// Main GenAI service that orchestrates all components
pub struct GenAIService {
    config: GenAIConfig,
    clients: dashmap::DashMap<String, Arc<dyn LLMClient>>,
    cache: Option<Arc<dyn cache::Cache>>,
    batch_queue: Option<Arc<batch::BatchQueue>>,
    metrics: Arc<metrics::Metrics>,
}

impl GenAIService {
    /// Create a new GenAI service with configuration
    pub async fn new(config: GenAIConfig) -> Result<Self> {
        let cache = if config.cache.enabled {
            Some(Arc::new(
                cache::SemanticCache::new(&config.cache).await?
            ) as Arc<dyn cache::Cache>)
        } else {
            None
        };
        
        let batch_queue = if config.batch.enabled {
            Some(Arc::new(batch::BatchQueue::new(&config.batch)))
        } else {
            None
        };
        
        let metrics = Arc::new(metrics::Metrics::new());
        
        Ok(Self {
            config,
            clients: dashmap::DashMap::new(),
            cache,
            batch_queue,
            metrics,
        })
    }
    
    /// Register a new LLM client backend
    pub fn register_client(&self, name: String, client: Arc<dyn LLMClient>) {
        self.clients.insert(name, client);
    }
    
    /// Generate a response using the specified backend
    pub async fn generate(&self, backend: &str, prompt: &Prompt) -> Result<Response> {
        let client = self.clients
            .get(backend)
            .ok_or_else(|| GenAIError::Config(format!("Backend '{}' not found", backend)))?;
        
        // Check cache first if enabled
        if let Some(cache) = &self.cache {
            if let Some(response) = cache.get(prompt).await? {
                self.metrics.record_cache_hit();
                return Ok(response);
            }
        }
        
        // Check if we should batch this request
        if let Some(batch_queue) = &self.batch_queue {
            if batch_queue.should_batch() {
                return batch_queue.enqueue(backend, prompt, client.clone()).await;
            }
        }
        
        // Generate response
        let start = std::time::Instant::now();
        let response = client.generate(prompt).await?;
        let latency = start.elapsed().as_millis() as u64;
        
        // Update metrics
        self.metrics.record_request(backend, latency, response.usage.total_tokens);
        
        // Cache the response if enabled
        if let Some(cache) = &self.cache {
            cache.put(prompt, &response).await?;
        }
        
        Ok(response)
    }
    
    /// Generate a streaming response
    pub async fn generate_stream(
        &self,
        backend: &str,
        prompt: &Prompt,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String>> + Send + Unpin>> {
        let client = self.clients
            .get(backend)
            .ok_or_else(|| GenAIError::Config(format!("Backend '{}' not found", backend)))?;
        
        client.generate_stream(prompt).await
    }
    
    /// Get service metrics
    pub fn metrics(&self) -> &metrics::Metrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prompt_creation() {
        let prompt = Prompt {
            system: Some("You are a helpful assistant.".to_string()),
            user: "What is Rust?".to_string(),
            context: None,
            max_tokens: Some(100),
            temperature: Some(0.7),
        };
        
        assert_eq!(prompt.user, "What is Rust?");
        assert_eq!(prompt.max_tokens, Some(100));
    }
}