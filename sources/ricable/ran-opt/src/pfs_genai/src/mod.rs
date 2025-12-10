//! PFS-GenAI: Generative AI Abstraction Service
//! 
//! This module provides a comprehensive abstraction layer for LLM integration with:
//! - Async client traits for multiple backends (OpenAI, local models)
//! - Token-efficient prompt engineering with compression algorithms
//! - Response caching with semantic similarity using vector databases
//! - Streaming response handlers for real-time output
//! - Batch inference queuing for efficient processing
//! - Connection pooling and comprehensive metrics
//! 
//! # Architecture
//! 
//! The service is designed with the following components:
//! 
//! ## Core Traits
//! - `LLMClient`: Main trait for all LLM backends
//! - `Cache`: Interface for response caching systems
//! - `PromptCompressor`: Interface for prompt optimization
//! 
//! ## Backend Support
//! - OpenAI API integration with streaming support
//! - Local model inference using Candle framework
//! - Extensible backend system for future providers
//! 
//! ## Optimization Features
//! - Semantic caching with vector similarity search
//! - Prompt compression using substitution and template patterns
//! - Batch processing with configurable queue management
//! - Connection pooling with semaphore-based limiting
//! 
//! ## Monitoring
//! - Comprehensive metrics with Prometheus integration
//! - Request latency and token usage tracking
//! - Cache hit/miss rates and batch processing stats
//! - Health checks and service status monitoring
//! 
//! # Usage Example
//! 
//! ```rust
//! use pfs_genai::{GenAIService, GenAIConfig, CacheConfig, BatchConfig};
//! use pfs_genai::backends::{OpenAIConfig, BackendFactory};
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Configure the service
//!     let config = GenAIConfig {
//!         max_tokens: 4096,
//!         temperature: 0.7,
//!         cache: CacheConfig {
//!             enabled: true,
//!             vector_db_url: "http://localhost:6333".to_string(),
//!             similarity_threshold: 0.8,
//!             ttl_secs: 3600,
//!             max_size_mb: 1024,
//!         },
//!         batch: BatchConfig {
//!             enabled: true,
//!             max_batch_size: 10,
//!             batch_timeout_ms: 100,
//!         },
//!         pool_size: 10,
//!         timeout_secs: 30,
//!     };
//! 
//!     // Create the service
//!     let service = GenAIService::new(config).await?;
//! 
//!     // Register backends
//!     let openai_config = OpenAIConfig {
//!         api_key: "your-api-key".to_string(),
//!         model: "gpt-3.5-turbo".to_string(),
//!         ..Default::default()
//!     };
//!     
//!     let openai_client = BackendFactory::create_openai(openai_config, 10);
//!     service.register_client("openai".to_string(), openai_client);
//! 
//!     // Use the service
//!     let prompt = Prompt {
//!         system: Some("You are a helpful assistant.".to_string()),
//!         user: "Explain how RAN optimization works.".to_string(),
//!         context: None,
//!         max_tokens: Some(500),
//!         temperature: Some(0.7),
//!     };
//! 
//!     let response = service.generate("openai", &prompt).await?;
//!     println!("Response: {}", response.text);
//! 
//!     Ok(())
//! }
//! ```

// Re-export main types and traits
pub use crate::{
    GenAIService, GenAIConfig, CacheConfig, BatchConfig,
    LLMClient, Prompt, Response, TokenUsage, ResponseMetadata,
    Result, GenAIError,
};

// Re-export module contents
pub use client::*;
pub use cache::*;
pub use compression::*;
pub use streaming::*;
pub use batch::*;
pub use metrics::*;

// Module declarations
pub mod client;
pub mod cache;
pub mod compression;
pub mod streaming;
pub mod backends;
pub mod batch;
pub mod metrics;

// Default implementations and utilities
pub mod utils {
    use super::*;
    
    /// Create a default configuration for development
    pub fn default_dev_config() -> GenAIConfig {
        GenAIConfig {
            max_tokens: 2048,
            temperature: 0.7,
            cache: CacheConfig {
                enabled: true,
                vector_db_url: "http://localhost:6333".to_string(),
                similarity_threshold: 0.8,
                ttl_secs: 1800, // 30 minutes
                max_size_mb: 512,
            },
            batch: BatchConfig {
                enabled: true,
                max_batch_size: 5,
                batch_timeout_ms: 100,
            },
            pool_size: 5,
            timeout_secs: 30,
        }
    }
    
    /// Create a default configuration for production
    pub fn default_prod_config() -> GenAIConfig {
        GenAIConfig {
            max_tokens: 4096,
            temperature: 0.7,
            cache: CacheConfig {
                enabled: true,
                vector_db_url: "http://qdrant:6333".to_string(),
                similarity_threshold: 0.85,
                ttl_secs: 3600, // 1 hour
                max_size_mb: 2048,
            },
            batch: BatchConfig {
                enabled: true,
                max_batch_size: 10,
                batch_timeout_ms: 50,
            },
            pool_size: 20,
            timeout_secs: 60,
        }
    }
    
    /// Create a minimal configuration for testing
    pub fn default_test_config() -> GenAIConfig {
        GenAIConfig {
            max_tokens: 100,
            temperature: 0.5,
            cache: CacheConfig {
                enabled: false,
                vector_db_url: "".to_string(),
                similarity_threshold: 0.9,
                ttl_secs: 300,
                max_size_mb: 10,
            },
            batch: BatchConfig {
                enabled: false,
                max_batch_size: 2,
                batch_timeout_ms: 10,
            },
            pool_size: 2,
            timeout_secs: 10,
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::client::MockClient;
    
    #[tokio::test]
    async fn test_service_integration() {
        let config = utils::default_test_config();
        let service = GenAIService::new(config).await.unwrap();
        
        // Register a mock client
        let mock_client = std::sync::Arc::new(MockClient::new("test-model".to_string(), 2));
        service.register_client("mock".to_string(), mock_client);
        
        // Test generation
        let prompt = Prompt {
            system: Some("Test system".to_string()),
            user: "Test user message".to_string(),
            context: None,
            max_tokens: Some(50),
            temperature: Some(0.5),
        };
        
        let response = service.generate("mock", &prompt).await.unwrap();
        assert!(!response.text.is_empty());
        assert_eq!(response.model, "test-model");
    }
    
    #[tokio::test]
    async fn test_service_with_compression() {
        let config = utils::default_test_config();
        let service = GenAIService::new(config).await.unwrap();
        
        // Test compression service
        let compressor = compression::CompressionService::new();
        
        let prompt = Prompt {
            system: Some("You are a helpful assistant".to_string()),
            user: "Please provide information about network performance".to_string(),
            context: None,
            max_tokens: None,
            temperature: None,
        };
        
        let (compressed_prompt, stats) = compressor.compress(&prompt, None).unwrap();
        assert!(stats.compression_ratio <= 1.0);
        assert!(compressed_prompt.user.contains("pls"));
        assert!(compressed_prompt.user.contains("info"));
    }
    
    #[tokio::test]
    async fn test_service_metrics() {
        let config = utils::default_test_config();
        let service = GenAIService::new(config).await.unwrap();
        
        // Register a mock client
        let mock_client = std::sync::Arc::new(MockClient::new("test-model".to_string(), 2));
        service.register_client("mock".to_string(), mock_client);
        
        // Make a request to generate metrics
        let prompt = Prompt {
            system: None,
            user: "Test".to_string(),
            context: None,
            max_tokens: None,
            temperature: None,
        };
        
        let _response = service.generate("mock", &prompt).await.unwrap();
        
        // Check metrics
        let metrics = service.metrics();
        let summary = metrics.get_summary().await;
        
        assert!(summary.uptime_seconds > 0);
        assert!(summary.backend_summaries.contains_key("mock"));
    }
}