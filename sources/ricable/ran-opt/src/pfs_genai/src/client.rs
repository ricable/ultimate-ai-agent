//! Async LLM client traits and base implementations

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Semaphore;
use futures::Stream;
use crate::{LLMClient, Prompt, Response, Result, GenAIError, TokenUsage, ResponseMetadata};

/// Connection pool for managing concurrent requests
pub struct ConnectionPool {
    /// Semaphore to limit concurrent connections
    semaphore: Arc<Semaphore>,
    
    /// Maximum concurrent connections
    max_connections: usize,
}

impl ConnectionPool {
    pub fn new(max_connections: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_connections)),
            max_connections,
        }
    }
    
    pub async fn acquire(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.semaphore.acquire().await.expect("Semaphore closed")
    }
    
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

/// Base client with connection pooling and retry logic
pub struct BaseClient {
    pool: ConnectionPool,
    timeout: std::time::Duration,
    max_retries: u32,
}

impl BaseClient {
    pub fn new(pool_size: usize, timeout_secs: u64, max_retries: u32) -> Self {
        Self {
            pool: ConnectionPool::new(pool_size),
            timeout: std::time::Duration::from_secs(timeout_secs),
            max_retries,
        }
    }
    
    /// Execute a request with retry logic
    pub async fn execute_with_retry<F, T>(&self, mut f: F) -> Result<T>
    where
        F: FnMut() -> futures::future::BoxFuture<'static, Result<T>>,
    {
        let mut attempts = 0;
        loop {
            match f().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.max_retries {
                        return Err(e);
                    }
                    
                    // Exponential backoff
                    let delay = std::time::Duration::from_millis(
                        100 * (2_u64.pow(attempts))
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
}

/// Mock client for testing
pub struct MockClient {
    model: String,
    base: BaseClient,
}

impl MockClient {
    pub fn new(model: String, pool_size: usize) -> Self {
        Self {
            model,
            base: BaseClient::new(pool_size, 30, 3),
        }
    }
}

#[async_trait]
impl LLMClient for MockClient {
    async fn generate(&self, prompt: &Prompt) -> Result<Response> {
        let _permit = self.base.pool.acquire().await;
        
        // Simulate processing time
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        let response_text = format!(
            "Mock response to: {}",
            prompt.user.chars().take(50).collect::<String>()
        );
        
        let prompt_tokens = self.count_tokens(&prompt.user).await?;
        let completion_tokens = self.count_tokens(&response_text).await?;
        
        Ok(Response {
            text: response_text,
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            model: self.model.clone(),
            metadata: ResponseMetadata {
                latency_ms: 100,
                cached: false,
                cache_similarity: None,
                batched: false,
            },
        })
    }
    
    async fn generate_batch(&self, prompts: &[Prompt]) -> Result<Vec<Response>> {
        let _permit = self.base.pool.acquire().await;
        
        let mut responses = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            responses.push(self.generate(prompt).await?);
        }
        
        Ok(responses)
    }
    
    async fn generate_stream(
        &self,
        prompt: &Prompt,
    ) -> Result<Box<dyn Stream<Item = Result<String>> + Send + Unpin>> {
        let response = self.generate(prompt).await?;
        let words: Vec<String> = response.text
            .split_whitespace()
            .map(|s| format!("{} ", s))
            .collect();
        
        Ok(Box::new(futures::stream::iter(
            words.into_iter().map(Ok)
        )))
    }
    
    fn model_name(&self) -> &str {
        &self.model
    }
    
    async fn count_tokens(&self, text: &str) -> Result<usize> {
        // Simple approximation: ~4 chars per token
        Ok(text.len() / 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mock_client() {
        let client = MockClient::new("mock-model".to_string(), 10);
        
        let prompt = Prompt {
            system: None,
            user: "Test prompt".to_string(),
            context: None,
            max_tokens: None,
            temperature: None,
        };
        
        let response = client.generate(&prompt).await.unwrap();
        assert!(response.text.contains("Mock response"));
        assert_eq!(response.model, "mock-model");
    }
    
    #[tokio::test]
    async fn test_connection_pool() {
        let pool = ConnectionPool::new(2);
        
        let permit1 = pool.acquire().await;
        let permit2 = pool.acquire().await;
        
        assert_eq!(pool.available_permits(), 0);
        
        drop(permit1);
        assert_eq!(pool.available_permits(), 1);
        
        drop(permit2);
        assert_eq!(pool.available_permits(), 2);
    }
}