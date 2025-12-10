//! Backend implementations for different LLM providers

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tokio_stream::wrappers::ReceiverStream;
use tokio::sync::mpsc;

use crate::{
    LLMClient, Prompt, Response, Result, GenAIError, TokenUsage, ResponseMetadata,
    streaming::{StreamingToken, TokenMetadata},
};

/// Configuration for OpenAI backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub max_retries: u32,
    pub request_timeout_secs: u64,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: "".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: 4096,
            temperature: 0.7,
            max_retries: 3,
            request_timeout_secs: 30,
        }
    }
}

/// OpenAI API request format
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

/// OpenAI API response format
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: Option<OpenAIMessage>,
    delta: Option<OpenAIMessage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// OpenAI streaming response
#[derive(Debug, Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamDelta {
    content: Option<String>,
}

/// OpenAI backend implementation
pub struct OpenAIBackend {
    config: OpenAIConfig,
    client: Client,
    semaphore: Arc<Semaphore>,
}

impl OpenAIBackend {
    pub fn new(config: OpenAIConfig, pool_size: usize) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            config,
            client,
            semaphore: Arc::new(Semaphore::new(pool_size)),
        }
    }
    
    fn create_messages(&self, prompt: &Prompt) -> Vec<OpenAIMessage> {
        let mut messages = Vec::new();
        
        if let Some(system) = &prompt.system {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }
        
        messages.push(OpenAIMessage {
            role: "user".to_string(),
            content: prompt.user.clone(),
        });
        
        if let Some(context) = &prompt.context {
            for ctx in context {
                messages.push(OpenAIMessage {
                    role: "user".to_string(),
                    content: ctx.clone(),
                });
            }
        }
        
        messages
    }
    
    async fn make_request(&self, request: OpenAIRequest) -> Result<OpenAIResponse> {
        let _permit = self.semaphore.acquire().await.map_err(|_| {
            GenAIError::Backend("Failed to acquire semaphore".to_string())
        })?;
        
        let response = self.client
            .post(&format!("{}/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(GenAIError::Backend(format!("OpenAI API error: {}", error_text)));
        }
        
        let openai_response: OpenAIResponse = response.json().await.map_err(|e| {
            GenAIError::Backend(format!("Failed to parse OpenAI response: {}", e))
        })?;
        
        Ok(openai_response)
    }
}

#[async_trait]
impl LLMClient for OpenAIBackend {
    async fn generate(&self, prompt: &Prompt) -> Result<Response> {
        let start = Instant::now();
        
        let messages = self.create_messages(prompt);
        
        let request = OpenAIRequest {
            model: self.config.model.clone(),
            messages,
            max_tokens: prompt.max_tokens.or(Some(self.config.max_tokens)),
            temperature: prompt.temperature.or(Some(self.config.temperature)),
            stream: false,
        };
        
        let openai_response = self.make_request(request).await?;
        
        let choice = openai_response.choices.into_iter().next()
            .ok_or_else(|| GenAIError::Backend("No choices in response".to_string()))?;
        
        let message = choice.message
            .ok_or_else(|| GenAIError::Backend("No message in choice".to_string()))?;
        
        let usage = openai_response.usage.unwrap_or(OpenAIUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        });
        
        let latency = start.elapsed().as_millis() as u64;
        
        Ok(Response {
            text: message.content,
            usage: TokenUsage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            },
            model: openai_response.model,
            metadata: ResponseMetadata {
                latency_ms: latency,
                cached: false,
                cache_similarity: None,
                batched: false,
            },
        })
    }
    
    async fn generate_batch(&self, prompts: &[Prompt]) -> Result<Vec<Response>> {
        let mut responses = Vec::with_capacity(prompts.len());
        
        // For now, process sequentially. Could be optimized with concurrent requests
        for prompt in prompts {
            let mut response = self.generate(prompt).await?;
            response.metadata.batched = true;
            responses.push(response);
        }
        
        Ok(responses)
    }
    
    async fn generate_stream(
        &self,
        prompt: &Prompt,
    ) -> Result<Box<dyn Stream<Item = Result<String>> + Send + Unpin>> {
        let messages = self.create_messages(prompt);
        
        let request = OpenAIRequest {
            model: self.config.model.clone(),
            messages,
            max_tokens: prompt.max_tokens.or(Some(self.config.max_tokens)),
            temperature: prompt.temperature.or(Some(self.config.temperature)),
            stream: true,
        };
        
        let (tx, rx) = mpsc::channel(100);
        
        let client = self.client.clone();
        let config = self.config.clone();
        let semaphore = self.semaphore.clone();
        
        tokio::spawn(async move {
            let _permit = semaphore.acquire().await.expect("Failed to acquire semaphore");
            
            let response = match client
                .post(&format!("{}/chat/completions", config.base_url))
                .header("Authorization", format!("Bearer {}", config.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
            {
                Ok(response) => response,
                Err(e) => {
                    let _ = tx.send(Err(GenAIError::Network(e))).await;
                    return;
                }
            };
            
            if !response.status().is_success() {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                let _ = tx.send(Err(GenAIError::Backend(format!("OpenAI API error: {}", error_text)))).await;
                return;
            }
            
            let mut bytes_stream = response.bytes_stream();
            let mut buffer = Vec::new();
            
            while let Some(chunk) = bytes_stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.extend_from_slice(&bytes);
                        
                        // Process complete lines
                        while let Some(line_end) = buffer.iter().position(|&b| b == b'\n') {
                            let line = buffer.drain(..=line_end).collect::<Vec<_>>();
                            let line_str = String::from_utf8_lossy(&line);
                            
                            if line_str.starts_with("data: ") {
                                let data = &line_str[6..];
                                if data.trim() == "[DONE]" {
                                    return;
                                }
                                
                                if let Ok(stream_response) = serde_json::from_str::<OpenAIStreamResponse>(data) {
                                    if let Some(choice) = stream_response.choices.into_iter().next() {
                                        if let Some(content) = choice.delta.content {
                                            if tx.send(Ok(content)).await.is_err() {
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(GenAIError::Network(e))).await;
                        return;
                    }
                }
            }
        });
        
        Ok(Box::new(ReceiverStream::new(rx)))
    }
    
    fn model_name(&self) -> &str {
        &self.config.model
    }
    
    async fn count_tokens(&self, text: &str) -> Result<usize> {
        // Simple approximation for now
        // In production, you'd use tiktoken or similar
        Ok(text.len() / 4)
    }
}

/// Configuration for local model backend using Candle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub device: String, // "cpu" or "cuda"
}

impl Default for LocalModelConfig {
    fn default() -> Self {
        Self {
            model_path: "".to_string(),
            tokenizer_path: "".to_string(),
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            device: "cpu".to_string(),
        }
    }
}

/// Local model backend using Candle for inference
pub struct LocalModelBackend {
    config: LocalModelConfig,
    // TODO: Add candle model and tokenizer fields
    // model: Arc<dyn candle_core::Model>,
    // tokenizer: Arc<tokenizers::Tokenizer>,
    semaphore: Arc<Semaphore>,
}

impl LocalModelBackend {
    pub fn new(config: LocalModelConfig, pool_size: usize) -> Result<Self> {
        // TODO: Load model and tokenizer using Candle
        // For now, just create the structure
        
        Ok(Self {
            config,
            semaphore: Arc::new(Semaphore::new(pool_size)),
        })
    }
}

#[async_trait]
impl LLMClient for LocalModelBackend {
    async fn generate(&self, prompt: &Prompt) -> Result<Response> {
        let _permit = self.semaphore.acquire().await.map_err(|_| {
            GenAIError::Backend("Failed to acquire semaphore".to_string())
        })?;
        
        let start = Instant::now();
        
        // TODO: Implement actual model inference using Candle
        // For now, return a mock response
        
        let response_text = format!("Local model response to: {}", 
            prompt.user.chars().take(50).collect::<String>());
        
        let prompt_tokens = self.count_tokens(&prompt.user).await?;
        let completion_tokens = self.count_tokens(&response_text).await?;
        
        let latency = start.elapsed().as_millis() as u64;
        
        Ok(Response {
            text: response_text,
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            model: "local-model".to_string(),
            metadata: ResponseMetadata {
                latency_ms: latency,
                cached: false,
                cache_similarity: None,
                batched: false,
            },
        })
    }
    
    async fn generate_batch(&self, prompts: &[Prompt]) -> Result<Vec<Response>> {
        let mut responses = Vec::with_capacity(prompts.len());
        
        for prompt in prompts {
            let mut response = self.generate(prompt).await?;
            response.metadata.batched = true;
            responses.push(response);
        }
        
        Ok(responses)
    }
    
    async fn generate_stream(
        &self,
        prompt: &Prompt,
    ) -> Result<Box<dyn Stream<Item = Result<String>> + Send + Unpin>> {
        // TODO: Implement streaming inference
        // For now, just split the response into chunks
        
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
        "local-model"
    }
    
    async fn count_tokens(&self, text: &str) -> Result<usize> {
        // TODO: Use actual tokenizer
        Ok(text.len() / 4)
    }
}

/// Factory for creating backend instances
pub struct BackendFactory;

impl BackendFactory {
    /// Create an OpenAI backend
    pub fn create_openai(config: OpenAIConfig, pool_size: usize) -> Arc<dyn LLMClient> {
        Arc::new(OpenAIBackend::new(config, pool_size))
    }
    
    /// Create a local model backend
    pub fn create_local_model(config: LocalModelConfig, pool_size: usize) -> Result<Arc<dyn LLMClient>> {
        Ok(Arc::new(LocalModelBackend::new(config, pool_size)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::MockClient;
    
    #[test]
    fn test_openai_config_default() {
        let config = OpenAIConfig::default();
        assert_eq!(config.model, "gpt-3.5-turbo");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
    }
    
    #[test]
    fn test_local_model_config_default() {
        let config = LocalModelConfig::default();
        assert_eq!(config.device, "cpu");
        assert_eq!(config.max_tokens, 2048);
    }
    
    #[tokio::test]
    async fn test_backend_factory() {
        let config = OpenAIConfig::default();
        let backend = BackendFactory::create_openai(config, 10);
        
        assert_eq!(backend.model_name(), "gpt-3.5-turbo");
    }
}