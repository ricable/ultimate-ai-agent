//! Streaming response handlers for real-time LLM output

use futures::{Stream, StreamExt, TryStreamExt};
use pin_project::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use serde::{Deserialize, Serialize};
use crate::{Result, GenAIError};

/// A streaming token from the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingToken {
    /// The text content of the token
    pub content: String,
    
    /// Position in the response
    pub position: usize,
    
    /// Whether this is the final token
    pub is_final: bool,
    
    /// Metadata about the token
    pub metadata: TokenMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMetadata {
    /// Timestamp when token was generated
    pub timestamp: u64,
    
    /// Confidence score (if available)
    pub confidence: Option<f32>,
    
    /// Token ID (if available)
    pub token_id: Option<u32>,
}

/// Configuration for streaming responses
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for streaming tokens
    pub buffer_size: usize,
    
    /// Timeout for individual tokens in milliseconds
    pub token_timeout_ms: u64,
    
    /// Whether to include metadata in tokens
    pub include_metadata: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            token_timeout_ms: 5000,
            include_metadata: true,
        }
    }
}

/// A buffered stream that collects tokens and emits them in chunks
#[pin_project]
pub struct BufferedTokenStream {
    #[pin]
    source: Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>>,
    buffer: Vec<StreamingToken>,
    buffer_size: usize,
    position: usize,
}

impl BufferedTokenStream {
    pub fn new(
        source: Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>>,
        buffer_size: usize,
    ) -> Self {
        Self {
            source,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
            position: 0,
        }
    }
}

impl Stream for BufferedTokenStream {
    type Item = Result<Vec<StreamingToken>>;
    
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        
        loop {
            match this.source.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(token))) => {
                    this.buffer.push(token.clone());
                    
                    if this.buffer.len() >= *this.buffer_size || token.is_final {
                        let tokens = std::mem::take(this.buffer);
                        return Poll::Ready(Some(Ok(tokens)));
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    if !this.buffer.is_empty() {
                        let tokens = std::mem::take(this.buffer);
                        return Poll::Ready(Some(Ok(tokens)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => {
                    if !this.buffer.is_empty() && this.buffer.len() < *this.buffer_size {
                        // Wait for more tokens or timeout
                        return Poll::Pending;
                    }
                    return Poll::Pending;
                }
            }
        }
    }
}

/// A stream that aggregates tokens into complete text chunks
#[pin_project]
pub struct TextAggregatorStream {
    #[pin]
    source: Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>>,
    accumulated_text: String,
    chunk_size: usize,
}

impl TextAggregatorStream {
    pub fn new(
        source: Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>>,
        chunk_size: usize,
    ) -> Self {
        Self {
            source,
            accumulated_text: String::new(),
            chunk_size,
        }
    }
}

impl Stream for TextAggregatorStream {
    type Item = Result<String>;
    
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        
        loop {
            match this.source.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(token))) => {
                    this.accumulated_text.push_str(&token.content);
                    
                    if this.accumulated_text.len() >= *this.chunk_size || token.is_final {
                        let text = std::mem::take(this.accumulated_text);
                        return Poll::Ready(Some(Ok(text)));
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    if !this.accumulated_text.is_empty() {
                        let text = std::mem::take(this.accumulated_text);
                        return Poll::Ready(Some(Ok(text)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// A stream processor that applies transformations to streaming tokens
pub struct StreamProcessor {
    config: StreamingConfig,
}

impl StreamProcessor {
    pub fn new(config: StreamingConfig) -> Self {
        Self { config }
    }
    
    /// Create a buffered stream from a token stream
    pub fn buffered(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>>,
    ) -> BufferedTokenStream {
        BufferedTokenStream::new(stream, self.config.buffer_size)
    }
    
    /// Create a text aggregator stream
    pub fn text_aggregator(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>>,
        chunk_size: usize,
    ) -> TextAggregatorStream {
        TextAggregatorStream::new(stream, chunk_size)
    }
    
    /// Filter tokens based on confidence threshold
    pub fn filter_by_confidence(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>>,
        threshold: f32,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>> {
        Box::pin(stream.try_filter(move |token| {
            futures::future::ready(
                token.metadata.confidence.map_or(true, |conf| conf >= threshold)
            )
        }))
    }
    
    /// Rate limit the stream to emit tokens at a specific rate
    pub fn rate_limit(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>>,
        tokens_per_second: u64,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>> {
        let interval = std::time::Duration::from_millis(1000 / tokens_per_second);
        
        Box::pin(stream.then(move |token| {
            let interval = interval;
            async move {
                tokio::time::sleep(interval).await;
                token
            }
        }))
    }
}

/// A mock streaming source for testing
pub struct MockStreamingSource {
    text: String,
    position: usize,
    delay_ms: u64,
}

impl MockStreamingSource {
    pub fn new(text: String, delay_ms: u64) -> Self {
        Self {
            text,
            position: 0,
            delay_ms,
        }
    }
    
    pub fn into_stream(self) -> Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>> {
        let (tx, rx) = mpsc::channel(100);
        
        tokio::spawn(async move {
            let words: Vec<&str> = self.text.split_whitespace().collect();
            
            for (i, word) in words.iter().enumerate() {
                let token = StreamingToken {
                    content: format!("{} ", word),
                    position: i,
                    is_final: i == words.len() - 1,
                    metadata: TokenMetadata {
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64,
                        confidence: Some(0.9),
                        token_id: Some(i as u32),
                    },
                };
                
                if tx.send(Ok(token)).await.is_err() {
                    break;
                }
                
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
        });
        
        Box::pin(ReceiverStream::new(rx))
    }
}

/// Utility functions for stream handling
pub mod utils {
    use super::*;
    
    /// Convert a simple string stream to a token stream
    pub fn string_stream_to_tokens(
        stream: Pin<Box<dyn Stream<Item = Result<String>> + Send>>,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>> {
        Box::pin(stream.enumerate().map(|(i, result)| {
            result.map(|content| StreamingToken {
                content,
                position: i,
                is_final: false, // We don't know if it's final from a string stream
                metadata: TokenMetadata {
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                    confidence: None,
                    token_id: None,
                },
            })
        }))
    }
    
    /// Collect all tokens from a stream into a single string
    pub async fn collect_stream_to_string(
        stream: Pin<Box<dyn Stream<Item = Result<StreamingToken>> + Send>>,
    ) -> Result<String> {
        let tokens: Vec<StreamingToken> = stream.try_collect().await?;
        Ok(tokens.into_iter().map(|t| t.content).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    
    #[tokio::test]
    async fn test_mock_streaming_source() {
        let source = MockStreamingSource::new("Hello world test".to_string(), 10);
        let stream = source.into_stream();
        
        let tokens: Vec<StreamingToken> = stream.try_collect().await.unwrap();
        
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].content, "Hello ");
        assert_eq!(tokens[1].content, "world ");
        assert_eq!(tokens[2].content, "test ");
        assert!(tokens[2].is_final);
    }
    
    #[tokio::test]
    async fn test_buffered_token_stream() {
        let source = MockStreamingSource::new("Hello world test stream".to_string(), 1);
        let stream = source.into_stream();
        
        let processor = StreamProcessor::new(StreamingConfig::default());
        let mut buffered = processor.buffered(stream);
        
        let first_chunk = buffered.next().await.unwrap().unwrap();
        assert!(!first_chunk.is_empty());
    }
    
    #[tokio::test]
    async fn test_text_aggregator_stream() {
        let source = MockStreamingSource::new("Hello world test".to_string(), 1);
        let stream = source.into_stream();
        
        let processor = StreamProcessor::new(StreamingConfig::default());
        let mut aggregator = processor.text_aggregator(stream, 50);
        
        let text = aggregator.next().await.unwrap().unwrap();
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(text.contains("test"));
    }
    
    #[tokio::test]
    async fn test_stream_utils() {
        let source = MockStreamingSource::new("Hello world".to_string(), 1);
        let stream = source.into_stream();
        
        let collected = utils::collect_stream_to_string(stream).await.unwrap();
        assert_eq!(collected, "Hello world ");
    }
}