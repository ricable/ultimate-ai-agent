//! Response caching with semantic similarity using vector database

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use blake3::Hash;
use crate::{Prompt, Response, Result, GenAIError, CacheConfig};

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cached response
    pub response: Response,
    
    /// Timestamp when cached
    pub timestamp: u64,
    
    /// Hash of the original prompt
    pub prompt_hash: Hash,
    
    /// Embedding vector for semantic similarity
    pub embedding: Vec<f32>,
}

/// Cache interface
#[async_trait]
pub trait Cache: Send + Sync {
    /// Get a cached response for a prompt
    async fn get(&self, prompt: &Prompt) -> Result<Option<Response>>;
    
    /// Store a response in the cache
    async fn put(&self, prompt: &Prompt, response: &Response) -> Result<()>;
    
    /// Clear the cache
    async fn clear(&self) -> Result<()>;
    
    /// Get cache statistics
    async fn stats(&self) -> CacheStats;
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub memory_usage_mb: f64,
}

/// In-memory cache with TTL
pub struct MemoryCache {
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl MemoryCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats {
                hits: 0,
                misses: 0,
                entries: 0,
                memory_usage_mb: 0.0,
            })),
        }
    }
    
    /// Generate a cache key from prompt
    fn generate_key(&self, prompt: &Prompt) -> String {
        let content = format!("{:?}", prompt);
        blake3::hash(content.as_bytes()).to_hex().to_string()
    }
    
    /// Check if entry is expired
    fn is_expired(&self, entry: &CacheEntry) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        now - entry.timestamp > self.config.ttl_secs
    }
    
    /// Clean expired entries
    async fn cleanup_expired(&self) {
        let mut cache = self.cache.write().await;
        let initial_size = cache.len();
        
        cache.retain(|_, entry| !self.is_expired(entry));
        
        if cache.len() != initial_size {
            self.update_stats().await;
        }
    }
    
    /// Update cache statistics
    async fn update_stats(&self) {
        let cache = self.cache.read().await;
        let mut stats = self.stats.write().await;
        
        stats.entries = cache.len();
        
        // Estimate memory usage
        let avg_entry_size = if cache.is_empty() {
            0.0
        } else {
            // Rough estimate: 1KB per entry + response size
            let total_size: usize = cache.values()
                .map(|entry| entry.response.text.len() + 1024)
                .sum();
            total_size as f64 / (1024.0 * 1024.0) // Convert to MB
        };
        
        stats.memory_usage_mb = avg_entry_size;
    }
}

#[async_trait]
impl Cache for MemoryCache {
    async fn get(&self, prompt: &Prompt) -> Result<Option<Response>> {
        self.cleanup_expired().await;
        
        let key = self.generate_key(prompt);
        let cache = self.cache.read().await;
        
        if let Some(entry) = cache.get(&key) {
            if !self.is_expired(entry) {
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                
                let mut response = entry.response.clone();
                response.metadata.cached = true;
                return Ok(Some(response));
            }
        }
        
        let mut stats = self.stats.write().await;
        stats.misses += 1;
        Ok(None)
    }
    
    async fn put(&self, prompt: &Prompt, response: &Response) -> Result<()> {
        let key = self.generate_key(prompt);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let entry = CacheEntry {
            response: response.clone(),
            timestamp,
            prompt_hash: blake3::hash(format!("{:?}", prompt).as_bytes()),
            embedding: vec![], // TODO: Generate actual embedding
        };
        
        let mut cache = self.cache.write().await;
        cache.insert(key, entry);
        
        self.update_stats().await;
        Ok(())
    }
    
    async fn clear(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        
        let mut stats = self.stats.write().await;
        stats.entries = 0;
        stats.memory_usage_mb = 0.0;
        
        Ok(())
    }
    
    async fn stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
}

/// Semantic cache using vector database for similarity search
pub struct SemanticCache {
    memory_cache: MemoryCache,
    config: CacheConfig,
    // TODO: Add qdrant client for vector similarity search
}

impl SemanticCache {
    pub async fn new(config: &CacheConfig) -> Result<Self> {
        // TODO: Initialize qdrant client
        Ok(Self {
            memory_cache: MemoryCache::new(config.clone()),
            config: config.clone(),
        })
    }
    
    /// Generate embedding for text (placeholder)
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // TODO: Use a proper embedding model
        // For now, return a simple hash-based vector
        let hash = blake3::hash(text.as_bytes());
        let bytes = hash.as_bytes();
        
        Ok(bytes.chunks(4)
            .map(|chunk| {
                let mut arr = [0u8; 4];
                for (i, &b) in chunk.iter().enumerate() {
                    if i < 4 { arr[i] = b; }
                }
                f32::from_ne_bytes(arr)
            })
            .take(128) // Use 128-dimensional vectors
            .collect())
    }
    
    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (magnitude_a * magnitude_b)
    }
    
    /// Find similar prompts in cache
    async fn find_similar(&self, prompt: &Prompt) -> Result<Option<(Response, f32)>> {
        let prompt_text = format!("{} {}", 
            prompt.system.as_deref().unwrap_or(""),
            prompt.user
        );
        
        let query_embedding = self.generate_embedding(&prompt_text).await?;
        
        // Search in memory cache for similar embeddings
        let cache = self.memory_cache.cache.read().await;
        let mut best_match: Option<(Response, f32)> = None;
        
        for entry in cache.values() {
            if self.memory_cache.is_expired(entry) {
                continue;
            }
            
            let similarity = self.cosine_similarity(&query_embedding, &entry.embedding);
            
            if similarity >= self.config.similarity_threshold {
                if let Some((_, best_similarity)) = &best_match {
                    if similarity > *best_similarity {
                        let mut response = entry.response.clone();
                        response.metadata.cached = true;
                        response.metadata.cache_similarity = Some(similarity);
                        best_match = Some((response, similarity));
                    }
                } else {
                    let mut response = entry.response.clone();
                    response.metadata.cached = true;
                    response.metadata.cache_similarity = Some(similarity);
                    best_match = Some((response, similarity));
                }
            }
        }
        
        Ok(best_match)
    }
}

#[async_trait]
impl Cache for SemanticCache {
    async fn get(&self, prompt: &Prompt) -> Result<Option<Response>> {
        // First try exact match
        if let Some(response) = self.memory_cache.get(prompt).await? {
            return Ok(Some(response));
        }
        
        // Then try semantic similarity
        if let Some((response, _)) = self.find_similar(prompt).await? {
            return Ok(Some(response));
        }
        
        Ok(None)
    }
    
    async fn put(&self, prompt: &Prompt, response: &Response) -> Result<()> {
        // Generate embedding for semantic search
        let prompt_text = format!("{} {}", 
            prompt.system.as_deref().unwrap_or(""),
            prompt.user
        );
        
        let embedding = self.generate_embedding(&prompt_text).await?;
        
        // Store in memory cache with embedding
        let key = self.memory_cache.generate_key(prompt);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let entry = CacheEntry {
            response: response.clone(),
            timestamp,
            prompt_hash: blake3::hash(format!("{:?}", prompt).as_bytes()),
            embedding,
        };
        
        let mut cache = self.memory_cache.cache.write().await;
        cache.insert(key, entry);
        
        self.memory_cache.update_stats().await;
        
        // TODO: Also store in vector database for persistent semantic search
        
        Ok(())
    }
    
    async fn clear(&self) -> Result<()> {
        self.memory_cache.clear().await
    }
    
    async fn stats(&self) -> CacheStats {
        self.memory_cache.stats().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TokenUsage, ResponseMetadata};
    
    #[tokio::test]
    async fn test_memory_cache() {
        let config = CacheConfig {
            enabled: true,
            vector_db_url: "".to_string(),
            similarity_threshold: 0.8,
            ttl_secs: 300,
            max_size_mb: 100,
        };
        
        let cache = MemoryCache::new(config);
        
        let prompt = Prompt {
            system: None,
            user: "Test prompt".to_string(),
            context: None,
            max_tokens: None,
            temperature: None,
        };
        
        let response = Response {
            text: "Test response".to_string(),
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
            model: "test-model".to_string(),
            metadata: ResponseMetadata {
                latency_ms: 100,
                cached: false,
                cache_similarity: None,
                batched: false,
            },
        };
        
        // Should miss initially
        assert!(cache.get(&prompt).await.unwrap().is_none());
        
        // Store response
        cache.put(&prompt, &response).await.unwrap();
        
        // Should hit now
        let cached_response = cache.get(&prompt).await.unwrap().unwrap();
        assert_eq!(cached_response.text, "Test response");
        assert!(cached_response.metadata.cached);
        
        // Check stats
        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
    }
}