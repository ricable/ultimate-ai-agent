//! Context Management for Micro-Experts
//!
//! This module handles context window management, including compression,
//! summarization, and efficient token management for WASM environments.

use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use wasm_bindgen::prelude::*;

/// Context window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum context window size in tokens
    pub max_tokens: usize,
    /// Token compression threshold (0.0 to 1.0)
    pub compression_threshold: f32,
    /// Enable context summarization
    pub enable_summarization: bool,
    /// Maximum summary length as ratio of original
    pub max_summary_ratio: f32,
    /// Enable importance tracking
    pub enable_importance_tracking: bool,
    /// Sliding window overlap size
    pub sliding_window_overlap: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 32_000, // Reduced from Kimi-K2's 128K for WASM
            compression_threshold: 0.8,
            enable_summarization: true,
            max_summary_ratio: 0.3,
            enable_importance_tracking: true,
            sliding_window_overlap: 512,
        }
    }
}

/// Token representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// Token text
    pub text: String,
    /// Token ID (for efficient processing)
    pub id: u32,
    /// Importance score (0.0 to 1.0)
    pub importance: f32,
    /// Position in the original sequence
    pub position: usize,
    /// Timestamp when added
    pub timestamp: f64,
    /// Context type (user, assistant, system, etc.)
    pub context_type: ContextType,
}

/// Context type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextType {
    /// User input
    User,
    /// Assistant response
    Assistant,
    /// System message
    System,
    /// Tool call
    ToolCall,
    /// Tool response
    ToolResponse,
    /// Internal processing
    Internal,
}

/// Context segment for compression and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSegment {
    /// Segment identifier
    pub id: u64,
    /// Tokens in this segment
    pub tokens: Vec<Token>,
    /// Compressed representation (if compressed)
    pub compressed_data: Option<Vec<u8>>,
    /// Summary of the segment
    pub summary: Option<String>,
    /// Importance score for the entire segment
    pub importance_score: f32,
    /// Creation timestamp
    pub created_at: f64,
    /// Last access timestamp
    pub last_accessed: f64,
    /// Access count
    pub access_count: u32,
    /// Whether this segment is compressed
    pub is_compressed: bool,
}

/// Importance tracker for context relevance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceTracker {
    /// Keyword frequency map
    keyword_frequency: HashMap<String, u32>,
    /// Recent topic importance
    recent_topics: VecDeque<(String, f32)>,
    /// Context type weights
    context_type_weights: HashMap<ContextType, f32>,
    /// Recency decay factor
    recency_decay: f32,
    /// Maximum tracked topics
    max_topics: usize,
}

impl Default for ImportanceTracker {
    fn default() -> Self {
        let mut context_type_weights = HashMap::new();
        context_type_weights.insert(ContextType::User, 1.0);
        context_type_weights.insert(ContextType::Assistant, 0.8);
        context_type_weights.insert(ContextType::System, 0.6);
        context_type_weights.insert(ContextType::ToolCall, 0.7);
        context_type_weights.insert(ContextType::ToolResponse, 0.7);
        context_type_weights.insert(ContextType::Internal, 0.3);

        Self {
            keyword_frequency: HashMap::new(),
            recent_topics: VecDeque::new(),
            context_type_weights,
            recency_decay: 0.95,
            max_topics: 50,
        }
    }
}

/// Context compression strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// Remove least important tokens
    LeastImportant,
    /// Sliding window with overlap
    SlidingWindow,
    /// Summarization-based compression
    Summarization,
    /// Hybrid approach
    Hybrid,
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        CompressionStrategy::Hybrid
    }
}

/// Context window manager
#[wasm_bindgen]
pub struct ContextWindow {
    /// Configuration
    config: ContextConfig,
    /// Current token count
    current_tokens: usize,
    /// Context segments
    segments: VecDeque<ContextSegment>,
    /// Importance tracker
    importance_tracker: ImportanceTracker,
    /// Compression strategy
    compression_strategy: CompressionStrategy,
    /// Next segment ID
    next_segment_id: u64,
    /// Statistics
    stats: ContextStats,
}

/// Context statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStats {
    /// Total tokens processed
    pub total_tokens_processed: u64,
    /// Current active tokens
    pub active_tokens: usize,
    /// Compressed tokens
    pub compressed_tokens: usize,
    /// Number of segments
    pub segment_count: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Average importance score
    pub avg_importance: f32,
    /// Number of compressions performed
    pub compression_count: u64,
}

impl Default for ContextStats {
    fn default() -> Self {
        Self {
            total_tokens_processed: 0,
            active_tokens: 0,
            compressed_tokens: 0,
            segment_count: 0,
            compression_ratio: 1.0,
            avg_importance: 0.5,
            compression_count: 0,
        }
    }
}

#[wasm_bindgen]
impl ContextWindow {
    /// Create a new context window
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<ContextWindow, JsValue> {
        let config: ContextConfig = serde_json::from_str(config_json)
            .unwrap_or_else(|_| ContextConfig::default());

        Ok(ContextWindow {
            config,
            current_tokens: 0,
            segments: VecDeque::new(),
            importance_tracker: ImportanceTracker::default(),
            compression_strategy: CompressionStrategy::default(),
            next_segment_id: 1,
            stats: ContextStats::default(),
        })
    }

    /// Add content to the context window
    #[wasm_bindgen]
    pub fn add_content(&mut self, content: &str, context_type: &str) -> Result<(), JsValue> {
        let ctx_type = match context_type {
            "user" => ContextType::User,
            "assistant" => ContextType::Assistant,
            "system" => ContextType::System,
            "tool_call" => ContextType::ToolCall,
            "tool_response" => ContextType::ToolResponse,
            _ => ContextType::Internal,
        };

        let tokens = self.tokenize_content(content, ctx_type)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.add_tokens(tokens)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(())
    }

    /// Get the current context as a string
    #[wasm_bindgen]
    pub fn get_context(&self) -> String {
        let mut context_parts = Vec::new();

        for segment in &self.segments {
            if segment.is_compressed {
                if let Some(summary) = &segment.summary {
                    context_parts.push(format!("[SUMMARY: {}]", summary));
                }
            } else {
                let segment_text: String = segment.tokens
                    .iter()
                    .map(|t| &t.text)
                    .collect::<Vec<_>>()
                    .join(" ");
                context_parts.push(segment_text);
            }
        }

        context_parts.join("\n")
    }

    /// Get context statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.stats)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Force compression of the context
    #[wasm_bindgen]
    pub fn compress_context(&mut self) -> Result<(), JsValue> {
        self.perform_compression()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Clear the context window
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.segments.clear();
        self.current_tokens = 0;
        self.importance_tracker = ImportanceTracker::default();
        self.next_segment_id = 1;
        self.update_stats();
    }

    /// Get context window utilization
    #[wasm_bindgen]
    pub fn get_utilization(&self) -> f32 {
        if self.config.max_tokens > 0 {
            self.current_tokens as f32 / self.config.max_tokens as f32
        } else {
            0.0
        }
    }

    /// Check if compression is needed
    #[wasm_bindgen]
    pub fn needs_compression(&self) -> bool {
        self.get_utilization() >= self.config.compression_threshold
    }

    /// Get most important segments
    #[wasm_bindgen]
    pub fn get_important_segments(&self, count: usize) -> Result<String, JsValue> {
        let mut segments_with_scores: Vec<_> = self.segments
            .iter()
            .map(|s| (s, s.importance_score))
            .collect();

        segments_with_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let important_segments: Vec<_> = segments_with_scores
            .into_iter()
            .take(count)
            .map(|(segment, score)| {
                let text = if segment.is_compressed {
                    segment.summary.as_deref().unwrap_or("[Compressed]")
                } else {
                    &segment.tokens
                        .iter()
                        .map(|t| &t.text)
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                format!("Score: {:.3} - {}", score, text)
            })
            .collect();

        Ok(important_segments.join("\n---\n"))
    }
}

impl ContextWindow {
    /// Tokenize content into tokens
    fn tokenize_content(&mut self, content: &str, context_type: ContextType) -> Result<Vec<Token>> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut tokens = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let importance = self.calculate_token_importance(word, context_type);
            
            let token = Token {
                text: word.to_string(),
                id: self.hash_token(word),
                importance,
                position: self.stats.total_tokens_processed as usize + i,
                timestamp: Utils::now(),
                context_type,
            };

            tokens.push(token);

            // Update importance tracker
            self.importance_tracker.update_keyword_frequency(word);
        }

        Ok(tokens)
    }

    /// Add tokens to the context window
    fn add_tokens(&mut self, tokens: Vec<Token>) -> Result<()> {
        // Check if we need to compress before adding
        while self.current_tokens + tokens.len() > self.config.max_tokens {
            if !self.try_compress_context()? {
                // If compression didn't help enough, remove oldest segments
                self.evict_oldest_segment()?;
            }
        }

        // Create a new segment for these tokens
        let segment = ContextSegment {
            id: self.next_segment_id,
            tokens: tokens.clone(),
            compressed_data: None,
            summary: None,
            importance_score: self.calculate_segment_importance(&tokens),
            created_at: Utils::now(),
            last_accessed: Utils::now(),
            access_count: 1,
            is_compressed: false,
        };

        self.segments.push_back(segment);
        self.next_segment_id += 1;
        self.current_tokens += tokens.len();
        self.stats.total_tokens_processed += tokens.len() as u64;

        self.update_stats();
        Ok(())
    }

    /// Calculate importance score for a token
    fn calculate_token_importance(&self, word: &str, context_type: ContextType) -> f32 {
        let mut importance = 0.5; // Base importance

        // Context type weighting
        if let Some(&weight) = self.importance_tracker.context_type_weights.get(&context_type) {
            importance *= weight;
        }

        // Frequency-based importance (rare words are more important)
        let frequency = self.importance_tracker.keyword_frequency
            .get(&word.to_lowercase())
            .copied()
            .unwrap_or(0);

        let frequency_factor = if frequency == 0 {
            1.0
        } else {
            1.0 / (1.0 + (frequency as f32).ln())
        };

        importance *= frequency_factor;

        // Length-based importance (longer words tend to be more important)
        let length_factor = (word.len() as f32 / 10.0).min(1.5);
        importance *= length_factor;

        // Special keyword detection
        let special_keywords = [
            "important", "critical", "urgent", "remember", "note", "warning",
            "error", "problem", "solution", "result", "conclusion"
        ];

        if special_keywords.iter().any(|&keyword| word.to_lowercase().contains(keyword)) {
            importance *= 1.5;
        }

        importance.min(1.0).max(0.0)
    }

    /// Calculate importance score for a segment
    fn calculate_segment_importance(&self, tokens: &[Token]) -> f32 {
        if tokens.is_empty() {
            return 0.0;
        }

        let total_importance: f32 = tokens.iter().map(|t| t.importance).sum();
        let avg_importance = total_importance / tokens.len() as f32;

        // Boost importance for certain context types
        let context_boost = tokens.iter()
            .map(|t| match t.context_type {
                ContextType::User => 1.2,
                ContextType::System => 1.1,
                ContextType::ToolCall | ContextType::ToolResponse => 1.15,
                _ => 1.0,
            })
            .fold(1.0, f32::max);

        // Boost for segments with high-importance tokens
        let high_importance_count = tokens.iter()
            .filter(|t| t.importance > 0.7)
            .count();

        let high_importance_boost = if high_importance_count > 0 {
            1.0 + (high_importance_count as f32 / tokens.len() as f32) * 0.3
        } else {
            1.0
        };

        (avg_importance * context_boost * high_importance_boost).min(1.0)
    }

    /// Try to compress the context
    fn try_compress_context(&mut self) -> Result<bool> {
        match self.compression_strategy {
            CompressionStrategy::None => Ok(false),
            CompressionStrategy::LeastImportant => self.compress_least_important(),
            CompressionStrategy::SlidingWindow => self.compress_sliding_window(),
            CompressionStrategy::Summarization => self.compress_with_summarization(),
            CompressionStrategy::Hybrid => self.compress_hybrid(),
        }
    }

    /// Perform overall compression
    fn perform_compression(&mut self) -> Result<()> {
        self.try_compress_context()?;
        self.stats.compression_count += 1;
        self.update_stats();
        Ok(())
    }

    /// Compress by removing least important tokens
    fn compress_least_important(&mut self) -> Result<bool> {
        let target_reduction = (self.current_tokens as f32 * 0.2) as usize; // Remove 20%
        let mut tokens_removed = 0;

        // Sort segments by importance (ascending)
        let mut segment_indices: Vec<_> = (0..self.segments.len()).collect();
        segment_indices.sort_by(|&a, &b| {
            let importance_a = self.segments[a].importance_score;
            let importance_b = self.segments[b].importance_score;
            importance_a.partial_cmp(&importance_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove tokens from least important segments
        for &index in &segment_indices {
            if tokens_removed >= target_reduction {
                break;
            }

            let segment = &mut self.segments[index];
            if !segment.is_compressed && segment.tokens.len() > 10 {
                // Remove least important tokens from this segment
                segment.tokens.sort_by(|a, b| a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal));
                
                let remove_count = (segment.tokens.len() / 4).min(target_reduction - tokens_removed);
                for _ in 0..remove_count {
                    if !segment.tokens.is_empty() {
                        segment.tokens.remove(0);
                        tokens_removed += 1;
                        self.current_tokens -= 1;
                    }
                }

                // Update segment importance
                segment.importance_score = self.calculate_segment_importance(&segment.tokens);
            }
        }

        Ok(tokens_removed > 0)
    }

    /// Compress using sliding window
    fn compress_sliding_window(&mut self) -> Result<bool> {
        if self.segments.len() <= 2 {
            return Ok(false);
        }

        // Compress older segments, keeping overlap
        let overlap_size = self.config.sliding_window_overlap;
        let mut compressed = false;

        // Process segments from oldest to newest, except the last two
        let compress_count = self.segments.len().saturating_sub(2);
        for i in 0..compress_count {
            let segment = &mut self.segments[i];
            if !segment.is_compressed {
                // Keep only the last 'overlap_size' tokens
                if segment.tokens.len() > overlap_size {
                    let tokens_to_remove = segment.tokens.len() - overlap_size;
                    segment.tokens.drain(0..tokens_to_remove);
                    self.current_tokens -= tokens_to_remove;
                    compressed = true;
                }
                
                segment.is_compressed = true;
                segment.compressed_data = Some(self.compress_segment_data(segment)?);
            }
        }

        Ok(compressed)
    }

    /// Compress using summarization
    fn compress_with_summarization(&mut self) -> Result<bool> {
        let mut compressed = false;

        for segment in &mut self.segments {
            if !segment.is_compressed && segment.tokens.len() > 50 {
                // Create a summary of the segment
                let summary = self.create_segment_summary(segment)?;
                let original_token_count = segment.tokens.len();
                
                // Replace tokens with summary tokens
                let summary_tokens = self.tokenize_content(&summary, ContextType::Internal)?;
                
                if summary_tokens.len() < original_token_count {
                    self.current_tokens -= original_token_count;
                    self.current_tokens += summary_tokens.len();
                    self.stats.compressed_tokens += original_token_count - summary_tokens.len();
                    
                    segment.summary = Some(summary);
                    segment.tokens = summary_tokens;
                    segment.is_compressed = true;
                    compressed = true;
                }
            }
        }

        Ok(compressed)
    }

    /// Hybrid compression strategy
    fn compress_hybrid(&mut self) -> Result<bool> {
        // First try summarization for large segments
        let summarization_result = self.compress_with_summarization()?;
        
        // Then apply sliding window to older segments
        let sliding_result = self.compress_sliding_window()?;
        
        // Finally, remove least important tokens if still needed
        let least_important_result = if self.needs_compression() {
            self.compress_least_important()?
        } else {
            false
        };

        Ok(summarization_result || sliding_result || least_important_result)
    }

    /// Create a summary of a segment
    fn create_segment_summary(&self, segment: &ContextSegment) -> Result<String> {
        let text: String = segment.tokens
            .iter()
            .map(|t| &t.text)
            .collect::<Vec<_>>()
            .join(" ");

        // Simple extractive summarization
        let sentences: Vec<&str> = text.split('.').collect();
        let target_sentences = ((sentences.len() as f32) * self.config.max_summary_ratio).ceil() as usize;
        
        if target_sentences >= sentences.len() {
            return Ok(text);
        }

        // Score sentences by importance (number of important tokens)
        let mut sentence_scores: Vec<(usize, f32)> = sentences
            .iter()
            .enumerate()
            .map(|(i, sentence)| {
                let score = sentence.split_whitespace()
                    .map(|word| {
                        segment.tokens
                            .iter()
                            .find(|t| t.text == word)
                            .map(|t| t.importance)
                            .unwrap_or(0.0)
                    })
                    .sum::<f32>() / sentence.split_whitespace().count().max(1) as f32;
                (i, score)
            })
            .collect();

        // Sort by score and take top sentences
        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut selected_indices: Vec<usize> = sentence_scores
            .into_iter()
            .take(target_sentences)
            .map(|(i, _)| i)
            .collect();
        
        selected_indices.sort();

        let summary = selected_indices
            .into_iter()
            .map(|i| sentences[i])
            .collect::<Vec<_>>()
            .join(".");

        Ok(summary)
    }

    /// Compress segment data
    fn compress_segment_data(&self, segment: &ContextSegment) -> Result<Vec<u8>> {
        let text: String = segment.tokens
            .iter()
            .map(|t| &t.text)
            .collect::<Vec<_>>()
            .join(" ");

        // Simple compression using LZ4
        lz4_flex::compress_prepend_size(text.as_bytes())
            .map_err(|e| KimiError::CompressionError(format!("Failed to compress segment: {}", e)))
    }

    /// Evict the oldest segment
    fn evict_oldest_segment(&mut self) -> Result<()> {
        if let Some(segment) = self.segments.pop_front() {
            self.current_tokens -= segment.tokens.len();
        }
        Ok(())
    }

    /// Hash a token for ID generation
    fn hash_token(&self, token: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.active_tokens = self.current_tokens;
        self.stats.segment_count = self.segments.len();
        
        // Calculate compression ratio
        let total_processed = self.stats.total_tokens_processed as usize;
        if total_processed > 0 {
            self.stats.compression_ratio = self.current_tokens as f32 / total_processed as f32;
        }

        // Calculate average importance
        let total_importance: f32 = self.segments
            .iter()
            .map(|s| s.importance_score)
            .sum();
        
        if !self.segments.is_empty() {
            self.stats.avg_importance = total_importance / self.segments.len() as f32;
        }
    }
}

impl ImportanceTracker {
    /// Update keyword frequency
    fn update_keyword_frequency(&mut self, word: &str) {
        let normalized_word = word.to_lowercase();
        *self.keyword_frequency.entry(normalized_word.clone()).or_insert(0) += 1;

        // Add to recent topics if it's significant
        if word.len() > 3 && !self.is_stop_word(&normalized_word) {
            self.recent_topics.push_back((normalized_word, 1.0));
            
            // Maintain max topics limit
            if self.recent_topics.len() > self.max_topics {
                self.recent_topics.pop_front();
            }
        }
    }

    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "among", "within",
            "without", "along", "following", "across", "behind", "beyond", "plus",
            "except", "but", "up", "out", "around", "down", "off", "above", "below"
        ];

        STOP_WORDS.contains(&word)
    }
}

/// Context window factory for different use cases
pub struct ContextWindowFactory;

impl ContextWindowFactory {
    /// Create a context window optimized for chat applications
    pub fn create_chat_context() -> Result<ContextWindow, JsValue> {
        let config = ContextConfig {
            max_tokens: 16_000,
            compression_threshold: 0.75,
            enable_summarization: true,
            max_summary_ratio: 0.4,
            enable_importance_tracking: true,
            sliding_window_overlap: 256,
        };

        let config_json = serde_json::to_string(&config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        ContextWindow::new(&config_json)
    }

    /// Create a context window optimized for code analysis
    pub fn create_code_context() -> Result<ContextWindow, JsValue> {
        let config = ContextConfig {
            max_tokens: 24_000,
            compression_threshold: 0.8,
            enable_summarization: false, // Code shouldn't be summarized
            max_summary_ratio: 0.2,
            enable_importance_tracking: true,
            sliding_window_overlap: 512,
        };

        let config_json = serde_json::to_string(&config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        ContextWindow::new(&config_json)
    }

    /// Create a context window optimized for document processing
    pub fn create_document_context() -> Result<ContextWindow, JsValue> {
        let config = ContextConfig {
            max_tokens: 32_000,
            compression_threshold: 0.85,
            enable_summarization: true,
            max_summary_ratio: 0.25,
            enable_importance_tracking: true,
            sliding_window_overlap: 1024,
        };

        let config_json = serde_json::to_string(&config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        ContextWindow::new(&config_json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_window_creation() {
        let config = ContextConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        let context = ContextWindow::new(&config_json).unwrap();

        assert_eq!(context.current_tokens, 0);
        assert_eq!(context.segments.len(), 0);
    }

    #[test]
    fn test_token_importance_calculation() {
        let config = ContextConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        let mut context = ContextWindow::new(&config_json).unwrap();

        let importance = context.calculate_token_importance("important", ContextType::User);
        assert!(importance > 0.5);

        let common_word_importance = context.calculate_token_importance("the", ContextType::User);
        assert!(common_word_importance < importance);
    }

    #[test]
    fn test_context_compression() {
        let config = ContextConfig {
            max_tokens: 10,
            compression_threshold: 0.8,
            ..Default::default()
        };
        let config_json = serde_json::to_string(&config).unwrap();
        let mut context = ContextWindow::new(&config_json).unwrap();

        // Add content that exceeds the limit
        context.add_content("This is a very long sentence that should trigger compression", "user").unwrap();
        
        // Context should have been compressed or truncated
        assert!(context.current_tokens <= config.max_tokens);
    }

    #[test]
    fn test_importance_tracker() {
        let mut tracker = ImportanceTracker::default();
        
        tracker.update_keyword_frequency("important");
        tracker.update_keyword_frequency("important");
        tracker.update_keyword_frequency("normal");

        assert_eq!(tracker.keyword_frequency.get("important"), Some(&2));
        assert_eq!(tracker.keyword_frequency.get("normal"), Some(&1));
    }

    #[test]
    fn test_context_factory() {
        let chat_context = ContextWindowFactory::create_chat_context().unwrap();
        let code_context = ContextWindowFactory::create_code_context().unwrap();
        let doc_context = ContextWindowFactory::create_document_context().unwrap();

        // Verify different configurations
        assert!(chat_context.config.max_tokens < doc_context.config.max_tokens);
        assert!(!code_context.config.enable_summarization);
        assert!(doc_context.config.enable_summarization);
    }
}