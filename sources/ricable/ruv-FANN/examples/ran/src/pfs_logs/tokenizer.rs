use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Byte-Pair Encoding tokenizer for log data
#[derive(Debug)]
pub struct BPETokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    embeddings: HashMap<u32, Array1<f32>>,
    embedding_dim: usize,
    max_vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub min_frequency: usize,
    pub embedding_dim: usize,
}

impl BPETokenizer {
    pub fn new(max_vocab_size: usize) -> Self {
        let mut tokenizer = BPETokenizer {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            merges: Vec::new(),
            embeddings: HashMap::new(),
            embedding_dim: 256,
            max_vocab_size,
        };
        
        // Initialize with special tokens
        tokenizer.add_token("<PAD>", 0);
        tokenizer.add_token("<UNK>", 1);
        tokenizer.add_token("<START>", 2);
        tokenizer.add_token("<END>", 3);
        tokenizer.add_token("<SEP>", 4);
        
        // Initialize byte-level vocabulary
        for byte in 0..=255u8 {
            let token = format!("BYTE_{}", byte);
            let id = tokenizer.vocab.len() as u32;
            tokenizer.add_token(&token, id);
        }
        
        tokenizer
    }

    fn add_token(&mut self, token: &str, id: u32) {
        self.vocab.insert(token.to_string(), id);
        self.reverse_vocab.insert(id, token.to_string());
        
        // Initialize random embedding
        let embedding = Array1::from_shape_fn(self.embedding_dim, |_| {
            rand::random::<f32>() * 0.02 - 0.01
        });
        self.embeddings.insert(id, embedding);
    }

    /// Train BPE on a corpus of log messages
    pub fn train(&mut self, corpus: &[String], num_merges: usize) {
        // Tokenize corpus into bytes
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        
        for text in corpus {
            let words = self.pre_tokenize(text);
            for word in words {
                let byte_tokens = self.word_to_byte_tokens(&word);
                *word_freqs.entry(byte_tokens).or_insert(0) += 1;
            }
        }
        
        // Perform BPE merges
        for _ in 0..num_merges {
            if self.vocab.len() >= self.max_vocab_size {
                break;
            }
            
            // Find most frequent pair
            let pair_freqs = self.get_pair_frequencies(&word_freqs);
            if let Some((pair, _)) = pair_freqs.into_iter().next() {
                // Create new token
                let new_token = format!("{}{}", pair.0, pair.1);
                let new_id = self.vocab.len() as u32;
                self.add_token(&new_token, new_id);
                self.merges.push(pair.clone());
                
                // Update word frequencies
                word_freqs = self.merge_pair_in_words(word_freqs, &pair);
            } else {
                break;
            }
        }
    }

    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        // Split on whitespace and special characters
        let mut words = Vec::new();
        let mut current = String::new();
        
        for ch in text.chars() {
            if ch.is_whitespace() || ".,;:!?()[]{}\"'".contains(ch) {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                if !ch.is_whitespace() {
                    words.push(ch.to_string());
                }
            } else {
                current.push(ch);
            }
        }
        
        if !current.is_empty() {
            words.push(current);
        }
        
        words
    }

    fn word_to_byte_tokens(&self, word: &str) -> Vec<String> {
        word.bytes()
            .map(|b| format!("BYTE_{}", b))
            .collect()
    }

    fn get_pair_frequencies(&self, word_freqs: &HashMap<Vec<String>, usize>) -> Vec<((String, String), usize)> {
        let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
        
        for (word, freq) in word_freqs {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i].clone(), word[i + 1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }
        
        // Sort by frequency
        let mut sorted_pairs: Vec<_> = pair_freqs.into_iter().collect();
        sorted_pairs.sort_by_key(|(_, freq)| Reverse(*freq));
        sorted_pairs
    }

    fn merge_pair_in_words(
        &self,
        word_freqs: HashMap<Vec<String>, usize>,
        pair: &(String, String),
    ) -> HashMap<Vec<String>, usize> {
        let mut new_word_freqs = HashMap::new();
        
        for (word, freq) in word_freqs {
            let new_word = self.merge_pair_in_word(word, pair);
            *new_word_freqs.entry(new_word).or_insert(0) += freq;
        }
        
        new_word_freqs
    }

    fn merge_pair_in_word(&self, word: Vec<String>, pair: &(String, String)) -> Vec<String> {
        let mut new_word = Vec::new();
        let mut i = 0;
        
        while i < word.len() {
            if i < word.len() - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                new_word.push(format!("{}{}", pair.0, pair.1));
                i += 2;
            } else {
                new_word.push(word[i].clone());
                i += 1;
            }
        }
        
        new_word
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        tokens.push(self.vocab["<START>"]);
        
        let words = self.pre_tokenize(text);
        for word in words {
            let word_tokens = self.encode_word(&word);
            tokens.extend(word_tokens);
        }
        
        tokens.push(self.vocab["<END>"]);
        tokens
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut tokens = self.word_to_byte_tokens(word);
        
        // Apply merges
        for (left, right) in &self.merges {
            let merged = format!("{}{}", left, right);
            let mut i = 0;
            let mut new_tokens = Vec::new();
            
            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == *left && tokens[i + 1] == *right {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            
            tokens = new_tokens;
        }
        
        // Convert to IDs
        tokens.iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(self.vocab["<UNK>"]))
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut text = String::new();
        
        for &id in token_ids {
            if let Some(token) = self.reverse_vocab.get(&id) {
                if token.starts_with("BYTE_") {
                    if let Ok(byte_val) = token[5..].parse::<u8>() {
                        text.push(byte_val as char);
                    }
                } else if !["<START>", "<END>", "<PAD>", "<SEP>"].contains(&token.as_str()) {
                    // Decode merged tokens
                    let decoded = self.decode_merged_token(token);
                    text.push_str(&decoded);
                }
            }
        }
        
        text
    }

    fn decode_merged_token(&self, token: &str) -> String {
        // Recursively decode merged tokens
        if token.starts_with("BYTE_") {
            if let Ok(byte_val) = token[5..].parse::<u8>() {
                return (byte_val as char).to_string();
            }
        }
        
        // Try to split and decode
        for (left, right) in &self.merges {
            let merged = format!("{}{}", left, right);
            if token == merged {
                let left_decoded = self.decode_merged_token(left);
                let right_decoded = self.decode_merged_token(right);
                return format!("{}{}", left_decoded, right_decoded);
            }
        }
        
        token.to_string()
    }

    pub fn get_embedding(&self, token_id: u32) -> Array1<f32> {
        self.embeddings.get(&token_id)
            .cloned()
            .unwrap_or_else(|| self.embeddings[&self.vocab["<UNK>"]].clone())
    }

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    pub fn set_vocab(&mut self, vocab: HashMap<String, u32>) {
        self.vocab = vocab.clone();
        self.reverse_vocab.clear();
        
        for (token, id) in vocab {
            self.reverse_vocab.insert(id, token);
        }
    }

    /// Special tokenization for Ericsson log patterns
    pub fn tokenize_ericsson_log(&self, log: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        tokens.push(self.vocab["<START>"]);
        
        // Special handling for AMOS commands
        if log.contains("alt ") || log.contains("lget ") || log.contains("cvc ") {
            tokens.push(self.vocab.get("<AMOS>").copied().unwrap_or(self.vocab["<UNK>"]));
        }
        
        // Extract and tokenize structured parts
        let parts = self.extract_log_parts(log);
        for part in parts {
            match part {
                LogPart::Timestamp(ts) => {
                    tokens.push(self.vocab.get("<TIME>").copied().unwrap_or(self.vocab["<UNK>"]));
                    tokens.extend(self.encode_word(&ts));
                }
                LogPart::Level(level) => {
                    tokens.push(self.vocab.get(&format!("<{}>", level.to_uppercase()))
                        .copied()
                        .unwrap_or(self.vocab["<UNK>"]));
                }
                LogPart::Message(msg) => {
                    tokens.extend(self.encode(&msg));
                }
                LogPart::KeyValue(key, value) => {
                    tokens.extend(self.encode_word(&key));
                    tokens.push(self.vocab.get("=").copied().unwrap_or(self.vocab["<UNK>"]));
                    tokens.extend(self.encode_word(&value));
                }
            }
        }
        
        tokens.push(self.vocab["<END>"]);
        tokens
    }

    fn extract_log_parts(&self, log: &str) -> Vec<LogPart> {
        let mut parts = Vec::new();
        
        // Extract timestamp (assuming ISO format)
        if let Some(ts_match) = regex::Regex::new(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
            .unwrap()
            .find(log)
        {
            parts.push(LogPart::Timestamp(ts_match.as_str().to_string()));
        }
        
        // Extract log level
        for level in &["ERROR", "WARN", "INFO", "DEBUG", "TRACE"] {
            if log.contains(level) {
                parts.push(LogPart::Level(level.to_string()));
                break;
            }
        }
        
        // Extract key-value pairs
        if let Ok(kv_regex) = regex::Regex::new(r"(\w+)=([^\s]+)") {
            for cap in kv_regex.captures_iter(log) {
                parts.push(LogPart::KeyValue(
                    cap[1].to_string(),
                    cap[2].to_string(),
                ));
            }
        }
        
        // The rest is the message
        parts.push(LogPart::Message(log.to_string()));
        
        parts
    }
}

#[derive(Debug, Clone)]
enum LogPart {
    Timestamp(String),
    Level(String),
    Message(String),
    KeyValue(String, String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_tokenizer() {
        let mut tokenizer = BPETokenizer::new(1000);
        
        let corpus = vec![
            "ERROR: Connection failed".to_string(),
            "INFO: Connection established".to_string(),
            "ERROR: Timeout occurred".to_string(),
        ];
        
        tokenizer.train(&corpus, 50);
        
        let encoded = tokenizer.encode("ERROR: New connection failed");
        let decoded = tokenizer.decode(&encoded);
        
        assert!(decoded.contains("ERROR"));
        assert!(decoded.contains("connection"));
    }

    #[test]
    fn test_ericsson_log_tokenization() {
        let tokenizer = BPETokenizer::new(1000);
        
        let log = "2024-01-04 10:15:23 AMOS alt cell=12345 state=active";
        let tokens = tokenizer.tokenize_ericsson_log(log);
        
        assert!(tokens.len() > 0);
        assert_eq!(tokens[0], tokenizer.vocab["<START>"]);
        assert_eq!(tokens[tokens.len() - 1], tokenizer.vocab["<END>"]);
    }
}