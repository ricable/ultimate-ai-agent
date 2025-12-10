//! Agent Booster - Fast code editing library using tree-sitter and similarity matching
//!
//! This library provides functionality to automatically apply code edits by finding
//! the best matching location in existing code using tree-sitter parsing and text
//! similarity algorithms.

pub mod merge;
pub mod models;
pub mod templates;

#[cfg(feature = "tree-sitter-parser")]
pub mod parser;

#[cfg(not(feature = "tree-sitter-parser"))]
pub mod parser_lite;

#[cfg(not(feature = "tree-sitter-parser"))]
pub use parser_lite as parser;

pub mod similarity;

pub use models::{
    AgentBoosterError, CodeChunk, Config, EditMetadata, EditRequest, EditResult, Language,
    MergeStrategy, Result,
};
pub use templates::TemplateEngine;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

/// Main AgentBooster API
pub struct AgentBooster {
    parser: parser::Parser,
    merger: merge::Merger,
    config: Config,
}

impl AgentBooster {
    /// Create a new AgentBooster instance with the given configuration
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            parser: parser::Parser::new()?,
            merger: merge::Merger::new()?,
            config,
        })
    }

    /// Apply a single edit to code
    pub fn apply_edit(&mut self, request: EditRequest) -> Result<EditResult> {
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = Instant::now();

        // PHASE 1: Try template-based transformation first (bypasses similarity matching)
        if let Some(template_match) = TemplateEngine::try_template_transform(&request.original_code, &request.edit_snippet) {
            let syntax_valid = self.parser.validate_syntax(&template_match.transformed_code, request.language);
            let confidence_f32 = template_match.confidence as f32;

            if confidence_f32 >= request.confidence_threshold && syntax_valid {
                #[cfg(not(target_arch = "wasm32"))]
                let processing_time_ms = Some(start_time.elapsed().as_millis() as u64);

                #[cfg(target_arch = "wasm32")]
                let processing_time_ms = None;

                return Ok(EditResult {
                    merged_code: template_match.transformed_code,
                    confidence: confidence_f32,
                    strategy: MergeStrategy::ExactReplace,
                    metadata: EditMetadata {
                        chunks_found: 1,
                        best_similarity: 1.0,
                        syntax_valid,
                        processing_time_ms,
                    },
                });
            }
        }

        // PHASE 2: Fall back to similarity-based matching
        // Parse original code
        let tree = self
            .parser
            .parse(&request.original_code, request.language)?;
        let mut chunks = self.parser.extract_chunks(&tree, &request.original_code);

        // If no chunks found, treat entire file as one chunk
        if chunks.is_empty() {
            chunks.push(self.parser.extract_full_file(&request.original_code));
        }

        // Limit chunks if needed
        if chunks.len() > self.config.max_chunks {
            chunks.truncate(self.config.max_chunks);
        }

        let chunks_found = chunks.len();

        // Find best matching chunk
        let best_match = similarity::SimilarityMatcher::find_best_match(&request.edit_snippet, &chunks)
            .ok_or_else(|| AgentBoosterError::LowConfidence(0.0))?;

        let best_similarity = best_match.similarity;

        // Apply merge
        let merge_result = self.merger.merge(
            &request.original_code,
            &request.edit_snippet,
            &best_match,
            request.language,
            request.confidence_threshold,
        )?;

        #[cfg(not(target_arch = "wasm32"))]
        let processing_time_ms = Some(start_time.elapsed().as_millis() as u64);

        #[cfg(target_arch = "wasm32")]
        let processing_time_ms = None; // No timing in WASM

        Ok(EditResult {
            merged_code: merge_result.code,
            confidence: merge_result.confidence,
            strategy: merge_result.strategy,
            metadata: EditMetadata {
                chunks_found,
                best_similarity,
                syntax_valid: merge_result.syntax_valid,
                processing_time_ms,
            },
        })
    }

    /// Apply multiple edits in batch (sequentially for now)
    pub fn batch_apply(&mut self, requests: Vec<EditRequest>) -> Result<Vec<EditResult>> {
        requests
            .into_iter()
            .map(|req| self.apply_edit(req))
            .collect()
    }

    /// Get the current configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: Config) {
        self.config = config;
    }
}

impl Default for AgentBooster {
    fn default() -> Self {
        Self::new(Config::default()).expect("Failed to create AgentBooster")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function_replacement() {
        let mut booster = AgentBooster::new(Config::default()).unwrap();

        let request = EditRequest {
            original_code: "function foo() { return 1; }".to_string(),
            edit_snippet: "function foo() { return 2; }".to_string(),
            language: Language::JavaScript,
            confidence_threshold: 0.5,
        };

        let result = booster.apply_edit(request).unwrap();

        assert!(result.merged_code.contains("return 2"));
        assert!(result.confidence > 0.8);
        assert_eq!(result.strategy, MergeStrategy::ExactReplace);
    }

    #[test]
    fn test_class_method_edit() {
        let mut booster = AgentBooster::new(Config::default()).unwrap();

        let original = r#"
class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}
"#;

        let edit = r#"
    multiply(a, b) {
        return a * b;
    }
"#;

        let request = EditRequest {
            original_code: original.to_string(),
            edit_snippet: edit.to_string(),
            language: Language::JavaScript,
            confidence_threshold: 0.5,
        };

        let result = booster.apply_edit(request).unwrap();

        assert!(result.merged_code.contains("multiply"));
        assert!(result.merged_code.contains("add"));
        assert!(result.merged_code.contains("subtract"));
    }

    #[test]
    fn test_typescript_interface() {
        let mut booster = AgentBooster::new(Config::default()).unwrap();

        let original = r#"
interface User {
    name: string;
    email: string;
}
"#;

        let edit = r#"
interface Product {
    id: number;
    name: string;
    price: number;
}
"#;

        let request = EditRequest {
            original_code: original.to_string(),
            edit_snippet: edit.to_string(),
            language: Language::TypeScript,
            confidence_threshold: 0.5,
        };

        let result = booster.apply_edit(request).unwrap();

        assert!(result.merged_code.contains("User"));
        assert!(result.merged_code.contains("Product"));
    }

    #[test]
    fn test_low_confidence_rejection() {
        let mut booster = AgentBooster::new(Config {
            confidence_threshold: 0.9, // High threshold
            max_chunks: 50,
        })
        .unwrap();

        let request = EditRequest {
            original_code: "function foo() { return 1; }".to_string(),
            edit_snippet: "completely unrelated code".to_string(),
            language: Language::JavaScript,
            confidence_threshold: 0.9,
        };

        let result = booster.apply_edit(request);

        assert!(result.is_err());
    }

    #[test]
    fn test_batch_apply() {
        let mut booster = AgentBooster::new(Config::default()).unwrap();

        let requests = vec![
            EditRequest {
                original_code: "function foo() { return 1; }".to_string(),
                edit_snippet: "function foo() { return 2; }".to_string(),
                language: Language::JavaScript,
                confidence_threshold: 0.5,
            },
            EditRequest {
                original_code: "function bar() { return 3; }".to_string(),
                edit_snippet: "function bar() { return 4; }".to_string(),
                language: Language::JavaScript,
                confidence_threshold: 0.5,
            },
        ];

        let results = booster.batch_apply(requests).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].merged_code.contains("return 2"));
        assert!(results[1].merged_code.contains("return 4"));
    }

    #[test]
    fn test_empty_file() {
        let mut booster = AgentBooster::new(Config::default()).unwrap();

        let request = EditRequest {
            original_code: "".to_string(),
            edit_snippet: "function test() { return 1; }".to_string(),
            language: Language::JavaScript,
            confidence_threshold: 0.3, // Lower threshold for empty file
        };

        let result = booster.apply_edit(request).unwrap();

        assert!(result.merged_code.contains("function test"));
        assert_eq!(result.strategy, MergeStrategy::Append);
    }
}
