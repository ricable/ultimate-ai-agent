use crate::models::{AgentBoosterError, Language, MergeStrategy, Result};
use crate::parser::Parser;
use crate::similarity::SearchResult;
use crate::templates::TemplateEngine;

/// Merger for applying edits to code
pub struct Merger {
    parser: Parser,
}

impl Merger {
    pub fn new() -> Result<Self> {
        Ok(Self {
            parser: Parser::new()?,
        })
    }

    /// Apply merge using the best match and selected strategy
    pub fn merge(
        &mut self,
        original_code: &str,
        edit_snippet: &str,
        best_match: &SearchResult,
        language: Language,
        confidence_threshold: f32,
    ) -> Result<MergeResult> {
        // PHASE 1: Try template-based transformation first
        if let Some(template_match) = TemplateEngine::try_template_transform(original_code, edit_snippet) {
            // Template matched! Use template transformation
            let syntax_valid = self.parser.validate_syntax(&template_match.transformed_code, language);
            let confidence_f32 = template_match.confidence as f32;

            if confidence_f32 >= confidence_threshold && syntax_valid {
                return Ok(MergeResult {
                    code: template_match.transformed_code,
                    strategy: MergeStrategy::ExactReplace, // Templates are exact transformations
                    confidence: confidence_f32,
                    syntax_valid,
                });
            }
        }

        // PHASE 2: Fall back to similarity-based merge
        // Select strategy based on similarity
        let strategy = Self::select_strategy(best_match.similarity, confidence_threshold);

        // Apply the merge
        let merged_code = self.apply_merge(original_code, edit_snippet, best_match, strategy)?;

        // Validate syntax
        let syntax_valid = self.parser.validate_syntax(&merged_code, language);

        // Calculate final confidence
        let confidence = Self::calculate_confidence(best_match.similarity, strategy, syntax_valid);

        // Check if confidence meets threshold
        if confidence < confidence_threshold {
            return Err(AgentBoosterError::LowConfidence(confidence));
        }

        if !syntax_valid {
            return Err(AgentBoosterError::InvalidSyntax);
        }

        Ok(MergeResult {
            code: merged_code,
            strategy,
            confidence,
            syntax_valid,
        })
    }

    /// Select merge strategy based on similarity score
    fn select_strategy(similarity: f32, threshold: f32) -> MergeStrategy {
        match similarity {
            // Prefer replacement strategies over insertion for simple edits
            s if s >= 0.90 => MergeStrategy::ExactReplace,  // Very high similarity
            s if s >= 0.50 => MergeStrategy::FuzzyReplace,  // Medium-high similarity (var→const, add types)
            s if s >= 0.30 => MergeStrategy::InsertAfter,   // Low-medium similarity
            s if s >= threshold => MergeStrategy::InsertBefore,
            _ => MergeStrategy::Append,
        }
    }

    /// Apply the actual merge operation
    fn apply_merge(
        &self,
        original_code: &str,
        edit_snippet: &str,
        best_match: &SearchResult,
        strategy: MergeStrategy,
    ) -> Result<String> {
        match strategy {
            MergeStrategy::ExactReplace => {
                self.apply_exact_replace(original_code, edit_snippet, best_match)
            }
            MergeStrategy::FuzzyReplace => {
                self.apply_fuzzy_replace(original_code, edit_snippet, best_match)
            }
            MergeStrategy::InsertAfter => {
                self.apply_insert_after(original_code, edit_snippet, best_match)
            }
            MergeStrategy::InsertBefore => {
                self.apply_insert_before(original_code, edit_snippet, best_match)
            }
            MergeStrategy::Append => Ok(self.apply_append(original_code, edit_snippet)),
        }
    }

    /// Replace exact match with edit snippet
    fn apply_exact_replace(
        &self,
        original_code: &str,
        edit_snippet: &str,
        best_match: &SearchResult,
    ) -> Result<String> {
        let chunk = &best_match.chunk;

        // Replace the matched chunk with the edit snippet
        let before = &original_code[..chunk.start_byte];
        let after = &original_code[chunk.end_byte..];

        Ok(format!("{}{}{}", before, edit_snippet, after))
    }

    /// Replace with fuzzy matching (similar to exact but more lenient)
    fn apply_fuzzy_replace(
        &self,
        original_code: &str,
        edit_snippet: &str,
        best_match: &SearchResult,
    ) -> Result<String> {
        // For now, fuzzy replace is the same as exact replace
        // In a more advanced version, this could handle partial replacements
        self.apply_exact_replace(original_code, edit_snippet, best_match)
    }

    /// Insert edit snippet after matched chunk
    fn apply_insert_after(
        &self,
        original_code: &str,
        edit_snippet: &str,
        best_match: &SearchResult,
    ) -> Result<String> {
        let chunk = &best_match.chunk;

        let before = &original_code[..chunk.end_byte];
        let after = &original_code[chunk.end_byte..];

        // Add newlines for proper formatting
        Ok(format!("{}\n\n{}{}", before, edit_snippet, after))
    }

    /// Insert edit snippet before matched chunk
    fn apply_insert_before(
        &self,
        original_code: &str,
        edit_snippet: &str,
        best_match: &SearchResult,
    ) -> Result<String> {
        let chunk = &best_match.chunk;

        let before = &original_code[..chunk.start_byte];
        let after = &original_code[chunk.start_byte..];

        // Add newlines for proper formatting
        Ok(format!("{}{}\n\n{}", before, edit_snippet, after))
    }

    /// Append edit snippet to end of file
    fn apply_append(&self, original_code: &str, edit_snippet: &str) -> String {
        format!("{}\n\n{}", original_code.trim_end(), edit_snippet)
    }

    /// Calculate final confidence score
    fn calculate_confidence(similarity: f32, strategy: MergeStrategy, syntax_valid: bool) -> f32 {
        let mut confidence = similarity;

        // Adjust based on strategy
        confidence *= match strategy {
            MergeStrategy::ExactReplace => 1.0,
            MergeStrategy::FuzzyReplace => 0.95,
            MergeStrategy::InsertAfter => 0.85,
            MergeStrategy::InsertBefore => 0.80,
            MergeStrategy::Append => 0.60,
        };

        // Penalize if syntax is invalid
        if !syntax_valid {
            confidence *= 0.5;
        }

        confidence.clamp(0.0, 1.0)
    }
}

impl Default for Merger {
    fn default() -> Self {
        Self::new().expect("Failed to initialize merger")
    }
}

/// Result of a merge operation
#[derive(Debug, Clone)]
pub struct MergeResult {
    pub code: String,
    pub strategy: MergeStrategy,
    pub confidence: f32,
    pub syntax_valid: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::CodeChunk;

    fn create_test_match(code: &str, similarity: f32) -> SearchResult {
        SearchResult {
            chunk: CodeChunk {
                code: code.to_string(),
                start_byte: 0,
                end_byte: code.len(),
                start_line: 0,
                end_line: 0,
                node_type: "function_declaration".to_string(),
                parent_type: None,
            },
            similarity,
            chunk_index: 0,
        }
    }

    #[test]
    fn test_select_strategy() {
        assert_eq!(
            Merger::select_strategy(0.96, 0.5),
            MergeStrategy::ExactReplace
        );
        assert_eq!(
            Merger::select_strategy(0.85, 0.5),
            MergeStrategy::FuzzyReplace
        );
        assert_eq!(
            Merger::select_strategy(0.70, 0.5),
            MergeStrategy::InsertAfter
        );
        assert_eq!(
            Merger::select_strategy(0.55, 0.5),
            MergeStrategy::InsertBefore
        );
        assert_eq!(Merger::select_strategy(0.30, 0.5), MergeStrategy::Append);
    }

    #[test]
    fn test_exact_replace() {
        let mut merger = Merger::new().unwrap();
        let original = "function foo() { return 1; }\nfunction bar() { return 2; }";
        let edit = "function foo() { return 99; }";

        let mut test_match = create_test_match("function foo() { return 1; }", 0.96);
        test_match.chunk.start_byte = 0;
        test_match.chunk.end_byte = 28;

        let result = merger
            .merge(original, edit, &test_match, Language::JavaScript, 0.5)
            .unwrap();

        assert!(result.code.contains("return 99"));
        assert!(result.code.contains("bar"));
        assert_eq!(result.strategy, MergeStrategy::ExactReplace);
    }

    #[test]
    fn test_append() {
        let merger = Merger::new().unwrap();
        let original = "function foo() { return 1; }";
        let edit = "function bar() { return 2; }";

        let result = merger.apply_append(original, edit);

        assert!(result.contains("foo"));
        assert!(result.contains("bar"));
        assert!(result.lines().count() >= 3); // Should have newlines
    }

    #[test]
    fn test_calculate_confidence() {
        let high = Merger::calculate_confidence(0.95, MergeStrategy::ExactReplace, true);
        assert!(high > 0.9);

        let medium = Merger::calculate_confidence(0.80, MergeStrategy::InsertAfter, true);
        assert!(medium > 0.6 && medium < 0.9);

        let low = Merger::calculate_confidence(0.95, MergeStrategy::ExactReplace, false);
        assert!(low < 0.6); // Invalid syntax should lower confidence
    }

    #[test]
    fn test_low_confidence_error() {
        let mut merger = Merger::new().unwrap();
        let original = "function foo() { return 1; }";
        let edit = "completely different code";

        let test_match = create_test_match(original, 0.30); // Low similarity

        let result = merger.merge(original, edit, &test_match, Language::JavaScript, 0.5);

        assert!(result.is_err());
        match result {
            Err(AgentBoosterError::LowConfidence(_)) => {} // Expected
            _ => panic!("Should return LowConfidence error"),
        }
    }
}
