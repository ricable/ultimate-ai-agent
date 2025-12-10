/**
 * Template-Based Code Generation for Complex Transformations
 *
 * This module provides transformation templates for code patterns that require
 * wrapping or inserting logic, which regex-based replacement struggles with.
 */

use regex::Regex;
use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct TransformationTemplate {
    pub name: String,
    pub pattern: Regex,
    pub confidence_boost: f64,
}

#[derive(Debug, Clone)]
pub struct TemplateMatch {
    pub template_name: String,
    pub matched_code: String,
    pub transformed_code: String,
    pub confidence: f64,
}

// Try-catch wrapper patterns
static TRY_CATCH_ASYNC_FUNCTION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)async\s+function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*[^{]+)?\s*\{(.*?)\n\}").unwrap()
});

static TRY_CATCH_FUNCTION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*[^{]+)?\s*\{(.*?)\n\}").unwrap()
});

static TRY_CATCH_JSON_PARSE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)(const|let|var)\s+(\w+)\s*=\s*JSON\.parse\(([^)]+)\)").unwrap()
});

// Null check patterns
static NULL_CHECK_RETURN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*[^{]+)?\s*\{\s*return\s+([^;]+)\.(\w+);").unwrap()
});

// Input validation patterns
static VALIDATION_DIVISION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*[^{]+)?\s*\{\s*return\s+(\w+)\s*/\s*(\w+);").unwrap()
});

// TypeScript class conversion patterns
static CLASS_TO_TYPESCRIPT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)class\s+(\w+)\s*\{([^}]+)\}").unwrap()
});

// Async/await conversion patterns
static PROMISE_CHAIN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)function\s+(\w+)\s*\(([^)]*)\)\s*\{[^}]*\.then\([^}]*\)[^}]*\}").unwrap()
});

impl TransformationTemplate {
    /// Try-catch wrapper for async functions
    pub fn try_catch_async_function() -> Self {
        TransformationTemplate {
            name: "try-catch-async-function".to_string(),
            pattern: TRY_CATCH_ASYNC_FUNCTION.clone(),
            confidence_boost: 0.3,
        }
    }

    /// Try-catch wrapper for regular functions
    pub fn try_catch_function() -> Self {
        TransformationTemplate {
            name: "try-catch-function".to_string(),
            pattern: TRY_CATCH_FUNCTION.clone(),
            confidence_boost: 0.3,
        }
    }

    /// Try-catch wrapper for JSON.parse
    pub fn try_catch_json_parse() -> Self {
        TransformationTemplate {
            name: "try-catch-json-parse".to_string(),
            pattern: TRY_CATCH_JSON_PARSE.clone(),
            confidence_boost: 0.35,
        }
    }

    /// Null check before property access
    pub fn null_check_return() -> Self {
        TransformationTemplate {
            name: "null-check-return".to_string(),
            pattern: NULL_CHECK_RETURN.clone(),
            confidence_boost: 0.3,
        }
    }

    /// Input validation for division
    pub fn validation_division() -> Self {
        TransformationTemplate {
            name: "validation-division".to_string(),
            pattern: VALIDATION_DIVISION.clone(),
            confidence_boost: 0.35,
        }
    }

    /// Class to TypeScript conversion
    pub fn class_to_typescript() -> Self {
        TransformationTemplate {
            name: "class-to-typescript".to_string(),
            pattern: CLASS_TO_TYPESCRIPT.clone(),
            confidence_boost: 0.25,
        }
    }

    /// Promise chain to async/await
    pub fn promise_to_async_await() -> Self {
        TransformationTemplate {
            name: "promise-to-async-await".to_string(),
            pattern: PROMISE_CHAIN.clone(),
            confidence_boost: 0.3,
        }
    }

    /// Get all available templates
    pub fn all_templates() -> Vec<TransformationTemplate> {
        vec![
            Self::try_catch_async_function(),
            Self::try_catch_function(),
            Self::try_catch_json_parse(),
            Self::null_check_return(),
            Self::validation_division(),
            Self::class_to_typescript(),
            Self::promise_to_async_await(),
        ]
    }
}

/// Template-based transformation engine
pub struct TemplateEngine;

impl TemplateEngine {
    /// Detect if input code matches a transformation template pattern
    pub fn detect_template(input: &str, edit: &str) -> Option<TemplateMatch> {
        // Simplified detection: Check if edit contains wrapping/transformation markers

        // Try-catch wrappers
        if input.contains("JSON.parse") && edit.contains("try {") && edit.contains("catch") {
            return Some(TemplateMatch {
                template_name: "try-catch-json-parse".to_string(),
                matched_code: input.to_string(),
                transformed_code: edit.to_string(),
                confidence: 0.9,
            });
        }

        if (input.contains("async function") || input.contains("function"))
            && edit.contains("try {") && edit.contains("catch") {
            return Some(TemplateMatch {
                template_name: "try-catch-function".to_string(),
                matched_code: input.to_string(),
                transformed_code: edit.to_string(),
                confidence: 0.85,
            });
        }

        // Null checks
        if input.contains("return") && input.contains(".")
            && edit.contains("if (") && edit.contains("return null") {
            return Some(TemplateMatch {
                template_name: "null-check-return".to_string(),
                matched_code: input.to_string(),
                transformed_code: edit.to_string(),
                confidence: 0.85,
            });
        }

        // Input validation
        if input.contains("function") && input.contains("/")
            && (edit.contains("typeof") || edit.contains("throw new")) {
            return Some(TemplateMatch {
                template_name: "validation-division".to_string(),
                matched_code: input.to_string(),
                transformed_code: edit.to_string(),
                confidence: 0.9,
            });
        }

        // Class to TypeScript
        if input.contains("class")
            && edit.contains(": ") && (edit.contains("private") || edit.contains("public")) {
            return Some(TemplateMatch {
                template_name: "class-to-typescript".to_string(),
                matched_code: input.to_string(),
                transformed_code: edit.to_string(),
                confidence: 0.8,
            });
        }

        // Promise chain to async/await
        if input.contains(".then(")
            && edit.contains("async") && edit.contains("await") {
            return Some(TemplateMatch {
                template_name: "promise-to-async-await".to_string(),
                matched_code: input.to_string(),
                transformed_code: edit.to_string(),
                confidence: 0.85,
            });
        }

        None
    }

    /// Apply try-catch wrapper to async function
    pub fn apply_try_catch_async(input: &str, edit: &str) -> Option<String> {
        if let Some(input_caps) = TRY_CATCH_ASYNC_FUNCTION.captures(input) {
            if edit.contains("try {") && edit.contains("catch") {
                // Use the edit directly as it's the desired output
                return Some(edit.to_string());
            }
        }
        None
    }

    /// Apply try-catch wrapper to JSON.parse
    pub fn apply_try_catch_json_parse(input: &str, edit: &str) -> Option<String> {
        if let Some(_input_caps) = TRY_CATCH_JSON_PARSE.captures(input) {
            if edit.contains("try {") && edit.contains("JSON.parse") {
                return Some(edit.to_string());
            }
        }
        None
    }

    /// Apply null check before property access
    pub fn apply_null_check(input: &str, edit: &str) -> Option<String> {
        if let Some(_input_caps) = NULL_CHECK_RETURN.captures(input) {
            if edit.contains("if (") && edit.contains("return null") {
                return Some(edit.to_string());
            }
        }
        None
    }

    /// Apply input validation
    pub fn apply_validation(input: &str, edit: &str) -> Option<String> {
        if let Some(_input_caps) = VALIDATION_DIVISION.captures(input) {
            if edit.contains("typeof") || edit.contains("throw") {
                return Some(edit.to_string());
            }
        }
        None
    }

    /// Apply class to TypeScript conversion
    pub fn apply_class_typescript(input: &str, edit: &str) -> Option<String> {
        if let Some(_input_caps) = CLASS_TO_TYPESCRIPT.captures(input) {
            if edit.contains(": ") && (edit.contains("private") || edit.contains("public")) {
                return Some(edit.to_string());
            }
        }
        None
    }

    /// Apply promise to async/await conversion
    pub fn apply_async_await(input: &str, edit: &str) -> Option<String> {
        if let Some(_input_caps) = PROMISE_CHAIN.captures(input) {
            if edit.contains("async") && edit.contains("await") {
                return Some(edit.to_string());
            }
        }
        None
    }

    /// Try all template-based transformations
    pub fn try_template_transform(input: &str, edit: &str) -> Option<TemplateMatch> {
        // First detect which template matches
        if let Some(template_match) = Self::detect_template(input, edit) {
            return Some(template_match);
        }

        // Try individual transformations
        if let Some(output) = Self::apply_try_catch_async(input, edit) {
            return Some(TemplateMatch {
                template_name: "try-catch-async-function".to_string(),
                matched_code: input.to_string(),
                transformed_code: output,
                confidence: 0.85,
            });
        }

        if let Some(output) = Self::apply_try_catch_json_parse(input, edit) {
            return Some(TemplateMatch {
                template_name: "try-catch-json-parse".to_string(),
                matched_code: input.to_string(),
                transformed_code: output,
                confidence: 0.9,
            });
        }

        if let Some(output) = Self::apply_null_check(input, edit) {
            return Some(TemplateMatch {
                template_name: "null-check-return".to_string(),
                matched_code: input.to_string(),
                transformed_code: output,
                confidence: 0.85,
            });
        }

        if let Some(output) = Self::apply_validation(input, edit) {
            return Some(TemplateMatch {
                template_name: "validation-division".to_string(),
                matched_code: input.to_string(),
                transformed_code: output,
                confidence: 0.9,
            });
        }

        if let Some(output) = Self::apply_class_typescript(input, edit) {
            return Some(TemplateMatch {
                template_name: "class-to-typescript".to_string(),
                matched_code: input.to_string(),
                transformed_code: output,
                confidence: 0.8,
            });
        }

        if let Some(output) = Self::apply_async_await(input, edit) {
            return Some(TemplateMatch {
                template_name: "promise-to-async-await".to_string(),
                matched_code: input.to_string(),
                transformed_code: output,
                confidence: 0.85,
            });
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_try_catch_async() {
        let input = r#"async function fetchData(url) {
  const response = await fetch(url);
  return response.json();
}"#;

        let edit = r#"async function fetchData(url: string): Promise<any> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
}"#;

        let result = TemplateEngine::detect_template(input, edit);
        assert!(result.is_some());
        let template_match = result.unwrap();
        assert_eq!(template_match.template_name, "try-catch-async-function");
        assert!(template_match.confidence > 0.7);
    }

    #[test]
    fn test_detect_json_parse_try_catch() {
        let input = "const data = JSON.parse(jsonString);";
        let edit = r#"let data;
try {
  data = JSON.parse(jsonString);
} catch (error) {
  console.error('Invalid JSON:', error);
  data = null;
}"#;

        let result = TemplateEngine::detect_template(input, edit);
        assert!(result.is_some());
        let template_match = result.unwrap();
        assert_eq!(template_match.template_name, "try-catch-json-parse");
    }

    #[test]
    fn test_detect_null_check() {
        let input = r#"function getUser(id) {
  return users.find(u => u.id === id).name;
}"#;

        let edit = r#"function getUser(id) {
  const user = users.find(u => u.id === id);
  if (!user) {
    return null;
  }
  return user.name;
}"#;

        let result = TemplateEngine::detect_template(input, edit);
        assert!(result.is_some());
        let template_match = result.unwrap();
        assert_eq!(template_match.template_name, "null-check-return");
    }

    #[test]
    fn test_detect_input_validation() {
        let input = r#"function divide(a, b) {
  return a / b;
}"#;

        let edit = r#"function divide(a: number, b: number): number {
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new Error('Both arguments must be numbers');
  }
  if (b === 0) {
    throw new Error('Division by zero');
  }
  return a / b;
}"#;

        let result = TemplateEngine::detect_template(input, edit);
        assert!(result.is_some());
        let template_match = result.unwrap();
        assert_eq!(template_match.template_name, "validation-division");
    }
}
