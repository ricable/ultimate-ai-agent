//! Lite parser for WASM compatibility
//!
//! This module provides a pure Rust parser that doesn't depend on tree-sitter's C library.
//! It uses regex-based parsing for JavaScript/TypeScript code extraction.
//!
//! Trade-off: ~80% accuracy vs tree-sitter's ~95%, but compiles to WASM without issues.

use crate::models::{AgentBoosterError, CodeChunk, Language, Result};
use regex::Regex;

/// Placeholder tree type for lite parser (no actual tree structure)
pub struct LiteTree {
    code: String,
    language: Language,
}

/// Lite parser that works in WASM without tree-sitter C dependencies
///
/// This parser uses regex-based matching instead of tree-sitter's C library.
/// It provides ~80% accuracy vs tree-sitter's ~95%, but compiles to WASM.
pub struct Parser {
    // JavaScript/TypeScript patterns
    js_function_regex: Regex,
    js_class_regex: Regex,

    // Python patterns
    py_function_regex: Regex,
    py_class_regex: Regex,

    // Rust patterns
    rust_function_regex: Regex,
    rust_struct_regex: Regex,
    rust_impl_regex: Regex,

    // Go patterns
    go_function_regex: Regex,
    go_struct_regex: Regex,

    // Java patterns
    java_method_regex: Regex,
    java_class_regex: Regex,

    // C/C++ patterns
    c_function_regex: Regex,
    cpp_class_regex: Regex,
}

impl Parser {
    /// Create a new lite parser
    pub fn new() -> Result<Self> {
        Ok(Self {
            // JavaScript/TypeScript
            js_function_regex: Regex::new(
                r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            js_class_regex: Regex::new(
                r"(?m)^\s*(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            // Python
            py_function_regex: Regex::new(
                r"(?m)^\s*(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*:",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            py_class_regex: Regex::new(
                r"(?m)^\s*class\s+(\w+)(?:\([^)]*\))?\s*:",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            // Rust
            rust_function_regex: Regex::new(
                r"(?m)^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^\{]+)?\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            rust_struct_regex: Regex::new(
                r"(?m)^\s*(?:pub\s+)?struct\s+(\w+)(?:<[^>]*>)?\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            rust_impl_regex: Regex::new(
                r"(?m)^\s*impl(?:<[^>]*>)?\s+(\w+)(?:<[^>]*>)?\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            // Go
            go_function_regex: Regex::new(
                r"(?m)^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\([^)]*\)(?:\s*\([^)]*\))?\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            go_struct_regex: Regex::new(
                r"(?m)^\s*type\s+(\w+)\s+struct\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            // Java
            java_method_regex: Regex::new(
                r"(?m)^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            java_class_regex: Regex::new(
                r"(?m)^\s*(?:public\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            // C/C++
            c_function_regex: Regex::new(
                r"(?m)^\s*(?:static\s+)?(?:inline\s+)?(?:\w+\s*\*?\s+)?(\w+)\s*\([^)]*\)\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,

            cpp_class_regex: Regex::new(
                r"(?m)^\s*(?:template\s*<[^>]*>\s*)?class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*\{",
            )
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?,
        })
    }

    /// Parse code (lite version returns code directly, not a tree)
    pub fn parse(&mut self, code: &str, language: Language) -> Result<LiteTree> {
        Ok(LiteTree {
            code: code.to_string(),
            language,
        })
    }

    /// Extract semantic code chunks from code
    pub fn extract_chunks(&self, tree: &LiteTree, code: &str) -> Vec<CodeChunk> {
        let mut chunks = Vec::new();

        match tree.language {
            Language::JavaScript | Language::TypeScript => {
                self.extract_js_chunks(code, &mut chunks);
            }
            Language::Python => {
                self.extract_python_chunks(code, &mut chunks);
            }
            Language::Rust => {
                self.extract_rust_chunks(code, &mut chunks);
            }
            Language::Go => {
                self.extract_go_chunks(code, &mut chunks);
            }
            Language::Java => {
                self.extract_java_chunks(code, &mut chunks);
            }
            Language::C => {
                self.extract_c_chunks(code, &mut chunks);
            }
            Language::Cpp => {
                self.extract_cpp_chunks(code, &mut chunks);
            }
        }

        chunks
    }

    fn extract_js_chunks(&self, code: &str, chunks: &mut Vec<CodeChunk>) {
        // Extract functions
        for cap in self.js_function_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "function_declaration".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }

        // Extract classes
        for cap in self.js_class_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "class_declaration".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }
    }

    fn extract_python_chunks(&self, code: &str, chunks: &mut Vec<CodeChunk>) {
        // Extract functions
        for cap in self.py_function_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_python_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "function_definition".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }

        // Extract classes
        for cap in self.py_class_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_python_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "class_definition".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }
    }

    fn extract_rust_chunks(&self, code: &str, chunks: &mut Vec<CodeChunk>) {
        // Extract functions
        for cap in self.rust_function_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "function_item".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }

        // Extract structs
        for cap in self.rust_struct_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "struct_item".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }

        // Extract impl blocks
        for cap in self.rust_impl_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "impl_item".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }
    }

    fn extract_go_chunks(&self, code: &str, chunks: &mut Vec<CodeChunk>) {
        // Extract functions
        for cap in self.go_function_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "function_declaration".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }

        // Extract structs
        for cap in self.go_struct_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "type_declaration".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }
    }

    fn extract_java_chunks(&self, code: &str, chunks: &mut Vec<CodeChunk>) {
        // Extract classes
        for cap in self.java_class_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "class_declaration".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }

        // Extract methods
        for cap in self.java_method_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "method_declaration".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }
    }

    fn extract_c_chunks(&self, code: &str, chunks: &mut Vec<CodeChunk>) {
        // Extract functions
        for cap in self.c_function_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "function_definition".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }
    }

    fn extract_cpp_chunks(&self, code: &str, chunks: &mut Vec<CodeChunk>) {
        // Extract classes
        for cap in self.cpp_class_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "class_specifier".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }

        // Extract functions (C-style)
        for cap in self.c_function_regex.captures_iter(code) {
            if let Some(m) = cap.get(0) {
                let start = m.start();
                if let Some(code_text) = self.extract_block(code, start) {
                    chunks.push(CodeChunk {
                        code: code_text.clone(),
                        node_type: "function_definition".to_string(),
                        start_byte: start,
                        end_byte: start + code_text.len(),
                        start_line: code[..start].lines().count(),
                        end_line: code[..start + code_text.len()].lines().count(),
                        parent_type: None,
                    });
                }
            }
        }
    }

    /// Extract a Python code block by indentation
    fn extract_python_block(&self, code: &str, start: usize) -> Option<String> {
        let lines: Vec<&str> = code.lines().collect();
        let start_line_idx = code[..start].lines().count();

        if start_line_idx >= lines.len() {
            return None;
        }

        // Get the base indentation from the def/class line
        let base_line = lines[start_line_idx];
        let base_indent = base_line.len() - base_line.trim_start().len();

        let mut end_line_idx = start_line_idx + 1;

        // Find the end of the indented block
        while end_line_idx < lines.len() {
            let line = lines[end_line_idx];
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                end_line_idx += 1;
                continue;
            }

            let line_indent = line.len() - line.trim_start().len();

            // If indentation is less than or equal to base, we've reached the end
            if line_indent <= base_indent {
                break;
            }

            end_line_idx += 1;
        }

        let block_lines = &lines[start_line_idx..end_line_idx];
        Some(block_lines.join("\n"))
    }

    /// Extract a code block by finding matching braces
    fn extract_block(&self, code: &str, start: usize) -> Option<String> {
        let bytes = code.as_bytes();

        // Find the opening brace
        let mut brace_start = start;
        while brace_start < bytes.len() && bytes[brace_start] != b'{' {
            brace_start += 1;
        }

        if brace_start >= bytes.len() {
            return None;
        }

        // Count braces to find matching closing brace
        let mut depth = 0;
        let mut pos = brace_start;

        while pos < bytes.len() {
            match bytes[pos] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        // Found matching brace
                        return Some(code[start..=pos].to_string());
                    }
                }
                _ => {}
            }
            pos += 1;
        }

        None
    }

    /// Validate syntax by checking for balanced braces/parens/brackets
    pub fn validate_syntax(&self, code: &str, _language: Language) -> bool {
        let mut paren_depth = 0;
        let mut brace_depth = 0;
        let mut bracket_depth = 0;

        for ch in code.chars() {
            match ch {
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                _ => {}
            }

            // Check for negative depth (closing before opening)
            if paren_depth < 0 || brace_depth < 0 || bracket_depth < 0 {
                return false;
            }
        }

        // All should be balanced
        paren_depth == 0 && brace_depth == 0 && bracket_depth == 0
    }

    /// Extract full file as a single chunk (fallback)
    pub fn extract_full_file(&self, code: &str) -> CodeChunk {
        CodeChunk {
            code: code.to_string(),
            node_type: "program".to_string(),
            start_byte: 0,
            end_byte: code.len(),
            start_line: 0,
            end_line: code.lines().count(),
            parent_type: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_function() {
        let mut parser = Parser::new().unwrap();
        let code = r#"
function hello() {
    console.log("Hello");
}
"#;

        let tree = parser.parse(code, Language::JavaScript).unwrap();
        let chunks = parser.extract_chunks(&tree, code);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].node_type, "function_declaration");
        assert!(chunks[0].code.contains("hello"));
    }

    #[test]
    fn test_parse_class() {
        let mut parser = Parser::new().unwrap();
        let code = r#"
class Person {
    constructor(name) {
        this.name = name;
    }
}
"#;

        let tree = parser.parse(code, Language::JavaScript).unwrap();
        let chunks = parser.extract_chunks(&tree, code);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].node_type, "class_declaration");
        assert!(chunks[0].code.contains("Person"));
    }

    #[test]
    fn test_validate_syntax() {
        let parser = Parser::new().unwrap();

        assert!(parser.validate_syntax("function f() { return 42; }", Language::JavaScript));
        assert!(!parser.validate_syntax("function f() { return 42;", Language::JavaScript));
        assert!(!parser.validate_syntax("function f() return 42; }", Language::JavaScript));
    }

    #[test]
    fn test_extract_block() {
        let parser = Parser::new().unwrap();
        let code = "function test() { return { a: 1 }; }";

        let block = parser.extract_block(code, 0);
        assert!(block.is_some());
        assert_eq!(block.unwrap(), code);
    }
}
