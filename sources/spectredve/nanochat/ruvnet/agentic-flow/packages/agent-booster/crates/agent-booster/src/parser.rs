use crate::models::{AgentBoosterError, CodeChunk, Language, Result};
use tree_sitter::{Parser as TSParser, Tree};

/// Parser for extracting code chunks using tree-sitter
pub struct Parser {
    js_parser: TSParser,
    ts_parser: TSParser,
}

impl Parser {
    /// Create a new parser with JavaScript and TypeScript support
    pub fn new() -> Result<Self> {
        let mut js_parser = TSParser::new();
        js_parser
            .set_language(&tree_sitter_javascript::language())
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?;

        let mut ts_parser = TSParser::new();
        ts_parser
            .set_language(&tree_sitter_typescript::language_typescript())
            .map_err(|e| AgentBoosterError::ParseError(e.to_string()))?;

        Ok(Self {
            js_parser,
            ts_parser,
        })
    }

    /// Parse code into a tree-sitter Tree
    pub fn parse(&mut self, code: &str, language: Language) -> Result<Tree> {
        let parser = match language {
            Language::JavaScript => &mut self.js_parser,
            Language::TypeScript => &mut self.ts_parser,
            // For other languages in native build, fall back to JavaScript parser
            // (This is only used when tree-sitter feature is enabled, which is native-only)
            _ => &mut self.js_parser,
        };

        parser
            .parse(code, None)
            .ok_or_else(|| AgentBoosterError::ParseError("Failed to parse code".to_string()))
    }

    /// Extract semantic code chunks from parsed tree
    pub fn extract_chunks(&self, tree: &Tree, code: &str) -> Vec<CodeChunk> {
        let mut chunks = Vec::new();
        let root = tree.root_node();

        self.extract_chunks_recursive(root, code, None, &mut chunks);

        // Sort by start position
        chunks.sort_by_key(|c| c.start_byte);
        chunks
    }

    fn extract_chunks_recursive(
        &self,
        node: tree_sitter::Node,
        code: &str,
        parent_type: Option<String>,
        chunks: &mut Vec<CodeChunk>,
    ) {
        let node_type = node.kind();

        // Extract meaningful code blocks
        let is_chunk = matches!(
            node_type,
            "function_declaration"
                | "function_expression"
                | "arrow_function"
                | "method_definition"
                | "class_declaration"
                | "class"
                | "interface_declaration"
                | "type_alias_declaration"
                | "enum_declaration"
                | "variable_declaration"
                | "export_statement"
                | "import_statement"
        );

        if is_chunk && !node.is_error() {
            let start_byte = node.start_byte();
            let end_byte = node.end_byte();

            if let Ok(chunk_code) = std::str::from_utf8(&code.as_bytes()[start_byte..end_byte]) {
                let start_line = node.start_position().row;
                let end_line = node.end_position().row;

                chunks.push(CodeChunk {
                    code: chunk_code.to_string(),
                    start_byte,
                    end_byte,
                    start_line,
                    end_line,
                    node_type: node_type.to_string(),
                    parent_type: parent_type.clone(),
                });
            }
        }

        // Recurse into children
        let current_type = Some(node_type.to_string());
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_chunks_recursive(child, code, current_type.clone(), chunks);
        }
    }

    /// Validate that code has valid syntax
    pub fn validate_syntax(&mut self, code: &str, language: Language) -> bool {
        match self.parse(code, language) {
            Ok(tree) => !tree.root_node().has_error(),
            Err(_) => false,
        }
    }

    /// Extract full file as a single chunk (fallback for small files)
    pub fn extract_full_file(&self, code: &str) -> CodeChunk {
        let lines = code.lines().count();
        CodeChunk {
            code: code.to_string(),
            start_byte: 0,
            end_byte: code.len(),
            start_line: 0,
            end_line: lines.saturating_sub(1),
            node_type: "file".to_string(),
            parent_type: None,
        }
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new().expect("Failed to initialize parser")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_javascript() {
        let mut parser = Parser::new().unwrap();
        let code = "function hello() { return 'world'; }";
        let tree = parser.parse(code, Language::JavaScript).unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn test_parse_typescript() {
        let mut parser = Parser::new().unwrap();
        let code = "function hello(): string { return 'world'; }";
        let tree = parser.parse(code, Language::TypeScript).unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn test_extract_chunks() {
        let mut parser = Parser::new().unwrap();
        let code = r#"
function foo() {
    return 1;
}

class Bar {
    method() {
        return 2;
    }
}
"#;
        let tree = parser.parse(code, Language::JavaScript).unwrap();
        let chunks = parser.extract_chunks(&tree, code);

        assert!(chunks.len() >= 2, "Should extract function and class");
        assert!(chunks.iter().any(|c| c.node_type == "function_declaration"));
        assert!(chunks.iter().any(|c| c.node_type == "class_declaration"));
    }

    #[test]
    fn test_validate_syntax() {
        let mut parser = Parser::new().unwrap();
        assert!(parser.validate_syntax("function test() {}", Language::JavaScript));
        assert!(!parser.validate_syntax("function test() {", Language::JavaScript));
    }
}
