use serde::{Deserialize, Serialize};

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    JavaScript,
    TypeScript,
    Python,
    Rust,
    Go,
    Java,
    C,
    Cpp,
}

impl Language {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "javascript" | "js" => Ok(Language::JavaScript),
            "typescript" | "ts" => Ok(Language::TypeScript),
            "python" | "py" => Ok(Language::Python),
            "rust" | "rs" => Ok(Language::Rust),
            "go" => Ok(Language::Go),
            "java" => Ok(Language::Java),
            "c" => Ok(Language::C),
            "cpp" | "c++" | "cxx" => Ok(Language::Cpp),
            _ => Err(AgentBoosterError::UnsupportedLanguage(s.to_string())),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Language::JavaScript => "javascript",
            Language::TypeScript => "typescript",
            Language::Python => "python",
            Language::Rust => "rust",
            Language::Go => "go",
            Language::Java => "java",
            Language::C => "c",
            Language::Cpp => "cpp",
        }
    }
}

/// Merge strategy for applying edits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Replace exact match with high confidence
    ExactReplace,
    /// Replace with fuzzy text matching
    FuzzyReplace,
    /// Insert after matched location
    InsertAfter,
    /// Insert before matched location
    InsertBefore,
    /// Append to end of file
    Append,
}

impl MergeStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            MergeStrategy::ExactReplace => "exact_replace",
            MergeStrategy::FuzzyReplace => "fuzzy_replace",
            MergeStrategy::InsertAfter => "insert_after",
            MergeStrategy::InsertBefore => "insert_before",
            MergeStrategy::Append => "append",
        }
    }
}

/// A code chunk extracted from parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// The code text
    pub code: String,
    /// Start byte position in original file
    pub start_byte: usize,
    /// End byte position in original file
    pub end_byte: usize,
    /// Start line number (0-indexed)
    pub start_line: usize,
    /// End line number (0-indexed)
    pub end_line: usize,
    /// Node type from tree-sitter (e.g., "function_declaration")
    pub node_type: String,
    /// Parent node type if available
    pub parent_type: Option<String>,
}

impl CodeChunk {
    pub fn line_count(&self) -> usize {
        self.end_line.saturating_sub(self.start_line) + 1
    }
}

/// Request to apply an edit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditRequest {
    /// Original source code
    pub original_code: String,
    /// Edit snippet to apply
    pub edit_snippet: String,
    /// Programming language
    pub language: Language,
    /// Minimum confidence threshold (0.0 - 1.0)
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
}

fn default_confidence_threshold() -> f32 {
    0.5
}

/// Result of applying an edit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditResult {
    /// Merged code after applying edit
    pub merged_code: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Strategy used for merging
    pub strategy: MergeStrategy,
    /// Metadata about the edit
    pub metadata: EditMetadata,
}

/// Metadata about an edit operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditMetadata {
    /// Number of chunks extracted from original code
    pub chunks_found: usize,
    /// Best similarity score found
    pub best_similarity: f32,
    /// Whether syntax validation passed
    pub syntax_valid: bool,
    /// Processing time in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_ms: Option<u64>,
}

/// Configuration for AgentBooster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Minimum confidence threshold for applying edits
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    /// Maximum number of chunks to consider
    #[serde(default = "default_max_chunks")]
    pub max_chunks: usize,
}

fn default_max_chunks() -> usize {
    50
}

impl Default for Config {
    fn default() -> Self {
        Self {
            confidence_threshold: default_confidence_threshold(),
            max_chunks: default_max_chunks(),
        }
    }
}

/// Errors that can occur in AgentBooster
#[derive(Debug, thiserror::Error)]
pub enum AgentBoosterError {
    #[error("Failed to parse code: {0}")]
    ParseError(String),

    #[error("No suitable merge location found (confidence: {0:.2})")]
    LowConfidence(f32),

    #[error("Merge resulted in invalid syntax")]
    InvalidSyntax,

    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, AgentBoosterError>;
