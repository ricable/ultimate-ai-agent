//! # ReasoningBank MCP
//!
//! MCP server implementation with reasoning tools and resources.
//!
//! This crate provides Model Context Protocol integration for Claude
//! and other AI systems to use reasoning capabilities.

pub mod server;
pub mod tools;

pub use server::{McpServer, McpConfig};
pub use tools::{McpTool, McpResource};

use thiserror::Error;

/// Result type for MCP operations
pub type Result<T> = std::result::Result<T, McpError>;

/// Errors that can occur in MCP operations
#[derive(Error, Debug)]
pub enum McpError {
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Storage error: {0}")]
    Storage(#[from] reasoningbank_storage::StorageError),

    #[error("Learning error: {0}")]
    Learning(#[from] reasoningbank_learning::LearningError),

    #[error("Network error: {0}")]
    Network(#[from] reasoningbank_network::NetworkError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}
